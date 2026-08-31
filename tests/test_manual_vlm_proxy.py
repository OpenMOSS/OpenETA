from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from agent.runtime.planner import _parse_backend_payload
from tools.manual_vlm_openeta import (
    OpenETAProtocolAdapter,
    build_operator_summary,
    build_tool_form_catalog,
    classify_request,
    detect_response_mode,
    extract_tool_audit_records,
    extract_tool_catalog,
    serialize_decision_xml,
)
from tools.manual_vlm_protocol import GenericProtocolAdapter, load_protocol_adapter
from tools.manual_vlm_proxy import (
    ManualVLMServer,
    RequestStore,
    build_wire_audit,
    load_default_adapter_spec,
    load_console_html,
)


def _json_request(url: str, value: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(value).encode() if value is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _start_server(*, adapter=None) -> tuple[ManualVLMServer, str, threading.Thread]:
    server = ManualVLMServer(
        ("127.0.0.1", 0),
        store=RequestStore(adapter=adapter),
        decision_timeout_s=2,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}", thread


def test_completion_waits_for_manual_response_and_preserves_wire_request() -> None:
    server, base, thread = _start_server()
    completion: dict = {}

    def call_provider() -> None:
        status, body = _json_request(
            base + "/v1/chat/completions",
            {
                "model": "human-vlm",
                "messages": [
                    {"role": "system", "content": "exact system text"},
                    {"role": "user", "content": "choose one tool"},
                ],
                "temperature": 0,
            },
        )
        completion.update({"status": status, "body": body})

    provider_thread = threading.Thread(target=call_provider)
    provider_thread.start()
    try:
        for _ in range(50):
            _, listing = _json_request(base + "/api/requests")
            if listing["requests"]:
                break
            time.sleep(0.02)
        pending = listing["requests"][0]
        assert pending["status"] == "pending"
        _, detail = _json_request(base + f"/api/requests/{pending['id']}")
        assert detail["messages"][0]["content"] == "exact system text"
        assert detail["request_options"]["temperature"] == 0

        xml = "<decision><kind>tool_call</kind><name>observe</name><parameters/></decision>"
        status, _ = _json_request(
            base + f"/api/requests/{pending['id']}/response", {"content": xml}
        )
        assert status == 200
        provider_thread.join(timeout=2)
        assert completion["status"] == 200
        assert completion["body"]["choices"][0]["message"]["content"] == xml
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_structured_decision_endpoint_serializes_typed_nested_parameters() -> None:
    server, base, thread = _start_server(adapter=OpenETAProtocolAdapter())
    completion: dict = {}

    def call_provider() -> None:
        status, body = _json_request(
            base + "/v1/chat/completions",
            {
                "model": "human-vlm",
                "messages": [{"role": "user", "content": "choose a tool"}],
            },
        )
        completion.update({"status": status, "body": body})

    provider_thread = threading.Thread(target=call_provider)
    provider_thread.start()
    try:
        for _ in range(50):
            _, listing = _json_request(base + "/api/requests")
            if listing["requests"]:
                break
            time.sleep(0.02)
        request_id = listing["requests"][0]["id"]
        decision = {
            "kind": "tool_call",
            "name": "example_tool",
            "parameters": {
                "count": 3,
                "enabled": True,
                "config": {"labels": ["a&b", "<target>"], "threshold": 0.25},
            },
            "reasoning": "Human form submission.",
        }
        status, _ = _json_request(
            base + f"/api/requests/{request_id}/response",
            {
                "intent": {
                    "type": "tool_call",
                    "name": decision["name"],
                    "arguments": decision["parameters"],
                    "reasoning": decision["reasoning"],
                }
            },
        )
        assert status == 200
        provider_thread.join(timeout=2)

        xml = completion["body"]["choices"][0]["message"]["content"]
        parsed, errors = _parse_backend_payload(xml)
        assert errors == []
        assert parsed == decision
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_decision_serializer_rejects_missing_required_envelope_fields() -> None:
    with pytest.raises(ValueError, match="decision.kind"):
        serialize_decision_xml({"name": "observe", "parameters": {}})


def test_inline_image_is_rendered_through_a_small_binary_endpoint() -> None:
    server, base, thread = _start_server()
    image_bytes = b"\x89PNG\r\n\x1a\nmanual-vlm-test"
    encoded = base64.b64encode(image_bytes).decode()

    def call_provider() -> None:
        _json_request(
            base + "/v1/chat/completions",
            {
                "model": "human-vlm",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "overlay follows"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
            },
        )

    provider_thread = threading.Thread(target=call_provider)
    provider_thread.start()
    try:
        for _ in range(50):
            _, listing = _json_request(base + "/api/requests")
            if listing["requests"]:
                break
            time.sleep(0.02)
        request_id = listing["requests"][0]["id"]
        _, detail = _json_request(base + f"/api/requests/{request_id}")
        image_part = detail["messages"][0]["content"][1]
        assert image_part["wire_url_kind"] == "data_url"
        assert "base64" not in json.dumps(detail)
        with urllib.request.urlopen(base + image_part["image_url"]["url"], timeout=2) as response:
            assert response.headers.get_content_type() == "image/png"
            assert response.read() == image_bytes
        _json_request(
            base + f"/api/requests/{request_id}/cancel", {"reason": "test complete"}
        )
        provider_thread.join(timeout=2)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_root_serves_human_operator_console() -> None:
    server, base, thread = _start_server()
    try:
        with urllib.request.urlopen(base + "/", timeout=2) as response:
            html = response.read().decode("utf-8")
        assert response.headers.get_content_type() == "text/html"
        assert 'data-view="operate"' in html
        assert 'data-view="audit"' in html
        assert "Human VLM Console" in html
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_tool_catalog_is_extracted_from_stable_wire_prompt_without_runtime_imports() -> None:
    stable = {
        "schema_version": "openeta.planner_static_context.v1",
        "agent_context_schema_version": "openeta.agent_context.v2",
        "available_tools": [
            {
                "name": "move_to",
                "description": "Move the end effector.",
                "parameters": {"target_pose": "world-frame pose"},
                "effect": "world_mutating",
            },
            {"name": "observe", "parameters": {}},
        ],
        "tool_references": [{"name": "move_to"}, {"name": "observe"}],
    }
    body = {
        "messages": [
            {
                "role": "system",
                "content": "Stable tool context follows.\n" + json.dumps(stable),
            }
        ]
    }
    tools = extract_tool_catalog(body)
    assert [tool["name"] for tool in tools] == ["move_to", "observe"]
    assert tools[0]["parameters"] == {"target_pose": "world-frame pose"}
    assert detect_response_mode(body) == "decision"


def test_tool_form_catalog_extracts_required_optional_defaults_and_choices() -> None:
    stable = {
        "schema_version": "openeta.planner_static_context.v1",
        "available_tools": [
            {
                "name": "create_simulator_env",
                "description": "Create an environment.",
                "parameters": {
                    "env_id": "required exact environment id",
                    "seed": "optional deterministic seed; defaults to 0",
                    "render_mode": "optional render mode; defaults to rgb_array",
                    "task": "optional task text",
                },
            },
            {
                "name": "sam3",
                "parameters": {
                    "mode": "default | text | points; defaults to text",
                    "points": {
                        "type": "array",
                        "description": "Point prompts.",
                        "required": False,
                        "default": [],
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Optional minimum detection confidence.",
                        "required": False,
                    },
                },
            },
        ],
    }
    body = {"messages": [{"role": "system", "content": json.dumps(stable)}]}

    forms = {item["name"]: item for item in build_tool_form_catalog(body)}
    create_fields = {item["name"]: item for item in forms["create_simulator_env"]["fields"]}
    sam_fields = {item["name"]: item for item in forms["sam3"]["fields"]}

    assert create_fields["env_id"]["required"] is True
    assert create_fields["env_id"]["value_type"] == "string"
    assert create_fields["task"]["required"] is False
    assert create_fields["task"]["has_default"] is False
    assert create_fields["seed"]["default"] == 0
    assert create_fields["seed"]["value_type"] == "integer"
    assert create_fields["render_mode"]["default"] == "rgb_array"
    assert create_fields["render_mode"]["value_type"] == "string"
    assert sam_fields["mode"]["choices"] == ["default", "text", "points"]
    assert sam_fields["mode"]["value_type"] == "string"
    assert sam_fields["mode"]["default"] == "text"
    assert sam_fields["points"]["value_type"] == "json"
    assert sam_fields["points"]["default"] == []
    assert sam_fields["confidence_threshold"]["value_type"] == "number"
    assert sam_fields["confidence_threshold"]["required"] is False


def _move_to_form_body(env_id: str, *, host_pose: dict | None = None) -> dict:
    stable = {
        "schema_version": "openeta.planner_static_context.v1",
        "available_tools": [
            {
                "name": "move_to",
                "description": "Move according to the active simulator profile.",
                "parameters": {"target_pose": "single-arm target"},
            },
            {
                "name": "move_eefs",
                "description": "Backend-only bimanual implementation.",
                "parameters": {"left_x": "left X", "right_x": "right X"},
            },
        ],
    }
    tool_context = {
        "objective": {"active_environment_task": {"env_id": env_id}},
    }
    if host_pose is not None:
        tool_context["host_resolved_inputs"] = {
            "move_to": {
                "status": "ready",
                "call_parameters": {"target_pose": host_pose},
            }
        }
    dynamic = {"instruction": "Choose one action.", "tool_context": tool_context}
    return {
        "messages": [
            {"role": "system", "content": json.dumps(stable)},
            {"role": "user", "content": json.dumps(dynamic)},
        ]
    }


def test_move_to_form_uses_both_eefs_for_behavior_without_exposing_move_eefs() -> None:
    forms = build_tool_form_catalog(_move_to_form_body("openeta/behavior_picking_up_trash-v0"))
    assert [form["name"] for form in forms] == ["move_to"]
    form = forms[0]
    fields = {field["name"]: field for field in form["fields"]}

    assert form["name"] == "move_to"
    assert form["control_profile"] == "bimanual"
    assert {"left_x", "left_y", "left_z", "right_x", "right_y", "right_z"}.issubset(
        fields
    )
    assert all(fields[name]["required"] is True for name in (
        "left_x", "left_y", "left_z", "right_x", "right_y", "right_z"
    ))
    assert "target_pose" not in fields


def test_move_to_form_uses_one_primitive_eef_for_single_arm_simulators() -> None:
    host_pose = {
        "frame": "world",
        "xyz": [0.1, 0.2, 0.3],
        "compiled_grasp_id": "compiled:1",
        "waypoint_role": "hover",
    }
    body = _move_to_form_body("openeta/libero_libero_10_task1-v0", host_pose=host_pose)
    form = build_tool_form_catalog(body)[0]
    fields = {field["name"]: field for field in form["fields"]}

    assert form["control_profile"] == "single"
    assert {"x", "y", "z"}.issubset(fields)
    assert "left_x" not in fields
    assert "target_pose" not in fields
    assert all("suggested_value" not in fields[axis] for axis in ("x", "y", "z"))

    encoded = OpenETAProtocolAdapter().encode_response(
        body,
        {
            "decision": {
                "kind": "tool_call",
                "name": "move_to",
                "parameters": {"x": 0.11, "y": 0.2, "z": 0.3},
                "reasoning": "Small visual correction.",
            }
        },
        request_id="move-single",
    )
    parsed, errors = _parse_backend_payload(encoded.message["content"])
    assert errors == []
    target = parsed["parameters"]["target_pose"]
    assert target["xyz"] == [0.11, 0.2, 0.3]
    assert target["compiled_grasp_id"] == "compiled:1"
    assert target["waypoint_role"] == "hover"


def test_grasp_form_requires_operator_to_copy_ready_host_bundle() -> None:
    stable = {
        "schema_version": "openeta.planner_static_context.v1",
        "available_tools": [
            {
                "name": "grasp_pose_estimate",
                "description": "Generate grasp candidates.",
                "parameters": {
                    "bundle_id": (
                        "preferred opaque id from host_resolved_inputs.grasp_pose_estimate; "
                        "when supplied, the host resolves aligned inputs atomically"
                    ),
                    "mode": "targeted or scene; defaults to targeted",
                    "object_mask": (
                        "complete SAM3 artifact; required for targeted mode and omitted "
                        "for scene mode"
                    ),
                    "hints": "optional semantic hints with max_gripper_width_m",
                },
            }
        ],
    }
    dynamic = {
        "instruction": "Choose one action.",
        "tool_context": {
            "host_resolved_inputs": {
                "grasp_pose_estimate": {
                    "status": "ready",
                    "bundle_id": "grasp:abc123",
                    "call_parameters": {"bundle_id": "grasp:abc123"},
                }
            }
        },
    }
    body = {
        "messages": [
            {"role": "system", "content": json.dumps(stable)},
            {"role": "user", "content": json.dumps(dynamic)},
        ]
    }

    form = build_tool_form_catalog(body)[0]
    fields = {field["name"]: field for field in form["fields"]}

    assert form["exclusive_parameter"] == "bundle_id"
    assert "host_input" not in form
    assert fields["bundle_id"]["value_type"] == "string"
    assert "suggested_value" not in fields["bundle_id"]
    assert fields["object_mask"]["required"] is False
    assert fields["object_mask"]["value_type"] == "json"
    assert fields["hints"]["value_type"] == "json"


def test_response_mode_detects_isolated_json_request() -> None:
    assert (
        detect_response_mode(
            {
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": "Return JSON."}],
            }
        )
        == "json"
    )


def test_operator_summary_projects_main_turn_without_dropping_wire_audit() -> None:
    stable = {
        "schema_version": "openeta.planner_static_context.v1",
        "agent_context_schema_version": "openeta.agent_context.v2",
        "available_tools": [
            {
                "name": "observe",
                "description": "Get a fresh observation.",
                "parameters": {},
            }
        ],
        "relevant_skills": [
            {"name": "pick", "description": "Acquire the requested object."}
        ],
    }
    dynamic = {
        "instruction": "Choose exactly one next OpenETA action.",
        "attempt": 2,
        "validation_errors": ["move_to requires target_pose"],
        "tool_context": {
            "objective": {
                "task": "pick up the butter",
                "active_environment_task": {"env_id": "openeta/test-v0"},
            },
            "current_observation": {
                "status": "available",
                "summary": {
                    "camera_ids": ["agentview"],
                    "object_count": 1,
                    "objects": [{"name": "butter"}],
                    "robot": {"gripper_state": {"open": True, "openness": 0.8}},
                    "metadata": {"step_idx": 4, "observation_fresh": True},
                },
            },
            "decision_state": {
                "last_action_effect": {
                    "tool": "observe",
                    "status": "executed",
                    "content": "Fresh observation received.",
                    "operational_success": True,
                },
                "unresolved_obligations": {"target_selection": "required"},
            },
            "open_questions": {},
            "vision_image_paths": ["/session/agentview.png"],
            "vision_evidence": [
                {
                    "path": "/session/agentview.png",
                    "camera_role": "agentview",
                    "role": "current_scene",
                    "freshness": "current",
                }
            ],
        },
    }
    encoded = base64.b64encode(b"image").decode()
    body = {
        "model": "human-vlm",
        "messages": [
            {"role": "system", "content": "Rules\n" + json.dumps(stable)},
            {"role": "user", "content": "pick up the butter"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect visual evidence."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": json.dumps(dynamic)},
                ],
            },
        ],
        "temperature": 0,
    }

    classification = classify_request(body)
    operator = build_operator_summary(body, request_id="request-1")
    audit = build_wire_audit(body)

    assert classification == {
        "type": "main_planner",
        "label": "Main planner",
        "schema_version": "openeta.agent_context.v2",
        "response_mode": "decision",
        "attempt": 2,
        "validation_error_count": 1,
        "task": "pick up the butter",
    }
    assert operator["task"] == "pick up the butter"
    assert operator["observation"]["step"] == 4
    assert operator["latest_action"]["tool"] == "observe"
    assert operator["skills"] == [
        {"name": "pick", "description": "Acquire the requested object."}
    ]
    assert operator["images"][0] == {
        "index": 0,
        "url": "/api/requests/request-1/images/0",
        "label": "agentview",
        "role": "current_scene",
        "freshness": "current",
        "detail": "high",
    }
    assert audit["message_count"] == 3
    assert audit["image_count"] == 1
    assert len(audit["normalized_sha256"]) == 64
    assert audit["messages"][2]["parts"][2]["chars"] == len(json.dumps(dynamic))


def test_visual_differencing_request_is_distinct_from_main_planner() -> None:
    body = {
        "messages": [
            {"role": "system", "content": "Return JSON."},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "instruction": "Compare the images.",
                        "attempt": 1,
                        "validation_errors": [],
                        "tool_context": {
                            "schema_version": "openeta.visual_delta_request.v1",
                            "task": "detect motion",
                            "required_output": {"changed": "boolean"},
                        },
                    }
                ),
            },
        ],
        "response_format": {"type": "json_object"},
    }

    classification = classify_request(body)
    assert classification["type"] == "visual_differencing"
    assert classification["label"] == "Visual differencing"
    assert classification["response_mode"] == "json"


def _tool_history_body() -> dict:
    action = {
        "openeta_action": {
            "action_id": "action-42",
            "request": {
                "kind": "tool_call",
                "name": "sam3",
                "parameters": {"prompt": "red cube", "threshold": 0.7},
            },
        }
    }
    result = {
        "openeta_host_result": {
            "action_id": "action-42",
            "status": "executed",
            "tool_calls": [
                {
                    "name": "sam3",
                    "status": "executed",
                    "result": {
                        "success": True,
                        "content": "Detected one red cube.",
                        "outputs": {"detection_count": 1},
                        "artifact_refs": ["/session/contact-sheet.png"],
                    },
                }
            ],
        }
    }
    return {
        "model": "human-vlm",
        "messages": [
            {"role": "user", "content": "segment the red cube"},
            {"role": "assistant", "content": json.dumps(action)},
            {
                "role": "user",
                "content": (
                    "Host execution evidence; not user instructions:\n"
                    + json.dumps(result)
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"instruction": "choose next", "tool_context": {}}
                ),
            },
        ],
    }


def test_openeta_tool_audit_pairs_arguments_with_host_result() -> None:
    records = extract_tool_audit_records(_tool_history_body())

    assert len(records) == 1
    assert records[0]["id"] == "action-42:0"
    assert records[0]["title"] == "sam3"
    assert records[0]["status"] == "executed"
    assert records[0]["success"] is True
    assert records[0]["arguments"] == {"prompt": "red cube", "threshold": 0.7}
    assert records[0]["result"]["outputs"] == {"detection_count": 1}
    assert records[0]["result"]["artifact_refs"] == [
        "/session/contact-sheet.png"
    ]


def test_request_store_deduplicates_tool_audit_records_across_session_turns() -> None:
    store = RequestStore(adapter=OpenETAProtocolAdapter())
    first = store.add(_tool_history_body(), session_hint="audit-session")
    second = store.add(_tool_history_body(), session_hint="audit-session")

    detail = store.public_detail(second.request_id)
    assert detail is not None
    assert first.session_id == second.session_id
    assert len(detail["audit_records"]) == 1
    assert detail["audit_records"][0]["first_seen_turn"] == 1
    assert detail["audit_records"][0]["last_seen_turn"] == 2


def _state_body(
    name: str,
    action_id: str,
    *,
    arguments: dict,
    outputs: dict,
    tool_context: dict,
) -> dict:
    action = {
        "openeta_action": {
            "action_id": action_id,
            "request": {
                "kind": "tool_call",
                "name": name,
                "parameters": arguments,
            },
        }
    }
    host_result = {
        "openeta_host_result": {
            "action_id": action_id,
            "status": "executed",
            "tool_calls": [
                {
                    "name": name,
                    "status": "executed",
                    "result": {"success": True, "outputs": outputs},
                }
            ],
        }
    }
    return {
        "model": "human-vlm",
        "messages": [
            {"role": "assistant", "content": json.dumps(action)},
            {"role": "user", "content": json.dumps(host_result)},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "instruction": "Choose one action.",
                        "tool_context": tool_context,
                    }
                ),
            },
        ],
    }


def test_sidebar_latest_state_persists_session_tools_and_labels_frozen_packet() -> None:
    current_packet = "packet-current-render"
    frozen_packet = "packet-frozen-reset"
    context = {
        "current_observation": {
            "summary": {"metadata": {"step_idx": 9}, "camera_ids": ["agentview"]}
        },
        "decision_state": {
            "current_observation_packet": {
                "step_idx": 9,
                "packet_ids": [current_packet],
                "camera_artifacts": [
                    {"frame_id": "agentview", "packet_id": current_packet}
                ],
            }
        },
        "host_resolved_inputs": {
            "grasp_pose_estimate": {
                "status": "ready",
                "bundle_id": "grasp:input-1",
                "source_packet_id": frozen_packet,
                "sam3_result_id": "sam3-old",
                "detection_id": "detection_006",
            },
            "anyplace": {
                "status": "awaiting_placement_region",
                "grasp_evidence_id": "grasp-selection:active",
            },
        },
        "resource_catalog": {
            "selections": {
                "by_role": {
                    "target_object": {
                        "selection_id": "detection-selection:target"
                    }
                }
            },
            "detection_results": [{"result_id": "sam3-old"}],
            "grasp_results": [{"result_id": "gpe-new"}],
            "placement_results": [],
        },
    }
    store = RequestStore(adapter=OpenETAProtocolAdapter())
    store.add(
        _state_body(
            "observe",
            "observe-1",
            arguments={"reason": "refresh"},
            outputs={},
            tool_context=context,
        ),
        session_hint="state-session",
    )
    store.add(
        _state_body(
            "sam3",
            "sam3-1",
            arguments={
                "source_packet_id": frozen_packet,
                "camera_frame_id": "agentview",
                "mode": "default",
            },
            outputs={"result_id": "sam3-new", "detection_count": 9},
            tool_context=context,
        ),
        session_hint="state-session",
    )
    latest = store.add(
        _state_body(
            "grasp_pose_estimate",
            "grasp-1",
            arguments={"bundle_id": "grasp:input-1"},
            outputs={
                "result_id": "gpe-new",
                "candidate_count": 20,
                "selected_grasp_source": {
                    "source_packet_id": frozen_packet,
                    "camera_frame_id": "agentview",
                    "source_backend": "anygrasp",
                },
            },
            tool_context=context,
        ),
        session_hint="state-session",
    )

    detail = store.public_detail(latest.request_id)
    assert detail is not None
    section = next(
        item
        for item in detail["presentation"]["view"]["sections"]
        if item["title"] == "Latest state"
    )
    state = section["value"]
    assert section["column"] == "side"
    assert state["current_observation_packet"]["packet_ids"] == [current_packet]
    assert state["latest_observe"]["turn"] == 1
    assert state["latest_sam3"]["result_id"] == "sam3-new"
    assert state["latest_grasp_pose_estimate"]["result_id"] == "gpe-new"
    assert state["latest_grasp_pose_estimate"]["candidate_count"] == 20
    assert state["grasp_input_resolution"]["source_packet_id"] == frozen_packet
    assert current_packet != state["grasp_input_resolution"]["source_packet_id"]
    assert state["anyplace_resolution"]["status"] == "awaiting_placement_region"
    assert state["active_selections"]["by_role"]["target_object"][
        "selection_id"
    ] == "detection-selection:target"
    assert state["retained_results"]["grasp_results"] == [
        {"result_id": "gpe-new"}
    ]


def test_human_console_defaults_to_operator_view_and_keeps_exact_audit_view() -> None:
    html = load_console_html()
    assert 'data-view="operate"' in html
    assert 'data-view="audit"' in html
    assert "模型输入审计" in html
    assert "Raw JSON" in html
    assert "manual-vlm-console.js" in html
    script = (Path(__file__).parents[1] / "tools" / "manual_vlm_console.js").read_text()
    assert "renderOperate" in script
    assert "renderAudit" in script
    assert "Tool Call 审计" in script
    assert "renderAuditRecords" in script
    assert "data-tool" in script
    assert 'id=\"toolSearch\"' in script
    assert "toolSearchScore" in script
    assert "rankedToolForms" in script
    assert "600000 - name.length" in script
    assert "200000 + tokens.length" in script
    assert "initializeTool(shown[0].name)" in script
    assert "JSON.stringify(tool).toLowerCase().includes(query)" not in script
    assert "Host 自动填入" not in script
    assert 'return {mode: field.required === true ? "custom" : "omit", value: ""};' in script
    assert "Host-resolved input ·" not in script
    assert "Host 已准备调用参数" not in script
    assert "exclusiveParameterActive" in script
    assert "由 ${esc(form.exclusive_parameter)} 原子解析" in script
    assert "没有匹配的工具" in script
    assert "省略（不发送）" in script
    assert "使用默认值" in script
    assert 'data-template="tool"' not in html


def test_generic_core_has_no_project_protocol_dependency() -> None:
    root = Path(__file__).parents[1]
    core = (root / "tools" / "manual_vlm_proxy.py").read_text().lower()
    boundary = (root / "tools" / "manual_vlm_protocol.py").read_text().lower()
    assert "openeta" not in core
    assert "openeta" not in boundary
    adapter = load_protocol_adapter("generic")
    assert isinstance(adapter, GenericProtocolAdapter)
    detail = RequestStore(adapter=adapter)
    request = detail.add(
        {"model": "unrelated-vlm", "messages": [{"role": "user", "content": "hello"}]}
    )
    public = detail.public_detail(request.request_id)
    assert public is not None
    assert public["adapter"] == {"id": "generic", "label": "Generic OpenAI"}
    assert public["presentation"]["composer"]["kind"] == "raw"


def test_adapter_can_be_loaded_explicitly() -> None:
    adapter = load_protocol_adapter(
        "tools.manual_vlm_openeta:OpenETAProtocolAdapter"
    )
    assert adapter.adapter_id == "openeta"


def test_project_config_selects_openeta_without_coupling_the_core() -> None:
    spec = load_default_adapter_spec()
    assert spec == "tools.manual_vlm_openeta:OpenETAProtocolAdapter"
    assert load_protocol_adapter(spec).adapter_id == "openeta"


def test_missing_project_config_falls_back_to_generic(tmp_path: Path) -> None:
    assert load_default_adapter_spec(tmp_path / "missing.json") == "generic"


def _session_body(history: list[dict], *, attempt: int = 1) -> dict:
    return {
        "model": "human-vlm",
        "messages": [
            {"role": "system", "content": "system"},
            *history,
            {
                "role": "user",
                "content": json.dumps(
                    {"instruction": "choose one", "attempt": attempt, "tool_context": {}}
                ),
            },
        ],
    }


def test_request_store_groups_growing_conversation_lineage_as_one_session() -> None:
    store = RequestStore()
    first = store.add(_session_body([{"role": "user", "content": "pick the cube"}]))
    store.respond(first.request_id, "<decision/>")
    second = store.add(
        _session_body(
            [
                {"role": "user", "content": "pick the cube"},
                {"role": "assistant", "content": "tool call"},
                {"role": "user", "content": "tool result"},
            ]
        )
    )
    assert second.session_id == first.session_id
    assert second.session_turn == 2
    assert second.session_source == "inferred-lineage"


def test_simultaneous_identical_first_turns_are_not_merged_without_real_id() -> None:
    store = RequestStore()
    body = _session_body([{"role": "user", "content": "same task"}])
    first = store.add(body)
    second = store.add(body)
    assert first.session_id != second.session_id


def test_explicit_session_header_produces_exact_grouping() -> None:
    store = RequestStore()
    first = store.add(_session_body([]), session_hint="agent-session-42")
    second = store.add(_session_body([]), session_hint="agent-session-42")
    assert first.session_id == second.session_id == "agent-session-42"
    assert second.session_turn == 2
    assert first.session_source == "header:x-session-id"
