from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from agent.runtime.planner import _parse_backend_payload
from tools.manual_vlm_proxy import (
    ManualVLMServer,
    RequestStore,
    build_operator_summary,
    build_tool_form_catalog,
    build_wire_audit,
    classify_request,
    detect_response_mode,
    extract_tool_catalog,
    load_console_html,
    serialize_decision_xml,
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


def _start_server() -> tuple[ManualVLMServer, str, threading.Thread]:
    server = ManualVLMServer(
        ("127.0.0.1", 0),
        store=RequestStore(),
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
    server, base, thread = _start_server()
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
            base + f"/api/requests/{request_id}/response", {"decision": decision}
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
        assert "OpenETA Human VLM" in html
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
    assert detect_response_mode(body) == "xml"


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
        "response_mode": "xml",
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


def test_human_console_defaults_to_operator_view_and_keeps_exact_audit_view() -> None:
    html = load_console_html()
    assert 'data-view="operate"' in html
    assert 'data-view="audit"' in html
    assert "模型输入审计" in html
    assert "Raw JSON" in html
    assert "renderOperator" in html
    assert "renderAudit" in html
    assert "composerToolSelect" in html
    assert "省略（不发送）" in html
    assert "使用默认值" in html
    assert 'data-template="tool"' not in html


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
    assert first.session_source == "header"
