from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from mcp.server.fastmcp.utilities.types import Image
from mcp.types import TextContent

import tools.embodied_mcp_server as server
from tools.embodied_gateway import GatewayResult


def test_operator_server_instructions_can_be_absent_without_fallback_text() -> None:
    assert server._operator_server_instructions("none_v1") is None
    assert "One LIBERO episode is active" in str(
        server._operator_server_instructions("compact_v1")
    )


def test_operator_mcp_result_contains_text_and_native_image_block(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "observation.png"
    image.write_bytes(b"not-a-real-image-but-the-block-is-path-backed")

    class FakeGateway:
        def observe(self, views=None) -> GatewayResult:
            return GatewayResult(
                True,
                {
                    "kind": "observation",
                    "success": True,
                    "observation_id": "obs-1",
                    "returned_views": ["agentview"],
                    "actual_grip_site_xyz_m": [0.0, 0.0, 0.5],
                    "gripper_aperture_mm": 80.0,
                },
                images=[image],
            )

    monkeypatch.setattr(server, "_GATEWAY", FakeGateway())
    result = server.observe()
    assert isinstance(result[0], TextContent)
    assert result[0].type == "text"
    assert isinstance(result[1], Image)
    assert result[1].path == image


def test_gripper_only_failure_uses_generic_public_error_contract() -> None:
    result = GatewayResult(
        False,
        {
            "kind": "move_to",
            "success": False,
            "motion_status": "not_requested",
            "actual_grip_site_xyz_m": [0.2, -0.01, 1.2],
            "returned_views": ["agentview", "wrist"],
            "issue_code": "remote_action_failed",
            "message": "Step failed: executing action in terminated episode",
            "retryable": False,
        },
    )

    blocks = server._content(result, public_tool="move_to")

    assert json.loads(blocks[0].text) == {
        "success": False,
        "reason": "remote_action_failed",
        "retryable": False,
        "message": "Step failed: executing action in terminated episode",
    }


def test_internal_abort_control_finalizes_live_episode(tmp_path: Path, monkeypatch) -> None:
    class FakeGateway:
        root = tmp_path

        def finish_episode(self, outcome: str, *, reason: str = "") -> GatewayResult:
            assert outcome == "abort"
            return GatewayResult(
                True,
                {
                    "kind": "episode_finish",
                    "success": True,
                    "outcome": "abort",
                    "episode_status": "aborted",
                    "message": reason,
                },
            )

    monkeypatch.setattr(server, "_GATEWAY", FakeGateway())
    monkeypatch.setattr(server, "_CONTEXT_SEQ", 0)
    http = ThreadingHTTPServer(("127.0.0.1", 0), server._ManualControlHandler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{http.server_port}/abort",
            data=b'{"reason":"stream stopped"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            payload = json.load(response)
        assert payload["success"] is True
        assert payload["text"]["outcome"] == "abort"
        context = (tmp_path / "operator_context.jsonl").read_text()
        assert '"tool": "manual_abort"' in context
    finally:
        http.shutdown()
        http.server_close()


def test_internal_abort_preserves_latched_native_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeGateway:
        root = tmp_path

        def __init__(self) -> None:
            self.outcomes: list[str] = []

        def finish_episode(self, outcome: str, *, reason: str = "") -> GatewayResult:
            self.outcomes.append(outcome)
            if outcome == "abort":
                return GatewayResult(
                    False,
                    {
                        "success": False,
                        "reason": "task_already_completed",
                    },
                )
            assert outcome == "success"
            return GatewayResult(
                True,
                {
                    "kind": "episode_finish",
                    "success": True,
                    "outcome": "success",
                    "episode_status": "completed",
                    "message": reason,
                },
            )

    gateway = FakeGateway()
    monkeypatch.setattr(server, "_GATEWAY", gateway)
    monkeypatch.setattr(server, "_CONTEXT_SEQ", 0)
    http = ThreadingHTTPServer(("127.0.0.1", 0), server._ManualControlHandler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{http.server_port}/abort",
            data=b'{"reason":"stream stopped"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            payload = json.load(response)
        assert gateway.outcomes == ["abort", "success"]
        assert payload["success"] is True
        assert payload["text"]["outcome"] == "success"
        context = (tmp_path / "operator_context.jsonl").read_text()
        assert '"tool": "manual_finalize_success"' in context
        assert '"tool": "manual_abort"' not in context
    finally:
        http.shutdown()
        http.server_close()


def test_failure_evidence_is_derived_from_durable_action_trace(
    tmp_path: Path,
) -> None:
    events = [
        {
            "kind": "action",
            "action_id": "action-000001",
            "payload": {"request": {"stage": "close_gripper"}},
        },
        {
            "kind": "tool_result",
            "action_id": "action-000001",
            "payload": {"success": True},
        },
        {
            "kind": "action",
            "action_id": "action-000002",
            "payload": {"request": {"stage": "move_to_selected_grasp"}},
        },
        {
            "kind": "tool_result",
            "action_id": "action-000002",
            "payload": {"success": False},
        },
    ]
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    class FakeGateway:
        root = tmp_path
        current_record = {"observation_id": "obs-000009"}

    evidence, attempts = server._failure_evidence_from_trace(FakeGateway())

    assert evidence == [
        "obs-000009",
        "action-000001",
        "action-000002",
    ]
    assert attempts == [
        "action-000001: close_gripper (tool_result_success=true)",
        "action-000002: move_to_selected_grasp (tool_result_success=false)",
    ]


def test_compact_failure_finish_does_not_require_operator_to_repeat_attempts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "events.jsonl").write_text(
        json.dumps(
            {
                "kind": "action",
                "action_id": "action-000001",
                "payload": {"request": {"stage": "close_gripper"}},
            }
        )
        + "\n"
        + json.dumps(
            {
                "kind": "tool_result",
                "action_id": "action-000001",
                "payload": {"success": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeGateway:
        root = tmp_path
        task = "test task"
        current_record = {"observation_id": "obs-000003"}

        def finish_episode(
            self,
            outcome: str,
            *,
            reason: str,
            failure_postmortem,
            operator_feedback,
        ) -> GatewayResult:
            assert outcome == "failure"
            assert reason == "object was not retained"
            assert failure_postmortem["evidence_refs"] == [
                "obs-000003",
                "action-000001",
            ]
            assert failure_postmortem["recovery_attempts"] == [
                "action-000001: close_gripper (tool_result_success=true)"
            ]
            assert operator_feedback is None
            return GatewayResult(
                True,
                {
                    "kind": "episode_finish",
                    "success": True,
                    "outcome": "failure",
                },
            )

    monkeypatch.setattr(server, "_GATEWAY", FakeGateway())

    blocks = server._finish_episode_compact(
        "failure",
        reason="object was not retained",
    )

    payload = json.loads(blocks[0].text)
    assert payload["success"] is True
    assert payload["outcome"] == "failure"


def test_compact_failure_finish_requests_public_reason_not_hidden_postmortem(
    monkeypatch,
) -> None:
    class FakeGateway:
        def finish_episode(self, *_args, **_kwargs) -> GatewayResult:
            raise AssertionError("blank failure reason must be rejected at the public boundary")

    monkeypatch.setattr(server, "_GATEWAY", FakeGateway())
    monkeypatch.setattr(server, "_COMPACT_INPUT_SCHEMA", True)

    blocks = server._finish_episode_compact("failure")

    payload = json.loads(blocks[0].text)
    assert payload == {
        "success": False,
        "reason": "failure_reason_required",
        "retryable": True,
        "required_fields": ["reason"],
    }


def test_exact_operator_context_projection_is_retained(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "pose.png"
    image.write_bytes(b"png")

    class FakeGateway:
        root = tmp_path

    monkeypatch.setattr(server, "_GATEWAY", FakeGateway())
    monkeypatch.setattr(server, "_CONTEXT_SEQ", 0)
    server._record_operator_context(
        tool="refine_grasp_pose",
        arguments={
            "base_grasp_id": "grasp_000",
            "translation_delta_mm": [0, 0, 20],
        },
        blocks=[
            TextContent(type="text", text='{"kind":"grasp_pose_refinement","success":true}'),
            Image(path=image),
        ],
    )

    event = json.loads((tmp_path / "operator_context.jsonl").read_text())
    assert event["seq"] == 1
    assert event["tool"] == "refine_grasp_pose"
    assert event["arguments"]["translation_delta_mm"] == [0, 0, 20]
    assert event["response_text_blocks"] == [
        '{"kind":"grasp_pose_refinement","success":true}'
    ]
    assert event["response_image_paths"] == [str(image)]


def test_operator_context_projection_keeps_nested_pydantic_arguments_structured(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeGateway:
        root = tmp_path

    intervention = server.SingleVariableIntervention(
        independent_variable="grip-site Z",
        control_condition="current Z",
        treatment_condition="Z+5 mm",
        held_constant=["seed", "task", "orientation"],
        predicted_effect="retention improves",
        primary_metric="close-lift retention",
        adoption_criterion="repeatable matched-trial improvement",
        execution_scope="current_tools",
        current_tool_plan=["move_to Z+5 mm", "lift and inspect"],
        attempted_in_episode=False,
        attempt_evidence_refs=[],
        planned_current_tool_trials=1,
        completed_current_tool_trials=0,
        remaining_current_tool_actions=["move_to Z+5 mm", "lift and inspect"],
        contradicting_attempts=[],
        operator_claimed_exhaustion_reason="",
    )
    postmortem = server.FailurePostmortem(
        progress_stopped_at="retention",
        expected_observation="The bowl follows the gripper.",
        actual_observation="The bowl remains on the table.",
        evidence_refs=["obs-8 agentview"],
        recovery_attempts=["centered close did not retain"],
        diagnostic_hypotheses=[
            server.DiagnosticHypothesis(
                suspected_layer="gripper_or_contact",
                explanation="The grip site may be vertically misaligned.",
                supporting_evidence="The fingers closed above the bowl.",
                missing_or_conflicting_evidence="Contacts are unavailable.",
                confidence="medium",
            )
        ],
        proposed_intervention=intervention,
    )
    monkeypatch.setattr(server, "_GATEWAY", FakeGateway())
    monkeypatch.setattr(server, "_CONTEXT_SEQ", 0)

    server._record_operator_context(
        tool="finish_episode",
        arguments={
            "outcome": "failure",
            "failure_postmortem": postmortem,
        },
        blocks=[TextContent(type="text", text='{"success":false}')],
    )

    event = json.loads((tmp_path / "operator_context.jsonl").read_text())
    recorded = event["arguments"]["failure_postmortem"]
    assert isinstance(recorded, dict)
    assert recorded["progress_stopped_at"] == "retention"
    assert recorded["proposed_intervention"]["execution_scope"] == (
        "current_tools"
    )
    assert event["response_text_blocks"] == ['{"success":false}']
    assert event["response_image_paths"] == []
