from __future__ import annotations

import json

import pytest

from adapter.protocol import CameraFrame, EnvAction, EnvObservation, RobotState
from agent.backends.planner import PlannerBackend, PlannerBackendRequest, PlannerBackendResult
from agent.runtime.actions import PipelineStatus
from agent.runtime.memory import AgentMemory
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import (
    PlannerDecision,
    ToolCallingPlanner,
    _project_latest_tool_outputs,
    build_tool_context,
)
from agent.runtime.skills import build_default_skill_registry
from agent.tools.registry import build_default_tool_registry


class RecordingBackend(PlannerBackend):
    def __init__(self) -> None:
        self.requests: list[PlannerBackendRequest] = []

    def decide(self, request: PlannerBackendRequest) -> PlannerBackendResult:
        self.requests.append(request)
        return PlannerBackendResult(
            payload={
                "kind": "response",
                "name": "talk",
                "parameters": {"message": "state assessed"},
                "reasoning": "current evidence inspected",
            }
        )


def _observation() -> EnvObservation:
    return EnvObservation(
        task="pick the cube",
        cameras=[
            CameraFrame(
                frame_id="agentview",
                role="scene_primary",
                rgb=[[[0, 0, 0]]],
                timestamp_s=12.5,
            )
        ],
        robot=RobotState(gripper_state={"open": 1.0}),
        objects=[{"name": "cube"}],
        metadata={
            "step_idx": 3,
            "image_artifacts": [
                {
                    "kind": "rgb",
                    "frame_id": "agentview",
                    "role": "scene_primary",
                    "path": "/session/current.png",
                }
            ],
        },
    )


def test_agent_context_prioritizes_current_evidence_and_agent_memory() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    memory.save_fact("agent_plan", {"next_subgoal": "locate cube"}, source="save_memory")

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )
    agent_context = context["agent_context"]

    assert agent_context["schema_version"] == "openeta.agent_context.v2"
    assert agent_context["objective"]["task"] == "pick the cube"
    assert agent_context["current_observation"]["status"] == "available"
    evidence = agent_context["current_observation"]["visual_evidence"]
    assert evidence[0]["evidence_id"] == "current_observation:3:agentview"
    assert evidence[0]["freshness"] == "current"
    assert agent_context["agent_working_state"]["facts"] == {
        "agent_plan": memory.agent_working_state["agent_plan"]
    }
    assert "required_action" not in str(agent_context)
    compatibility_payload = {key: value for key, value in context.items() if key != "agent_context"}
    assert len(json.dumps(agent_context)) < len(json.dumps(compatibility_payload))


def test_latest_probe_projection_keeps_short_ik_handoff_and_omits_pose_payload() -> None:
    short_request = {
        "tool": "ik_preview_check",
        "parameters": {"probe_id": "probe:short", "waypoint_index": 0},
    }
    projected = _project_latest_tool_outputs(
        "prepare_attachment_probe",
        {
            "schema_version": "openeta.articulated_attachment_probe.v1",
            "probe_id": "probe:short",
            "frozen_path": [{"xyz": [0.1, 0.2, 0.3], "trace": "x" * 12_000}],
            "ik_preview_requests": [short_request],
            "execution_handoff": {
                "tool": "move_to",
                "parameters": {"ik_receipt_id": "<receipt>"},
            },
        },
    )

    assert isinstance(projected, dict)
    assert projected["ik_preview_requests"] == [short_request]
    assert "frozen_path" not in projected


def test_agent_context_bounds_artifact_index_and_does_not_duplicate_tool_schemas() -> None:
    memory = AgentMemory()
    memory.start_session(task="inspect accumulated artifacts")
    for index in range(50):
        memory.save_artifact(
            f"artifact-{index:02d}",
            {
                "type": "diagnostic",
                "path": f"/session/artifact-{index:02d}.json",
                "payload": "x" * 2_000,
            },
            source="test",
        )

    tools = build_default_tool_registry()
    tools.bind_handler("observe", lambda _context: {"success": True})
    agent_context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )["agent_context"]

    artifacts = agent_context["artifacts"]
    assert artifacts["__index__"]["total_count"] == 50
    assert artifacts["__index__"]["truncated"] is True
    assert "artifact-49" in artifacts
    assert "artifact-00" not in artifacts
    assert "python_exec" in artifacts["__index__"]["query"]["files"]
    assert all(set(reference) == {"name"} for reference in agent_context["tool_references"])
    assert any("description" in reference for reference in agent_context["available_tools"])


def test_observation_disambiguates_measured_aperture_from_close_command() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    memory.save_fact(
        "gripper_command_state",
        {"position": 0, "state": "closed", "latched": True, "scene_epoch": 1},
        source="acknowledged_gripper_command",
    )
    observation = _observation()
    observation.robot.gripper_state = {"open": True, "openness": 0.67}

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    evidence = context["agent_context"]["current_observation"]["summary"]["gripper_evidence"]
    assert evidence["measured_aperture"] == {
        "open_fraction": 0.67,
        "legacy_threshold_open": True,
    }
    assert evidence["commanded_state"]["position"] == 0
    assert evidence["attachment_status"] == "unknown_without_co_motion_evidence"
    assert "not a command" in evidence["semantics"]
    assert "not attachment proof" in evidence["semantics"]


def test_main_planner_backend_receives_agent_owned_context() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    backend = RecordingBackend()
    planner = ToolCallingPlanner(backend=backend)

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    assert len(backend.requests) == 1
    assert decision.action_type == "response"
    assert decision.action == "talk"
    sent = backend.requests[0].tool_context
    assert sent["schema_version"] == "openeta.agent_context.v2"
    assert sent["vision_evidence"][0]["role"] == "current_scene"
    assert "You own task decomposition" in backend.requests[0].system_prompt


def test_removed_task_policy_facts_are_purged_when_a_session_resumes(tmp_path) -> None:
    root = tmp_path / ".openeta_memory"
    memory = AgentMemory(store=JsonMemoryStore(root))
    memory.start_session(task="pick the cube", session_id="session-1")
    memory.save_fact("agent_plan", {"next_subgoal": "locate cube"}, source="save_memory")
    legacy_entry = {
        "value": {"status": "required", "stage": "descend"},
        "source": "save_memory",
        "timestamp_s": 1.0,
    }
    memory.facts["grasp_execution"] = dict(legacy_entry)
    memory.agent_working_state["grasp_execution"] = {
        **legacy_entry,
        "ownership": "agent",
        "freshness": "agent_managed",
    }
    memory._save_working_memory()

    resumed = AgentMemory(store=JsonMemoryStore(root))
    resumed.resume_session("session-1")

    assert set(resumed.agent_working_state) == {"agent_plan"}
    assert resumed.agent_working_state["agent_plan"]["ownership"] == "agent"
    assert "grasp_execution" not in resumed.facts
    persisted = JsonMemoryStore(root).load_working_memory()
    assert "grasp_execution" not in persisted["facts"]
    assert "grasp_execution" not in persisted["agent_working_state"]


def test_removed_task_policy_names_cannot_be_recreated() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")

    with pytest.raises(ValueError, match="reserved tombstone"):
        memory.save_fact("grasp_execution", {"stage": "close"}, source="save_memory")


def test_world_evidence_is_derived_with_provenance_and_scene_freshness() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    memory.save_fact(
        "selected_sam3_detection",
        {"id": "detection-1", "scene_epoch": 0},
        source="sam3_selection",
    )
    memory.save_fact(
        "gripper_command_state",
        {"position": 0, "scene_epoch": 0},
        source="gripper_control",
    )

    current = memory.world_evidence_context()
    assert current["selected_target"]["freshness"] == "current_scene_epoch"
    assert current["selected_target"]["provenance"]["source"] == "sam3_selection"
    assert current["gripper_command"]["freshness"] == "commanded_not_observed"

    memory.save_fact("scene_epoch", {"epoch": 1}, source="runtime")
    assert memory.world_evidence_context()["selected_target"]["freshness"] == "stale_scene_epoch"


def test_agent_context_projects_ik_receipt_bank_without_mutating_memory() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    receipts = [
        {
            "receipt_id": f"ik-{index}",
            "classification": "reachable",
            "reason_code": "ik_solution_found",
            "orientation_policy": "explicit",
            "pose_policy_signature": f"pose-{index}",
            "object_scene_epoch": 0,
            "robot_motion_epoch": index,
            "target_pose": {
                "xyz": [0.1, 0.2, 0.3],
                "waypoint_role": "grasp_contact",
                "rotation_matrix": [[1.0, 0.0, 0.0]] * 3,
            },
            "suggestions": ["large diagnostic payload " * 50],
        }
        for index in range(12)
    ]
    memory.save_fact(
        "ik_preview_receipts",
        {
            "schema_version": "openeta.ik_preview_receipt_index.v1",
            "latest": receipts[-1],
            "receipts": receipts,
        },
        source="ik_preview_check",
    )

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )["agent_context"]
    projected = context["world_evidence"]["ik_preview_receipts"]["value"]

    assert projected["receipt_count"] == 12
    assert len(projected["index"]) == 12
    assert projected["latest"]["receipt_id"] == "ik-11"
    assert "suggestions" not in projected["index"][0]
    assert len(memory.ik_preview_receipts()["receipts"]) == 12


def test_wrist_segmentation_miss_does_not_invalidate_coarse_grasp_evidence() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    memory.save_fact(
        "sam3_no_detection",
        {
            "result_id": "sam3-wrist-miss",
            "frame_id": "wrist",
            "source_packet_id": "packet-wrist-4",
            "target_prompt": "cube",
            "scene_epoch": 0,
        },
        source="sam3",
    )

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )["agent_context"]
    impact = context["open_questions"]["perception_failure"]["workflow_impact"]

    assert impact["classification"] == "optional_wrist_refinement_unavailable"
    assert impact["coarse_grasp_invalidated"] is False
    assert impact["host_policy"] == "advisory_only"


def test_infrastructure_failures_open_advisory_tool_health_circuit() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")

    def record(*, success: bool, failure_code: str = "mcp_timeout") -> None:
        memory.add_action(
            EnvAction(
                action_type="tool_call",
                command={
                    "status": "executed" if success else "failed",
                    "request": {
                        "kind": "tool_call",
                        "name": "molmopoint",
                        "parameters": {"sources": []},
                    },
                    "tool_calls": [
                        {
                            "name": "molmopoint",
                            "status": "executed" if success else "failed",
                            "parameters": {"sources": []},
                            "result": {
                                "success": success,
                                "content": "ok" if success else "backend timed out",
                                "details": {
                                    "diagnostics": []
                                    if success
                                    else [{"code": failure_code}],
                                },
                            },
                        }
                    ],
                },
            )
        )

    record(success=False)
    assert memory.tool_health()["molmopoint"]["status"] == "degraded"
    record(success=False)

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )["agent_context"]
    health = context["decision_state"]["tool_health"]["molmopoint"]
    assert health["status"] == "circuit_open"
    assert health["host_enforcement"] == "advisory_only"

    # A structured semantic failure proves that the backend answered, so it
    # closes the infrastructure circuit even though the task result failed.
    record(success=False, failure_code="inconsistent_point_outputs")
    assert memory.tool_health()["molmopoint"]["status"] == "healthy"

    record(success=True)
    assert memory.tool_health()["molmopoint"]["status"] == "healthy"
    assert memory.tool_health()["molmopoint"]["consecutive_infrastructure_failures"] == 0


def test_transition_projection_coalesces_repeated_zero_reward_receipts() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    for index in range(2):
        memory.add_action(
            EnvAction(
                action_type="tool_call",
                command={
                    "status": "executed",
                    "request": {
                        "kind": "tool_call",
                        "name": "python_exec",
                        "parameters": {"code": f"result = {index}"},
                    },
                    "tool_calls": [
                        {
                            "name": "python_exec",
                            "status": "executed",
                            "result": {"success": True, "details": {"outputs": {}}},
                        }
                    ],
                },
            )
        )
        memory.record_environment_receipt(
            reward=0.0,
            terminated=False,
            truncated=False,
        )

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )["agent_context"]
    projected = context["transition_ledger"]

    assert len(memory.transition_ledger()) == 4
    assert [row["tool"] for row in projected] == [
        "python_exec",
        "python_exec",
        "environment_receipt",
    ]
    assert "latest of 2" in projected[-1]["projection_note"]


def test_compiled_grasp_is_retained_as_read_only_evidence() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    compiled = {
        "schema_version": "openeta.compiled_grasp_seed.v1",
        "compiled_grasp_id": "compiled-1",
        "candidate_id": "grasp-1",
        "scene_epoch": 0,
        "approach_world_xyz": [0.0, 0.0, -1.0],
        "hover_pose": {"frame": "world", "xyz": [0.1, 0.2, 0.3]},
        "contact_pose": {"frame": "world", "xyz": [0.1, 0.2, 0.15]},
    }
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {"kind": "tool_call", "name": "compile_grasp_seed"},
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "compile_grasp_seed",
                        "status": "executed",
                        "result": {"success": True, "details": {"outputs": compiled}},
                    }
                ],
            },
        )
    )

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )["agent_context"]
    retained = next(
        artifact for artifact in context["artifacts"].values() if artifact.get("type") == "compiled_grasp"
    )
    assert retained["compiled_grasp_id"] == "compiled-1"
    assert retained["hover_pose"]["xyz"] == [0.1, 0.2, 0.3]
    assert retained["contact_pose"]["xyz"] == [0.1, 0.2, 0.15]


def test_robot_motion_and_object_scene_epochs_are_independent() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")

    def successful_action(name: str, parameters: dict) -> EnvAction:
        return EnvAction(
            action_type="tool_call",
            command={
                "request": {"kind": "tool_call", "name": name, "parameters": parameters},
                "status": "executed",
                "tool_calls": [
                    {
                        "name": name,
                        "status": "executed",
                        "parameters": parameters,
                        "result": {"success": True, "details": {"outputs": {}}},
                    }
                ],
            },
        )

    memory.add_action(successful_action("move_to", {"target_pose": {"frame": "world", "xyz": [0, 0, 1]}}))
    assert memory.robot_motion_epoch() == 1
    assert memory.object_scene_epoch() == 0

    memory.add_action(successful_action("gripper_control", {"position": 0}))
    assert memory.robot_motion_epoch() == 2
    assert memory.object_scene_epoch() == 0

    memory.add_action(successful_action("gripper_control", {"position": 1}))
    assert memory.robot_motion_epoch() == 3
    assert memory.object_scene_epoch() == 0

    changed = _observation()
    changed.metadata["object_scene_change"] = {
        "changed": True,
        "change_id": "cube-displaced-1",
        "reason": "fresh dual-view evidence shows object displacement",
    }
    memory.add_observation(changed)
    assert memory.robot_motion_epoch() == 3
    assert memory.object_scene_epoch() == 1

    # Re-materializing the same observation evidence is idempotent.
    memory.add_observation(changed)
    assert memory.object_scene_epoch() == 1


def test_zero_step_motion_does_not_advance_robot_motion_epoch() -> None:
    memory = AgentMemory()
    memory.start_session(task="reach the target")

    def move_action(*, steps: int, start: list[float], end: list[float]) -> EnvAction:
        parameters = {"target_pose": {"frame": "world", "xyz": [0.1, 0.2, 0.3]}}
        return EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": parameters,
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "executed",
                        "parameters": parameters,
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "motion_summary": {
                                        "steps_executed": steps,
                                        "start": {"xyz": start},
                                        "end": {"xyz": end},
                                        "reached_target": True,
                                    }
                                }
                            },
                        },
                    }
                ],
            },
        )

    memory.add_action(
        move_action(
            steps=0,
            start=[0.09, 0.2, 0.3],
            end=[0.09, 0.2, 0.3],
        )
    )
    assert memory.robot_motion_epoch() == 0
    assert any(event.event_type == "world_mutation_noop" for event in memory.events)

    memory.add_action(
        move_action(
            steps=1,
            start=[0.09, 0.2, 0.3],
            end=[0.1, 0.2, 0.3],
        )
    )
    assert memory.robot_motion_epoch() == 1


def test_failed_motion_with_partial_execution_advances_robot_motion_epoch() -> None:
    memory = AgentMemory()
    memory.start_session(task="reach the target")
    parameters = {"target_pose": {"frame": "world", "xyz": [0.1, 0.2, 0.3]}}
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": parameters,
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "executed",
                        "parameters": parameters,
                        "result": {
                            "success": False,
                            "details": {
                                "operational_success": False,
                                "semantic_outcome": "target_not_reached",
                                "outputs": {
                                    "motion_summary": {
                                        "steps_executed": 11,
                                        "start": {"xyz": [0.0, 0.2, 0.3]},
                                        "end": {"xyz": [0.09, 0.2, 0.3]},
                                        "reached_target": False,
                                        "stop_reason": "control_step_failed",
                                    }
                                },
                            },
                        },
                    }
                ],
            },
        )
    )

    assert memory.robot_motion_epoch() == 1
    advanced = [
        event
        for event in memory.events
        if event.event_type == "world_epochs_advanced"
    ]
    assert advanced[-1].payload["robot_motion_epoch"] == 1


def test_failed_motion_without_execution_receipt_does_not_advance_epoch() -> None:
    memory = AgentMemory()
    memory.start_session(task="reach the target")
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {"kind": "tool_call", "name": "move_to"},
                "status": "blocked",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "skipped",
                        "result": {
                            "success": False,
                            "details": {"operational_success": False},
                        },
                    }
                ],
            },
        )
    )

    assert memory.robot_motion_epoch() == 0


def test_move_requires_exact_current_feasible_ik_receipt() -> None:
    memory = AgentMemory()
    memory.start_session(task="reach the target")
    target_pose = {
        "frame": "world",
        "xyz": [0.8, 0.0, 0.1],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "ik_preview_check",
                    "parameters": {"target_pose": target_pose},
                },
                "status": "failed",
                "tool_calls": [
                    {
                        "name": "ik_preview_check",
                        "status": "failed",
                        "parameters": {"target_pose": target_pose},
                        "result": {
                            "success": False,
                            "details": {
                                "outputs": {
                                    "ik_preview_receipt": {
                                        "schema_version": "openeta.ik_preview_receipt.v1",
                                        "receipt_id": "ik-hard-1",
                                        "classification": "hard_infeasible",
                                        "reason_code": "target_outside_workspace",
                                    }
                                }
                            },
                        },
                    }
                ],
            },
        )
    )

    pipeline = ActionPipeline()
    tools = build_default_tool_registry()
    skills = build_default_skill_registry()
    exact = pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="move_to",
            parameters={"ik_receipt_id": "ik-hard-1"},
        ),
        observation=_observation(),
        tools=tools,
        skills=skills,
        memory=memory,
    )
    assert exact.status == PipelineStatus.BLOCKED
    repair = exact.metadata["repair_bundle"]
    assert repair["code"] == "ik_target_hard_infeasible"
    assert repair["latest_ik_preview"]["receipt_id"] == "ik-hard-1"
    assert any(call["tool"] == "observe" for call in repair["allowed_next_calls"])

    adjusted_pose = {**target_pose, "xyz": [0.78, 0.0, 0.1]}
    adjusted = pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="move_to",
            parameters={"target_pose": adjusted_pose},
        ),
        observation=_observation(),
        tools=tools,
        skills=skills,
        memory=memory,
    )
    assert adjusted.status == PipelineStatus.BLOCKED
    assert adjusted.metadata["repair_bundle"]["code"] == (
        "invalid_ik_receipt_reference"
    )
    assert "ik_receipt_id" in adjusted.tool_calls[0].reason

    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "ik_preview_check",
                    "parameters": {"target_pose": adjusted_pose},
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "ik_preview_check",
                        "status": "executed",
                        "parameters": {"target_pose": adjusted_pose},
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "ik_preview_receipt": {
                                        "schema_version": "openeta.ik_preview_receipt.v1",
                                        "receipt_id": "ik-feasible-adjusted",
                                        "classification": "feasible",
                                    }
                                }
                            },
                        },
                    }
                ],
            },
        )
    )
    authorized = pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="move_to",
            parameters={"ik_receipt_id": "ik-feasible-adjusted"},
        ),
        observation=_observation(),
        tools=tools,
        skills=skills,
        memory=memory,
    )
    assert authorized.status != PipelineStatus.BLOCKED


def test_partial_failed_motion_invalidates_preexisting_ik_receipt() -> None:
    memory = AgentMemory()
    memory.start_session(task="reach the target")
    target_pose = {
        "frame": "world",
        "xyz": [0.2, 0.1, 0.3],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "ik_preview_check",
                    "parameters": {"target_pose": target_pose},
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "ik_preview_check",
                        "status": "executed",
                        "parameters": {"target_pose": target_pose},
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "ik_preview_receipt": {
                                        "schema_version": "openeta.ik_preview_receipt.v1",
                                        "receipt_id": "ik-before-partial-failure",
                                        "classification": "feasible",
                                    }
                                }
                            },
                        },
                    }
                ],
            },
        )
    )
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {"target_pose": target_pose},
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "executed",
                        "parameters": {"target_pose": target_pose},
                        "result": {
                            "success": False,
                            "details": {
                                "operational_success": False,
                                "outputs": {
                                    "motion_summary": {
                                        "steps_executed": 3,
                                        "start": {"xyz": [0.0, 0.0, 0.3]},
                                        "end": {"xyz": [0.03, 0.0, 0.3]},
                                        "reached_target": False,
                                    }
                                },
                            },
                        },
                    }
                ],
            },
        )
    )

    blocked = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="move_to",
            parameters={"ik_receipt_id": "ik-before-partial-failure"},
        ),
        observation=_observation(),
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert blocked.status is PipelineStatus.BLOCKED
    assert blocked.metadata["repair_bundle"]["code"] == "invalid_ik_receipt_reference"
    assert "stale" in blocked.tool_calls[0].reason


def test_repairable_ik_pose_remains_available_as_adjustment_anchor() -> None:
    memory = AgentMemory()
    memory.start_session(task="reach the target")
    target_pose = {
        "frame": "world",
        "xyz": [0.4, 0.0, 0.2],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "ik_preview_check",
                    "parameters": {"target_pose": target_pose},
                },
                "status": "failed",
                "tool_calls": [
                    {
                        "name": "ik_preview_check",
                        "status": "failed",
                        "parameters": {"target_pose": target_pose},
                        "result": {
                            "success": False,
                            "details": {
                                "outputs": {
                                    "ik_preview_receipt": {
                                        "schema_version": "openeta.ik_preview_receipt.v1",
                                        "receipt_id": "ik-repairable-1",
                                        "classification": "repairable",
                                        "reason_code": "full_pose_infeasible",
                                    }
                                }
                            },
                        },
                    }
                ],
            },
        )
    )
    error = memory.ik_execution_gate_error(
        tool_name="move_to", parameters={"target_pose": target_pose}
    )
    assert error is not None
    assert error.startswith("ik_preview_not_feasible:")
    assert "useful repair evidence" in error


def test_wrist_alignment_bundle_hides_host_paths_from_planner_call() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    memory.save_fact(
        "selected_sam3_detection",
        {
            "result_id": "sam-wrist",
            "id": "detection_000",
            "evidence_role": "target_object",
            "source_packet_id": "packet-wrist",
            "frame_id": "robot0_eye_in_hand",
            "camera_role": "wrist",
            "mask_ref": "/session/wrist-mask.png",
            "scene_epoch": 0,
                "source_observation": {
                "packet_id": "packet-wrist",
                "frame_id": "robot0_eye_in_hand",
                    "role": "wrist",
                    "object_scene_epoch": 0,
                    "robot_motion_epoch": 0,
                "depth": "/session/wrist-depth.png",
                "intrinsics": {
                    "fx": 100.0,
                    "fy": 100.0,
                    "cx": 32.0,
                    "cy": 32.0,
                    "scale": 1000.0,
                    "width": 64,
                    "height": 64,
                },
                "extrinsics": {"type": "T_gripper_cam", "frame": "eef"},
                "current_eef_pose": {
                    "xyz": [0.1, 0.2, 0.3],
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        },
        source="test",
    )
    memory.save_artifact(
        "compiled-wrist",
        {
            "type": "compiled_grasp",
            "schema_version": "openeta.compiled_grasp_seed.v1",
            "compiled_grasp_id": "compiled-wrist",
        },
        source="test",
    )
    memory.save_fact(
        "grasp_provenance",
        {"compiled_grasp_id": "compiled-wrist", "object_scene_epoch": 0},
        source="test",
    )

    assert memory._refresh_wrist_alignment_bundle() is True
    public = memory.wrist_alignment_bundle()
    assert public["status"] == "ready"
    assert public["call_parameters"] == {"bundle_id": public["bundle_id"]}
    resolved = memory.resolve_wrist_alignment_bundle(public["bundle_id"])
    assert resolved["parameters"]["target_mask"] == "/session/wrist-mask.png"
    assert resolved["parameters"]["current_eef_pose"]["xyz"] == [0.1, 0.2, 0.3]
    assert resolved["parameters"]["source_robot_motion_epoch"] == 0

    memory._advance_runtime_epochs(
        tool="move_to",
        object_scene_changed=False,
        source="test_robot_motion",
    )
    stale = memory.wrist_alignment_bundle()
    assert stale["status"] == "stale_robot_motion"
    assert stale["bundle_id"] is None
    with pytest.raises(ValueError, match="stale robot-motion epoch"):
        memory.resolve_wrist_alignment_bundle(public["bundle_id"])


def test_exact_instance_mismatch_cannot_be_overridden_by_sam_selection() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the green bottle")
    memory.save_fact(
        "pending_sam3_selection",
        {
            "result_id": "sam-wrong-bottle",
            "evidence_role": "target_object",
            "target_prompt": "bottle",
            "candidates": [{"id": "detection_000", "mask_ref": "orange.png"}],
            "identity_conflict": {
                "decision": "mismatch",
                "confidence": 0.97,
                "reason": "reference is green; candidate is orange",
            },
        },
        source="asset_reference",
    )

    with pytest.raises(ValueError, match="target_identity_conflict"):
        memory.resolve_sam3_selection(
            result_id="sam-wrong-bottle",
            detection_id="detection_000",
            selection_source="main_agent_vlm",
            reason="generic bottle semantics",
        )

    assert memory.pending_sam3_selection() is not None
    assert memory.selected_sam3_detection() is None


def test_new_target_detection_requires_explicit_identity_relation() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the salad dressing")
    memory.save_fact(
        "pending_sam3_selection",
        {
            "result_id": "sam-initial",
            "evidence_role": "target_object",
            "target_prompt": "salad dressing bottle",
            "source_packet_id": "packet-1",
            "candidates": [{"id": "detection_000", "mask_ref": "green.png"}],
        },
        source="test",
    )
    initial = memory.resolve_sam3_selection(
        result_id="sam-initial",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
        reason="green bottle appears to be the target",
    )
    anchor_id = initial["identity_anchor_id"]

    memory.save_fact(
        "pending_sam3_selection",
        {
            "result_id": "sam-new-view",
            "evidence_role": "target_object",
            "target_prompt": "salad dressing bottle",
            "source_packet_id": "packet-2",
            "candidates": [{"id": "detection_001", "mask_ref": "orange.png"}],
        },
        source="test",
    )
    with pytest.raises(ValueError, match="target_identity_confirmation_required"):
        memory.resolve_sam3_selection(
            result_id="sam-new-view",
            detection_id="detection_001",
            selection_source="main_agent_vlm",
            reason="new camera view",
        )
    assert memory.pending_sam3_selection() is not None

    continued = memory.resolve_sam3_selection(
        result_id="sam-new-view",
        detection_id="detection_001",
        selection_source="main_agent_vlm",
        reason="same bottle confirmed by color, shape, and relative location",
        identity_anchor_id=anchor_id,
        identity_relation="same_instance",
    )
    assert continued["identity_anchor_id"] == anchor_id
    assert continued["identity_continuity"] == "agent_confirmed_same_instance"


def test_agent_can_explicitly_replace_a_misidentified_target_anchor() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the salad dressing")
    memory.save_fact(
        "pending_sam3_selection",
        {
            "result_id": "sam-wrong",
            "evidence_role": "target_object",
            "target_prompt": "salad dressing bottle",
            "source_packet_id": "packet-1",
            "candidates": [{"id": "detection_000", "mask_ref": "green.png"}],
        },
        source="test",
    )
    wrong = memory.resolve_sam3_selection(
        result_id="sam-wrong",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
        reason="initial distant-view guess",
    )
    old_anchor_id = wrong["identity_anchor_id"]
    memory.save_fact(
        "pending_sam3_selection",
        {
            "result_id": "sam-corrected",
            "evidence_role": "target_object",
            "target_prompt": "salad dressing bottle",
            "source_packet_id": "packet-2",
            "candidates": [{"id": "detection_001", "mask_ref": "orange.png"}],
        },
        source="test",
    )

    corrected = memory.resolve_sam3_selection(
        result_id="sam-corrected",
        detection_id="detection_001",
        selection_source="main_agent_vlm",
        reason="fresh close view shows orange bottle is salad dressing; green is another item",
        identity_anchor_id=old_anchor_id,
        identity_relation="replace_misidentified_anchor",
    )

    assert corrected["identity_anchor_id"] != old_anchor_id
    assert corrected["replaced_identity_anchor_id"] == old_anchor_id
    assert corrected["identity_continuity"] == "anchor_replaced_after_misidentification"

def test_fresh_observation_and_motion_reconciliation_remain_host_invariants() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    tools = build_default_tool_registry()
    skills = build_default_skill_registry()
    pipeline = ActionPipeline()
    memory.save_fact(
        "motion_reconciliation",
        {"status": "required", "scene_epoch": 0},
        source="transport_unknown",
    )

    blocked = pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="move_to",
            parameters={"target_pose": {"frame": "world", "xyz": [0.1, 0.2, 0.3]}},
        ),
        observation=_observation(),
        tools=tools,
        skills=skills,
        memory=memory,
    )
    observe = pipeline.compile(
        PlannerDecision(action_type="tool_call", action="observe", parameters={}),
        observation=_observation(),
        tools=tools,
        skills=skills,
        memory=memory,
    )

    assert blocked.status == PipelineStatus.BLOCKED
    assert "transport-unknown" in blocked.tool_calls[0].reason
    assert observe.status != PipelineStatus.BLOCKED


def test_agent_context_warns_on_packet_churn_without_forcing_next_tool() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    for packet_id in ("packet-1", "packet-2"):
        memory.add_action(
            EnvAction(
                action_type="tool_call",
                command={
                    "status": "executed",
                    "request": {
                        "kind": "tool_call",
                        "name": "propose_wrist_viewpoints",
                        "parameters": {
                            "compiled_grasp_id": "compiled-1",
                            "camera_frame_id": "wrist",
                            "source_packet_id": packet_id,
                        },
                    },
                    "tool_calls": [
                        {
                            "name": "propose_wrist_viewpoints",
                            "status": "executed",
                            "result": {
                                "success": True,
                                "content": "proposal ready",
                                "details": {
                                    "schema_version": "openeta.tool_result.v1",
                                    "effect": "read_only",
                                    "operational_success": True,
                                    "semantic_outcome": "completed",
                                    "outputs": {
                                        "proposal_id": "wrist_viewpoint:stable",
                                        "candidates": [{"candidate_id": "wrist_view_00"}],
                                    },
                                },
                            },
                        }
                    ],
                },
            )
        )

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    warning = context["agent_context"]["decision_state"]["no_progress_tool_loop"]
    assert warning["tool"] == "propose_wrist_viewpoints"
    assert warning["equivalent_call_count"] == 2
    assert warning["packet_ids_changed"] is True
    assert warning["host_policy"].startswith("reflection_warning_only")
    latest_outputs = context["agent_context"]["decision_state"]["last_action_effect"][
        "outputs"
    ]
    assert latest_outputs["proposal_id"] == "wrist_viewpoint:stable"
    assert latest_outputs["candidates"][0]["candidate_id"] == "wrist_view_00"


def test_agent_context_surfaces_feasible_ik_preview_until_exact_reference_is_dispatched() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    memory.add_observation(_observation())
    target_pose = {
        "frame": "world",
        "xyz": [0.10, -0.20, 0.30],
        "viewpoint_proposal_id": "wrist-view:1",
        "viewpoint_id": "candidate-0",
    }
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "status": "executed",
                "request": {
                    "kind": "tool_call",
                    "name": "ik_preview_check",
                    "parameters": {
                        "target_pose": target_pose,
                        "preserve_current_orientation": True,
                    },
                },
                "tool_calls": [
                    {
                        "name": "ik_preview_check",
                        "status": "executed",
                        "parameters": {
                            "target_pose": target_pose,
                            "preserve_current_orientation": True,
                        },
                        "result": {
                            "success": True,
                            "details": {
                                "operational_success": True,
                                "semantic_outcome": "ik_feasible",
                                "outputs": {
                                    "ik_preview_receipt": {
                                        "schema_version": "openeta.ik_preview_receipt.v1",
                                        "receipt_id": "ik-wrist-view-1",
                                        "classification": "feasible",
                                        "orientation_policy": "preserve_current",
                                    }
                                },
                            },
                        },
                    }
                ],
            },
        )
    )

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )
    decision_state = context["agent_context"]["decision_state"]
    pending = decision_state["pending_execution_receipts"]
    assert pending["host_policy"].startswith("capability_index_only")
    assert pending["receipts"] == [
        {
            "receipt_id": "ik-wrist-view-1",
            "status": "previewed_not_executed",
            "classification": "feasible",
            "orientation_policy": "preserve_current",
            "target_signature": memory.ik_preview_receipts()["latest"][
                "target_signature"
            ],
            "execution_reference": {
                "tool": "move_to",
                "parameters": {"ik_receipt_id": "ik-wrist-view-1"},
            },
            "request_reference": {
                "viewpoint_proposal_id": "wrist-view:1",
                "viewpoint_id": "candidate-0",
            },
        }
    ]
    assert decision_state["unresolved_obligations"]["preview_execution_gap"][
        "pending_receipt_ids"
    ] == ["ik-wrist-view-1"]

    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "status": "executed",
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {"ik_receipt_id": "ik-wrist-view-1"},
                },
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "executed",
                        "parameters": {"ik_receipt_id": "ik-wrist-view-1"},
                        "result": {
                            "success": False,
                            "details": {
                                "operational_success": False,
                                "semantic_outcome": "target_not_reached",
                                "outputs": {},
                            },
                        },
                    }
                ],
            },
        )
    )
    after_dispatch = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )["agent_context"]["decision_state"]
    assert after_dispatch["pending_execution_receipts"] is None
    assert "preview_execution_gap" not in after_dispatch["unresolved_obligations"]


def test_agent_context_warns_on_multi_tool_negative_cycle_at_same_visual_epochs() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    memory.add_observation(_observation())

    def record(name: str, semantic_outcome: str) -> None:
        memory.add_action(
            EnvAction(
                action_type="tool_call",
                command={
                    "status": "executed",
                    "request": {
                        "kind": "tool_call",
                        "name": name,
                        "parameters": {"source_packet_id": f"packet-{name}"},
                    },
                    "tool_calls": [
                        {
                            "name": name,
                            "status": "executed",
                            "result": {
                                "success": True,
                                "details": {
                                    "operational_success": True,
                                    "semantic_outcome": semantic_outcome,
                                    "outputs": {},
                                },
                            },
                        }
                    ],
                },
            )
        )

    record("compute_wrist_alignment", "requires_better_view")
    record("sam3", "no_detection")
    record("select_sam3_detection", "completed")
    record("compute_wrist_alignment", "requires_better_view")

    warning = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )["agent_context"]["decision_state"]["no_progress_tool_loop"]
    assert warning["trigger_type"] == "semantic_state_cycle_without_world_change"
    assert warning["tool"] == "compute_wrist_alignment"
    assert warning["semantic_outcome"] == "requires_better_view"
    assert warning["equivalent_outcome_count"] == 2
    assert warning["intervening_tools"] == ["sam3", "select_sam3_detection"]
    assert warning["state_anchor"]["object_scene_epoch"] == 0
    assert warning["state_anchor"]["robot_motion_epoch"] == 0
    assert warning["state_anchor"]["visual_signature"]
    assert warning["host_policy"].startswith("reflection_warning_only")


def test_semantic_cycle_warning_resets_after_visual_observation_changes() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    first = _observation()
    memory.add_observation(first)

    def alignment() -> None:
        memory.add_action(
            EnvAction(
                action_type="tool_call",
                command={
                    "status": "executed",
                    "request": {
                        "kind": "tool_call",
                        "name": "compute_wrist_alignment",
                        "parameters": {},
                    },
                    "tool_calls": [
                        {
                            "name": "compute_wrist_alignment",
                            "status": "executed",
                            "result": {
                                "success": True,
                                "details": {
                                    "semantic_outcome": "requires_better_view",
                                    "outputs": {},
                                },
                            },
                        }
                    ],
                },
            )
        )

    alignment()
    changed = _observation()
    changed.cameras[0].rgb = [[[255, 255, 255]]]
    changed.metadata["step_idx"] = 4
    memory.add_observation(changed)
    alignment()

    warning = build_tool_context(
        observation=changed,
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )["agent_context"]["decision_state"]["no_progress_tool_loop"]
    assert warning is None


def test_agent_context_warns_on_interleaved_sam3_selection_cycle_and_exposes_bundle() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")

    def record(name: str, parameters: dict[str, object]) -> None:
        memory.add_action(
            EnvAction(
                action_type="tool_call",
                command={
                    "status": "executed",
                    "request": {
                        "kind": "tool_call",
                        "name": name,
                        "parameters": parameters,
                    },
                    "tool_calls": [
                        {
                            "name": name,
                            "status": "executed",
                            "parameters": parameters,
                            "result": {
                                "success": True,
                                "details": {
                                    "operational_success": True,
                                    "semantic_outcome": "completed",
                                    "outputs": {},
                                },
                            },
                        }
                    ],
                },
            )
        )

    for index, packet_id in enumerate(("packet-1", "packet-2"), start=1):
        record(
            "sam3",
            {
                "source_packet_id": packet_id,
                "camera_frame_id": "agentview",
                "mode": "text",
                "prompt": "cube",
                "evidence_role": "target_object",
            },
        )
        record(
            "select_sam3_detection",
            {
                "sam3_result_id": f"result-{index}",
                "detection_id": "detection_000",
                "evidence_role": "target_object",
                "identity_anchor_id": "target:cube",
                "identity_relation": "same_instance",
                "reason": f"same cube on packet {packet_id}",
            },
        )
    memory.save_fact(
        "grasp_input_bundles",
        {
            "schema_version": "openeta.grasp_input_bundle_store.v1",
            "active_bundle_id": "grasp:ready",
            "public": {
                "status": "ready",
                "bundle_id": "grasp:ready",
                "target_evidence_id": "sam3:result-2:detection_000",
                "object_scene_epoch": 0,
            },
            "bundles": {},
        },
        source="host_provenance_bundle_resolver",
    )

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    warning = context["agent_context"]["decision_state"]["no_progress_tool_loop"]
    assert warning["trigger_type"] == "interleaved_equivalent_read_only_cycle"
    assert warning["tool"] == "select_sam3_detection"
    assert warning["equivalent_call_count"] == 2
    assert warning["intervening_tools"] == ["sam3"]
    assert warning["reusable_bundles"]["grasp_pose_estimate"]["bundle_id"] == (
        "grasp:ready"
    )
    assert warning["host_policy"].startswith("reflection_warning_only")


def test_context_warns_when_motion_retries_converge_to_same_wrong_pose() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    target = {
        "frame": "world",
        "xyz": [0.04, -0.10, 0.16],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    for actual in (
        [-0.0071, -0.1613, 0.1285],
        [-0.0076, -0.1610, 0.1279],
    ):
        memory.add_action(
            EnvAction(
                action_type="tool_call",
                command={
                    "status": "executed",
                    "request": {
                        "kind": "tool_call",
                        "name": "move_to",
                        "parameters": {"target_pose": target},
                    },
                    "tool_calls": [
                        {
                            "name": "move_to",
                            "status": "executed",
                            "parameters": {"target_pose": target},
                            "result": {
                                "success": True,
                                "details": {
                                    "operational_success": False,
                                    "semantic_outcome": "target_not_reached",
                                    "outputs": {
                                        "motion_summary": {
                                            "reached_target": False,
                                            "steps_executed": 150,
                                            "position_error_m": 0.083,
                                            "stop_reason": "iteration_limit",
                                            "end": {"xyz": actual},
                                        }
                                    },
                                },
                            },
                        }
                    ],
                },
            )
        )

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    warning = context["agent_context"]["decision_state"]["no_progress_tool_loop"]
    assert warning["trigger_type"] == "repeated_failed_motion_attractor"
    assert warning["tool"] == "move_to"
    assert warning["attempt_count"] == 2
    assert "same wrong EEF pose" in warning["interpretation"]
    assert warning["host_policy"].startswith("reflection_warning_only")
