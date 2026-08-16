from __future__ import annotations

import json

import pytest

from adapter.protocol import CameraFrame, EnvAction, EnvObservation, RobotState
from agent.backends.planner import PlannerBackend, PlannerBackendRequest, PlannerBackendResult
from agent.runtime.actions import PipelineStatus
from agent.runtime.memory import AgentMemory
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import PlannerDecision, ToolCallingPlanner, build_tool_context
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
    assert memory.object_scene_epoch() == 1

    memory.add_action(successful_action("gripper_control", {"position": 1}))
    assert memory.robot_motion_epoch() == 3
    assert memory.object_scene_epoch() == 2


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
