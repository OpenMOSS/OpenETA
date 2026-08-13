from __future__ import annotations

import json

from adapter.protocol import CameraFrame, EnvAction, EnvObservation, RobotState
from agent.backends.planner import PlannerBackend, PlannerBackendRequest, PlannerBackendResult
from agent.runtime.memory import AgentMemory
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.actions import PipelineStatus
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import (
    PlannerContextConfig,
    PlannerDecision,
    ToolCallingPlanner,
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


def test_agent_context_prioritizes_current_evidence_without_task_phase() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    memory.save_fact("agent_plan", {"next_subgoal": "locate cube"}, source="save_memory")
    memory.save_fact(
        "grasp_execution",
        {"status": "required", "stage": "descend", "required_action": {"name": "move_to"}},
        source="runtime_grasp_transition",
    )

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
    assert "grasp_execution" not in agent_context
    assert "required_action" not in str(agent_context)
    legacy_payload = {key: value for key, value in context.items() if key != "agent_context"}
    assert len(json.dumps(agent_context)) < len(json.dumps(legacy_payload))


def test_main_planner_backend_receives_agent_context_not_runtime_policy_context() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    backend = RecordingBackend()
    planner = ToolCallingPlanner(backend=backend)

    planner.plan(
        _observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    assert len(backend.requests) == 1
    sent = backend.requests[0].tool_context
    assert sent["schema_version"] == "openeta.agent_context.v2"
    assert sent["vision_evidence"][0]["role"] == "current_scene"
    assert "grasp_execution" not in sent
    assert "grasp_estimation_fallback_obligation" not in sent


def test_agent_working_state_persists_separately_from_runtime_facts(tmp_path) -> None:
    root = tmp_path / ".openeta_memory"
    memory = AgentMemory(store=JsonMemoryStore(root))
    memory.start_session(task="pick the cube", session_id="session-1")
    memory.save_fact("agent_plan", {"next_subgoal": "locate cube"}, source="save_memory")
    memory.save_fact(
        "grasp_execution",
        {"status": "required", "stage": "descend"},
        source="runtime_grasp_transition",
    )

    resumed = AgentMemory(store=JsonMemoryStore(root))
    resumed.resume_session("session-1")

    assert set(resumed.agent_working_state) == {"agent_plan"}
    assert resumed.agent_working_state["agent_plan"]["ownership"] == "agent"
    assert "grasp_execution" in resumed.facts
    assert "grasp_execution" not in resumed.agent_working_state


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
    assert current["selected_target"]["provenance"] == {
        "source": "sam3_selection",
        "timestamp_s": memory.facts["selected_sam3_detection"]["timestamp_s"],
        "evidence_kind": "perception_result",
    }
    assert current["gripper_command"]["freshness"] == "commanded_not_observed"

    memory.save_fact("scene_epoch", {"epoch": 1}, source="runtime")
    stale = memory.world_evidence_context()

    assert stale["selected_target"]["freshness"] == "stale_scene_epoch"


def test_agent_policy_mode_does_not_dispatch_host_task_stage() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    memory.save_fact(
        "grasp_execution",
        {
            "status": "required",
            "stage": "close",
            "required_action": {"name": "gripper_control", "parameters": {"position": 0}},
        },
        source="runtime_grasp_transition",
    )
    backend = RecordingBackend()
    planner = ToolCallingPlanner(
        backend=backend,
        context_config=PlannerContextConfig(host_task_policy_enabled=False),
    )

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    assert len(backend.requests) == 1
    assert decision.action_type == "response"
    assert decision.action == "talk"
    assert decision.metadata.get("execution_model") != "host_obligation_dispatch"
    assert "You own task decomposition" in backend.requests[0].system_prompt
    assert "required next action" in backend.requests[0].system_prompt
    assert "follow host-generated grasp_execution stages" not in backend.requests[0].system_prompt


def test_agent_policy_mode_retains_compiled_grasp_as_read_only_evidence() -> None:
    memory = AgentMemory(task_state_tracking_enabled=False)
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
                        "result": {
                            "success": True,
                            "details": {"outputs": compiled},
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
    )["agent_context"]

    assert memory.grasp_execution() is None
    retained = next(
        artifact
        for artifact in context["artifacts"].values()
        if artifact.get("type") == "compiled_grasp"
    )
    assert retained["compiled_grasp_id"] == "compiled-1"
    assert retained["hover_pose"]["xyz"] == [0.1, 0.2, 0.3]
    assert retained["contact_pose"]["xyz"] == [0.1, 0.2, 0.15]


def test_agent_policy_mode_keeps_fresh_observation_host_invariant() -> None:
    memory = AgentMemory(task_state_tracking_enabled=False)
    memory.start_session(task="pick the cube")
    backend = RecordingBackend()
    planner = ToolCallingPlanner(
        backend=backend,
        context_config=PlannerContextConfig(host_task_policy_enabled=False),
    )
    observation = _observation()
    observation.metadata["fresh_observation_required"] = True
    tools = build_default_tool_registry()
    tools.bind_handler("observe", lambda _context: {"success": True})

    decision = planner.plan(
        observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )

    assert backend.requests == []
    assert decision.action_type == "tool_call"
    assert decision.action == "observe"
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"
    assert decision.metadata["host_obligation"]["schema_version"] == (
        "openeta.fresh_observation_obligation.v1"
    )


def test_agent_policy_pipeline_does_not_enforce_legacy_grasp_stage() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    memory.save_fact(
        "grasp_execution",
        {
            "status": "required",
            "stage": "close",
            "required_action": {"name": "gripper_control", "parameters": {"position": 0}},
        },
        source="runtime_grasp_transition",
    )
    decision = PlannerDecision(
        action_type="tool_call",
        action="move_to",
        parameters={"target_pose": {"frame": "world", "xyz": [0.1, 0.2, 0.3]}},
    )
    tools = build_default_tool_registry()
    skills = build_default_skill_registry()

    legacy = ActionPipeline().compile(
        decision,
        observation=_observation(),
        tools=tools,
        skills=skills,
        memory=memory,
    )
    agent_owned = ActionPipeline(task_execution_gate_enabled=False).compile(
        decision,
        observation=_observation(),
        tools=tools,
        skills=skills,
        memory=memory,
    )

    assert legacy.status == PipelineStatus.BLOCKED
    assert agent_owned.status != PipelineStatus.BLOCKED


def test_agent_policy_memory_does_not_advance_legacy_task_stage() -> None:
    memory = AgentMemory(task_state_tracking_enabled=False)
    memory.start_session(task="pick the cube")
    execution = {
        "status": "required",
        "stage": "close",
        "required_action": {"name": "gripper_control", "parameters": {"position": 0}},
    }
    memory.save_fact(
        "grasp_execution",
        execution,
        source="runtime_grasp_transition",
    )
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "gripper_control",
                    "parameters": {"position": 0},
                },
                "tool_calls": [
                    {
                        "name": "gripper_control",
                        "status": "executed",
                        "parameters": {"position": 0},
                        "result": {"success": True, "details": {}},
                    }
                ],
            },
        )
    )

    assert memory.grasp_execution() == execution
    assert not any(event.event_type == "grasp_execution_transition" for event in memory.events)


def test_agent_policy_context_hides_resumed_legacy_task_policy_facts() -> None:
    memory = AgentMemory(task_state_tracking_enabled=False)
    memory.start_session(task="pick the cube")
    memory.save_fact(
        "grasp_candidate_policy",
        {
            "status": "active",
            "active_candidate": {"id": "legacy-grasp"},
            "scene_epoch": 0,
        },
        source="legacy_runtime",
    )
    memory.save_fact(
        "grasp_execution",
        {
            "status": "required",
            "stage": "close",
            "required_action": {"name": "gripper_control", "parameters": {"position": 0}},
        },
        source="legacy_runtime",
    )

    context = memory.planning_context()

    assert context["grasp_candidate_policy"] is None
    assert context["grasp_execution"] is None
    assert "grasp_candidates" not in context["world_evidence"]
    assert memory.grasp_candidate_gate_error(tool_name="move_to", parameters={}) is None


def test_agent_policy_world_evidence_keeps_tool_candidates_without_host_policy() -> None:
    memory = AgentMemory(task_state_tracking_enabled=False)
    memory.start_session(task="pick the cube")
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {
                        "name": "grasp_pose_estimate",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "grasp_candidates": [
                                        {
                                            "id": "candidate-1",
                                            "score": 0.8,
                                            "translation": [0.1, 0.2, 0.3],
                                        }
                                    ],
                                    "source": {
                                        "rgb": "/session/rgb.png",
                                        "depth": "/session/depth.npy",
                                        "camera_frame_id": "agentview",
                                    },
                                }
                            },
                        },
                    }
                ]
            },
        )
    )

    evidence = memory.world_evidence_context()["grasp_candidates"]

    assert memory.grasp_candidate_policy() is None
    assert evidence["value"]["grasp_candidates"][0]["id"] == "candidate-1"
    assert evidence["provenance"]["source"] == "grasp_pose_estimate"
    assert evidence["freshness"] == "current_scene_epoch"
    assert "host-generated" not in evidence["value"]["next_tool_hint"]


def test_agent_policy_pipeline_keeps_motion_reconciliation_invariant() -> None:
    memory = AgentMemory(task_state_tracking_enabled=False)
    memory.start_session(task="pick the cube")
    memory.save_fact(
        "motion_reconciliation",
        {"status": "required", "scene_epoch": 0},
        source="transport_unknown",
    )
    tools = build_default_tool_registry()
    skills = build_default_skill_registry()
    pipeline = ActionPipeline(task_execution_gate_enabled=False)

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
