"""Position diagnostics never stand in for remote operation completion."""

from copy import deepcopy

import pytest

from adapter.protocol import EnvAction, EnvObservation, RobotState
from agent.runtime.actions import PipelineStatus
from agent.runtime.episode import is_agent_terminal_response
from agent.runtime.memory import AgentMemory
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import PlannerDecision, _invariant_obligation_decision
from agent.runtime.skills import build_default_skill_registry
from agent.tools.registry import ToolResult, build_default_tool_registry


TARGET = {"frame": "world", "xyz": [0.1, 0.2, 0.3]}


def _unknown_action(tool="move_to"):
    resolved = (
        {"position": 0} if tool == "gripper_control"
        else {"target_pose": deepcopy(TARGET), "tolerance": 0.01}
    )
    public = {"ik_receipt_id": "ik-before-timeout"} if tool == "move_to" else resolved
    return EnvAction(action_type="tool_call", command={
        "request": {"kind": "tool_call", "name": tool, "parameters": public},
        "status": "failed",
        "tool_calls": [{
            "name": tool,
            "status": "failed",
            "parameters": resolved,
            "result": {"success": False, "details": {
                "outputs": {"motion_outcome": "unknown", "reconciliation_required": True},
            }},
        }],
    })


def _observation(xyz=None):
    return EnvObservation(task="move safely", cameras=[], robot=RobotState(
        end_effector_pose={"xyz": list(TARGET["xyz"] if xyz is None else xyz)},
        gripper_state={"openness": 0.4},
    ))


def _memory(tool="move_to"):
    memory = AgentMemory()
    memory.start_session(task="move safely")
    memory.add_action(_unknown_action(tool))
    return memory


@pytest.mark.parametrize("tool", ["move_to", "follow_eef_trajectory", "gripper_control"])
def test_matching_position_never_unlocks_unknown_remote_operation(tool):
    memory = _memory(tool)
    initial_epoch = memory.robot_motion_epoch()
    for _ in range(4):
        memory.add_observation(_observation())
        reconciliation = memory.motion_reconciliation()
        assert reconciliation["position_reconciliation_status"] == "completed"
        assert reconciliation["status"] == "unresolved"
        assert reconciliation["remote_completion_verified"] is False
        assert memory.motion_reconciliation_gate_error(tool_name="move_to")
        assert memory.motion_reconciliation_gate_error(tool_name="observe") is None
        assert memory.robot_motion_epoch() == initial_epoch + 1


def test_stable_target_miss_is_not_remote_failure_receipt():
    memory = _memory()
    for _ in range(3):
        memory.add_observation(_observation([0.4, 0.2, 0.3]))
    reconciliation = memory.motion_reconciliation()
    assert reconciliation["position_reconciliation_status"] == "failed"
    assert reconciliation["status"] == "unresolved"
    assert reconciliation["remote_completion_verified"] is False
    assert memory.motion_reconciliation_gate_error(tool_name="gripper_control")


@pytest.mark.parametrize("status", ["required", "unresolved", "completed", "failed", "unknown"])
def test_legacy_snapshot_verdict_does_not_authorize_resumption(tmp_path, status):
    memory = AgentMemory(store=JsonMemoryStore(tmp_path))
    memory.start_session(task="move safely", session_id="unknown-operation")
    memory.save_fact("motion_reconciliation", {
        "tool": "move_to", "status": status,
        "intended_parameters": {"target_pose": deepcopy(TARGET)},
    }, source="same_handle_observation")
    resumed = AgentMemory(store=JsonMemoryStore(tmp_path))
    resumed.resume_session("unknown-operation")
    assert resumed.motion_reconciliation_gate_error(tool_name="move_to")
    resumed.add_observation(_observation())
    assert resumed.motion_reconciliation()["status"] == "unresolved"
    assert resumed.motion_reconciliation()["position_reconciliation_status"] == "completed"
    assert resumed.motion_reconciliation_gate_error(tool_name="move_to")


@pytest.mark.parametrize("status", ["required", "unresolved", "completed", "failed"])
def test_planner_stops_instead_of_forcing_endless_observations(status):
    tools = build_default_tool_registry()
    decision = _invariant_obligation_decision({
        "motion_reconciliation": {"tool": "move_to", "status": status},
        "fresh_observation_obligation": {"required": True},
    }, tools=tools)
    assert decision.action_type == "response"
    assert decision.action == "talk"
    plan = ActionPipeline().compile(
        decision, observation=_observation(), tools=tools,
        skills=build_default_skill_registry(), memory=_memory(),
    )
    assert plan.status is not PipelineStatus.BLOCKED
    assert is_agent_terminal_response(plan.to_env_action())
    assert not plan.tool_calls
    assert "Task success has not been established" in decision.parameters["message"]


def test_matched_observation_cannot_dispatch_next_command_but_allows_inspection():
    memory = _memory()
    observation = _observation()
    memory.add_observation(observation)
    tools = build_default_tool_registry()
    dispatched = []

    def record(context):
        dispatched.append(context.spec.name)
        return ToolResult(True)

    tools.bind_handler("gripper_control", record)
    tools.bind_handler("observe", record)
    pipeline = ActionPipeline()
    blocked = pipeline.compile(
        PlannerDecision(action_type="tool_call", action="gripper_control", parameters={"position": 1}),
        observation=observation, tools=tools, skills=build_default_skill_registry(), memory=memory,
    )
    assert blocked.status is PipelineStatus.BLOCKED
    assert blocked.metadata["motion_reconciliation_gate"]["blocked"] is True
    assert not dispatched
    observed = pipeline.compile(
        PlannerDecision(action_type="tool_call", action="observe", parameters={}),
        observation=observation, tools=tools, skills=build_default_skill_registry(), memory=memory,
    )
    assert observed.status is PipelineStatus.EXECUTED
    assert dispatched == ["observe"]
    assert memory.motion_reconciliation_gate_error(tool_name="gripper_control")


def test_normal_fresh_observation_obligation_is_unchanged_without_unknown_motion():
    tools = build_default_tool_registry()
    tools.bind_handler("observe", lambda _context: ToolResult(True))
    decision = _invariant_obligation_decision({
        "fresh_observation_obligation": {"required": True},
    }, tools=tools)
    assert decision.action_type == "tool_call"
    assert decision.action == "observe"


def test_unknown_trajectory_without_single_endpoint_stays_unresolved():
    action = _unknown_action("follow_eef_trajectory")
    action.command["tool_calls"][0]["parameters"] = {"trajectory": [deepcopy(TARGET)]}
    memory = AgentMemory()
    memory.start_session(task="move safely")
    memory.add_action(action)
    memory.add_observation(_observation())
    assert memory.motion_reconciliation()["position_reconciliation_status"] == "unresolved"
    assert memory.motion_reconciliation_gate_error(tool_name="move_to")
