"""Removing strategy vetoes must not transfer or manufacture evidence."""

import pytest

from agent.runtime.memory import AgentMemory, GRASP_PROVENANCE_KEY


def _alternative_memory():
    memory = AgentMemory()
    memory.start_session(task="pick the selected object")
    memory.save_fact("selected_sam3_detection", {
        "result_id": "segmentation", "id": "one",
        "identity_anchor_id": "object-one",
    }, source="select_sam3_detection")
    for branch in ("A", "B"):
        pose = {"frame": "world", "xyz": [0.1, 0.2, 0.3],
                "compiled_grasp_id": branch, "waypoint_role": "grasp_contact"}
        memory.save_artifact(branch, {
            "type": "compiled_grasp", "compiled_grasp_id": branch,
            "scene_epoch": 0, "contact_pose": pose,
            "target_anchor_world_xyz": [0.1, 0.2, 0.3],
        }, source="compile_grasp_seed")
        provenance = {"compiled_grasp_id": branch, "object_scene_epoch": 0,
                      "target_evidence_id": "sam3:segmentation:one",
                      "target_identity_anchor_id": "object-one"}
        memory.save_fact(GRASP_PROVENANCE_KEY, provenance, source="compile_grasp_seed")
        memory.record("grasp_provenance_bound", {"provenance": provenance})
    return memory


def test_compiling_alternative_does_not_revoke_same_instance_branch():
    memory = _alternative_memory()
    pose = memory._compiled_grasp_artifact("A")["contact_pose"]
    assert memory.compiled_grasp_target_gate_error(
        tool_name="move_to", parameters={"target_pose": pose}) is None
    authorization = memory.resolve_compiled_contact_authorization(pose)
    assert authorization["compiled_grasp_id"] == "A"
    assert authorization["target_evidence_id"] == "sam3:segmentation:one"
    assert memory.attachment_evidence() is None
    assert memory.ik_execution_gate_error(
        tool_name="move_to", parameters={"target_pose": pose}) is not None


@pytest.mark.parametrize("change", ["different_instance", "invalidated", "stale"])
def test_old_branch_cannot_borrow_contact_authorization(change):
    memory = _alternative_memory()
    pose = memory._compiled_grasp_artifact("A")["contact_pose"]
    if change == "different_instance":
        memory.save_fact("selected_sam3_detection", {
            "result_id": "another", "id": "two", "identity_anchor_id": "object-two",
        }, source="select_sam3_detection")
    elif change == "invalidated":
        memory.record("compiled_grasp_contact_geometry_invalidated", {"compiled_grasp_id": "A"})
    else:
        memory._compiled_grasp_artifact("A")["scene_epoch"] = 9
    with pytest.raises(ValueError):
        memory.resolve_compiled_contact_authorization(pose)


@pytest.mark.parametrize("evidence", ["missing", "stale", "invalidated"])
def test_real_proxy_dispatch_does_not_turn_optional_binding_into_close_gate(evidence):
    from adapter.protocol import EnvObservation, RobotState
    from agent.backends.planner import StaticPlannerBackend
    from agent.runtime.planner import ToolCallingPlanner
    from agent.runtime.runtime import OpenEtaAgentRuntime
    from agent.tools.registry import build_default_tool_registry
    from agent.tools.sim_mcp import SimulatorMcpToolProxyConfig, bind_simulator_mcp_tool_handlers

    calls = []

    class Transport:
        def call_tool(self, name, arguments, *, timeout_s=None):
            calls.append((name, arguments))
            return {"success": True, "steps_executed": 40,
                    "gripper_state": {"open": False, "openness": 0.4}}

    runtime = OpenEtaAgentRuntime(
        tools=bind_simulator_mcp_tool_handlers(
            build_default_tool_registry(), transport=Transport(),
            config=SimulatorMcpToolProxyConfig(session_id="autonomy", handle="fixture"),
            tool_names=("gripper_control",)),
        rollout_enabled=False,
        planner=ToolCallingPlanner(StaticPlannerBackend({
            "kind": "tool_call", "name": "gripper_control", "parameters": {"position": 0},
        })),
    )
    runtime.start_session(task="choose a finger-only recovery")
    provenance = {"compiled_grasp_id": "unusable", "object_scene_epoch": 0}
    if evidence == "invalidated":
        provenance["contact_geometry_invalidated_at_s"] = 1
    runtime.memory.save_fact(GRASP_PROVENANCE_KEY, provenance, source="compile_grasp_seed")
    if evidence == "stale":
        runtime.memory.save_artifact("unusable", {
            "type": "compiled_grasp", "compiled_grasp_id": "unusable", "scene_epoch": 9,
            "contact_pose": {"frame": "world", "xyz": [0.1, 0.2, 0.3],
                             "compiled_grasp_id": "unusable", "waypoint_role": "grasp_contact"},
        }, source="compile_grasp_seed")
    runtime.act(EnvObservation(task="choose a finger-only recovery", cameras=[], robot=RobotState()))
    assert len(calls) == 1
    assert calls[0][0] == "gripper_close"
    assert "contact_authorization" not in calls[0][1]
    assert runtime.memory.attachment_evidence() is None
