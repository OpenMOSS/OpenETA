from copy import deepcopy

import pytest

from adapter.protocol import EnvAction, EnvObservation, RobotState
from agent.runtime.memory import AgentMemory
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import PlannerDecision, PlannerContextConfig, build_tool_context, _validate_tool_parameters
from agent.runtime.skills import build_default_skill_registry
from agent.runtime.interface_profiles import profile_parameter_errors, project_profile_evidence
from agent.tools.bundle_proposals import bind_profile_proposals
from agent.tools.registry import build_default_tool_registry, ToolResult


POSE = {"frame": "world", "xyz": [0.1, 0.2, 0.3], "rotation_matrix": [[1,0,0],[0,1,0],[0,0,1]]}


def memory_at(tmp_path, profile="bundle_stage3"):
    memory = AgentMemory(artifact_root=tmp_path / "artifacts", store=JsonMemoryStore(root=tmp_path / "memory"),
                         agent_interface_profile=profile)
    memory.start_session(task="fixture", session_id="stage-fixture")
    return memory


def action(name, outputs, parameters=None):
    return EnvAction(action_type="tool_call", command={"request": {"kind": "tool_call", "name": name, "parameters": parameters or {}},
        "status": "executed", "tool_calls": [{"name": name, "status": "executed", "parameters": parameters or {},
        "result": {"success": True, "details": {"outputs": outputs}}}]})


def run_tool(memory, name, parameters, tools=None, profile=None):
    tools = tools or build_default_tool_registry()
    profile = profile or memory.agent_interface_profile
    if not tools.can_execute("propose_motion_target"):
        bind_profile_proposals(tools, profile, lambda: memory)
    plan = ActionPipeline(agent_interface_profile=profile).compile(
        PlannerDecision(action_type="tool_call", action=name, parameters=parameters),
        observation=EnvObservation(task="fixture", cameras=[], robot=RobotState()),
        tools=tools, skills=build_default_skill_registry(), memory=memory)
    return plan


@pytest.mark.parametrize("reference", [
    {"target_pose": POSE}, {"compiled_grasp_id": "c", "waypoint_role": "grasp_contact"},
    {"compiled_grasp_id": "c", "path_fraction": 0.5},
    {"viewpoint_proposal_id": "v", "candidate_id": "one"}, {"probe_id": "p", "waypoint_index": 0},
])
def test_stage2_preserves_every_native_ik_source_as_authored_bundle(tmp_path, reference):
    memory = memory_at(tmp_path, "bundle_stage2")
    assert profile_parameter_errors("bundle_stage2", "ik_preview_check", reference)
    assert not profile_parameter_errors("bundle_stage2", "propose_motion_target", reference)
    plan = run_tool(memory, "propose_motion_target", reference)
    assert plan.status.value == "executed"
    assert memory.tool_handoffs() == []  # Handler cannot commit to memory on its own.
    memory.add_action(plan.to_env_action())
    bundle = memory.tool_handoffs()[-1]
    assert memory.resolve_tool_bundle("ik_preview_check", {"bundle_id": bundle["bundle_id"]}) == reference
    restored = AgentMemory(artifact_root=memory.artifact_root, store=memory.store, agent_interface_profile="bundle_stage2")
    restored.resume_session("stage-fixture")
    assert restored.tool_handoffs()[-1]["bundle_id"] == bundle["bundle_id"]
    assert restored.tool_handoffs()[-1]["current_epoch"] is False


@pytest.mark.parametrize("tool,outputs,binding", [
    ("camera_pose_to_world", {"world_pose": {"frame": "world", "translation_xyz": [0.1,0.2,0.3], "rotation_matrix": POSE["rotation_matrix"]}}, "target_pose"),
    ("compute_wrist_alignment", {"adjusted_contact_pose": POSE}, "target_pose"),
    ("prepare_attachment_probe", {"probe_id": "p", "frozen_path": [POSE]}, "probe_id"),
])
def test_stage2_producers_publish_targets_without_new_agent_copying(tmp_path, tool, outputs, binding):
    memory = memory_at(tmp_path, "bundle_stage2")
    memory.add_action(action(tool, outputs))
    bundle = memory.tool_handoffs()[-1]
    assert binding in memory.resolve_tool_bundle("ik_preview_check", {"bundle_id": bundle["bundle_id"]})


def test_stage3_selection_and_rejection_resolve_same_detection_set(tmp_path):
    memory = memory_at(tmp_path)
    memory.add_action(action("sam3", {"result_id": "s1", "detection_count": 2}))
    ref = memory.tool_handoffs()[-1]
    for tool, choices in (("select_sam3_detection", {"detection_id": "d0", "identity_relation": "same_instance", "identity_anchor_id": "anchor"}),
                          ("reject_sam3_detections", {"reason": "wrong object"})):
        public = {"bundle_id": ref["bundle_id"], **choices}
        assert not profile_parameter_errors("bundle_stage3", tool, public)
        assert not _validate_tool_parameters(tool, public)
        assert memory.resolve_tool_bundle(tool, public) == {"sam3_result_id": "s1", **choices}
        assert profile_parameter_errors("bundle_stage3", tool, {"sam3_result_id": "s1", **choices})


@pytest.mark.parametrize("tool,kind,native", [("grasp_pose_estimate", "grasp_input", "grasp:old"), ("anyplace", "placement_input", "anyplace:old")])
def test_input_manifest_does_not_change_native_evidence_authority(tmp_path, tool, kind, native):
    memory = memory_at(tmp_path)
    ref = memory.register_tool_bundle(kind=kind, producer="fixture", reference_parameters={"native_bundle_id": native},
        summary={"consumer_tool": tool, "native_reference": native})
    assert profile_parameter_errors("bundle_stage3", tool, {"bundle_id": native})
    assert memory.resolve_tool_bundle(tool, {"bundle_id": ref["bundle_id"]}) == {"bundle_id": native}
    # The manifest alone is not the native provenance bundle. Real native gate
    # must still reject it before a backend handler can be reached.
    calls = []
    tools = build_default_tool_registry()
    tools.bind_handler(tool, lambda ctx: calls.append(ctx.parameters) or ToolResult(True))
    plan = run_tool(memory, tool, {"bundle_id": ref["bundle_id"]}, tools)
    assert not calls and plan.status.value == "blocked"


def test_trajectory_composition_keeps_order_and_rejects_unregistered_input(tmp_path):
    memory = memory_at(tmp_path)
    refs = [memory.register_tool_bundle(kind="ik_result", producer="fixture", robot_bound=True,
                reference_parameters={"ik_receipt_id": f"ik-{n}"}, summary={"receipt_id": f"ik-{n}"}) for n in range(2)]
    ids = [r["bundle_id"] for r in reversed(refs)]
    plan = run_tool(memory, "compose_ik_trajectory", {"bundle_ids": ids})
    assert plan.status.value == "executed"
    memory.add_action(plan.to_env_action())
    route = memory.tool_handoffs()[-1]
    assert memory.resolve_tool_bundle("follow_eef_trajectory", {"bundle_id": route["bundle_id"]}) == {"ik_receipt_ids": ["ik-1", "ik-0"]}
    calls = []
    tools = build_default_tool_registry()
    tools.bind_handler("follow_eef_trajectory", lambda ctx: calls.append(ctx) or ToolResult(True))
    denied = run_tool(memory, "follow_eef_trajectory", {"bundle_id": route["bundle_id"]}, tools)
    assert denied.status.value == "blocked" and not calls  # No executable receipts were forged.
    for values in ([ids[0], ids[0]], ["bnd-" + "a"*32]):
        denied = run_tool(memory, "compose_ik_trajectory", {"bundle_ids": values})
        assert denied.status.value == "failed"


def test_profile_guidance_never_relabels_historical_native_call_as_executable():
    refs = [{"kind": "grasp_input", "bundle_id": "bnd-new", "current_epoch": True, "summary": {"native_reference": "grasp:old"}}]
    original = {"host_resolved_inputs": {"grasp_pose_estimate": {"status": "ready", "bundle_id": "grasp:old", "call_parameters": {"bundle_id": "grasp:old"}}},
                "recorded_call": {"tool": "grasp_pose_estimate", "parameters": {"bundle_id": "grasp:old"}},
                "allowed_next_calls": [{"tool": "ik_preview_check", "parameters": {"target_pose": POSE}}]}
    snapshot = deepcopy(original)
    result = project_profile_evidence(original, "bundle_stage3", refs)
    assert original == snapshot
    assert result["recorded_call"] == original["recorded_call"]
    assert result["host_resolved_inputs"]["grasp_pose_estimate"]["bundle_id"] == "bnd-new"
    assert result["allowed_next_calls"][0]["tool"] == "propose_motion_target"


@pytest.mark.parametrize("profile", ["bundle_stage2", "bundle_stage3"])
def test_full_agent_projection_includes_authoring_and_single_ik_reference(tmp_path, profile):
    memory = memory_at(tmp_path, profile)
    tools = build_default_tool_registry()
    tools.bind_handler("ik_preview_check", lambda ctx: ToolResult(True))
    bind_profile_proposals(tools, profile, lambda: memory)
    ctx = build_tool_context(observation=EnvObservation(task="fixture", cameras=[], robot=RobotState()), memory=memory,
        tools=tools, skills=build_default_skill_registry(), config=PlannerContextConfig(agent_interface_profile=profile))["agent_context"]
    references = {r["name"]: r for r in ctx["available_tools"]}
    assert "propose_motion_target" in references
    assert references["ik_preview_check"]["parameters"]["required"] == ["bundle_id"]
    assert "target_pose" not in references["ik_preview_check"]["parameters"]["properties"]
    assert ("compose_ik_trajectory" in references) == (profile == "bundle_stage3")
