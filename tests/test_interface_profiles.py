from copy import deepcopy

import pytest

from adapter.protocol import EnvAction, EnvObservation, RobotState
from agent.backends.planner import StaticPlannerBackend, _planner_user_prompt
from agent.runtime.interface_profiles import profile_parameter_errors
from agent.runtime.memory import AgentMemory
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import PlannerContextConfig, PlannerDecision, ToolCallingPlanner, _pending_ik_execution_index
from agent.runtime.skills import build_default_skill_registry
from agent.tools.registry import build_default_tool_registry, ToolResult


@pytest.mark.parametrize("tool,old,new", [
    ("move_to", {"ik_receipt_id": "ik-1"}, {"bundle_id": "bnd-" + "1" * 32}),
    ("compile_grasp_seed", {"grasp_result_id": "gpe-1", "candidate_id": "c"},
     {"bundle_id": "bnd-" + "1" * 32, "candidate_id": "c"}),
])
def test_profile_schema_admission_and_pipeline_share_boundary(tool, old, new):
    assert profile_parameter_errors("bundle_stage1", tool, old)
    assert not profile_parameter_errors("bundle_stage1", tool, new)
    assert profile_parameter_errors("bundle_stage1", tool, {**old, **new})
    assert not profile_parameter_errors("legacy_compatible", tool, old)
    calls = []
    tools = build_default_tool_registry()
    tools.bind_handler(tool, lambda ctx: calls.append(ctx))
    plan = ActionPipeline(agent_interface_profile="bundle_stage1").compile(
        PlannerDecision(action_type="tool_call", action=tool, parameters=old),
        observation=EnvObservation(task="test", cameras=[], robot=RobotState()),
        tools=tools, skills=build_default_skill_registry(),
    )
    assert plan.status.value == "blocked" and not calls
    assert "interface_profile_gate" in plan.metadata


def test_actual_planner_request_and_repair_use_experimental_schema(tmp_path):
    class Backend(StaticPlannerBackend):
        def __init__(self):
            super().__init__([
                {"kind": "tool_call", "name": "move_to", "parameters": {"ik_receipt_id": "ik-1"}},
                {"kind": "tool_call", "name": "move_to", "parameters": {"bundle_id": "bnd-" + "1" * 32}},
            ])
            self.requests = []
        def decide(self, request):
            self.requests.append(deepcopy(request))
            return super().decide(request)
    backend = Backend()
    memory = AgentMemory(artifact_root=tmp_path)
    memory.start_session(task="fixture")
    tools = build_default_tool_registry()
    tools.bind_handler("move_to", lambda ctx: ToolResult(True))
    planner = ToolCallingPlanner(backend, context_config=PlannerContextConfig(agent_interface_profile="bundle_stage1"))
    planner.plan(EnvObservation(task="fixture", cameras=[], robot=RobotState()),
                 memory=memory, tools=tools, skills=build_default_skill_registry())
    assert len(backend.requests) == 2
    context = backend.requests[0].tool_context
    schema = next(r for r in context["available_tools"] if r["name"] == "move_to")["parameters"]
    assert schema["required"] == ["bundle_id"]
    assert "ik_receipt_id" not in schema["properties"] and "oneOf" not in schema
    wire = _planner_user_prompt(backend.requests[1])
    assert "bundle_stage1" in wire and "no native-reference fallback" in wire


def test_strict_bundle_resolution_preserves_live_gate_and_agent_history(tmp_path):
    memory = AgentMemory(artifact_root=tmp_path)
    memory.start_session(task="fixture")
    pose = {"frame": "world", "xyz": [0.1, 0.2, 0.3],
            "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    action = EnvAction(action_type="tool_call", command={
        "request": {"kind": "tool_call", "name": "ik_preview_check", "parameters": {"target_pose": pose}},
        "status": "executed", "tool_calls": [{"name": "ik_preview_check", "status": "executed",
        "parameters": {"target_pose": pose}, "result": {"success": True, "details": {"outputs": {
            "ik_preview_receipt": {"receipt_id": "ik-1", "classification": "feasible",
            "target_pose": pose, "orientation_policy": "explicit_orientation", "reason_code": "ik_solution_found"}
        }}}}],
    })
    memory.add_action(action)
    bundle_id = memory.tool_handoffs()[0]["bundle_id"]
    index = _pending_ik_execution_index(memory, profile="bundle_stage1")
    assert index["receipts"][0]["execution_reference"]["parameters"] == {"bundle_id": bundle_id}
    calls = []
    tools = build_default_tool_registry()
    tools.bind_handler("move_to", lambda ctx: calls.append(ctx.parameters) or ToolResult(True))
    plan = ActionPipeline(agent_interface_profile="bundle_stage1").compile(
        PlannerDecision(action_type="tool_call", action="move_to", parameters={"bundle_id": bundle_id}),
        observation=EnvObservation(task="fixture", cameras=[], robot=RobotState()),
        tools=tools, skills=build_default_skill_registry(), memory=memory,
    )
    assert calls and plan.status.value == "executed"
    assert plan.request.parameters == {"ik_receipt_id": "ik-1"}
    memory.add_action(plan.to_env_action())
    actions = [x for x in memory.conversation.items if x.role == "assistant" and x.kind == "action"]
    assert actions[-1].data["request"]["parameters"] == {"bundle_id": bundle_id}


def test_unknown_profile_fails_closed():
    with pytest.raises(ValueError, match="Unknown host"):
        ActionPipeline(agent_interface_profile="typo")


@pytest.mark.parametrize("tool,extra", [
    ("compile_grasp_seed", {"candidate_id": "c"}), ("move_to", {}),
    ("ik_preview_check", {}), ("follow_eef_trajectory", {}),
    ("select_sam3_detection", {"detection_id": "d"}),
    ("reject_sam3_detections", {"reason": "not target"}),
    ("grasp_pose_estimate", {}), ("anyplace", {}),
])
def test_all_migrated_consumers_reject_native_or_whitespace_ids(tool, extra):
    bundle = "bnd-" + "a" * 32
    assert not profile_parameter_errors("bundle_stage3", tool, {"bundle_id": bundle, **extra})
    for bad in ("grasp:legacy", " " + bundle + " ", "bnd-1"):
        assert profile_parameter_errors("bundle_stage3", tool, {"bundle_id": bad, **extra})


def test_repair_feedback_separates_required_fields_from_optional_nulls():
    params = {"bundle_id": "bnd-" + "a" * 32, "detection_id": "detection_002"}
    errors = profile_parameter_errors("bundle_stage3", "select_sam3_detection",
                                      {**params, "identity_anchor_id": None})
    assert errors
    assert "Required fields: detection_id, bundle_id" in errors[0]
    assert "Optional fields may be omitted" in errors[0]
    assert "do not fill them with null" in errors[0]
    assert not profile_parameter_errors("bundle_stage3", "select_sam3_detection", params)
