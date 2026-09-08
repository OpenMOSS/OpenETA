from __future__ import annotations

import json
from pathlib import Path

import pytest

from adapter.protocol import EnvAction, EnvObservation, RobotState
from agent.runtime.memory import AgentMemory, ROBOT_MOTION_EPOCH_KEY
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import PlannerDecision, _validate_tool_parameters, build_tool_context
from agent.runtime.skills import build_default_skill_registry
from agent.runtime.tool_bundles import BUNDLE_INDEX_KEY
from agent.tools.contracts import build_default_tool_contract_catalog, check_tool_request_conformance
from agent.tools.registry import build_default_tool_registry, ToolResult
from agent.tools.coding import PythonExecConfig, PythonExecRuntime
from agent.tools.registry import ToolExecutionContext


def _memory(tmp_path):
    memory = AgentMemory(artifact_root=tmp_path / "artifacts")
    memory.start_session(task="pick", session_id="test-session")
    return memory


def _action(name, outputs, parameters=None, success=True):
    return EnvAction(action_type="tool_call", command={
        "request": {"kind": "tool_call", "name": name, "parameters": parameters or {}},
        "status": "executed", "tool_calls": [{
            "name": name, "status": "executed", "parameters": parameters or {},
            "result": {"success": success, "details": {"outputs": outputs}},
        }],
    })


def _compile(memory, tool, parameters, calls):
    tools = build_default_tool_registry()
    def handler(context):
        calls.append(context.parameters)
        return ToolResult(True)
    tools.bind_handler(tool, handler)
    return ActionPipeline().compile(
        PlannerDecision(action_type="tool_call", action=tool, parameters=parameters),
        observation=EnvObservation(task="pick", cameras=[], robot=RobotState()),
        tools=tools, skills=build_default_skill_registry(), memory=memory,
    )


def test_candidate_output_is_durable_inspectable_handoff(tmp_path):
    memory = _memory(tmp_path)
    memory.add_action(_action("grasp_pose_estimate", {
        "result_id": "gpe-1", "grasp_candidates": [{"id": "candidate-7"}],
    }))
    ref = memory.tool_handoffs()[0]
    assert Path(ref["path"]).is_file()
    assert ref["summary"]["candidate_ids"] == ["candidate-7"]
    assert memory.resolve_tool_bundle("compile_grasp_seed", {
        "bundle_id": ref["bundle_id"], "candidate_id": "candidate-7",
    }) == {"grasp_result_id": "gpe-1", "candidate_id": "candidate-7"}
    assert BUNDLE_INDEX_KEY not in memory.planning_context()["working_memory"]["facts"]
    context = build_tool_context(
        observation=EnvObservation(task="pick", cameras=[], robot=RobotState()), memory=memory,
        tools=build_default_tool_registry(), skills=build_default_skill_registry(),
    )
    state = context["agent_context"]["decision_state"]
    assert state["tool_handoffs"][0]["bundle_id"] == ref["bundle_id"]
    assert "artifacts.read_bundle" in state["bundle_usage"]


def test_current_source_handoffs_survive_long_pose_history(tmp_path):
    memory = _memory(tmp_path)
    sources = []
    for kind in ("grasp_candidates", "sam3_detections", "grasp_input", "placement_input"):
        sources.append(memory.register_tool_bundle(kind=kind, producer="fixture",
            reference_parameters={}, summary={})["bundle_id"])
    for n in range(40):
        memory.register_tool_bundle(kind="target_pose", producer="fixture",
            reference_parameters={"target_pose": {"frame": "world", "xyz": [n / 100, 0, 0.3]}},
            summary={}, robot_bound=True)
    refs = memory.tool_handoffs()
    assert len(refs) == 12
    assert set(sources).issubset({r["bundle_id"] for r in refs})
    assert all(r["current_epoch"] for r in refs)
    assert len(memory.tool_handoffs(limit=1)) == 1
    assert len(memory.tool_handoffs(limit=100)) == 32
    newer = memory.register_tool_bundle(kind="grasp_candidates", producer="fixture",
        reference_parameters={}, summary={})
    assert newer["bundle_id"] in {r["bundle_id"] for r in memory.tool_handoffs()}
    assert sources[0] not in {r["bundle_id"] for r in memory.tool_handoffs()}
    memory.save_fact(ROBOT_MOTION_EPOCH_KEY, {"epoch": 1}, source="runtime")
    assert all(not r["current_epoch"] for r in memory.tool_handoffs() if r["kind"] == "target_pose")


def test_ik_handoff_summary_keeps_target_and_parent_association(tmp_path):
    memory = _memory(tmp_path)
    pose = {"frame": "world", "xyz": [0.1, 0.2, 0.3],
            "source_grasp_id": "candidate-7", "compiled_grasp_id": "compiled-1",
            "waypoint_role": "grasp_clearance"}
    target = memory.register_tool_bundle(kind="target_pose", producer="fixture",
        reference_parameters={"target_pose": pose}, summary={}, robot_bound=True)
    action = _action("ik_preview_check", {"ik_preview_receipt": {
        "receipt_id": "ik-1", "classification": "inconclusive", "target_pose": pose,
        "orientation_policy": "preserve_current_orientation", "reason_code": "ik_search_no_solution",
    }}, {"target_pose": pose})
    action.command["metadata"] = {"bundle_request": {"parameters": {"bundle_id": target["bundle_id"]}}}
    memory.add_action(action)
    summary = next(r["summary"] for r in memory.tool_handoffs() if r["kind"] == "ik_result")
    assert summary["target_bundle_id"] == target["bundle_id"]
    assert summary["target_pose"] == pose
    assert summary["classification"] == "inconclusive"


def test_bundle_keeps_existing_compiled_pose_resolution(tmp_path):
    memory = _memory(tmp_path)
    pose = {"frame": "world", "xyz": [0.1, 0.2, 0.3],
            "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "compiled_grasp_id": "compiled-1", "waypoint_role": "grasp_clearance"}
    memory.save_artifact("compiled-1", {
        "type": "compiled_grasp", "compiled_grasp_id": "compiled-1",
        "scene_epoch": 0, "hover_pose": pose,
    }, source="compile_grasp_seed")
    memory.add_action(_action("compile_grasp_seed", {
        "compiled_grasp_id": "compiled-1", "candidate_id": "candidate-7", "hover_pose": pose,
    }))
    ref = memory.tool_handoffs()[0]
    calls = []
    plan = _compile(memory, "ik_preview_check", {"bundle_id": ref["bundle_id"]}, calls)
    assert plan.status.value == "executed"
    assert calls[0]["target_pose"] == pose
    assert plan.metadata["bundle_request"]["parameters"] == {"bundle_id": ref["bundle_id"]}


@pytest.mark.parametrize("classification,allowed", [("feasible", True), ("inconclusive", False), ("hard_infeasible", False)])
def test_ik_bundle_does_not_change_execution_authority(tmp_path, classification, allowed):
    memory = _memory(tmp_path)
    pose = {"frame": "world", "xyz": [0.1, 0.2, 0.3],
            "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    memory.add_action(_action("ik_preview_check", {"ik_preview_receipt": {
        "receipt_id": "ik-1", "classification": classification,
        "target_pose": pose, "orientation_policy": "explicit_orientation",
        "reason_code": "ik_solution_found" if allowed else "ik_search_no_solution",
    }}, {"target_pose": pose}, success=allowed))
    calls = []
    ref = memory.tool_handoffs()[0]
    plan = _compile(memory, "move_to", {"bundle_id": ref["bundle_id"]}, calls)
    assert bool(calls) is allowed
    if not allowed:
        assert plan.status.value == "blocked"
    memory.save_fact(ROBOT_MOTION_EPOCH_KEY, {"epoch": 1}, source="runtime")
    calls.clear()
    plan = _compile(memory, "move_to", {"bundle_id": ref["bundle_id"]}, calls)
    assert plan.status.value == "blocked"
    assert not calls


def test_bundle_rejects_type_override_tamper_and_unregistered_file(tmp_path):
    memory = _memory(tmp_path)
    ref = memory.register_tool_bundle(kind="grasp_candidates", producer="fixture",
        reference_parameters={"grasp_result_id": "gpe-1"}, summary={})
    with pytest.raises(ValueError, match="type mismatch"):
        memory.resolve_tool_bundle("move_to", {"bundle_id": ref["bundle_id"]})
    with pytest.raises(ValueError, match="mixed"):
        memory.resolve_tool_bundle("compile_grasp_seed", {
            "bundle_id": ref["bundle_id"], "grasp_result_id": "gpe-2", "candidate_id": "c",
        })
    other = _memory(tmp_path / "other")
    with pytest.raises(ValueError, match="unknown host bundle"):
        other.resolve_tool_bundle("compile_grasp_seed", {"bundle_id": ref["bundle_id"]})
    other.save_fact(BUNDLE_INDEX_KEY, {ref["bundle_id"]: ref}, source="save_memory")
    with pytest.raises(ValueError, match="unknown host bundle"):
        other.resolve_tool_bundle("compile_grasp_seed", {"bundle_id": ref["bundle_id"]})
    path = Path(ref["path"])
    path.chmod(0o600)
    path.write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        memory.resolve_tool_bundle("compile_grasp_seed", {"bundle_id": ref["bundle_id"]})


def test_invalid_host_bundle_cannot_recursively_resolve_another_bundle(tmp_path):
    memory = _memory(tmp_path)
    ref = memory.register_tool_bundle(kind="ik_result", producer="bad-host-fixture",
        reference_parameters={"bundle_id": "bnd-missing"}, summary={})
    calls = []
    plan = _compile(memory, "move_to", {"bundle_id": ref["bundle_id"]}, calls)
    assert plan.status.value == "blocked"
    assert not calls


def test_handoff_registration_survives_resume_without_reviving_stale_evidence(tmp_path):
    root = tmp_path / "artifacts"
    memory = AgentMemory(store=JsonMemoryStore(tmp_path / "memory"), artifact_root=root)
    memory.start_session(task="pick", session_id="resume-bundles")
    memory.add_action(_action("grasp_pose_estimate", {
        "result_id": "gpe-1", "grasp_candidates": [{"id": "c"}],
    }))
    bundle_id = memory.tool_handoffs()[0]["bundle_id"]
    resumed = AgentMemory(store=JsonMemoryStore(tmp_path / "memory"), artifact_root=root)
    resumed.resume_session("resume-bundles")
    ref = resumed.tool_handoffs()[0]
    assert ref["bundle_id"] == bundle_id
    assert Path(ref["path"]).is_file()
    assert ref["current_epoch"] is False
    # Resume intentionally advances evidence epochs. Persistence does not
    # silently revive a previous environment's execution/perception authority.
    with pytest.raises(ValueError, match="stale"):
        resumed.resolve_tool_bundle("compile_grasp_seed", {
            "bundle_id": bundle_id, "candidate_id": "c",
        })


def test_python_can_read_bundle_but_cannot_rewrite_host_manifest(tmp_path):
    memory = _memory(tmp_path)
    ref = memory.register_tool_bundle(kind="grasp_candidates", producer="fixture",
        reference_parameters={"grasp_result_id": "gpe-1"}, summary={"candidate_ids": ["c"]})
    runtime = PythonExecRuntime(PythonExecConfig(session_root=str(tmp_path)))
    result = runtime.handler(ToolExecutionContext(name="python_exec", spec=build_default_tool_registry().get("python_exec"), parameters={
        "code": f"result = artifacts.read_bundle({ref['bundle_id']!r})['summary']",
    }))
    assert result.success, result.details
    assert result.details["outputs"]["result"] == {"candidate_ids": ["c"]}
    result = runtime.handler(ToolExecutionContext(name="python_exec", spec=build_default_tool_registry().get("python_exec"), parameters={
        "code": f"open({ref['path']!r}, 'w').write('{{}}')",
    }))
    assert not result.success
    assert json.loads(Path(ref["path"]).read_text())["bundle_id"] == ref["bundle_id"]


@pytest.mark.parametrize("tool,params", [
    ("compile_grasp_seed", {"bundle_id": "bnd-test", "candidate_id": "c"}),
    ("ik_preview_check", {"bundle_id": "bnd-test"}),
    ("move_to", {"bundle_id": "bnd-test"}),
])
def test_bundle_schema_and_planner_acceptance_match(tool, params):
    contract = build_default_tool_contract_catalog(build_default_tool_registry().list()).get(tool)
    assert not check_tool_request_conformance(contract, params)
    assert not _validate_tool_parameters(tool, params)


def test_repeated_ik_search_is_visible_and_epoch_scoped(tmp_path):
    memory = _memory(tmp_path)
    def record(receipt_id, budget):
        pose = {"xyz": [0.1, 0.2, 0.3]}
        memory.add_action(_action("ik_preview_check", {"ik_preview_receipt": {
            "receipt_id": receipt_id, "classification": "inconclusive", "target_pose": pose,
            "reachability": {"solver": {"search_budget": {"max_attempts": budget}}},
        }}, {"target_pose": pose}, False))
        return memory.ik_preview_receipts()["latest"]["retry_history"]
    assert record("r1", 24)["attempt_count"] == 1
    second = record("r2", 24)
    assert second["attempt_count"] == 2
    assert second["same_search_previously_failed"]
    assert record("r3", 64)["attempt_count"] == 1
    memory.save_fact(ROBOT_MOTION_EPOCH_KEY, {"epoch": 1}, source="runtime")
    assert record("r4", 24)["attempt_count"] == 1


@pytest.mark.parametrize("extra", [{"compiled_grasp_id": "c"}, {"candidate_id": "c"}, {"waypoint_role": "grasp_contact"}])
def test_partial_reference_cannot_be_mixed_with_pose_bundle(extra):
    params = {"bundle_id": "bnd-test", **extra}
    contract = build_default_tool_contract_catalog(build_default_tool_registry().list()).get("ik_preview_check")
    assert check_tool_request_conformance(contract, params)
    assert _validate_tool_parameters("ik_preview_check", params)


@pytest.mark.parametrize("mode,fields", [
    ("points", {"points": [{"x": 1, "y": 1, "label": 1}], "prompt": "mug"}),
    ("text", {"points": [{"x": 1, "y": 1, "label": 1}], "prompt": "mug"}),
    ("points", {"positive_points": [{"x": 1, "y": 1, "label": 1}], "prompt": "mug"}),
])
def test_sam3_mixed_point_text_prompts_rejected_by_both_contracts(mode, fields):
    params = {"source_packet_id": "obs-1", "mode": mode, **fields}
    contract = build_default_tool_contract_catalog(build_default_tool_registry().list()).get("sam3")
    assert check_tool_request_conformance(contract, params)
    assert _validate_tool_parameters("sam3", params)
