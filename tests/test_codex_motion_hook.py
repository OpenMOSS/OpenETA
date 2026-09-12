"""Exercise native motion composition through the real planner/runner/gates."""
import json
from copy import deepcopy

import pytest

from agent.runtime.episode import DummyEpisodeEnvironment, OpenEtaEpisodeRunner
from agent.runtime.memory import AgentMemory
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import PlannerContextConfig, ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.tools.bundle_proposals import bind_profile_proposals, proposal_contract
from agent.tools.contracts import build_default_tool_contract_catalog
from agent.tools.handlers import bind_dummy_tool_handlers
from agent.tools.registry import ToolResult, build_default_tool_registry
from agent.tools.sim_mcp import _ik_preview_receipt, _ik_execution_authorization
from tools.codex_host import CodexHost, SubmittedBackend

POSE = {"frame": "world", "xyz": [.1, .2, 1.], "quat_xyzw": [1., 0., 0., 0.]}


@pytest.fixture
def rig(tmp_path):
    backend = SubmittedBackend()
    tools = bind_dummy_tool_handlers(build_default_tool_registry())
    catalog = build_default_tool_contract_catalog(tools.list())
    for name in ("propose_motion_target", "compose_ik_trajectory"):
        catalog.register(proposal_contract(name))
    runtime = OpenEtaAgentRuntime(
        planner=ToolCallingPlanner(backend, max_validation_retries=0, tool_contract_catalog=catalog,
            context_config=PlannerContextConfig(agent_interface_profile="bundle_stage3")),
        pipeline=ActionPipeline(agent_interface_profile="bundle_stage3", tool_contract_catalog=catalog),
        memory=AgentMemory(artifact_root=tmp_path / "artifacts"), tools=tools)
    bind_profile_proposals(tools, "bundle_stage3", lambda: runtime.memory)
    state = {"calls": [], "feasible": True, "reason": "ik_solution_found", "stop_reason": "target_reached"}

    def preview(ctx):
        state["calls"].append(("ik_preview_check", deepcopy(ctx.parameters)))
        if state.get("transport_failure"):
            raise TimeoutError("Synthetic IK transport timeout")
        reach = {"feasible": state["feasible"], "status": "reachable" if state["feasible"] else "unknown",
                 "kinematic_status": "reachable" if state["feasible"] else "unknown",
                 "reason_code": state["reason"], "collision": {"checked": True, "collision": False},
                 "position_only_reachable": True, "orientation_only_reachable": True}
        receipt = _ik_preview_receipt(ctx.parameters, reach)
        authorization = _ik_execution_authorization(receipt)
        return ToolResult(True, "Synthetic IK evidence", {"outputs": {
            "ik_preview_receipt": receipt, "ik_receipt_id": receipt["receipt_id"],
            "execution_authorization": authorization, "reachability": reach,
            "motion_execution_ref": {"ik_receipt_id": receipt["receipt_id"]} if authorization["authorized_for_move_to"] else {}}})

    def move(ctx):
        state["calls"].append(("move_to", deepcopy(ctx.parameters)))
        return ToolResult(True, "Synthetic motion receipt", {"outputs": {"motion_summary": {
            "reached_target": state["stop_reason"] == "target_reached", "stop_reason": state["stop_reason"]}}})

    tools.bind_handler("ik_preview_check", preview, replace=True)
    tools.bind_handler("move_to", move, replace=True)
    runner = OpenEtaEpisodeRunner(runtime=runtime, environment=DummyEpisodeEnvironment())
    runner.start(task="move fixture", max_turns=30, max_tool_calls=30, timeout_s=60)
    runtime.memory.add_observation(runner.current_observation)
    host = CodexHost(runner, backend, output=tmp_path / "host", max_requests=30)
    assert host.motion_hook_enabled
    yield host, state
    host.close()


def response(result):
    return json.loads(result.content[0].text)


def test_one_native_pose_request_checks_then_moves_exact_pose_and_tolerances(rig):
    host, state = rig
    result = host.call("move_to", {"target_pose": POSE, "tolerance": .007, "ori_tolerance": .12, "num_steps": 200})
    assert not result.isError, response(result).get("error")
    assert [n for n, _ in state["calls"]] == ["ik_preview_check", "move_to"]
    check, move = [p for _, p in state["calls"]]
    assert check["target_pose"]["xyz"] == move["target_pose"]["xyz"] == POSE["xyz"]
    assert check["position_tolerance_m"] == move["tolerance"] == .007
    assert check["orientation_tolerance_rad"] == move["ori_tolerance"] == .12
    assert move["enable_collision_check"] is True
    assert move["num_steps"] == 200
    assert host.requests == 1 and host.runner.turn_index == 3 and host.runner.tool_call_count == 3
    commands = [json.loads(s) for s in (host.output / "host-commands.jsonl").read_text().splitlines()]
    assert all(r["parent_request"]["parameters"]["target_pose"] == POSE for r in commands)
    assert response(result)["motion_hook"]["reason_code"] == "target_reached"


@pytest.mark.parametrize("reason", ["ik_search_no_solution", "ik_search_timeout"])
def test_inconclusive_preview_never_dispatches_motion(rig, reason):
    host, state = rig
    state.update(feasible=None, reason=reason)
    result = host.call("move_to", {"target_pose": POSE})
    assert result.isError
    assert response(result)["error"]["code"] == reason
    assert [n for n, _ in state["calls"]] == ["ik_preview_check"]
    assert response(result)["motion_hook"]["motion_dispatched"] is False


def test_target_bundle_skips_proposal_and_retains_embedded_tolerance(rig):
    host, state = rig
    assert not host.call("propose_motion_target", {"target_pose": POSE, "position_tolerance_m": .006}).isError
    bid = host.runtime.memory.tool_handoffs()[-1]["bundle_id"]
    result = host.call("move_to", {"bundle_id": bid})
    assert not result.isError, response(result).get("error")
    assert len(response(result)["motion_hook"]["stages"]) == 2
    assert state["calls"][0][1]["position_tolerance_m"] == state["calls"][1][1]["tolerance"] == .006


def test_unknown_bundle_and_mixed_reference_never_call_simulator(rig):
    host, state = rig
    for args in ({"bundle_id": "bnd-" + "a" * 32},
                 {"bundle_id": "bnd-" + "a" * 32, "target_pose": POSE}):
        assert host.call("move_to", args).isError
    assert state["calls"] == []


def test_budget_expiry_between_preview_and_move_stops_motion(rig):
    host, state = rig
    host.runner.max_turns = 2
    result = host.call("move_to", {"target_pose": POSE})
    assert result.isError and host.closed
    assert [n for n, _ in state["calls"]] == ["ik_preview_check"]


def test_host_invariant_substitution_stops_hook(rig, monkeypatch):
    from agent.runtime.planner import PlannerDecision
    host, state = rig
    monkeypatch.setattr("agent.runtime.planner._invariant_obligation_decision",
        lambda *a, **kw: PlannerDecision(action_type="tool_call", action="observe", parameters={}))
    assert host.call("move_to", {"target_pose": POSE}).isError
    assert state["calls"] == []
    assert host.backend.pending is None


def test_ik_success_does_not_hide_controller_iteration_limit(rig):
    host, state = rig
    state["stop_reason"] = "iteration_limit"
    result = host.call("move_to", {"target_pose": POSE})
    assert not result.isError, response(result).get("error")
    hook = response(result)["motion_hook"]
    assert hook["execution_authorization"]["authorized_for_move_to"] is True
    assert hook["reason_code"] == "iteration_limit"
    assert hook["motion_summary"]["reached_target"] is False


@pytest.mark.parametrize("tolerance,ori_tolerance", [(.001, .02), (.002, .05)])
def test_existing_ik_bundle_is_rechecked_with_new_execution_tolerance(rig, tolerance, ori_tolerance):
    host, state = rig
    assert not host.call("propose_motion_target", {"target_pose": POSE}).isError
    target = host.runtime.memory.tool_handoffs()[-1]["bundle_id"]
    assert not host.call("ik_preview_check", {"bundle_id": target, "position_tolerance_m": .002,
                                             "orientation_tolerance_rad": .05}).isError
    checked = next(b["bundle_id"] for b in host.runtime.memory.tool_handoffs() if b["kind"] == "ik_result")
    state["calls"].clear()
    result = host.call("move_to", {"bundle_id": checked, "tolerance": tolerance, "ori_tolerance": ori_tolerance})
    assert not result.isError, response(result).get("error")
    assert [n for n, _ in state["calls"]] == ["ik_preview_check", "move_to"]
    assert state["calls"][0][1]["position_tolerance_m"] == state["calls"][1][1]["tolerance"] == tolerance
    assert state["calls"][0][1]["orientation_tolerance_rad"] == state["calls"][1][1]["ori_tolerance"] == ori_tolerance


def test_stale_target_bundle_never_reaches_ik_or_motion(rig, monkeypatch):
    host, state = rig
    assert not host.call("propose_motion_target", {"target_pose": POSE}).isError
    bid = host.runtime.memory.tool_handoffs()[-1]["bundle_id"]
    epoch = host.runtime.memory.object_scene_epoch()
    monkeypatch.setattr(host.runtime.memory, "object_scene_epoch", lambda: epoch + 1)
    assert host.call("move_to", {"bundle_id": bid}).isError
    assert not state["calls"]


def test_tool_call_budget_covers_internal_hook_stages(rig):
    host, state = rig
    # The runner's owned budget is used by the registry for all nested stages.
    host.runner._tool_call_budget._limit = 2
    result = host.call("move_to", {"target_pose": POSE})
    assert result.isError
    assert all(n != "move_to" for n, _ in state["calls"])


def test_ik_transport_failure_does_not_fall_back_to_motion(rig):
    host, state = rig
    state["transport_failure"] = True
    result = host.call("move_to", {"target_pose": POSE})
    assert result.isError
    assert response(result)["motion_hook"]["failure_stage"] == "ik_preview_check"
    assert response(result)["motion_hook"]["motion_dispatched"] is False
    assert [n for n, _ in state["calls"]] == ["ik_preview_check"]


def test_collision_stop_after_good_ik_is_exposed_separately(rig):
    host, state = rig
    state["stop_reason"] = "collision_detected"
    result = host.call("move_to", {"target_pose": POSE})
    hook = response(result)["motion_hook"]
    assert hook["execution_authorization"]["authorized_for_move_to"] is True
    assert hook["reason_code"] == "collision_detected"
    assert hook["motion_summary"]["reached_target"] is False
