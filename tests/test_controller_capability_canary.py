"""Fail-closed gates of the live canary; these fixtures are not live evidence."""

from argparse import Namespace
from copy import deepcopy

import pytest

from scripts import controller_capability_canary as canary


def arguments(**overrides):
    return Namespace(**{
        "url": "http://127.0.0.1:8766/sse",
        "env_id": "openeta/libero_libero_object_task2-v0",
        "seed": 2, "image_size": 256, "delta_z_m": 0.02,
        "tolerance_m": 0.003, "max_steps": 40, "timeout_s": 10.0,
        "expected_controller_id": "robosuite.osc_pose",
        "disable_collision_check": False, "expected_move_rejection_code": "",
        **overrides,
    })


@pytest.fixture
def backend(monkeypatch):
    controller = {
        "controller_id": "robosuite.osc_pose",
        "command_interface": "normalized_cartesian_delta_pose",
        "goal_executor": "openeta.outer_closed_loop_cartesian.v1",
    }
    move = {"reached_target": True, "steps_executed": 4, "stop_reason": "target_reached"}
    replies = {
        "create_env": {"handle": "new-handle", "session_id": "new-session",
                       "control_spec": {"controller": controller}},
        "reset_env": {"end_effector_pose": {"xyz": [0.0, 0.0, 0.3]}},
        "ik_preview_check": {"status": "reachable", "collision": {
            "checked": True, "scene_objects_included": True, "detected": False,
            "world_checked": True, "self_checked": True,
        }},
        "move_to": {**move, "end": {"xyz": [0.0, 0.0, 0.32]}, "controller_receipt": {
            **controller, **move,
            "schema_version": "openeta.controller_execution_receipt.v1",
        }, "collision": {"detected": False, "world_checked": True, "self_checked": True}},
        "close_env": {"ok": True},
    }
    calls = []

    class Transport:
        def call_tool(self, name, parameters, *, timeout_s):
            calls.append((name, parameters, timeout_s))
            reply = replies[name]
            if isinstance(reply, Exception):
                raise reply
            return deepcopy(reply)

    monkeypatch.setattr(canary, "SseSimulatorMcpTransport", lambda url: Transport())
    return replies, calls


def test_canary_requires_motion_receipt_and_confirmed_cleanup(backend):
    _, calls = backend
    report = canary.run(arguments())
    assert report["passed"] is True
    assert report["checks"]["cleanup_confirmed"] is True
    assert [call[0] for call in calls] == [
        "create_env", "reset_env", "ik_preview_check", "move_to", "close_env",
    ]
    assert calls[-2][1]["enable_collision_check"] is True
    assert calls[-3][1]["check_endpoint_collision"] is True
    assert calls[-3][1]["include_scene_objects"] is True
    assert calls[-1][1]["handle"] == "new-handle"


def _seeded_backend(backend, monkeypatch):
    replies, calls = backend
    monkeypatch.setattr(canary, "uuid4", lambda: Namespace(hex="fixture-seed"))
    controller = {
        "controller_id": "mink.robosuite_joint_velocity", "command_interface": "joint_velocity",
        "goal_executor": "openeta.worker_mink_goal.v1",
    }
    replies["create_env"]["control_spec"]["controller"] = controller
    replies["ik_preview_check"]["best_candidate"] = {"joint_positions": [0.1] * 7}
    replies["move_to"]["controller_receipt"].update({
        **controller, "execution_policy": "ik_preview_seeded_cartesian_goal",
        "ik_execution_seed_receipt_id": "controller-canary-fixture-seed",
    })
    return replies, calls, arguments(expected_controller_id=controller["controller_id"], require_ik_seed=True)


def test_seed_canary_forwards_exact_preview_and_requires_consumption_receipt(backend, monkeypatch):
    _, calls, args = _seeded_backend(backend, monkeypatch)
    report = canary.run(args)
    assert report["passed"] is True
    move = next(parameters for name, parameters, _ in calls if name == "move_to")
    assert move["ik_execution_seed"]["joint_positions"] == [0.1] * 7
    assert move["ik_execution_seed"]["receipt_id"] == report["seed_probe"]["receipt_id"]
    assert report["checks"]["ik_seed_execution_policy_used"] is True


def test_canary_dispatches_through_real_runtime_receipt_resolution(backend, monkeypatch):
    from agent.tools.sim_mcp import _ik_preview_receipt

    replies, calls, args = _seeded_backend(backend, monkeypatch)
    args.through_agent_runtime = True
    receipt = _ik_preview_receipt({
        "target_pose": {"frame": "world", "xyz": [0.0, 0.0, 0.32]},
        "preserve_current_orientation": True,
        "position_tolerance_m": 0.003, "orientation_tolerance_rad": 0.05,
        "check_endpoint_collision": True, "include_scene_objects": True,
    }, replies["ik_preview_check"])
    replies["move_to"]["controller_receipt"]["ik_execution_seed_receipt_id"] = receipt["receipt_id"]
    report = canary.run(args)
    assert report.get("error") is None
    assert report["passed"] is True
    assert report["seed_probe"]["source"] == "agent_runtime_exact_receipt_resolution"
    runtime = report["agent_runtime_probe"]
    assert runtime["request"]["ik_receipt_id"] == receipt["receipt_id"]
    assert runtime["pipeline_status"] == "executed"
    assert runtime["tool_admission"]["admitted_this_run"] == 1
    assert runtime["remote_motion_calls"] == 1
    assert [name for name, _, _ in calls] == [
        "create_env", "reset_env", "ik_preview_check", "move_to", "close_env",
    ]
    assert calls[3][2] == args.timeout_s


@pytest.mark.parametrize("override", [{"require_ik_seed": False}, {"disable_collision_check": True}])
def test_runtime_probe_requires_seed_and_collision_checks_before_rpc(backend, override):
    _, calls = backend
    with pytest.raises(ValueError):
        canary.run(arguments(through_agent_runtime=True, **override))
    assert calls == []


@pytest.mark.parametrize("joints", [None, [], [0.1] * 6, [True] * 7, [float("nan")] * 7, ["0"] * 7])
def test_seed_canary_rejects_invalid_seed_before_motion(backend, monkeypatch, joints):
    replies, calls, args = _seeded_backend(backend, monkeypatch)
    replies["ik_preview_check"]["best_candidate"]["joint_positions"] = joints
    assert canary.run(args)["passed"] is False
    assert "move_to" not in [name for name, _, _ in calls]
    assert calls[-1][0] == "close_env"


@pytest.mark.parametrize("field", ["execution_policy", "ik_execution_seed_receipt_id"])
def test_seed_canary_does_not_accept_unseeded_or_unbound_motion(backend, monkeypatch, field):
    replies, _, args = _seeded_backend(backend, monkeypatch)
    replies["move_to"]["controller_receipt"][field] = "other"
    assert canary.run(args)["passed"] is False


def test_seed_probe_does_not_claim_osc_consumption(backend):
    _, calls = backend
    with pytest.raises(ValueError, match="requires Mink"):
        canary.run(arguments(require_ik_seed=True))
    assert calls == []


def test_controller_mismatch_stops_before_reset_or_motion(backend):
    replies, calls = backend
    replies["create_env"]["control_spec"]["controller"]["controller_id"] = "other"
    report = canary.run(arguments())
    assert report["passed"] is False
    assert report["checks"]["declared_expected_controller"] is False
    assert [call[0] for call in calls] == ["create_env", "close_env"]


@pytest.mark.parametrize("status", ["inconclusive", "unreachable", None])
def test_nonreachable_preview_never_executes_motion(backend, status):
    replies, calls = backend
    replies["ik_preview_check"] = {"status": status}
    report = canary.run(arguments())
    assert report["passed"] is False
    assert not any(call[0] == "move_to" for call in calls)
    assert calls[-1][0] == "close_env"


@pytest.mark.parametrize("reply", [{}, {"ok": False}, {"pending": True}, RuntimeError("close failed")])
def test_cleanup_failure_cannot_pass(backend, reply):
    replies, _ = backend
    replies["close_env"] = reply
    report = canary.run(arguments())
    assert report["passed"] is False
    assert report["checks"]["cleanup_confirmed"] is False
    assert report["create"]["handle"] == "new-handle"


def test_motion_exception_still_attempts_cleanup_and_reports_failure(backend):
    replies, calls = backend
    replies["move_to"] = TimeoutError("motion outcome unknown")
    report = canary.run(arguments())
    assert report["passed"] is False
    assert report["error"]["type"] == "TimeoutError"
    assert calls[-1][0] == "close_env"


@pytest.mark.parametrize("end", [None, {}, {"xyz": [0.0, 0.0, 0.3]},
                                  {"xyz": [False, 0.0, 0.32]},
                                  {"xyz": [0.0, 0.0, float("nan")]}])
def test_reported_target_hit_requires_measured_endpoint(backend, end):
    replies, _ = backend
    replies["move_to"]["end"] = end
    report = canary.run(arguments())
    assert report["passed"] is False
    assert report["checks"]["measured_endpoint_within_tolerance"] is False


@pytest.mark.parametrize("steps", [0, False, -1, 41])
def test_motion_probe_cannot_pass_with_invalid_or_zero_steps(backend, steps):
    replies, _ = backend
    replies["move_to"]["steps_executed"] = steps
    replies["move_to"]["controller_receipt"]["steps_executed"] = steps
    assert canary.run(arguments())["passed"] is False


@pytest.mark.parametrize("overrides", [
    {"max_steps": 0}, {"max_steps": True}, {"timeout_s": float("inf")},
    {"timeout_s": 0}, {"delta_z_m": float("nan")},
    {"delta_z_m": 0.002}, {"tolerance_m": -0.1},
])
def test_invalid_budget_or_noop_target_rejected_before_creation(backend, overrides):
    _, calls = backend
    with pytest.raises(ValueError):
        canary.run(arguments(**overrides))
    assert calls == []


def test_expected_safe_rejection_can_pass_but_requires_cleanup(backend):
    replies, _ = backend
    replies["move_to"] = {
        "reached_target": False, "steps_executed": 0, "code": "collision_rejected",
    }
    report = canary.run(arguments(expected_move_rejection_code="collision_rejected"))
    assert report["passed"] is True
    assert report["checks"]["move_safely_rejected"] is True


@pytest.mark.parametrize("patch", [
    {"checked": False}, {"checked": 1}, {"available": False},
    {"scene_objects_included": False}, {"detected": True},
    {"world_checked": False}, {"self_checked": False}, {"self_checked": 1},
])
def test_incomplete_endpoint_collision_receipt_stops_before_motion(backend, patch):
    replies, calls = backend
    replies["ik_preview_check"]["collision"].update(patch)
    report = canary.run(arguments())
    assert report["passed"] is False
    assert report["checks"]["endpoint_collision_coverage"] is False
    assert not any(name == "move_to" for name, _, _ in calls)
    assert calls[-1][0] == "close_env"


@pytest.mark.parametrize("collision", [
    None, {}, {"detected": False, "available": False},
    {"detected": False, "world_checked": True, "self_checked": False},
    {"detected": True, "world_checked": True, "self_checked": True},
])
def test_reaching_target_with_unavailable_collision_monitoring_cannot_pass(backend, collision):
    replies, _ = backend
    replies["move_to"]["collision"] = collision
    report = canary.run(arguments())
    assert report["passed"] is False
    assert report["probe"]["collision"] == collision
    assert report["checks"]["motion_collision_coverage"] is False


def test_explicit_collision_disabled_debug_run_is_not_acceptance(backend):
    _, calls = backend
    report = canary.run(arguments(disable_collision_check=True))
    assert report["passed"] is False
    assert report["checks"]["collision_checks_requested"] is False
    assert calls[-2][1]["enable_collision_check"] is False


@pytest.mark.parametrize("patch", [{"x": 0.5}, {"num_steps": 400}, {"enable_collision_check": False}])
def test_runtime_probe_guard_rejects_changed_motion_before_rpc(backend, monkeypatch, patch):
    from agent.tools.sim_mcp import SimulatorMcpToolProxy

    _, calls, args = _seeded_backend(backend, monkeypatch)
    args.through_agent_runtime = True
    original = SimulatorMcpToolProxy._move_to_arguments

    def changed(self, parameters, *, metadata=None):
        return {**original(self, parameters, metadata=metadata), **patch}

    monkeypatch.setattr(SimulatorMcpToolProxy, "_move_to_arguments", changed)
    report = canary.run(args)
    assert report["passed"] is False
    assert "move_to" not in [name for name, _, _ in calls]
    assert calls[-1][0] == "close_env"


def test_runtime_motion_timeout_never_retries_and_still_closes(backend, monkeypatch):
    replies, calls, args = _seeded_backend(backend, monkeypatch)
    args.through_agent_runtime = True
    replies["move_to"] = TimeoutError("remote motion outcome unknown")
    report = canary.run(args)
    assert report["passed"] is False
    assert [name for name, _, _ in calls].count("move_to") == 1
    assert calls[-1][0] == "close_env"
