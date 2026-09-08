"""Offline solver tests: synthetic FK is not a LIBERO success claim."""
from __future__ import annotations

import inspect
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sim.reachability import _solve_problem, check_endpoint_reachability
from sim.ik_search_policy import DEFAULT_IK_MAX_ATTEMPTS, DEFAULT_IK_MAX_NFEV_PER_ATTEMPT, DEFAULT_IK_TIMEOUT_S
from agent.tools.sim_mcp import _ik_preview_receipt, _ik_execution_authorization, _ik_recovery_options, _response_diagnostics


def _problem(monkeypatch):
    pytest.importorskip("scipy", reason="numerical solver checks run in the LIBERO Python environment")
    data = SimpleNamespace(qpos=np.zeros(3), qvel=np.zeros(3),
                           site_xpos=np.zeros((1, 3)), xmat=np.eye(3).reshape(1, 9))
    def forward(model, state):
        state.site_xpos[0] = state.qpos
    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(mj_forward=forward))
    return {"model": object(), "data": data, "current_qpos": np.zeros(3),
            "current_arm_q": np.zeros(3), "current_eef_quat_xyzw": [0, 0, 0, 1],
            "qpos_indices": [0, 1, 2], "lower": np.full(3, -1.0),
            "upper": np.full(3, 1.0), "site_id": 0, "body_id": 0, "backend": "synthetic"}


@pytest.mark.parametrize("target,found", [([0.2, 0.1, -0.3], True), ([4, 0, 0], False)])
def test_search_distinguishes_solution_from_no_solution(monkeypatch, target, found):
    problem = _problem(monkeypatch)
    original = problem["current_qpos"].copy()
    result = _solve_problem(problem, target_xyz=target, target_quat_xyzw=[0, 0, 0, 1],
        preserve_current_orientation=True, position_tolerance_m=0.002,
        orientation_tolerance_rad=0.05, max_attempts=4, max_nfev_per_attempt=50, timeout_s=2)
    assert np.array_equal(problem["current_qpos"], original)
    assert result["status"] == ("reachable" if found else "unknown")
    assert result["feasible"] is (True if found else None)
    assert result["solver"]["search_budget"] == {
        "max_attempts": 4, "max_nfev_per_attempt": 50, "timeout_s": 2,
    }
    receipt = _ik_preview_receipt({"target_pose": {"xyz": target}}, result)
    assert _ik_execution_authorization(receipt)["authorized_for_move_to"] is found
    if not found:
        assert result["reason_code"] == "ik_search_no_solution"
        assert receipt["classification"] == "inconclusive"
        assert _ik_recovery_options(receipt)[1]["action"] == "inspect_search_evidence"
        diagnostic = _response_diagnostics(result)[0]
        assert diagnostic["code"] == "ik_search_no_solution"
        assert diagnostic["failure_class"] == "ik_search_inconclusive"


def test_search_timeout_is_unknown_and_records_effective_budget(monkeypatch):
    problem = _problem(monkeypatch)
    ticks = iter(range(100))
    monkeypatch.setattr("sim.reachability.time.monotonic", lambda: next(ticks))
    result = _solve_problem(problem, target_xyz=[0.2, 0, 0], target_quat_xyzw=None,
        preserve_current_orientation=True, position_tolerance_m=0.002,
        orientation_tolerance_rad=0.05, max_attempts=4, max_nfev_per_attempt=50, timeout_s=0.1)
    assert result["status"] == "unknown"
    assert result["reason_code"] == "ik_search_timeout"
    assert result["solver"]["search_budget"]["timeout_s"] == 0.1


def test_direct_and_mcp_defaults_share_host_search_policy():
    from sim.mcp_server.server import ik_preview_check
    expected = {"max_attempts": DEFAULT_IK_MAX_ATTEMPTS,
                "max_nfev_per_attempt": DEFAULT_IK_MAX_NFEV_PER_ATTEMPT,
                "timeout_s": DEFAULT_IK_TIMEOUT_S}
    assert expected == {"max_attempts": 64, "max_nfev_per_attempt": 1000, "timeout_s": 30}
    for function in (check_endpoint_reachability, ik_preview_check):
        signature = inspect.signature(function)
        assert {k: signature.parameters[k].default for k in expected} == expected


def test_timeout_keeps_completed_search_work_and_best_residual(monkeypatch):
    problem = _problem(monkeypatch)
    from sim.reachability import _SearchTimeout
    calls = []
    def solve(residual, seed, **kwargs):
        if calls:
            raise _SearchTimeout
        calls.append(1)
        residual(seed)
        return SimpleNamespace(x=seed, nfev=7)
    monkeypatch.setattr("scipy.optimize.least_squares", solve)
    result = _solve_problem(problem, target_xyz=[4, 0, 0], target_quat_xyzw=None,
        preserve_current_orientation=True, position_tolerance_m=0.002,
        orientation_tolerance_rad=0.05, max_attempts=4, max_nfev_per_attempt=50, timeout_s=2)
    assert result["reason_code"] == "ik_search_timeout"
    assert result["solver"]["attempts_completed"] == 1
    assert result["solver"]["function_evaluations"] == 7
    assert result["solver"]["residual_evaluations"] == 1
    assert result["best_candidate"]["position_error_m"] == pytest.approx(4)
