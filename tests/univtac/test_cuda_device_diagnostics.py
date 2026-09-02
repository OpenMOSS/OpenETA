from __future__ import annotations

import json

import pytest

from scripts.univtac.probe_uipc_device import _ensure_simulation_playing
from sim.envs.univtac.cuda_device_diagnostics import (
    classify_uipc_runs,
    compare_cuda_modes,
    gate_followups,
    validate_fixed_seed,
)


def _cuda(uuid: str, pci: str, *, success: bool = True) -> dict:
    return {
        "success": success,
        "nvidia_smi_logical_zero": {"uuid": uuid, "pci_bus_id": pci},
    }


def test_headless_uipc_harness_starts_stopped_timeline_once() -> None:
    class FakeSimulation:
        def __init__(self) -> None:
            self.playing = False
            self.play_calls = 0

        def play(self) -> None:
            self.play_calls += 1
            self.playing = True

    simulation = FakeSimulation()
    original = lambda instance: instance.playing
    assert _ensure_simulation_playing(simulation, original) == (True, True)
    assert _ensure_simulation_playing(simulation, original) == (True, False)
    assert simulation.play_calls == 1


def test_cuda_modes_must_map_to_same_physical_gpu() -> None:
    same = compare_cuda_modes(_cuda("gpu-a", "0000:01"), _cuda("gpu-a", "0000:01"))
    different = compare_cuda_modes(
        _cuda("gpu-a", "0000:01"), _cuda("gpu-b", "0000:02")
    )
    assert same["classification"] == "same_physical_device"
    assert different["classification"] == "cuda_device_ordinal_mismatch"


def test_cuda_failure_is_preserved_even_with_same_mapping() -> None:
    comparison = compare_cuda_modes(
        _cuda("gpu-a", "0000:01", success=False), _cuda("gpu-a", "0000:01")
    )
    assert comparison["d0_success"] is False
    assert comparison["same_physical_gpu_uuid"] is True


def test_only_fixed_ftp1_seeds_are_accepted() -> None:
    for seed in (1000000, 1000001, 1000002):
        assert validate_fixed_seed(seed) == seed
    with pytest.raises(ValueError):
        validate_fixed_seed(1000003)


def test_followup_seeds_require_planner_or_reset_return() -> None:
    base = {"seed": 1000000, "planning_call_count": 0, "reset_returned": False}
    assert gate_followups(base) == []
    assert gate_followups({**base, "planning_call_count": 1}) == [1000001, 1000002]
    assert gate_followups({**base, "reset_returned": True}) == [1000001, 1000002]


def test_uipc_requires_two_passing_runs_in_same_mode() -> None:
    passing = {
        "mode": "d0_visible_gpu0",
        "returncode": 0,
        "completed_step": True,
        "invalid_device": False,
        "cleanup_complete": True,
    }
    summary = classify_uipc_runs([passing, dict(passing)])
    assert summary["consecutive_two_pass"] is True
    assert summary["classification"] == "uipc_sentinel_passed"


def test_uipc_intervening_same_mode_failure_breaks_pass_streak() -> None:
    passing = {
        "mode": "d0_visible_gpu0",
        "returncode": 0,
        "completed_step": True,
        "invalid_device": False,
        "cleanup_complete": True,
    }
    failed = {**passing, "returncode": 1, "completed_step": False}
    summary = classify_uipc_runs([passing, failed, passing])
    assert summary["consecutive_two_pass"] is False


def test_persistent_invalid_device_is_not_a_planner_failure() -> None:
    failed = {
        "mode": "d0_visible_gpu0",
        "returncode": -6,
        "completed_step": False,
        "invalid_device": True,
        "cleanup_complete": True,
    }
    summary = classify_uipc_runs([failed, {**failed, "mode": "d1_unset_visible_devices"}])
    assert summary["classification"] == "persistent_uipc_invalid_device_clean_state"
    assert "planner" not in json.dumps(summary).lower()


def test_clean_native_no_step_is_not_external_or_planner_failure() -> None:
    early_exit = {
        "mode": "d0_visible_gpu0",
        "returncode": 0,
        "completed_step": False,
        "invalid_device": False,
        "cleanup_complete": True,
        "native_result_missing": True,
        "stage": "official_main_started",
    }
    summary = classify_uipc_runs(
        [early_exit, {**early_exit, "mode": "d1_unset_visible_devices"}]
    )
    assert summary["classification"] == "uipc_no_step_clean_state"
    assert summary["consecutive_two_pass"] is False
    assert "planner" not in summary["classification"]


def test_clean_uipc_timeout_is_classified_separately() -> None:
    timed_out = {
        "mode": "d0_visible_gpu0",
        "returncode": 0,
        "timed_out": True,
        "completed_step": False,
        "invalid_device": False,
        "cleanup_complete": True,
        "final_process_group_members": [],
        "new_gpu_pids_after": [],
    }
    summary = classify_uipc_runs(
        [timed_out, {**timed_out, "mode": "d1_unset_visible_devices"}]
    )
    assert summary["classification"] == "uipc_sentinel_timeout_clean_state"
    assert summary["consecutive_two_pass"] is False
