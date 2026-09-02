from __future__ import annotations

import inspect

from scripts.univtac import run_isaac51_runtime_gates as runner
from scripts.univtac.probe_isaac51_torch import required_architecture
from sim.envs.univtac.isaac51_runtime_validation import gate_sequence_decision


def _result(classification: str = "task_unsuccessful"):
    return {"classification": classification, "seed_contract_valid": True}


def test_gate_order_is_g0_l0_c0_then_conditional_h0() -> None:
    assert gate_sequence_decision({}) == "G0"
    assert gate_sequence_decision({"G0": _result()}) == "L0"
    assert gate_sequence_decision({"G0": _result(), "L0": _result()}) == "C0"
    first_three = {"G0": _result(), "L0": _result(), "C0": _result()}
    assert gate_sequence_decision(first_three, c0_tactile_valid=False) is None
    assert gate_sequence_decision(first_three, c0_tactile_valid=True) == "H0"


def test_seed_contract_violation_stops_all_later_gates() -> None:
    violated = {"classification": "seed_contract_violation", "seed_contract_valid": False}
    assert gate_sequence_decision({"G0": violated}, c0_tactile_valid=True) is None


def test_runtime_error_stops_all_later_gates() -> None:
    assert gate_sequence_decision({"G0": _result("runtime_error")}) is None


def test_runner_never_invokes_sysctl_or_sudo() -> None:
    source = inspect.getsource(runner)
    assert "run_noninteractive_sysctl" not in source
    assert '"sudo"' not in source
    assert "sysctl -w" not in source


def test_legacy_environment_operations_are_read_only_fingerprints() -> None:
    source = inspect.getsource(runner.fingerprint_environment)
    assert "conda_explicit" in source
    assert "conda_export" in source
    assert "pip_freeze" in source
    assert "install" not in source
    assert "update" not in source
    assert "remove" not in source


def test_clean_environment_does_not_inherit_cuda_12_8_library_path(monkeypatch) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda-12.8/lib64")
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda-12.8")
    environment = runner.clean_environment()
    assert "LD_LIBRARY_PATH" not in environment
    assert "CUDA_HOME" not in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"


def test_failed_legacy_fingerprints_are_never_reported_unchanged() -> None:
    failed = {
        "success": False,
        "commands": {"conda_explicit": {"returncode": 1, "sha256": None}},
    }
    comparison = runner.compare_fingerprints(failed, failed)
    assert comparison["verification_complete"] is False
    assert comparison["unchanged"] is False
    assert comparison["classification"] == "verification_failed"


def test_conda_explicit_fingerprint_ignores_one_time_notice_progress() -> None:
    clean = "# platform: linux-64\n@EXPLICIT\nhttps://repo/pkg-a.tar.bz2\n"
    noisy = "Retrieving notices: done\n" + clean
    assert runner._normalize_fingerprint_output("conda_explicit", clean) == (
        runner._normalize_fingerprint_output("conda_explicit", noisy)
    )


def test_blackwell_compute_capability_maps_to_sm120() -> None:
    assert required_architecture((12, 0)) == "sm_120"


def test_supplemental_export_drift_does_not_override_stable_package_sets() -> None:
    before = {
        "success": True,
        "commands": {
            "conda_explicit": {"returncode": 0, "sha256": "a"},
            "pip_freeze": {"returncode": 0, "sha256": "b"},
            "conda_export": {"returncode": 0, "sha256": "c"},
        },
    }
    after = {
        "success": True,
        "commands": {
            "conda_explicit": {"returncode": 0, "sha256": "a"},
            "pip_freeze": {"returncode": 0, "sha256": "b"},
            "conda_export": {"returncode": 0, "sha256": "d"},
        },
    }
    comparison = runner.compare_fingerprints(before, after)
    assert comparison["verification_complete"] is True
    assert comparison["unchanged"] is True
    assert comparison["comparisons"]["conda_export"] is False
