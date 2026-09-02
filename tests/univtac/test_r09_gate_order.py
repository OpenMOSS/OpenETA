from __future__ import annotations

import inspect

from scripts.univtac import install_isaac51_blackwell_runtime as installer
from scripts.univtac import run_isaac51_task_gates as task_runner
from scripts.univtac import validate_isaac51_blackwell_runtime as smoke_runner


def test_installer_never_runs_official_installer_or_starts_isaac() -> None:
    source = inspect.getsource(installer)
    assert "scripts/install.sh" not in source
    assert "AppLauncher" not in source
    assert "smoke_isaac51.py" not in source
    assert '"sysctl"' not in source
    assert '"sudo"' not in source


def test_installer_gate_order_is_clone_e0_p0_i0_i1_i2_n0() -> None:
    source = inspect.getsource(installer.main)
    tokens = (
        'stage="E0"',
        'manifest["stages"]["P0"]',
        'manifest["stages"]["I0"]',
        'manifest["stages"]["I1"]',
        'manifest["stages"]["I2"]',
        'stage="N0"',
    )
    positions = [source.index(token) for token in tokens]
    assert positions == sorted(positions)
    assert source.count('config["install"]["isaaclab_requirement"]') == 2


def test_smokes_are_sequential_taxim_only() -> None:
    source = inspect.getsource(smoke_runner.main)
    assert '(("S0", "smoke_first"), ("S1", "smoke_second"))' in source
    assert '"--backend", "taxim"' in source
    assert "pix2pix" not in source


def test_task_gate_runtime_does_not_load_agent_or_model() -> None:
    source = inspect.getsource(task_runner)
    assert "FTP1InferenceWrapper" not in source
    assert "FTP1ChunkPolicy" not in source
    assert '"ftp1_model_loaded": False' in source
    assert '"openeta_imported": False' in source


def test_g0_failure_stops_but_l0_task_failure_can_continue() -> None:
    successful = {"seed_contract_valid": True, "classification": "episode_saved", "episode_saved": True, "cleanup_success": True}
    unsuccessful = {"seed_contract_valid": True, "classification": "task_unsuccessful", "episode_saved": False, "cleanup_success": True}
    assert task_runner.should_continue("G0", successful) is True
    assert task_runner.should_continue("G0", unsuccessful) is False
    assert task_runner.should_continue("L0", unsuccessful) is True


def test_cleanup_failure_stops_and_cannot_be_classified_viable() -> None:
    dirty = {"seed_contract_valid": True, "classification": "task_unsuccessful", "episode_saved": False, "cleanup_success": False}
    assert task_runner.should_continue("L0", dirty) is False
    assert task_runner.classify({"G0": {**dirty, "episode_saved": True}, "L0": dirty}) == "cleanup_incomplete"


def test_h0_requires_clean_c0_and_both_smokes() -> None:
    c0 = {"seed_contract_valid": True, "reset_returned": True, "expert_trajectory_completed": True, "cleanup_success": True}
    smoke = {"status": "completed", "cleanup_complete": True, "runs": {"S0": {"passed": True}, "S1": {"passed": True}}}
    assert task_runner.c0_allows_h0(c0, smoke) is True
    c0["cleanup_success"] = False
    assert task_runner.c0_allows_h0(c0, smoke) is False
