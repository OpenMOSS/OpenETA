from __future__ import annotations

import inspect

from scripts.univtac import run_blackwell_native_bridge as runner
from sim.envs.univtac.blackwell_adaptation import EXPECTED_GATE_ORDER, next_gate


def test_gate_order_is_a_strict_prefix() -> None:
    completed = []
    for expected in EXPECTED_GATE_ORDER:
        assert next_gate(completed) == expected
        completed.append(expected)
    assert next_gate(completed) is None


def test_curobo_is_after_two_successful_core_runs() -> None:
    assert EXPECTED_GATE_ORDER.index("C0_BUILD") > EXPECTED_GATE_ORDER.index("U1_RUN2")


def test_no_safe_curobo_smoke_is_unresolved_not_fabricated() -> None:
    prefix = list(EXPECTED_GATE_ORDER[:-1])
    assert next_gate(prefix, curobo_smoke_available=False) is None


def test_runner_does_not_install_or_start_isaac_and_never_changes_sysctl() -> None:
    source = inspect.getsource(runner)
    assert "AppLauncher" not in source
    assert "smoke_isaac51" not in source
    assert '"isaacsim"' not in source
    assert '"isaaclab"' not in source
    assert '"sudo"' not in source
    assert "sysctl -w" not in source


def test_runner_fingerprints_legacy_r07_and_uv_environments() -> None:
    source = inspect.getsource(runner)
    assert "fingerprints_before" in source
    assert '"legacy"' in source
    assert '"r07"' in source
    assert '"openeta_uv"' in source
    assert "conda_meta_fingerprint" in source


def test_vendor_tree_remains_unmodified() -> None:
    import subprocess
    from pathlib import Path

    completed = subprocess.run(
        ["git", "diff", "--quiet", "--", "third_party/ftp1-policy/UniVTAC"],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
    )
    assert completed.returncode == 0
