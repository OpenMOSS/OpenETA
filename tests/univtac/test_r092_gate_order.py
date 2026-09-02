from __future__ import annotations

import inspect

from scripts.univtac import resume_isaac51_r092 as runner


def test_r092_is_a0_fail_closed_before_any_environment_change() -> None:
    source = inspect.getsource(runner.main)
    assert source.index("audit_packaging_compatibility.py") < source.index('manifest["stages"]["A0"] = "failed"')
    assert '"packaging_install_invocations": 0' in source
    assert '"isaac_actual_install_invocations": 0' in source
    assert '"not_run_due_to_gate"' in source


def test_r092_runner_cannot_modify_or_clone_environment() -> None:
    source = inspect.getsource(runner)
    for forbidden in ("pip\", \"install", "conda\", \"create", "--no-deps", "--upgrade", "scripts/install.sh", "AppLauncher", '"sudo"', "sysctl -w"):
        assert forbidden not in source


def test_runtime_label_and_gate_order_are_fixed() -> None:
    assert runner.R092_RUNTIME_VARIANT.endswith("+isaaclab_packaging23_compatibility_bridge_v1")
    assert runner.STAGES[:7] == ("A0", "A1", "A2", "A3", "A4", "A5", "N23")
    assert runner.STAGES[-4:] == ("G0", "L0", "C0", "H0")
