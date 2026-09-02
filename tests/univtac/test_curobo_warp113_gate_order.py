from __future__ import annotations

import inspect

from scripts.univtac import run_curobo_warp113_validation as runner


def test_gate_order_and_no_isaac_or_task() -> None:
    source = inspect.getsource(runner.main)
    tokens = (
        'manifest["gates"]["W0"]',
        'manifest["gates"]["W1"]',
        'manifest["gates"]["W2"]',
        'manifest["gates"]["C0"]',
        'manifest["gates"]["C1"]',
        'manifest["gates"]["C2"]',
        'manifest["gates"]["C3"]',
        'manifest["gates"]["C4"]',
        'manifest["gates"]["U2"]',
    )
    positions = [source.index(token) for token in tokens]
    assert positions == sorted(positions)
    assert "AppLauncher" not in source
    assert "task.reset" not in source


def test_build_is_bounded_and_uses_pinned_wrappers() -> None:
    source = inspect.getsource(runner)
    assert source.count('"pip", "install"') == 1
    assert "compiler_wrapper_paths(pinned_source)" in source
    assert "openeta-univtac-main/scripts/toolchains" not in source
    assert '"--no-deps", "--no-build-isolation"' in source
    assert '"additional_retry_allowed": False' not in source
    assert 'parser.add_argument("--invalid-invocation-root"' in source
    assert "shutil.copy2(invalid_process" in source
    assert 'record["realpath"], "--version"' in source


def test_collision_finite_precedes_libuipc() -> None:
    source = inspect.getsource(runner.main)
    assert source.index("probe_curobo_collision_finite.py") < source.index("probe_libuipc_core.py")


def test_viable_is_assigned_only_after_environment_and_cleanup() -> None:
    source = inspect.getsource(runner.main)
    assert source.index("environment_valid = False") < source.index('"classification": "blackwell_native_bridge_viable"')
    assert source.index("cleanup_valid = False") < source.index('"classification": "blackwell_native_bridge_viable"')


def test_static_warning_process_must_finish_and_cleanup() -> None:
    source = inspect.getsource(runner.main)
    assert "static_audit_process_usable(binary_audit, static)" in source
    assert "static_audit_covers_binaries(static, binaries)" in source


def test_invalid_invocation_evidence_requires_failed_clean_process() -> None:
    source = inspect.getsource(runner.main)
    assert 'invalid_payload.get("returncode") != 1' in source
    assert 'invalid_payload.get("timed_out") is not False' in source
    assert 'invalid_payload.get("cleanup_complete") is not True' in source


def test_environment_diff_compares_all_non_curobo_pip_lines() -> None:
    source = inspect.getsource(runner.main)
    assert 'semantic_diff["pip_dependencies_unchanged"]' in source
