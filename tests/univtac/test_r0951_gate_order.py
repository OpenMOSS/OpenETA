from __future__ import annotations

import inspect

from scripts.univtac import build_reproducible_legacy_wheels as builder
from scripts.univtac import resume_isaac51_r0951 as runner


def test_bridge_runs_before_cache_or_install() -> None:
    source = inspect.getsource(runner.main)
    assert source.index('manifest["stages"]["R0"] = "passed"') < source.index('bridge_root = output / "legacy_sdist_bridge"')
    assert '"wheel_cache_created": False' in source
    assert '"total_isaac_install_invocations": 1' in source
    assert '"authorized_offline_retry_invocations": 0' in source
    assert '"isaac_started": False' in source and '"agent_started": False' in source


def test_builder_has_two_runs_no_network_build_and_no_third_build() -> None:
    source = inspect.getsource(builder.main)
    assert "for run_index in (1, 2):" in source
    assert '"pip", "wheel", "--no-deps", "--no-build-isolation", "--no-cache-dir"' in source
    assert 'config["builder_environment"]' in source
    assert 'HTTP_PROXY' in inspect.getsource(builder.builder_environment)
    assert "run_index in (1, 2, 3)" not in source


def test_builder_never_installs_business_packages_into_r09() -> None:
    source = inspect.getsource(builder.main)
    assert '"--target"' in inspect.getsource(builder.import_smoke)
    assert 'state["build_invocations"][name] += 1' in source
    assert "AppLauncher" not in source
