from __future__ import annotations

import inspect

from scripts.univtac import resume_isaac51_r095 as runner


def test_artifact_lock_precedes_cache_download_and_offline_install() -> None:
    source = inspect.getsource(runner.main)
    assert source.index('manifest["stages"]["R0"] = "passed"') < source.index('artifact_root = output / "artifact_lock"')
    assert 'manifest["stages"]["L0"] = "failed"' in source
    assert '"wheel_cache_created": False' in source
    assert '"transport_attempts": 0' in source
    assert '"total_isaac_install_invocations": 1' in source
    assert '"authorized_offline_retry_invocations": 0' in source
    assert "pip\", \"install" not in source


def test_r095_does_not_start_isaac_or_agent_and_preserves_seed_stages() -> None:
    source = inspect.getsource(runner)
    assert '"isaac_started": False' in source
    assert '"agent_started": False' in source
    assert '"G0", "TASK_L0", "C0", "H0"' in source
    assert "AppLauncher" not in source
    assert "sudo" not in source and "sysctl -w" not in source
