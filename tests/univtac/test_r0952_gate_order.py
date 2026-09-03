from __future__ import annotations

import inspect
import subprocess

from scripts.univtac import download_isaac_wheelhouse as downloader
from scripts.univtac import resume_isaac51_r0952 as runner


def test_r0952r3_revision_and_r1_evidence_precede_transport() -> None:
    source = inspect.getsource(runner.main)
    assert source.index("inspect_execution_revision(ROOT)") < source.index(
        "push_execution_revision()"
    )
    assert source.index("validate_r0952r1_evidence(r0952r1)") < source.index(
        'manifest["stages"]["T0"] = "passed"'
    )
    assert source.index('manifest["stages"]["T0"] = "passed"') < source.index(
        'manifest["stages"]["T1"] = "passed"'
    )
    assert source.index('manifest["stages"]["T1"] = "passed"') < source.index(
        "download_process = managed("
    )


def test_r0952r3_stops_after_offline_dry_run_without_install() -> None:
    source = inspect.getsource(runner.main)
    assert '"actual_install_executed": False' in source
    assert 'manifest["stages"]["O0"] = "passed"' in source
    assert "pip install" not in source
    assert "AppLauncher" not in source
    assert '"task_started": False' in source
    assert '"agent_started": False' in source


def test_downloader_behavior_file_is_not_part_of_r3_revision() -> None:
    source = inspect.getsource(runner.main)
    assert "download_isaac_wheelhouse.py" in source
    assert "download_process = managed(" in source
    assert downloader.__file__ is not None


def test_push_retries_only_once_for_allowed_transport_failure(monkeypatch) -> None:
    calls = []
    sleeps = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 1, "", "gnutls handshake failed")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    result = runner.push_execution_revision(wait_seconds=30)
    assert result["returncode"] == 0
    assert result["attempt_count"] == 2
    assert calls[1][1:3] == ["-c", "http.version=HTTP/1.1"]
    assert sleeps == [30]


def test_push_does_not_retry_unrelated_failure(monkeypatch) -> None:
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, "", "permission denied")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner.push_execution_revision(wait_seconds=0)
    assert result["returncode"] == 1
    assert result["attempt_count"] == 1
    assert len(calls) == 1
