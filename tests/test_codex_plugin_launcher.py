import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import codex_plugin_smoke as launcher


def fake_login(tmp_path, monkeypatch, *, api_key=False):
    source = tmp_path / "original-codex-home"
    source.mkdir()
    auth = {"auth_mode": "apikey", "OPENAI_API_KEY": "fixture"} if api_key else {
        "auth_mode": "chatgpt", "tokens": {"access_token": "synthetic-test-only"}}
    (source / "auth.json").write_text(json.dumps(auth))
    monkeypatch.setenv("CODEX_HOME", str(source))
    return source


def test_private_install_does_not_modify_login_or_inherit_api_provider(tmp_path, monkeypatch):
    source = fake_login(tmp_path, monkeypatch)
    before = (source / "auth.json").read_bytes()
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key")
    monkeypatch.setenv("OPENETA_LLM_API_BASE", "https://not-used.invalid")
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: commands.append((command, kw)))
    home, work, env = launcher.prepare(tmp_path / "run", ["--task", "fixture"], "gpt-5.6-sol", "medium", "codex")
    assert (source / "auth.json").read_bytes() == before
    assert env["CODEX_HOME"] == str(home)
    assert "OPENAI_API_KEY" not in env and "OPENETA_LLM_API_BASE" not in env
    assert home.stat().st_mode & 0o777 == 0o700
    assert (home / "auth.json").stat().st_mode & 0o777 == 0o600
    config = (home / "config.toml").read_text()
    assert 'forced_login_method = "chatgpt"' in config
    assert 'mcp_optional_startup_grace_ms = 0' in config
    assert 'default_tools_approval_mode = "approve"' in config
    assert "shell_tool = false" in config
    assert len(commands) == 2
    assert all(kw["cwd"] == work for _, kw in commands)


def test_api_key_login_is_rejected_before_install(tmp_path, monkeypatch):
    fake_login(tmp_path, monkeypatch, api_key=True)
    with pytest.raises(RuntimeError, match="ChatGPT"):
        launcher.prepare(tmp_path / "run", [], "gpt-5.6-sol", "medium", "codex")
    assert not (tmp_path / "run/codex-home/auth.json").exists()


def test_pinned_catalog_is_private_and_preserves_model_instructions(tmp_path, monkeypatch):
    fake_login(tmp_path, monkeypatch)
    catalog = {"models": [{"slug": "gpt-6-astra", "tool_mode": None,
                          "base_instructions": "unchanged model instructions"}]}
    source = tmp_path / "source-catalog.json"
    source.write_text(json.dumps(catalog))
    before = source.read_bytes()
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: None)
    home, _, _ = launcher.prepare(tmp_path / "run", [], "gpt-6-astra", "medium", "codex",
                                  model_catalog_json=source)
    copied = home / "model-catalog.json"
    assert json.loads(copied.read_text()) == catalog
    assert source.read_bytes() == before
    assert f'model_catalog_json = {json.dumps(str(copied))}' in (home / "config.toml").read_text()
    assert 'shell_tool = false' in (home / "config.toml").read_text()


def test_failed_install_removes_temporary_auth_and_reports_failure(tmp_path, monkeypatch):
    source = fake_login(tmp_path, monkeypatch)
    before = (source / "auth.json").read_bytes()
    def fail(*a, **kw):
        raise subprocess.CalledProcessError(1, ["codex", "plugin", "add"])
    monkeypatch.setattr(subprocess, "run", fail)
    output = tmp_path / "run"
    args = SimpleNamespace(output=output, sim_url="http://127.0.0.1:1/sse", env_id="fixture",
        task="fixture", seed=0, timeout=1, max_requests=2, sam3_url="", anygrasp_url="",
        model="gpt-5.6-sol", effort="medium", codex="codex", observe_only=True, controller="mink_joint_velocity")
    assert launcher.run(args) == 1
    assert not (output / "codex-home/auth.json").exists()
    assert (source / "auth.json").read_bytes() == before
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "setup_failed"
    assert not summary["integration_passed"]
    assert not summary["task_success"]


@pytest.mark.parametrize("cleanup_ok,official_success,expected", [
    (True, False, 0), (False, False, 1), (True, True, 0),
])
def test_completed_codex_requires_cleanup_and_separates_task_success(
    tmp_path, monkeypatch, cleanup_ok, official_success, expected,
):
    output = tmp_path / "run"
    def prepared(output, *args):
        host_args = args[0]
        assert host_args[host_args.index("--expected-controller") + 1] == "mink_joint_velocity"
        home = output / "codex-home"
        home.mkdir(parents=True)
        (home / "auth.json").write_text("synthetic-test-only")
        (output / "host").mkdir()
        (output / "host/host-status.json").write_text(json.dumps({
            "closed": True, "cleanup": {"ok": cleanup_ok}, "tool_calls": 1,
            "official_task_success": official_success,
        }))
        return home, output, {}
    monkeypatch.setattr(launcher, "prepare", prepared)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw:
        SimpleNamespace(pid=99999999, returncode=0, wait=lambda **kw: 0))
    monkeypatch.setattr(launcher.os, "killpg", lambda *a: None)
    args = SimpleNamespace(output=output, sim_url="http://127.0.0.1:1/sse", env_id="fixture",
        task="fixture", seed=0, timeout=1, max_requests=2, sam3_url="", anygrasp_url="",
        model="gpt-5.6-sol", effort="medium", codex="codex", observe_only=True, controller="mink_joint_velocity")
    assert launcher.run(args) == expected
    summary = json.loads((output / "summary.json").read_text())
    assert summary["integration_passed"] is cleanup_ok
    assert summary["task_success"] is official_success
    assert not (output / "codex-home/auth.json").exists()
