"""Endpoint pinning and real launcher argv tests; no live services involved."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agent.cli import openeta_cli as cli_module


@pytest.fixture
def cli(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module.OpenEtaCli, "_build_runtime", lambda self: None)
    return cli_module.OpenEtaCli()


class Transport:
    def __init__(self, url):
        self.url = url
        self.calls = []

    def call_tool(self, name, arguments, *, timeout_s=None):
        self.calls.append((name, dict(arguments)))
        return {"ok": True, "already_closed": False, "cleanup_errors": []}


@pytest.fixture
def endpoints(cli, monkeypatch):
    made = []
    refreshed = []

    def factory(url):
        made.append(Transport(url))
        return made[-1]

    def refresh(owner):
        refreshed.append(owner.state.simulator_mcp_url)
        owner.state.simulator_mcp_tool_catalog = {"url": owner.state.simulator_mcp_url}

    monkeypatch.setattr(cli_module, "SseSimulatorMcpTransport", factory)
    monkeypatch.setattr(cli_module, "_refresh_simulator_mcp_tool_catalog", refresh)
    monkeypatch.setattr(cli_module, "_load_sim_mcp_url", lambda: "http://old.example/sse")
    cli_module._ensure_simulator_mcp_transport(cli)
    return made, refreshed


def test_explicit_endpoint_overrides_file_and_effective_registry_without_rewriting(cli, endpoints, tmp_path):
    path = tmp_path / ".mcp.json"
    original = json.dumps({"mcpServers": {
        "openeta": {"url": "http://alias.example/sse"},
        "openeta-sim": {"url": "http://old.example/sse"},
        "openeta-sam3": {"url": "http://sam.example/sse"},
    }})
    path.write_text(original)
    cli.state.simulator_mcp_url_override = "http://127.0.0.1:18766/sse"
    actual = cli_module._ensure_simulator_mcp_transport(cli)
    cli._refresh_mcp_registry()
    assert actual.url == "http://127.0.0.1:18766/sse"
    assert cli.state.simulator_mcp_tool_catalog["url"] == actual.url
    assert cli.state.mcp_registry["source"] == ".mcp.json + --simulator-mcp-url"
    assert {server["name"]: server["url"] for server in cli.state.mcp_registry["servers"]} == {
        "openeta-sim": actual.url, "openeta-sam3": "http://sam.example/sse",
    }
    assert cli.state.mcp_registry["server_count"] == 2
    assert path.read_text() == original


@pytest.mark.parametrize("state", ["active_handle", "closing", "close_failed", "in_progress"])
@pytest.mark.parametrize("new_url", ["http://new.example/sse", ""])
def test_unconfirmed_environment_cannot_repoint_or_drop_transport(cli, endpoints, monkeypatch, state, new_url):
    made, refreshed = endpoints
    original = made[0]
    config = cli.state.simulator_mcp_config
    if state == "active_handle":
        config.handle = "old-handle"
    elif state == "in_progress":
        config.close_in_progress = True
    else:
        config.close_state = state
    monkeypatch.setattr(cli_module, "_load_sim_mcp_url", lambda: new_url)
    with pytest.raises(RuntimeError, match="cleanup is unconfirmed"):
        cli_module._ensure_simulator_mcp_transport(cli)
    assert cli.state.simulator_mcp_transport is original
    assert cli.state.simulator_mcp_url == "http://old.example/sse"
    assert len(made) == len(refreshed) == 1


def test_rejected_repoint_still_closes_on_original_service(cli, endpoints, monkeypatch):
    made, _ = endpoints
    config = cli.state.simulator_mcp_config
    config.handle, config.session_id = "owned-handle", "owned-session"
    monkeypatch.setattr(cli_module, "_load_sim_mcp_url", lambda: "http://new.example/sse")
    with pytest.raises(RuntimeError):
        cli_module._ensure_simulator_mcp_transport(cli)
    assert cli.close()["ok"] is True
    assert made[0].calls == [("close_env", {"handle": "owned-handle", "session_id": "owned-session"})]
    assert config.handle == ""
    assert cli_module._ensure_simulator_mcp_transport(cli).url == "http://new.example/sse"
    assert len(made) == 2


def test_same_endpoint_reuses_transport_with_active_handle(cli, endpoints):
    made, refreshed = endpoints
    cli.state.simulator_mcp_config.handle = "owned-handle"
    assert cli_module._ensure_simulator_mcp_transport(cli) is made[0]
    assert len(made) == len(refreshed) == 1


def test_active_worker_without_returned_handle_blocks_endpoint_replacement(cli, endpoints):
    made, _ = endpoints
    idle_queries = []
    cli.state.episode_runner = SimpleNamespace(
        wait_for_idle=lambda **kwargs: idle_queries.append(kwargs) or False,
    )
    cli.state.simulator_mcp_url_override = "http://new.example/sse"
    with pytest.raises(RuntimeError, match="worker is still active"):
        cli_module._ensure_simulator_mcp_transport(cli)
    assert idle_queries == [{"timeout_s": 0}]
    assert cli.state.simulator_mcp_transport is made[0]
    assert len(made) == 1


def test_invalid_configured_endpoint_does_not_repoint_existing_transport(cli, endpoints, monkeypatch):
    made, _ = endpoints
    monkeypatch.setattr(cli_module, "_load_sim_mcp_url", lambda: "http://host:bad/sse")
    with pytest.raises(ValueError, match="HTTP"):
        cli_module._ensure_simulator_mcp_transport(cli)
    assert cli.state.simulator_mcp_transport is made[0]
    assert cli.state.simulator_mcp_url == made[0].url
    assert len(made) == 1


def test_failed_constructor_retains_previous_endpoint_and_catalog(cli, endpoints, monkeypatch):
    original = cli.state.simulator_mcp_transport
    old_catalog = dict(cli.state.simulator_mcp_tool_catalog)
    cli.state.simulator_mcp_url_override = "http://new.example/sse"

    def fail(url):
        raise RuntimeError("constructor failed")

    monkeypatch.setattr(cli_module, "SseSimulatorMcpTransport", fail)
    with pytest.raises(RuntimeError, match="constructor failed"):
        cli_module._ensure_simulator_mcp_transport(cli)
    assert cli.state.simulator_mcp_transport is original
    assert cli.state.simulator_mcp_url == original.url
    assert cli.state.simulator_mcp_tool_catalog == old_catalog


def test_removed_endpoint_clears_idle_transport_and_catalog(cli, endpoints, monkeypatch):
    monkeypatch.setattr(cli_module, "_load_sim_mcp_url", lambda: "")
    assert cli_module._ensure_simulator_mcp_transport(cli) is None
    assert cli.state.simulator_mcp_url == ""
    assert cli.state.simulator_mcp_tool_catalog == {}


@pytest.mark.parametrize("servers,expected", [
    ({"sam3": {"url": "http://sam.example/sse"}}, ""),
    ({"openeta": {"url": "http://alias.example/sse"}}, "http://alias.example/sse"),
    ({"openeta": {"url": "http://alias.example/sse"}, "openeta-sim": {"url": "http://named.example/sse"}}, "http://named.example/sse"),
])
def test_only_named_simulator_or_legacy_alias_is_selected(tmp_path, servers, expected):
    path = tmp_path / ".mcp.json"
    path.write_text(json.dumps({"mcpServers": servers}))
    assert cli_module._load_sim_mcp_url(path) == expected


def test_shared_runtime_dispatch_and_cleanup_use_explicit_endpoint(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".mcp.json").write_text(json.dumps({
        "mcpServers": {"openeta-sim": {"url": "http://old.example/sse"}},
    }))
    made = []

    def factory(url):
        made.append(Transport(url))
        return made[-1]

    monkeypatch.setattr(cli_module, "SseSimulatorMcpTransport", factory)
    runtime_cli = cli_module.OpenEtaCli(simulator_mcp_url="http://127.0.0.1:18766/sse")
    runtime_cli.state.simulator_mcp_config.handle = "fixture-handle"
    runtime_cli.state.simulator_mcp_config.session_id = "fixture-session"
    try:
        runtime_cli._require_runtime().tools.call("observe", {})
    finally:
        assert runtime_cli.close()["ok"] is True
    assert len(made) == 1
    assert made[0].url == "http://127.0.0.1:18766/sse"
    assert [name for name, _ in made[0].calls] == ["render_env", "close_env"]
    assert all(arguments["handle"] == "fixture-handle" for _, arguments in made[0].calls)


@pytest.mark.parametrize("url", [
    "", "relative/path", "file:///tmp/server", "http:///sse", "http://host:0/sse",
    "http://host:99999/sse", "http://host:bad/sse", "http://[broken/sse",
    "http://user:SECRET@host/sse", "http://host/sse?token=SECRET",
    "http://host/sse#fragment", "http://host/\nsse", " http://host/sse",
])
def test_invalid_explicit_url_fails_before_cli_creation_without_echoing_secrets(monkeypatch, capsys, url):
    monkeypatch.setattr(cli_module, "OpenEtaCli", lambda **kwargs: pytest.fail("CLI must not be constructed"))
    with pytest.raises(SystemExit) as error:
        cli_module.main(["--simulator-mcp-url", url])
    assert error.value.code == 2
    assert "SECRET" not in capsys.readouterr().err


@pytest.mark.parametrize("url", ["http://127.0.0.1:18766/sse", "https://sim.example/mcp", "http://[::1]:18766/sse"])
def test_url_argument_reaches_constructor_before_runtime_build(monkeypatch, url):
    captured = []
    monkeypatch.setattr(cli_module.OpenEtaCli, "_build_runtime", lambda self: captured.append(self.state.simulator_mcp_url_override))
    monkeypatch.setattr(cli_module.OpenEtaCli, "run", lambda self: None)
    monkeypatch.setattr(cli_module.OpenEtaCli, "close", lambda self: None)
    assert cli_module.main(["--simulator-mcp-url", url]) == 0
    assert captured == [url]


@pytest.mark.parametrize("task", ["goal0", "object0", "long0", "spatial0", "long9"])
@pytest.mark.parametrize("port", ["8766", "18766"])
def test_real_launchers_check_and_pass_same_simulator_port(tmp_path, task, port):
    source = Path(__file__).resolve().parents[1] / "scripts" / f"run_human_vlm_libero_{task}.sh"
    root = tmp_path / "isolated repo"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    launcher = scripts / source.name
    shutil.copy2(source, launcher)
    binaries = root / ".venv" / "bin"
    binaries.mkdir(parents=True)
    (binaries / "openeta").write_text(
        f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n"
    )
    (binaries / "curl").write_text("#!/bin/sh\nexit 0\n")
    (binaries / "nc").write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "from pathlib import Path\nPath(os.environ['NC_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    for path in binaries.iterdir():
        path.chmod(0o700)
    nc_capture = root / "nc.json"
    env = {**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
           "OPENETA_LOCAL_SIM_PORT": port, "OPENETA_SIMULATOR_TIMEOUT_S": "45",
           "NC_CAPTURE": str(nc_capture)}
    result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True, timeout=5, check=True)
    argv = json.loads(result.stdout.splitlines()[-1])
    assert argv[argv.index("--simulator-mcp-url") + 1] == f"http://127.0.0.1:{port}/sse"
    assert argv[argv.index("--simulator-timeout-s") + 1] == "45"
    assert json.loads(nc_capture.read_text())[-2:] == ["127.0.0.1", port]
