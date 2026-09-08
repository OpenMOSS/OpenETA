"""Host-local creation/cleanup races; no network or real simulator resources."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from agent.cli import openeta_cli as cli_module
from agent.tools import sim_mcp
from agent.tools.registry import build_default_tool_registry
from agent.tools.sim_mcp import (
    SimulatorMcpEpisodeConfig, SimulatorMcpEpisodeEnvironment,
    SimulatorMcpToolProxyConfig, bind_simulator_mcp_tool_handlers,
)


class Transport:
    def __init__(self, block_at="create_env"):
        self.block_at = block_at
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []
        self.failure = None
        self.close_failure = False

    def call_tool(self, name, arguments, *, timeout_s=None):
        self.calls.append(name)
        if name == self.block_at:
            self.entered.set()
            assert self.release.wait(3), "test did not release transport"
            if self.failure:
                raise self.failure
        if name == "create_env":
            return {"success": True, "handle": "owned", "session_id": "session"}
        if name == "close_env":
            return {"ok": not self.close_failure}
        return {"success": True, "cameras": [], "robot": {}}


@pytest.fixture
def rig(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module.OpenEtaCli, "_build_runtime", lambda self: None)
    cli = cli_module.OpenEtaCli()
    config = cli.state.simulator_mcp_config
    config.materialize_images = config.materialize_text = False
    transport = Transport()
    cli.state.simulator_mcp_transport = transport
    cli.state.simulator_mcp_url = "http://original.invalid/sse"
    environment = SimulatorMcpEpisodeEnvironment(
        transport=transport, tool_proxy_config=config,
        config=SimulatorMcpEpisodeConfig(env_id="fixture", startup_attempts=1),
    )
    registry = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(), transport=transport, config=config,
        tool_names=("create_simulator_env", "close_simulator_env", "observe"),
    )
    return cli, config, transport, environment, registry


@pytest.mark.parametrize("entry", ["tool", "episode"])
@pytest.mark.parametrize("phase", ["create_env", "reset_env"])
def test_startup_blocks_all_close_entries_duplicate_start_and_endpoint_replacement(rig, monkeypatch, entry, phase):
    cli, config, transport, environment, registry = rig
    transport.block_at = phase
    create = lambda: registry.call("create_simulator_env", {"env_id": "fixture"})
    start = create if entry == "tool" else lambda: environment.reset(task="fixture")
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(start)
        try:
            assert transport.entered.wait(1)
            assert config.startup_in_progress
            assert config.handle == ("owned" if phase == "reset_env" else "")
            assert cli.close()["pending"] is True
            assert cli._shutdown_result is None
            assert environment.close()["pending"] is True
            close_result = registry.call("close_simulator_env")
            assert not close_result.success
            assert close_result.details["outputs"]["pending"] is True
            assert not create().success
            with pytest.raises(RuntimeError, match="startup"):
                environment.reset(task="duplicate")
            assert not registry.call("observe").success
            monkeypatch.setattr(cli_module, "_load_sim_mcp_url", lambda: "http://new.invalid/sse")
            with pytest.raises(RuntimeError, match="cleanup is unconfirmed"):
                cli_module._ensure_simulator_mcp_transport(cli)
            assert cli.state.simulator_mcp_transport is transport
            assert transport.calls == (["create_env", "reset_env"] if phase == "reset_env" else ["create_env"])
        finally:
            transport.release.set()
        result = future.result(timeout=2)
    if entry == "tool":
        assert result.success
    assert not config.startup_in_progress
    assert config.handle == "owned"
    assert cli.close()["ok"] is True
    assert config.handle == ""
    assert transport.calls == ["create_env", "reset_env", "close_env"]


@pytest.mark.parametrize("entry", ["tool", "episode"])
@pytest.mark.parametrize("phase", ["create_env", "reset_env"])
def test_local_startup_reservation_is_released_on_transport_failure(rig, entry, phase):
    _, config, transport, environment, registry = rig
    transport.block_at = phase
    transport.release.set()
    transport.failure = TimeoutError("fixture transport failed")
    if entry == "tool":
        assert not registry.call("create_simulator_env", {"env_id": "fixture"}).success
    else:
        with pytest.raises(TimeoutError):
            environment.reset(task="fixture")
    assert not config.startup_in_progress
    # A known handle remains available for cleanup; a failed create with no
    # returned identity is still an unresolved remote-outcome problem.
    assert config.handle == ("owned" if phase == "reset_env" else "")


def test_cli_failed_close_is_retryable_and_blocks_create_while_in_flight(rig):
    cli, config, transport, _, registry = rig
    config.handle, config.session_id = "owned", "session"
    transport.block_at = "close_env"
    transport.close_failure = True
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(cli.close)
        try:
            assert transport.entered.wait(1)
            assert cli.close()["pending"] is True
            assert not registry.call("close_simulator_env").success
            assert not registry.call("create_simulator_env", {"env_id": "fixture"}).success
        finally:
            transport.release.set()
        assert future.result(timeout=2)["ok"] is False
    assert config.handle == "owned" and config.close_state == "close_failed"
    transport.close_failure = False
    assert cli.close()["ok"] is True
    assert cli.close()["ok"] is True
    assert config.handle == "" and config.close_state == "closed"
    assert transport.calls == ["close_env", "close_env"]


def test_invalid_create_releases_reservation_without_dispatch(rig):
    _, config, transport, _, registry = rig
    assert not registry.call("create_simulator_env", {"env_id": ""}).success
    assert not config.startup_in_progress
    assert transport.calls == []


def test_cli_cached_close_does_not_hide_later_environment(rig):
    cli, config, transport, _, registry = rig
    assert cli.close()["skipped"] is True
    transport.release.set()
    assert registry.call("create_simulator_env", {"env_id": "fixture"}).success
    assert config.handle == "owned"
    assert cli.close()["closed"] is True
    assert transport.calls == ["create_env", "reset_env", "close_env"]


def test_cli_cannot_acknowledge_known_handle_without_transport(rig):
    cli, config, _, _, _ = rig
    config.handle = "owned"
    cli.state.simulator_mcp_transport = None
    assert cli.close()["ok"] is False
    assert config.handle == "owned" and config.close_state == "close_failed"


def test_episode_cannot_replace_another_active_shared_identity(rig):
    _, config, transport, environment, _ = rig
    config.handle = "someone-elses-handle"
    with pytest.raises(RuntimeError, match="already active"):
        environment.reset(task="fixture")
    assert not config.startup_in_progress and transport.calls == []


def test_returned_identity_survives_create_callback_failure(rig):
    _, config, transport, _, _ = rig
    transport.release.set()

    def fail_callback(*args):
        raise RuntimeError("callback failed")

    registry = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(), transport=transport, config=config,
        response_callback=fail_callback, tool_names=("create_simulator_env",),
    )
    # Registry converts handler failures to structured results.
    result = registry.call("create_simulator_env", {"env_id": "fixture"})
    assert not result.success
    assert not config.startup_in_progress
    assert (config.handle, config.session_id) == ("owned", "session")
    assert transport.calls == ["create_env"]


@pytest.mark.parametrize("cancel_check", [1, 2, 3])
@pytest.mark.parametrize("close_failure", [False, True])
def test_cancelled_startup_cleans_known_identity_or_retains_it_for_retry(rig, monkeypatch, cancel_check, close_failure):
    _, config, transport, _, registry = rig
    transport.release.set()
    transport.close_failure = close_failure
    checks = 0

    def cancelled(context):
        nonlocal checks
        checks += 1
        return checks >= cancel_check

    monkeypatch.setattr(sim_mcp, "_context_execution_cancelled", cancelled)
    result = registry.call("create_simulator_env", {"env_id": "fixture"})
    assert not result.success
    assert "cleaned up" not in result.content
    assert not config.startup_in_progress
    assert config.handle == ("owned" if close_failure else "")
    assert transport.calls == (["create_env", "reset_env", "close_env"]
                               if cancel_check == 3 else ["create_env", "close_env"])
    if close_failure:
        assert config.close_state == "close_failed"
        transport.close_failure = False
        assert registry.call("close_simulator_env").success
        assert config.handle == ""
