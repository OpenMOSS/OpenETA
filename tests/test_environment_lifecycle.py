from __future__ import annotations

import asyncio
import socket
import threading
from types import SimpleNamespace

import pytest

from adapter.environment_lifecycle import close_response_error
from agent.tools.registry import build_default_tool_registry
from agent.tools.sim_mcp import (
    SimulatorMcpEpisodeConfig, SimulatorMcpEpisodeEnvironment, SimulatorMcpToolProxyConfig,
    bind_simulator_mcp_tool_handlers,
)
from sim import bench_worker
from sim.mcp_server import collision, server, session
from sim.mcp_server.env_lifecycle import close_managed_environment, environment_lock


class Manager:
    def __init__(self, responses):
        self.responses = list(responses)
        self.remote_calls = 0
        self.release_calls = 0

    def proxy_handle_op(self, meta, path, method="GET"):
        self.remote_calls += 1
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def release_worker(self, worker_url):
        self.release_calls += 1

    def call_tool(self, name, arguments, *, timeout_s=None):
        return self.proxy_handle_op({}, name)


def _envs():
    return {"session": {"handle": {"remote_handle": "remote", "worker_url": "worker"}}}


def test_mcp_observation_waits_for_complete_same_handle_control(monkeypatch):
    envs = _envs()
    envs["session"]["other"] = {"remote_handle": "other", "phase": "idle"}
    monkeypatch.setattr(server, "_session_envs", envs)
    monkeypatch.setattr(server, "_touch_session", lambda sid: None)
    monkeypatch.setattr(server, "_proxy_observe", lambda meta: {"phase": meta["phase"]})
    middle = threading.Event()
    finish_control = threading.Event()
    observation_requested = threading.Event()
    observation_returned = threading.Event()
    results = []
    errors = []

    @server._serialized_env_control
    def controlled_motion(handle, *, session_id):
        # Model the gap between two outer-loop worker steps. The worker-level
        # render lock is free here, but the full control has not completed.
        envs[session_id][handle]["phase"] = "intermediate"
        middle.set()
        if not finish_control.wait(2):
            errors.append("control release timed out")
        envs[session_id][handle]["phase"] = "final"

    def observe():
        observation_requested.set()
        try:
            results.append(server.observe_env.__wrapped__("handle", session_id="session"))
        except Exception as exc:
            errors.append(str(exc))
        finally:
            observation_returned.set()

    control_thread = threading.Thread(target=controlled_motion, args=("handle",),
                                      kwargs={"session_id": "session"})
    observe_thread = threading.Thread(target=observe)
    control_thread.start()
    try:
        assert middle.wait(1)
        observe_thread.start()
        assert observation_requested.wait(1)
        assert not observation_returned.wait(0.05)
        # Different environments must remain independent while this one waits.
        assert server.observe_env.__wrapped__("other", session_id="session") == {"phase": "idle"}
    finally:
        finish_control.set()
        control_thread.join(2)
        if observe_thread.ident is not None:
            observe_thread.join(2)
    assert not control_thread.is_alive() and not observe_thread.is_alive()
    assert not errors
    assert results == [{"phase": "final"}]


@pytest.mark.parametrize("state", ["closing", "close_failed", "closed"])
def test_mcp_observation_does_not_read_a_retiring_environment(monkeypatch, state):
    envs = _envs()
    envs["session"]["handle"]["close_lifecycle"] = {"state": state}
    monkeypatch.setattr(server, "_session_envs", envs)
    monkeypatch.setattr(server, "_touch_session", lambda sid: None)
    calls = []
    monkeypatch.setattr(server, "_proxy_observe", lambda meta: calls.append(meta) or {})
    result = server.observe_env.__wrapped__("handle", session_id="session")
    assert result["code"] == "environment_closing"
    assert calls == []


@pytest.mark.parametrize("response", [
    {"error": "HTTP unavailable"}, {"ok": False}, {"success": False},
    {"isError": True}, {"ok": True, "cleanup_errors": ["still alive"]},
    {"ok": True, "pending": True}, {}, None, "ok", TimeoutError("lost acknowledgement"),
])
def test_failed_remote_cleanup_preserves_identity_and_reference_for_retry(response):
    manager = Manager([response, {"ok": True}])
    envs = _envs()
    retired = []

    def close():
        return close_managed_environment(
            session_id="session", handle="handle", envs=envs, manager=manager,
            retire=lambda meta: retired.append(meta["remote_handle"]),
        )

    assert close()["ok"] is False
    assert envs["session"]["handle"]["close_lifecycle"]["state"] == "close_failed"
    assert manager.release_calls == 0
    assert retired == []
    assert close()["ok"] is True
    assert manager.release_calls == 1
    assert retired == ["remote"]
    assert envs["session"] == {}
    assert close()["already_closed"] is True
    assert manager.remote_calls == 2


@pytest.mark.parametrize("failing_phase", ["release", "retire"])
def test_retry_after_remote_ack_does_not_repeat_completed_phases(failing_phase):
    manager = Manager([{"ok": True}])
    envs = _envs()
    attempts = 0
    original_release = manager.release_worker

    def fail_once(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected local cleanup failure before side effect")
        if failing_phase == "release":
            original_release(*args)

    if failing_phase == "release":
        manager.release_worker = fail_once
    retire = fail_once if failing_phase == "retire" else lambda meta: None
    for expected in (False, True):
        result = close_managed_environment(
            session_id="session", handle="handle", envs=envs, manager=manager, retire=retire,
        )
        assert result["ok"] is expected
    assert manager.remote_calls == 1
    assert manager.release_calls == 1


@pytest.mark.parametrize("tool_path", [False, True])
def test_agent_close_keeps_bound_handle_until_confirmed(tool_path):
    transport = Manager([TimeoutError("timeout"), {"error": "remote failed"}, {"ok": True}])
    config = SimulatorMcpToolProxyConfig(session_id="session", handle="handle")
    if tool_path:
        tools = bind_simulator_mcp_tool_handlers(
            build_default_tool_registry(), transport=transport, config=config,
            tool_names=("close_simulator_env",),
        )
        close = lambda: tools.call("close_simulator_env").success
    else:
        environment = SimulatorMcpEpisodeEnvironment(
            transport=transport, tool_proxy_config=config,
            config=SimulatorMcpEpisodeConfig(env_id="test", session_id="session", handle="handle"),
        )
        close = lambda: environment.close()["ok"]
    for expected in (False, False, True):
        assert close() is expected
        assert config.handle == ("" if expected else "handle")
        assert config.close_state == ("closed" if expected else "close_failed")
    assert transport.remote_calls == 3


def test_ttl_cleanup_retains_failed_handle_and_sweep_registration(monkeypatch):
    manager = Manager([{"error": "worker unavailable"}, {"ok": True}])
    envs = _envs()
    cache = {"session": {("worker", "remote"): {"frame": "retained"}}}
    activity = {"session": 1.0}
    retired = []
    monkeypatch.setattr(session, "_get_mgr", lambda: manager)
    monkeypatch.setattr(session, "_session_envs", envs)
    monkeypatch.setattr(session, "_session_last_obs", cache)
    monkeypatch.setattr(session, "_session_last_activity", activity)
    monkeypatch.setattr(collision, "remove_checker", retired.append)
    assert session._cleanup_session("session")["ok"] is False
    assert "session" in activity and "handle" in envs["session"]
    assert cache["session"] and not retired and manager.release_calls == 0
    assert session._cleanup_session("session")["ok"] is True
    assert "session" not in activity and "session" not in envs and "session" not in cache
    assert retired == ["handle"]
    assert manager.release_calls == 1


def test_control_cannot_reuse_environment_after_failed_close(monkeypatch):
    manager = Manager([{"error": "worker unavailable"}])
    envs = _envs()
    monkeypatch.setattr(server, "_session_envs", envs)
    monkeypatch.setattr(server, "_get_mgr", lambda: manager)
    monkeypatch.setattr(server, "_touch_session", lambda sid: None)
    assert server.close_env.__wrapped__("handle", session_id="session")["ok"] is False
    result = server.step_env.__wrapped__("handle", action=[0.0], session_id="session")
    assert result["code"] == "environment_closing"
    assert manager.remote_calls == 1


def test_cleanup_lock_identity_survives_waiting_callers():
    lock = environment_lock("session", "handle")
    started = threading.Event()
    entered = threading.Event()

    def waiter():
        waiting_lock = environment_lock("session", "handle")
        assert waiting_lock is lock
        started.set()
        with waiting_lock:
            entered.set()

    with lock:
        thread = threading.Thread(target=waiter)
        thread.start()
        assert started.wait(1)
        assert environment_lock("session", "handle") is lock
        assert not entered.is_set()
    thread.join(1)
    assert entered.is_set()


def test_worker_close_failure_preserves_environment_then_idempotently_retries(monkeypatch):
    class Environment:
        calls = 0

        def close(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("close failed")

    env = Environment()
    monkeypatch.setattr(bench_worker, "_envs", {"handle": env})
    monkeypatch.setattr(bench_worker, "_last_obs", {"handle": {"frame": "saved"}})
    monkeypatch.setattr(bench_worker, "_done_handles", {"handle"})
    monkeypatch.setattr(bench_worker, "_terminal_step_results", {"handle": {"reward": 1}})
    first = bench_worker._close_environment("handle", bench="dummy")
    assert first["ok"] is False
    assert bench_worker._envs["handle"] is env
    assert bench_worker._last_obs["handle"]
    assert "handle" in bench_worker._done_handles
    assert bench_worker._close_environment("handle", bench="dummy")["ok"] is True
    assert not bench_worker._envs and not bench_worker._last_obs
    assert not bench_worker._done_handles and not bench_worker._terminal_step_results
    assert bench_worker._close_environment("handle", bench="dummy")["already_closed"] is True
    assert env.calls == 2


def test_worker_close_waits_off_event_loop_and_serializes_duplicate_requests(monkeypatch):
    # Some outer sandboxes allow socketpair creation but deny send(), silently
    # preventing asyncio's cross-thread self-pipe from waking the selector.
    # Probe that capability before starting threads; exercise this test outside
    # that sandbox too, rather than hanging the whole core test suite.
    try:
        left, right = socket.socketpair()
        try:
            left.send(b"x")
        finally:
            left.close()
            right.close()
    except PermissionError:
        pytest.skip("outer sandbox denies asyncio cross-thread self-pipe send")
    entered = threading.Event()
    release = threading.Event()

    class Environment:
        calls = 0

        def close(self):
            self.calls += 1
            entered.set()
            assert release.wait(2)

    env = Environment()
    monkeypatch.setattr(bench_worker, "_envs", {"handle": env})
    monkeypatch.setattr(bench_worker.app.state, "sim_executor", None, raising=False)
    request = SimpleNamespace(path_params={"handle": "handle"}, app=SimpleNamespace(state=SimpleNamespace(bench="dummy")))

    async def exercise():
        first = asyncio.create_task(bench_worker.close_env(request))
        assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 1), timeout=1.5)
        second = asyncio.create_task(bench_worker.close_env(request))
        await asyncio.sleep(0)
        release.set()
        return await asyncio.wait_for(asyncio.gather(first, second), timeout=2)

    async def bounded_exercise():
        return await asyncio.wait_for(exercise(), timeout=4)

    replies = asyncio.run(bounded_exercise())
    assert all(reply.status_code == 200 for reply in replies)
    assert env.calls == 1


def test_worker_explicit_absence_is_a_valid_close_acknowledgement(monkeypatch):
    monkeypatch.setattr(bench_worker, "_envs", {})
    assert close_response_error(bench_worker._close_environment("missing", bench="dummy")) is None


def test_close_failure_blocks_new_proxy_calls_until_cleanup_is_confirmed():
    transport = Manager([TimeoutError("lost acknowledgement"), {"ok": True}])
    config = SimulatorMcpToolProxyConfig(session_id="session", handle="handle")
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(), transport=transport, config=config,
        tool_names=("close_simulator_env", "observe"),
    )
    assert not tools.call("close_simulator_env").success
    blocked = tools.call("observe")
    assert not blocked.success
    assert blocked.details["diagnostics"][0]["code"] == "environment_closing"
    assert transport.remote_calls == 1
    assert tools.call("close_simulator_env").success


def test_behavior_worker_reference_is_not_released_until_process_exit_is_confirmed():
    from sim.mcp_server.worker_mgr import BenchWorkerHandle, BenchWorkerManager

    class Process:
        running = True

        def terminate(self):
            pass

        def kill(self):
            pass

        def wait(self, timeout):
            if self.running:
                raise TimeoutError("process did not exit")
            return 0

        def poll(self):
            return None if self.running else 0

    process = Process()
    worker = BenchWorkerHandle(bench="behavior", port=1, process=process, base_url="worker", env_count=1)
    manager = BenchWorkerManager.__new__(BenchWorkerManager)
    manager._lock = threading.RLock()
    manager._pools = {"behavior": [worker]}
    with pytest.raises(RuntimeError, match="exit is unconfirmed"):
        manager.release_worker("worker")
    assert manager._pools["behavior"] == [worker]
    assert worker.env_count == 1
    process.running = False
    manager.release_worker("worker")
    assert manager._pools["behavior"] == []


def test_startup_retry_cannot_forget_environment_when_cleanup_fails():
    transport = Manager([
        {"handle": "handle", "session_id": "session"},
        TimeoutError("timeout"),
        {"error": "close not acknowledged"},
    ])
    environment = SimulatorMcpEpisodeEnvironment(
        transport=transport,
        config=SimulatorMcpEpisodeConfig(env_id="test", startup_attempts=2, startup_retry_delay_s=0),
    )
    with pytest.raises(RuntimeError, match="unconfirmed cleanup"):
        environment.reset(task="test")
    assert environment.config.handle == "handle"
    assert environment.tool_proxy_config.close_state == "close_failed"
    assert transport.remote_calls == 3
    with pytest.raises(RuntimeError, match="cleanup is unconfirmed"):
        environment.reset(task="test")
    assert transport.remote_calls == 3
