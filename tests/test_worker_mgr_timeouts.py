from __future__ import annotations

from concurrent.futures import Future
import threading
from typing import Any

from sim.mcp_server import server, worker_mgr
from sim.mcp_server.worker_mgr import BenchWorkerHandle, BenchWorkerManager


def test_delete_proxy_uses_bounded_cleanup_timeout(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_proxy(
        self: BenchWorkerHandle,
        method: str,
        path: str,
        body: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict:
        calls.append(
            {
                "method": method,
                "path": path,
                "body": body,
                "timeout_s": timeout_s,
            }
        )
        return {"ok": True}

    monkeypatch.setenv("OPENETA_WORKER_DELETE_TIMEOUT_S", "7.5")
    monkeypatch.setattr(BenchWorkerHandle, "proxy", fake_proxy)
    manager = BenchWorkerManager.__new__(BenchWorkerManager)
    meta = {"worker_url": "http://127.0.0.1:9999"}

    assert manager.proxy_handle_op(meta, "/env/one", method="DELETE") == {
        "ok": True
    }
    assert manager.proxy_handle_op(meta, "/env/one", method="POST") == {
        "ok": True
    }
    assert calls == [
        {
            "method": "DELETE",
            "path": "/env/one",
            "body": None,
            "timeout_s": 7.5,
        },
        {
            "method": "POST",
            "path": "/env/one",
            "body": None,
            "timeout_s": None,
        },
    ]


def test_worker_timeout_keeps_structured_transport_detail(monkeypatch) -> None:
    def timeout(*_args, **_kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr(worker_mgr.urllib.request, "urlopen", timeout)
    worker = BenchWorkerHandle(
        bench="",
        port=0,
        process=None,
        base_url="http://worker",
    )

    result = worker.proxy("DELETE", "/env/remote", timeout_s=2.0)

    assert result["ok"] is False
    assert result["error_detail"] == {
        "kind": "transport_timeout",
        "message": "Worker request failed: timed out",
        "error_type": "TimeoutError",
        "method": "DELETE",
        "path": "/env/remote",
        "timeout_s": 2.0,
    }


def test_background_refresh_does_not_replace_cache_with_busy_marker(monkeypatch) -> None:
    cache = {"sid": {"remote": {"observation": "last-good-frame"}}}
    monkeypatch.setattr(worker_mgr, "_session_last_obs", cache)
    monkeypatch.setattr(
        worker_mgr,
        "_proxy_render_all",
        lambda _url, _handles: {
            "by_handle": {"remote": {"skipped": True, "reason": "env_busy"}}
        },
    )

    worker_mgr._refresh_cache_for_worker("http://worker", ["remote"], "sid")

    assert cache["sid"]["remote"] == {"observation": "last-good-frame"}


def test_render_all_uses_dedicated_short_timeout(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_proxy(
        self: BenchWorkerHandle,
        method: str,
        path: str,
        body: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict:
        calls.append(
            {
                "method": method,
                "path": path,
                "body": body,
                "timeout_s": timeout_s,
            }
        )
        return {"by_handle": {}}

    monkeypatch.setattr(BenchWorkerHandle, "proxy", fake_proxy)

    assert worker_mgr._proxy_render_all("http://worker", ["one", "two"]) == {
        "by_handle": {}
    }
    assert calls == [
        {
            "method": "POST",
            "path": "/render_all",
            "body": {"handles": ["one", "two"]},
            "timeout_s": 2.0,
        }
    ]


def test_proxy_step_forwards_render_and_camera_response_controls(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    class Manager:
        def proxy_handle_op(self, meta, path, method="GET", body=None):
            calls.append(
                {
                    "meta": meta,
                    "path": path,
                    "method": method,
                    "body": body,
                }
            )
            return {"observation": {"robot": {}}}

    meta = {
        "remote_handle": "remote",
        "worker_url": "http://worker",
        "_sid": "sid",
    }
    monkeypatch.setattr(worker_mgr, "_get_mgr", lambda: Manager())
    monkeypatch.setattr(worker_mgr, "_session_last_obs", {})

    result = worker_mgr._proxy_step(
        meta,
        [0.1, 0.2],
        num_steps=3,
        render=False,
        include_cameras=False,
    )

    assert result == {"observation": {"robot": {}}}
    assert calls == [
        {
            "meta": meta,
            "path": "/env/remote/step",
            "method": "POST",
            "body": {
                "action": [0.1, 0.2],
                "num_steps": 3,
                "render": False,
                "include_cameras": False,
            },
        }
    ]


def test_background_refresh_has_one_inflight_per_session_worker(monkeypatch) -> None:
    class RecordingExecutor:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, tuple[Any, ...]]] = []
            self.futures: list[Future] = []

        def submit(self, fn, *args):
            future: Future = Future()
            self.calls.append((fn, args))
            self.futures.append(future)
            return future

    executor = RecordingExecutor()
    monkeypatch.setattr(worker_mgr, "_render_executor", executor)
    monkeypatch.setattr(worker_mgr, "_render_inflight", {})

    assert worker_mgr._submit_render_refresh("http://worker", ["one"], "sid") is True
    assert worker_mgr._submit_render_refresh("http://worker", ["two"], "sid") is False
    assert worker_mgr._submit_render_refresh("http://worker", ["two"], "other") is True
    assert len(executor.calls) == 2

    executor.futures[0].set_result(None)
    assert worker_mgr._submit_render_refresh("http://worker", ["two"], "sid") is True
    assert len(executor.calls) == 3
    for future in executor.futures[1:]:
        future.set_result(None)


def test_close_timeout_stays_draining_until_retry_is_confirmed(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class Manager:
        def __init__(self) -> None:
            self.responses = [
                {
                    "ok": False,
                    "error": "Worker request failed: timed out",
                    "error_detail": {
                        "kind": "transport_timeout",
                        "message": "Worker request failed: timed out",
                        "method": "DELETE",
                        "path": "/env/remote",
                        "timeout_s": 10.0,
                    },
                },
                {"ok": True},
            ]

        def proxy_handle_op(self, meta, path, method="GET"):
            calls.append((method, path))
            return self.responses.pop(0)

        def release_worker(self, worker_url):
            calls.append(("release", worker_url))

    manager = Manager()
    envs = {
        "sid": {
            "local": {
                "remote_handle": "remote",
                "worker_url": "http://worker",
                "backend": "libero",
            }
        }
    }
    monkeypatch.setattr(server, "_get_mgr", lambda: manager)
    monkeypatch.setattr(server, "_touch_session", lambda _sid: None)
    monkeypatch.setattr(server, "remove_checker", lambda _handle: None)
    monkeypatch.setattr(server, "_session_envs", envs)
    monkeypatch.setattr(server, "_session_last_obs", {"sid": {"remote": {}}})
    monkeypatch.setattr(server, "_world_actions_inflight", {})

    first = server.close_env.__wrapped__("local", session_id="sid")

    assert first["ok"] is False
    assert first["already_closed"] is False
    assert first["cleanup_state"] == "draining"
    assert first["cleanup_error_details"][0]["kind"] == "transport_timeout"
    assert envs["sid"]["local"]["_cleanup_state"] == "draining"
    assert ("release", "http://worker") not in calls
    active = server.list_active_envs.__wrapped__(session_id="sid")
    assert active["count"] == 1
    assert active["draining_count"] == 1
    blocked_action = server.step_env.__wrapped__(
        "local", [0.0] * 7, session_id="sid"
    )
    assert blocked_action["ok"] is False
    assert blocked_action["code"] == "cleanup_in_progress"
    assert calls == [("DELETE", "/env/remote")]

    second = server.close_env.__wrapped__("local", session_id="sid")

    assert second["ok"] is True
    assert second["already_closed"] is False
    assert envs["sid"] == {}
    assert calls == [
        ("DELETE", "/env/remote"),
        ("DELETE", "/env/remote"),
        ("release", "http://worker"),
    ]


def test_release_retry_does_not_repeat_confirmed_remote_delete(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class Manager:
        def __init__(self) -> None:
            self.release_attempts = 0

        def proxy_handle_op(self, meta, path, method="GET"):
            calls.append((method, path))
            return {"ok": True}

        def release_worker(self, worker_url):
            self.release_attempts += 1
            calls.append(("release", worker_url))
            if self.release_attempts == 1:
                raise RuntimeError("pool accounting busy")

    manager = Manager()
    envs = {
        "sid": {
            "local": {
                "remote_handle": "remote",
                "worker_url": "http://worker",
            }
        }
    }
    monkeypatch.setattr(server, "_get_mgr", lambda: manager)
    monkeypatch.setattr(server, "_touch_session", lambda _sid: None)
    monkeypatch.setattr(server, "remove_checker", lambda _handle: None)
    monkeypatch.setattr(server, "_session_envs", envs)
    monkeypatch.setattr(server, "_session_last_obs", {"sid": {}})
    monkeypatch.setattr(server, "_world_actions_inflight", {})

    first = server.close_env.__wrapped__("local", session_id="sid")
    second = server.close_env.__wrapped__("local", session_id="sid")

    assert first["ok"] is False
    assert first["cleanup_state"] == "draining"
    assert first["cleanup_error_details"][0]["stage"] == "release_worker"
    assert second["ok"] is True
    assert calls == [
        ("DELETE", "/env/remote"),
        ("release", "http://worker"),
        ("release", "http://worker"),
    ]


def test_running_move_rejects_second_world_action_without_worker_call(monkeypatch) -> None:
    entered_step = threading.Event()
    release_step = threading.Event()
    proxy_calls: list[str] = []
    move_results: list[dict] = []
    initial = {
        "observation": {
            "robot": {
                "end_effector_pose": {"xyz": [0.0, 0.0, 0.0]},
                "joint_velocities": [0.0] * 7,
            }
        }
    }
    final = {
        "observation": {
            "robot": {
                "end_effector_pose": {"xyz": [0.1, 0.0, 0.0]},
                "joint_velocities": [0.0] * 7,
            }
        },
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
    }

    def blocking_proxy_step(
        meta,
        action,
        num_steps=1,
        render=True,
        include_cameras=None,
    ):
        del include_cameras
        proxy_calls.append("step")
        entered_step.set()
        assert release_step.wait(timeout=5)
        return final

    monkeypatch.setattr(server, "_touch_session", lambda _sid: None)
    monkeypatch.setattr(
        server,
        "_session_envs",
        {
            "sid": {
                "handle": {
                    "backend": "libero",
                    "action_dim": 7,
                    "remote_handle": "remote",
                }
            }
        },
    )
    monkeypatch.setattr(server, "_session_last_obs", {"sid": {"remote": initial}})
    monkeypatch.setattr(server, "_proxy_step", blocking_proxy_step)
    monkeypatch.setattr(server, "_proxy_render", lambda _meta: final)
    monkeypatch.setattr(server, "_world_actions_inflight", {})

    thread = threading.Thread(
        target=lambda: move_results.append(
            server.move_to.__wrapped__(
                "handle",
                0.1,
                0.0,
                0.0,
                num_steps=1,
                settle_steps=0,
                enable_collision_check=False,
                session_id="sid",
            )
        )
    )
    thread.start()
    assert entered_step.wait(timeout=5)

    rejected = server.step_env.__wrapped__(
        "handle", [0.0] * 7, session_id="sid"
    )

    assert rejected["ok"] is False
    assert rejected["code"] == "action_in_progress"
    assert rejected["error_detail"]["active_action"] == "move_to"
    assert rejected["error_detail"]["requested_action"] == "step_env"
    assert proxy_calls == ["step"]

    release_step.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert move_results[0]["steps_executed"] == 1
