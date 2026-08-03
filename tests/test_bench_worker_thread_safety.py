from __future__ import annotations

import asyncio
import json
import threading

import pytest

import sim.bench_worker as bench_worker
from sim.bench_worker import (
    _EnvUnavailableError,
    _close_env_locked,
    _env_errors,
    _env_lock,
    _env_locks,
    _env_locks_guard,
    _env_render_is_busy,
    _env_step_busy_until,
    _envs,
    _last_obs,
    _note_env_step_activity,
    _run_env_locked,
    _run_sim_call,
    _step_with_image,
    _shutdown_sim_executor,
    _startup_sim_executor,
    _try_run_env_locked,
    app,
    render_all_envs,
)


def _clear_lock(handle: str) -> None:
    with _env_locks_guard:
        _env_locks.pop(handle, None)
    _envs.pop(handle, None)
    _last_obs.pop(handle, None)
    _env_errors.pop(handle, None)
    _env_step_busy_until.pop(handle, None)


def test_dashboard_render_skips_when_environment_is_busy() -> None:
    handle = "busy-env"
    lock = _env_lock(handle)
    lock.acquire()
    try:
        called = False

        def render() -> str:
            nonlocal called
            called = True
            return "rendered"

        assert _try_run_env_locked(handle, render) is None
        assert _env_render_is_busy(handle) is True
        assert called is False
    finally:
        lock.release()
        _clear_lock(handle)


def test_world_changing_call_owns_per_environment_lock() -> None:
    handle = "step-env"
    entered = threading.Event()
    release = threading.Event()

    def step() -> str:
        entered.set()
        assert release.wait(timeout=2)
        return "stepped"

    result: list[str] = []
    worker = threading.Thread(target=lambda: result.append(_run_env_locked(handle, step)))
    worker.start()
    assert entered.wait(timeout=2)
    assert _try_run_env_locked(handle, lambda: "rendered") is None
    release.set()
    worker.join(timeout=2)

    assert result == ["stepped"]
    assert _try_run_env_locked(handle, lambda: "rendered") == "rendered"
    _clear_lock(handle)


def test_lock_wrapper_forwards_simulator_handle_keyword() -> None:
    env_handle = "lock-key"

    def reset(*, handle: str) -> str:
        return handle

    assert _run_env_locked(env_handle, reset, handle="sim-handle") == "sim-handle"
    _clear_lock(env_handle)


def test_dashboard_render_skips_between_steps_in_motion_burst() -> None:
    handle = "moving-env"
    _note_env_step_activity(handle)

    assert _try_run_env_locked(handle, lambda: "rendered") is None

    _clear_lock(handle)


def test_close_keeps_lock_and_rejects_operation_that_captured_old_env() -> None:
    handle = "closing-env"
    operation_captured_env = threading.Event()
    allow_operation_to_lock = threading.Event()
    operation_ran = False
    errors: list[BaseException] = []

    class Env:
        closed = False

        def close(self) -> None:
            self.closed = True

    env = Env()
    _envs[handle] = env
    _last_obs[handle] = {"stale": True}
    _env_errors[handle] = "stale error"
    _env_step_busy_until[handle] = float("inf")
    lock = _env_lock(handle)

    def stale_operation() -> None:
        nonlocal operation_ran
        captured_env = _envs[handle]
        operation_captured_env.set()
        assert allow_operation_to_lock.wait(timeout=2)
        try:
            _run_env_locked(
                handle,
                lambda: setattr(env, "operation_ran", True),
                _expected_env=captured_env,
            )
            operation_ran = True
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=stale_operation)
    worker.start()
    assert operation_captured_env.wait(timeout=2)

    assert _close_env_locked(handle, env, close_simulator=True) is True
    assert env.closed is True
    assert _env_lock(handle) is lock
    assert handle not in _envs
    assert handle not in _last_obs
    assert handle not in _env_errors
    assert handle not in _env_step_busy_until

    allow_operation_to_lock.set()
    worker.join(timeout=2)

    assert worker.is_alive() is False
    assert operation_ran is False
    assert len(errors) == 1
    assert isinstance(errors[0], _EnvUnavailableError)
    assert not hasattr(env, "operation_ran")
    _clear_lock(handle)


def test_dashboard_render_rechecks_step_grace_after_acquiring_lock() -> None:
    handle = "grace-race-env"
    called = False

    class GraceActivatingLock:
        released = False

        def acquire(self, *, blocking: bool) -> bool:
            assert blocking is False
            _env_step_busy_until[handle] = float("inf")
            return True

        def release(self) -> None:
            self.released = True

    lock = GraceActivatingLock()
    with _env_locks_guard:
        _env_locks[handle] = lock  # type: ignore[assignment]

    def render() -> str:
        nonlocal called
        called = True
        return "rendered"

    assert _try_run_env_locked(handle, render) is None
    assert called is False
    assert lock.released is True
    _clear_lock(handle)


def test_control_step_keeps_raw_camera_cache_but_omits_response_cameras() -> None:
    handle = "camera-free-control-step"
    rgb = [[[1, 2, 3], [4, 5, 6]]]
    depth = [[0.1, 0.2]]

    class Env:
        def step(self, _act):
            return (
                {
                    "task_description": "test",
                    "cameras": {
                        "wrist": {
                            "rgb": rgb,
                            "depth": depth,
                            "intrinsics": {},
                            "extrinsics": {},
                        }
                    },
                    "proprio": {},
                },
                0.0,
                False,
                False,
                {},
            )

    try:
        result = _step_with_image(
            Env(),
            [0.0],
            handle=handle,
            render=False,
            include_cameras=False,
        )

        assert result["observation"]["cameras"] == []
        assert _last_obs[handle]["cameras"]["wrist"]["rgb"] is rgb
        assert _last_obs[handle]["cameras"]["wrist"]["depth"] is depth
    finally:
        _clear_lock(handle)


def test_simulator_calls_and_render_all_stay_on_create_thread(monkeypatch) -> None:
    handle = "thread-affinity-env"
    caller_thread_id = threading.get_ident()
    calls: list[tuple[str, int]] = []

    def record(name: str) -> dict:
        thread_id = threading.get_ident()
        calls.append((name, thread_id))
        return {"operation": name, "thread_id": thread_id}

    class Env:
        def reset(self) -> dict:
            return record("reset")

        def step(self) -> dict:
            return record("step")

        def observe(self) -> dict:
            return record("observe")

        def check(self) -> dict:
            return record("check")

        def render(self) -> dict:
            return record("render_all")

        def close(self) -> None:
            record("close")

    def create() -> Env:
        record("create")
        return Env()

    class RenderAllRequest:
        method = "POST"
        app = app

        async def json(self) -> dict:
            return {"handles": [handle]}

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            executor = app.state.sim_executor
            try:
                env = await _run_sim_call(create)
                _envs[handle] = env
                _env_lock(handle)

                for method_name in ("reset", "step", "observe", "check"):
                    result = await _run_sim_call(
                        _run_env_locked,
                        handle,
                        getattr(env, method_name),
                        _expected_env=env,
                    )
                    assert result["operation"] == method_name

                monkeypatch.setattr(
                    bench_worker,
                    "_observe_with_image",
                    lambda render_env, handle="": render_env.render(),
                )
                response = await render_all_envs(RenderAllRequest())
                payload = json.loads(response.body)
                assert payload["by_handle"][handle]["operation"] == "render_all"

                assert await _run_sim_call(
                    _close_env_locked,
                    handle,
                    env,
                    close_simulator=True,
                ) is True
            finally:
                _clear_lock(handle)

        assert app.state.sim_executor is None
        with pytest.raises(RuntimeError, match="after shutdown"):
            executor.submit(lambda: None)

    asyncio.run(scenario())

    assert [name for name, _ in calls] == [
        "create",
        "reset",
        "step",
        "observe",
        "check",
        "render_all",
        "close",
    ]
    simulator_thread_ids = {thread_id for _, thread_id in calls}
    assert len(simulator_thread_ids) == 1
    assert simulator_thread_ids != {caller_thread_id}


def test_render_all_skips_instead_of_queueing_behind_busy_env(monkeypatch) -> None:
    handle = "render-all-busy-env"
    entered = threading.Event()
    release = threading.Event()
    render_called = False

    class Env:
        pass

    env = Env()

    def blocking_world_call() -> None:
        entered.set()
        assert release.wait(timeout=2)

    def render(*args, **kwargs) -> dict:
        nonlocal render_called
        render_called = True
        return {"rendered": True}

    class RenderAllRequest:
        method = "POST"
        app = app

        async def json(self) -> dict:
            return {"handles": [handle]}

    async def scenario() -> None:
        _startup_sim_executor()
        _envs[handle] = env
        _env_lock(handle)
        blocker = asyncio.create_task(_run_sim_call(
            _run_env_locked,
            handle,
            blocking_world_call,
            _expected_env=env,
        ))
        try:
            while not entered.is_set():
                await asyncio.sleep(0)
            response = await asyncio.wait_for(
                render_all_envs(RenderAllRequest()),
                timeout=0.5,
            )
            payload = json.loads(response.body)
            assert payload["by_handle"][handle] == {
                "skipped": True,
                "reason": "env_busy",
            }
            assert render_called is False
        finally:
            release.set()
            await blocker
            _clear_lock(handle)
            _shutdown_sim_executor()

    monkeypatch.setattr(bench_worker, "_observe_with_image", render)
    asyncio.run(scenario())
