from __future__ import annotations

from sim import bench_worker


class _TerminalEnv:
    def __init__(self) -> None:
        self.step_calls = 0
        self.reset_calls = 0

    def step(self, _action):
        self.step_calls += 1
        return (
            {},
            1.0,
            True,
            False,
            {"source": "official_terminal_step", "step_calls": self.step_calls},
        )

    def reset(self, *, seed=None):
        self.reset_calls += 1
        return {}, {"seed": seed}

    def render(self):
        return None


def _clear_handle(handle: str) -> None:
    bench_worker._envs.pop(handle, None)
    bench_worker._last_obs.pop(handle, None)
    bench_worker._done_handles.discard(handle)
    bench_worker._terminal_step_results.pop(handle, None)
    with bench_worker._obs_locks_guard:
        bench_worker._obs_locks.pop(handle, None)


def test_repeated_step_replays_positive_terminal_result_without_actuation() -> None:
    handle = "terminal-replay-probe"
    env = _TerminalEnv()
    _clear_handle(handle)
    bench_worker._envs[handle] = env
    try:
        first = bench_worker._step_with_image(
            env,
            [0.0],
            handle=handle,
            render=False,
        )
        replayed = bench_worker._step_with_image(
            env,
            [1.0],
            handle=handle,
            render=False,
        )

        assert env.step_calls == 1
        assert first["reward"] == replayed["reward"] == 1.0
        assert first["terminated"] is replayed["terminated"] is True
        assert first["truncated"] is replayed["truncated"] is False
        assert first["observation"] == replayed["observation"]
        assert first["info"]["source"] == "official_terminal_step"
        assert replayed["info"]["source"] == "official_terminal_step"
        assert replayed["info"]["terminal_result_replayed"] is True
    finally:
        _clear_handle(handle)


def test_reset_clears_terminal_result_and_allows_a_new_real_step() -> None:
    handle = "terminal-reset-probe"
    env = _TerminalEnv()
    _clear_handle(handle)
    bench_worker._envs[handle] = env
    try:
        first = bench_worker._step_with_image(
            env,
            [0.0],
            handle=handle,
            render=False,
        )
        bench_worker._reset_with_image(env, seed=7, handle=handle)
        second = bench_worker._step_with_image(
            env,
            [0.0],
            handle=handle,
            render=False,
        )

        assert first["reward"] == second["reward"] == 1.0
        assert env.reset_calls == 1
        assert env.step_calls == 2
        assert second["info"]["step_calls"] == 2
        assert "terminal_result_replayed" not in second["info"]
    finally:
        _clear_handle(handle)
