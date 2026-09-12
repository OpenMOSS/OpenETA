import pytest

from adapter.protocol import CameraFrame
from sim import bench_worker as worker


@pytest.mark.parametrize("render,terminated,truncated", [
    (False, False, False), (True, False, False),
    (False, True, False), (False, False, True),
])
def test_intermediate_step_skips_pixels_but_preserves_cached_and_final_evidence(
    monkeypatch, render, terminated, truncated,
):
    handle = "intermediate-camera-probe"
    raw = {"cameras": {"agentview": {"rgb": [[[120, 30, 40]]], "depth": [[1.0]]}},
           "robot": {"joint_positions": [0.25]}}
    class Env:
        calls = 0
        def step(self, action):
            self.calls += 1
            return raw, 1.0 if terminated else 0.0, terminated, truncated, {}
    env = Env()
    monkeypatch.setitem(worker._envs, handle, env)
    monkeypatch.setattr(worker, "_inject_render_frame", lambda *a: None)
    original = CameraFrame.from_dict
    converted = []
    def from_dict(*args, **kwargs):
        converted.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(CameraFrame, "from_dict", from_dict)
    try:
        result = worker._step_with_image(env, [0.0], handle=handle, render=render)
        keep_images = render or terminated or truncated
        assert bool(converted) is keep_images
        assert bool(result["observation"]["cameras"]) is keep_images
        assert result["observation"]["robot"]["joint_positions"] == [0.25]
        assert worker._last_obs[handle] is raw
        assert raw["cameras"]["agentview"]["rgb"] == [[[120, 30, 40]]]
        if terminated or truncated:
            again = worker._step_with_image(env, [1.0], handle=handle, render=False)
            assert env.calls == 1
            assert again["observation"] == result["observation"]
            assert again["reward"] == result["reward"]
    finally:
        worker._last_obs.pop(handle, None)
        worker._done_handles.discard(handle)
        worker._terminal_step_results.pop(handle, None)
        with worker._obs_locks_guard:
            worker._obs_locks.pop(handle, None)
