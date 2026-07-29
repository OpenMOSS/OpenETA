from __future__ import annotations

import numpy as np
import pytest

from sim.bench_worker import _inject_render_frame


@pytest.mark.parametrize("backend", ["behavior", "robocasa"])
def test_worker_does_not_inject_uncalibrated_render_into_direct_rgbd(
    backend: str,
) -> None:
    class FakeEnv:
        _backend = backend

        def render(self):
            raise AssertionError("calibrated direct observations must be reused")

    calibrated = {
        "rgb": np.zeros((2, 3, 3), dtype=np.uint8),
        "intrinsics": {"fx": 10.0},
        "extrinsics": {"camera_frame": "opencv"},
    }
    observation = {"cameras": {"scene": calibrated}}

    _inject_render_frame(FakeEnv(), observation)

    assert observation["cameras"] == {"scene": calibrated}


def test_worker_preserves_legacy_libero_render_injection() -> None:
    class FakeEnv:
        _backend = "libero"

        @staticmethod
        def render():
            return np.full((2, 3, 3), 7, dtype=np.uint8)

    observation = {
        "cameras": {
            "agentview": {
                "rgb": np.zeros((2, 3, 3), dtype=np.uint8),
                "extrinsics": {"camera_frame": "opengl"},
            }
        }
    }

    _inject_render_frame(FakeEnv(), observation)

    assert set(observation["cameras"]) == {"agentview", "render"}
    np.testing.assert_array_equal(
        observation["cameras"]["render"]["rgb"],
        np.full((2, 3, 3), 7, dtype=np.uint8),
    )
