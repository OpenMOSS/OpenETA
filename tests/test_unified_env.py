from __future__ import annotations

import numpy as np

from sim.unified_env import UnifiedEnv


def _normalise_libero_proprio(quaternion_xyzw: list[float]) -> np.ndarray:
    env = object.__new__(UnifiedEnv)
    env._include_objects = False
    raw = {
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array(quaternion_xyzw),
    }
    return env._normalise_libero(raw)["proprio"]["ee_pose"]


def test_libero_eef_quaternion_preserves_robosuite_xyzw_order() -> None:
    pose = _normalise_libero_proprio([0.5, -0.5, 0.5, 0.5])

    np.testing.assert_allclose(pose, [0.1, 0.2, 0.3, 0.5, -0.5, 0.5, 0.5])


def test_libero_identity_eef_quaternion_uses_unified_xyzw_contract() -> None:
    pose = _normalise_libero_proprio([0.0, 0.0, 0.0, 1.0])

    np.testing.assert_allclose(pose[3:], [0.0, 0.0, 0.0, 1.0])
