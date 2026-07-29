from __future__ import annotations

import numpy as np

from adapter.protocol import EnvObservation
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


def test_libero_camera_contract_remains_role_free(monkeypatch) -> None:
    env = object.__new__(UnifiedEnv)
    env._include_objects = False
    monkeypatch.setattr(
        env,
        "_extract_camera_params",
        lambda name, image_width, image_height: {
            "intrinsics": {
                "fx": 100.0,
                "fy": 100.0,
                "cx": image_width / 2.0,
                "cy": image_height / 2.0,
            },
            "extrinsics": {
                "pos": [1.0, 2.0, 3.0],
                "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "matrix_layout": "row_major",
                "frame_transform": "camera_to_world",
                "camera_frame": "opengl",
            },
        },
    )
    top = np.array([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.uint8)
    wrist = np.array([[[7, 8, 9]], [[10, 11, 12]]], dtype=np.uint8)

    normalized = env._normalise_libero(
        {
            "agentview_image": top,
            "robot0_eye_in_hand_image": wrist,
        }
    )
    observation = EnvObservation.from_dict(normalized)
    mcp_cameras = observation.to_mcp_dict()["cameras"]

    assert list(normalized["cameras"]) == ["agentview", "wrist"]
    assert all("role" not in camera for camera in normalized["cameras"].values())
    assert [camera.frame_id for camera in observation.cameras] == ["agentview", "wrist"]
    assert [camera.role for camera in observation.cameras] == ["", ""]
    assert all("role" not in camera for camera in mcp_cameras)
    assert all("depth_encoding" not in camera for camera in mcp_cameras)
    assert all("depth_scale" not in camera for camera in mcp_cameras)
    assert all(
        camera.extrinsics
        == {
            "pos": [1.0, 2.0, 3.0],
            "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "matrix_layout": "row_major",
            "frame_transform": "camera_to_world",
            "camera_frame": "opengl",
        }
        for camera in observation.cameras
    )
    np.testing.assert_array_equal(
        np.asarray(observation.cameras[0].rgb),
        np.flipud(top),
    )


def test_behavior_and_robocasa_publish_additive_camera_roles(monkeypatch) -> None:
    env = object.__new__(UnifiedEnv)
    env._backend = "robocasa"
    env._include_objects = False
    monkeypatch.setattr(env, "_depth_to_metres", lambda value: np.asarray(value))
    monkeypatch.setattr(
        env,
        "_robocasa_camera_params",
        lambda name, image_width, image_height: {
            "fx": 100.0,
            "fy": 100.0,
            "cx": image_width / 2.0,
            "cy": image_height / 2.0,
            "width": image_width,
            "height": image_height,
            "extrinsics": {
                "pos": [0.0, 0.0, 0.0],
                "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "matrix_layout": "row_major",
                "frame_transform": "camera_to_world",
                "camera_frame": "opengl",
            },
        },
    )

    behavior = EnvObservation.from_dict(
        env._normalise_behavior(
            {
                "main_images": np.zeros((2, 3, 3), dtype=np.uint8),
                "wrist_images": np.zeros((2, 2, 3, 3), dtype=np.uint8),
            }
        )
    )
    robocasa = EnvObservation.from_dict(
        env._normalise_robocasa(
            {
                "robot0_agentview_left_image": np.zeros((2, 3, 3), dtype=np.uint8),
                "robot0_agentview_left_depth": np.ones((2, 3), dtype=np.float32),
                "robot0_agentview_right_image": np.zeros((2, 3, 3), dtype=np.uint8),
                "robot0_agentview_right_depth": np.ones((2, 3), dtype=np.float32),
                "robot0_eye_in_hand_image": np.zeros((2, 3, 3), dtype=np.uint8),
                "robot0_eye_in_hand_depth": np.ones((2, 3), dtype=np.float32),
            }
        )
    )

    assert [(camera.frame_id, camera.role) for camera in behavior.cameras] == [
        ("zed_head", "scene_primary"),
        ("wrist_left", "wrist_secondary"),
        ("wrist_right", "wrist_primary"),
    ]
    assert [(camera.frame_id, camera.role) for camera in robocasa.cameras] == [
        ("agentview_left", "scene_primary"),
        ("agentview_right", "scene_secondary"),
        ("wrist", "wrist_primary"),
    ]
