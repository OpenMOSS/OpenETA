from __future__ import annotations

import numpy as np

from sim.camera_conventions import (
    normalise_camera_to_world_opencv,
    quaternion_xyzw_to_rotation_matrix,
)


def test_renderer_to_opencv_conversion_right_multiplies_noncommuting_rotation() -> None:
    rotation_world_renderer = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    expected = rotation_world_renderer @ np.diag([1.0, -1.0, -1.0])

    packet = normalise_camera_to_world_opencv(
        position_xyz=[0.4, -0.2, 1.1],
        rotation_camera_to_world=rotation_world_renderer,
        source_camera_frame="opengl",
        normalized_from="mujoco_opengl",
    )

    actual = np.asarray(packet["mat"]).reshape(3, 3)
    np.testing.assert_allclose(actual, expected)
    assert not np.allclose(actual, np.diag([1.0, -1.0, -1.0]) @ rotation_world_renderer)

    depth = 0.75
    point_opencv = np.asarray([0.0, 0.0, depth])
    position = np.asarray(packet["pos"])
    point_world = actual @ point_opencv + position
    recovered = actual.T @ (point_world - position)
    np.testing.assert_allclose(recovered, point_opencv)
    assert recovered[2] > 0.0


def test_already_opencv_camera_to_world_is_not_converted_twice() -> None:
    rotation = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    packet = normalise_camera_to_world_opencv(
        position_xyz=[1.0, 2.0, 3.0],
        rotation_camera_to_world=rotation,
        source_camera_frame="opencv",
        normalized_from="pre_normalized",
    )

    np.testing.assert_allclose(np.asarray(packet["mat"]).reshape(3, 3), rotation)
    assert packet["camera_frame"] == "opencv"
    assert packet["image_origin"] == "top_left"


def test_usd_quaternion_conversion_uses_native_rotation_on_the_left() -> None:
    half_sqrt = np.sqrt(0.5)
    rotation_world_usd = quaternion_xyzw_to_rotation_matrix(
        [0.0, 0.0, half_sqrt, half_sqrt]
    )

    packet = normalise_camera_to_world_opencv(
        position_xyz=[0.0, 0.0, 0.0],
        rotation_camera_to_world=rotation_world_usd,
        source_camera_frame="omnigibson_usd",
        normalized_from="omnigibson_usd",
    )

    expected = rotation_world_usd @ np.diag([1.0, -1.0, -1.0])
    np.testing.assert_allclose(np.asarray(packet["mat"]).reshape(3, 3), expected)
