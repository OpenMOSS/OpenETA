from __future__ import annotations

from typing import Any

from agent.tools.handlers import bind_dummy_tool_handlers
from agent.tools.registry import TOOL_RESULT_SCHEMA_VERSION, build_default_tool_registry


def _candidate(**overrides: Any) -> dict[str, Any]:
    candidate = {
        "id": "grasp_000",
        "frame": "camera",
        "score": 0.9,
        "translation_xyz": [0.1, 0.2, 0.3],
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "depth": 0.03,
        "width": 0.06,
        "height": 0.03,
        "gripper_tip_position_xyz": [0.2, 0.2, 0.3],
    }
    candidate.update(overrides)
    return candidate


def test_camera_pose_to_world_tool_spec_is_read_only_geometry() -> None:
    spec = build_default_tool_registry().get("camera_pose_to_world")

    assert spec.category == "geometry"
    assert spec.effect.value == "read_only"
    assert spec.allows_batched_observation is True
    assert "camera_to_world" in spec.parameters
    assert "camera_extrinsics" in spec.parameters


def test_camera_pose_to_world_transforms_standard_opencv_camera_to_world_payload() -> None:
    tools = bind_dummy_tool_handlers(build_default_tool_registry())
    result = tools.call(
        "camera_pose_to_world",
        {
            "camera_frame_id": "agentview",
            "camera_pose": _candidate(),
            "camera_to_world": {
                "camera_frame": "opencv",
                "camera_to_world": [
                    [0.0, -1.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
        },
    )

    assert result.success is True
    assert result.details["schema_version"] == TOOL_RESULT_SCHEMA_VERSION
    outputs = result.details["outputs"]
    assert outputs["camera_frame_id"] == "agentview"
    assert outputs["input_camera_frame"] == "opencv"
    assert outputs["camera_to_world_frame"] == "opencv"
    assert outputs["camera_to_world_format"] == "camera_to_world"
    assert outputs["world_pose"] == {
        "id": "grasp_000",
        "frame": "world",
        "score": 0.9,
        "translation_xyz": [0.8, 2.1, 3.3],
        "rotation_matrix": [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "depth": 0.03,
        "width": 0.06,
        "height": 0.03,
        "gripper_tip_position_xyz": [0.8, 2.2, 3.3],
    }


def test_camera_pose_to_world_transforms_sim_pos_mat_row_major_opengl_payload() -> None:
    tools = bind_dummy_tool_handlers(build_default_tool_registry())
    result = tools.call(
        "camera_pose_to_world",
        {
            "camera_pose": _candidate(rotation_matrix=None, gripper_tip_position_xyz=None),
            "camera_extrinsics": {
                "pos": [1.0, 2.0, 3.0],
                "mat": [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            },
        },
    )

    assert result.success is True
    outputs = result.details["outputs"]
    assert outputs["camera_to_world_format"] == "pos_mat"
    assert outputs["camera_to_world_matrix_layout"] == "row_major"
    assert outputs["input_camera_frame"] == "opencv"
    assert outputs["camera_to_world_frame"] == "opengl"
    assert outputs["world_pose"]["translation_xyz"] == [1.1, 1.8, 2.7]


def test_camera_pose_to_world_can_override_pos_mat_layout_and_frame() -> None:
    tools = bind_dummy_tool_handlers(build_default_tool_registry())
    result = tools.call(
        "camera_pose_to_world",
        {
            "camera_pose": _candidate(rotation_matrix=None, gripper_tip_position_xyz=None),
            "camera_extrinsics": {
                "camera_frame": "opencv",
                "matrix_layout": "row_major",
                "pos": [1.0, 2.0, 3.0],
                "mat": [
                    0.0,
                    -1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            },
        },
    )

    assert result.success is True
    outputs = result.details["outputs"]
    assert outputs["camera_to_world_format"] == "pos_mat"
    assert outputs["camera_to_world_matrix_layout"] == "row_major"
    assert outputs["camera_to_world_frame"] == "opencv"
    assert outputs["world_pose"]["translation_xyz"] == [0.8, 2.1, 3.3]


def test_camera_pose_to_world_supports_4x4_matrix_payload() -> None:
    tools = bind_dummy_tool_handlers(build_default_tool_registry())
    result = tools.call(
        "camera_pose_to_world",
        {
            "camera_pose": _candidate(rotation_matrix=None, gripper_tip_position_xyz=None),
            "camera_extrinsics": {
                "matrix": [
                    [1.0, 0.0, 0.0, 0.5],
                    [0.0, 1.0, 0.0, -0.5],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            },
        },
    )

    assert result.success is True
    outputs = result.details["outputs"]
    assert outputs["camera_to_world_format"] == "matrix"
    assert outputs["camera_to_world_matrix_layout"] == "row_major"
    assert outputs["world_pose"]["translation_xyz"] == [0.6, -0.3, 1.3]
    assert outputs["world_pose"]["rotation_matrix"] is None
    assert outputs["world_pose"]["gripper_tip_position_xyz"] is None


def test_camera_pose_to_world_fails_closed_for_non_camera_frame() -> None:
    tools = bind_dummy_tool_handlers(build_default_tool_registry())
    result = tools.call(
        "camera_pose_to_world",
        {
            "camera_pose": _candidate(frame="world"),
            "camera_extrinsics": {"pos": [0.0, 0.0, 0.0], "mat": [1.0, 0.0, 0.0] * 3},
        },
    )

    assert result.success is False
    assert result.details["outputs"]["reason"] == "invalid_frame"
    assert result.details["diagnostics"][0]["code"] == "invalid_frame"


def test_camera_pose_to_world_supports_explicit_opengl_to_opencv_bridge() -> None:
    tools = bind_dummy_tool_handlers(build_default_tool_registry())
    result = tools.call(
        "camera_pose_to_world",
        {
            "camera_pose": _candidate(
                camera_frame="opengl",
                translation_xyz=[0.1, -0.2, -0.3],
                rotation_matrix=[
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                gripper_tip_position_xyz=None,
            ),
            "camera_to_world": {
                "camera_frame": "opencv",
                "camera_to_world": [
                    [1.0, 0.0, 0.0, 0.5],
                    [0.0, 1.0, 0.0, -0.5],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
        },
    )

    assert result.success is True
    outputs = result.details["outputs"]
    assert outputs["input_camera_frame"] == "opengl"
    assert outputs["camera_to_world_frame"] == "opencv"
    assert outputs["world_pose"]["translation_xyz"] == [0.6, -0.3, 1.3]
    assert outputs["world_pose"]["rotation_matrix"] == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def test_camera_pose_to_world_fails_closed_for_malformed_extrinsics() -> None:
    tools = bind_dummy_tool_handlers(build_default_tool_registry())
    result = tools.call(
        "camera_pose_to_world",
        {
            "camera_pose": _candidate(),
            "camera_extrinsics": {"pos": [0.0, 0.0], "mat": [1.0, 0.0, 0.0]},
        },
    )

    assert result.success is False
    assert result.details["outputs"]["reason"] == "invalid_camera_to_world"
