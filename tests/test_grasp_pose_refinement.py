from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from agent.tools.embodied_perception import ObservationFrame
from agent.tools.grasp_pose_refinement import (
    GraspPoseRefinementError,
    backproject_masked_world,
    camera_to_world_opencv,
    derive_mask_side_pose,
    rigid_transform,
)


def _frame(tmp_path: Path, depth: np.ndarray) -> ObservationFrame:
    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.png"
    Image.new("RGB", (depth.shape[1], depth.shape[0]), (20, 30, 40)).save(rgb_path)
    Image.fromarray(depth.astype(np.uint16), mode="I;16").save(depth_path)
    return ObservationFrame(
        artifact_root=tmp_path,
        observation_id="obs-1",
        frame_id="frame-1",
        camera_id="agentview",
        rgb_path=rgb_path,
        depth_path=depth_path,
        intrinsics={"fx": 2.0, "fy": 2.0, "cx": 1.0, "cy": 1.0, "scale": 1000.0},
        extrinsics={
            "pos": [1.0, 2.0, 3.0],
            "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "matrix_layout": "row_major",
            "frame_transform": "camera_to_world",
            "camera_frame": "opencv",
        },
    )


def test_backproject_masked_world_preserves_pixel_correspondence(tmp_path: Path) -> None:
    depth = np.full((3, 3), 1000, dtype=np.uint16)
    frame = _frame(tmp_path, depth)
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 2] = 255
    mask_path = tmp_path / "mask.png"
    Image.fromarray(mask, mode="L").save(mask_path)

    cloud = backproject_masked_world(frame, mask_path)

    np.testing.assert_array_equal(cloud.pixels_uv, [[2, 1]])
    np.testing.assert_allclose(cloud.points_world, [[1.5, 2.0, 4.0]])


def test_opengl_camera_bridge_is_applied_exactly_once() -> None:
    value = camera_to_world_opencv(
        {
            "pos": [0.0, 0.0, 0.0],
            "mat": np.eye(3).reshape(-1).tolist(),
            "matrix_layout": "row_major",
            "frame_transform": "camera_to_world",
            "camera_frame": "opengl",
        }
    )

    np.testing.assert_allclose(value[:3, :3], np.diag([1.0, -1.0, -1.0]))


def test_mask_side_pose_has_explicit_panda_axis_and_offset_semantics(tmp_path: Path) -> None:
    depth = np.asarray(
        [
            [1100, 1100, 1100, 1100, 1100],
            [1000, 1000, 1000, 1000, 1000],
            [1000, 1000, 1000, 1000, 1000],
            [1000, 1000, 1000, 1000, 1000],
            [500, 1000, 1000, 1000, 1000],
        ],
        dtype=np.uint16,
    )
    frame = _frame(tmp_path, depth)
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.full(depth.shape, 255, dtype=np.uint8), mode="L").save(mask_path)

    result = derive_mask_side_pose(
        frame,
        mask_path=mask_path,
        detection_id="detection_000",
        side="y_min",
        insertion_depth_m=0.08,
        surface_floor_quantile=0.05,
        top_quantile=0.99,
    )

    transform = result.world_from_grip_site
    np.testing.assert_allclose(transform[:3, 0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(transform[:3, 2], [0.0, 0.0, -1.0])
    assert np.linalg.det(transform[:3, :3]) == 1.0
    assert result.diagnostics["insertion_depth_m"] == 0.08
    assert result.diagnostics["masked_point_count"] == 25
    assert "surface_floor_z_m" not in result.diagnostics
    assert "not support-plane or collision clearance" in result.diagnostics[
        "mask_low_quantile_semantics"
    ]
    assert transform[2, 3] == result.diagnostics["surface_top_z_m"] - 0.08


def test_mask_side_pose_is_deterministic(tmp_path: Path) -> None:
    depth = np.full((5, 5), 1000, dtype=np.uint16)
    frame = _frame(tmp_path, depth)
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.full(depth.shape, 255, dtype=np.uint8), mode="L").save(mask_path)

    first = derive_mask_side_pose(
        frame,
        mask_path=mask_path,
        detection_id="detection_000",
        side="x_max",
        insertion_depth_m=0.04,
    )
    second = derive_mask_side_pose(
        frame,
        mask_path=mask_path,
        detection_id="detection_000",
        side="x_max",
        insertion_depth_m=0.04,
    )

    assert first.pose_id == second.pose_id
    np.testing.assert_array_equal(first.world_from_grip_site, second.world_from_grip_site)
    assert first.diagnostics == second.diagnostics


def test_rigid_transform_rejects_left_handed_pose() -> None:
    transform = np.eye(4)
    transform[2, 2] = -1.0

    try:
        rigid_transform(transform)
    except GraspPoseRefinementError as exc:
        assert "right-handed" in str(exc)
    else:
        raise AssertionError("Expected left-handed transform rejection")
