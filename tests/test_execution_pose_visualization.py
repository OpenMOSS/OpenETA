from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from agent.tools.execution_pose_visualization import (
    backproject_rgbd_world,
    render_execution_point_cloud,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.png"
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[..., 0] = np.linspace(30, 220, 64, dtype=np.uint8)
    rgb[..., 1] = 120
    rgb[..., 2] = 180
    Image.fromarray(rgb).save(rgb_path)
    Image.fromarray(np.full((48, 64), 1000, dtype=np.uint16)).save(depth_path)
    intrinsics = {"fx": 60.0, "fy": 60.0, "cx": 32.0, "cy": 24.0, "scale": 1000.0}
    extrinsics = {
        "pos": [0.0, 0.0, 0.0],
        "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "frame_transform": "camera_to_world",
        "camera_frame": "opencv",
    }
    return rgb_path, depth_path, intrinsics, extrinsics


def test_backproject_rgbd_world_preserves_colored_samples(tmp_path: Path) -> None:
    rgb, depth, intrinsics, extrinsics = _fixture(tmp_path)

    points, colors = backproject_rgbd_world(
        rgb, depth, intrinsics=intrinsics, extrinsics=extrinsics
    )

    assert points.shape == (48 * 64, 3)
    assert colors.shape == (48 * 64, 3)
    np.testing.assert_allclose(points[:, 2], 1.0)


def test_execution_point_cloud_renders_four_view_distance_board(
    tmp_path: Path,
) -> None:
    rgb, depth, intrinsics, extrinsics = _fixture(tmp_path)
    target = np.eye(4)
    target[:3, 3] = [0.0, 0.0, 1.0]
    actual = np.eye(4)
    actual[:3, 3] = [0.02, -0.01, 1.01]

    output, diagnostics = render_execution_point_cloud(
        rgb,
        depth,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        target_world_from_grip_site=target,
        actual_world_from_grip_site=actual,
        output_path=tmp_path / "comparison.png",
        local_radius_m=0.4,
    )

    assert output.is_file()
    assert Image.open(output).size == (780, 852)
    assert diagnostics["point_count"] > 0
    np.testing.assert_allclose(
        diagnostics["position_error_xyz_m"], [0.02, -0.01, 0.01]
    )
    assert diagnostics["position_error_m"] > 0.024
