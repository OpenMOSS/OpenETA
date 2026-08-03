"""Local point-cloud diagnostics for commanded versus observed EEF poses."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agent.tools.grasp_pose_refinement import camera_to_world_opencv, rigid_transform


TARGET_COLOR = (35, 225, 255)
ACTUAL_COLOR = (255, 205, 35)
ERROR_COLOR = (255, 80, 165)


def backproject_rgbd_world(
    rgb_path: str | Path,
    depth_path: str | Path,
    *,
    intrinsics: Mapping[str, Any],
    extrinsics: Mapping[str, Any],
    depth_truncation_m: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project one retained RGB-D frame to colored world points."""

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    depth = np.asarray(Image.open(depth_path), dtype=np.float64)
    if depth.ndim != 2 or rgb.shape[:2] != depth.shape:
        raise ValueError("RGB and depth dimensions differ")
    try:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        scale = float(intrinsics.get("scale", 1000.0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid RGB-D intrinsics") from exc
    if not all(math.isfinite(value) and value > 0.0 for value in (fx, fy, scale)):
        raise ValueError("invalid RGB-D intrinsics")
    depth_m = depth / scale
    valid = (
        np.isfinite(depth_m)
        & (depth_m > 0.0)
        & (depth_m < float(depth_truncation_m))
    )
    rows, columns = np.where(valid)
    if len(rows) == 0:
        raise ValueError("RGB-D frame has no valid depth samples")
    z = depth_m[rows, columns]
    camera_points = np.stack(
        [
            (columns.astype(np.float64) - cx) * z / fx,
            (rows.astype(np.float64) - cy) * z / fy,
            z,
        ],
        axis=1,
    )
    world_from_camera = camera_to_world_opencv(extrinsics)
    points = (
        camera_points @ world_from_camera[:3, :3].T
        + world_from_camera[:3, 3]
    )
    return points, rgb[rows, columns]


def render_execution_point_cloud(
    rgb_path: str | Path,
    depth_path: str | Path,
    *,
    intrinsics: Mapping[str, Any],
    extrinsics: Mapping[str, Any],
    target_world_from_grip_site: Any,
    actual_world_from_grip_site: Any,
    output_path: str | Path,
    jaw_width_m: float = 0.08,
    local_radius_m: float = 0.22,
    max_points: int = 24_000,
) -> tuple[Path, dict[str, Any]]:
    """Render top/front/side/isometric local cloud views with two EEF poses."""

    target = rigid_transform(
        target_world_from_grip_site, name="target_world_from_grip_site"
    )
    actual = rigid_transform(
        actual_world_from_grip_site, name="actual_world_from_grip_site"
    )
    points, colors = backproject_rgbd_world(
        rgb_path,
        depth_path,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )
    center = (target[:3, 3] + actual[:3, 3]) / 2.0
    distances = np.linalg.norm(points - center, axis=1)
    effective_radius_m = float(local_radius_m)
    local = distances <= effective_radius_m
    if int(np.count_nonzero(local)) < 64:
        # Keep diagnostics available even when a malformed or very distant
        # target sits outside the normal contact-scale crop. Expanding the
        # context makes that mismatch visible instead of silently falling back
        # to the noisy RGB overlay.
        effective_radius_m = max(
            effective_radius_m,
            min(0.60, float(np.quantile(distances, 0.10)) + 0.03),
        )
        local = distances <= effective_radius_m
    points = points[local]
    colors = colors[local]
    if len(points) == 0:
        raise ValueError("No point-cloud samples near TARGET/ACTUAL")
    if len(points) > max_points:
        rng = np.random.default_rng(17)
        indices = np.sort(rng.choice(len(points), max_points, replace=False))
        points = points[indices]
        colors = colors[indices]

    error_xyz = actual[:3, 3] - target[:3, 3]
    error_m = float(np.linalg.norm(error_xyz))
    panel_size = 390
    header_height = 72
    canvas = Image.new("RGB", (panel_size * 2, header_height + panel_size * 2), (17, 19, 24))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((12, 10), "LOCAL RGB-D POINT CLOUD EXECUTION CHECK", fill=(245, 245, 245), font=font)
    draw.text(
        (12, 30),
        (
            f"TARGET cyan   ACTUAL yellow   error {error_m * 1000.0:.1f} mm   "
            f"dxyz [{error_xyz[0] * 1000.0:+.1f}, {error_xyz[1] * 1000.0:+.1f}, {error_xyz[2] * 1000.0:+.1f}] mm"
        ),
        fill=(205, 210, 220),
        font=font,
    )
    draw.text(
        (12, 49),
        "judge contact depth/table clearance here; use RGB only as semantic context",
        fill=(150, 160, 175),
        font=font,
    )

    z_axis = np.asarray([0.0, 0.0, 1.0])
    iso_view = _unit(np.asarray([1.0, -1.0, 0.85]))
    iso_u = _unit(np.cross(z_axis, iso_view))
    iso_v = _unit(np.cross(iso_view, iso_u))
    views = [
        ("TOP  XY", np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0]), np.asarray([0.0, 0.0, 1.0])),
        ("FRONT  XZ", np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 1.0]), np.asarray([0.0, -1.0, 0.0])),
        ("SIDE  YZ", np.asarray([0.0, 1.0, 0.0]), np.asarray([0.0, 0.0, 1.0]), np.asarray([1.0, 0.0, 0.0])),
        ("ISO", iso_u, iso_v, iso_view),
    ]
    for index, (title, axis_u, axis_v, view_dir) in enumerate(views):
        x0 = (index % 2) * panel_size
        y0 = header_height + (index // 2) * panel_size
        _render_panel(
            canvas,
            origin=(x0, y0),
            size=panel_size,
            title=title,
            points=points,
            colors=colors,
            center=center,
            axis_u=axis_u,
            axis_v=axis_v,
            view_dir=view_dir,
            target=target,
            actual=actual,
            jaw_width_m=float(jaw_width_m),
            error_m=error_m,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG")
    diagnostics = {
        "point_count": int(len(points)),
        "requested_local_radius_m": float(local_radius_m),
        "local_radius_m": effective_radius_m,
        "position_error_m": error_m,
        "position_error_xyz_m": error_xyz.tolist(),
        "target_world_from_grip_site": target.tolist(),
        "actual_world_from_grip_site": actual.tolist(),
    }
    return output, diagnostics


def _render_panel(
    canvas: Image.Image,
    *,
    origin: tuple[int, int],
    size: int,
    title: str,
    points: np.ndarray,
    colors: np.ndarray,
    center: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    view_dir: np.ndarray,
    target: np.ndarray,
    actual: np.ndarray,
    jaw_width_m: float,
    error_m: float,
) -> None:
    x0, y0 = origin
    panel = Image.new("RGB", (size, size), (24, 27, 33))
    draw = ImageDraw.Draw(panel)
    relative = points - center
    projected_u = relative @ axis_u
    projected_v = relative @ axis_v
    pose_points = np.stack([target[:3, 3], actual[:3, 3]]) - center
    all_u = np.concatenate([projected_u, pose_points @ axis_u])
    all_v = np.concatenate([projected_v, pose_points @ axis_v])
    span = max(
        0.08,
        float(np.quantile(np.abs(all_u), 0.995)),
        float(np.quantile(np.abs(all_v), 0.995)),
        jaw_width_m,
        error_m * 1.5,
    )
    scale = (size * 0.43) / span

    depth = relative @ view_dir
    order = np.argsort(depth)[::2]
    point_u = (size / 2.0 + projected_u[order] * scale).astype(np.int32)
    point_v = (size / 2.0 - projected_v[order] * scale).astype(np.int32)
    point_colors = colors[order]
    valid = (point_u >= 1) & (point_u < size - 1) & (point_v >= 18) & (point_v < size - 1)
    pixels = panel.load()
    for u, v, color in zip(point_u[valid], point_v[valid], point_colors[valid]):
        muted = tuple(int(35 + 0.72 * int(channel)) for channel in color)
        pixels[int(u), int(v)] = muted

    def project(value: np.ndarray) -> tuple[float, float]:
        delta = value - center
        return (
            size / 2.0 + float(delta @ axis_u) * scale,
            size / 2.0 - float(delta @ axis_v) * scale,
        )

    target_px = project(target[:3, 3])
    actual_px = project(actual[:3, 3])
    draw.line((*target_px, *actual_px), fill=ERROR_COLOR, width=3)
    _draw_gripper(draw, project, target, jaw_width_m, TARGET_COLOR, "TARGET")
    _draw_gripper(draw, project, actual, jaw_width_m, ACTUAL_COLOR, "ACTUAL")
    draw.rectangle((0, 0, size - 1, size - 1), outline=(65, 70, 82), width=1)
    draw.rectangle((6, 5, 92, 20), fill=(15, 17, 21), outline=(80, 86, 100))
    draw.text((10, 8), title, fill=(240, 240, 245))
    canvas.paste(panel, (x0, y0))


def _draw_gripper(
    draw: ImageDraw.ImageDraw,
    project: Any,
    transform: np.ndarray,
    jaw_width_m: float,
    color: tuple[int, int, int],
    label: str,
) -> None:
    center = transform[:3, 3]
    closing = transform[:3, 0]
    approach = transform[:3, 2]
    jaw_a = center + closing * jaw_width_m / 2.0
    jaw_b = center - closing * jaw_width_m / 2.0
    approach_tip = center + approach * 0.065
    c = project(center)
    a = project(jaw_a)
    b = project(jaw_b)
    tip = project(approach_tip)
    draw.line((*a, *b), fill=color, width=5)
    draw.line((*c, *tip), fill=color, width=4)
    draw.ellipse((c[0] - 6, c[1] - 6, c[0] + 6, c[1] + 6), outline=color, width=3)
    draw.text((c[0] + 8, c[1] - 14), label, fill=color, stroke_width=2, stroke_fill=(10, 12, 15))


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("zero-length view vector")
    return value / norm


__all__ = ["backproject_rgbd_world", "render_execution_point_cloud"]
