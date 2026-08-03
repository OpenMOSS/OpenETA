"""Deterministic LIBERO point-cloud views and model-authored pose marks.

The model owns the geometric decision.  This module only renders calibrated
orthographic views, maps view pixels back to world coordinates, and converts
three explicit marks into a right-handed Panda grip-site frame.
"""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agent.tools.grasp_pose_refinement import camera_to_world_opencv


VIEW_SPECS: dict[str, dict[str, Any]] = {
    "pointcloud_top": {
        "horizontal_axis": "x",
        "vertical_axis": "y",
        "hidden_axis": "z",
        "horizontal_range_m": [-0.45, 0.35],
        "vertical_range_m": [-0.35, 0.50],
        "visibility": "highest_z",
    },
    "pointcloud_front": {
        "horizontal_axis": "x",
        "vertical_axis": "z",
        "hidden_axis": "y",
        "horizontal_range_m": [-0.45, 0.35],
        "vertical_range_m": [0.85, 1.35],
        "visibility": "nearest_negative_y",
    },
    "pointcloud_side": {
        "horizontal_axis": "y",
        "vertical_axis": "z",
        "hidden_axis": "x",
        "horizontal_range_m": [-0.35, 0.50],
        "vertical_range_m": [0.85, 1.35],
        "visibility": "nearest_negative_x",
    },
}

# The original LIBERO tabletop scene uses this calibrated world-frame box.
# Keep it as the default for existing callers and serialized view semantics.
# Object-suite tasks use a different MuJoCo scene scale, so callers that are
# rendering a live cloud may supply scene-derived bounds instead.
LIBERO_WORKSPACE_BOUNDS = np.asarray(
    [[-0.45, 0.35], [-0.35, 0.50], [0.85, 1.35]],
    dtype=np.float64,
)
# A task-agnostic LIBERO robot workspace for scenes whose origin is near the
# tabletop (the object-suite uses z≈0 while the spatial-suite uses z≈1).
# This is a rendering envelope, not an object detector or privileged pose
# source.  It prevents distant walls/background depth from determining the
# orthographic view scale.
LIBERO_OBJECT_WORKSPACE_BOUNDS = np.asarray(
    [[-0.60, 0.60], [-0.60, 0.70], [-0.15, 0.60]],
    dtype=np.float64,
)

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
AXIS_COLORS = {
    "x": (255, 80, 70),
    "y": (80, 220, 100),
    "z": (80, 140, 255),
}
# Shared-axis agreement tolerance for two-view intersections, matching the
# 15 mm convention already used by merge_world_vector_projections.
CONSISTENCY_TOLERANCE_M = 0.015
RAY_CLICK_TOLERANCE_M = 0.02
# An Agentview click is a semantic binding to the aligned RGB-D visible
# feature, not an unconstrained request for any point along the same camera
# ray.  The complementary orthographic click confirms that visible depth
# layer.  Hidden/free-space points remain expressible with a new point_id and
# two ordinary point-cloud views.
VISIBLE_SURFACE_LAYER_TOLERANCE_M = 0.02
VISIBLE_SURFACE_MARKER_COLOR = (80, 255, 255)
POINT_COLORS = (
    (255, 80, 70),
    (80, 210, 255),
    (110, 255, 120),
    (255, 210, 70),
    (210, 120, 255),
    (255, 150, 80),
)


@dataclass(frozen=True, slots=True)
class PointCloudView:
    name: str
    image_path: Path
    lookup_path: Path
    metadata_path: Path
    width: int
    height: int
    spec: dict[str, Any]
    clean_image_path: Path | None = None

    def public_dict(self, root: Path, *, absolute_paths: bool = False) -> dict[str, Any]:
        def ref(path: Path) -> str:
            if absolute_paths:
                return str(path.resolve())
            try:
                return str(path.relative_to(root))
            except ValueError:
                return str(path)

        return {
            "view": self.name,
            "image_path": ref(self.image_path),
            "clean_image_path": (
                ref(self.clean_image_path)
                if self.clean_image_path is not None
                else ref(self.image_path)
            ),
            "lookup_path": ref(self.lookup_path),
            "metadata_path": ref(self.metadata_path),
            "width": self.width,
            "height": self.height,
            **self.spec,
        }


def render_observation_pointcloud_views(
    observation_record: Mapping[str, Any],
    *,
    artifact_root: str | Path,
    output_root: str | Path,
    image_size: int = 384,
    camera_ids: tuple[str, ...] = ("agentview",),
    metric_edge_ticks: bool = False,
    metric_tick_band_style: str | None = None,
) -> dict[str, PointCloudView]:
    """Back-project retained RGB-D frames and render calibrated views.

    ``camera_ids`` defaults to the historical single ``agentview`` path.  A
    A host-side LIBERO probe can pass several virtual free-camera frame ids;
    all frames are then fused in the same world coordinate system before the
    orthographic views are drawn.  The Operator-facing API remains unchanged.
    """
    root = Path(artifact_root)
    world_points, colors = world_pointcloud_from_record(
        observation_record, artifact_root=root, camera_ids=camera_ids
    )
    return render_world_pointcloud_views(
        world_points,
        colors,
        observation_record=observation_record,
        artifact_root=artifact_root,
        output_root=output_root,
        image_size=image_size,
        metric_edge_ticks=metric_edge_ticks,
        metric_tick_band_style=metric_tick_band_style,
    )


def render_world_pointcloud_views(
    world_points: np.ndarray,
    colors: np.ndarray,
    *,
    observation_record: Mapping[str, Any],
    artifact_root: str | Path,
    output_root: str | Path,
    image_size: int = 384,
    grip_site_xyz: list[float] | None = None,
    grip_site_rotation: list[list[float]] | None = None,
    grip_site_aperture_m: float | None = None,
    finger_pad_contact_centers_world_m: list[list[float]] | None = None,
    target_grip_site_xyz: list[float] | None = None,
    metric_edge_ticks: bool = False,
    metric_tick_band_style: str | None = None,
    compact_grip_site_overlay: bool = False,
    world_bounds: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> dict[str, PointCloudView]:
    """Render calibrated projections from an already-fused world cloud.

    ``world_bounds`` is optional by design.  Omitting it preserves the
    historical LIBERO spatial box; supplying it lets a live scene with a
    different coordinate scale (for example LIBERO object tasks around
    ``z=0``) retain and display its actual geometry without changing the
    Operator-facing contract.
    """

    bounds = _normalize_world_bounds(
        LIBERO_WORKSPACE_BOUNDS if world_bounds is None else world_bounds
    )
    # Keep the calibrated tabletop workspace by default.  A caller may pass
    # scene-derived bounds for a task whose world origin/scale differs.
    workspace = np.all(
        (world_points >= bounds[:, 0]) & (world_points <= bounds[:, 1]),
        axis=1,
    )
    world_points = world_points[workspace]
    colors = colors[workspace]
    if len(world_points) == 0:
        raise ValueError("world point-cloud bounds contain no valid points")

    output = Path(output_root) / str(observation_record.get("observation_id"))
    output.mkdir(parents=True, exist_ok=True)
    if grip_site_xyz is None or grip_site_rotation is None:
        observed_xyz, observed_rotation = _observation_grip_site_pose(
            observation_record
        )
        if grip_site_xyz is None:
            grip_site_xyz = observed_xyz
        if grip_site_rotation is None:
            grip_site_rotation = observed_rotation
        if grip_site_aperture_m is None:
            grip_site_aperture_m = _observation_gripper_aperture_m(
                observation_record
            )
    if finger_pad_contact_centers_world_m is None:
        finger_pad_contact_centers_world_m = (
            _observation_finger_pad_contact_centers_world_m(observation_record)
        )
    views: dict[str, PointCloudView] = {}
    view_specs = _view_specs_for_world_bounds(bounds)
    for name, raw_spec in view_specs.items():
        spec = dict(raw_spec)
        spec["world_frame_directions"] = {
            "screen_right": f"+{str(spec['horizontal_axis']).upper()}",
            "screen_up": f"+{str(spec['vertical_axis']).upper()}",
        }
        spec["axis_colors"] = {
            axis.upper(): list(color) for axis, color in AXIS_COLORS.items()
        }
        spec["grid_interval_m"] = 0.1
        if grip_site_xyz is not None:
            spec["actual_grip_site_xyz_m"] = grip_site_xyz
        if grip_site_rotation is not None:
            rotation = np.asarray(grip_site_rotation, dtype=np.float64)
            spec["actual_grip_site_jaw_axis_world"] = rotation[:, 0].tolist()
            spec["actual_grip_site_approach_axis_world"] = rotation[:, 2].tolist()
            if finger_pad_contact_centers_world_m is not None:
                spec["actual_finger_pad_contact_centers_world_m"] = (
                    finger_pad_contact_centers_world_m
                )
            if isinstance(grip_site_aperture_m, (int, float)) and math.isfinite(
                float(grip_site_aperture_m)
            ):
                spec["actual_gripper_aperture_m"] = float(
                    grip_site_aperture_m
                )
        if target_grip_site_xyz is not None:
            spec["commanded_grip_site_xyz_m"] = target_grip_site_xyz
        if compact_grip_site_overlay:
            spec["current_grip_site_overlay"] = "micro_marker_v1"
        xyz, clean_canvas = _orthographic_view(
            world_points,
            colors,
            spec=spec,
            size=int(image_size),
            metric_edge_ticks=metric_edge_ticks,
            metric_tick_band_style=metric_tick_band_style,
        )
        _overlay_xyz, canvas = _orthographic_view(
            world_points,
            colors,
            spec=spec,
            size=int(image_size),
            grip_site_xyz=grip_site_xyz,
            grip_site_rotation=grip_site_rotation,
            grip_site_aperture_m=grip_site_aperture_m,
            finger_pad_contact_centers_world_m=(
                finger_pad_contact_centers_world_m
            ),
            target_grip_site_xyz=target_grip_site_xyz,
            metric_edge_ticks=metric_edge_ticks,
            metric_tick_band_style=metric_tick_band_style,
            compact_grip_site_overlay=compact_grip_site_overlay,
        )
        image_path = output / f"{name}.png"
        clean_image_path = output / f"{name}.clean.png"
        lookup_path = output / f"{name}.world_xyz.npz"
        metadata_path = output / f"{name}.json"
        Image.fromarray(canvas, mode="RGB").save(image_path)
        Image.fromarray(clean_canvas, mode="RGB").save(clean_image_path)
        np.savez_compressed(
            lookup_path,
            xyz_m=xyz.astype(np.float32),
            valid=np.isfinite(xyz[..., 0]),
        )
        metadata = {
            "schema_version": "openeta.pointcloud_view.v1",
            "observation_id": observation_record.get("observation_id"),
            "camera_id": "agentview",
            "frame": "world",
            "world_bounds_m": bounds.tolist(),
            "view": name,
            "width": int(image_size),
            "height": int(image_size),
            **spec,
            "pixel_coordinates": "u right, v down, zero-based",
            "point_mark_rule": (
                "Every point requires clicks in two complementary orthographic "
                "views. Each click fixes its two displayed world axes and the "
                "pair intersects into one world-frame 3D point. The rendered "
                "cloud is visualization only and never validates or changes it."
            ),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        views[name] = PointCloudView(
            name=name,
            image_path=image_path,
            lookup_path=lookup_path,
            metadata_path=metadata_path,
            width=int(image_size),
            height=int(image_size),
            spec=spec,
            clean_image_path=clean_image_path,
        )
    return views


def world_pointcloud_from_record(
    observation_record: Mapping[str, Any],
    *,
    artifact_root: str | Path,
    camera_ids: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse one or more retained RGB-D frames into world-frame points.

    This is deliberately ordinary RGB-D back-projection.  It does not read
    simulator object poses, segmentation, meshes, or privileged state.  Each
    frame must carry its own intrinsics and camera-to-world extrinsics.
    """

    root = Path(artifact_root)
    requested = set(camera_ids) if camera_ids is not None else None
    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    raw_frames = observation_record.get("frames", [])
    if not isinstance(raw_frames, list):
        raise ValueError("observation has no retained camera frames")
    for raw in raw_frames:
        if not isinstance(raw, Mapping):
            continue
        camera_id = str(raw.get("camera_id") or "")
        if requested is not None and camera_id not in requested:
            continue
        rgb_raw = raw.get("rgb_path")
        depth_raw = raw.get("depth_path")
        if not rgb_raw or not depth_raw:
            continue
        rgb_path = Path(rgb_raw)
        depth_path = Path(depth_raw)
        if not rgb_path.is_absolute():
            rgb_path = root / rgb_path
        if not depth_path.is_absolute():
            depth_path = root / depth_path
        metadata = raw.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        intrinsics = metadata.get("anygrasp_intrinsics") or metadata.get("intrinsics")
        extrinsics = metadata.get("extrinsics")
        if not isinstance(intrinsics, Mapping) or not isinstance(extrinsics, Mapping):
            continue
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        depth = np.asarray(Image.open(depth_path), dtype=np.float64)
        scale = float(intrinsics.get("scale", 1000.0))
        if scale <= 0.0 or depth.ndim != 2 or rgb.shape[:2] != depth.shape:
            continue
        depth = depth / scale
        valid = np.isfinite(depth) & (depth > 0.0)
        rows, columns = np.where(valid)
        if len(rows) == 0:
            continue
        z = depth[rows, columns]
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        camera_points = np.stack(
            [(columns - cx) * z / fx, (rows - cy) * z / fy, z], axis=1
        )
        world_from_camera = camera_to_world_opencv(extrinsics)
        points.append(
            camera_points @ world_from_camera[:3, :3].T
            + world_from_camera[:3, 3]
        )
        colors.append(rgb[rows, columns])
    if not points:
        raise ValueError("no valid RGB-D frames for world point-cloud fusion")
    return np.concatenate(points, axis=0), np.concatenate(colors, axis=0)


def workspace_pointcloud(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    bounds: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop an ordinary world cloud to a world-frame workspace.

    The default remains the original LIBERO tabletop box.  Live gateway code
    can pass scene-derived bounds for object-suite scenes.
    """

    normalized = _normalize_world_bounds(
        LIBERO_WORKSPACE_BOUNDS if bounds is None else bounds
    )
    mask = np.all(
        (points >= normalized[:, 0]) & (points <= normalized[:, 1]),
        axis=1,
    )
    return points[mask], colors[mask]


def world_bounds_from_points(
    points: np.ndarray,
    *,
    quantile: float = 0.01,
    padding_m: float = 0.03,
    minimum_extent_m: float = 0.10,
) -> np.ndarray:
    """Derive conservative scene bounds from ordinary RGB-D points.

    This is deliberately geometry-only: it uses finite point coordinates,
    never simulator object identities or privileged poses.  Small quantile
    trimming removes a few distant depth outliers, while the minimum extent
    keeps thin scenes markable in all three projections.
    """

    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    finite = array[np.isfinite(array).all(axis=1)]
    if len(finite) == 0:
        raise ValueError("cannot derive world bounds from empty points")
    q = min(max(float(quantile), 0.0), 0.25)
    lower = np.quantile(finite, q, axis=0)
    upper = np.quantile(finite, 1.0 - q, axis=0)
    pad = max(0.0, float(padding_m))
    extent = np.maximum(upper - lower, float(minimum_extent_m))
    center = (lower + upper) / 2.0
    half = extent / 2.0 + pad
    return np.column_stack((center - half, center + half))


def operator_scene_bounds(points: np.ndarray) -> np.ndarray:
    """Choose stable task-agnostic view bounds from observable geometry."""

    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    finite = array[np.isfinite(array).all(axis=1)]
    if len(finite) == 0:
        raise ValueError("cannot choose scene bounds from empty points")
    object_mask = np.all(
        (finite >= LIBERO_OBJECT_WORKSPACE_BOUNDS[:, 0])
        & (finite <= LIBERO_OBJECT_WORKSPACE_BOUNDS[:, 1]),
        axis=1,
    )
    object_points = finite[object_mask]
    # A spatial task has a dense tabletop around z≈1.  Object-suite scenes
    # have their task geometry near z≈0; use the broad object workspace there.
    if len(object_points) and float(np.median(object_points[:, 2])) < 0.60:
        return LIBERO_OBJECT_WORKSPACE_BOUNDS.copy()
    return LIBERO_WORKSPACE_BOUNDS.copy()


def world_bounds_including_points(
    bounds: np.ndarray | Sequence[Sequence[float]],
    points: np.ndarray | Sequence[Sequence[float]],
    *,
    padding_m: float = 0.10,
) -> np.ndarray:
    """Expand rendered bounds to include finite workspace anchors."""

    normalized = _normalize_world_bounds(bounds).copy()
    anchors = np.asarray(points, dtype=np.float64)
    if anchors.ndim == 1:
        anchors = anchors.reshape(1, -1)
    if anchors.ndim != 2 or anchors.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    finite = anchors[np.isfinite(anchors).all(axis=1)]
    if len(finite) == 0:
        return normalized
    padding = max(0.0, float(padding_m))
    normalized[:, 0] = np.minimum(
        normalized[:, 0],
        np.min(finite, axis=0) - padding,
    )
    normalized[:, 1] = np.maximum(
        normalized[:, 1],
        np.max(finite, axis=0) + padding,
    )
    return normalized


def operator_scene_lookat(
    points: np.ndarray,
    *,
    bounds: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    """Choose a neutral camera target from observable scene geometry."""

    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    finite = array[np.isfinite(array).all(axis=1)]
    if len(finite) == 0:
        raise ValueError("cannot choose scene look-at from empty points")
    normalized = _normalize_world_bounds(
        operator_scene_bounds(finite) if bounds is None else bounds
    )
    inside = np.all(
        (finite >= normalized[:, 0]) & (finite <= normalized[:, 1]),
        axis=1,
    )
    bounded = finite[inside]
    if len(bounded) == 0:
        bounded = finite
    target = np.array(
        [
            float(np.median(bounded[:, 0])),
            float(np.median(bounded[:, 1])),
            float(np.quantile(bounded[:, 2], 0.65)),
        ],
        dtype=np.float64,
    )
    return np.clip(target, normalized[:, 0], normalized[:, 1])


def _normalize_world_bounds(
    bounds: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    normalized = np.asarray(bounds, dtype=np.float64)
    if normalized.shape != (3, 2) or not np.isfinite(normalized).all():
        raise ValueError("world bounds must have shape (3, 2) and be finite")
    if np.any(normalized[:, 1] <= normalized[:, 0]):
        raise ValueError("world bounds must have positive extents")
    return normalized


def _view_specs_for_world_bounds(bounds: np.ndarray) -> dict[str, dict[str, Any]]:
    """Build orthographic ranges from a scene box, preserving view axes."""

    normalized = _normalize_world_bounds(bounds)
    result: dict[str, dict[str, Any]] = {}
    for name, raw in VIEW_SPECS.items():
        spec = dict(raw)
        h = AXIS_INDEX[str(spec["horizontal_axis"])]
        v = AXIS_INDEX[str(spec["vertical_axis"])]
        spec["horizontal_range_m"] = normalized[h].tolist()
        spec["vertical_range_m"] = normalized[v].tolist()
        result[name] = spec
    return result


def voxel_fuse_pointcloud(
    points: np.ndarray, colors: np.ndarray, *, voxel_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse repeated samples in each voxel without inventing surfaces.

    The consensus stage has already removed unsupported depth samples. A
    centroid is sufficient for the remaining samples inside one small voxel
    and can be computed in one vectorized pass. The historical per-voxel
    median loop scanned the entire point array once per occupied voxel, making
    a normal 600k-point / 30k-voxel LIBERO cloud effectively O(N * V).
    """

    if len(points) == 0:
        return points, colors
    voxel = float(voxel_m)
    if not np.isfinite(voxel) or voxel <= 0.0:
        raise ValueError("voxel_m must be a positive finite number")
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if colors.shape != points.shape:
        raise ValueError("colors must have the same shape as points")

    keys = np.floor(points / voxel).astype(np.int64)
    _unique, inverse, counts = np.unique(
        keys,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    fused_points = np.column_stack(
        [
            np.bincount(inverse, weights=points[:, axis], minlength=len(counts))
            for axis in range(3)
        ]
    )
    fused_colors = np.column_stack(
        [
            np.bincount(inverse, weights=colors[:, axis], minlength=len(counts))
            for axis in range(3)
        ]
    )
    fused_points /= counts[:, None]
    fused_colors /= counts[:, None]
    return fused_points, np.rint(fused_colors).clip(0, 255).astype(np.uint8)


def multiview_consensus_pointcloud(
    points: np.ndarray,
    colors: np.ndarray,
    observation_record: Mapping[str, Any],
    *,
    artifact_root: str | Path,
    min_support: int = 2,
    tolerance_m: float = 0.015,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Keep surfaces whose metric depth is reproduced by another view."""

    root = Path(artifact_root)
    frames = observation_record.get("frames", [])
    if not isinstance(frames, list) or len(frames) < 2:
        return points, colors, {
            "min_support": int(min_support),
            "input_points": int(len(points)),
            "supported_points": int(len(points)),
        }
    images: list[tuple[Mapping[str, Any], np.ndarray]] = []
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        depth_path = Path(str(frame.get("depth_path", "")))
        if not depth_path.is_absolute():
            depth_path = root / depth_path
        if depth_path.is_file():
            images.append(
                (frame, np.asarray(Image.open(depth_path), dtype=np.float64) / 1000.0)
            )
    support = np.zeros(len(points), dtype=np.int16)
    for frame, depth in images:
        metadata = frame.get("metadata", {})
        metadata = metadata if isinstance(metadata, Mapping) else {}
        intrinsics = metadata.get("intrinsics", {})
        extrinsics = metadata.get("extrinsics", {})
        if not isinstance(intrinsics, Mapping) or not isinstance(extrinsics, Mapping):
            continue
        try:
            fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
            cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
            rotation = np.asarray(extrinsics["mat"], dtype=np.float64).reshape(3, 3)
            translation = np.asarray(extrinsics["pos"], dtype=np.float64).reshape(3)
        except (KeyError, TypeError, ValueError):
            continue
        camera = (points - translation) @ rotation
        z = camera[:, 2]
        u = np.rint(camera[:, 0] / np.maximum(z, 1e-9) * fx + cx).astype(np.int64)
        v = np.rint(camera[:, 1] / np.maximum(z, 1e-9) * fy + cy).astype(np.int64)
        inside = (
            (z > 0)
            & (u >= 0)
            & (u < depth.shape[1])
            & (v >= 0)
            & (v < depth.shape[0])
        )
        observed = np.zeros(len(points), dtype=np.float64)
        observed[inside] = depth[v[inside], u[inside]]
        same_surface = (
            inside
            & (observed > 0)
            & (np.abs(observed - z) <= np.maximum(float(tolerance_m), 0.02 * z))
        )
        support += same_surface.astype(np.int16)
    keep = support >= int(min_support)
    return points[keep], colors[keep], {
        "min_support": int(min_support),
        "tolerance_m": float(tolerance_m),
        "input_points": int(len(points)),
        "supported_points": int(keep.sum()),
    }


def pointcloud_coverage(points: np.ndarray, *, voxel_m: float = 0.005) -> int:
    """Return occupied voxel count for point-cloud quality diagnostics."""

    if len(points) == 0:
        return 0
    return int(
        np.unique(np.floor(points / float(voxel_m)).astype(np.int64), axis=0).shape[0]
    )


def render_pointcloud_contact_sheet(
    views: Mapping[str, PointCloudView], *, output_root: str | Path
) -> Path | None:
    """Combine clean calibrated views into one compact Operator-facing image.

    The global metric views are for reading scene geometry. Robot pose,
    TARGET/ACTUAL, and pad-footprint overlays belong in execution records and
    point-specific zooms; placing them over the scene can hide an object or be
    mistaken for object geometry after the robot has moved.
    """

    ordered = [views[name] for name in ("pointcloud_top", "pointcloud_front", "pointcloud_side") if name in views]
    if not ordered:
        raise ValueError("no point-cloud views available")
    images = [
        Image.open(
            view.clean_image_path
            if view.clean_image_path is not None
            and view.clean_image_path.is_file()
            else view.image_path
        ).convert("RGB")
        for view in ordered
    ]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    sheet = Image.new("RGB", (width * 2, height * 2), (0, 0, 0))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % 2) * width, (index // 2) * height))
    path = Path(output_root) / "pointcloud_contact_sheet.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def mark_world_point(
    view: PointCloudView,
    *,
    u: int,
    v: int,
    pending_constraint: Mapping[str, Any] | None = None,
    enforce_visible_surface_layer: bool = False,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Resolve explicit view clicks into one world point.

    The first click records a two-axis constraint. The second click supplies
    the missing axis and must agree on the shared axis.  For an Agentview
    semantic ray with aligned RGB-D, the complementary click must also select
    the same visible depth layer.  Hidden or free-space points use a new
    point_id with two ordinary point-cloud views.
    """

    u = max(0, min(view.width - 1, int(u)))
    v = max(0, min(view.height - 1, int(v)))
    click = _world_axes_from_view_click(view, u=u, v=v)
    if pending_constraint is None:
        return None, {
            "status": "pending",
            "source_kind": "orthographic_plane_coordinate",
            "requested_pixel_xy": [u, v],
            "pending_constraint": click,
            "needs_complementary_view": _complementary_views(view.name),
        }
    if str(pending_constraint.get("source_kind") or "") == "agentview_camera_ray":
        shortcut_mode = str(pending_constraint.get("shortcut_mode") or "")
        visible_surface = pending_constraint.get("visible_surface")
        if (
            shortcut_mode == "agentview_first_visible_surface"
            and isinstance(visible_surface, Mapping)
            and isinstance(visible_surface.get("xyz_m"), (list, tuple))
            and len(visible_surface["xyz_m"]) == 3
        ):
            # The complementary point-cloud click is only the UI commit
            # gesture in this profile.  The authoritative point is the
            # calibrated RGB-D surface at the original Agentview pixel, not
            # the quantized orthographic projection.
            xyz = np.asarray(visible_surface["xyz_m"], dtype=np.float64)
            if not np.isfinite(xyz).all():
                return None, {
                    "status": "invalid_visible_surface",
                    "source_kind": "agentview_first_visible_surface",
                    "requested_pixel_xy": [u, v],
                    "pending_constraint": pending_constraint,
                }
            return xyz, {
                "status": "solved",
                "source_kind": "agentview_first_visible_surface",
                "requested_pixel_xy": [u, v],
                "shared_axis": "camera_ray",
                "shared_axis_residual_m": 0.0,
                "consistency_tolerance_m": 0.0,
                "view_contributions_m": {
                    str(pending_constraint.get("view") or "agentview"): {
                        "surface_xyz_m": xyz.tolist(),
                        "pixel_xy": list(
                            pending_constraint.get("pixel_xy") or []
                        ),
                    },
                    str(click["view"]): {
                        axis: float(value)
                        for axis, value in click["components_m"].items()
                    },
                },
                "uncertainty_m": {
                    axis: float(click["axis_pixel_size_m"][axis])
                    for axis in click["fixed_axes"]
                },
                "ray_parameter_m": float(
                    visible_surface.get("ray_parameter_m") or 0.0
                ),
                "pending_constraint": None,
                "visible_surface": dict(visible_surface),
                "visible_surface_delta_m": 0.0,
                # Host-side provenance for optional replay/rendering.  The
                # compact public mark_point result remains only point_id +
                # xyz_m, while a renderer experiment can preserve the
                # calibrated line of sight that produced this surface point.
                "agentview_source_ray": {
                    "camera_id": str(
                        pending_constraint.get("view") or "agentview"
                    ),
                    "pixel_xy": list(
                        pending_constraint.get("pixel_xy") or []
                    ),
                    "origin_xyz_m": list(
                        pending_constraint.get("origin_xyz_m") or []
                    ),
                    "direction_xyz": list(
                        pending_constraint.get("direction_xyz") or []
                    ),
                },
            }
        solved = solve_camera_ray_constraint(pending_constraint, click)
    else:
        solved = solve_point_constraints(pending_constraint, click)
    status = str(solved["status"])
    if status == "complementary_view_required":
        return None, {
            **solved,
            "source_kind": "orthographic_plane_coordinate",
            "requested_pixel_xy": [u, v],
            "pending_constraint": pending_constraint,
        }
    if status != "solved":
        is_camera_ray = (
            str(pending_constraint.get("source_kind") or "")
            == "agentview_camera_ray"
        )
        visible_surface = pending_constraint.get("visible_surface")
        visible_surface_ray_pixel_xy = (
            _project_click_xyz_to_view_pixel(
                click,
                np.asarray(visible_surface["xyz_m"], dtype=np.float64),
            )
            if is_camera_ray
            and enforce_visible_surface_layer
            and isinstance(visible_surface, Mapping)
            and isinstance(visible_surface.get("xyz_m"), (list, tuple))
            else None
        )
        return None, {
            **solved,
            "source_kind": str(
                pending_constraint.get("source_kind")
                or "orthographic_plane_coordinate"
            ),
            "requested_pixel_xy": [u, v],
            "pending_constraint": (
                pending_constraint
                if is_camera_ray or status == "complementary_view_required"
                else click
            ),
            **(
                {
                    "visible_surface_ray_pixel_xy":
                        visible_surface_ray_pixel_xy
                }
                if visible_surface_ray_pixel_xy is not None
                else {}
            ),
        }
    xyz = np.asarray(solved["xyz_m"], dtype=np.float64)
    visible_surface = pending_constraint.get("visible_surface")
    visible_surface_delta_m: float | None = None
    if (
        isinstance(visible_surface, Mapping)
        and isinstance(visible_surface.get("ray_parameter_m"), (int, float))
        and isinstance(solved.get("ray_parameter_m"), (int, float))
    ):
        visible_surface_delta_m = float(solved["ray_parameter_m"]) - float(
            visible_surface["ray_parameter_m"]
        )
        if (
            enforce_visible_surface_layer
            and abs(visible_surface_delta_m)
            > VISIBLE_SURFACE_LAYER_TOLERANCE_M
        ):
            visible_surface_ray_pixel_xy = _project_click_xyz_to_view_pixel(
                click,
                np.asarray(visible_surface["xyz_m"], dtype=np.float64),
            )
            return None, {
                "status": "different_visible_depth_layer",
                "source_kind": "agentview_camera_ray",
                "requested_pixel_xy": [u, v],
                "visible_surface": dict(visible_surface),
                "visible_surface_delta_m": visible_surface_delta_m,
                "visible_surface_layer_tolerance_m":
                    VISIBLE_SURFACE_LAYER_TOLERANCE_M,
                **(
                    {
                        "visible_surface_ray_pixel_xy":
                            visible_surface_ray_pixel_xy
                    }
                    if visible_surface_ray_pixel_xy is not None
                    else {}
                ),
                "pending_constraint": pending_constraint,
            }
    return xyz, {
        "status": "solved",
        "source_kind": str(solved.get("source_kind") or "two_view_intersection"),
        "requested_pixel_xy": [u, v],
        "shared_axis": solved["shared_axis"],
        "shared_axis_residual_m": solved["shared_axis_residual_m"],
        "consistency_tolerance_m": solved["consistency_tolerance_m"],
        "view_contributions_m": solved["view_contributions_m"],
        "uncertainty_m": solved["uncertainty_m"],
        **(
            {"ray_parameter_m": solved["ray_parameter_m"]}
            if solved.get("ray_parameter_m") is not None
            else {}
        ),
        **(
            {
                "visible_surface": dict(visible_surface),
                "visible_surface_delta_m": visible_surface_delta_m,
            }
            if isinstance(visible_surface, Mapping)
            and visible_surface_delta_m is not None
            else {}
        ),
        "pending_constraint": None,
    }


def camera_ray_from_image_click(
    observation_record: Mapping[str, Any],
    *,
    camera_id: str,
    u: int,
    v: int,
    artifact_root: str | Path | None = None,
    world_bounds: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Convert one calibrated perspective-image click into a world ray.

    When aligned depth is available, the exact clicked RGB-D surface is stored
    as the semantic depth reference.  The ray is still projected into
    orthographic views so the caller can visually confirm the same feature.
    """

    frames = observation_record.get("frames", [])
    frame = next(
        (
            item
            for item in frames
            if isinstance(item, Mapping)
            and str(item.get("camera_id") or "") == str(camera_id)
        ),
        None,
    )
    if not isinstance(frame, Mapping):
        raise ValueError(f"observation has no {camera_id} frame")
    metadata = frame.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    intrinsics = metadata.get("intrinsics")
    extrinsics = metadata.get("extrinsics")
    if not isinstance(intrinsics, Mapping) or not isinstance(extrinsics, Mapping):
        raise ValueError(f"{camera_id} frame has no calibrated camera metadata")
    width = int(
        intrinsics.get("width")
        or frame.get("width")
        or round(float(intrinsics.get("cx") or 0.0) * 2.0)
    )
    height = int(
        intrinsics.get("height")
        or frame.get("height")
        or round(float(intrinsics.get("cy") or 0.0) * 2.0)
    )
    if width <= 0 or height <= 0:
        raise ValueError(f"{camera_id} frame has invalid dimensions")
    u = max(0, min(width - 1, int(u)))
    v = max(0, min(height - 1, int(v)))
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    camera_direction = np.asarray(
        [(u - cx) / fx, (v - cy) / fy, 1.0],
        dtype=np.float64,
    )
    camera_direction /= np.linalg.norm(camera_direction)
    world_from_camera = camera_to_world_opencv(extrinsics)
    world_direction = world_from_camera[:3, :3] @ camera_direction
    world_direction /= np.linalg.norm(world_direction)
    origin = world_from_camera[:3, 3]
    clipped = clip_world_ray_to_workspace(
        origin,
        world_direction,
        bounds=world_bounds,
    )
    result = {
        "view": str(camera_id),
        "source_kind": "agentview_camera_ray",
        "pixel_xy": [u, v],
        "image_size": [width, height],
        "origin_xyz_m": origin.tolist(),
        "direction_xyz": world_direction.tolist(),
        "workspace_t_range_m": (
            [float(clipped[0]), float(clipped[1])] if clipped is not None else None
        ),
        "fixed_axes": [],
        "hidden_axis": None,
    }
    depth_raw = frame.get("depth_path")
    if isinstance(depth_raw, str) and depth_raw:
        depth_path = Path(depth_raw)
        if not depth_path.is_absolute() and artifact_root is not None:
            depth_path = Path(artifact_root) / depth_path
        if depth_path.is_file():
            depth = np.asarray(Image.open(depth_path), dtype=np.float64)
            if depth.ndim == 2 and v < depth.shape[0] and u < depth.shape[1]:
                depth_z_m = float(depth[v, u])
                scale = float(intrinsics.get("scale", 1000.0))
                if scale > 0.0:
                    depth_z_m /= scale
                if math.isfinite(depth_z_m) and depth_z_m > 0.0:
                    camera_point = np.asarray(
                        [
                            (u - cx) * depth_z_m / fx,
                            (v - cy) * depth_z_m / fy,
                            depth_z_m,
                        ],
                        dtype=np.float64,
                    )
                    visible_world = (
                        world_from_camera[:3, :3] @ camera_point
                        + world_from_camera[:3, 3]
                    )
                    result["visible_surface"] = {
                        "source": "aligned_rgbd_exact_pixel",
                        "pixel_xy": [u, v],
                        "xyz_m": visible_world.tolist(),
                        "ray_parameter_m": float(np.linalg.norm(camera_point)),
                        "selection_role":
                            "agentview_visible_feature_depth_reference",
                    }
    return result


def clip_world_ray_to_workspace(
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    *,
    bounds: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> tuple[float, float] | None:
    """Return the forward ray interval inside the rendered world bounds."""

    origin = np.asarray(origin_xyz, dtype=np.float64).reshape(3)
    direction = np.asarray(direction_xyz, dtype=np.float64).reshape(3)
    bounds = _normalize_world_bounds(
        LIBERO_WORKSPACE_BOUNDS if bounds is None else bounds
    )
    t_min = 0.0
    t_max = float("inf")
    for axis in range(3):
        if abs(float(direction[axis])) < 1e-12:
            if not (bounds[axis, 0] <= origin[axis] <= bounds[axis, 1]):
                return None
            continue
        t0 = (bounds[axis, 0] - origin[axis]) / direction[axis]
        t1 = (bounds[axis, 1] - origin[axis]) / direction[axis]
        t0, t1 = min(t0, t1), max(t0, t1)
        t_min = max(t_min, float(t0))
        t_max = min(t_max, float(t1))
        if t_max < t_min:
            return None
    if not math.isfinite(t_max) or t_max < 0.0:
        return None
    return max(0.0, t_min), t_max


def solve_camera_ray_constraint(
    ray: Mapping[str, Any],
    click: Mapping[str, Any],
    *,
    consistency_tolerance_m: float = RAY_CLICK_TOLERANCE_M,
) -> dict[str, Any]:
    """Select one 3D point on a calibrated camera ray with an ortho click."""

    origin = np.asarray(ray["origin_xyz_m"], dtype=np.float64)
    direction = np.asarray(ray["direction_xyz"], dtype=np.float64)
    axes = [AXIS_INDEX[str(axis)] for axis in click["fixed_axes"]]
    desired = np.asarray(
        [float(click["components_m"][axis]) for axis in click["fixed_axes"]],
        dtype=np.float64,
    )
    projected_origin = origin[axes]
    projected_direction = direction[axes]
    denominator = float(projected_direction @ projected_direction)
    if denominator < 1e-12:
        return {
            "status": "ray_parallel_to_view",
            "consistency_tolerance_m": float(consistency_tolerance_m),
        }
    t = float(projected_direction @ (desired - projected_origin) / denominator)
    t_range = ray.get("workspace_t_range_m")
    ray_parameter_clamped = False
    if isinstance(t_range, (list, tuple)) and len(t_range) == 2:
        range_min = max(0.0, float(t_range[0]))
        range_max = float(t_range[1])
        if not (range_min <= t <= range_max):
            # The published ray endpoints are integer pixels. Mapping one of
            # those endpoint pixels back into metric view coordinates can put
            # the least-squares ray parameter a fraction of a pixel outside
            # the clipped workspace interval. Accept only that quantization
            # case: clamp to the endpoint, then require the ordinary metric
            # ray consistency tolerance. A genuinely off-segment click still
            # fails because its residual to the clamped endpoint is too large.
            clamped_t = min(range_max, max(range_min, t))
            clamped_xyz = origin + clamped_t * direction
            clamped_residual = float(
                np.linalg.norm(clamped_xyz[axes] - desired)
            )
            if clamped_residual <= float(consistency_tolerance_m):
                t = clamped_t
                ray_parameter_clamped = True
            else:
                nearest_pixel_xy = _project_click_xyz_to_view_pixel(
                    click,
                    clamped_xyz,
                )
                return {
                    "status": "outside_ray_workspace",
                    "ray_parameter_m": t,
                    "nearest_workspace_ray_parameter_m": clamped_t,
                    "nearest_workspace_residual_m": clamped_residual,
                    "consistency_tolerance_m": float(consistency_tolerance_m),
                    **(
                        {"nearest_ray_pixel_xy": nearest_pixel_xy}
                        if nearest_pixel_xy is not None
                        else {}
                    ),
                }
    elif t < 0.0:
        return {
            "status": "outside_ray_workspace",
            "ray_parameter_m": t,
            "consistency_tolerance_m": float(consistency_tolerance_m),
        }
    xyz = origin + t * direction
    residual = float(np.linalg.norm(xyz[axes] - desired))
    if residual > float(consistency_tolerance_m):
        nearest_pixel_xy = _project_click_xyz_to_view_pixel(click, xyz)
        return {
            "status": "inconsistent_views",
            "shared_axis": "camera_ray",
            "shared_axis_residual_m": residual,
            "consistency_tolerance_m": float(consistency_tolerance_m),
            **(
                {"nearest_ray_pixel_xy": nearest_pixel_xy}
                if nearest_pixel_xy is not None
                else {}
            ),
        }
    uncertainty_m = {
        axis: float(click["axis_pixel_size_m"][axis])
        for axis in click["fixed_axes"]
    }
    return {
        "status": "solved",
        "source_kind": "agentview_ray_orthographic_intersection",
        "xyz_m": xyz.tolist(),
        "shared_axis": "camera_ray",
        "shared_axis_residual_m": residual,
        "consistency_tolerance_m": float(consistency_tolerance_m),
        "ray_parameter_m": t,
        "ray_parameter_clamped": ray_parameter_clamped,
        "view_contributions_m": {
            str(ray.get("view") or "agentview"): {"ray_parameter_m": t},
            str(click["view"]): {
                axis: float(value)
                for axis, value in click["components_m"].items()
            },
        },
        "uncertainty_m": uncertainty_m,
    }


def _complementary_views(view_name: str) -> list[str]:
    return [name for name in VIEW_SPECS if name != view_name]


def _world_axes_from_view_click(
    view: PointCloudView, *, u: int, v: int
) -> dict[str, Any]:
    """Map one orthographic click to the view's two displayed world axes."""

    h_axis = str(view.spec["horizontal_axis"])
    v_axis = str(view.spec["vertical_axis"])
    h0, h1 = (float(x) for x in view.spec["horizontal_range_m"])
    v0, v1 = (float(x) for x in view.spec["vertical_range_m"])
    components = {
        h_axis: h0 + (u / max(1, view.width - 1)) * (h1 - h0),
        v_axis: v1 - (v / max(1, view.height - 1)) * (v1 - v0),
    }
    return {
        "view": view.name,
        "pixel_xy": [int(u), int(v)],
        "view_width": int(view.width),
        "view_height": int(view.height),
        "view_horizontal_range_m": [float(h0), float(h1)],
        "view_vertical_range_m": [float(v0), float(v1)],
        "components_m": components,
        "fixed_axes": [h_axis, v_axis],
        "hidden_axis": str(view.spec["hidden_axis"]),
        "axis_pixel_size_m": {
            h_axis: (h1 - h0) / max(1, view.width - 1),
            v_axis: (v1 - v0) / max(1, view.height - 1),
        },
    }


def _project_click_xyz_to_view_pixel(
    click: Mapping[str, Any],
    xyz: np.ndarray,
) -> list[int] | None:
    """Project a world point into the orthographic view that produced click.

    This is guidance for a failed camera-ray intersection. It does not snap
    the click and is intentionally optional for old serialized constraints.
    """

    try:
        width = int(click["view_width"])
        height = int(click["view_height"])
        h_axis = AXIS_INDEX[str(click["fixed_axes"][0])]
        v_axis = AXIS_INDEX[str(click["fixed_axes"][1])]
        h0, h1 = (float(value) for value in click["view_horizontal_range_m"])
        v0, v1 = (float(value) for value in click["view_vertical_range_m"])
        point = np.asarray(xyz, dtype=np.float64).reshape(3)
        u = int(round((float(point[h_axis]) - h0) / (h1 - h0) * (width - 1)))
        v = int(round((v1 - float(point[v_axis])) / (v1 - v0) * (height - 1)))
        return [
            max(0, min(width - 1, u)),
            max(0, min(height - 1, v)),
        ]
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
        return None


def solve_point_constraints(
    pending_constraint: Mapping[str, Any],
    click: Mapping[str, Any],
    *,
    consistency_tolerance_m: float = CONSISTENCY_TOLERANCE_M,
) -> dict[str, Any]:
    """Intersect two complementary orthographic clicks into one 3D point.

    Each click fixes the two world axes its view displays; any complementary
    pair covers all three axes and shares exactly one.  The shared axis must
    agree within ``consistency_tolerance_m``; the solved coordinate takes the
    mean on the shared axis and each view's own value on its unique axis.
    """

    if str(pending_constraint["view"]) == str(click["view"]):
        return {
            "status": "complementary_view_required",
            "clicked_view": str(click["view"]),
            "needs_complementary_view": _complementary_views(str(click["view"])),
        }
    shared = next(
        axis
        for axis in pending_constraint["fixed_axes"]
        if axis in click["fixed_axes"]
    )
    pending_components = pending_constraint["components_m"]
    click_components = click["components_m"]
    residual = abs(float(pending_components[shared]) - float(click_components[shared]))
    if residual > float(consistency_tolerance_m):
        return {
            "status": "inconsistent_views",
            "shared_axis": shared,
            "shared_axis_residual_m": float(residual),
            "consistency_tolerance_m": float(consistency_tolerance_m),
        }
    xyz = np.full(3, np.nan, dtype=np.float64)
    xyz[AXIS_INDEX[shared]] = (
        float(pending_components[shared]) + float(click_components[shared])
    ) / 2.0
    contributions: dict[str, dict[str, float]] = {}
    uncertainty_m: dict[str, float] = {}
    for constraint in (pending_constraint, click):
        view_components: dict[str, float] = {}
        for axis, value in constraint["components_m"].items():
            if axis == shared:
                continue
            xyz[AXIS_INDEX[axis]] = float(value)
            view_components[axis] = float(value)
            uncertainty_m[axis] = float(constraint["axis_pixel_size_m"][axis])
        contributions[str(constraint["view"])] = view_components
    uncertainty_m[shared] = float(residual) + float(
        np.mean([c["axis_pixel_size_m"][shared] for c in (pending_constraint, click)])
    )
    return {
        "status": "solved",
        "xyz_m": xyz.tolist(),
        "shared_axis": shared,
        "shared_axis_residual_m": float(residual),
        "consistency_tolerance_m": float(consistency_tolerance_m),
        "view_contributions_m": contributions,
        "uncertainty_m": uncertainty_m,
    }


def solve_grasp_center(
    pending_constraint: Mapping[str, Any],
    click: Mapping[str, Any],
    *,
    consistency_tolerance_m: float = CONSISTENCY_TOLERANCE_M,
) -> dict[str, Any]:
    """Compatibility alias for older host-side callers."""

    return solve_point_constraints(
        pending_constraint,
        click,
        consistency_tolerance_m=consistency_tolerance_m,
    )


def project_world_vector_to_view(
    view: PointCloudView, vector_xyz: np.ndarray
) -> tuple[float, float]:
    """Project a world vector into a calibrated orthographic view in pixels."""

    vector = np.asarray(vector_xyz, dtype=np.float64).reshape(3)
    h_axis = AXIS_INDEX[str(view.spec["horizontal_axis"])]
    v_axis = AXIS_INDEX[str(view.spec["vertical_axis"])]
    h0, h1 = (float(x) for x in view.spec["horizontal_range_m"])
    v0, v1 = (float(x) for x in view.spec["vertical_range_m"])
    return (
        float(vector[h_axis]) / (h1 - h0) * (view.width - 1),
        -float(vector[v_axis]) / (v1 - v0) * (view.height - 1),
    )


def world_vector_from_view_delta(
    view: PointCloudView, *, start_xy: tuple[int, int], end_xy: tuple[int, int]
) -> dict[str, Any]:
    """Convert a 2D view vector into its two world-axis components."""

    du = float(end_xy[0] - start_xy[0])
    dv = float(end_xy[1] - start_xy[1])
    h0, h1 = (float(x) for x in view.spec["horizontal_range_m"])
    v0, v1 = (float(x) for x in view.spec["vertical_range_m"])
    h_component = du / max(1, view.width - 1) * (h1 - h0)
    v_component = -dv / max(1, view.height - 1) * (v1 - v0)
    components = {
        str(view.spec["horizontal_axis"]): float(h_component),
        str(view.spec["vertical_axis"]): float(v_component),
    }
    vector = np.zeros(3, dtype=np.float64)
    for axis, value in components.items():
        vector[AXIS_INDEX[axis]] = value
    return {
        "view": view.name,
        "start_pixel_xy": [int(start_xy[0]), int(start_xy[1])],
        "end_pixel_xy": [int(end_xy[0]), int(end_xy[1])],
        "pixel_delta_xy": [float(du), float(dv)],
        "vector_world_partial_xyz_m": vector.tolist(),
        "components_m": components,
        "plane_axes": [
            str(view.spec["horizontal_axis"]),
            str(view.spec["vertical_axis"]),
        ],
        "length_m": float(np.linalg.norm(vector)),
    }


def annotate_vector_views(
    views: Mapping[str, PointCloudView],
    projections: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    output_root: str | Path,
) -> dict[str, Path]:
    """Draw the exact 2D vector marks back onto their calibrated views.

    This is intentionally a rendering-only operation: no hidden geometry is
    inferred and no vector is moved or normalized.  Each source view gets the
    start/end pixels supplied by the Operator, making pixel-to-world scaling
    and view-axis conventions directly inspectable in Chrome.
    """

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, Path] = {}
    colors = [(255, 210, 70), (255, 110, 90), (100, 220, 255), (150, 255, 120)]
    for name, view in views.items():
        image = Image.open(view.image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for role_index, (role, by_view) in enumerate(projections.items()):
            item = by_view.get(name)
            if not isinstance(item, Mapping):
                continue
            start = item.get("start_pixel_xy")
            end = item.get("end_pixel_xy")
            if not (isinstance(start, list) and isinstance(end, list) and len(start) == 2 and len(end) == 2):
                continue
            color = colors[role_index % len(colors)]
            sx, sy = (int(start[0]), int(start[1]))
            ex, ey = (int(end[0]), int(end[1]))
            draw.line((sx, sy, ex, ey), fill=color, width=4)
            draw.ellipse((sx - 6, sy - 6, sx + 6, sy + 6), fill=color)
            dx, dy = float(ex - sx), float(ey - sy)
            length = math.hypot(dx, dy) or 1.0
            ux, uy = dx / length, dy / length
            px, py = -uy, ux
            tip = (ex, ey)
            left = (ex - 11 * ux + 6 * px, ey - 11 * uy + 6 * py)
            right = (ex - 11 * ux - 6 * px, ey - 11 * uy - 6 * py)
            draw.polygon([tip, left, right], fill=color)
            draw.text((ex + 8, ey + 4), str(role), fill=color, font=_font(13))
        path = output / f"{name}.vectors.png"
        image.save(path)
        rendered[name] = path
    return rendered


def merge_world_vector_projections(
    projections: Mapping[str, Mapping[str, Any]], *, consistency_tolerance_m: float = CONSISTENCY_TOLERANCE_M
) -> dict[str, Any] | None:
    """Merge two complementary view projections into one checked 3D vector."""

    values: dict[str, list[float]] = {"x": [], "y": [], "z": []}
    for item in projections.values():
        components = item.get("components_m", {})
        if not isinstance(components, Mapping):
            continue
        for axis in values:
            value = components.get(axis)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values[axis].append(float(value))
    if any(not entries for entries in values.values()):
        return None
    conflicts = {
        axis: entries
        for axis, entries in values.items()
        if max(entries) - min(entries) > float(consistency_tolerance_m)
    }
    if conflicts:
        raise ValueError(
            "vector projection conflict: "
            + json.dumps(conflicts, ensure_ascii=False, separators=(",", ":"))
        )
    vector = np.asarray([float(np.mean(values[axis])) for axis in ("x", "y", "z")])
    length = float(np.linalg.norm(vector))
    if length < 0.01:
        raise ValueError("marked vector is shorter than 1 cm")
    return {
        "frame": "world",
        "vector_world_xyz_m": vector.tolist(),
        "length_m": length,
        "source_views": sorted(projections),
        "component_observations_m": values,
        "consistency_tolerance_m": float(consistency_tolerance_m),
    }


def pose_from_points(
    marks: Mapping[str, Mapping[str, Any]],
    *,
    position_point_id: str,
    approach_from_point_id: str,
    jaw_toward_point_id: str,
) -> dict[str, Any]:
    """Build a Panda grip-site pose from three ordinary world points."""

    required = {
        position_point_id,
        approach_from_point_id,
        jaw_toward_point_id,
    }
    if not required.issubset(marks):
        missing = sorted(required.difference(marks))
        raise ValueError(f"missing marked point ids: {', '.join(missing)}")
    center = np.asarray(marks[position_point_id]["xyz_m"], dtype=np.float64)
    approach_ref = np.asarray(marks[approach_from_point_id]["xyz_m"], dtype=np.float64)
    jaw_ref = np.asarray(marks[jaw_toward_point_id]["xyz_m"], dtype=np.float64)

    # The approach reference is where the gripper comes from, so local +Z
    # points from that reference toward the contact center.
    z_axis = center - approach_ref
    x_hint = jaw_ref - center
    approach_span = float(np.linalg.norm(z_axis))
    jaw_span = float(np.linalg.norm(x_hint))
    if approach_span < 0.02:
        raise ValueError("approach point must be at least 2 cm from position point")
    if jaw_span < 0.02:
        raise ValueError("jaw point must be at least 2 cm from position point")
    z_axis /= approach_span
    x_axis = x_hint - z_axis * float(np.dot(x_hint, z_axis))
    orthogonal_span = float(np.linalg.norm(x_axis))
    if orthogonal_span < 0.01:
        raise ValueError("jaw direction is nearly parallel to approach direction")
    x_axis /= orthogonal_span
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise ValueError("marked pose did not form a right-handed frame")
    return {
        "schema_version": "openeta.marked_pose.v1",
        "frame": "world",
        "eef_frame": "panda_grip_site",
        "position_xyz_m": center.tolist(),
        "rotation_matrix": rotation.tolist(),
        "approach_axis_world": z_axis.tolist(),
        "jaw_axis_world": x_axis.tolist(),
        "approach_reference_span_m": approach_span,
        "jaw_reference_span_m": jaw_span,
        "point_ids": {
            "position": position_point_id,
            "approach_from": approach_from_point_id,
            "jaw_toward": jaw_toward_point_id,
        },
    }


def pose_from_marks(marks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """Compatibility wrapper for the former role-based pose convention."""

    required = {"grasp_center", "approach_reference", "jaw_reference"}
    if not required.issubset(marks):
        return None
    return pose_from_points(
        marks,
        position_point_id="grasp_center",
        approach_from_point_id="approach_reference",
        jaw_toward_point_id="jaw_reference",
    )


def _draw_direction_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: tuple[int, int, int],
) -> None:
    """Draw one compact, label-free image-space direction arrow."""

    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    length = math.hypot(dx, dy)
    if length < 3.0:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head_length = min(9.0, max(5.0, length * 0.22))
    head_half_width = min(5.0, max(3.0, head_length * 0.55))
    base = (
        end[0] - head_length * ux,
        end[1] - head_length * uy,
    )
    draw.line((*start, *base), fill=color, width=3)
    draw.polygon(
        (
            end,
            (
                int(round(base[0] + head_half_width * px)),
                int(round(base[1] + head_half_width * py)),
            ),
            (
                int(round(base[0] - head_half_width * px)),
                int(round(base[1] - head_half_width * py)),
            ),
        ),
        fill=color,
    )


def _draw_direction_segment(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: tuple[int, int, int],
) -> None:
    """Draw a compact, label-free line without implying a travel direction."""

    if math.hypot(float(end[0] - start[0]), float(end[1] - start[1])) < 3.0:
        return
    draw.line((*start, *end), fill=color, width=3)


def annotate_views(
    views: Mapping[str, PointCloudView],
    marks: Mapping[str, Mapping[str, Any]],
    *,
    output_root: str | Path,
    source_paths: Mapping[str, str | Path] | None = None,
    solved_agentview_anchor_ray_visual_mode: str | None = None,
) -> dict[str, Path]:
    """Render the caller-selected active marks into every orthographic view.

    The gateway deliberately passes only the point(s) involved in the current
    authoring operation.  Historical solved marks remain in the structured
    mark ledger; drawing that entire ledger into every later planning image
    makes dense episodes progressively unreadable.
    """

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, Path] = {}
    for name, view in views.items():
        source_ref = (source_paths or {}).get(name)
        source = Path(source_ref) if source_ref is not None else view.image_path
        try:
            image = Image.open(source).convert("RGB")
        except (OSError, ValueError):
            continue
        draw = ImageDraw.Draw(image)
        for point_index, (point_id, item) in enumerate(sorted(marks.items())):
            xyz = np.asarray(item["xyz_m"], dtype=np.float64)
            u, v = project_world_to_view(view, xyz)
            color = POINT_COLORS[point_index % len(POINT_COLORS)]
            if (
                solved_agentview_anchor_ray_visual_mode
                in {
                    "surface_inward_segment_v1",
                    "surface_inward_segment_v2",
                }
                and str(item.get("source_kind") or "")
                == "agentview_first_visible_surface"
            ):
                ray = item.get("agentview_source_ray")
                direction = (
                    np.asarray(ray.get("direction_xyz"), dtype=np.float64)
                    if isinstance(ray, Mapping)
                    else np.asarray([], dtype=np.float64)
                )
                if (
                    direction.shape == (3,)
                    and np.isfinite(direction).all()
                    and float(np.linalg.norm(direction)) > 1e-9
                ):
                    direction /= np.linalg.norm(direction)
                    # A short local segment preserves the calibrated
                    # camera-to-surface geometry without covering the whole
                    # scene. The arrow points from the visible surface into
                    # the scene; it does not infer an object center.
                    ray_start = xyz - 0.02 * direction
                    ray_end = xyz + 0.08 * direction
                    ray_start_px = project_world_to_view(view, ray_start)
                    ray_end_px = project_world_to_view(view, ray_end)
                    if (
                        solved_agentview_anchor_ray_visual_mode
                        == "surface_inward_segment_v2"
                    ):
                        _draw_direction_segment(
                            draw,
                            ray_start_px,
                            ray_end_px,
                            color=VISIBLE_SURFACE_MARKER_COLOR,
                        )
                    else:
                        _draw_direction_arrow(
                            draw,
                            ray_start_px,
                            ray_end_px,
                            color=VISIBLE_SURFACE_MARKER_COLOR,
                        )
            radius = 8
            draw.ellipse(
                (u - radius, v - radius, u + radius, v + radius),
                outline=color,
                width=4,
            )
            draw.line((u - 12, v, u + 12, v), fill=color, width=2)
            draw.line((u, v - 12, u, v + 12), fill=color, width=2)
            draw.text((u + 10, v + 7), point_id, fill=color, font=_font(13))
        fingerprint_payload: dict[str, Any] = {
            point_id: item.get("xyz_m")
            for point_id, item in sorted(marks.items())
        }
        if solved_agentview_anchor_ray_visual_mode is not None:
            fingerprint_payload["_solved_agentview_anchor_ray"] = {
                "mode": solved_agentview_anchor_ray_visual_mode,
                "rays": {
                    point_id: item.get("agentview_source_ray")
                    for point_id, item in sorted(marks.items())
                    if item.get("agentview_source_ray") is not None
                },
            }
        suffix = _artifact_fingerprint(fingerprint_payload)
        path = output / f"{name}.marked.{suffix}.png"
        image.save(path)
        rendered[name] = path
    return rendered


def annotate_active_grip_site_views(
    views: Mapping[str, PointCloudView],
    *,
    output_root: str | Path,
    point_id: str,
    target_position_xyz_m: Sequence[float],
    target_pad_contact_centers_world_m: Sequence[Sequence[float]] | None = None,
    source_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Path]:
    """Render one persistent, lightweight authored grip-site target.

    This is the planning point most recently selected through
    ``position_point_id``. It deliberately omits historical measurement marks
    and pose axes: the full candidate orientation remains available in
    ``move_to(preview_only=true)``, while this overlay keeps the chosen
    grip-site center and pad footprint visible across later observations.
    """

    target = np.asarray(target_position_xyz_m, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("active grip-site target must be a finite world point")
    pad_points = _finite_pad_points(target_pad_contact_centers_world_m)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = _artifact_fingerprint(
        {
            "point_id": str(point_id),
            "target_position_xyz_m": target.tolist(),
            "target_pad_contact_centers_world_m": (
                pad_points.tolist() if pad_points is not None else None
            ),
        }
    )
    rendered: dict[str, Path] = {}
    for name, view in views.items():
        source_ref = (source_paths or {}).get(name)
        source = Path(source_ref) if source_ref is not None else view.image_path
        try:
            image = Image.open(source).convert("RGB")
        except Exception:
            continue
        draw = ImageDraw.Draw(image)
        target_u, target_v = project_world_to_view(view, target)
        color = (255, 150, 20)
        draw.rectangle(
            (target_u - 7, target_v - 7, target_u + 7, target_v + 7),
            outline=color,
            width=3,
        )
        draw.line(
            (target_u - 11, target_v, target_u + 11, target_v),
            fill=color,
            width=2,
        )
        draw.line(
            (target_u, target_v - 11, target_u, target_v + 11),
            fill=color,
            width=2,
        )
        draw.text(
            (target_u + 10, target_v - 12),
            f"ACTIVE grip_site {point_id}",
            fill=(255, 175, 35),
            font=_font(11),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        if pad_points is not None:
            projected_pads = [
                project_world_to_view(view, pad) for pad in pad_points
            ]
            draw.line(
                (
                    projected_pads[0][0],
                    projected_pads[0][1],
                    projected_pads[1][0],
                    projected_pads[1][1],
                ),
                fill=color,
                width=2,
            )
            for index, (pad_u, pad_v) in enumerate(projected_pads, start=1):
                draw.rectangle(
                    (pad_u - 5, pad_v - 5, pad_u + 5, pad_v + 5),
                    outline=color,
                    width=2,
                )
                draw.text(
                    (pad_u + 6, pad_v + 2),
                    f"PAD{index}",
                    fill=(255, 175, 35),
                    font=_font(9),
                    stroke_width=2,
                    stroke_fill=(0, 0, 0),
                )
        path = output / f"{name}.active-grip-site.{fingerprint}.png"
        image.save(path)
        rendered[name] = path
    return rendered


def annotate_pose_preview_views(
    views: Mapping[str, PointCloudView],
    *,
    output_root: str | Path,
    target_position_xyz_m: Sequence[float],
    target_rotation_matrix: Sequence[Sequence[float]],
    target_pad_contact_centers_world_m: Sequence[Sequence[float]] | None = None,
    target_pad_sweep_start_centers_world_m: Sequence[Sequence[float]] | None = None,
    target_pad_boxes: Sequence[Mapping[str, Any]] | None = None,
    actual_position_xyz_m: Sequence[float] | None = None,
    actual_rotation_matrix: Sequence[Sequence[float]] | None = None,
    compact_labels: bool = False,
) -> dict[str, Path]:
    """Render an uncluttered actual-versus-candidate Panda pose inspection.

    This is a preview of the exact resolved ``move_to`` target, not a grasp
    proposal or an automatic correction.  Historical point marks are omitted
    so the candidate pad footprint and axes remain legible.
    """

    target = np.asarray(target_position_xyz_m, dtype=np.float64)
    target_rotation = np.asarray(target_rotation_matrix, dtype=np.float64)
    if (
        target.shape != (3,)
        or target_rotation.shape != (3, 3)
        or not np.isfinite(target).all()
        or not np.isfinite(target_rotation).all()
    ):
        raise ValueError("pose preview target must be a finite rigid pose")
    actual = (
        np.asarray(actual_position_xyz_m, dtype=np.float64)
        if actual_position_xyz_m is not None
        else None
    )
    actual_rotation = (
        np.asarray(actual_rotation_matrix, dtype=np.float64)
        if actual_rotation_matrix is not None
        else None
    )
    pad_points = _finite_pad_points(target_pad_contact_centers_world_m)
    sweep_start_points = _finite_pad_points(
        target_pad_sweep_start_centers_world_m
    )
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = _artifact_fingerprint(
        {
            "target_position_xyz_m": target.tolist(),
            "target_rotation_matrix": target_rotation.tolist(),
        }
    )
    rendered: dict[str, Path] = {}
    for name, view in views.items():
        source = (
            view.clean_image_path
            if view.clean_image_path is not None
            and view.clean_image_path.is_file()
            else view.image_path
        )
        try:
            image = Image.open(source).convert("RGB")
        except Exception:
            continue
        draw = ImageDraw.Draw(image)
        if (
            actual is not None
            and actual.shape == (3,)
            and np.isfinite(actual).all()
            and actual_rotation is not None
            and actual_rotation.shape == (3, 3)
            and np.isfinite(actual_rotation).all()
        ):
            actual_u, actual_v = project_world_to_view(view, actual)
            draw.ellipse(
                (actual_u - 5, actual_v - 5, actual_u + 5, actual_v + 5),
                outline=(255, 0, 255),
                width=2,
            )
            if not compact_labels:
                _draw_pose_axis(
                    draw,
                    spec=view.spec,
                    center=actual,
                    direction=actual_rotation[:, 0],
                    size=view.width,
                    color=(205, 80, 205),
                    label="ACT JAW",
                )
                _draw_pose_axis(
                    draw,
                    spec=view.spec,
                    center=actual,
                    direction=actual_rotation[:, 2],
                    size=view.width,
                    color=(100, 150, 205),
                    label="ACT APP",
                )
        target_u, target_v = project_world_to_view(view, target)
        draw.ellipse(
            (target_u - 8, target_v - 8, target_u + 8, target_v + 8),
            outline=(255, 150, 20),
            width=4,
        )
        if not compact_labels:
            draw.text(
                (target_u + 10, target_v - 12),
                "CANDIDATE grip_site",
                fill=(255, 170, 30),
                font=_font(11),
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
        _draw_pose_axis(
            draw,
            spec=view.spec,
            center=target,
            direction=target_rotation[:, 0],
            size=view.width,
            color=(255, 190, 40),
            label="" if compact_labels else "JAW",
        )
        _draw_pose_axis(
            draw,
            spec=view.spec,
            center=target,
            direction=target_rotation[:, 2],
            size=view.width,
            color=(40, 220, 255),
            label="" if compact_labels else "APP",
        )
        if pad_points is not None:
            projected_pads = [
                project_world_to_view(view, pad) for pad in pad_points
            ]
            draw.line(
                (
                    projected_pads[0][0],
                    projected_pads[0][1],
                    projected_pads[1][0],
                    projected_pads[1][1],
                ),
                fill=(255, 150, 20),
                width=3,
            )
            for index, (pad_u, pad_v) in enumerate(projected_pads, start=1):
                draw.rectangle(
                    (pad_u - 6, pad_v - 6, pad_u + 6, pad_v + 6),
                    outline=(255, 150, 20),
                    width=3,
                )
                if not compact_labels:
                    draw.text(
                        (pad_u + 8, pad_v + 3),
                        f"PAD{index}",
                        fill=(255, 170, 30),
                        font=_font(10),
                        stroke_width=2,
                        stroke_fill=(0, 0, 0),
                    )
        if pad_points is not None and sweep_start_points is not None:
            for start, end in zip(sweep_start_points, pad_points):
                start_u, start_v = project_world_to_view(view, start)
                end_u, end_v = project_world_to_view(view, end)
                draw.line(
                    (start_u, start_v, end_u, end_v),
                    fill=(255, 210, 80),
                    width=2,
                )
                draw.ellipse(
                    (start_u - 4, start_v - 4, start_u + 4, start_v + 4),
                    outline=(255, 210, 80),
                    width=2,
                )
        for box in target_pad_boxes or ():
            try:
                center = np.asarray(box["center_world_m"], dtype=np.float64)
                rotation = np.asarray(box["rotation_world"], dtype=np.float64)
                half_size = np.asarray(box["half_size_m"], dtype=np.float64)
            except (KeyError, TypeError, ValueError):
                continue
            corners: list[tuple[int, int]] = []
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        corner = center + rotation @ (
                            half_size
                            * np.asarray((sx, sy, sz), dtype=np.float64)
                        )
                        corners.append(project_world_to_view(view, corner))
            hull = _convex_hull_2d(corners)
            if len(hull) >= 3:
                draw.polygon(hull, outline=(255, 150, 20), width=2)
        legend_y = max(47, image.height - 42)
        draw.rectangle((0, legend_y, image.width, image.height), fill=(0, 0, 0))
        draw.text(
            (8, legend_y + 3),
            (
                "orange = requested pads/JAW   pale lines = closing sweep   cyan = APP"
                if compact_labels
                else "orange = requested pads/JAW   pale lines = geometric closing sweep   cyan = candidate APP"
            ),
            fill=(255, 190, 60),
            font=_font(11),
        )
        draw.text(
            (8, legend_y + 21),
            "magenta/dim axes = current actual pose",
            fill=(230, 120, 230),
            font=_font(11),
        )
        path = output / f"{name}.pose-preview.{fingerprint}.png"
        image.save(path)
        rendered[name] = path
    return rendered


def render_move_feedback_crop(
    view: PointCloudView,
    *,
    output_path: str | Path,
    target_position_xyz_m: Sequence[float],
    actual_position_xyz_m: Sequence[float],
    actual_pad_contact_centers_world_m: Sequence[Sequence[float]] | None = None,
    actual_pad_boxes: Sequence[Mapping[str, Any]] | None = None,
    compact_actual_pad_geometry: bool = False,
    render_actual_pad_contact_corridor: bool = False,
    render_actual_pad_closing_cues: bool = False,
    output_size_px: int = 384,
    minimum_source_crop_px: int = 96,
    margin_source_px: int = 24,
) -> Path:
    """Render a local, non-clickable TARGET-versus-ACTUAL motion view.

    The crop is derived from the clean calibrated orthographic source, so
    historical authored marks cannot leak into closed-loop motion feedback.
    Its resized pixels deliberately do *not* share the source view coordinate
    system and therefore must never be accepted by ``mark_point``.
    """

    target = np.asarray(target_position_xyz_m, dtype=np.float64)
    actual = np.asarray(actual_position_xyz_m, dtype=np.float64)
    if (
        target.shape != (3,)
        or actual.shape != (3,)
        or not np.isfinite(target).all()
        or not np.isfinite(actual).all()
    ):
        raise ValueError("move feedback target and actual must be finite XYZ")
    if output_size_px < 64:
        raise ValueError("move feedback output_size_px must be at least 64")
    source = (
        view.clean_image_path
        if view.clean_image_path is not None
        and view.clean_image_path.is_file()
        else view.image_path
    )
    image = Image.open(source).convert("RGB")
    pad_points = _finite_pad_points(actual_pad_contact_centers_world_m)
    pad_boxes = (
        _finite_pad_boxes(actual_pad_boxes)
        if compact_actual_pad_geometry
        else []
    )
    contact_faces = (
        _actual_pad_inner_contact_faces(pad_boxes, pad_points)
        if render_actual_pad_contact_corridor
        else []
    )
    world_points = [target, actual]
    if pad_points is not None:
        world_points.extend(pad_points)
    pad_box_corners: list[list[np.ndarray]] = []
    for box in pad_boxes:
        corners: list[np.ndarray] = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corners.append(
                        box["center_world_m"]
                        + box["rotation_world"]
                        @ (
                            box["half_size_m"]
                            * np.asarray((sx, sy, sz), dtype=np.float64)
                        )
                    )
        pad_box_corners.append(corners)
        world_points.extend(corners)
    for face in contact_faces:
        world_points.extend(face)
    projected = [
        project_world_to_view(view, np.asarray(point, dtype=np.float64))
        for point in world_points
    ]

    min_u = min(point[0] for point in projected) - int(margin_source_px)
    max_u = max(point[0] for point in projected) + int(margin_source_px)
    min_v = min(point[1] for point in projected) - int(margin_source_px)
    max_v = max(point[1] for point in projected) + int(margin_source_px)
    crop_size = max(
        int(minimum_source_crop_px),
        max_u - min_u,
        max_v - min_v,
    )
    crop_size = min(crop_size, image.width, image.height)
    center_u = (min_u + max_u) / 2.0
    center_v = (min_v + max_v) / 2.0
    left = int(round(center_u - crop_size / 2.0))
    top = int(round(center_v - crop_size / 2.0))
    left = max(0, min(image.width - crop_size, left))
    top = max(0, min(image.height - crop_size, top))
    right = left + crop_size
    bottom = top + crop_size
    resampling = getattr(Image, "Resampling", Image).NEAREST
    cropped = image.crop((left, top, right, bottom)).resize(
        (int(output_size_px), int(output_size_px)),
        resample=resampling,
    )
    draw = ImageDraw.Draw(cropped)
    scale = float(output_size_px) / float(crop_size)

    def local(pixel: tuple[int, int]) -> tuple[int, int]:
        return (
            int(round((pixel[0] - left) * scale)),
            int(round((pixel[1] - top) * scale)),
        )

    target_uv = local(projected[0])
    actual_uv = local(projected[1])
    draw.line(
        (*actual_uv, *target_uv),
        fill=(245, 245, 245),
        width=2 if compact_actual_pad_geometry else 3,
    )

    target_color = (60, 255, 120)
    actual_color = (255, 0, 255)
    radius = 7 if compact_actual_pad_geometry else 9
    draw.ellipse(
        (
            target_uv[0] - radius,
            target_uv[1] - radius,
            target_uv[0] + radius,
            target_uv[1] + radius,
        ),
        outline=target_color,
        width=3 if compact_actual_pad_geometry else 4,
    )
    if not compact_actual_pad_geometry:
        draw.text(
            (target_uv[0] + 10, target_uv[1] - 20),
            "T",
            fill=target_color,
            font=_font(14),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    actual_radius = 5 if compact_actual_pad_geometry else 7
    draw.ellipse(
        (
            actual_uv[0] - actual_radius,
            actual_uv[1] - actual_radius,
            actual_uv[0] + actual_radius,
            actual_uv[1] + actual_radius,
        ),
        outline=actual_color,
        width=2 if compact_actual_pad_geometry else 3,
    )
    cross_radius = 7 if compact_actual_pad_geometry else 11
    draw.line(
        (
            actual_uv[0] - cross_radius,
            actual_uv[1],
            actual_uv[0] + cross_radius,
            actual_uv[1],
        ),
        fill=actual_color,
        width=2 if compact_actual_pad_geometry else 3,
    )
    draw.line(
        (
            actual_uv[0],
            actual_uv[1] - cross_radius,
            actual_uv[0],
            actual_uv[1] + cross_radius,
        ),
        fill=actual_color,
        width=2 if compact_actual_pad_geometry else 3,
    )
    if not compact_actual_pad_geometry:
        draw.text(
            (actual_uv[0] + 10, actual_uv[1] + 4),
            "A",
            fill=actual_color,
            font=_font(14),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    if pad_points is not None and not contact_faces:
        pad_projection_end = 2 + len(pad_points)
        pads = [local(pixel) for pixel in projected[2:pad_projection_end]]
        pad_color = (255, 110, 40)
        if compact_actual_pad_geometry:
            for pad_u, pad_v in pads:
                draw.ellipse(
                    (pad_u - 3, pad_v - 3, pad_u + 3, pad_v + 3),
                    fill=pad_color,
                )
        else:
            draw.line((*pads[0], *pads[1]), fill=pad_color, width=3)
            for pad_u, pad_v in pads:
                draw.rectangle(
                    (pad_u - 6, pad_v - 6, pad_u + 6, pad_v + 6),
                    outline=pad_color,
                    width=3,
                )

    projected_pad_centers: list[tuple[int, int]] = []
    if pad_points is not None:
        pad_projection_end = 2 + len(pad_points)
        projected_pad_centers = [
            local(pixel) for pixel in projected[2:pad_projection_end]
        ]

    if compact_actual_pad_geometry:
        pad_box_projection_start = 2 + (
            len(pad_points) if pad_points is not None else 0
        )
        offset = pad_box_projection_start
        pad_color = (255, 110, 40)
        for corners in pad_box_corners:
            projected_corners = projected[offset : offset + len(corners)]
            offset += len(corners)
            hull = _convex_hull_2d([local(pixel) for pixel in projected_corners])
            if len(hull) >= 3:
                draw.polygon(
                    hull,
                    outline=pad_color,
                    width=2 if contact_faces else 3,
                )
        if len(contact_faces) == 2:
            face_projection_start = offset
            projected_faces: list[list[tuple[int, int]]] = []
            for face in contact_faces:
                face_pixels = projected[
                    face_projection_start : face_projection_start + len(face)
                ]
                face_projection_start += len(face)
                projected_faces.append(
                    [local(pixel) for pixel in face_pixels]
                )
            corridor_hull = _convex_hull_2d(
                [pixel for face in projected_faces for pixel in face]
            )
            if len(corridor_hull) >= 3:
                draw.polygon(
                    corridor_hull,
                    outline=(255, 210, 80),
                    width=2,
                )
            for face_pixels in projected_faces:
                face_hull = _convex_hull_2d(face_pixels)
                if len(face_hull) >= 3:
                    draw.polygon(
                        face_hull,
                        outline=(255, 235, 130),
                        width=3,
                    )
            if (
                render_actual_pad_closing_cues
                and len(projected_pad_centers) == 2
            ):
                _draw_inward_closing_cues(
                    draw,
                    projected_pad_centers,
                )
    else:
        h_axis = str(view.spec["horizontal_axis"]).upper()
        v_axis = str(view.spec["vertical_axis"]).upper()
        h_color = AXIS_COLORS[h_axis.lower()]
        v_color = AXIS_COLORS[v_axis.lower()]
        draw.line((14, 20, 54, 20), fill=h_color, width=3)
        draw.polygon(((54, 20), (46, 15), (46, 25)), fill=h_color)
        draw.text(
            (59, 10),
            f"+{h_axis}",
            fill=h_color,
            font=_font(12),
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
        draw.line((14, 34, 14, 4), fill=v_color, width=3)
        draw.polygon(((14, 4), (9, 12), (19, 12)), fill=v_color)
        draw.text(
            (23, 2),
            f"+{v_axis}",
            fill=v_color,
            font=_font(12),
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(output)
    return output


def render_move_feedback_local_cloud(
    view: PointCloudView,
    *,
    output_path: str | Path,
    world_points: np.ndarray,
    colors_rgb: np.ndarray,
    target_position_xyz_m: Sequence[float],
    actual_position_xyz_m: Sequence[float],
    actual_pad_contact_centers_world_m: Sequence[Sequence[float]] | None = None,
    actual_pad_boxes: Sequence[Mapping[str, Any]] | None = None,
    render_actual_pad_contact_corridor: bool = False,
    render_actual_pad_closing_cues: bool = False,
    output_size_px: int = 384,
    minimum_span_m: float = 0.14,
    margin_m: float = 0.025,
    hidden_margin_m: float = 0.08,
) -> Path:
    """Render stopped-pose feedback from a local current-cloud slice.

    This rerasterizes observation-bound 3D points near the stopped grip-site
    instead of enlarging a global orthographic image. Distant depth layers
    cannot hide the local object, while live pad boxes keep metric geometry.
    """

    points = np.asarray(world_points, dtype=np.float64)
    colors = np.asarray(colors_rgb, dtype=np.uint8)
    target = np.asarray(target_position_xyz_m, dtype=np.float64)
    actual = np.asarray(actual_position_xyz_m, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or colors.shape != points.shape
        or target.shape != (3,)
        or actual.shape != (3,)
        or not np.isfinite(target).all()
        or not np.isfinite(actual).all()
    ):
        raise ValueError("local move feedback requires finite XYZ/RGB arrays")

    pad_points = _finite_pad_points(actual_pad_contact_centers_world_m)
    pad_boxes = _finite_pad_boxes(actual_pad_boxes)
    geometry_points = [target, actual]
    if pad_points is not None:
        geometry_points.extend(pad_points)
    for box in pad_boxes:
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    geometry_points.append(
                        box["center_world_m"]
                        + box["rotation_world"]
                        @ (
                            box["half_size_m"]
                            * np.asarray((sx, sy, sz), dtype=np.float64)
                        )
                    )
    geometry = np.asarray(geometry_points, dtype=np.float64)
    h_axis = AXIS_INDEX[str(view.spec["horizontal_axis"])]
    v_axis = AXIS_INDEX[str(view.spec["vertical_axis"])]
    hidden_axis = AXIS_INDEX[str(view.spec["hidden_axis"])]

    h_min = float(np.min(geometry[:, h_axis]) - margin_m)
    h_max = float(np.max(geometry[:, h_axis]) + margin_m)
    v_min = float(np.min(geometry[:, v_axis]) - margin_m)
    v_max = float(np.max(geometry[:, v_axis]) + margin_m)
    span = max(float(minimum_span_m), h_max - h_min, v_max - v_min)
    h_center = (h_min + h_max) / 2.0
    v_center = (v_min + v_max) / 2.0
    h_range = [h_center - span / 2.0, h_center + span / 2.0]
    v_range = [v_center - span / 2.0, v_center + span / 2.0]
    hidden_range = [
        float(np.min(geometry[:, hidden_axis]) - hidden_margin_m),
        float(np.max(geometry[:, hidden_axis]) + hidden_margin_m),
    ]

    finite = np.isfinite(points).all(axis=1)
    inside = (
        finite
        & (points[:, h_axis] >= h_range[0])
        & (points[:, h_axis] <= h_range[1])
        & (points[:, v_axis] >= v_range[0])
        & (points[:, v_axis] <= v_range[1])
        & (points[:, hidden_axis] >= hidden_range[0])
        & (points[:, hidden_axis] <= hidden_range[1])
    )
    local_points = points[inside]
    local_colors = colors[inside]
    if len(local_points) == 0:
        return render_move_feedback_crop(
            view,
            output_path=output_path,
            target_position_xyz_m=target,
            actual_position_xyz_m=actual,
            actual_pad_contact_centers_world_m=pad_points,
            actual_pad_boxes=actual_pad_boxes,
            compact_actual_pad_geometry=True,
            render_actual_pad_contact_corridor=(
                render_actual_pad_contact_corridor
            ),
            render_actual_pad_closing_cues=render_actual_pad_closing_cues,
            output_size_px=output_size_px,
        )

    u = np.clip(
        np.rint(
            (local_points[:, h_axis] - h_range[0])
            / (h_range[1] - h_range[0])
            * (output_size_px - 1)
        ).astype(int),
        0,
        output_size_px - 1,
    )
    v = np.clip(
        np.rint(
            (v_range[1] - local_points[:, v_axis])
            / (v_range[1] - v_range[0])
            * (output_size_px - 1)
        ).astype(int),
        0,
        output_size_px - 1,
    )
    pixel = v * output_size_px + u
    priority = local_points[:, hidden_axis]
    if str(view.spec["visibility"]) in {
        "nearest_negative_y",
        "nearest_negative_x",
    }:
        priority = -priority
    order = np.lexsort((-priority, pixel))
    sorted_pixels = pixel[order]
    first = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
    chosen = order[first]
    base_u, base_v = u[chosen], v[chosen]
    chosen_colors = np.clip(
        local_colors[chosen].astype(np.float64) * 1.45 + 24.0,
        0.0,
        255.0,
    ).astype(np.uint8)
    canvas = np.full(
        (output_size_px, output_size_px, 3),
        14,
        dtype=np.uint8,
    )
    # This closes sub-pixel RGB-D holes only; it does not complete geometry.
    for dv, du in (
        (0, 0),
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ):
        uu = np.clip(base_u + du, 0, output_size_px - 1)
        vv = np.clip(base_v + dv, 0, output_size_px - 1)
        empty = np.all(canvas[vv, uu] == 14, axis=1)
        canvas[vv[empty], uu[empty]] = chosen_colors[empty]

    output = Path(output_path)
    source_path = output.with_name(f"{output.stem}.local-cloud.png")
    metadata_path = output.with_name(f"{output.stem}.local-cloud.json")
    source_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGB").save(source_path)
    local_spec = dict(view.spec)
    local_spec["horizontal_range_m"] = h_range
    local_spec["vertical_range_m"] = v_range
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "openeta.move_feedback_local_cloud.v1",
                "source_view": view.name,
                "width": int(output_size_px),
                "height": int(output_size_px),
                "horizontal_axis": local_spec["horizontal_axis"],
                "vertical_axis": local_spec["vertical_axis"],
                "hidden_axis": local_spec["hidden_axis"],
                "horizontal_range_m": h_range,
                "vertical_range_m": v_range,
                "hidden_range_m": hidden_range,
                "visibility": local_spec["visibility"],
                "target_position_xyz_m": target.tolist(),
                "actual_position_xyz_m": actual.tolist(),
                "actual_pad_contact_centers_world_m": (
                    pad_points.tolist() if pad_points is not None else None
                ),
                "actual_pad_boxes": [
                    {
                        "center_world_m": box["center_world_m"].tolist(),
                        "rotation_world": box["rotation_world"].tolist(),
                        "half_size_m": box["half_size_m"].tolist(),
                    }
                    for box in pad_boxes
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    local_view = PointCloudView(
        name=view.name,
        image_path=source_path,
        clean_image_path=source_path,
        lookup_path=view.lookup_path,
        metadata_path=view.metadata_path,
        width=int(output_size_px),
        height=int(output_size_px),
        spec=local_spec,
    )
    return render_move_feedback_crop(
        local_view,
        output_path=output,
        target_position_xyz_m=target,
        actual_position_xyz_m=actual,
        actual_pad_contact_centers_world_m=pad_points,
        actual_pad_boxes=actual_pad_boxes,
        compact_actual_pad_geometry=True,
        render_actual_pad_contact_corridor=(
            render_actual_pad_contact_corridor
        ),
        render_actual_pad_closing_cues=render_actual_pad_closing_cues,
        output_size_px=output_size_px,
        minimum_source_crop_px=output_size_px,
        margin_source_px=0,
    )


def render_pose_local_inspection_contact_sheet(
    views: Mapping[str, PointCloudView],
    *,
    output_path: str | Path,
    target_position_xyz_m: Sequence[float],
    source_paths: Mapping[str, str | Path] | None = None,
    crop_size_px: int = 192,
    scale: int = 2,
    strip_text_overlays: bool = True,
) -> Path:
    """Create a generic zoomed inspection sheet around a candidate grip site.

    The crop is purely visual: it does not infer object identity, collision
    validity, or a better grasp.  It preserves the same calibrated projection
    and overlays already present in each pose-preview image.
    """

    target = np.asarray(target_position_xyz_m, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("local inspection target must be a finite 3-vector")
    panels: list[Image.Image] = []
    labels: list[str] = []
    half = max(32, int(crop_size_px) // 2)
    for name in ("pointcloud_top", "pointcloud_front", "pointcloud_side"):
        view = views.get(name)
        if view is None:
            continue
        source = (
            Path(source_paths[name])
            if source_paths is not None and name in source_paths
            else view.image_path
        )
        if not source.is_file():
            continue
        if not hasattr(view, "spec"):
            continue
        try:
            image = Image.open(source).convert("RGB")
        except Exception:
            continue
        u, v = project_world_to_view(view, target)
        left = max(0, min(image.width - 1, int(round(u)) - half))
        top = max(0, min(image.height - 1, int(round(v)) - half))
        right = min(image.width, left + 2 * half)
        bottom = min(image.height, top + 2 * half)
        crop = image.crop((left, top, right, bottom))
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.NEAREST)
        if strip_text_overlays:
            # Pose-preview source images reserve their bottom 42 pixels for a
            # prose legend.  Cropping that legend into the local sheet makes
            # the enlarged geometry harder to inspect, so blank only the
            # corresponding source-image band.  Candidate/pad/axis geometry
            # remains untouched.
            legend_top = max(47, image.height - 42)
            overlap_top = max(top, legend_top)
            overlap_bottom = min(bottom, image.height)
            if overlap_top < overlap_bottom:
                draw = ImageDraw.Draw(crop)
                y0 = (overlap_top - top) * scale
                y1 = (overlap_bottom - top) * scale
                draw.rectangle((0, y0, crop.width, y1), fill=(0, 0, 0))
        panels.append(crop)
        labels.append(name.replace("pointcloud_", ""))
    if not panels:
        return None
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    # Keep the contact sheet image-only.  The panel order and source-view
    # names remain available in the structured tool result/metadata; drawing
    # captions here obscures the small local geometry the sheet is meant to
    # expose to the Operator.
    sheet = Image.new("RGB", (width * len(panels), height), (18, 18, 18))
    for index, (panel, label) in enumerate(zip(panels, labels)):
        sheet.paste(panel, (index * width, 0))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def render_pose_candidate_frame_inspection(
    views: Mapping[str, PointCloudView],
    *,
    output_path: str | Path,
    target_position_xyz_m: Sequence[float],
    target_rotation_matrix: Sequence[Sequence[float]],
    target_pad_boxes: Sequence[Mapping[str, Any]] | None = None,
    extent_m: float = 0.12,
    size: int = 384,
) -> Path | None:
    """Render local candidate-frame cross-sections from observed geometry.

    This is a visualization-only transform of the existing calibrated cloud.
    It does not segment, snap, score, or alter the candidate pose.  Candidate
    local +X is the jaw direction and +Z is the approach direction.
    """

    target = np.asarray(target_position_xyz_m, dtype=np.float64)
    rotation = np.asarray(target_rotation_matrix, dtype=np.float64)
    if (
        target.shape != (3,)
        or rotation.shape != (3, 3)
        or not np.isfinite(target).all()
        or not np.isfinite(rotation).all()
    ):
        return None
    clouds: list[np.ndarray] = []
    for view in views.values():
        try:
            payload = np.load(view.lookup_path)
            xyz = np.asarray(payload["xyz_m"], dtype=np.float64)
            valid = np.asarray(payload["valid"], dtype=bool)
            clouds.append(xyz[valid])
        except (AttributeError, KeyError, OSError, TypeError, ValueError):
            continue
    if not clouds:
        return None
    world = np.concatenate(clouds, axis=0)
    local = (world - target) @ rotation
    radius = max(0.04, float(extent_m))
    keep = np.max(np.abs(local), axis=1) <= radius
    local = local[keep]
    if len(local) == 0:
        return None
    # Voxelize only for legibility and deterministic rendering.
    keys = np.floor((local + radius) / 0.003).astype(np.int64)
    _, unique = np.unique(keys, axis=0, return_index=True)
    local = local[np.sort(unique)]

    panels: list[Image.Image] = []
    specs = (
        (0, 2, "JAW / APPROACH", "+X jaw", "+Z approach"),
        (1, 2, "LATERAL / APPROACH", "+Y lateral", "+Z approach"),
    )
    local_boxes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for box in target_pad_boxes or ():
        try:
            center_world = np.asarray(box["center_world_m"], dtype=np.float64)
            rotation_world = np.asarray(box["rotation_world"], dtype=np.float64)
            half_size = np.asarray(box["half_size_m"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            continue
        if (
            center_world.shape == (3,)
            and rotation_world.shape == (3, 3)
            and half_size.shape == (3,)
        ):
            local_boxes.append(
                (
                    rotation.T @ (center_world - target),
                    rotation.T @ rotation_world,
                    half_size,
                )
            )
    for horizontal, vertical, title, x_label, z_label in specs:
        panel = Image.new("RGB", (size, size), (12, 12, 12))
        draw = ImageDraw.Draw(panel)
        for value in np.linspace(-radius, radius, 9):
            px = int(round((value + radius) / (2 * radius) * (size - 1)))
            py = int(round((radius - value) / (2 * radius) * (size - 1)))
            draw.line((px, 0, px, size - 1), fill=(34, 34, 34), width=1)
            draw.line((0, py, size - 1, py), fill=(34, 34, 34), width=1)
        u = np.rint(
            (local[:, horizontal] + radius) / (2 * radius) * (size - 1)
        ).astype(int)
        v = np.rint(
            (radius - local[:, vertical]) / (2 * radius) * (size - 1)
        ).astype(int)
        depth_axis = 1 if horizontal == 0 else 0
        depth = np.abs(local[:, depth_axis])
        brightness = np.clip(235 - depth / radius * 150, 70, 235).astype(int)
        for px, py, value in zip(u, v, brightness):
            draw.point((int(px), int(py)), fill=(int(value), int(value), int(value)))
        for center, box_rotation, half_size in local_boxes:
            corners: list[tuple[int, int]] = []
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        corner = center + box_rotation @ (
                            half_size * np.asarray((sx, sy, sz))
                        )
                        px = int(
                            round(
                                (corner[horizontal] + radius)
                                / (2 * radius)
                                * (size - 1)
                            )
                        )
                        py = int(
                            round(
                                (radius - corner[vertical])
                                / (2 * radius)
                                * (size - 1)
                            )
                        )
                        corners.append((px, py))
            hull = _convex_hull_2d(corners)
            if len(hull) >= 3:
                draw.polygon(hull, outline=(255, 155, 20), width=3)
        origin = size // 2
        draw.ellipse(
            (origin - 6, origin - 6, origin + 6, origin + 6),
            outline=(255, 155, 20),
            width=3,
        )
        draw.rectangle((0, 0, size, 43), fill=(0, 0, 0))
        draw.text((8, 5), title, fill=(255, 255, 255), font=_font(13))
        draw.text(
            (8, 24),
            f"right={x_label}  up={z_label}  span=±{radius * 1000:.0f}mm",
            fill=(190, 190, 190),
            font=_font(10),
        )
        panels.append(panel)
    sheet = Image.new("RGB", (size * 2, size), (10, 10, 10))
    for index, panel in enumerate(panels):
        sheet.paste(panel, (index * size, 0))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def render_pose_candidate_frame_preview_views(
    world_points: np.ndarray,
    colors_rgb: np.ndarray,
    *,
    output_root: str | Path,
    target_position_xyz_m: Sequence[float],
    target_rotation_matrix: Sequence[Sequence[float]],
    target_pad_contact_centers_world_m: Sequence[Sequence[float]] | None = None,
    target_pad_sweep_start_centers_world_m: Sequence[Sequence[float]] | None = None,
    target_pad_boxes: Sequence[Mapping[str, Any]] | None = None,
    target_pad_sweep_start_boxes: Sequence[Mapping[str, Any]] | None = None,
    target_pad_capture_corridor_boxes: (
        Sequence[Mapping[str, Any]] | None
    ) = None,
    extent_m: float = 0.06,
    hidden_margin_m: float = 0.02,
    size: int = 384,
    axis_label_semantics: str = "local_xyz_v1",
    visual_mode: str = "current_candidate_frame_v1",
    markable: bool = False,
) -> dict[str, Path]:
    """Render two clean candidate-gripper-frame pose-preview views.

    The renderer emits the existing ``pointcloud_front`` and
    ``pointcloud_side`` image slots. A caller may expose those slots through
    stable markable image references without changing their pixels or count.
    Both views keep candidate local +X (jaw closing axis) horizontal:

    - ``pointcloud_front`` shows local X-Z (jaw versus approach height).
    - ``pointcloud_side`` shows local X-Y (jaw versus finger depth).

    ``axis_label_semantics`` only changes the minimal glyph labels.  The
    historical ``local_xyz_v1`` mode keeps X/Y/Z for immutable replay of older
    profiles.  ``jaw_lat_app_v2`` labels the same candidate-local basis as
    JAW/LAT/APP so it cannot be mistaken for the world frame.

    This is a rigid visualization transform of the current observed cloud. It
    does not segment, snap, score, or alter the requested pose.
    """

    if visual_mode not in {
        "current_candidate_frame_v1",
        "candidate_corridor_local_v2",
    }:
        raise ValueError(
            "visual_mode must be current_candidate_frame_v1 or "
            "candidate_corridor_local_v2"
        )
    points = np.asarray(world_points, dtype=np.float64)
    colors = np.asarray(colors_rgb, dtype=np.uint8)
    target = np.asarray(target_position_xyz_m, dtype=np.float64)
    rotation = np.asarray(target_rotation_matrix, dtype=np.float64)
    radius = max(0.04, float(extent_m))
    # This visual mode uses the same slots, dimensions, axes, and geometry
    # inputs with a slightly tighter local decision surface. It is a renderer
    # choice only: it does not snap, score, segment, or infer contact.
    if visual_mode == "candidate_corridor_local_v2":
        radius = max(0.04, min(radius, 0.05))
    hidden_margin = max(0.005, float(hidden_margin_m))
    output_size = max(64, int(size))
    if axis_label_semantics not in {
        "local_xyz_v1",
        "jaw_lat_app_v2",
        "jaw_lat_app_v3",
    }:
        raise ValueError(
            "axis_label_semantics must be local_xyz_v1, jaw_lat_app_v2, "
            "or jaw_lat_app_v3"
        )
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or colors.shape != points.shape
        or target.shape != (3,)
        or rotation.shape != (3, 3)
        or not np.isfinite(points).all()
        or not np.isfinite(target).all()
        or not np.isfinite(rotation).all()
        or not math.isfinite(radius)
        or not math.isfinite(hidden_margin)
    ):
        raise ValueError(
            "candidate-frame preview requires finite XYZ/RGB and rigid pose"
        )

    local = (points - target) @ rotation
    inside = np.max(np.abs(local), axis=1) <= radius
    local = local[inside]
    local_colors = colors[inside]
    if len(local):
        # A 2 mm voxel is fine enough to retain thin rims and handles while
        # avoiding repeated samples from dominating the canonical projection.
        keys = np.floor((local + radius) / 0.002).astype(np.int64)
        _, unique = np.unique(keys, axis=0, return_index=True)
        unique = np.sort(unique)
        local = local[unique]
        local_colors = local_colors[unique]

    pad_points_world = _finite_pad_points(
        target_pad_contact_centers_world_m
    )
    pad_points_local = (
        (pad_points_world - target) @ rotation
        if pad_points_world is not None
        else None
    )
    sweep_points_world = _finite_pad_points(
        target_pad_sweep_start_centers_world_m
    )
    sweep_points_local = (
        (sweep_points_world - target) @ rotation
        if sweep_points_world is not None
        else None
    )
    def localize_boxes(
        boxes: Sequence[Mapping[str, Any]] | None,
    ) -> list[dict[str, np.ndarray]]:
        localized: list[dict[str, np.ndarray]] = []
        for box in _finite_pad_boxes(boxes):
            localized.append(
                {
                    "center": (box["center_world_m"] - target) @ rotation,
                    "rotation": rotation.T @ box["rotation_world"],
                    "half_size": box["half_size_m"],
                }
            )
        return localized

    def box_corners(box: Mapping[str, np.ndarray]) -> list[np.ndarray]:
        return [
            box["center"]
            + box["rotation"]
            @ (
                box["half_size"]
                * np.asarray((sx, sy, sz), dtype=np.float64)
            )
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]

    local_boxes = localize_boxes(target_pad_boxes)
    local_sweep_start_boxes = localize_boxes(
        target_pad_sweep_start_boxes
    )
    capture_corridor_boxes_world = _finite_pad_boxes(
        target_pad_capture_corridor_boxes
    )
    capture_corridor_faces_world = _actual_pad_inner_contact_faces(
        capture_corridor_boxes_world,
        sweep_points_world,
    )
    capture_corridor_faces_local = [
        [
            (np.asarray(point, dtype=np.float64) - target) @ rotation
            for point in face
        ]
        for face in capture_corridor_faces_world
    ]
    local_box_corners: list[np.ndarray] = []
    for box in (*local_boxes, *local_sweep_start_boxes):
        local_box_corners.extend(box_corners(box))
    corridor_geometry: list[np.ndarray] = list(local_box_corners)
    if pad_points_local is not None:
        corridor_geometry.extend(pad_points_local)
    if sweep_points_local is not None:
        corridor_geometry.extend(sweep_points_local)
    for face in capture_corridor_faces_local:
        corridor_geometry.extend(face)
    corridor_geometry_array = (
        np.asarray(corridor_geometry, dtype=np.float64)
        if corridor_geometry
        else np.zeros((0, 3), dtype=np.float64)
    )

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint_payload = {
        "target_position_xyz_m": target.tolist(),
        "target_rotation_matrix": rotation.tolist(),
        "extent_m": radius,
        "hidden_margin_m": hidden_margin,
        "size": output_size,
        "visual_mode": visual_mode,
    }
    if local_sweep_start_boxes:
        fingerprint_payload["target_pad_sweep_start_boxes"] = [
            {
                "center": box["center"].tolist(),
                "rotation": box["rotation"].tolist(),
                "half_size": box["half_size"].tolist(),
            }
            for box in local_sweep_start_boxes
        ]
    if capture_corridor_boxes_world:
        fingerprint_payload["target_pad_capture_corridor_boxes"] = [
            {
                "center_world_m": box["center_world_m"].tolist(),
                "rotation_world": box["rotation_world"].tolist(),
                "half_size_m": box["half_size_m"].tolist(),
            }
            for box in capture_corridor_boxes_world
        ]
    # Preserve the v1 fingerprint exactly. New semantic-label or renderer
    # modes get a distinct immutable artifact identity.
    if (
        axis_label_semantics != "local_xyz_v1"
        or visual_mode != "current_candidate_frame_v1"
    ):
        fingerprint_payload["axis_label_semantics"] = axis_label_semantics
    if visual_mode != "current_candidate_frame_v1":
        fingerprint_payload["visual_mode"] = visual_mode
    if markable:
        fingerprint_payload["mark_point_mapping"] = (
            "candidate_frame_two_view_v1"
        )
    fingerprint = _artifact_fingerprint(fingerprint_payload)
    rendered: dict[str, Path] = {}
    if axis_label_semantics in {"jaw_lat_app_v2", "jaw_lat_app_v3"}:
        specs = (
            # In the front view, +APP is the direction from the approach
            # reference toward the candidate grip-site. For the common
            # top-down pose this points toward the table/object. Put that
            # positive direction on screen-down so the local preview agrees
            # with the physical front view instead of looking vertically
            # inverted.
            (
                "pointcloud_front",
                0,
                2,
                1,
                "x",
                "z",
                "JAW",
                "APP",
                -1 if axis_label_semantics == "jaw_lat_app_v3" else 1,
            ),
            (
                "pointcloud_side",
                0,
                1,
                2,
                "x",
                "y",
                "JAW",
                "LAT",
                1,
            ),
        )
    else:
        specs = (
            ("pointcloud_front", 0, 2, 1, "x", "z", "X", "Z", 1),
            ("pointcloud_side", 0, 1, 2, "x", "y", "X", "Y", 1),
        )
    for (
        name,
        horizontal,
        vertical,
        hidden,
        h_color_axis,
        v_color_axis,
        h_label,
        v_label,
        vertical_screen_sign,
    ) in specs:
        canvas = np.full(
            (output_size, output_size, 3),
            14,
            dtype=np.uint8,
        )
        grid_step_m = 0.02
        grid_values = np.arange(
            -math.floor(radius / grid_step_m) * grid_step_m,
            radius + grid_step_m * 0.5,
            grid_step_m,
        )
        for value in grid_values:
            pixel = int(
                round((value + radius) / (2.0 * radius) * (output_size - 1))
            )
            if 0 <= pixel < output_size:
                canvas[:, pixel] = (30, 30, 30)
                row = (
                    output_size - 1 - pixel
                    if vertical_screen_sign > 0
                    else pixel
                )
                canvas[row, :] = (30, 30, 30)

        if len(corridor_geometry_array):
            hidden_min = (
                float(np.min(corridor_geometry_array[:, hidden]))
                - hidden_margin
            )
            hidden_max = (
                float(np.max(corridor_geometry_array[:, hidden]))
                + hidden_margin
            )
        else:
            hidden_min, hidden_max = -hidden_margin, hidden_margin
        visible = (
            (local[:, hidden] >= hidden_min)
            & (local[:, hidden] <= hidden_max)
        )
        view_local = local[visible]
        view_colors = local_colors[visible]
        if len(view_local):
            u = np.clip(
                np.rint(
                    (view_local[:, horizontal] + radius)
                    / (2.0 * radius)
                    * (output_size - 1)
                ).astype(int),
                0,
                output_size - 1,
            )
            v = np.clip(
                np.rint(
                        (
                            radius
                            - vertical_screen_sign
                            * view_local[:, vertical]
                        )
                    / (2.0 * radius)
                    * (output_size - 1)
                ).astype(int),
                0,
                output_size - 1,
            )
            pixel = v * output_size + u
            # Prefer the observed surface nearest the candidate's central
            # cross-section. This is deterministic visibility selection, not
            # geometric completion or object inference.
            order = np.lexsort((np.abs(view_local[:, hidden]), pixel))
            sorted_pixels = pixel[order]
            first = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
            chosen = order[first]
            base_u, base_v = u[chosen], v[chosen]
            chosen_colors = np.clip(
                view_colors[chosen].astype(np.float64) * 1.45 + 24.0,
                0.0,
                255.0,
            ).astype(np.uint8)
            for dv, du in (
                (0, 0),
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ):
                uu = np.clip(base_u + du, 0, output_size - 1)
                vv = np.clip(base_v + dv, 0, output_size - 1)
                empty = np.all(canvas[vv, uu] <= 30, axis=1)
                canvas[vv[empty], uu[empty]] = chosen_colors[empty]

        image = Image.fromarray(canvas, mode="RGB")

        def project_local(point: np.ndarray) -> tuple[int, int]:
            return (
                int(
                    round(
                        (float(point[horizontal]) + radius)
                        / (2.0 * radius)
                        * (output_size - 1)
                    )
                ),
                int(
                    round(
                        (
                            radius
                            - vertical_screen_sign * float(point[vertical])
                        )
                        / (2.0 * radius)
                        * (output_size - 1)
                    )
                ),
            )

        # A close preview can optionally show the full collision-box volume
        # swept from nominal open to nominal closed.  This is a translucent
        # geometric overlay only: it does not infer collision, contact, or a
        # grasp.  Pair boxes by finger and take the projected convex hull of
        # both endpoint boxes, which is the exact 2-D projection of a rigid
        # translational sweep along the jaw axis.
        if (
            len(local_boxes) == 2
            and len(local_sweep_start_boxes) == 2
        ):
            sweep_overlay = Image.new(
                "RGBA",
                image.size,
                (0, 0, 0, 0),
            )
            sweep_draw = ImageDraw.Draw(sweep_overlay)
            for start_box, end_box in zip(
                local_sweep_start_boxes,
                local_boxes,
            ):
                sweep_hull = _convex_hull_2d(
                    [
                        project_local(corner)
                        for corner in (
                            *box_corners(start_box),
                            *box_corners(end_box),
                        )
                    ]
                )
                if len(sweep_hull) >= 3:
                    sweep_draw.polygon(
                        sweep_hull,
                        fill=(255, 205, 70, 52),
                        outline=(255, 215, 90, 210),
                        width=2,
                )
            image = Image.alpha_composite(
                image.convert("RGBA"),
                sweep_overlay,
            ).convert("RGB")

        # Treatment-only translucent fills are composited separately so the
        # observed RGB-D surface remains legible beneath the geometry.
        geometry_fill = Image.new("RGBA", image.size, (0, 0, 0, 0))
        geometry_fill_draw = ImageDraw.Draw(geometry_fill)
        draw = ImageDraw.Draw(image)
        if len(capture_corridor_faces_local) == 2:
            projected_faces = [
                [project_local(point) for point in face]
                for face in capture_corridor_faces_local
            ]
            corridor_hull = _convex_hull_2d(
                [pixel for face in projected_faces for pixel in face]
            )
            if len(corridor_hull) >= 3:
                if visual_mode == "candidate_corridor_local_v2":
                    geometry_fill_draw.polygon(
                        corridor_hull,
                        fill=(255, 210, 80, 52),
                    )
                draw.polygon(
                    corridor_hull,
                    outline=(255, 210, 80),
                    width=3,
                )
            for face_pixels in projected_faces:
                face_hull = _convex_hull_2d(face_pixels)
                if len(face_hull) >= 3:
                    draw.polygon(
                        face_hull,
                        outline=(255, 235, 130),
                        width=3,
                    )

        draw = ImageDraw.Draw(image)
        if pad_points_local is not None:
            projected_pads = [
                project_local(point) for point in pad_points_local
            ]
            draw.line(
                (*projected_pads[0], *projected_pads[1]),
                fill=(255, 150, 20),
                width=3,
            )
            for pad_u, pad_v in projected_pads:
                draw.rectangle(
                    (pad_u - 5, pad_v - 5, pad_u + 5, pad_v + 5),
                    outline=(255, 150, 20),
                    width=3,
                )
        if pad_points_local is not None and sweep_points_local is not None:
            for start, end in zip(sweep_points_local, pad_points_local):
                start_u, start_v = project_local(start)
                end_u, end_v = project_local(end)
                draw.line(
                    (start_u, start_v, end_u, end_v),
                    fill=(255, 215, 90),
                    width=2,
                )
                draw.ellipse(
                    (start_u - 4, start_v - 4, start_u + 4, start_v + 4),
                    outline=(255, 215, 90),
                    width=2,
                )
        for box in local_boxes:
            corners: list[tuple[int, int]] = []
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        corner = box["center"] + box["rotation"] @ (
                            box["half_size"]
                            * np.asarray((sx, sy, sz), dtype=np.float64)
                        )
                        corners.append(project_local(corner))
            hull = _convex_hull_2d(corners)
            if len(hull) >= 3:
                if visual_mode == "candidate_corridor_local_v2":
                    geometry_fill_draw.polygon(
                        hull,
                        fill=(255, 150, 20, 44),
                    )
                draw.polygon(hull, outline=(255, 150, 20), width=3)

        if visual_mode == "candidate_corridor_local_v2":
            image = Image.alpha_composite(
                image.convert("RGBA"),
                geometry_fill,
            ).convert("RGB")
            draw = ImageDraw.Draw(image)

        origin = output_size // 2
        draw.ellipse(
            (origin - 6, origin - 6, origin + 6, origin + 6),
            outline=(255, 150, 20),
            width=3,
        )
        # Minimal candidate-local axis glyphs: no title, legend, or prose.
        axis_origin = (
            (18, output_size - 18)
            if vertical_screen_sign > 0
            else (18, 18)
        )
        h_color = AXIS_COLORS[h_color_axis]
        v_color = AXIS_COLORS[v_color_axis]
        draw.line(
            (*axis_origin, axis_origin[0] + 30, axis_origin[1]),
            fill=h_color,
            width=3,
        )
        draw.polygon(
            (
                (axis_origin[0] + 30, axis_origin[1]),
                (axis_origin[0] + 22, axis_origin[1] - 5),
                (axis_origin[0] + 22, axis_origin[1] + 5),
            ),
            fill=h_color,
        )
        draw.text(
            (axis_origin[0] + 34, axis_origin[1] - 8),
            h_label,
            fill=h_color,
            font=_font(10),
        )
        draw.line(
            (
                *axis_origin,
                axis_origin[0],
                axis_origin[1] - 30 * vertical_screen_sign,
            ),
            fill=v_color,
            width=3,
        )
        draw.polygon(
            (
                (
                    axis_origin[0],
                    axis_origin[1] - 30 * vertical_screen_sign,
                ),
                (
                    axis_origin[0] - 5,
                    axis_origin[1] - 22 * vertical_screen_sign,
                ),
                (
                    axis_origin[0] + 5,
                    axis_origin[1] - 22 * vertical_screen_sign,
                ),
            ),
            fill=v_color,
        )
        draw.text(
            (
                axis_origin[0] - 4,
                axis_origin[1]
                - (45 if vertical_screen_sign > 0 else -8),
            ),
            v_label,
            fill=v_color,
            font=_font(10),
        )

        path = output / f"{name}.candidate-frame.{fingerprint}.png"
        image.save(path)
        rendered[name] = path

    metadata_path = output / f"candidate-frame.{fingerprint}.json"
    if axis_label_semantics in {"jaw_lat_app_v2", "jaw_lat_app_v3"}:
        schema_version = (
            "openeta.pose_preview_candidate_frame_views.v3"
            if axis_label_semantics == "jaw_lat_app_v3"
            else "openeta.pose_preview_candidate_frame_views.v2"
        )
        view_metadata = {
            "pointcloud_front": {
                "screen_right": "+JAW",
                "screen_up": (
                    "-APP"
                    if axis_label_semantics == "jaw_lat_app_v3"
                    else "+APP"
                ),
                **(
                    {"screen_down": "+APP"}
                    if axis_label_semantics == "jaw_lat_app_v3"
                    else {}
                ),
                "hidden_axis": "LAT",
            },
            "pointcloud_side": {
                "screen_right": "+JAW",
                "screen_up": "+LAT",
                "hidden_axis": "APP",
            },
        }
        candidate_axes_world = {
            "JAW": rotation[:, 0].tolist(),
            "LAT": rotation[:, 1].tolist(),
            "APP": rotation[:, 2].tolist(),
        }
    else:
        schema_version = "openeta.pose_preview_candidate_frame_views.v1"
        view_metadata = {
            "pointcloud_front": {
                "screen_right": "+X jaw",
                "screen_up": "+Z approach",
                "hidden_axis": "Y",
            },
            "pointcloud_side": {
                "screen_right": "+X jaw",
                "screen_up": "+Y lateral",
                "hidden_axis": "Z",
            },
        }
        candidate_axes_world = None
    metadata = {
        "schema_version": schema_version,
        "frame": "candidate_panda_grip_site",
        "extent_m": radius,
        "hidden_margin_m": hidden_margin,
        "visual_mode": visual_mode,
        "views": view_metadata,
        "visualization_only": True,
    }
    if markable:
        metadata.update(
            {
                "mark_point_mapping": "candidate_frame_two_view_v1",
                "candidate_origin_world_m": target.tolist(),
                "candidate_rotation_world": rotation.tolist(),
                "image_size_xy": [output_size, output_size],
            }
        )
    if local_sweep_start_boxes:
        metadata["closing_sweep_geometry"] = (
            "pad_collision_box_swept_volume_v1"
        )
    if capture_corridor_faces_local:
        metadata["closing_corridor_geometry"] = (
            "open_inner_pad_faces_capture_corridor_v1"
        )
    if candidate_axes_world is not None:
        metadata["candidate_axes_world"] = candidate_axes_world
    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return rendered


def annotate_pending_constraint_views(
    views: Mapping[str, PointCloudView],
    pending: Mapping[str, Any],
    *,
    output_root: str | Path,
    point_id: str = "point",
    marks: Mapping[str, Mapping[str, Any]] | None = None,
    source_paths: Mapping[str, str | Path] | None = None,
    show_visible_surface_marker: bool = False,
) -> dict[str, Path]:
    """Draw the unresolved orthographic constraint in every view.

    A first point click fixes the two coordinates visible in its source
    projection and leaves the hidden coordinate free.  In each complementary
    orthographic view this is therefore rendered as a line (the projected
    ray), rather than as a guessed 3D point.
    """

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    components = pending.get("components_m", {})
    source_view = str(pending.get("view") or "")
    is_camera_ray = str(pending.get("source_kind") or "") == "agentview_camera_ray"
    rendered: dict[str, Path] = {}
    color = (255, 230, 70)
    suffix = _artifact_fingerprint(
        {
            "point_id": point_id,
            "source_kind": pending.get("source_kind"),
            "pixel_xy": pending.get("pixel_xy"),
            "components_m": pending.get("components_m"),
            "origin_xyz_m": pending.get("origin_xyz_m"),
            "direction_xyz": pending.get("direction_xyz"),
        }
    )
    for name, view in views.items():
        source_ref = (source_paths or {}).get(name)
        source = Path(source_ref) if source_ref is not None else view.image_path
        image = Image.open(source).convert("RGB")
        draw = ImageDraw.Draw(image)
        # Only explicitly selected active references are drawn.  The gateway
        # normally passes none while a new point is pending, keeping the
        # constraint ray/line and scene geometry legible.  Callers may still
        # request a small, relevant reference set when needed.
        for point_index, (marked_id, item) in enumerate(
            sorted((marks or {}).items())
        ):
            xyz = np.asarray(item["xyz_m"], dtype=np.float64)
            marked_u, marked_v = project_world_to_view(view, xyz)
            marked_color = POINT_COLORS[point_index % len(POINT_COLORS)]
            draw.ellipse(
                (
                    marked_u - 8,
                    marked_v - 8,
                    marked_u + 8,
                    marked_v + 8,
                ),
                outline=marked_color,
                width=4,
            )
            draw.line(
                (marked_u - 12, marked_v, marked_u + 12, marked_v),
                fill=marked_color,
                width=2,
            )
            draw.line(
                (marked_u, marked_v - 12, marked_u, marked_v + 12),
                fill=marked_color,
                width=2,
            )
            draw.text(
                (marked_u + 10, marked_v + 7),
                marked_id,
                fill=marked_color,
                font=_font(13),
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
        if is_camera_ray:
            origin = np.asarray(pending["origin_xyz_m"], dtype=np.float64)
            direction = np.asarray(pending["direction_xyz"], dtype=np.float64)
            t_range = pending.get("workspace_t_range_m")
            if not (
                isinstance(t_range, (list, tuple))
                and len(t_range) == 2
            ):
                draw.text(
                    (10, 10),
                    f"{point_id} ray misses workspace",
                    fill=color,
                    font=_font(13),
                )
            else:
                start = origin + float(t_range[0]) * direction
                end = origin + float(t_range[1]) * direction
                start_uv = project_world_to_view(view, start)
                end_uv = project_world_to_view(view, end)
                draw.line(
                    (*start_uv, *end_uv),
                    fill=(255, 210, 40),
                    width=7,
                )
                draw.line(
                    (*start_uv, *end_uv),
                    fill=(255, 70, 210),
                    width=3,
                )
                for u, v in (start_uv, end_uv):
                    draw.ellipse(
                        (u - 5, v - 5, u + 5, v + 5),
                        outline=color,
                        width=2,
                    )
                visible_surface = (
                    pending.get("visible_surface")
                    if show_visible_surface_marker
                    else None
                )
                if (
                    isinstance(visible_surface, Mapping)
                    and isinstance(
                        visible_surface.get("xyz_m"),
                        (list, tuple),
                    )
                ):
                    visible_uv = project_world_to_view(
                        view,
                        np.asarray(
                            visible_surface["xyz_m"],
                            dtype=np.float64,
                        ),
                    )
                    draw.ellipse(
                        (
                            visible_uv[0] - 8,
                            visible_uv[1] - 8,
                            visible_uv[0] + 8,
                            visible_uv[1] + 8,
                        ),
                        outline=VISIBLE_SURFACE_MARKER_COLOR,
                        width=4,
                    )
                    draw.ellipse(
                        (
                            visible_uv[0] - 2,
                            visible_uv[1] - 2,
                            visible_uv[0] + 2,
                            visible_uv[1] + 2,
                        ),
                        fill=VISIBLE_SURFACE_MARKER_COLOR,
                    )
                anchor = (
                    max(4, min(view.width - 160, (start_uv[0] + end_uv[0]) // 2 + 8)),
                    max(4, min(view.height - 22, (start_uv[1] + end_uv[1]) // 2 + 8)),
                )
                draw.text(
                    anchor,
                    f"{point_id} agentview ray",
                    fill=color,
                    font=_font(13),
                    stroke_width=2,
                    stroke_fill=(0, 0, 0),
                )
            path = output / f"{name}.pending.{suffix}.png"
            image.save(path)
            rendered[name] = path
            continue
        h_axis = str(view.spec["horizontal_axis"])
        v_axis = str(view.spec["vertical_axis"])
        h_fixed = h_axis in components
        v_fixed = v_axis in components
        if h_fixed:
            h0, h1 = (float(x) for x in view.spec["horizontal_range_m"])
            hu = int(round((float(components[h_axis]) - h0) / (h1 - h0) * (view.width - 1)))
            hu = max(0, min(view.width - 1, hu))
        else:
            hu = None
        if v_fixed:
            v0, v1 = (float(x) for x in view.spec["vertical_range_m"])
            vv = int(round((v1 - float(components[v_axis])) / (v1 - v0) * (view.height - 1)))
            vv = max(0, min(view.height - 1, vv))
        else:
            vv = None
        if hu is not None and vv is not None:
            draw.ellipse((hu - 9, vv - 9, hu + 9, vv + 9), outline=color, width=4)
            draw.line((hu - 14, vv, hu + 14, vv), fill=color, width=2)
            draw.line((hu, vv - 14, hu, vv + 14), fill=color, width=2)
            anchor = (hu + 12, vv + 8)
        elif hu is not None:
            draw.line((hu, 0, hu, view.height - 1), fill=color, width=3)
            anchor = (hu + 8, 10)
        elif vv is not None:
            draw.line((0, vv, view.width - 1, vv), fill=color, width=3)
            anchor = (10, vv + 8)
        else:
            draw.rectangle((2, 2, view.width - 3, view.height - 3), outline=color, width=3)
            anchor = (10, 10)
        label = (
            f"{point_id} ray ({source_view})"
            if name != source_view
            else f"{point_id} click"
        )
        draw.text(anchor, label, fill=color, font=_font(13))
        path = output / f"{name}.pending.{suffix}.png"
        image.save(path)
        rendered[name] = path
    return rendered


def annotate_rejected_point_click(
    source_path: str | Path,
    *,
    output_root: str | Path,
    point_id: str,
    view: str,
    pixel_xy: Sequence[int],
) -> Path:
    """Add one ephemeral, pixel-precise marker for the rejected click."""

    source = Path(source_path)
    image = Image.open(source).convert("RGB")
    if len(pixel_xy) != 2:
        raise ValueError("pixel_xy must contain exactly two coordinates")
    u = max(0, min(image.width - 1, int(pixel_xy[0])))
    v = max(0, min(image.height - 1, int(pixel_xy[1])))
    draw = ImageDraw.Draw(image)
    color = (255, 70, 70)
    draw.ellipse((u - 4, v - 4, u + 4, v + 4), outline=color, width=1)
    draw.point((u, v), fill=(255, 255, 255))
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    suffix = _artifact_fingerprint(
        {
            "point_id": point_id,
            "view": view,
            "pixel_xy": [u, v],
        }
    )
    path = output / f"{view}.rejected-current.{suffix}.png"
    image.save(path)
    return path


def projected_camera_ray_segments(
    views: Mapping[str, PointCloudView],
    ray: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return the exact visible line segment for a calibrated camera ray.

    This exposes only the projection already drawn by
    :func:`annotate_pending_constraint_views`. It does not choose a point on
    the ray or inspect point-cloud geometry.
    """

    if str(ray.get("source_kind") or "") != "agentview_camera_ray":
        return {}
    t_range = ray.get("workspace_t_range_m")
    if not (
        isinstance(t_range, (list, tuple))
        and len(t_range) == 2
    ):
        return {}
    origin = np.asarray(ray["origin_xyz_m"], dtype=np.float64)
    direction = np.asarray(ray["direction_xyz"], dtype=np.float64)
    start = origin + float(t_range[0]) * direction
    end = origin + float(t_range[1]) * direction
    visible_surface = ray.get("visible_surface")
    visible_xyz = (
        np.asarray(visible_surface["xyz_m"], dtype=np.float64)
        if isinstance(visible_surface, Mapping)
        and isinstance(visible_surface.get("xyz_m"), (list, tuple))
        else None
    )
    return {
        name: {
            "view": name,
            "pixel_start_xy": list(project_world_to_view(view, start)),
            "pixel_end_xy": list(project_world_to_view(view, end)),
            **(
                {
                    "visible_surface_pixel_xy": list(
                        project_world_to_view(view, visible_xyz)
                    )
                }
                if visible_xyz is not None
                else {}
            ),
            "displayed_world_axes": [
                str(view.spec["horizontal_axis"]),
                str(view.spec["vertical_axis"]),
            ],
            "world_start_xyz_m": start.tolist(),
            "world_end_xyz_m": end.tolist(),
        }
        for name, view in views.items()
    }


def project_world_to_view(view: PointCloudView, xyz: np.ndarray) -> tuple[int, int]:
    h_axis = AXIS_INDEX[str(view.spec["horizontal_axis"])]
    v_axis = AXIS_INDEX[str(view.spec["vertical_axis"])]
    h0, h1 = (float(x) for x in view.spec["horizontal_range_m"])
    v0, v1 = (float(x) for x in view.spec["vertical_range_m"])
    u = int(round((float(xyz[h_axis]) - h0) / (h1 - h0) * (view.width - 1)))
    v = int(round((v1 - float(xyz[v_axis])) / (v1 - v0) * (view.height - 1)))
    return (
        max(0, min(view.width - 1, u)),
        max(0, min(view.height - 1, v)),
    )


def inspect_world_point_projection(
    view: PointCloudView,
    xyz: np.ndarray,
    *,
    search_radius_px: int = 8,
) -> dict[str, Any]:
    """Describe where one authoritative world point lands in a view.

    This is validation metadata only: it never changes the selected point or
    substitutes another depth sample. ``support`` reports whether the
    projected pixel (or a nearby pixel) contains rendered cloud geometry.
    """

    u, v = project_world_to_view(view, np.asarray(xyz, dtype=np.float64))
    payload = np.load(view.lookup_path)
    valid = np.asarray(payload["valid"], dtype=bool)
    hit = _nearest_valid_pixel(valid, u=u, v=v, radius=search_radius_px)
    return {
        "view": view.name,
        "pixel_xy": [int(u), int(v)],
        "valid_at_projection": bool(valid[v, u]),
        "support": bool(hit is not None),
        "support_pixel_xy": [int(hit[0]), int(hit[1])] if hit is not None else None,
        "search_radius_px": int(search_radius_px),
    }


def _layout_metric_tick_labels(
    draw: Any,
    tick_labels: Sequence[tuple[tuple[int, int], str]],
    *,
    font: Any,
    declutter: bool,
) -> list[tuple[tuple[int, int], str, tuple[int, int, int, int]]]:
    layouts: list[
        tuple[tuple[int, int], str, tuple[int, int, int, int]]
    ] = []
    occupied: list[tuple[int, int, int, int]] = []
    for position, label in tick_labels:
        left, top, right, bottom = draw.textbbox(
            position,
            label,
            font=font,
            stroke_width=1,
        )
        box = (left - 1, top - 1, right + 1, bottom + 1)
        if declutter and any(
            not (
                box[2] < other[0]
                or other[2] < box[0]
                or box[3] < other[1]
                or other[3] < box[1]
            )
            for other in occupied
        ):
            continue
        layouts.append((position, label, box))
        occupied.append(box)
    return layouts


def _orthographic_view(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    spec: Mapping[str, Any],
    size: int,
    grip_site_xyz: list[float] | None = None,
    grip_site_rotation: list[list[float]] | None = None,
    grip_site_aperture_m: float | None = None,
    finger_pad_contact_centers_world_m: list[list[float]] | None = None,
    target_grip_site_xyz: list[float] | None = None,
    metric_edge_ticks: bool = False,
    metric_tick_band_style: str | None = None,
    compact_grip_site_overlay: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    h_axis = AXIS_INDEX[str(spec["horizontal_axis"])]
    v_axis = AXIS_INDEX[str(spec["vertical_axis"])]
    hidden_axis = AXIS_INDEX[str(spec["hidden_axis"])]
    h0, h1 = (float(x) for x in spec["horizontal_range_m"])
    v0, v1 = (float(x) for x in spec["vertical_range_m"])
    inside = (
        (points[:, h_axis] >= h0)
        & (points[:, h_axis] <= h1)
        & (points[:, v_axis] >= v0)
        & (points[:, v_axis] <= v1)
    )
    p = points[inside]
    c = colors[inside]
    u = np.clip(
        np.rint((p[:, h_axis] - h0) / (h1 - h0) * (size - 1)).astype(int),
        0,
        size - 1,
    )
    v = np.clip(
        np.rint((v1 - p[:, v_axis]) / (v1 - v0) * (size - 1)).astype(int),
        0,
        size - 1,
    )
    pixel = v * size + u
    visibility = str(spec["visibility"])
    priority = p[:, hidden_axis]
    if visibility in {"nearest_negative_y", "nearest_negative_x"}:
        priority = -priority
    order = np.lexsort((-priority, pixel))
    sorted_pixels = pixel[order]
    first = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
    chosen = order[first]

    xyz = np.full((size, size, 3), np.nan, dtype=np.float64)
    canvas = np.full((size, size, 3), 14, dtype=np.uint8)
    _draw_grid(canvas, spec)
    base_u, base_v = u[chosen], v[chosen]
    # RGB-D views from LIBERO can be very dark from a side projection. Keep
    # the source hue, but lift the display floor and expand contrast so that
    # thin object contours remain markable. This is a rendering-only change;
    # lookup XYZ and authoritative colors are unchanged.
    chosen_colors = c[chosen].astype(np.float64)
    chosen_colors = np.clip(chosen_colors * 1.75 + 28.0, 0.0, 255.0).astype(
        np.uint8
    )
    for dv, du in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        uu = np.clip(base_u + du, 0, size - 1)
        vv = np.clip(base_v + dv, 0, size - 1)
        empty = ~np.isfinite(xyz[vv, uu, 0])
        xyz[vv[empty], uu[empty]] = p[chosen[empty]]
        canvas[vv[empty], uu[empty]] = chosen_colors[empty]

    if metric_edge_ticks and metric_tick_band_style == "readable_v1":
        # Keep the metric labels legible when bright point samples occupy the
        # first scene row below the title. This changes pixels only; the XYZ
        # lookup remains the unmodified calibrated projection.
        tick_band_end = min(size, 60)
        if tick_band_end > 45:
            canvas[45:tick_band_end] = (
                canvas[45:tick_band_end].astype(np.uint16) // 4
            ).astype(np.uint8)
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    h_name = str(spec["horizontal_axis"]).upper()
    v_name = str(spec["vertical_axis"]).upper()
    title = f"WORLD FRAME  right = +{h_name}  up = +{v_name}"
    draw.rectangle((0, 0, size, 45), fill=(0, 0, 0))
    draw.text((8, 4), title, fill=(255, 255, 255), font=_font(13))
    draw.line((12, 34, 54, 34), fill=AXIS_COLORS[h_name.lower()], width=3)
    draw.polygon(((54, 34), (46, 29), (46, 39)), fill=AXIS_COLORS[h_name.lower()])
    draw.text((59, 25), f"+{h_name}", fill=AXIS_COLORS[h_name.lower()], font=_font(13))
    draw.line((112, 39, 112, 19), fill=AXIS_COLORS[v_name.lower()], width=3)
    draw.polygon(((112, 19), (107, 27), (117, 27)), fill=AXIS_COLORS[v_name.lower()])
    draw.text((121, 25), f"+{v_name}", fill=AXIS_COLORS[v_name.lower()], font=_font(13))
    # Keep metric coordinate cues visible without changing the geometry or
    # adding an opaque panel.  These labels are deliberately tied to the
    # calibrated view ranges, so they remain useful for every task/object.
    if metric_edge_ticks:
        tick_font = _font(9)
        tick_color = (170, 170, 170)
        tick_interval = float(spec.get("grid_interval_m", 0.1))
        tick_labels: list[tuple[tuple[int, int], str]] = []
        for value in np.arange(
            math.ceil(h0 / tick_interval) * tick_interval,
            h1 + tick_interval / 2,
            tick_interval,
        ):
            tick_u = int(round((float(value) - h0) / (h1 - h0) * (size - 1)))
            if 0 <= tick_u < size:
                tick_labels.append(
                    (
                        (max(0, min(size - 28, tick_u - 12)), 46),
                        f"{value:+.1f}",
                    )
                )
        for value in np.arange(
            math.ceil(v0 / tick_interval) * tick_interval,
            v1 + tick_interval / 2,
            tick_interval,
        ):
            tick_v = int(round((v1 - float(value)) / (v1 - v0) * (size - 1)))
            if 0 <= tick_v < size:
                tick_labels.append(
                    (
                        (2, max(46, min(size - 14, tick_v - 5))),
                        f"{value:+.1f}",
                    )
                )
        label_layouts = _layout_metric_tick_labels(
            draw,
            tick_labels,
            font=tick_font,
            declutter=metric_tick_band_style == "readable_v3",
        )
        if metric_tick_band_style in {"readable_v2", "readable_v3"}:
            for _position, _label, box in label_layouts:
                draw.rectangle(
                    box,
                    fill=(0, 0, 0),
                )
        for position, label, _box in label_layouts:
            draw.text(
                position,
                label,
                fill=tick_color,
                font=tick_font,
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
    if target_grip_site_xyz is not None:
        target = np.asarray(target_grip_site_xyz, dtype=np.float64)
        if target.shape == (3,) and np.isfinite(target).all():
            target_u, target_v = _project_spec_pixel(spec, target, size=size)
            if 0 <= target_u < size and 0 <= target_v < size:
                target_color = (60, 255, 120)
                if compact_grip_site_overlay:
                    draw.ellipse(
                        (
                            target_u - 3,
                            target_v - 3,
                            target_u + 3,
                            target_v + 3,
                        ),
                        outline=target_color,
                        width=1,
                    )
                    draw.point((target_u, target_v), fill=target_color)
                else:
                    draw.ellipse(
                        (
                            target_u - 8,
                            target_v - 8,
                            target_u + 8,
                            target_v + 8,
                        ),
                        outline=target_color,
                        width=3,
                    )
                    draw.text(
                        (target_u + 10, target_v - 10),
                        "TARGET",
                        fill=target_color,
                        font=_font(11),
                        stroke_width=2,
                        stroke_fill=(0, 0, 0),
                    )
    if grip_site_xyz is not None:
        grip = np.asarray(grip_site_xyz, dtype=np.float64)
        if grip.shape == (3,) and np.isfinite(grip).all():
            u, v = _project_spec_pixel(spec, grip, size=size)
            if 0 <= u < size and 0 <= v < size:
                grip_color = (255, 0, 255)
                if compact_grip_site_overlay:
                    draw.ellipse(
                        (u - 3, v - 3, u + 3, v + 3),
                        outline=grip_color,
                        width=1,
                    )
                    draw.point((u, v), fill=grip_color)
                else:
                    draw.ellipse(
                        (u - 6, v - 6, u + 6, v + 6),
                        outline=grip_color,
                        width=2,
                    )
                    draw.line(
                        (u - 9, v, u + 9, v),
                        fill=grip_color,
                        width=2,
                    )
                    draw.line(
                        (u, v - 9, u, v + 9),
                        fill=grip_color,
                        width=2,
                    )
                if target_grip_site_xyz is not None:
                    target = np.asarray(target_grip_site_xyz, dtype=np.float64)
                    if target.shape == (3,) and np.isfinite(target).all():
                        target_uv = _project_spec_pixel(
                            spec, target, size=size
                        )
                        draw.line(
                            (u, v, target_uv[0], target_uv[1]),
                            fill=(255, 255, 255),
                            width=2,
                        )
                rotation = np.asarray(grip_site_rotation, dtype=np.float64)
                if rotation.shape == (3, 3) and np.isfinite(rotation).all():
                    pad_points = _finite_pad_points(
                        finger_pad_contact_centers_world_m
                    )
                    if pad_points is None and (
                        isinstance(grip_site_aperture_m, (int, float))
                        and math.isfinite(float(grip_site_aperture_m))
                    ):
                        half_aperture = max(
                            0.0, float(grip_site_aperture_m)
                        ) / 2.0
                        pad_points = np.asarray(
                            (
                                grip - rotation[:, 0] * half_aperture,
                                grip + rotation[:, 0] * half_aperture,
                            ),
                            dtype=np.float64,
                        )
                    if pad_points is not None:
                        pad_color = (255, 110, 40)
                        for index, pad in enumerate(pad_points, start=1):
                            pad_u, pad_v = _project_spec_pixel(
                                spec, pad, size=size
                            )
                            if 0 <= pad_u < size and 0 <= pad_v < size:
                                if compact_grip_site_overlay:
                                    draw.ellipse(
                                        (
                                            pad_u - 2,
                                            pad_v - 2,
                                            pad_u + 2,
                                            pad_v + 2,
                                        ),
                                        outline=pad_color,
                                        width=1,
                                    )
                                    draw.point(
                                        (pad_u, pad_v),
                                        fill=pad_color,
                                    )
                                else:
                                    draw.rectangle(
                                        (
                                            pad_u - 5,
                                            pad_v - 5,
                                            pad_u + 5,
                                            pad_v + 5,
                                        ),
                                        outline=pad_color,
                                        width=3,
                                    )
                                    draw.text(
                                        (pad_u + 7, pad_v + 3),
                                        f"PAD{index}",
                                        fill=pad_color,
                                        font=_font(10),
                                        stroke_width=2,
                                        stroke_fill=(0, 0, 0),
                                    )
                    # Persistent planning views answer where the gripper and
                    # pads are. Full orientation axes belong to explicit pose
                    # preview/inspection, where they can be compared without
                    # obscuring scene geometry on every closed-loop frame.
        if not compact_grip_site_overlay:
            xyz_text = "MAGENTA = current grip_site  XYZ " + " ".join(
                f"{value:+.3f}" for value in grip_site_xyz
            )
            text_y = max(47, size - 22)
            draw.rectangle((0, text_y - 2, size, size), fill=(0, 0, 0))
            draw.text(
                (8, text_y),
                xyz_text,
                fill=(255, 120, 255),
                font=_font(12),
            )
    canvas = np.asarray(image).copy()
    return xyz, canvas


def _draw_grid(canvas: np.ndarray, spec: Mapping[str, Any]) -> None:
    size = canvas.shape[0]
    interval = float(spec.get("grid_interval_m", 0.1))
    h0, h1 = (float(value) for value in spec["horizontal_range_m"])
    v0, v1 = (float(value) for value in spec["vertical_range_m"])
    for value in np.arange(math.ceil(h0 / interval) * interval, h1 + interval / 2, interval):
        u = int(round((value - h0) / (h1 - h0) * (size - 1)))
        if 0 <= u < size:
            canvas[:, u] = (37, 37, 37)
    for value in np.arange(math.ceil(v0 / interval) * interval, v1 + interval / 2, interval):
        v = int(round((v1 - value) / (v1 - v0) * (size - 1)))
        if 0 <= v < size:
            canvas[v, :] = (37, 37, 37)


def _project_spec_pixel(
    spec: Mapping[str, Any], xyz: np.ndarray, *, size: int
) -> tuple[int, int]:
    h_axis = AXIS_INDEX[str(spec["horizontal_axis"])]
    v_axis = AXIS_INDEX[str(spec["vertical_axis"])]
    h0, h1 = (float(value) for value in spec["horizontal_range_m"])
    v0, v1 = (float(value) for value in spec["vertical_range_m"])
    u = int(round((float(xyz[h_axis]) - h0) / (h1 - h0) * (size - 1)))
    v = int(round((v1 - float(xyz[v_axis])) / (v1 - v0) * (size - 1)))
    return u, v


def _observation_grip_site_pose(
    observation_record: Mapping[str, Any],
) -> tuple[list[float] | None, list[list[float]] | None]:
    metadata = observation_record.get("metadata")
    if not isinstance(metadata, Mapping):
        return None, None
    robot = metadata.get("robot")
    if not isinstance(robot, Mapping):
        simulator = metadata.get("simulator_observation")
        robot = simulator.get("robot") if isinstance(simulator, Mapping) else None
    pose = robot.get("end_effector_pose") if isinstance(robot, Mapping) else None
    raw_xyz = pose.get("xyz") if isinstance(pose, Mapping) else None
    try:
        xyz = np.asarray(raw_xyz, dtype=np.float64)
    except (TypeError, ValueError):
        return None, None
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        return None, None
    quat = pose.get("quat_xyzw") if isinstance(pose, Mapping) else None
    rotation = _rotation_matrix_from_quat_xyzw(quat)
    return xyz.tolist(), rotation.tolist() if rotation is not None else None


def _observation_grip_site_xyz(
    observation_record: Mapping[str, Any],
) -> list[float] | None:
    """Compatibility helper for callers that need translation only."""

    return _observation_grip_site_pose(observation_record)[0]


def _observation_gripper_aperture_m(
    observation_record: Mapping[str, Any],
) -> float | None:
    metadata = observation_record.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    robot = metadata.get("robot")
    if not isinstance(robot, Mapping):
        simulator = metadata.get("simulator_observation")
        robot = simulator.get("robot") if isinstance(simulator, Mapping) else None
    state = robot.get("gripper_state") if isinstance(robot, Mapping) else None
    aperture = state.get("aperture_m") if isinstance(state, Mapping) else None
    if (
        not isinstance(aperture, (int, float))
        or not math.isfinite(float(aperture))
    ):
        return None
    return max(0.0, float(aperture))


def _finite_pad_points(value: Any) -> np.ndarray | None:
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if points.shape != (2, 3) or not np.isfinite(points).all():
        return None
    return points


def _finite_pad_boxes(
    value: Any,
) -> list[dict[str, np.ndarray]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    boxes: list[dict[str, np.ndarray]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            continue
        try:
            center = np.asarray(raw["center_world_m"], dtype=np.float64)
            rotation = np.asarray(raw["rotation_world"], dtype=np.float64)
            half_size = np.asarray(raw["half_size_m"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            continue
        if (
            center.shape != (3,)
            or rotation.shape != (3, 3)
            or half_size.shape != (3,)
            or not np.isfinite(center).all()
            or not np.isfinite(rotation).all()
            or not np.isfinite(half_size).all()
            or np.any(half_size <= 0.0)
        ):
            continue
        boxes.append(
            {
                "center_world_m": center,
                "rotation_world": rotation,
                "half_size_m": half_size,
            }
        )
    return boxes


def _actual_pad_inner_contact_faces(
    boxes: Sequence[Mapping[str, np.ndarray]],
    contact_centers: np.ndarray | None,
) -> list[list[np.ndarray]]:
    """Return the two live inner pad faces from collision-box geometry."""

    if len(boxes) != 2 or contact_centers is None:
        return []
    faces: list[list[np.ndarray]] = []
    for box, contact_center in zip(boxes, contact_centers):
        center = np.asarray(box["center_world_m"], dtype=np.float64)
        rotation = np.asarray(box["rotation_world"], dtype=np.float64)
        half_size = np.asarray(box["half_size_m"], dtype=np.float64)
        local_offset = rotation.T @ (
            np.asarray(contact_center, dtype=np.float64) - center
        )
        normal_axis = int(np.argmax(np.abs(local_offset)))
        tangent_axes = [axis for axis in range(3) if axis != normal_axis]
        face: list[np.ndarray] = []
        for first_sign, second_sign in (
            (-1.0, -1.0),
            (-1.0, 1.0),
            (1.0, 1.0),
            (1.0, -1.0),
        ):
            face.append(
                np.asarray(contact_center, dtype=np.float64)
                + rotation[:, tangent_axes[0]]
                * half_size[tangent_axes[0]]
                * first_sign
                + rotation[:, tangent_axes[1]]
                * half_size[tangent_axes[1]]
                * second_sign
            )
        faces.append(face)
    return faces


def _draw_inward_closing_cues(
    draw: ImageDraw.ImageDraw,
    projected_contact_centers: Sequence[tuple[int, int]],
) -> None:
    """Draw two compact arrowheads showing the measured jaw-closing axis."""

    if len(projected_contact_centers) != 2:
        return
    first = np.asarray(projected_contact_centers[0], dtype=np.float64)
    second = np.asarray(projected_contact_centers[1], dtype=np.float64)
    separation = second - first
    distance = float(np.linalg.norm(separation))
    if not math.isfinite(distance) or distance < 18.0:
        return
    unit = separation / distance
    normal = np.asarray((-unit[1], unit[0]), dtype=np.float64)
    shaft_length = min(34.0, max(14.0, distance * 0.28))
    inset = min(7.0, distance * 0.08)
    head_length = min(9.0, shaft_length * 0.38)
    head_half_width = min(6.0, head_length * 0.7)
    color = (255, 245, 120)
    for contact_center, inward in ((first, unit), (second, -unit)):
        start = contact_center + inward * inset
        tip = start + inward * shaft_length
        base = tip - inward * head_length
        draw.line(
            (
                int(round(start[0])),
                int(round(start[1])),
                int(round(base[0])),
                int(round(base[1])),
            ),
            fill=color,
            width=3,
        )
        draw.polygon(
            (
                (int(round(tip[0])), int(round(tip[1]))),
                (
                    int(round(base[0] + normal[0] * head_half_width)),
                    int(round(base[1] + normal[1] * head_half_width)),
                ),
                (
                    int(round(base[0] - normal[0] * head_half_width)),
                    int(round(base[1] - normal[1] * head_half_width)),
                ),
            ),
            fill=color,
        )


def _observation_finger_pad_contact_centers_world_m(
    observation_record: Mapping[str, Any],
) -> list[list[float]] | None:
    metadata = observation_record.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    robot = metadata.get("robot")
    if not isinstance(robot, Mapping):
        simulator = metadata.get("simulator_observation")
        robot = simulator.get("robot") if isinstance(simulator, Mapping) else None
    state = robot.get("gripper_state") if isinstance(robot, Mapping) else None
    geometry = state.get("geometry") if isinstance(state, Mapping) else None
    points = (
        geometry.get("finger_pad_inner_contact_centers_world_m")
        if isinstance(geometry, Mapping)
        else None
    )
    finite = _finite_pad_points(points)
    return finite.tolist() if finite is not None else None


def _rotation_matrix_from_quat_xyzw(value: Any) -> np.ndarray | None:
    try:
        quat = np.asarray(value, dtype=np.float64).reshape(4)
    except (TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm < 1e-12:
        return None
    x, y, z, w = quat / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _draw_pose_axis(
    draw: ImageDraw.ImageDraw,
    *,
    spec: Mapping[str, Any],
    center: np.ndarray,
    direction: np.ndarray,
    size: int,
    color: tuple[int, int, int],
    label: str,
    length_m: float = 0.08,
) -> None:
    start = _project_spec_pixel(spec, center, size=size)
    end = _project_spec_pixel(
        spec,
        center + np.asarray(direction, dtype=np.float64) * float(length_m),
        size=size,
    )
    if start == end:
        return
    draw.line((*start, *end), fill=color, width=4)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    draw.polygon(
        [
            end,
            (end[0] - 10 * ux + 5 * px, end[1] - 10 * uy + 5 * py),
            (end[0] - 10 * ux - 5 * px, end[1] - 10 * uy - 5 * py),
        ],
        fill=color,
    )
    if label:
        draw.text(
            (end[0] + 5, end[1] + 3),
            label,
            fill=color,
            font=_font(11),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )


def _artifact_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:10]


def _convex_hull_2d(
    points: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return a deterministic convex hull for a small projected point set."""

    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique

    def cross(
        origin: tuple[int, int],
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> int:
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower: list[tuple[int, int]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[int, int]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _nearest_valid_pixel(
    valid: np.ndarray, *, u: int, v: int, radius: int
) -> tuple[int, int] | None:
    best: tuple[float, int, int] | None = None
    for vv in range(max(0, v - radius), min(valid.shape[0], v + radius + 1)):
        for uu in range(max(0, u - radius), min(valid.shape[1], u + radius + 1)):
            if not valid[vv, uu]:
                continue
            candidate = (float((uu - u) ** 2 + (vv - v) ** 2), uu, vv)
            if best is None or candidate < best:
                best = candidate
    return None if best is None else (best[1], best[2])


def _font(size: int) -> Any:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()
