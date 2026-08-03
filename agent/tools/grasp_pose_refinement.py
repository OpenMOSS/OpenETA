"""Deterministic, observation-only grasp-pose refinement helpers.

This module deliberately knows nothing about LIBERO object state.  It consumes
one retained RGB-D camera frame, one explicit binary mask, and explicit policy
parameters.  The output pose is already expressed as a world-frame robosuite
Panda ``grip_site`` transform, so downstream rendering and execution do not
need to guess backend axis conventions.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from collections.abc import Mapping

import numpy as np
from PIL import Image

from agent.tools.embodied_perception import ObservationFrame


MaskSide = Literal["x_min", "x_max", "y_min", "y_max"]


class GraspPoseRefinementError(ValueError):
    """Raised when an observation-only pose cannot be derived safely."""


@dataclass(frozen=True, slots=True)
class MaskedWorldCloud:
    """Masked depth samples with retained pixel/world correspondence."""

    points_world: np.ndarray
    pixels_uv: np.ndarray
    world_from_camera_opencv: np.ndarray


@dataclass(frozen=True, slots=True)
class MaskSidePose:
    """One immutable world-frame Panda grip-site pose and its provenance."""

    pose_id: str
    observation_id: str
    frame_id: str
    detection_id: str
    side: MaskSide
    world_from_grip_site: np.ndarray
    diagnostics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pose_id": self.pose_id,
            "observation_id": self.observation_id,
            "frame_id": self.frame_id,
            "detection_id": self.detection_id,
            "side": self.side,
            "pose_frame": "world",
            "eef_frame": "panda_grip_site",
            "transform_world_from_grip_site": self.world_from_grip_site.tolist(),
            "diagnostics": self.diagnostics,
        }


def rigid_transform(value: Any, *, name: str = "transform") -> np.ndarray:
    """Validate and return a finite, right-handed homogeneous transform."""

    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise GraspPoseRefinementError(f"{name} must be a finite 4x4 transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise GraspPoseRefinementError(f"{name} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise GraspPoseRefinementError(f"{name} rotation must be right-handed")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise GraspPoseRefinementError(f"{name} has an invalid homogeneous row")
    return transform


def camera_to_world_opencv(extrinsics: Mapping[str, Any]) -> np.ndarray:
    """Normalize retained camera-to-world extrinsics to OpenCV camera axes."""

    frame_transform = str(extrinsics.get("frame_transform") or "camera_to_world")
    if frame_transform != "camera_to_world":
        raise GraspPoseRefinementError(
            f"unsupported frame_transform: {frame_transform!r}"
        )
    raw_position = extrinsics.get("pos")
    raw_rotation = extrinsics.get("mat")
    try:
        position = np.asarray(raw_position, dtype=np.float64)
        rotation_values = np.asarray(raw_rotation, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GraspPoseRefinementError("invalid camera extrinsics") from exc
    if position.shape != (3,) or not np.isfinite(position).all():
        raise GraspPoseRefinementError("camera extrinsics pos must be finite xyz")
    if rotation_values.shape == (9,):
        rotation = rotation_values.reshape(3, 3)
        if str(extrinsics.get("matrix_layout") or "row_major") == "column_major":
            rotation = rotation.T
    elif rotation_values.shape == (3, 3):
        rotation = rotation_values
    else:
        raise GraspPoseRefinementError("camera extrinsics mat must contain 9 values")
    camera_frame = str(
        extrinsics.get("camera_frame")
        or extrinsics.get("frame_convention")
        or "opengl"
    ).lower()
    if camera_frame in {"opencv", "cv", "pinhole"}:
        world_from_cv_rotation = rotation
    elif camera_frame in {"opengl", "gl"}:
        world_from_cv_rotation = rotation @ np.diag([1.0, -1.0, -1.0])
    else:
        raise GraspPoseRefinementError(
            f"unsupported camera frame: {camera_frame!r}"
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = world_from_cv_rotation
    transform[:3, 3] = position
    return rigid_transform(transform, name="world_from_camera_opencv")


def backproject_masked_world(
    frame: ObservationFrame,
    mask_path: str | Path,
    *,
    depth_truncation_m: float = 3.0,
) -> MaskedWorldCloud:
    """Back-project one retained binary mask into the LIBERO world frame."""

    depth = np.asarray(Image.open(frame.depth_path), dtype=np.float64)
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise GraspPoseRefinementError("depth and mask dimensions differ")
    try:
        fx = float(frame.intrinsics["fx"])
        fy = float(frame.intrinsics["fy"])
        cx = float(frame.intrinsics["cx"])
        cy = float(frame.intrinsics["cy"])
        scale = float(frame.intrinsics.get("scale", 1000.0))
    except (KeyError, TypeError, ValueError) as exc:
        raise GraspPoseRefinementError("invalid camera intrinsics") from exc
    if not all(math.isfinite(value) and value > 0.0 for value in (fx, fy, scale)):
        raise GraspPoseRefinementError("invalid camera intrinsics")
    depth_m = depth / scale
    valid = mask & np.isfinite(depth_m) & (depth_m > 0.0) & (
        depth_m < float(depth_truncation_m)
    )
    rows, columns = np.where(valid)
    if len(rows) == 0:
        raise GraspPoseRefinementError("selected mask has no valid depth samples")
    z = depth_m[rows, columns]
    camera_points = np.stack(
        [
            (columns.astype(np.float64) - cx) * z / fx,
            (rows.astype(np.float64) - cy) * z / fy,
            z,
        ],
        axis=1,
    )
    world_from_camera = camera_to_world_opencv(frame.extrinsics)
    world_points = (
        camera_points @ world_from_camera[:3, :3].T
        + world_from_camera[:3, 3]
    )
    return MaskedWorldCloud(
        points_world=world_points,
        pixels_uv=np.stack([columns, rows], axis=1).astype(np.int32),
        world_from_camera_opencv=world_from_camera,
    )


def derive_mask_side_pose(
    frame: ObservationFrame,
    *,
    mask_path: str | Path,
    detection_id: str,
    side: MaskSide,
    insertion_depth_m: float,
    surface_floor_quantile: float = 0.05,
    top_quantile: float = 0.99,
) -> MaskSidePose:
    """Derive an explicit top-down side pose from masked world geometry.

    ``insertion_depth_m`` is intentionally an exposed policy input.  It moves
    the Panda grip site downward from the robust top surface estimate; this is
    not inferred from object identity and is never silently changed.
    """

    if side not in {"x_min", "x_max", "y_min", "y_max"}:
        raise GraspPoseRefinementError(f"unsupported mask side: {side!r}")
    if not math.isfinite(insertion_depth_m) or not 0.0 <= insertion_depth_m <= 0.20:
        raise GraspPoseRefinementError("insertion_depth_m must be in [0, 0.20]")
    for name, value in (
        ("surface_floor_quantile", surface_floor_quantile),
        ("top_quantile", top_quantile),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise GraspPoseRefinementError(f"{name} must be in [0, 1]")
    if surface_floor_quantile >= top_quantile:
        raise GraspPoseRefinementError(
            "surface_floor_quantile must be smaller than top_quantile"
        )

    cloud = backproject_masked_world(frame, mask_path)
    points = cloud.points_world
    floor_z = float(np.quantile(points[:, 2], surface_floor_quantile))
    surface = points[points[:, 2] >= floor_z]
    if len(surface) < 16:
        raise GraspPoseRefinementError(
            f"too few surface points after quantile filter: {len(surface)}"
        )
    x_min, y_min = np.min(surface[:, :2], axis=0)
    x_max, y_max = np.max(surface[:, :2], axis=0)
    center_x = float((x_min + x_max) / 2.0)
    center_y = float((y_min + y_max) / 2.0)
    top_z = float(np.quantile(surface[:, 2], top_quantile))
    side_spec: dict[MaskSide, tuple[np.ndarray, np.ndarray]] = {
        "x_min": (
            np.asarray([x_min, center_y, top_z - insertion_depth_m]),
            np.asarray([1.0, 0.0, 0.0]),
        ),
        "x_max": (
            np.asarray([x_max, center_y, top_z - insertion_depth_m]),
            np.asarray([-1.0, 0.0, 0.0]),
        ),
        "y_min": (
            np.asarray([center_x, y_min, top_z - insertion_depth_m]),
            np.asarray([0.0, 1.0, 0.0]),
        ),
        "y_max": (
            np.asarray([center_x, y_max, top_z - insertion_depth_m]),
            np.asarray([0.0, -1.0, 0.0]),
        ),
    }
    translation, closing_axis = side_spec[side]
    approach_axis = np.asarray([0.0, 0.0, -1.0])
    lateral_axis = np.cross(approach_axis, closing_axis)
    lateral_axis /= np.linalg.norm(lateral_axis)
    rotation = np.column_stack([closing_axis, lateral_axis, approach_axis])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    transform = rigid_transform(transform, name="world_from_grip_site")

    policy = {
        "method": "mask_side_top_down_v1",
        "side": side,
        "insertion_depth_m": float(insertion_depth_m),
        "surface_floor_quantile": float(surface_floor_quantile),
        "top_quantile": float(top_quantile),
    }
    digest_input = {
        "observation_id": frame.observation_id,
        "frame_id": frame.frame_id,
        "detection_id": detection_id,
        "policy": policy,
        "transform": np.round(transform, 12).tolist(),
    }
    digest = hashlib.sha256(
        json.dumps(digest_input, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    pose_id = f"refined_{side}_{round(insertion_depth_m * 1000):03d}mm_{digest}"
    diagnostics = {
        **policy,
        "mask_ref": str(Path(mask_path).expanduser().resolve()),
        "masked_point_count": int(len(points)),
        "surface_point_count": int(len(surface)),
        "mask_low_quantile_z_m": floor_z,
        "mask_low_quantile_semantics": (
            "selected-mask world-z quantile; not support-plane or collision clearance"
        ),
        "surface_bounds_world_m": {
            "x_min": float(x_min),
            "x_max": float(x_max),
            "y_min": float(y_min),
            "y_max": float(y_max),
        },
        "surface_top_z_m": top_z,
        "translation_world_m": translation.tolist(),
        "closing_axis_world": closing_axis.tolist(),
        "approach_axis_world": approach_axis.tolist(),
    }
    return MaskSidePose(
        pose_id=pose_id,
        observation_id=frame.observation_id,
        frame_id=frame.frame_id,
        detection_id=str(detection_id),
        side=side,
        world_from_grip_site=transform,
        diagnostics=diagnostics,
    )


__all__ = [
    "GraspPoseRefinementError",
    "MaskSide",
    "MaskSidePose",
    "MaskedWorldCloud",
    "backproject_masked_world",
    "camera_to_world_opencv",
    "derive_mask_side_pose",
    "rigid_transform",
]
