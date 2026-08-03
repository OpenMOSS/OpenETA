#!/usr/bin/env python3
"""Compare retained AnyGrasp and GraspGen-X poses in a Viser sidecar.

The viewer is intentionally read-only with respect to the simulator.  It binds
one retained RGB-D observation to immutable proposal files, renders both
backends with the same Panda geometry, and exposes a small local HTTP control
surface for deterministic camera views and screenshots.

Run this script with the GraspGen-X Python 3.11 environment.  It keeps Viser,
trimesh, and model-specific assets out of OpenETA's default environment.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from dataclasses import asdict, dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image


PANDA_GRIP_SITE_OFFSET_M = 0.097
PANDA_MAX_APERTURE_M = 0.08
SUPPORT_CLEARANCE_REQUIRED_MM = 20.0
ANYGRASP_TO_PANDA_AXES = np.asarray(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
BACKEND_COLORS = {
    "anygrasp": (35, 210, 255),
    "graspgenx": (245, 80, 190),
    "refined": (70, 235, 115),
    "target": (35, 225, 255),
    "actual": (255, 205, 35),
}


@dataclass(frozen=True)
class GraspPose:
    pose_id: str
    label: str
    backend: Literal["anygrasp", "graspgenx", "refined", "target", "actual"]
    backend_rank: int
    score: float | None
    world_from_gripper_base: np.ndarray
    world_from_grip_site: np.ndarray
    source_id: str
    branch: str | None = None
    aperture_m: float | None = None
    aperture_semantics: Literal[
        "proposed", "measured", "physical_max", "unknown"
    ] = "unknown"
    aperture_status: Literal[
        "within_robot_limit", "exceeds_robot_limit", "unknown"
    ] = "unknown"
    robot_max_aperture_m: float = PANDA_MAX_APERTURE_M
    viewer_mesh_aperture_m: float = PANDA_MAX_APERTURE_M
    scene_clearance_mm: float | None = None
    support_clearance_mm: float | None = None
    support_clearance_required_mm: float = SUPPORT_CLEARANCE_REQUIRED_MM
    support_clearance_status: Literal["eligible", "unsafe", "unknown"] = "unknown"

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("world_from_gripper_base")
        value.pop("world_from_grip_site")
        value["score_scope"] = f"{self.backend}_only"
        value["renderable"] = True
        value["selectable"] = self.support_clearance_status == "eligible"
        return value


def _finite_aperture_m(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    aperture = float(value)
    return aperture if aperture >= 0.0 else None


def _aperture_status(aperture_m: float | None) -> Literal[
    "within_robot_limit", "exceeds_robot_limit", "unknown"
]:
    if aperture_m is None:
        return "unknown"
    if aperture_m > PANDA_MAX_APERTURE_M + 1e-6:
        return "exceeds_robot_limit"
    return "within_robot_limit"


@dataclass(frozen=True)
class SceneCloud:
    points_world: np.ndarray
    colors_rgb: np.ndarray
    target_points_world: np.ndarray
    target_surface_points_world: np.ndarray
    target_low_points_world: np.ndarray
    center_world: np.ndarray
    radius_m: float
    camera_to_world_opencv: np.ndarray
    support_plane_point_world: np.ndarray | None
    support_plane_normal_world: np.ndarray | None
    support_plane_offset_m: float | None
    diagnostics: dict[str, Any]


def estimate_support_plane(
    points_world: Any,
    *,
    minimum_points: int = 32,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Fit an upward-facing plane and return point, unit normal, and offset.

    The signed-distance convention is ``normal @ point + offset``. Positive
    distances are above the support plane. A small robust refit keeps isolated
    object/edge samples from rotating the local plane estimate.
    """

    points = np.asarray(points_world, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or len(points) < minimum_points
        or not np.isfinite(points).all()
    ):
        return None

    def fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
        center = np.median(values, axis=0)
        _, singular_values, vh = np.linalg.svd(values - center, full_matrices=False)
        if len(singular_values) < 3 or singular_values[1] <= 1e-9:
            return None
        normal = vh[-1]
        normal /= np.linalg.norm(normal)
        if normal[2] < 0.0:
            normal = -normal
        offset = -float(np.dot(normal, center))
        point = center - (float(np.dot(normal, center)) + offset) * normal
        return point, normal, offset

    initial = fit(points)
    if initial is None:
        return None
    _point, normal, offset = initial
    residuals = np.abs(points @ normal + offset)
    median_residual = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median_residual)))
    threshold = max(0.003, median_residual + 3.0 * 1.4826 * mad)
    inliers = points[residuals <= threshold]
    if len(inliers) >= minimum_points:
        refined = fit(inliers)
        if refined is not None:
            return refined
    return initial


def signed_collision_mesh_clearance_mm(
    world_from_gripper_base: Any,
    collision_mesh_vertices: Any,
    *,
    support_plane_normal_world: Any | None,
    support_plane_offset_m: float | None,
) -> float | None:
    """Return the minimum signed collision-mesh distance to a support plane."""

    if support_plane_normal_world is None or support_plane_offset_m is None:
        return None
    transform = _rigid_transform(
        world_from_gripper_base, name="world_from_gripper_base"
    )
    vertices = np.asarray(collision_mesh_vertices, dtype=np.float64)
    normal = np.asarray(support_plane_normal_world, dtype=np.float64)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or len(vertices) == 0
        or not np.isfinite(vertices).all()
        or normal.shape != (3,)
        or not np.isfinite(normal).all()
        or not math.isfinite(float(support_plane_offset_m))
    ):
        return None
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-9:
        return None
    normal = normal / normal_norm
    offset = float(support_plane_offset_m) / normal_norm
    vertices_world = vertices @ transform[:3, :3].T + transform[:3, 3]
    return float(np.min(vertices_world @ normal + offset) * 1000.0)


def pose_with_support_clearance(
    pose: GraspPose,
    collision_mesh_vertices: Any,
    *,
    support_plane_normal_world: Any | None,
    support_plane_offset_m: float | None,
) -> GraspPose:
    clearance_mm = signed_collision_mesh_clearance_mm(
        pose.world_from_gripper_base,
        collision_mesh_vertices,
        support_plane_normal_world=support_plane_normal_world,
        support_plane_offset_m=support_plane_offset_m,
    )
    status: Literal["eligible", "unsafe", "unknown"] = "unknown"
    if clearance_mm is not None:
        status = (
            "eligible"
            if clearance_mm >= SUPPORT_CLEARANCE_REQUIRED_MM
            else "unsafe"
        )
    return replace(
        pose,
        support_clearance_mm=clearance_mm,
        support_clearance_required_mm=SUPPORT_CLEARANCE_REQUIRED_MM,
        support_clearance_status=status,
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in value
    )


def _rigid_transform(value: Any, *, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-3):
        raise ValueError(f"{name} rotation is not right-handed")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous row")
    return transform


def load_scene_cloud(
    sample_dir: Path,
    *,
    expected_observation_id: str | None = None,
    max_scene_points: int = 120_000,
    local_scene_radius_m: float = 0.42,
    seed: int = 17,
) -> SceneCloud:
    """Back-project a GraspGen-X sample into its retained world frame."""

    metadata = _read_json(sample_dir / "meta_data.json")
    source_observation_id = metadata.get("observation_id")
    if (
        expected_observation_id is not None
        and source_observation_id != expected_observation_id
    ):
        raise ValueError(
            "Viser sample observation mismatch: "
            f"expected {expected_observation_id!r}, got {source_observation_id!r}"
        )
    if metadata.get("camera_frame") not in {None, "opencv"}:
        raise ValueError("Viser sample camera_pose must use the OpenCV camera frame")
    rgb = np.asarray(Image.open(sample_dir / "rgb.png").convert("RGB"), dtype=np.uint8)
    depth = np.asarray(np.load(sample_dir / "depth.npy"), dtype=np.float64)
    segmentation = np.asarray(Image.open(sample_dir / "seg.png"), dtype=np.uint8)
    if depth.shape != rgb.shape[:2] or segmentation.shape != depth.shape:
        raise ValueError("RGB, depth, and segmentation dimensions differ")

    intrinsics = np.asarray(metadata.get("intrinsics"), dtype=np.float64)
    camera_to_world = _rigid_transform(
        metadata.get("camera_pose"), name="camera_pose"
    )
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("intrinsics must be a finite 3x3 matrix")

    valid = np.isfinite(depth) & (depth > 0.0)
    rows, columns = np.where(valid)
    z = depth[rows, columns]
    camera_points = np.stack(
        [
            (columns - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (rows - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        ],
        axis=1,
    )
    world_points = (
        camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    )
    colors = rgb[rows, columns]
    target_selector = segmentation[rows, columns] > 0
    target_points = world_points[target_selector]
    if len(target_points) == 0:
        raise ValueError("The retained segmentation has no valid target points")

    # A full camera frustum is actively harmful for close pose inspection:
    # walls and distant surfaces can sit between the deterministic pose camera
    # and the target.  Keep a target-centred local scene so table clearance and
    # nearby obstacles remain visible without turning the view into an opaque
    # shell of unrelated points.
    target_mask_low_quantile_z = float(np.quantile(target_points[:, 2], 0.05))
    target_surface = target_points[
        target_points[:, 2] >= target_mask_low_quantile_z
    ]
    target_low = target_points[
        target_points[:, 2] < target_mask_low_quantile_z
    ]
    focus_center = metadata.get("focus_center_world")
    center = (
        np.asarray(focus_center, dtype=np.float64)
        if isinstance(focus_center, list) and len(focus_center) == 3
        else np.median(target_surface, axis=0)
    )
    radial_distance = np.linalg.norm(world_points[:, :2] - center[:2], axis=1)
    support_candidates = world_points[
        (radial_distance >= 0.06)
        & (radial_distance <= 0.18)
        & (world_points[:, 2] <= target_mask_low_quantile_z + 0.01)
    ]
    support_plane = estimate_support_plane(support_candidates)
    support_plane_point = support_plane[0] if support_plane is not None else None
    support_plane_normal = support_plane[1] if support_plane is not None else None
    support_plane_offset = support_plane[2] if support_plane is not None else None
    local_selector = (
        np.linalg.norm(world_points - center, axis=1) <= local_scene_radius_m
    )
    world_points = world_points[local_selector]
    colors = colors[local_selector]

    if len(world_points) > max_scene_points:
        rng = np.random.default_rng(seed)
        selected = np.sort(
            rng.choice(len(world_points), max_scene_points, replace=False)
        )
        world_points = world_points[selected]
        colors = colors[selected]

    target_radius = float(
        np.linalg.norm(target_surface - center, axis=1).max()
    )
    radius = max(0.15, min(0.45, target_radius * 5.0))
    visible_mask_surface_gap_m = (
        float(
            np.min(
                target_surface @ support_plane_normal + support_plane_offset
            )
        )
        if support_plane_normal is not None and support_plane_offset is not None
        else None
    )
    return SceneCloud(
        points_world=np.asarray(world_points, dtype=np.float32),
        colors_rgb=np.asarray(colors, dtype=np.uint8),
        target_points_world=np.asarray(target_points, dtype=np.float32),
        target_surface_points_world=np.asarray(target_surface, dtype=np.float32),
        target_low_points_world=np.asarray(target_low, dtype=np.float32),
        center_world=np.asarray(center, dtype=np.float64),
        radius_m=radius,
        camera_to_world_opencv=camera_to_world,
        support_plane_point_world=(
            np.asarray(support_plane_point, dtype=np.float64)
            if support_plane_point is not None
            else None
        ),
        support_plane_normal_world=(
            np.asarray(support_plane_normal, dtype=np.float64)
            if support_plane_normal is not None
            else None
        ),
        support_plane_offset_m=(
            float(support_plane_offset)
            if support_plane_offset is not None
            else None
        ),
        diagnostics={
            "scene_mode": metadata.get("scene_mode", "proposal"),
            "source_observation_id": source_observation_id,
            "source_frame_id": metadata.get("frame_id"),
            "source_action_id": metadata.get("action_id"),
            "camera_frame": metadata.get("camera_frame", "opencv"),
            "target_point_count": int(len(target_points)),
            "target_surface_point_count": int(len(target_surface)),
            "target_low_point_count": int(len(target_low)),
            "target_mask_low_quantile": 0.05,
            "target_mask_low_quantile_z_m": target_mask_low_quantile_z,
            "target_top_z_m": float(np.max(target_points[:, 2])),
            "support_plane_status": (
                "estimated" if support_plane is not None else "unknown"
            ),
            "support_plane_point_world": (
                support_plane_point.tolist()
                if support_plane_point is not None
                else None
            ),
            "support_plane_normal_world": (
                support_plane_normal.tolist()
                if support_plane_normal is not None
                else None
            ),
            "support_plane_offset_m": support_plane_offset,
            "support_plane_candidate_count": int(len(support_candidates)),
            "visible_mask_surface_gap_above_support_m": (
                visible_mask_surface_gap_m
            ),
            "visible_mask_surface_gap_semantics": (
                "minimum signed distance from mask-filtered target samples; "
                "not robot collision clearance"
            ),
        },
    )


def load_graspgenx_poses(
    results_path: Path,
    *,
    proposal_set_id: str,
    top_k: int | None = None,
) -> list[GraspPose]:
    payload = _read_json(results_path)
    raw_grasps = payload.get("grasps")
    if not isinstance(raw_grasps, list):
        raise ValueError("GraspGen-X results have no grasps list")
    output: list[GraspPose] = []
    for fallback_rank, item in enumerate(raw_grasps):
        if top_k is not None and len(output) >= top_k:
            break
        if not isinstance(item, dict):
            continue
        rank = int(item.get("rank", fallback_rank))
        base = _rigid_transform(
            item.get("transform_world_from_gripper"),
            name=f"GraspGen-X rank {rank}",
        )
        grip_site_value = item.get("transform_world_from_grip_site")
        if grip_site_value is None:
            grip_site = base.copy()
            grip_site[:3, 3] += PANDA_GRIP_SITE_OFFSET_M * base[:3, 2]
        else:
            grip_site = _rigid_transform(
                grip_site_value, name=f"GraspGen-X grip site rank {rank}"
            )
        proposed_aperture = _finite_aperture_m(item.get("width"))
        aperture = (
            proposed_aperture
            if proposed_aperture is not None
            else PANDA_MAX_APERTURE_M
        )
        output.append(
            GraspPose(
                pose_id=f"{proposal_set_id}/graspgenx/{rank:03d}",
                label=f"GX{rank}",
                backend="graspgenx",
                backend_rank=rank,
                score=float(item.get("score", 0.0)),
                world_from_gripper_base=base,
                world_from_grip_site=grip_site,
                source_id=str(item.get("id", f"graspgenx_{rank:03d}")),
                branch=str(item["branch"]) if item.get("branch") is not None else None,
                aperture_m=aperture,
                aperture_semantics=(
                    "proposed" if proposed_aperture is not None else "physical_max"
                ),
                aperture_status=_aperture_status(aperture),
                scene_clearance_mm=(
                    float(item["scene_clearance_mm"])
                    if item.get("scene_clearance_mm") is not None
                    else None
                ),
            )
        )
    return output


def load_graspgenx_prefilter_poses(
    prefilter_path: Path,
    *,
    proposal_set_id: str,
    raw_ranks: list[int],
) -> list[GraspPose]:
    """Load explicit collision-eligible raw ranks without score substitution."""

    payload = _read_json(prefilter_path)
    values = payload.get("grasps")
    if not isinstance(values, list):
        raise ValueError("GraspGen-X prefilter has no grasps list")
    by_rank = {
        int(item["raw_rank"]): item
        for item in values
        if isinstance(item, dict) and item.get("raw_rank") is not None
    }
    output: list[GraspPose] = []
    for raw_rank in raw_ranks:
        if raw_rank not in by_rank:
            raise ValueError(f"Unknown GraspGen-X raw rank: {raw_rank}")
        item = by_rank[raw_rank]
        if item.get("collision_free") is not True:
            raise ValueError(
                f"GraspGen-X raw rank {raw_rank} is not collision eligible"
            )
        base = _rigid_transform(
            item.get("transform_world_from_gripper"),
            name=f"GraspGen-X raw rank {raw_rank}",
        )
        grip_site = base.copy()
        grip_site[:3, 3] += PANDA_GRIP_SITE_OFFSET_M * base[:3, 2]
        proposed_aperture = _finite_aperture_m(item.get("width"))
        aperture = (
            proposed_aperture
            if proposed_aperture is not None
            else PANDA_MAX_APERTURE_M
        )
        output.append(
            GraspPose(
                pose_id=f"{proposal_set_id}/graspgenx/raw-{raw_rank:03d}",
                label=f"GXr{raw_rank}",
                backend="graspgenx",
                backend_rank=raw_rank,
                score=float(item.get("score", 0.0)),
                world_from_gripper_base=base,
                world_from_grip_site=grip_site,
                source_id=f"graspgenx_raw_{raw_rank:03d}",
                branch=str(item["branch"]) if item.get("branch") is not None else None,
                aperture_m=aperture,
                aperture_semantics=(
                    "proposed" if proposed_aperture is not None else "physical_max"
                ),
                aperture_status=_aperture_status(aperture),
                scene_clearance_mm=(
                    float(item["scene_clearance_m"]) * 1000.0
                    if item.get("scene_clearance_m") is not None
                    else None
                ),
            )
        )
    return output


def load_anygrasp_poses(
    response_path: Path,
    *,
    camera_to_world_opencv: np.ndarray,
    proposal_set_id: str,
    top_k: int | None = None,
    expected_result_id: str | None = None,
) -> list[GraspPose]:
    """Convert canonical camera-frame candidates to Panda execution geometry.

    The supervisor supplies ``grasp_candidates.canonical.json``.  The legacy
    ``details.grasp_candidates`` shape remains readable for standalone use, but
    raw backend response ordering is never selected by the supervisor.
    """

    payload = _read_json(response_path)
    canonical = payload.get("grasp_candidates")
    if isinstance(canonical, list):
        schema_version = payload.get("schema_version")
        if schema_version != "openeta.canonical_grasp_candidates.v1":
            raise ValueError(
                "AnyGrasp canonical artifact has an unsupported schema_version"
            )
        result_id = payload.get("result_id")
        if expected_result_id is not None and result_id != expected_result_id:
            raise ValueError(
                "AnyGrasp canonical artifact result identity mismatch: "
                f"expected {expected_result_id!r}, got {result_id!r}"
            )
        raw_grasps = canonical
    else:
        details = payload.get("details", {})
        raw_grasps = (
            details.get("grasp_candidates") if isinstance(details, dict) else None
        )
    if not isinstance(raw_grasps, list):
        raise ValueError("AnyGrasp artifact has no grasp_candidates list")
    camera_to_world = _rigid_transform(
        camera_to_world_opencv, name="camera_to_world_opencv"
    )
    output: list[GraspPose] = []
    for fallback_rank, item in enumerate(raw_grasps):
        if top_k is not None and len(output) >= top_k:
            break
        if not isinstance(item, dict):
            continue
        if item.get("frame") != "camera" or item.get("camera_frame") != "opencv":
            raise ValueError("AnyGrasp candidate must be in the OpenCV camera frame")
        rank = int(item.get("rank", fallback_rank))
        camera_rotation = np.asarray(item.get("rotation_matrix"), dtype=np.float64)
        camera_translation = np.asarray(item.get("translation_xyz"), dtype=np.float64)
        if camera_rotation.shape != (3, 3) or camera_translation.shape != (3,):
            raise ValueError(f"AnyGrasp rank {rank} has malformed pose fields")
        world_from_anygrasp = camera_to_world[:3, :3] @ camera_rotation
        world_from_panda = world_from_anygrasp @ ANYGRASP_TO_PANDA_AXES
        grip_site_position = (
            camera_to_world[:3, :3] @ camera_translation
            + camera_to_world[:3, 3]
        )
        grip_site = np.eye(4, dtype=np.float64)
        grip_site[:3, :3] = world_from_panda
        grip_site[:3, 3] = grip_site_position
        grip_site = _rigid_transform(grip_site, name=f"AnyGrasp grip site rank {rank}")
        base = grip_site.copy()
        base[:3, 3] -= PANDA_GRIP_SITE_OFFSET_M * grip_site[:3, 2]
        proposed_aperture = _finite_aperture_m(item.get("width"))
        output.append(
            GraspPose(
                pose_id=f"{proposal_set_id}/anygrasp/{rank:03d}",
                label=f"AG{rank}",
                backend="anygrasp",
                backend_rank=rank,
                score=float(item.get("score", 0.0)),
                world_from_gripper_base=base,
                world_from_grip_site=grip_site,
                source_id=str(item.get("id", f"grasp_{rank:03d}")),
                aperture_m=proposed_aperture,
                aperture_semantics=(
                    "proposed" if proposed_aperture is not None else "unknown"
                ),
                aperture_status=_aperture_status(proposed_aperture),
            )
        )
    return output


def load_execution_poses(
    comparison_path: Path,
    *,
    proposal_set_id: str,
) -> list[GraspPose]:
    """Load exact TARGET/ACTUAL world-frame Panda grip-site transforms."""

    payload = _read_json(comparison_path)
    proposed_aperture = _finite_aperture_m(
        payload.get("proposed_jaw_width_m", payload.get("jaw_width_m"))
    )
    measured_aperture = _finite_aperture_m(payload.get("measured_aperture_m"))
    robot_max_aperture = (
        _finite_aperture_m(payload.get("robot_max_aperture_m"))
        or PANDA_MAX_APERTURE_M
    )
    viewer_mesh_aperture = (
        _finite_aperture_m(payload.get("viewer_mesh_aperture_m"))
        or PANDA_MAX_APERTURE_M
    )
    output: list[GraspPose] = []
    for backend, label, key in (
        ("target", "TARGET", "target_world_from_grip_site"),
        ("actual", "ACTUAL", "actual_world_from_grip_site"),
    ):
        grip_site = _rigid_transform(payload.get(key), name=key)
        base = grip_site.copy()
        base[:3, 3] -= PANDA_GRIP_SITE_OFFSET_M * grip_site[:3, 2]
        aperture = proposed_aperture if backend == "target" else measured_aperture
        output.append(
            GraspPose(
                pose_id=f"{proposal_set_id}/{backend}",
                label=label,
                backend=backend,  # type: ignore[arg-type]
                backend_rank=0,
                score=None,
                world_from_gripper_base=base,
                world_from_grip_site=grip_site,
                source_id=str(payload.get("source_grasp_id") or backend),
                branch="commanded" if backend == "target" else "observed",
                aperture_m=aperture,
                aperture_semantics=(
                    "proposed"
                    if backend == "target" and aperture is not None
                    else "measured"
                    if backend == "actual" and aperture is not None
                    else "unknown"
                ),
                aperture_status=_aperture_status(aperture),
                robot_max_aperture_m=robot_max_aperture,
                viewer_mesh_aperture_m=viewer_mesh_aperture,
            )
        )
    return output


def camera_preset(
    preset: str,
    *,
    target_center: np.ndarray,
    scene_radius_m: float,
    focus_pose: GraspPose | None = None,
) -> dict[str, np.ndarray | float]:
    """Return a deterministic Viser camera position/look-at/up triple."""

    center = np.asarray(target_center, dtype=np.float64)
    distance = max(0.3, float(scene_radius_m) * 2.2)
    if preset == "scene":
        direction = np.asarray([1.0, -1.0, 0.72])
        look_at = center
        up = np.asarray([0.0, 0.0, 1.0])
    elif preset == "top":
        direction = np.asarray([0.0, 0.0, 1.0])
        look_at = center
        up = np.asarray([0.0, 1.0, 0.0])
    elif preset == "front":
        direction = np.asarray([0.0, -1.0, 0.25])
        look_at = center
        up = np.asarray([0.0, 0.0, 1.0])
    elif preset == "side":
        direction = np.asarray([1.0, 0.0, 0.25])
        look_at = center
        up = np.asarray([0.0, 0.0, 1.0])
    elif preset in {"pose_approach", "pose_jaws"}:
        if focus_pose is None:
            raise ValueError(f"{preset} requires focus_pose")
        pose = focus_pose.world_from_grip_site
        look_at = pose[:3, 3]
        if preset == "pose_approach":
            direction = -pose[:3, 2]
            up = pose[:3, 1]
        else:
            direction = pose[:3, 1]
            # A top-down Panda pose has local +Z pointing toward world down.
            # Using that as the camera up vector turns the whole scene upside
            # down and makes orbit controls feel pole-limited.  Keep jaws views
            # world-upright while removing any component parallel to the view.
            world_up = np.asarray([0.0, 0.0, 1.0])
            up = world_up - np.dot(world_up, direction) * direction
            if np.linalg.norm(up) < 1e-8:
                up = pose[:3, 0]
        distance = max(0.34, float(scene_radius_m) * 2.0)
    else:
        raise ValueError(f"Unknown camera preset: {preset}")
    direction = direction / np.linalg.norm(direction)
    up = up / np.linalg.norm(up)
    return {
        "position": look_at + distance * direction,
        "look_at": look_at,
        "up_direction": up,
        "fov": math.radians(50.0),
    }


def orbit_camera_state(
    position: Any,
    look_at: Any,
    *,
    azimuth_delta_deg: float = 0.0,
    elevation_delta_deg: float = 0.0,
    zoom_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Orbit one camera around its look-at point using world-up spherical axes."""

    camera = np.asarray(position, dtype=np.float64)
    focus = np.asarray(look_at, dtype=np.float64)
    if camera.shape != (3,) or focus.shape != (3,):
        raise ValueError("camera position and look_at must be xyz vectors")
    values = (azimuth_delta_deg, elevation_delta_deg, zoom_scale)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("orbit values must be finite")
    if not 0.2 <= float(zoom_scale) <= 5.0:
        raise ValueError("zoom_scale must be in [0.2, 5.0]")
    relative = camera - focus
    radius = float(np.linalg.norm(relative))
    if radius <= 1e-8:
        raise ValueError("camera is at its look_at point")
    azimuth = math.atan2(relative[1], relative[0]) + math.radians(
        float(azimuth_delta_deg)
    )
    elevation = math.asin(max(-1.0, min(1.0, relative[2] / radius)))
    elevation += math.radians(float(elevation_delta_deg))
    elevation = max(math.radians(-88.0), min(math.radians(88.0), elevation))
    radius *= float(zoom_scale)
    cosine = math.cos(elevation)
    new_position = focus + radius * np.asarray(
        [
            cosine * math.cos(azimuth),
            cosine * math.sin(azimuth),
            math.sin(elevation),
        ]
    )
    view = focus - new_position
    view /= np.linalg.norm(view)
    world_up = np.asarray([0.0, 0.0, 1.0])
    up = world_up - np.dot(world_up, view) * view
    if np.linalg.norm(up) < 1e-6:
        up = np.asarray([0.0, 1.0, 0.0])
    up /= np.linalg.norm(up)
    return {"position": new_position, "look_at": focus, "up_direction": up}


def _plane_frame(normal_world: Any) -> np.ndarray:
    normal = np.asarray(normal_world, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    reference = (
        np.asarray([1.0, 0.0, 0.0])
        if abs(float(normal[0])) < 0.9
        else np.asarray([0.0, 1.0, 0.0])
    )
    axis_y = np.cross(normal, reference)
    axis_y /= np.linalg.norm(axis_y)
    axis_x = np.cross(axis_y, normal)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack([axis_x, axis_y, normal])
    return transform


class GraspPoseViewer:
    def __init__(
        self,
        *,
        scene: SceneCloud,
        poses: list[GraspPose],
        proposal_set_id: str,
        episode_root: Path,
        observation_id: str,
        host: str,
        port: int,
        artifact_root: Path,
        visual_mesh: Any,
        collision_mesh: Any,
        scene_mode: str = "proposal",
        proposal_id: str | None = None,
        result_id: str | None = None,
    ) -> None:
        import viser
        from graspgenx.utils.viser_utils import matrix_to_wxyz_position

        if not poses:
            raise ValueError("At least one grasp pose is required")
        self._viser = viser
        self._matrix_to_wxyz_position = matrix_to_wxyz_position
        self.scene = scene
        self.proposal_set_id = proposal_set_id
        self.episode_root = episode_root.expanduser().resolve()
        self.observation_id = str(observation_id)
        self.artifact_root = artifact_root
        self.scene_mode = str(scene_mode)
        self.proposal_id = str(proposal_id) if proposal_id is not None else None
        self.result_id = str(result_id) if result_id is not None else None
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.server = viser.ViserServer(
            host=host, port=port, label="OpenETA grasp inspector", verbose=True
        )
        self.lock = threading.RLock()
        self.render_lock = threading.Lock()
        self.revision = 1
        self.scope = "default"
        self.show_backend = {
            "anygrasp": True,
            "graspgenx": True,
            "refined": True,
            "target": True,
            "actual": True,
        }
        self.focus_pose_id = poses[0].pose_id
        self.clients: dict[int, Any] = {}
        self.pose_handles: dict[str, list[Any]] = {}
        self.capture_index = 0

        self.mesh_vertices = np.asarray(visual_mesh.vertices, dtype=np.float32)
        self.mesh_faces = np.asarray(visual_mesh.faces, dtype=np.uint32)
        self.collision_mesh_vertices = np.asarray(
            collision_mesh.vertices, dtype=np.float64
        )
        self.poses = [
            pose_with_support_clearance(
                pose,
                self.collision_mesh_vertices,
                support_plane_normal_world=scene.support_plane_normal_world,
                support_plane_offset_m=scene.support_plane_offset_m,
            )
            for pose in poses
        ]
        self.pose_by_id = {pose.pose_id: pose for pose in self.poses}

        self.server.scene.set_up_direction("+z")
        self.scene_cloud_handle = self.server.scene.add_point_cloud(
            "/scene/rgbd",
            points=scene.points_world,
            colors=scene.colors_rgb,
            point_size=0.0012,
            point_shape="rounded",
        )
        self.target_surface_handle = self.server.scene.add_point_cloud(
            "/scene/target_surface",
            points=scene.target_surface_points_world,
            colors=(255, 185, 35),
            point_size=0.0020,
            point_shape="rounded",
        )
        self.target_low_handle = self.server.scene.add_point_cloud(
            "/scene/target_low_mask_points",
            points=scene.target_low_points_world,
            colors=(255, 70, 70),
            point_size=0.0024,
            point_shape="rounded",
        )
        support_known = (
            scene.support_plane_point_world is not None
            and scene.support_plane_normal_world is not None
            and scene.support_plane_offset_m is not None
        )
        self.support_plane_known = support_known
        support_point = (
            np.asarray(scene.support_plane_point_world, dtype=np.float64)
            if support_known
            else np.asarray(
                [
                    scene.center_world[0],
                    scene.center_world[1],
                    float(np.quantile(scene.target_points_world[:, 2], 0.10)),
                ],
                dtype=np.float64,
            )
        )
        support_normal = (
            np.asarray(scene.support_plane_normal_world, dtype=np.float64)
            if support_known
            else np.asarray([0.0, 0.0, 1.0])
        )
        support_frame = _plane_frame(support_normal)
        support_frame[:3, 3] = support_point + 0.003 * support_normal
        support_wxyz, support_position = self._matrix_to_wxyz_position(
            support_frame
        )
        self.support_grid_handle = self.server.scene.add_grid(
            "/scene/estimated_support_plane",
            width=0.30,
            height=0.30,
            plane="xy",
            cell_color=(85, 170, 255),
            cell_thickness=0.7,
            cell_size=0.02,
            section_color=(45, 115, 230),
            section_thickness=1.3,
            section_size=0.10,
            plane_color=(80, 145, 245),
            plane_opacity=0.10,
            wxyz=support_wxyz,
            position=support_position,
        )
        if support_known:
            target_signed_distances = (
                scene.target_surface_points_world @ support_normal
                + float(scene.support_plane_offset_m)
            )
            visible_center = np.asarray(
                scene.target_surface_points_world[
                    int(np.argmin(target_signed_distances))
                ],
                dtype=np.float32,
            )
            support_center = np.asarray(
                visible_center
                - float(np.min(target_signed_distances)) * support_normal,
                dtype=np.float32,
            )
        else:
            support_center = np.asarray(support_point, dtype=np.float32)
            visible_center = support_center.copy()
        self.support_gap_handle = self.server.scene.add_line_segments(
            "/scene/visible_surface_gap",
            points=np.asarray([[support_center, visible_center]], dtype=np.float32),
            colors=(255, 215, 70),
            line_width=2.0,
        )
        self.support_label_handle = self.server.scene.add_label(
            "/scene/support_label",
            text=(
                (
                    "estimated support plane | mask-filtered visible surface gap "
                    f"{float(scene.diagnostics['visible_mask_surface_gap_above_support_m']) * 1000:.1f}mm | "
                    "pose labels show collision-mesh clearance"
                )
                if support_known
                else "support plane unknown | autonomous grasp execution disabled"
            ),
            position=support_center,
            font_screen_scale=0.75,
            depth_test=False,
            anchor="top-center",
        )
        self.server.scene.add_frame(
            "/scene/world",
            axes_length=0.12,
            axes_radius=0.004,
        )
        self.support_grid_handle.visible = support_known
        self.support_gap_handle.visible = support_known
        for pose in self.poses:
            self._add_pose_node(pose)

        if self.scene_mode == "execution":
            # The execution exporter marks a geometric focus neighbourhood,
            # not a semantic object mask.  Proposal-style orange/red emphasis
            # would recolor the robot, table, and nearby objects and obscure
            # the exact cyan/yellow TARGET/ACTUAL comparison.
            self.target_surface_handle.visible = False
            self.target_low_handle.visible = False
            self.support_gap_handle.visible = False
            self.support_label_handle.visible = False

        self._install_gui()
        self._apply_visibility(increment=False)

        self._install_client_tracking()

        self._write_state()

    def _install_client_tracking(self) -> None:
        """Track browser clients, including ones connected during scene setup.

        Chrome can finish its websocket handshake while a large point cloud and
        pose set are still being added. Viser does not replay that earlier
        connection when the callback is registered later, so bootstrap from
        ``get_clients()`` after installing the callbacks.
        """

        @self.server.on_client_connect
        def _on_client(client: Any) -> None:
            with self.lock:
                self.clients[int(client.client_id)] = client
                self._set_client_camera(client, "scene", None)
                self._write_state()

        @self.server.on_client_disconnect
        def _on_disconnect(client: Any) -> None:
            with self.lock:
                self.clients.pop(int(client.client_id), None)
                self._write_state()

        for client in self.server.get_clients().values():
            _on_client(client)

    def _sync_connected_clients(self) -> None:
        """Refresh client handles from Visers authoritative connection map."""

        server = getattr(self, "server", None)
        if server is None or not hasattr(server, "get_clients"):
            return
        connected = server.get_clients()
        with self.lock:
            self.clients = {
                int(client_id): client for client_id, client in connected.items()
            }

    def _add_pose_node(self, pose: GraspPose) -> None:
        color = BACKEND_COLORS[pose.backend]
        wxyz, position = self._matrix_to_wxyz_position(
            pose.world_from_gripper_base.astype(float)
        )
        root = f"/grasps/{pose.backend}/{pose.label}"
        mesh_handle = self.server.scene.add_mesh_simple(
            f"{root}/panda",
            vertices=self.mesh_vertices,
            faces=self.mesh_faces,
            color=color,
            opacity=0.68 if pose.backend == "refined" else 0.62,
            side="double",
            flat_shading=False,
            wxyz=wxyz,
            position=position,
        )
        frame_handle = self.server.scene.add_frame(
            f"{root}/axes",
            axes_length=0.055,
            axes_radius=0.002,
            wxyz=wxyz,
            position=position,
        )
        grip_position = pose.world_from_grip_site[:3, 3]
        aperture_handle = None
        if pose.aperture_m is not None:
            closing_axis = pose.world_from_grip_site[:3, 0]
            jaw_a = grip_position + closing_axis * pose.aperture_m * 0.5
            jaw_b = grip_position - closing_axis * pose.aperture_m * 0.5
            aperture_handle = self.server.scene.add_line_segments(
                f"{root}/aperture_guide",
                points=np.asarray([[jaw_a, jaw_b]], dtype=np.float32),
                colors=(
                    (255, 70, 70)
                    if pose.aperture_status == "exceeds_robot_limit"
                    else color
                ),
                line_width=4.0,
            )
        details = pose.label
        if pose.aperture_m is not None:
            details += (
                f"  {pose.aperture_semantics}-aperture "
                f"{pose.aperture_m * 1000.0:.1f}mm"
            )
            if pose.aperture_status == "exceeds_robot_limit":
                details += (
                    f" > Panda {pose.robot_max_aperture_m * 1000.0:.0f}mm"
                )
        else:
            details += "  aperture unknown"
        details += (
            f"  mesh=physical-open {pose.viewer_mesh_aperture_m * 1000.0:.0f}mm"
        )
        if self.scene_mode != "execution":
            if pose.score is not None:
                details += f"  {pose.score:.4f}"
            if pose.scene_clearance_mm is not None:
                details += f"  backend-clr {pose.scene_clearance_mm:.1f}mm"
            if pose.support_clearance_mm is not None:
                details += (
                    f"  support {pose.support_clearance_mm:.1f}mm"
                    f"/{pose.support_clearance_required_mm:.0f}mm"
                )
            else:
                details += "  support unknown"
            details += f"  {pose.support_clearance_status}"
            if pose.branch:
                details += f"  {pose.branch}"
        label_handle = self.server.scene.add_label(
            f"{root}/label",
            text=details,
            position=grip_position,
            font_screen_scale=0.58 if self.scene_mode == "execution" else 0.9,
            depth_test=False,
            anchor="bottom-center",
        )
        self.pose_handles[pose.pose_id] = [
            handle
            for handle in (mesh_handle, frame_handle, aperture_handle, label_handle)
            if handle is not None
        ]

    def _install_gui(self) -> None:
        if self.scene_mode == "execution":
            self.server.gui.add_markdown(
                "# OpenETA execution inspector\n"
                "Cyan = commanded TARGET, yellow = simulator-reported ACTUAL. "
                "Each colored aperture guide is numeric: TARGET uses the proposed "
                "width and ACTUAL uses the measured simulator aperture. The Panda "
                "mesh itself stays at the physical-open 80 mm geometry. "
                "The point cloud is reconstructed from the exact post-action "
                "RGB-D observation. Viewing never moves the robot."
            )
        else:
            self.server.gui.add_markdown(
                "# OpenETA grasp inspector\n"
                "Cyan = AnyGrasp, magenta = GraspGen-X, green = explicit refined "
                "Panda grip-site pose. Scores are backend-local. "
                "A colored jaw line shows the proposal aperture; red means it "
                "exceeds the Panda 80 mm physical limit. The mesh stays physical-open. "
                "Viewing never moves the robot."
            )
        self.server.gui.add_markdown(
            "**Navigation:** left-drag orbit, right-drag pan, wheel zoom; "
            "W/A/S/D move, Q/E elevate, arrow keys rotate. Viser's orbit has "
            "a pole limit; use Q/E plus arrow keys or a named camera preset to "
            "cross that pole."
        )
        show_scene = self.server.gui.add_checkbox("Show RGB-D scene", True)
        proposal_scene = self.scene_mode != "execution"
        show_surface = self.server.gui.add_checkbox(
            "Show clean target surface" if proposal_scene else "Show focus samples",
            proposal_scene,
        )
        show_low = self.server.gui.add_checkbox(
            "Show low mask points" if proposal_scene else "Show low focus samples",
            proposal_scene,
        )
        show_support = self.server.gui.add_checkbox("Show estimated table plane", True)
        show_ag = self.server.gui.add_checkbox("Show AnyGrasp", True)
        show_gx = self.server.gui.add_checkbox("Show GraspGen-X", True)
        show_refined = self.server.gui.add_checkbox("Show refined poses", True)
        show_target = self.server.gui.add_checkbox("Show TARGET", True)
        show_actual = self.server.gui.add_checkbox("Show ACTUAL", True)
        scope = self.server.gui.add_dropdown(
            "Pose scope", ("default", "all", "focus"), initial_value="default"
        )
        options = tuple(f"{pose.label} | {pose.pose_id}" for pose in self.poses)
        focus = self.server.gui.add_dropdown(
            "Focus pose", options, initial_value=options[0]
        )
        status = self.server.gui.add_markdown("")

        def update(_event: Any = None) -> None:
            selected_label = str(focus.value).split(" | ", 1)[0]
            selected = next(
                pose for pose in self.poses if pose.label == selected_label
            )
            with self.lock:
                self.scene_cloud_handle.visible = bool(show_scene.value)
                self.target_surface_handle.visible = bool(show_surface.value)
                self.target_low_handle.visible = bool(show_low.value)
                self.support_grid_handle.visible = (
                    bool(show_support.value) and self.support_plane_known
                )
                self.support_gap_handle.visible = (
                    bool(show_support.value)
                    and proposal_scene
                    and self.support_plane_known
                )
                self.support_label_handle.visible = (
                    bool(show_support.value) and proposal_scene
                )
                self.show_backend["anygrasp"] = bool(show_ag.value)
                self.show_backend["graspgenx"] = bool(show_gx.value)
                self.show_backend["refined"] = bool(show_refined.value)
                self.show_backend["target"] = bool(show_target.value)
                self.show_backend["actual"] = bool(show_actual.value)
                self.scope = str(scope.value)
                self.focus_pose_id = selected.pose_id
                self._apply_visibility()
                status.content = self._status_markdown()

        show_scene.on_update(update)
        show_surface.on_update(update)
        show_low.on_update(update)
        show_support.on_update(update)
        show_ag.on_update(update)
        show_gx.on_update(update)
        show_refined.on_update(update)
        show_target.on_update(update)
        show_actual.on_update(update)
        scope.on_update(update)
        focus.on_update(update)

        for preset, label in (
            ("scene", "Camera: scene"),
            ("top", "Camera: top"),
            ("front", "Camera: front"),
            ("side", "Camera: side"),
            ("pose_approach", "Camera: approach"),
            ("pose_jaws", "Camera: jaws"),
        ):
            button = self.server.gui.add_button(label)

            @button.on_click
            def _camera(event: Any, chosen: str = preset) -> None:
                if event.client is None:
                    return
                with self.lock:
                    self._set_client_camera(
                        event.client, chosen, self.focus_pose_id
                    )

        capture_button = self.server.gui.add_button(
            "Capture current view", color="green"
        )

        @capture_button.on_click
        def _capture(event: Any) -> None:
            if event.client is None:
                return
            result = self.capture(
                viewer_id=int(event.client.client_id), camera="current"
            )
            if result.get("success"):
                event.client.add_notification(
                    title="Grasp view captured",
                    body=str(result["image_ref"]),
                    loading=False,
                    with_close_button=True,
                    auto_close=5000,
                )
            else:
                event.client.add_notification(
                    title="Capture failed",
                    body=str(result.get("message")),
                    loading=False,
                    with_close_button=True,
                    color="red",
                )
        status.content = self._status_markdown()

    def _status_markdown(self) -> str:
        shown = self.displayed_pose_ids()
        focus = self.pose_by_id[self.focus_pose_id]
        aperture = (
            f"{focus.aperture_m * 1000.0:.1f} mm ({focus.aperture_status})"
            if focus.aperture_m is not None
            else "unknown"
        )
        support = (
            f"{focus.support_clearance_status} "
            f"({focus.support_clearance_mm:.1f} / "
            f"{focus.support_clearance_required_mm:.0f} mm)"
            if focus.support_clearance_mm is not None
            else f"{focus.support_clearance_status} (unknown; execution disabled)"
        )
        return (
            f"**revision:** {self.revision}  \n"
            f"**focus:** {focus.label}  \n"
            f"**aperture:** {focus.aperture_semantics} {aperture}  \n"
            f"**support clearance:** {support}  \n"
            f"**visible poses:** {len(shown)} / {len(self.poses)}"
        )

    def displayed_pose_ids(self) -> list[str]:
        visible: list[str] = []
        for pose in self.poses:
            if not self.show_backend[pose.backend]:
                continue
            if self.scope == "focus" and pose.pose_id != self.focus_pose_id:
                continue
            if self.scope == "default" and pose.backend_rank != 0:
                continue
            visible.append(pose.pose_id)
        return visible

    def _apply_visibility(self, *, increment: bool = True) -> None:
        displayed = set(self.displayed_pose_ids())
        for pose_id, handles in self.pose_handles.items():
            visible = pose_id in displayed
            for handle in handles:
                handle.visible = visible
        if increment:
            self.revision += 1
        self._write_state()

    def _set_client_camera(
        self, client: Any, preset: str, focus_pose_id: str | None
    ) -> None:
        focus_pose = (
            self.pose_by_id.get(focus_pose_id) if focus_pose_id else None
        )
        value = camera_preset(
            preset,
            target_center=self.scene.center_world,
            scene_radius_m=self.scene.radius_m,
            focus_pose=focus_pose,
        )
        client.camera.position = value["position"]
        client.camera.look_at = value["look_at"]
        client.camera.up_direction = value["up_direction"]
        client.camera.fov = float(value["fov"])

    def _orbit_client_camera(
        self,
        client: Any,
        *,
        azimuth_delta_deg: float,
        elevation_delta_deg: float,
        zoom_scale: float,
    ) -> None:
        focus = self.pose_by_id[self.focus_pose_id].world_from_grip_site[:3, 3]
        value = orbit_camera_state(
            client.camera.position,
            focus,
            azimuth_delta_deg=azimuth_delta_deg,
            elevation_delta_deg=elevation_delta_deg,
            zoom_scale=zoom_scale,
        )
        client.camera.position = value["position"]
        client.camera.look_at = value["look_at"]
        client.camera.up_direction = value["up_direction"]

    def state(self) -> dict[str, Any]:
        self._sync_connected_clients()
        with self.lock:
            return {
                "success": True,
                "kind": "grasp_inspector",
                "proposal_set_id": self.proposal_set_id,
                "scene_mode": self.scene_mode,
                "episode_root": str(self.episode_root),
                "observation_id": self.observation_id,
                "proposal_id": self.proposal_id,
                "result_id": self.result_id,
                "state_revision": self.revision,
                "visibility": {
                    "anygrasp": self.show_backend["anygrasp"],
                    "graspgenx": self.show_backend["graspgenx"],
                    "refined": self.show_backend["refined"],
                    "target": self.show_backend["target"],
                    "actual": self.show_backend["actual"],
                    "pose_scope": self.scope,
                },
                "focused_pose_id": self.focus_pose_id,
                "displayed_pose_ids": self.displayed_pose_ids(),
                "poses": [pose.public_dict() for pose in self.poses],
                "support_clearance_required_mm": SUPPORT_CLEARANCE_REQUIRED_MM,
                "support_plane_status": (
                    "estimated"
                    if getattr(self, "support_plane_known", False)
                    else "unknown"
                ),
                "scene_diagnostics": dict(self.scene.diagnostics),
                "viewer_clients": [
                    {"viewer_id": str(client_id), "connected": True}
                    for client_id in sorted(self.clients)
                ],
            }

    def _write_state(self) -> None:
        path = self.artifact_root / "state.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.state(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def configure(self, request: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if "show_anygrasp" in request:
                self.show_backend["anygrasp"] = bool(request["show_anygrasp"])
            if "show_graspgenx" in request:
                self.show_backend["graspgenx"] = bool(request["show_graspgenx"])
            if "show_refined" in request:
                self.show_backend["refined"] = bool(request["show_refined"])
            if "show_target" in request:
                self.show_backend["target"] = bool(request["show_target"])
            if "show_actual" in request:
                self.show_backend["actual"] = bool(request["show_actual"])
            if "pose_scope" in request:
                value = str(request["pose_scope"])
                if value not in {"default", "all", "focus"}:
                    return self._failure("invalid_pose_scope", value)
                self.scope = value
            if request.get("focus_pose_id") is not None:
                pose_id = str(request["focus_pose_id"])
                if pose_id not in self.pose_by_id:
                    return self._failure("unknown_pose", pose_id)
                self.focus_pose_id = pose_id
            if self.scope == "focus" and self.focus_pose_id not in self.pose_by_id:
                return self._failure("unknown_pose", self.focus_pose_id)
            self._apply_visibility()
            preset = str(request.get("camera_preset", "keep"))
            if preset != "keep":
                client_result = self._resolve_client(request.get("viewer_id"))
                if isinstance(client_result, dict):
                    return client_result
                try:
                    self._set_client_camera(
                        client_result, preset, self.focus_pose_id
                    )
                except ValueError as exc:
                    return self._failure("invalid_camera_preset", str(exc))
            if any(
                key in request
                for key in (
                    "orbit_azimuth_deg",
                    "orbit_elevation_deg",
                    "zoom_scale",
                )
            ):
                client_result = self._resolve_client(request.get("viewer_id"))
                if isinstance(client_result, dict):
                    return client_result
                try:
                    self._orbit_client_camera(
                        client_result,
                        azimuth_delta_deg=float(
                            request.get("orbit_azimuth_deg", 0.0)
                        ),
                        elevation_delta_deg=float(
                            request.get("orbit_elevation_deg", 0.0)
                        ),
                        zoom_scale=float(request.get("zoom_scale", 1.0)),
                    )
                except (TypeError, ValueError) as exc:
                    return self._failure("invalid_camera_orbit", str(exc))
            return self.state()

    def add_pose(self, request: dict[str, Any]) -> dict[str, Any]:
        """Register one immutable world-frame Panda grip-site pose live."""

        with self.lock:
            pose_id = str(request.get("pose_id") or "").strip()
            if not pose_id:
                return self._failure("missing_pose_id", "pose_id is required")
            if pose_id in self.pose_by_id:
                return self._failure("pose_exists", pose_id)
            if request.get("observation_id") not in {None, self.observation_id}:
                return self._failure(
                    "stale_pose",
                    f"pose observation {request.get('observation_id')!r} does not match {self.observation_id!r}",
                )
            try:
                grip_site = _rigid_transform(
                    request.get("transform_world_from_grip_site"),
                    name="transform_world_from_grip_site",
                )
            except ValueError as exc:
                return self._failure("invalid_pose", str(exc))
            refined_rank = sum(
                1 for pose in self.poses if pose.backend == "refined"
            )
            label = str(request.get("label") or f"R{refined_rank}")
            base = grip_site.copy()
            base[:3, 3] -= PANDA_GRIP_SITE_OFFSET_M * grip_site[:3, 2]
            pose = GraspPose(
                pose_id=pose_id,
                label=label,
                backend="refined",
                backend_rank=refined_rank,
                score=(
                    float(request["score"])
                    if isinstance(request.get("score"), (int, float))
                    else None
                ),
                world_from_gripper_base=base,
                world_from_grip_site=grip_site,
                source_id=str(request.get("source_id") or pose_id),
                branch=str(request.get("method") or "explicit")[:80],
            )
            pose = pose_with_support_clearance(
                pose,
                self.collision_mesh_vertices,
                support_plane_normal_world=self.scene.support_plane_normal_world,
                support_plane_offset_m=self.scene.support_plane_offset_m,
            )
            self.poses.append(pose)
            self.pose_by_id[pose.pose_id] = pose
            self._add_pose_node(pose)
            self.focus_pose_id = pose.pose_id
            self.scope = "focus"
            self.show_backend["refined"] = True
            self._apply_visibility()
            for client in self.clients.values():
                self._set_client_camera(client, "pose_jaws", pose.pose_id)
            artifact = self.artifact_root / "registered" / f"{_safe_name(pose_id)}.json"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(
                json.dumps(
                    {
                        "pose": pose.public_dict(),
                        "transform_world_from_grip_site": grip_site.tolist(),
                        "request": request,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            result = self.state()
            result["added_pose_id"] = pose.pose_id
            result["added_pose"] = pose.public_dict()
            result["support_clearance_mm"] = pose.support_clearance_mm
            result["support_clearance_required_mm"] = (
                pose.support_clearance_required_mm
            )
            result["support_clearance_status"] = pose.support_clearance_status
            result["artifact_ref"] = str(artifact)
            return result

    def _resolve_client(self, viewer_id: Any) -> Any | dict[str, Any]:
        self._sync_connected_clients()
        if not self.clients:
            return self._failure(
                "viewer_not_connected", "No Viser browser client is connected."
            )
        if viewer_id is None:
            if len(self.clients) != 1:
                return self._failure(
                    "ambiguous_viewer",
                    f"Connected viewer IDs: {sorted(self.clients)}",
                )
            return next(iter(self.clients.values()))
        try:
            numeric = int(viewer_id)
        except (TypeError, ValueError):
            return self._failure("unknown_viewer", str(viewer_id))
        if numeric not in self.clients:
            return self._failure("unknown_viewer", str(viewer_id))
        return self.clients[numeric]

    def capture(self, *, viewer_id: Any, camera: str) -> dict[str, Any]:
        with self.lock:
            client_result = self._resolve_client(viewer_id)
            if isinstance(client_result, dict):
                return client_result
            client = client_result
            if camera != "current":
                try:
                    self._set_client_camera(client, camera, self.focus_pose_id)
                except ValueError as exc:
                    return self._failure("invalid_camera_preset", str(exc))
            self.capture_index += 1
            index = self.capture_index
            camera_state = {
                "position": np.asarray(client.camera.position).tolist(),
                "look_at": np.asarray(client.camera.look_at).tolist(),
                "up_direction": np.asarray(client.camera.up_direction).tolist(),
                "wxyz": np.asarray(client.camera.wxyz).tolist(),
                "fov_rad": float(client.camera.fov),
            }

        result: dict[str, Any] = {}

        def render() -> None:
            try:
                with self.render_lock:
                    result["image"] = client.get_render(
                        height=768,
                        width=1024,
                        transport_format="png",
                    )
            except Exception as exc:  # pragma: no cover - browser/runtime path
                result["error"] = repr(exc)

        worker = threading.Thread(target=render, daemon=True)
        worker.start()
        worker.join(timeout=12.0)
        if worker.is_alive():
            return self._failure("capture_timeout", "Browser render timed out.")
        if "error" in result:
            return self._failure("capture_failed", str(result["error"]))
        image_path = self.artifact_root / "views" / f"{index:06d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(result["image"], dtype=np.uint8)).save(image_path)
        metadata = {
            "success": True,
            "kind": "grasp_inspector_capture",
            "view_id": f"view-{index:06d}",
            "proposal_set_id": self.proposal_set_id,
            "state_revision": self.revision,
            "camera_source": camera,
            "camera_state": camera_state,
            "displayed_pose_ids": self.displayed_pose_ids(),
            "image_ref": str(image_path),
        }
        image_path.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._write_state()
        return metadata

    def _failure(self, code: str, message: str) -> dict[str, Any]:
        return {
            "success": False,
            "code": code,
            "retryable": code
            in {
                "viewer_not_connected",
                "ambiguous_viewer",
                "unknown_viewer",
                "unknown_pose",
                "capture_timeout",
                "capture_failed",
            },
            "message": message,
            "proposal_set_id": self.proposal_set_id,
            "state_revision": self.revision,
        }


class ControlServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], viewer: GraspPoseViewer) -> None:
        self.viewer = viewer
        super().__init__(address, ControlHandler)


class ControlHandler(BaseHTTPRequestHandler):
    server: ControlServer

    def _respond(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _request_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/state":
            self._respond(self.server.viewer.state())
        else:
            self._respond(
                {"success": False, "code": "not_found"},
                status=HTTPStatus.NOT_FOUND,
            )

    def do_POST(self) -> None:  # noqa: N802
        try:
            request = self._request_json()
            if self.path == "/configure":
                result = self.server.viewer.configure(request)
            elif self.path == "/add_pose":
                result = self.server.viewer.add_pose(request)
            elif self.path == "/capture":
                result = self.server.viewer.capture(
                    viewer_id=request.get("viewer_id"),
                    camera=str(request.get("camera", "current")),
                )
            else:
                self._respond(
                    {"success": False, "code": "not_found"},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            self._respond(result, status=200 if result.get("success") else 409)
        except (ValueError, json.JSONDecodeError) as exc:
            self._respond(
                {"success": False, "code": "invalid_request", "message": str(exc)},
                status=HTTPStatus.BAD_REQUEST,
            )

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[grasp-viewer-control] {format % args}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--observation-id", required=True)
    parser.add_argument("--proposal-id")
    parser.add_argument("--result-id")
    parser.add_argument("--graspgenx-results", type=Path)
    parser.add_argument("--graspgenx-prefilter", type=Path)
    parser.add_argument(
        "--graspgenx-prefilter-raw-ranks", nargs="*", type=int, default=[]
    )
    parser.add_argument("--anygrasp-response", type=Path)
    parser.add_argument("--execution-comparison", type=Path)
    parser.add_argument("--proposal-set-id", default="ps-retained-000001")
    parser.add_argument("--top-k-per-backend", type=int, default=20)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=8082)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/tmp/openeta-grasp-inspector"),
    )
    parser.add_argument(
        "--graspgenx-assets",
        type=Path,
        default=Path("third_party/GraspGenX/assets"),
    )
    args = parser.parse_args()

    from graspgenx.x_grippers import resolve_gripper_info

    scene = load_scene_cloud(
        args.sample_dir.expanduser().resolve(),
        expected_observation_id=args.observation_id,
    )
    poses: list[GraspPose] = []
    if args.graspgenx_results is not None:
        poses.extend(
            load_graspgenx_poses(
                args.graspgenx_results.expanduser().resolve(),
                proposal_set_id=args.proposal_set_id,
                top_k=args.top_k_per_backend,
            )
        )
    if args.graspgenx_prefilter_raw_ranks:
        if args.graspgenx_prefilter is None:
            parser.error(
                "--graspgenx-prefilter is required with raw-rank selection"
            )
        poses.extend(
            load_graspgenx_prefilter_poses(
                args.graspgenx_prefilter.expanduser().resolve(),
                proposal_set_id=args.proposal_set_id,
                raw_ranks=args.graspgenx_prefilter_raw_ranks,
            )
        )
    if args.anygrasp_response is not None:
        poses.extend(
            load_anygrasp_poses(
                args.anygrasp_response.expanduser().resolve(),
                camera_to_world_opencv=scene.camera_to_world_opencv,
                proposal_set_id=args.proposal_set_id,
                top_k=args.top_k_per_backend,
                expected_result_id=args.result_id,
            )
        )
    if args.execution_comparison is not None:
        poses.extend(
            load_execution_poses(
                args.execution_comparison.expanduser().resolve(),
                proposal_set_id=args.proposal_set_id,
            )
        )
    poses.sort(key=lambda pose: (pose.backend, pose.backend_rank))
    gripper_info = resolve_gripper_info(
        "franka_panda", assets_dir=str(args.graspgenx_assets)
    )
    viewer = GraspPoseViewer(
        scene=scene,
        poses=poses,
        proposal_set_id=args.proposal_set_id,
        episode_root=args.episode_root,
        observation_id=args.observation_id,
        host=args.host,
        port=args.port,
        artifact_root=args.artifact_root.expanduser().resolve(),
        visual_mesh=gripper_info.visual_mesh,
        collision_mesh=gripper_info.collision_mesh,
        scene_mode="execution" if args.execution_comparison is not None else "proposal",
        proposal_id=args.proposal_id,
        result_id=args.result_id,
    )
    control = ControlServer((args.control_host, args.control_port), viewer)
    control_thread = threading.Thread(target=control.serve_forever, daemon=True)
    control_thread.start()
    print(
        json.dumps(
            {
                "viewer_url": f"http://{args.host}:{args.port}",
                "control_url": f"http://{args.control_host}:{args.control_port}",
                "state_path": str(viewer.artifact_root / "state.json"),
                "pose_count": len(poses),
                "default_displayed_pose_ids": viewer.displayed_pose_ids(),
            },
            indent=2,
        ),
        flush=True,
    )
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        control.shutdown()
        control.server_close()
        viewer.server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
