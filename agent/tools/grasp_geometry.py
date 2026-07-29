"""Deterministic geometry tools for compiling and refining grasp seeds."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from adapter.protocol import JsonDict
from agent.runtime.calibration_registry import DEFAULT_GRASP_CALIBRATION_PROFILE
from agent.tools.grasp_strategies import (
    DEFAULT_GRASP_STRATEGY_ROOT,
    GraspStrategyError,
    load_grasp_strategies,
    public_grasp_strategy,
    select_grasp_strategy,
    strategy_alignment_policy,
    strategy_candidate_filter,
    strategy_grasp_width_bounds,
    strategy_motion_policy,
    strategy_pose_policy,
)
from agent.tools.registry import ToolExecutionContext, ToolHandler, ToolResult, make_tool_result


LEGACY_GRASP_CALIBRATION_SCHEMA = "libero.grasp_to_eef_calibration.v1"
GRASP_CALIBRATION_SCHEMA = "libero.grasp_to_eef_calibration.v2"
SUPPORTED_GRASP_CALIBRATION_SCHEMAS = {
    LEGACY_GRASP_CALIBRATION_SCHEMA,
    GRASP_CALIBRATION_SCHEMA,
}
COMPILED_GRASP_SCHEMA = "openeta.compiled_grasp_seed.v1"
WRIST_ALIGNMENT_SCHEMA = "openeta.wrist_alignment.v1"
DEFAULT_GRASP_PROFILE = DEFAULT_GRASP_CALIBRATION_PROFILE
_OPENCV_TO_OPENGL = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
_PANDA_TOP_DOWN_ROTATION = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
_WORLD_NEGATIVE_Z = [0.0, 0.0, -1.0]
_MIN_SAFE_HOVER_DISTANCE_M = 0.15
_DEFAULT_REFINEMENT_HOVER_CLEARANCE_M = 0.20
_ARTICULATED_HANDLE_APPROACH_MODES = {"top_down", "front", "side"}


class GraspGeometryError(ValueError):
    """Raised when a grasp geometry contract cannot be satisfied."""


class GraspCandidateRejected(GraspGeometryError):
    """Raised when one estimator candidate violates a strategy constraint."""

    def __init__(
        self,
        message: str,
        *,
        rejection_code: str,
        recovery_class: str = "none",
    ) -> None:
        super().__init__(message)
        self.rejection_code = rejection_code
        self.recovery_class = recovery_class


def build_compile_grasp_seed_handler(
    profile_path: str | Path = DEFAULT_GRASP_PROFILE,
    *,
    strategy_root: (
        str | Path | Callable[[ToolExecutionContext], str | Path]
    ) = DEFAULT_GRASP_STRATEGY_ROOT,
) -> ToolHandler:
    """Build a compiler with fixed embodiment calibration and task strategies."""

    resolved_profile = Path(profile_path)

    def handler(context: ToolExecutionContext) -> ToolResult:
        try:
            selected_strategy_root = (
                strategy_root(context) if callable(strategy_root) else strategy_root
            )
            profile, profile_sha256 = _load_profile(resolved_profile)
            outputs = compile_grasp_seed(
                context.parameters,
                profile=profile,
                profile_sha256=profile_sha256,
                strategies=load_grasp_strategies(Path(selected_strategy_root)),
            )
        except GraspCandidateRejected as exc:
            camera_pose = context.parameters.get("camera_pose")
            camera_pose = camera_pose if isinstance(camera_pose, Mapping) else {}
            candidate_id = str(camera_pose.get("id") or "")
            return make_tool_result(
                context,
                success=False,
                content=f"grasp seed candidate rejected: {exc}",
                outputs={
                    "reason": "grasp_seed_candidate_rejected",
                    "candidate_rejection": True,
                    "candidate_id": candidate_id,
                    "rejection_code": exc.rejection_code,
                    "recovery_class": exc.recovery_class,
                },
                diagnostics=[
                    {
                        "code": "grasp_seed_candidate_rejected",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "candidate_rejection": True,
                        "candidate_id": candidate_id,
                        "rejection_code": exc.rejection_code,
                        "recovery_class": exc.recovery_class,
                    }
                ],
            )
        except (
            OSError,
            json.JSONDecodeError,
            GraspGeometryError,
            GraspStrategyError,
        ) as exc:
            return make_tool_result(
                context,
                success=False,
                content=f"grasp seed compilation failed: {exc}",
                outputs={"reason": "grasp_seed_compile_failed"},
                diagnostics=[
                    {
                        "code": "grasp_seed_compile_failed",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            )
        return make_tool_result(
            context,
            success=True,
            content="normalized grasp seed compiled to staged world-frame EEF poses",
            outputs=outputs,
        )

    return handler


def build_wrist_alignment_handler() -> ToolHandler:
    """Build a read-only mask/depth wrist alignment calculator."""

    def handler(context: ToolExecutionContext) -> ToolResult:
        try:
            outputs = compute_wrist_alignment(context.parameters)
        except (OSError, GraspGeometryError) as exc:
            return make_tool_result(
                context,
                success=False,
                content=f"wrist alignment failed: {exc}",
                outputs={"reason": "wrist_alignment_failed"},
                diagnostics=[
                    {
                        "code": "wrist_alignment_failed",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            )
        return make_tool_result(
            context,
            success=True,
            content="bounded wrist alignment correction computed",
            outputs=outputs,
        )

    return handler


def compile_grasp_seed(
    parameters: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
    profile_sha256: str,
    strategies: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    candidate = _mapping(parameters.get("camera_pose"), "camera_pose")
    extrinsics = _mapping(parameters.get("camera_extrinsics"), "camera_extrinsics")
    target_geometry_family = str(
        parameters.get("target_geometry_family") or parameters.get("target_class") or ""
    ).strip()
    approach_mode = str(parameters.get("approach_mode") or "").strip().lower()
    if approach_mode and approach_mode not in _ARTICULATED_HANDLE_APPROACH_MODES:
        raise GraspGeometryError(
            "approach_mode must be one of front, side, or top_down"
        )
    requested_strategy_id = str(parameters.get("strategy_id") or "").strip()
    scene_epoch = _nonnegative_int(parameters.get("scene_epoch"), "scene_epoch")
    requested_pregrasp_distance = _bounded_float(
        parameters.get("pregrasp_distance_m", _MIN_SAFE_HOVER_DISTANCE_M),
        "pregrasp_distance_m",
        0.04,
        0.16,
    )
    # Hover is a clearance pose, not a task-tuned contact correction. Keep at
    # least 15 cm along the grasp approach normal before wrist alignment.
    pregrasp_distance = max(_MIN_SAFE_HOVER_DISTANCE_M, requested_pregrasp_distance)
    candidate_fallback = (
        parameters.get("candidate_fallback") is True
        or candidate.get("candidate_fallback") is True
    )
    _validate_profile(profile, target_class=target_geometry_family)

    candidate_id = str(candidate.get("id") or "").strip()
    final_refinable_fallback = candidate.get("final_refinable_fallback") is True
    if not candidate_id:
        raise GraspGeometryError("camera_pose.id is required")
    if str(candidate.get("frame") or "") != "camera":
        raise GraspGeometryError("camera_pose.frame must be 'camera'")
    if str(candidate.get("camera_frame") or "opencv").lower() != "opencv":
        raise GraspGeometryError("camera_pose.camera_frame must be 'opencv'")
    max_gripper_width = _bounded_float(
        profile.get("max_gripper_width_m"),
        "max_gripper_width_m",
        0.001,
        0.2,
    )
    width = _bounded_float(
        candidate.get("width"),
        "camera_pose.width",
        0.0,
        max_gripper_width,
    )
    calibration_id = str(profile.get("calibration_id") or "")
    available_strategies = (
        load_grasp_strategies()
        if strategies is None and profile.get("schema_version") == GRASP_CALIBRATION_SCHEMA
        else list(strategies or [])
    )
    strategy, strategy_selection = select_grasp_strategy(
        available_strategies,
        calibration_id=calibration_id,
        target_geometry_family=target_geometry_family,
        strategy_id=requested_strategy_id,
    )
    if approach_mode:
        if target_geometry_family not in {"articulated_handle", "drawer_handle"}:
            raise GraspGeometryError(
                "approach_mode is reserved for articulated_handle geometry"
            )
        pose_policy = strategy_pose_policy(strategy) if strategy is not None else {}
        preserves_candidate = (
            pose_policy.get("orientation") == "preserve_candidate"
            and pose_policy.get("approach_axis") == "preserve_candidate"
        )
        if approach_mode == "top_down" and preserves_candidate:
            raise GraspGeometryError(
                "top_down approach_mode requires a top-down grasp strategy"
            )
        if approach_mode in {"front", "side"} and not preserves_candidate:
            raise GraspGeometryError(
                f"{approach_mode} approach_mode requires a preserve-candidate strategy"
            )
    legacy_restricted = (
        _mapping(profile.get("restricted_geometry"), "restricted_geometry")
        if profile.get("schema_version") == LEGACY_GRASP_CALIBRATION_SCHEMA
        else None
    )
    if strategy is not None:
        width_bounds = strategy_grasp_width_bounds(strategy)
        if width_bounds[1] > max_gripper_width:
            raise GraspGeometryError("strategy grasp width exceeds calibration max_gripper_width_m")
    elif legacy_restricted is not None:
        legacy_widths = _vector(
            legacy_restricted.get("width_bounds_m"),
            2,
            "restricted_geometry.width_bounds_m",
        )
        width_bounds = (legacy_widths[0], legacy_widths[1])
    else:
        width_bounds = (0.0, max_gripper_width)
    if (
        not final_refinable_fallback
        and (width < width_bounds[0] or width > width_bounds[1])
    ):
        raise GraspCandidateRejected(
            f"candidate width {width:.4f} m is outside active strategy bounds "
            f"[{width_bounds[0]:.4f}, {width_bounds[1]:.4f}]",
            rejection_code="strategy_width_out_of_bounds",
            recovery_class="perception_refinable",
        )

    r_camera_grasp = _rotation(candidate.get("rotation_matrix"), "camera_pose.rotation_matrix")
    p_camera_grasp = _vector(candidate.get("translation_xyz"), 3, "camera_pose.translation_xyz")
    r_world_cv, p_world_camera = _opencv_camera_to_world(extrinsics)

    transform = _mapping(profile.get("T_grasp_eef"), "T_grasp_eef")
    r_grasp_eef = _rotation(transform.get("rotation_matrix"), "T_grasp_eef.rotation_matrix")
    p_grasp_eef = _vector(transform.get("translation_xyz"), 3, "T_grasp_eef.translation_xyz")

    r_world_grasp = _matmul3(r_world_cv, r_camera_grasp)
    r_world_eef = _matmul3(r_world_grasp, r_grasp_eef)
    p_world_grasp = _add(_matvec3(r_world_cv, p_camera_grasp), p_world_camera)
    p_world_eef = _add(p_world_grasp, _matvec3(r_world_grasp, p_grasp_eef))
    approach_world = _normalise([r_world_grasp[row][0] for row in range(3)], "approach")
    native_downward_alignment = max(-1.0, min(1.0, -approach_world[2]))
    orientation_clamped = False
    alignment_policy: JsonDict = {"target_region": "mask_centroid"}
    motion_policy: JsonDict = {}
    if strategy is not None:
        candidate_filter = strategy_candidate_filter(strategy)
        min_alignment = candidate_filter.get("min_downward_alignment")
        if (
            not final_refinable_fallback
            and min_alignment is not None
            and native_downward_alignment < float(min_alignment)
            and not candidate_fallback
        ):
            raise GraspCandidateRejected(
                "candidate native downward alignment "
                f"{native_downward_alignment:.3f} is below active strategy minimum "
                f"{float(min_alignment):.3f}",
                rejection_code="strategy_alignment_rejected",
                recovery_class="perception_refinable",
            )
        pose_policy = strategy_pose_policy(strategy)
        if pose_policy.get("orientation") == "top_down":
            r_world_eef = [list(row) for row in _PANDA_TOP_DOWN_ROTATION]
            orientation_clamped = True
        elif pose_policy.get("orientation") == "top_down_preserve_yaw":
            r_world_eef = _top_down_preserve_yaw(r_world_eef)
            orientation_clamped = True
        if pose_policy.get("approach_axis") == "world_-Z":
            approach_world = list(_WORLD_NEGATIVE_Z)
        alignment_policy.update(strategy_alignment_policy(strategy))
        motion_policy.update(strategy_motion_policy(strategy))
    elif legacy_restricted is not None and profile.get("status") == "candidate":
        r_world_eef = [list(row) for row in _PANDA_TOP_DOWN_ROTATION]
        approach_world = list(_WORLD_NEGATIVE_Z)
        orientation_clamped = True
    p_hover = [p_world_eef[index] - pregrasp_distance * approach_world[index] for index in range(3)]
    compiled_identity: JsonDict = {
        "candidate_id": candidate_id,
        "candidate": candidate,
        "extrinsics": extrinsics,
        "profile_sha256": profile_sha256,
        "strategy_id": strategy.get("strategy_id") if strategy is not None else None,
        "pregrasp_distance_m": pregrasp_distance,
        "scene_epoch": scene_epoch,
    }
    if approach_mode:
        compiled_identity["approach_mode"] = approach_mode
    compiled_id = hashlib.sha256(
        json.dumps(
            compiled_identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:20]

    pose_common = {
        "frame": "world",
        "rotation_matrix": _round_matrix(r_world_eef),
        "source_grasp_id": candidate_id,
        "compiled_grasp_id": compiled_id,
        "calibration_id": calibration_id,
        "scene_epoch": scene_epoch,
    }
    if approach_mode:
        pose_common["approach_mode"] = approach_mode
    precontact_distance = motion_policy.get("precontact_distance_m")
    precontact_pose = None
    if precontact_distance is not None:
        p_precontact = [
            p_world_eef[index] - float(precontact_distance) * approach_world[index]
            for index in range(3)
        ]
        precontact_pose = {
            **pose_common,
            "xyz": _round_vector(p_precontact),
            "grasp_stage": "precontact",
        }
    return {
        "schema_version": COMPILED_GRASP_SCHEMA,
        "compiled_grasp_id": compiled_id,
        "candidate_id": candidate_id,
        "camera_frame_id": str(parameters.get("camera_frame_id") or ""),
        "scene_epoch": scene_epoch,
        "target_class": target_geometry_family,
        "target_geometry_family": target_geometry_family,
        **({"approach_mode": approach_mode} if approach_mode else {}),
        "calibration_id": calibration_id,
        "calibration_status": str(profile.get("status") or ""),
        "not_validated": profile.get("status") != "validated",
        "profile_sha256": profile_sha256,
        "approach_world_xyz": _round_vector(approach_world),
        "native_downward_alignment": round(native_downward_alignment, 6),
        "hover_offset_world_xyz": _round_vector(
            [-pregrasp_distance * component for component in approach_world]
        ),
        "gripper_width_m": width,
        "final_refinable_fallback": final_refinable_fallback,
        "requested_pregrasp_distance_m": requested_pregrasp_distance,
        "pregrasp_distance_m": pregrasp_distance,
        "orientation_clamped": orientation_clamped,
        "strategy_id": strategy.get("strategy_id") if strategy is not None else None,
        "strategy_status": strategy.get("status") if strategy is not None else None,
        "strategy_selection": strategy_selection,
        "outside_validated_strategy_scope": (
            strategy is None or strategy.get("status") != "validated"
        ),
        "candidate_fallback": candidate_fallback,
        **(
            {
                "fallback_reason": str(
                    parameters.get("fallback_reason")
                    or candidate.get("fallback_reason")
                )
            }
            if parameters.get("fallback_reason") or candidate.get("fallback_reason")
            else {}
        ),
        "hover_pose": {
            **pose_common,
            "xyz": _round_vector(p_hover),
            "grasp_stage": "hover",
        },
        "contact_pose": {
            **pose_common,
            "xyz": _round_vector(p_world_eef),
            "grasp_stage": "contact",
        },
        "precontact_pose": precontact_pose,
        "grasp_strategy": public_grasp_strategy(strategy),
        "alignment_policy": alignment_policy,
        "motion_policy": motion_policy,
        "warning": (
            (
                "No validated task-family strategy matched; preserving the grasp "
                "estimator orientation and approach as a coarse reference. "
            )
            if strategy is None
            else (
                f"Using {strategy.get('status')} task-family strategy "
                f"{strategy.get('strategy_id')}. "
            )
        )
        + (
            (
                "All articulated-handle approach modes failed; this is the one "
                "global score-selected fallback and remains subject to motion and "
                "attachment gates. "
            )
            if candidate_fallback
            and str(
                parameters.get("fallback_reason")
                or candidate.get("fallback_reason")
                or ""
            )
            == "all_approach_modes_failed"
            else (
                "All ranked candidates failed their strategy geometry checks; this is a "
                "score-selected fallback and remains subject to motion and attachment "
                "gates. "
            )
            if candidate_fallback
            else ""
        )
        + (
            "Calibration/strategy outputs remain references; hover alignment "
            "and attachment gates are mandatory."
        ),
    }


def grasp_candidate_approach_world(
    camera_pose: Mapping[str, Any],
    camera_extrinsics: Mapping[str, Any],
) -> list[float]:
    """Return a normalized candidate approach using compiler frame conventions."""

    r_camera_grasp = _rotation(
        camera_pose.get("rotation_matrix"),
        "camera_pose.rotation_matrix",
    )
    r_world_cv, _ = _opencv_camera_to_world(camera_extrinsics)
    r_world_grasp = _matmul3(r_world_cv, r_camera_grasp)
    return _normalise(
        [r_world_grasp[row][0] for row in range(3)],
        "approach",
    )


def camera_optical_forward_world(camera_extrinsics: Mapping[str, Any]) -> list[float]:
    """Return the OpenCV optical +Z direction in world coordinates."""

    r_world_cv, _ = _opencv_camera_to_world(camera_extrinsics)
    return _normalise([r_world_cv[row][2] for row in range(3)], "camera optical forward")


def grasp_refinement_hover_pose(
    camera_pose: Mapping[str, Any],
    camera_extrinsics: Mapping[str, Any],
    *,
    scene_epoch: int,
    recovery_id: str,
    clearance_m: float = _DEFAULT_REFINEMENT_HOVER_CLEARANCE_M,
) -> JsonDict:
    """Build a target-centric observation hover without trusting rejected orientation."""

    candidate_id = str(camera_pose.get("id") or "").strip()
    if not candidate_id:
        raise GraspGeometryError("camera_pose.id is required")
    if str(camera_pose.get("frame") or "") != "camera":
        raise GraspGeometryError("camera_pose.frame must be 'camera'")
    if str(camera_pose.get("camera_frame") or "opencv").lower() != "opencv":
        raise GraspGeometryError("camera_pose.camera_frame must be 'opencv'")
    clearance = _bounded_float(
        clearance_m,
        "clearance_m",
        _MIN_SAFE_HOVER_DISTANCE_M,
        0.30,
    )
    p_camera_target = _vector(
        camera_pose.get("translation_xyz"),
        3,
        "camera_pose.translation_xyz",
    )
    r_world_cv, p_world_camera = _opencv_camera_to_world(camera_extrinsics)
    p_world_target = _add(_matvec3(r_world_cv, p_camera_target), p_world_camera)
    return {
        "frame": "world",
        "xyz": _round_vector(
            [
                p_world_target[0],
                p_world_target[1],
                p_world_target[2] + clearance,
            ]
        ),
        "grasp_stage": "grasp_estimation_refinement_hover",
        "source_grasp_id": candidate_id,
        "recovery_id": recovery_id,
        "scene_epoch": _nonnegative_int(scene_epoch, "scene_epoch"),
    }


def compute_wrist_alignment(parameters: Mapping[str, Any]) -> JsonDict:
    compiled = _mapping(parameters.get("compiled_grasp"), "compiled_grasp")
    if compiled.get("schema_version") != COMPILED_GRASP_SCHEMA:
        raise GraspGeometryError("compiled_grasp has an unsupported schema")
    target_mask = Path(str(parameters.get("target_mask") or ""))
    depth_path = Path(str(parameters.get("depth") or ""))
    if not target_mask.is_file() or not depth_path.is_file():
        raise GraspGeometryError("target_mask and depth must be existing local files")
    intrinsics = _mapping(parameters.get("intrinsics"), "intrinsics")
    fx = _positive_float(intrinsics.get("fx"), "intrinsics.fx")
    fy = _positive_float(intrinsics.get("fy"), "intrinsics.fy")
    cx = _finite_float(intrinsics.get("cx"), "intrinsics.cx")
    cy = _finite_float(intrinsics.get("cy"), "intrinsics.cy")
    scale = _positive_float(intrinsics.get("scale", 1000.0), "intrinsics.scale")
    desired = parameters.get("desired_pixel_xy", [cx, cy])
    desired_xy = _vector(desired, 2, "desired_pixel_xy")
    max_correction = _bounded_float(
        parameters.get("max_correction_m", 0.03),
        "max_correction_m",
        0.005,
        0.05,
    )

    alignment_policy = compiled.get("alignment_policy")
    alignment_policy = alignment_policy if isinstance(alignment_policy, dict) else {}
    target_region = str(alignment_policy.get("target_region") or "mask_centroid")
    u, v, depth_m, width, height = _mask_depth_target(
        target_mask,
        depth_path,
        scale=scale,
        desired_xy=desired_xy,
        target_region=target_region,
    )
    if not (0 <= desired_xy[0] < width and 0 <= desired_xy[1] < height):
        raise GraspGeometryError("desired_pixel_xy is outside the image")
    delta_camera = [
        (u - desired_xy[0]) * depth_m / fx,
        (v - desired_xy[1]) * depth_m / fy,
        0.0,
    ]
    r_world_cv, _ = _opencv_camera_to_world(
        _mapping(parameters.get("camera_extrinsics"), "camera_extrinsics")
    )
    delta_world = _matvec3(r_world_cv, delta_camera)
    norm = math.sqrt(sum(value * value for value in delta_world))
    if norm > max_correction:
        scale_factor = max_correction / norm
        delta_world = [value * scale_factor for value in delta_world]
    residual_px = math.hypot(u - desired_xy[0], v - desired_xy[1])

    current_pose = _mapping(parameters.get("current_eef_pose"), "current_eef_pose")
    current_xyz = _vector(current_pose.get("xyz"), 3, "current_eef_pose.xyz")
    contact_pose = _mapping(compiled.get("contact_pose"), "compiled_grasp.contact_pose")
    contact_xyz = _vector(contact_pose.get("xyz"), 3, "compiled_grasp.contact_pose.xyz")
    aligned_hover = dict(contact_pose)
    aligned_hover.update(
        {
            "xyz": _round_vector(_add(current_xyz, delta_world)),
            "grasp_stage": "align",
            "alignment_id": "",
        }
    )
    adjusted_contact = dict(contact_pose)
    adjusted_contact.update(
        {
            "xyz": _round_vector(_add(contact_xyz, delta_world)),
            "grasp_stage": "contact",
            "alignment_id": "",
        }
    )
    precontact_pose = compiled.get("precontact_pose")
    adjusted_precontact = None
    if isinstance(precontact_pose, Mapping):
        precontact_xyz = _vector(
            precontact_pose.get("xyz"), 3, "compiled_grasp.precontact_pose.xyz"
        )
        adjusted_precontact = dict(precontact_pose)
        adjusted_precontact.update(
            {
                "xyz": _round_vector(_add(precontact_xyz, delta_world)),
                "grasp_stage": "precontact",
                "alignment_id": "",
            }
        )
    alignment_id = hashlib.sha256(
        json.dumps(
            {
                "compiled_grasp_id": compiled.get("compiled_grasp_id"),
                "mask": hashlib.sha256(target_mask.read_bytes()).hexdigest(),
                "depth": hashlib.sha256(depth_path.read_bytes()).hexdigest(),
                "delta_world": delta_world,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:20]
    aligned_hover["alignment_id"] = alignment_id
    adjusted_contact["alignment_id"] = alignment_id
    if adjusted_precontact is not None:
        adjusted_precontact["alignment_id"] = alignment_id
    return {
        "schema_version": WRIST_ALIGNMENT_SCHEMA,
        "alignment_id": alignment_id,
        "compiled_grasp_id": compiled.get("compiled_grasp_id"),
        "candidate_id": compiled.get("candidate_id"),
        "scene_epoch": _nonnegative_int(parameters.get("scene_epoch"), "scene_epoch"),
        "target_pixel_xy": [round(u, 3), round(v, 3)],
        "desired_pixel_xy": _round_vector(desired_xy),
        "target_depth_m": round(depth_m, 6),
        "target_region": target_region,
        "residual_px_before": round(residual_px, 3),
        "correction_world_xyz": _round_vector(delta_world),
        "correction_clamped": norm > max_correction,
        "aligned_hover_pose": aligned_hover,
        "adjusted_contact_pose": adjusted_contact,
        "adjusted_precontact_pose": adjusted_precontact,
    }


def _load_profile(path: Path) -> tuple[JsonDict, str]:
    data = path.read_bytes()
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise GraspGeometryError("calibration profile must contain one JSON object")
    return payload, hashlib.sha256(data).hexdigest()


def _validate_profile(profile: Mapping[str, Any], *, target_class: str) -> None:
    schema_version = profile.get("schema_version")
    if schema_version not in SUPPORTED_GRASP_CALIBRATION_SCHEMAS:
        raise GraspGeometryError("unsupported calibration profile schema")
    if profile.get("status") not in {"candidate", "validated"}:
        raise GraspGeometryError("calibration status must be candidate or validated")
    required = {
        "robot_model": "Panda",
        "gripper_model": "PandaGripper",
        "grasp_frame": "graspnet",
        "eef_frame": "openeta_eef",
        "length_unit": "m",
        "rotation_convention": "active_column_vectors",
    }
    for key, expected in required.items():
        if profile.get(key) != expected:
            raise GraspGeometryError(f"calibration {key} does not match {expected}")
    if schema_version == LEGACY_GRASP_CALIBRATION_SCHEMA:
        restricted = _mapping(profile.get("restricted_geometry"), "restricted_geometry")
        if profile.get("status") == "candidate" and (
            restricted.get("approach_axis") != "world_-Z"
            or restricted.get("eef_orientation") != "top_down"
        ):
            raise GraspGeometryError(
                "legacy candidate calibration must restrict approach to world_-Z "
                "and EEF to top_down"
            )
        target_classes = restricted.get("target_classes")
        if not isinstance(target_classes, list) or target_class not in target_classes:
            raise GraspGeometryError(
                "legacy target_class must be one of "
                + ", ".join(str(value) for value in target_classes or [])
            )


def _camera_to_world(extrinsics: Mapping[str, Any]) -> tuple[list[list[float]], list[float]]:
    rotation_value = extrinsics.get("mat")
    if isinstance(rotation_value, list) and len(rotation_value) == 9:
        position = _vector(extrinsics.get("pos"), 3, "camera_extrinsics.pos")
        flat = _vector(rotation_value, 9, "camera_extrinsics.mat")
        layout = str(extrinsics.get("matrix_layout") or "row_major").lower()
        if layout == "column_major":
            rotation = [[flat[row + col * 3] for col in range(3)] for row in range(3)]
        else:
            rotation = [flat[0:3], flat[3:6], flat[6:9]]
        _rotation(rotation, "camera_extrinsics.mat")
        return rotation, position
    for key in ("camera_to_world", "pose_mat", "matrix"):
        matrix = extrinsics.get(key)
        if isinstance(matrix, list) and len(matrix) == 4:
            rows = [_vector(row, 4, f"camera_extrinsics.{key}") for row in matrix]
            rotation = _rotation([row[:3] for row in rows[:3]], f"camera_extrinsics.{key}")
            return rotation, [rows[0][3], rows[1][3], rows[2][3]]
    raise GraspGeometryError("camera_extrinsics must contain pos+mat or a 4x4 matrix")


def _opencv_camera_to_world(
    extrinsics: Mapping[str, Any],
) -> tuple[list[list[float]], list[float]]:
    """Return a transform whose local axes are OpenCV optical axes.

    Missing ``camera_frame`` keeps the historical OpenGL interpretation used
    by LIBERO and older simulator packets.  New simulator adapters must tag
    their normalized packet explicitly with ``camera_frame="opencv"``.
    """

    rotation, position = _camera_to_world(extrinsics)
    raw_frame = str(extrinsics.get("camera_frame") or "opengl")
    camera_frame = raw_frame.strip().lower().replace("-", "_").replace(" ", "_")
    if camera_frame in {"opencv", "opencv_optical", "cv"}:
        return rotation, position
    if camera_frame in {"opengl", "opengl_renderer", "mujoco", "renderer"}:
        return _matmul3(rotation, _OPENCV_TO_OPENGL), position
    raise GraspGeometryError(
        f"camera_extrinsics.camera_frame has unsupported value {raw_frame!r}"
    )


def _mask_depth_target(
    mask_path: Path,
    depth_path: Path,
    *,
    scale: float,
    desired_xy: Sequence[float],
    target_region: str,
) -> tuple[float, float, float, int, int]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - runtime dependency is already required.
        raise GraspGeometryError("Pillow is required for wrist alignment") from exc
    with Image.open(mask_path) as mask_image, Image.open(depth_path) as depth_image:
        mask = mask_image.convert("L")
        depth = depth_image.convert("I")
        if mask.size != depth.size:
            raise GraspGeometryError("target mask and depth dimensions differ")
        width, height = mask.size
        foreground: list[tuple[int, int, float]] = []
        mask_values = list(mask.get_flattened_data())
        depth_values = list(depth.get_flattened_data())
        for index, mask_value in enumerate(mask_values):
            if int(mask_value) <= 0:
                continue
            raw_depth = float(depth_values[index])
            if raw_depth > 0 and math.isfinite(raw_depth):
                foreground.append((index % width, index // width, raw_depth / scale))
    if len(foreground) < 5:
        raise GraspGeometryError("target mask has too few valid depth pixels")
    depths = sorted(sample[2] for sample in foreground)
    if target_region == "nearest_shallow_surface":
        shallow_depth = depths[max(0, int(len(depths) * 0.05) - 1)]
        tolerance = max(0.008, 0.015 * shallow_depth)
        shallow = [sample for sample in foreground if sample[2] <= shallow_depth + tolerance]
        if len(shallow) < 5:
            raise GraspGeometryError("target has too few shallow rim pixels")
        selected = min(
            shallow,
            key=lambda sample: (
                (sample[0] - desired_xy[0]) ** 2
                + (sample[1] - desired_xy[1]) ** 2,
                sample[1],
                sample[0],
            ),
        )
        return selected[0], selected[1], selected[2], width, height
    if target_region != "mask_centroid":
        raise GraspGeometryError(f"unsupported alignment target region: {target_region}")
    median_depth = depths[len(depths) // 2]
    tolerance = max(0.012, 0.025 * median_depth)
    inliers = [sample for sample in foreground if abs(sample[2] - median_depth) <= tolerance]
    if len(inliers) < 5:
        raise GraspGeometryError("target depth is too inconsistent for wrist alignment")
    return (
        sum(sample[0] for sample in inliers) / len(inliers),
        sum(sample[1] for sample in inliers) / len(inliers),
        median_depth,
        width,
        height,
    )


def _top_down_preserve_yaw(rotation: Sequence[Sequence[float]]) -> list[list[float]]:
    x_axis = [float(rotation[0][0]), float(rotation[1][0])]
    norm = math.hypot(*x_axis)
    if norm < 1e-6:
        x_axis = [-float(rotation[1][1]), float(rotation[0][1])]
        norm = math.hypot(*x_axis)
    if norm < 1e-6:
        return [list(row) for row in _PANDA_TOP_DOWN_ROTATION]
    cosine = x_axis[0] / norm
    sine = x_axis[1] / norm
    return [
        [cosine, sine, 0.0],
        [sine, -cosine, 0.0],
        [0.0, 0.0, -1.0],
    ]


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GraspGeometryError(f"{label} must be an object")
    return value


def _vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise GraspGeometryError(f"{label} must contain {length} finite numbers")
    parsed = [_finite_float(item, label) for item in value]
    return parsed


def _rotation(value: Any, label: str) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise GraspGeometryError(f"{label} must be a 3x3 rotation matrix")
    matrix = [_vector(row, 3, label) for row in value]
    for row in range(3):
        norm = sum(matrix[row][col] * matrix[row][col] for col in range(3))
        if not math.isclose(norm, 1.0, abs_tol=1e-4):
            raise GraspGeometryError(f"{label} is not orthonormal")
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if not math.isclose(determinant, 1.0, abs_tol=1e-4):
        raise GraspGeometryError(f"{label} must have determinant +1")
    return matrix


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise GraspGeometryError(f"{label} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GraspGeometryError(f"{label} must be finite") from exc
    if not math.isfinite(parsed):
        raise GraspGeometryError(f"{label} must be finite")
    return parsed


def _positive_float(value: Any, label: str) -> float:
    parsed = _finite_float(value, label)
    if parsed <= 0:
        raise GraspGeometryError(f"{label} must be positive")
    return parsed


def _bounded_float(value: Any, label: str, lower: float, upper: float) -> float:
    parsed = _finite_float(value, label)
    if parsed < lower or parsed > upper:
        raise GraspGeometryError(f"{label} must be in [{lower}, {upper}]")
    return parsed


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise GraspGeometryError(f"{label} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise GraspGeometryError(f"{label} must be a non-negative integer") from exc
    if parsed < 0:
        raise GraspGeometryError(f"{label} must be a non-negative integer")
    return parsed


def _matmul3(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> list[list[float]]:
    return [
        [sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3)]
        for row in range(3)
    ]


def _matvec3(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3)]


def _add(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [left[index] + right[index] for index in range(3)]


def _normalise(vector: Sequence[float], label: str) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm < 1e-9:
        raise GraspGeometryError(f"{label} has zero length")
    return [value / norm for value in vector]


def _round_vector(vector: Sequence[float]) -> list[float]:
    return [0.0 if abs(value) < 1e-12 else round(float(value), 12) for value in vector]


def _round_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [_round_vector(row) for row in matrix]
