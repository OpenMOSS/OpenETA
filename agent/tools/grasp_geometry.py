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
WRIST_ALIGNMENT_OPERATING_REGION_SCHEMA = (
    "openeta.wrist_alignment_operating_region.v1"
)
TARGET_MASK_QUALITY_SCHEMA = "openeta.target_mask_quality.v1"
WRIST_VIEWPOINT_PROPOSAL_SCHEMA = "openeta.wrist_viewpoint_proposal.v1"
DEFAULT_GRASP_PROFILE = DEFAULT_GRASP_CALIBRATION_PROFILE
_OPENCV_TO_OPENGL = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
_PANDA_TOP_DOWN_ROTATION = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
_WORLD_NEGATIVE_Z = [0.0, 0.0, -1.0]
_MIN_SAFE_HOVER_DISTANCE_M = 0.15
_DEFAULT_REFINEMENT_HOVER_CLEARANCE_M = 0.20
_DEFAULT_WRIST_ALIGNMENT_MAX_REFERENCE_DISTANCE_M = 0.08
_DEFAULT_WRIST_ALIGNMENT_RESIDUAL_STEP_LIMIT_M = 0.02
_DEFAULT_WRIST_ALIGNMENT_RESIDUAL_CUMULATIVE_LIMIT_M = 0.10
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


def rebase_camera_grasp_candidate(
    candidate: Mapping[str, Any],
    *,
    source_camera_extrinsics: Mapping[str, Any],
    target_camera_extrinsics: Mapping[str, Any],
) -> JsonDict:
    """Express one OpenCV camera-frame grasp in another calibrated camera.

    The grasp itself is unchanged in world space.  This host-side transform is
    used when a moving wrist camera produced the best pickup candidate but a
    fixed scene camera owns the aligned object/placement masks required by
    AnyPlace.  It must never be used without exact packet-owned extrinsics for
    both cameras.
    """

    if str(candidate.get("frame") or "") != "camera":
        raise GraspGeometryError("candidate.frame must be 'camera'")
    if str(candidate.get("camera_frame") or "opencv").lower() != "opencv":
        raise GraspGeometryError("candidate.camera_frame must be 'opencv'")
    source_rotation = _rotation(
        candidate.get("rotation_matrix"), "candidate.rotation_matrix"
    )
    source_translation = _vector(
        candidate.get("translation_xyz"), 3, "candidate.translation_xyz"
    )
    source_tip = _vector(
        candidate.get("gripper_tip_position_xyz"),
        3,
        "candidate.gripper_tip_position_xyz",
    )
    source_execution_center = _vector(
        candidate.get("execution_contact_center_xyz", source_translation),
        3,
        "candidate.execution_contact_center_xyz",
    )
    world_from_source, world_source_position = _opencv_camera_to_world(
        source_camera_extrinsics
    )
    world_from_target, world_target_position = _opencv_camera_to_world(
        target_camera_extrinsics
    )
    target_from_world = _transpose3(world_from_target)

    world_rotation = _matmul3(world_from_source, source_rotation)
    target_rotation = _matmul3(target_from_world, world_rotation)

    def rebase_point(point: Sequence[float]) -> list[float]:
        world_point = _add(_matvec3(world_from_source, point), world_source_position)
        relative_world = [
            world_point[index] - world_target_position[index] for index in range(3)
        ]
        return _matvec3(target_from_world, relative_world)

    rebased = json.loads(json.dumps(dict(candidate), ensure_ascii=False))
    rebased_translation = _round_vector(rebase_point(source_translation))
    rebased_rotation = _round_matrix(target_rotation)
    rebased["translation_xyz"] = rebased_translation
    rebased["gripper_tip_position_xyz"] = _round_vector(rebase_point(source_tip))
    rebased["execution_contact_center_xyz"] = _round_vector(
        rebase_point(source_execution_center)
    )
    rebased["rotation_matrix"] = rebased_rotation
    if "transform_matrix" in rebased:
        rebased["transform_matrix"] = [
            [*rebased_rotation[row], rebased_translation[row]] for row in range(3)
        ] + [[0.0, 0.0, 0.0, 1.0]]
    # This nested pose uses the estimator's native grasp basis in the original
    # source camera.  It is rendering provenance, not executable geometry, and
    # retaining it after a cross-camera rebase would publish contradictory
    # coordinate frames inside one candidate.
    rebased.pop("model_native_grasp_pose", None)
    rebased["frame"] = "camera"
    rebased["camera_frame"] = "opencv"
    return rebased


def rebase_camera_direction(
    direction: Sequence[float],
    *,
    source_camera_extrinsics: Mapping[str, Any],
    target_camera_extrinsics: Mapping[str, Any],
) -> list[float]:
    """Express a direction vector in another calibrated OpenCV camera."""

    source_direction = _normalise(
        _vector(direction, 3, "camera_direction"), "camera_direction"
    )
    world_from_source, _ = _opencv_camera_to_world(source_camera_extrinsics)
    world_from_target, _ = _opencv_camera_to_world(target_camera_extrinsics)
    target_direction = _matvec3(
        _transpose3(world_from_target),
        _matvec3(world_from_source, source_direction),
    )
    return _round_vector(_normalise(target_direction, "rebased_camera_direction"))


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
        refinement = outputs.get("execution_guidance", {}).get(
            "near_field_refinement", {}
        )
        refinement_note = (
            "; near-field wrist refinement is recommended by current mask/candidate "
            "evidence before first close"
            if isinstance(refinement, Mapping)
            and refinement.get("recommended") is True
            else ""
        )
        strategy_note = (
            "; the explicitly selected strategy remains the active contact branch; "
            "a later wrist estimate is separate raw evidence and must not silently "
            "replace or inherit it"
            if outputs.get("strategy_selection") == "explicit"
            else ""
        )
        return make_tool_result(
            context,
            success=True,
            content=(
                "normalized grasp seed compiled to world-frame EEF references; "
                "for the precision-critical contact reference, start with "
                "move_to tolerance=0.005 m and inspect the returned actual pose plus "
                "fresh wrist image before closing"
                + refinement_note
                + strategy_note
            ),
            outputs=outputs,
        )

    return handler


def build_wrist_alignment_handler(
    profile_path: str | Path = DEFAULT_GRASP_PROFILE,
) -> ToolHandler:
    """Build a wrist alignment calculator bound to one embodiment profile."""

    resolved_profile = Path(profile_path)

    def handler(context: ToolExecutionContext) -> ToolResult:
        try:
            profile, profile_sha256 = _load_profile(resolved_profile)
            outputs = compute_wrist_alignment(
                context.parameters,
                profile=profile,
                profile_sha256=profile_sha256,
            )
        except (OSError, json.JSONDecodeError, GraspGeometryError) as exc:
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
        operating_region = outputs.get("operating_region")
        if outputs.get("executable_reference") is not True:
            failed_checks = (
                operating_region.get("failed_checks", [])
                if isinstance(operating_region, dict)
                else []
            )
            return make_tool_result(
                context,
                success=True,
                content=(
                    "wrist alignment is non-executable; the geometry calculation "
                    "produced no motion pose. Do not repeat SAM3 on unchanged wrist "
                    "pixels. If the selected target mask is usable and the target is "
                    "visible, consume the ready wrist grasp_pose_estimate bundle for "
                    "a full 6-DoF re-estimate now. Reposition only when the target is "
                    "clipped/out of view or that bundle is unavailable"
                ),
                outputs=outputs,
                diagnostics=[
                    {
                        "code": "wrist_alignment_outside_operating_region",
                        "message": (
                            "one or more geometric work-envelope checks failed; inspect "
                            "outputs.operating_region.failed_checks"
                        ),
                        "failed_checks": failed_checks,
                    }
                ],
                semantic_outcome="requires_better_view",
                recovery_options=[
                    {
                        "action": "run_full_wrist_grasp_estimate",
                        "reason": (
                            "reuse the already selected fresh wrist mask through the "
                            "host-resolved grasp bundle when orientation, approach "
                            "direction, axial depth, or a clamped correction is uncertain; "
                            "do not segment the unchanged image again"
                        ),
                    },
                    {
                        "action": "gather_fresh_near_field_wrist_view",
                        "reason": (
                            "only when current evidence is clipped, out of view, or has "
                            "no ready full-estimate bundle, choose a new safe target-facing "
                            "view and then ground the same target instance"
                        ),
                    },
                ],
            )
        return make_tool_result(
            context,
            success=True,
            content="bounded wrist alignment correction computed inside its work envelope",
            outputs=outputs,
        )

    return handler


def build_wrist_viewpoint_proposal_handler() -> ToolHandler:
    """Build a read-only target-facing wrist-viewpoint proposer."""

    def handler(context: ToolExecutionContext) -> ToolResult:
        try:
            outputs = propose_wrist_viewpoints(context.parameters)
        except (OSError, GraspGeometryError) as exc:
            return make_tool_result(
                context,
                success=False,
                content=f"wrist viewpoint proposal failed: {exc}",
                outputs={"reason": "wrist_viewpoint_proposal_failed"},
                diagnostics=[
                    {
                        "code": "wrist_viewpoint_proposal_failed",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            )
        return make_tool_result(
            context,
            success=True,
            content=(
                "target-facing wrist viewpoint candidates proposed; preview one exact "
                "candidate with ik_preview_check before moving. The proposal remains "
                "valid across packet-id-only refreshes while its geometry epochs stay "
                "unchanged; consume it instead of requesting it again. After reaching "
                "the viewpoint, use the fresh post-motion wrist packet for target "
                "segmentation and alignment or a full wrist grasp estimate; reaching "
                "the viewpoint alone does not refine the older contact pose."
            ),
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
    articulated_options = parameters.get("articulated_handle_options")
    if articulated_options is not None and not isinstance(articulated_options, Mapping):
        raise GraspGeometryError("articulated_handle_options must be an object")
    nested_approach_mode = (
        articulated_options.get("approach_mode")
        if isinstance(articulated_options, Mapping)
        else None
    )
    if parameters.get("approach_mode") is not None and nested_approach_mode is not None:
        raise GraspGeometryError(
            "provide approach_mode only inside articulated_handle_options"
        )
    # The top-level spelling remains a host-side compatibility input for stored
    # artifacts; it is intentionally absent from the planner-facing ToolSpec.
    approach_mode = str(
        nested_approach_mode
        if nested_approach_mode is not None
        else parameters.get("approach_mode")
        or ""
    ).strip().lower()
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
    _validate_profile(profile, target_class=target_geometry_family)

    candidate_id = str(candidate.get("id") or "").strip()
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
    if width < width_bounds[0] or width > width_bounds[1]:
        raise GraspCandidateRejected(
            f"candidate width {width:.4f} m is outside active strategy bounds "
            f"[{width_bounds[0]:.4f}, {width_bounds[1]:.4f}]",
            rejection_code="strategy_width_out_of_bounds",
            recovery_class="perception_refinable",
        )

    r_camera_grasp = _rotation(candidate.get("rotation_matrix"), "camera_pose.rotation_matrix")
    # AnyGrasp and GraspGenX expose different native origins. The normalized
    # candidate resolves the physical grip-site contact point explicitly.
    execution_reference_point = str(
        candidate.get("execution_reference_point") or "translation_xyz"
    )
    p_camera_grasp = _vector(
        candidate.get(
            "execution_contact_center_xyz",
            candidate.get(execution_reference_point),
        ),
        3,
        "camera_pose.execution_contact_center_xyz",
    )
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
        if min_alignment is not None and native_downward_alignment < float(min_alignment):
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
    target_mask_quality = parameters.get("target_mask_quality")
    target_mask_quality = (
        dict(target_mask_quality)
        if isinstance(target_mask_quality, Mapping)
        else {}
    )
    mask_area_fraction = target_mask_quality.get("area_fraction")
    mask_area_fraction = (
        float(mask_area_fraction)
        if isinstance(mask_area_fraction, int | float)
        and not isinstance(mask_area_fraction, bool)
        and math.isfinite(float(mask_area_fraction))
        else None
    )
    source_camera_frame = str(parameters.get("camera_frame_id") or "")
    source_is_wrist = "wrist" in source_camera_frame.lower()
    refinement_reasons: list[str] = []
    if (
        not source_is_wrist
        and strategy is None
        and mask_area_fraction is not None
        and mask_area_fraction < 0.02
    ):
        refinement_reasons.append(
            "scene-view target mask occupies less than 2% of the image"
        )
    selection_advice = parameters.get("grasp_selection_advice")
    selection_advice = (
        dict(selection_advice) if isinstance(selection_advice, Mapping) else {}
    )
    if (
        not source_is_wrist
        and strategy is None
        and str(selection_advice.get("status") or "") == "skipped_single_candidate"
    ):
        refinement_reasons.append(
            "only one host-executable scene-view candidate was available for ranking"
        )
    near_field_refinement = {
        "recommended": bool(refinement_reasons),
        "reasons": refinement_reasons,
        "source_camera_frame_id": source_camera_frame,
        "target_mask_area_fraction": mask_area_fraction,
        "grasp_candidate_count": _nonnegative_int(
            parameters.get("grasp_candidate_count", 0),
            "grasp_candidate_count",
        ),
        "agent_discretion": True,
        "active_strategy_id": (
            strategy.get("strategy_id") if strategy is not None else None
        ),
        "strategy_continuity": (
            "This explicitly selected strategy remains the active contact branch. "
            "A later wrist grasp estimate is separate raw evidence: it neither "
            "inherits nor silently replaces this strategy. If the observed object "
            "posture still satisfies the strategy geometry, explicitly pass the "
            "same strategy_id when compiling a wrist candidate; otherwise abandon "
            "or change the strategy using fresh visual evidence."
            if strategy is not None
            else None
        ),
        "suggested_actions": [
            {
                "tool": "compute_wrist_alignment",
                "when": (
                    "fresh wrist mask is complete and only lateral contact placement "
                    "needs correction"
                ),
            },
            {
                "tool": "grasp_pose_estimate",
                "when": (
                    "fresh wrist evidence makes approach direction, orientation, or "
                    "contact depth uncertain"
                ),
            },
        ],
        "interpretation": (
            "This is evidence-based workflow advice, not a required task phase or "
            "movement authorization. If skipped, verify and state why current "
            "dual-view evidence already supports contact."
        ),
    }
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
            "waypoint_role": "grasp_precontact",
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
        "execution_reference_point": execution_reference_point,
        "calibration_status": str(profile.get("status") or ""),
        "not_validated": profile.get("status") != "validated",
        "profile_sha256": profile_sha256,
        "target_anchor_world_xyz": _round_vector(p_world_grasp),
        "approach_world_xyz": _round_vector(approach_world),
        "native_downward_alignment": round(native_downward_alignment, 6),
        "hover_offset_world_xyz": _round_vector(
            [-pregrasp_distance * component for component in approach_world]
        ),
        "gripper_width_m": width,
        "requested_pregrasp_distance_m": requested_pregrasp_distance,
        "pregrasp_distance_m": pregrasp_distance,
        "orientation_clamped": orientation_clamped,
        "strategy_id": strategy.get("strategy_id") if strategy is not None else None,
        "strategy_status": strategy.get("status") if strategy is not None else None,
        "strategy_selection": strategy_selection,
        "outside_validated_strategy_scope": (
            strategy is None or strategy.get("status") != "validated"
        ),
        "hover_pose": {
            **pose_common,
            "xyz": _round_vector(p_hover),
            "waypoint_role": "grasp_clearance",
        },
        "contact_pose": {
            **pose_common,
            "xyz": _round_vector(p_world_eef),
            "waypoint_role": "grasp_contact",
        },
        "precontact_pose": precontact_pose,
        "execution_guidance": {
            "schema_version": "openeta.compiled_grasp_motion_guidance.v1",
            "clearance": {
                "recommended_position_tolerance_m": 0.01,
                "interpretation": "coarse collision-clearance waypoint",
            },
            "contact": {
                "recommended_position_tolerance_m": 0.005,
                "recommended_orientation_tolerance_rad": 0.10,
                "precision_metric": "max_axis_absolute_error",
                "verify_fresh_wrist_contact_before_close": True,
                "interpretation": (
                    "A controller-level reached_target receipt at a looser tolerance "
                    "does not by itself establish object-relative finger engagement."
                ),
            },
            "lift_probe": {
                "recommended_position_tolerance_m": 0.01,
                "interpretation": (
                    "small Agent-chosen probe; attachment still requires visual "
                    "co-motion and source-vacancy evidence"
                ),
            },
            "near_field_refinement": near_field_refinement,
        },
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
            "Calibration and strategy outputs are geometric references; the Agent "
            "owns candidate choice, waypoint sequencing, and recovery."
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


def world_up_direction_camera(camera_extrinsics: Mapping[str, Any]) -> list[float]:
    """Express world +Z in the source camera's OpenCV coordinate frame."""

    r_world_cv, _ = _opencv_camera_to_world(camera_extrinsics)
    return _normalise(
        [r_world_cv[2][column] for column in range(3)],
        "world up in camera frame",
    )


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
        "waypoint_role": "grasp_refinement_clearance",
        "source_grasp_id": candidate_id,
        "recovery_id": recovery_id,
        "scene_epoch": _nonnegative_int(scene_epoch, "scene_epoch"),
    }


def assess_target_mask_quality(
    mask_path: str | Path,
    *,
    depth_path: str | Path | None = None,
    minimum_valid_depth_pixels: int = 5,
    minimum_depth_coverage: float = 0.5,
    minimum_border_clearance_fraction: float = 0.02,
) -> JsonDict:
    """Assess whether a selected mask is complete enough for targeted geometry.

    The verdict treats image-boundary contact or only a sliver of border margin
    as a hard visibility failure. A nearly full-frame mask can otherwise include
    foreground gripper/table pixels while appearing technically unclipped. Area
    and depth statistics remain explicit so callers and the Agent can distinguish
    a clipped view from a tiny or depth-poor target.
    """

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - runtime dependency is required.
        raise GraspGeometryError("Pillow is required for target-mask quality checks") from exc

    resolved_mask = Path(mask_path)
    if not resolved_mask.is_file():
        raise GraspGeometryError("target mask must be an existing local file")
    resolved_depth = Path(depth_path) if depth_path is not None else None
    if resolved_depth is not None and not resolved_depth.is_file():
        raise GraspGeometryError("aligned depth must be an existing local file")

    with Image.open(resolved_mask) as mask_image:
        mask = mask_image.convert("L")
        width, height = mask.size
        mask_values = list(mask.get_flattened_data())
    foreground_indices = [
        index for index, value in enumerate(mask_values) if int(value) > 0
    ]
    if not foreground_indices:
        return {
            "schema_version": TARGET_MASK_QUALITY_SCHEMA,
            "status": "empty_mask",
            "usable_for_targeted_geometry": False,
            "image_size_wh": [width, height],
            "foreground_pixel_count": 0,
            "area_fraction": 0.0,
            "bbox_xyxy": None,
            "border_clearance_px": None,
            "touches_image_boundary": False,
            "valid_depth_pixel_count": 0 if resolved_depth is not None else None,
            "depth_coverage": 0.0 if resolved_depth is not None else None,
            "failed_checks": ["mask_nonempty"],
        }

    xs = [index % width for index in foreground_indices]
    ys = [index // width for index in foreground_indices]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    border_clearance = min(min_x, min_y, width - 1 - max_x, height - 1 - max_y)
    border_clearance_fraction = border_clearance / max(1, min(width, height))
    valid_depth_count: int | None = None
    depth_coverage: float | None = None
    if resolved_depth is not None:
        with Image.open(resolved_depth) as depth_image:
            depth = depth_image.convert("I")
            if depth.size != (width, height):
                raise GraspGeometryError("target mask and aligned depth dimensions differ")
            depth_values = list(depth.get_flattened_data())
        valid_depth_count = sum(
            1
            for index in foreground_indices
            if float(depth_values[index]) > 0 and math.isfinite(float(depth_values[index]))
        )
        depth_coverage = valid_depth_count / len(foreground_indices)

    failed_checks: list[str] = []
    if border_clearance <= 0:
        failed_checks.append("mask_not_clipped")
    elif border_clearance_fraction < minimum_border_clearance_fraction:
        failed_checks.append("minimum_border_clearance")
    if valid_depth_count is not None and valid_depth_count < minimum_valid_depth_pixels:
        failed_checks.append("minimum_valid_depth_pixels")
    if depth_coverage is not None and depth_coverage < minimum_depth_coverage:
        failed_checks.append("minimum_depth_coverage")
    status = (
        "clipped_mask"
        if {"mask_not_clipped", "minimum_border_clearance"}.intersection(failed_checks)
        else "insufficient_depth"
        if failed_checks
        else "usable"
    )
    return {
        "schema_version": TARGET_MASK_QUALITY_SCHEMA,
        "status": status,
        "usable_for_targeted_geometry": not failed_checks,
        "image_size_wh": [width, height],
        "foreground_pixel_count": len(foreground_indices),
        "area_fraction": round(len(foreground_indices) / (width * height), 8),
        # SAM3 and public tool contracts use right/bottom-exclusive boxes.
        "bbox_xyxy": [min_x, min_y, max_x + 1, max_y + 1],
        "bbox_fill_fraction": round(
            len(foreground_indices)
            / ((max_x - min_x + 1) * (max_y - min_y + 1)),
            8,
        ),
        "border_clearance_px": border_clearance,
        "border_clearance_fraction": round(
            border_clearance_fraction, 8
        ),
        "minimum_border_clearance_fraction": round(
            minimum_border_clearance_fraction, 8
        ),
        "touches_image_boundary": border_clearance <= 0,
        "valid_depth_pixel_count": valid_depth_count,
        "depth_coverage": round(depth_coverage, 8) if depth_coverage is not None else None,
        "failed_checks": failed_checks,
    }


def propose_wrist_viewpoints(parameters: Mapping[str, Any]) -> JsonDict:
    """Generate target-facing wrist-camera poses from current eye-in-hand geometry."""

    compiled = _mapping(parameters.get("compiled_grasp"), "compiled_grasp")
    if compiled.get("schema_version") != COMPILED_GRASP_SCHEMA:
        raise GraspGeometryError("compiled_grasp has an unsupported schema")
    current_pose = _mapping(parameters.get("current_eef_pose"), "current_eef_pose")
    camera_extrinsics = _mapping(
        parameters.get("camera_extrinsics"), "camera_extrinsics"
    )
    object_epoch = _nonnegative_int(
        parameters.get("object_scene_epoch"), "object_scene_epoch"
    )
    robot_epoch = _nonnegative_int(
        parameters.get("robot_motion_epoch"), "robot_motion_epoch"
    )
    compiled_epoch = _nonnegative_int(
        compiled.get("scene_epoch"), "compiled_grasp.scene_epoch"
    )
    if compiled_epoch != object_epoch:
        raise GraspGeometryError(
            "compiled grasp belongs to a stale object-scene epoch; re-estimate the target"
        )

    target_anchor = compiled.get("target_anchor_world_xyz")
    if target_anchor is None:
        contact = _mapping(compiled.get("contact_pose"), "compiled_grasp.contact_pose")
        target_anchor = contact.get("xyz")
    p_world_target = _vector(target_anchor, 3, "compiled_grasp.target_anchor_world_xyz")
    p_world_eef = _vector(current_pose.get("xyz"), 3, "current_eef_pose.xyz")
    r_world_eef, orientation_source = _eef_rotation(current_pose)
    r_world_camera, p_world_camera, extrinsics_source = (
        _resolve_current_opencv_camera_to_world(
            camera_extrinsics,
            p_world_eef=p_world_eef,
            r_world_eef=r_world_eef,
        )
    )
    r_eef_camera = _matmul3(_transpose3(r_world_eef), r_world_camera)
    p_eef_camera = _matvec3(
        _transpose3(r_world_eef),
        [p_world_camera[index] - p_world_eef[index] for index in range(3)],
    )

    standoffs_value = parameters.get("standoff_m", [0.18, 0.22])
    if isinstance(standoffs_value, (int, float)) and not isinstance(standoffs_value, bool):
        standoffs_value = [standoffs_value]
    if not isinstance(standoffs_value, Sequence) or isinstance(
        standoffs_value, (str, bytes)
    ):
        raise GraspGeometryError("standoff_m must be one number or a list of numbers")
    standoffs = [
        _bounded_float(value, "standoff_m", 0.12, 0.35)
        for value in list(standoffs_value)[:3]
    ]
    if not standoffs:
        raise GraspGeometryError("standoff_m must contain at least one value")

    # Candidate offsets are observation choices, not a hidden motion sequence.
    lateral_offsets = [0.0, -0.04, 0.04]
    current_camera_x = [r_world_camera[row][0] for row in range(3)]
    candidates: list[JsonDict] = []
    for standoff in standoffs:
        for lateral in lateral_offsets:
            p_world_camera_goal = [
                p_world_target[0] + lateral,
                p_world_target[1],
                p_world_target[2] + standoff,
            ]
            optical_forward = _normalise(
                [
                    p_world_target[index] - p_world_camera_goal[index]
                    for index in range(3)
                ],
                "target-facing optical axis",
            )
            camera_x = _orthogonal_unit(current_camera_x, optical_forward)
            camera_y = _cross3(optical_forward, camera_x)
            r_world_camera_goal = [
                [camera_x[row], camera_y[row], optical_forward[row]]
                for row in range(3)
            ]
            r_world_eef_goal = _matmul3(
                r_world_camera_goal,
                _transpose3(r_eef_camera),
            )
            p_world_eef_goal = [
                p_world_camera_goal[index]
                - _matvec3(r_world_eef_goal, p_eef_camera)[index]
                for index in range(3)
            ]
            candidate_id = f"wrist_view_{len(candidates):02d}"
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "target_pose": {
                        "frame": "world",
                        "xyz": _round_vector(p_world_eef_goal),
                        "rotation_matrix": _round_matrix(r_world_eef_goal),
                        "waypoint_role": "wrist_observation_viewpoint",
                        "viewpoint_candidate_id": candidate_id,
                        "camera_frame_id": parameters.get("camera_frame_id"),
                        "compiled_grasp_id": compiled.get("compiled_grasp_id"),
                        "object_scene_epoch": object_epoch,
                    },
                    "camera_goal": {
                        "xyz": _round_vector(p_world_camera_goal),
                        "optical_forward_world_xyz": _round_vector(optical_forward),
                        "target_anchor_world_xyz": _round_vector(p_world_target),
                        "standoff_m": round(standoff, 6),
                        "lateral_offset_m": round(lateral, 6),
                    },
                    "requires_ik_preview": True,
                }
            )

    proposal_id = "wrist_viewpoint:" + hashlib.sha256(
        json.dumps(
            {
                "compiled_grasp_id": compiled.get("compiled_grasp_id"),
                "object_scene_epoch": object_epoch,
                "robot_motion_epoch": robot_epoch,
                "camera_frame_id": parameters.get("camera_frame_id"),
                "target_anchor_world_xyz": _round_vector(p_world_target),
                "current_eef_xyz": _round_vector(p_world_eef),
                "camera_mount_translation": _round_vector(p_eef_camera),
                "standoffs": standoffs,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:20]

    return {
        "schema_version": WRIST_VIEWPOINT_PROPOSAL_SCHEMA,
        "proposal_id": proposal_id,
        "compiled_grasp_id": compiled.get("compiled_grasp_id"),
        "source_packet_id": parameters.get("source_packet_id"),
        "camera_frame_id": parameters.get("camera_frame_id"),
        "object_scene_epoch": object_epoch,
        "robot_motion_epoch": robot_epoch,
        "target_anchor_world_xyz": _round_vector(p_world_target),
        "camera_mount": {
            "eef_orientation_source": orientation_source,
            "camera_extrinsics_source": extrinsics_source,
            "eef_to_camera_translation_xyz": _round_vector(p_eef_camera),
            "eef_to_camera_rotation_matrix": _round_matrix(r_eef_camera),
        },
        "candidates": candidates,
        "selection_policy": (
            "Agent chooses one candidate using workspace/collision evidence, runs an "
            "exact full-pose ik_preview_check, then moves only if that preview supports it"
        ),
        "validity": {
            "reusable_across_observation_packet_refresh": True,
            "invalidated_by": [
                "compiled_grasp_id_change",
                "object_scene_epoch_change",
                "robot_motion_epoch_change",
                "camera_mount_change",
            ],
            "interpretation": (
                "A newer source_packet_id alone does not invalidate this proposal. "
                "Reuse these candidates until the named geometry evidence changes."
            ),
        },
        "next_action_contract": {
            "consume_existing_proposal": True,
            "recommended_tool": "ik_preview_check",
            "parameter": "target_pose",
            "instruction": (
                "Copy one candidate.target_pose exactly into ik_preview_check. Do not "
                "call propose_wrist_viewpoints again only because a read-only turn "
                "created a newer observation packet."
            ),
        },
        "post_reach_evidence_contract": {
            "schema_version": "openeta.wrist_viewpoint_post_reach.v1",
            "agent_discretion": True,
            "viewpoint_reached_is_not_contact_refined": True,
            "fresh_packet_source": "current_observation.source_packet_id_after_move",
            "camera_frame_id": parameters.get("camera_frame_id"),
            "recommended_sequence": [
                {
                    "tool": "sam3",
                    "instruction": (
                        "segment the same target on the fresh post-motion wrist packet"
                    ),
                },
                {
                    "tool": "select_sam3_detection",
                    "instruction": (
                        "confirm cross-view identity continuity for the wrist mask"
                    ),
                },
                {
                    "tool": "compute_wrist_alignment_or_grasp_pose_estimate",
                    "instruction": (
                        "use bounded lateral alignment when orientation/depth remain "
                        "credible; otherwise run a full wrist-view grasp estimate"
                    ),
                },
            ],
            "interpretation": (
                "Reaching an observation viewpoint only gathers better evidence. Do "
                "not treat the older scene-view contact pose as newly verified merely "
                "because the viewpoint motion succeeded."
            ),
        },
    }


def compute_wrist_alignment(
    parameters: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
    profile_sha256: str,
) -> JsonDict:
    compiled = _mapping(parameters.get("compiled_grasp"), "compiled_grasp")
    if compiled.get("schema_version") != COMPILED_GRASP_SCHEMA:
        raise GraspGeometryError("compiled_grasp has an unsupported schema")
    calibration_id = str(profile.get("calibration_id") or "").strip()
    if not calibration_id:
        raise GraspGeometryError("calibration profile calibration_id is required")
    if str(compiled.get("calibration_id") or "") != calibration_id:
        raise GraspGeometryError(
            "compiled_grasp calibration_id does not match the loaded profile"
        )
    if str(compiled.get("profile_sha256") or "") != profile_sha256:
        raise GraspGeometryError(
            "compiled_grasp profile_sha256 does not match the loaded profile"
        )
    if "desired_pixel_xy" in parameters:
        raise GraspGeometryError(
            "desired_pixel_xy is host-derived from calibration and must not be supplied"
        )
    target_mask = Path(str(parameters.get("target_mask") or ""))
    depth_path = Path(str(parameters.get("depth") or ""))
    if not target_mask.is_file() or not depth_path.is_file():
        raise GraspGeometryError("target_mask and depth must be existing local files")
    intrinsics = _mapping(parameters.get("intrinsics"), "intrinsics")
    fx = _positive_float(intrinsics.get("fx"), "intrinsics.fx")
    fy = _positive_float(intrinsics.get("fy"), "intrinsics.fy")
    _finite_float(intrinsics.get("cx"), "intrinsics.cx")
    _finite_float(intrinsics.get("cy"), "intrinsics.cy")
    scale = _positive_float(intrinsics.get("scale", 1000.0), "intrinsics.scale")
    current_pose = _mapping(parameters.get("current_eef_pose"), "current_eef_pose")
    camera_extrinsics = _mapping(
        parameters.get("camera_extrinsics"), "camera_extrinsics"
    )
    desired_xy, gripper_projection, r_world_cv = _project_configured_gripper_center(
        profile=profile,
        profile_sha256=profile_sha256,
        current_eef_pose=current_pose,
        camera_extrinsics=camera_extrinsics,
        intrinsics=intrinsics,
    )
    max_correction = _bounded_float(
        parameters.get("max_correction_m", 0.03),
        "max_correction_m",
        0.005,
        0.05,
    )
    max_reference_distance = _bounded_float(
        parameters.get(
            "max_reference_distance_m",
            _DEFAULT_WRIST_ALIGNMENT_MAX_REFERENCE_DISTANCE_M,
        ),
        "max_reference_distance_m",
        0.01,
        0.20,
    )
    scene_epoch = _nonnegative_int(parameters.get("scene_epoch"), "scene_epoch")
    source_object_epoch = _nonnegative_int(
        parameters.get("source_object_scene_epoch", scene_epoch),
        "source_object_scene_epoch",
    )
    current_object_epoch = _nonnegative_int(
        parameters.get("current_object_scene_epoch", scene_epoch),
        "current_object_scene_epoch",
    )
    source_robot_epoch = _nonnegative_int(
        parameters.get("source_robot_motion_epoch", 0),
        "source_robot_motion_epoch",
    )
    current_robot_epoch = _nonnegative_int(
        parameters.get("current_robot_motion_epoch", source_robot_epoch),
        "current_robot_motion_epoch",
    )

    alignment_policy = compiled.get("alignment_policy")
    alignment_policy = alignment_policy if isinstance(alignment_policy, dict) else {}
    target_region = str(alignment_policy.get("target_region") or "mask_centroid")
    u, v, depth_m, width, height, mask_geometry = _mask_depth_target(
        target_mask,
        depth_path,
        scale=scale,
        desired_xy=desired_xy,
        target_region=target_region,
    )
    if not math.isclose(
        _positive_float(intrinsics.get("width"), "intrinsics.width"),
        float(width),
    ) or not math.isclose(
        _positive_float(intrinsics.get("height"), "intrinsics.height"),
        float(height),
    ):
        raise GraspGeometryError(
            "wrist intrinsics width/height do not match the mask and depth images"
        )
    if not (0 <= desired_xy[0] < width and 0 <= desired_xy[1] < height):
        raise GraspGeometryError("desired_pixel_xy is outside the image")
    delta_camera = [
        (u - desired_xy[0]) * depth_m / fx,
        (v - desired_xy[1]) * depth_m / fy,
        0.0,
    ]
    raw_delta_world = _matvec3(r_world_cv, delta_camera)
    raw_correction_norm = math.sqrt(
        sum(value * value for value in raw_delta_world)
    )
    delta_world = list(raw_delta_world)
    if raw_correction_norm > max_correction:
        scale_factor = max_correction / raw_correction_norm
        delta_world = [value * scale_factor for value in raw_delta_world]
    residual_px = math.hypot(u - desired_xy[0], v - desired_xy[1])

    current_xyz = _vector(current_pose.get("xyz"), 3, "current_eef_pose.xyz")
    hover_pose = _mapping(compiled.get("hover_pose"), "compiled_grasp.hover_pose")
    hover_xyz = _vector(hover_pose.get("xyz"), 3, "compiled_grasp.hover_pose.xyz")
    reference_distance = math.sqrt(
        sum((current_xyz[index] - hover_xyz[index]) ** 2 for index in range(3))
    )
    contact_pose = _mapping(compiled.get("contact_pose"), "compiled_grasp.contact_pose")
    contact_xyz = _vector(contact_pose.get("xyz"), 3, "compiled_grasp.contact_pose.xyz")
    aligned_hover = dict(contact_pose)
    aligned_hover.update(
        {
            "xyz": _round_vector(_add(current_xyz, delta_world)),
            "waypoint_role": "grasp_alignment_reference",
            "alignment_id": "",
        }
    )
    adjusted_contact = dict(contact_pose)
    adjusted_contact.update(
        {
            "xyz": _round_vector(_add(contact_xyz, delta_world)),
            "waypoint_role": "grasp_contact",
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
                "waypoint_role": "grasp_precontact",
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
                "gripper_center_projection": gripper_projection,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:20]
    aligned_hover["alignment_id"] = alignment_id
    adjusted_contact["alignment_id"] = alignment_id
    if adjusted_precontact is not None:
        adjusted_precontact["alignment_id"] = alignment_id

    residual_budget = parameters.get("residual_budget")
    residual_budget = residual_budget if isinstance(residual_budget, Mapping) else {}
    per_call_limit = _positive_float(
        residual_budget.get(
            "per_call_limit_m", _DEFAULT_WRIST_ALIGNMENT_RESIDUAL_STEP_LIMIT_M
        ),
        "residual_budget.per_call_limit_m",
    )
    cumulative_limit = _positive_float(
        residual_budget.get(
            "cumulative_limit_m",
            _DEFAULT_WRIST_ALIGNMENT_RESIDUAL_CUMULATIVE_LIMIT_M,
        ),
        "residual_budget.cumulative_limit_m",
    )
    cumulative_translation = _finite_float(
        residual_budget.get("cumulative_translation_m", 0.0),
        "residual_budget.cumulative_translation_m",
    )
    if cumulative_translation < 0.0:
        raise GraspGeometryError(
            "residual_budget.cumulative_translation_m must be non-negative"
        )
    previous_residual = _vector(
        residual_budget.get("last_residual_xyz_m", [0.0, 0.0, 0.0]),
        3,
        "residual_budget.last_residual_xyz_m",
    )
    pose_residuals = {
        "aligned_hover_pose": [
            float(aligned_hover["xyz"][index]) - hover_xyz[index]
            for index in range(3)
        ],
        "adjusted_contact_pose": list(delta_world),
    }
    if adjusted_precontact is not None:
        pose_residuals["adjusted_precontact_pose"] = list(delta_world)
    residual_steps = {
        name: math.sqrt(
            sum(
                (residual[index] - previous_residual[index]) ** 2
                for index in range(3)
            )
        )
        for name, residual in pose_residuals.items()
    }
    max_residual_step = max(residual_steps.values())
    projected_cumulative = cumulative_translation + max_residual_step

    checks = [
        {
            "code": "object_scene_epoch_current",
            "passed": (
                source_object_epoch == current_object_epoch
                and _nonnegative_int(compiled.get("scene_epoch"), "compiled_grasp.scene_epoch")
                == current_object_epoch
            ),
            "source_epoch": source_object_epoch,
            "compiled_epoch": compiled.get("scene_epoch"),
            "current_epoch": current_object_epoch,
        },
        {
            "code": "robot_motion_epoch_current",
            "passed": source_robot_epoch == current_robot_epoch,
            "source_epoch": source_robot_epoch,
            "current_epoch": current_robot_epoch,
        },
        {
            "code": "target_mask_not_clipped",
            "passed": mask_geometry["touches_image_boundary"] is False,
            "bbox_xyxy": mask_geometry["bbox_xyxy"],
            "border_clearance_px": mask_geometry["border_clearance_px"],
        },
        {
            "code": "eef_near_compiled_clearance",
            "passed": reference_distance <= max_reference_distance,
            "distance_m": round(reference_distance, 6),
            "max_distance_m": round(max_reference_distance, 6),
            "current_eef_xyz": _round_vector(current_xyz),
            "compiled_clearance_xyz": _round_vector(hover_xyz),
        },
        {
            "code": "correction_within_limit",
            "passed": raw_correction_norm <= max_correction,
            "raw_correction_m": round(raw_correction_norm, 6),
            "max_correction_m": round(max_correction, 6),
        },
        {
            "code": "aligned_references_within_residual_budget",
            "passed": (
                max_residual_step <= per_call_limit + 1e-6
                and projected_cumulative <= cumulative_limit + 1e-6
            ),
            "per_pose_increment_m": {
                name: round(value, 6) for name, value in residual_steps.items()
            },
            "max_increment_m": round(max_residual_step, 6),
            "per_call_limit_m": round(per_call_limit, 6),
            "projected_cumulative_m": round(projected_cumulative, 6),
            "cumulative_limit_m": round(cumulative_limit, 6),
        },
    ]
    failed_checks = [check for check in checks if check["passed"] is not True]
    executable_reference = not failed_checks
    operating_region = {
        "schema_version": WRIST_ALIGNMENT_OPERATING_REGION_SCHEMA,
        "status": (
            "within_operating_region"
            if executable_reference
            else "requires_better_view"
        ),
        "checks": checks,
        "failed_checks": failed_checks,
    }
    return {
        "schema_version": WRIST_ALIGNMENT_SCHEMA,
        "status": (
            "aligned_reference_ready"
            if executable_reference
            else "requires_better_view"
        ),
        "executable_reference": executable_reference,
        "alignment_id": alignment_id,
        "compiled_grasp_id": compiled.get("compiled_grasp_id"),
        "candidate_id": compiled.get("candidate_id"),
        "scene_epoch": scene_epoch,
        "source_object_scene_epoch": source_object_epoch,
        "current_object_scene_epoch": current_object_epoch,
        "source_robot_motion_epoch": source_robot_epoch,
        "current_robot_motion_epoch": current_robot_epoch,
        "target_pixel_xy": [round(u, 3), round(v, 3)],
        "desired_pixel_xy": _round_vector(desired_xy),
        "gripper_center_projection": gripper_projection,
        "target_depth_m": round(depth_m, 6),
        "target_region": target_region,
        "residual_px_before": round(residual_px, 3),
        "raw_correction_world_xyz": _round_vector(raw_delta_world),
        "correction_world_xyz": _round_vector(delta_world),
        "correction_clamped": raw_correction_norm > max_correction,
        "mask_geometry": mask_geometry,
        "operating_region": operating_region,
        "aligned_hover_pose": aligned_hover if executable_reference else None,
        "adjusted_contact_pose": adjusted_contact if executable_reference else None,
        "adjusted_precontact_pose": (
            adjusted_precontact if executable_reference else None
        ),
    }


def _project_configured_gripper_center(
    *,
    profile: Mapping[str, Any],
    profile_sha256: str,
    current_eef_pose: Mapping[str, Any],
    camera_extrinsics: Mapping[str, Any],
    intrinsics: Mapping[str, Any],
) -> tuple[list[float], JsonDict, list[list[float]]]:
    """Project the calibrated gripper centre into the current wrist image."""

    wrist_alignment = _mapping(profile.get("wrist_alignment"), "wrist_alignment")
    center_eef = _vector(
        wrist_alignment.get("eef_to_gripper_center_xyz"),
        3,
        "wrist_alignment.eef_to_gripper_center_xyz",
    )
    p_world_eef = _vector(current_eef_pose.get("xyz"), 3, "current_eef_pose.xyz")
    r_world_eef, orientation_source = _eef_rotation(current_eef_pose)
    p_world_center = _add(p_world_eef, _matvec3(r_world_eef, center_eef))

    r_world_cv, p_world_camera, extrinsics_source = (
        _resolve_current_opencv_camera_to_world(
            camera_extrinsics,
            p_world_eef=p_world_eef,
            r_world_eef=r_world_eef,
        )
    )
    camera_delta = [
        p_world_center[index] - p_world_camera[index] for index in range(3)
    ]
    p_camera_center = [
        sum(r_world_cv[row][column] * camera_delta[row] for row in range(3))
        for column in range(3)
    ]
    if p_camera_center[2] <= 1e-6:
        raise GraspGeometryError(
            "calibrated gripper center projects behind the wrist camera"
        )

    fx = _positive_float(intrinsics.get("fx"), "intrinsics.fx")
    fy = _positive_float(intrinsics.get("fy"), "intrinsics.fy")
    cx = _finite_float(intrinsics.get("cx"), "intrinsics.cx")
    cy = _finite_float(intrinsics.get("cy"), "intrinsics.cy")
    desired_xy = [
        fx * p_camera_center[0] / p_camera_center[2] + cx,
        fy * p_camera_center[1] / p_camera_center[2] + cy,
    ]
    width = _positive_float(intrinsics.get("width"), "intrinsics.width")
    height = _positive_float(intrinsics.get("height"), "intrinsics.height")
    if not (0 <= desired_xy[0] < width and 0 <= desired_xy[1] < height):
        raise GraspGeometryError(
            "calibrated gripper center projects outside the wrist image"
        )

    calibration_id = str(profile.get("calibration_id") or "")
    return desired_xy, {
        "schema_version": "openeta.gripper_center_projection.v1",
        "calibration_id": calibration_id,
        "profile_sha256": profile_sha256,
        "reference_frame": str(profile.get("eef_frame") or ""),
        "reference_point": str(wrist_alignment.get("reference_point") or ""),
        "eef_to_gripper_center_xyz": _round_vector(center_eef),
        "gripper_center_world_xyz": _round_vector(p_world_center),
        "gripper_center_camera_xyz": _round_vector(p_camera_center),
        "pixel_xy": _round_vector(desired_xy),
        "eef_orientation_source": orientation_source,
        "camera_extrinsics_source": extrinsics_source,
        "input_camera_frame": str(
            camera_extrinsics.get("camera_frame")
            or ("opencv" if extrinsics_source != "live_camera_to_world" else "opengl")
        ),
        "projection_camera_frame": "opencv",
    }, r_world_cv


def _resolve_current_opencv_camera_to_world(
    extrinsics: Mapping[str, Any],
    *,
    p_world_eef: Sequence[float],
    r_world_eef: Sequence[Sequence[float]],
) -> tuple[list[list[float]], list[float], str]:
    """Resolve live or calibrated wrist extrinsics into world camera pose."""

    if any(key in extrinsics for key in ("mat", "camera_to_world", "pose_mat", "matrix")):
        frame_transform = str(extrinsics.get("frame_transform") or "camera_to_world")
        if frame_transform != "camera_to_world":
            raise GraspGeometryError(
                "live wrist extrinsics frame_transform must be camera_to_world"
            )
        rotation, position = _opencv_camera_to_world(extrinsics)
        return rotation, position, "live_camera_to_world"

    mount_type = str(extrinsics.get("type") or "").strip()
    if mount_type == "T_gripper_cam":
        if str(extrinsics.get("frame") or "").strip() not in {"gripper", "eef", "tcp"}:
            raise GraspGeometryError(
                "T_gripper_cam extrinsics require frame gripper, eef, or tcp"
            )
        matrix = extrinsics.get("T_gripper_cam")
        if not isinstance(matrix, list) or len(matrix) != 4:
            raise GraspGeometryError("T_gripper_cam must be a 4x4 matrix")
        rows = [_vector(row, 4, "camera_extrinsics.T_gripper_cam") for row in matrix]
        r_eef_cv = _rotation(
            [row[:3] for row in rows[:3]],
            "camera_extrinsics.T_gripper_cam",
        )
        p_eef_camera = [rows[0][3], rows[1][3], rows[2][3]]
        r_world_cv = _matmul3(r_world_eef, r_eef_cv)
        p_world_camera = _add(
            p_world_eef,
            _matvec3(r_world_eef, p_eef_camera),
        )
        return r_world_cv, p_world_camera, "T_world_eef_x_T_gripper_cam"

    if mount_type == "T_base_cam":
        matrix = extrinsics.get("T_base_cam")
        if not isinstance(matrix, list) or len(matrix) != 4:
            raise GraspGeometryError("T_base_cam must be a 4x4 matrix")
        rows = [_vector(row, 4, "camera_extrinsics.T_base_cam") for row in matrix]
        rotation = _rotation(
            [row[:3] for row in rows[:3]],
            "camera_extrinsics.T_base_cam",
        )
        return rotation, [rows[0][3], rows[1][3], rows[2][3]], "T_base_cam"

    raise GraspGeometryError(
        "camera_extrinsics must provide live camera_to_world, T_gripper_cam, or T_base_cam"
    )


def _eef_rotation(pose: Mapping[str, Any]) -> tuple[list[list[float]], str]:
    rotation_matrix = pose.get("rotation_matrix")
    if rotation_matrix is not None:
        return _rotation(rotation_matrix, "current_eef_pose.rotation_matrix"), "rotation_matrix"

    quat = pose.get("quat_xyzw")
    if quat is not None:
        x, y, z, w = _vector(quat, 4, "current_eef_pose.quat_xyzw")
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-9:
            raise GraspGeometryError("current_eef_pose.quat_xyzw has zero length")
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        return [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], "quat_xyzw"

    rotvec = pose.get("rotvec")
    if rotvec is not None:
        rx, ry, rz = _vector(rotvec, 3, "current_eef_pose.rotvec")
        angle = math.sqrt(rx * rx + ry * ry + rz * rz)
        if angle < 1e-12:
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "rotvec"
        ax, ay, az = rx / angle, ry / angle, rz / angle
        c, s, one_minus_c = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
        return [
            [
                c + ax * ax * one_minus_c,
                ax * ay * one_minus_c - az * s,
                ax * az * one_minus_c + ay * s,
            ],
            [
                ay * ax * one_minus_c + az * s,
                c + ay * ay * one_minus_c,
                ay * az * one_minus_c - ax * s,
            ],
            [
                az * ax * one_minus_c - ay * s,
                az * ay * one_minus_c + ax * s,
                c + az * az * one_minus_c,
            ],
        ], "rotvec"

    raise GraspGeometryError(
        "current_eef_pose requires rotation_matrix, quat_xyzw, or rotvec"
    )


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
) -> tuple[float, float, float, int, int, JsonDict]:
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
        mask_coordinates: list[tuple[int, int]] = []
        mask_values = list(mask.get_flattened_data())
        depth_values = list(depth.get_flattened_data())
        for index, mask_value in enumerate(mask_values):
            if int(mask_value) <= 0:
                continue
            mask_coordinates.append((index % width, index // width))
            raw_depth = float(depth_values[index])
            if raw_depth > 0 and math.isfinite(raw_depth):
                foreground.append((index % width, index // width, raw_depth / scale))
    if len(foreground) < 5:
        raise GraspGeometryError("target mask has too few valid depth pixels")
    min_x = min(sample[0] for sample in mask_coordinates)
    max_x = max(sample[0] for sample in mask_coordinates)
    min_y = min(sample[1] for sample in mask_coordinates)
    max_y = max(sample[1] for sample in mask_coordinates)
    border_clearance = min(min_x, min_y, width - 1 - max_x, height - 1 - max_y)
    mask_geometry: JsonDict = {
        "bbox_xyxy": [min_x, min_y, max_x, max_y],
        "foreground_pixel_count": len(mask_coordinates),
        "valid_depth_pixel_count": len(foreground),
        "border_clearance_px": border_clearance,
        "touches_image_boundary": border_clearance <= 0,
    }
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
        return selected[0], selected[1], selected[2], width, height, mask_geometry
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
        mask_geometry,
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


def _transpose3(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(matrix[column][row]) for column in range(3)] for row in range(3)]


def _cross3(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    ]


def _orthogonal_unit(preferred: Sequence[float], normal: Sequence[float]) -> list[float]:
    projection = sum(float(preferred[index]) * float(normal[index]) for index in range(3))
    tangent = [
        float(preferred[index]) - projection * float(normal[index])
        for index in range(3)
    ]
    if math.sqrt(sum(value * value for value in tangent)) < 1e-6:
        fallback = [1.0, 0.0, 0.0] if abs(float(normal[0])) < 0.8 else [0.0, 1.0, 0.0]
        projection = sum(fallback[index] * float(normal[index]) for index in range(3))
        tangent = [
            fallback[index] - projection * float(normal[index]) for index in range(3)
        ]
    return _normalise(tangent, "camera roll reference")


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
