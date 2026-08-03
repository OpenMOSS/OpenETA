"""Stateful, observation-bound gateway for the clean Operator Codex.

This module is deliberately separate from the historical OpenETA agent
runtime.  It owns one simulator session and one ``EpisodeObservability``
writer, while reusing the existing simulator/SAM3/AnyGrasp MCP services and
the tested observation-bound perception facade.

The gateway's public methods return a small ``GatewayResult``.  The MCP
adapter turns that into concise text plus native MCP image content.  Full
numeric payloads remain in host-side episode artifacts and are never
required in the normal Operator response.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agent.runtime.image_artifacts import materialize_mcp_images
from agent.tools.embodied_perception import ObservationBoundPerception
from agent.tools.embodied_perception import resolve_observation_frame
from agent.tools.grasp_pose_refinement import (
    GraspPoseRefinementError,
    backproject_masked_world,
    camera_to_world_opencv,
    derive_mask_side_pose,
    rigid_transform,
)
from agent.tools.handlers import (
    AnyGraspDetectCallable,
    Sam3SegmentCallable,
    _camera_pose_to_world_handler,
    build_sse_anygrasp_mcp_grasper,
    build_sse_sam3_mcp_segmenter,
)
from agent.tools.registry import ToolExecutionContext, ToolResult, build_default_tool_registry
from agent.tools.sim_mcp import (
    SseSimulatorMcpTransport,
    mcp_dashboard_url,
    mcp_server_url_from_transport,
)
from logger.observability import EpisodeObservability
from sim.mcp_server.action_codecs import (
    LIBERO_PANDA_CLOSED_ENDPOINT_M,
    LIBERO_PANDA_NEAR_CLOSED_THRESHOLD_M,
    LIBERO_PANDA_OPEN_ENDPOINT_M,
)
from tools.grasp_inspector_client import GraspInspectorClient
from tools.operator_context_profiles import active_profile
from tools.pointcloud_pose_marking import (
    PointCloudView,
    annotate_active_grip_site_views,
    annotate_pose_preview_views,
    annotate_rejected_point_click,
    annotate_views,
    annotate_vector_views,
    camera_ray_from_image_click,
    mark_world_point,
    merge_world_vector_projections,
    multiview_consensus_pointcloud,
    pointcloud_coverage,
    pose_from_points,
    projected_camera_ray_segments,
    project_world_to_view,
    world_vector_from_view_delta,
    render_pointcloud_contact_sheet,
    render_move_feedback_crop,
    render_move_feedback_local_cloud,
    render_pose_candidate_frame_inspection,
    render_pose_candidate_frame_preview_views,
    render_pose_local_inspection_contact_sheet as render_pose_local_inspection_contact_sheet,
    render_world_pointcloud_views,
    annotate_pending_constraint_views,
    voxel_fuse_pointcloud,
    workspace_pointcloud,
    operator_scene_bounds,
    operator_scene_lookat,
    world_bounds_including_points,
    world_pointcloud_from_record,
)


PANDA_GRIP_SITE_REFERENCE_SPAN_M = 0.097
PANDA_MAX_APERTURE_M = LIBERO_PANDA_OPEN_ENDPOINT_M
# LIBERO's Panda finger joints retain a small mechanical/model clearance at
# the commanded closed endpoint.  This is the measured no-load endpoint, not
# the idealized zero-width jaw geometry.
PANDA_CLOSED_APERTURE_M = LIBERO_PANDA_CLOSED_ENDPOINT_M
PANDA_NEAR_CLOSED_THRESHOLD_M = LIBERO_PANDA_NEAR_CLOSED_THRESHOLD_M
SUPPORT_CLEARANCE_REQUIRED_MM = 20.0
PREGRASP_STANDOFF_M = 0.08
APPROACH_OPEN_MIN_APERTURE_MM = 65.0
CONTACT_MAX_TRANSLATION_STEP_M = 0.003
CONTACT_TRACKING_STALL_STEPS = 6
CONTACT_TRACKING_MIN_ALIGNED_PROGRESS_M = 0.001
CONTACT_TRACKING_CROSS_TRACK_TOLERANCE_M = 0.01
CONTACT_TRACKING_MIN_ERROR_IMPROVEMENT_M = 0.00005
PREGRASP_TRACKING_STALL_STEPS = 12
PREGRASP_TRACKING_MIN_ALIGNED_PROGRESS_M = 0.001
PREGRASP_TRACKING_CROSS_TRACK_TOLERANCE_M = 0.02
PREGRASP_TRACKING_MIN_ERROR_IMPROVEMENT_M = 0.00005
WORLD_ACTION_TIMEOUT_REPORT_CODES = {
    "capture_timeout",
    "remote_completion_unknown",
}
WORLD_ACTION_COMPONENTS = {
    "step",
    "move_to_pose",
    "nudge_end_effector",
    "move_to_selected_grasp",
    "move_to_selected_pregrasp",
    "approach_selected_grasp",
    "lift_grasp",
    "open_gripper",
    "close_gripper",
}
GRASP_INSPECTOR_SCENE_WAIT_S = 5.0
GRASP_INSPECTOR_SCENE_POLL_S = 0.1


def _exception_message(exc: BaseException) -> str:
    detail = str(exc).strip()
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


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


class GraspInspectorSceneNotReady(GraspPoseRefinementError):
    """Raised when a pose mutation cannot bind to the proposal viewer."""

    def __init__(self, state: Mapping[str, Any]) -> None:
        self.state = dict(state)
        super().__init__(
            str(state.get("message") or "The matching Viser proposal scene is not ready.")
        )


@dataclass(slots=True)
class GatewayResult:
    """Internal result split into Operator-visible and host-only channels."""

    success: bool
    text: dict[str, Any]
    images: list[Path] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class EmbodiedGateway:
    """Own one LIBERO-backed episode and expose semantic Operator actions."""

    def __init__(
        self,
        *,
        root: str | Path,
        env_id: str,
        task: str,
        seed: int = 0,
        image_width: int = 256,
        image_height: int = 256,
        simulator_url: str = "http://127.0.0.1:8765/sse",
        sam3_url: str = "http://127.0.0.1:8773/sse",
        anygrasp_url: str = "http://127.0.0.1:8774/sse",
        grasp_inspector_url: str = "http://127.0.0.1:8082",
        include_objects: bool = False,
        libero_dir: str | Path | None = None,
        transport: Any | None = None,
        sam3: Sam3SegmentCallable | None = None,
        sam3_points: Sam3SegmentCallable | None = None,
        anygrasp: AnyGraspDetectCallable | None = None,
        grasp_inspector: Any | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.env_id = str(env_id)
        self.task = str(task)
        self.seed = int(seed)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.include_objects = bool(include_objects)
        self.simulator_url = simulator_url
        self.sam3_url = sam3_url
        self.anygrasp_url = anygrasp_url
        self.grasp_inspector_url = grasp_inspector_url
        self.libero_dir = Path(
            libero_dir
            or os.environ.get("LIBERO_DIR")
            or Path(__file__).resolve().parents[1] / "third_party" / "LIBERO"
        ).expanduser().resolve()
        self.transport = transport or SseSimulatorMcpTransport(simulator_url)
        self._sam3_callable = sam3 or build_sse_sam3_mcp_segmenter(url=sam3_url)
        self._sam3_points_callable = (
            sam3_points
            or sam3
            or build_sse_sam3_mcp_segmenter(url=sam3_url, tool_name="segment_points")
        )
        raw_anygrasp = anygrasp or build_sse_anygrasp_mcp_grasper(url=anygrasp_url)
        self._anygrasp_callable = _build_anygrasp_depth_horizon_fallback(
            raw_anygrasp,
            artifact_root=self.root / "perception" / "anygrasp_depth_fallback",
        )
        self._grasp_inspector = grasp_inspector or GraspInspectorClient(
            grasp_inspector_url
        )

        self.session_id = ""
        self.handle = ""
        self._action_dim: int | None = None
        self.dashboard_url = ""
        self.writer: EpisodeObservability | None = None
        self.perception: ObservationBoundPerception | None = None
        self.current_record: dict[str, Any] | None = None
        self.current_sim_payload: dict[str, Any] = {}
        # Human-facing live viewer state.  This is deliberately separate from
        # the Operator tool response: the simulator pane can switch from the
        # latest RGB frame to a SAM3/AnyGrasp overlay as soon as a perception
        # call finishes, then switch back on the next observation.
        self.current_visualization: dict[str, Any] | None = None
        self.latest_grasp: ToolResult | None = None
        self.latest_grasp_proposal_id = ""
        self.latest_grasp_result_id = ""
        self.latest_segmentation: ToolResult | None = None
        self.selected_detection: dict[str, Any] | None = None
        self.selected_detection_observation_id = ""
        self.selected_grasp: dict[str, Any] | None = None
        self.selected_grasp_observation_id = ""
        self.last_task_success: bool | None = None
        # A native terminal-success signal is a one-way fact for the current
        # episode.  The simulator may report ``terminated=true`` during a
        # motion call before the Operator has finalized the episode; allowing
        # another world action after that point can destroy an already
        # satisfied task predicate.  Keep this separate from the public
        # ``last_task_success`` value so a later stale/false checker response
        # cannot erase the terminal evidence.
        self._task_success_latch: dict[str, Any] | None = None
        self.close_error: str | None = None
        self.failure_case: dict[str, Any] | None = None
        self.issues: list[dict[str, Any]] = []
        self._requested_outcome: str | None = None
        self._manual_mark_image: Path | None = None
        self._manual_marks: list[dict[str, Any]] = []
        self._pointcloud_views: dict[str, PointCloudView] = {}
        self._operator_pointcloud_points_world: np.ndarray | None = None
        self._operator_pointcloud_colors_rgb: np.ndarray | None = None
        self._operator_pointcloud_mode = (
            os.environ.get("OPENETA_OPERATOR_POINTCLOUD_MODE", "single-agentview")
            .strip()
            .lower()
        )
        if self._operator_pointcloud_mode not in {
            "single-agentview",
            "live-multiview-consensus",
        }:
            raise ValueError(
                "OPENETA_OPERATOR_POINTCLOUD_MODE must be "
                "'single-agentview' or 'live-multiview-consensus'"
            )
        operator_invariants = active_profile().manifest.get("invariants", {})
        self._operator_compact_result = (
            operator_invariants.get("operator_result_schema")
            == "compact_v1"
        )
        self._operator_gripper_reference_sent = False
        self._operator_observe_inspection_enabled = (
            operator_invariants.get("observe_inspection_policy") != "disabled"
        )
        self._operator_source_view_only_images = (
            operator_invariants.get("operator_image_return_policy")
            == "individual_source_views_only"
        )
        self._operator_mark_point_failure_feedback = operator_invariants.get(
            "mark_point_failure_feedback"
        )
        self._solved_point_agentview_pad_footprint = (
            active_profile()
            .manifest.get("invariants", {})
            .get("solved_point_agentview_projection")
            == "audit_with_current_orientation_pad_footprint"
        )
        self._solved_point_agentview_audit_return = (
            active_profile()
            .manifest.get("invariants", {})
            .get("solved_point_operator_evidence")
            == "source_view_zooms_plus_agentview_audit"
        )
        self._solved_point_current_only_feedback = (
            active_profile()
            .manifest.get("invariants", {})
            .get("solved_point_operator_images")
            == "current_mark_source_views_v1"
        )
        self._agentview_visible_surface_confirmation = (
            active_profile()
            .manifest.get("invariants", {})
            .get("agentview_ray_depth_policy")
            in {
                "confirm_aligned_rgbd_visible_surface",
                "solve_aligned_rgbd_visible_surface",
            }
        )
        self._agentview_first_visible_surface_shortcut = (
            active_profile()
            .manifest.get("invariants", {})
            .get("agentview_first_surface_shortcut")
            is True
        )
        self._all_returned_images_markable = (
            operator_invariants.get("mark_point_image_contract")
            == "all_returned_views_v1"
        )
        self._pose_candidate_identifier_namespace = (
            operator_invariants.get("pose_preview_view_namespace")
            == "candidate_v1"
        )
        self._pointcloud_current_reachable_envelope = (
            operator_invariants.get("pointcloud_world_bounds")
            == "scene_plus_current_grip_site_100mm_v1"
        )
        self._rejected_mark_current_click_feedback = (
            operator_invariants.get("mark_point_rejected_click_feedback")
            == "current_rejected_micro_marker_v1"
        )
        self._rgbd_surface_mark_views = {
            str(name)
            for name in operator_invariants.get(
                "mark_point_rgbd_surface_views",
                ["agentview"],
            )
        }
        self._mark_point_numeric_footprint = (
            active_profile()
            .manifest.get("invariants", {})
            .get("mark_point_footprint_policy")
            == "current_orientation_numeric_diagnostic_only"
        )
        self._compact_pose_preview_overlay = (
            active_profile()
            .manifest.get("invariants", {})
            .get("pose_preview_overlay")
            == "compact_geometry_only"
        )
        self._metric_pointcloud_edge_ticks = (
            active_profile()
            .manifest.get("invariants", {})
            .get("pointcloud_metric_edge_ticks")
            is True
        )
        self._metric_pointcloud_tick_band_style = operator_invariants.get(
            "pointcloud_metric_tick_band_style"
        )
        if self._metric_pointcloud_tick_band_style not in {
            None,
            "legacy_v1",
            "readable_v1",
            "readable_v2",
            "readable_v3",
        }:
            raise RuntimeError(
                "unsupported pointcloud_metric_tick_band_style: "
                f"{self._metric_pointcloud_tick_band_style!r}"
            )
        self._precise_current_grip_site_feedback = (
            operator_invariants.get("current_grip_site_image_feedback")
            == "micro_marker_v1"
        )
        self._precise_solved_mark_feedback = (
            operator_invariants.get("solved_mark_image_feedback")
            == "micro_marker_v1"
        )
        agentview_pose_preview_mode = operator_invariants.get(
            "pose_preview_agentview_overlay"
        )
        self._compact_agentview_pose_preview = (
            agentview_pose_preview_mode == "micro_marker_v1"
        )
        self._candidate_frame_pose_inspection = (
            active_profile()
            .manifest.get("invariants", {})
            .get("pose_preview_candidate_frame_inspection")
            == "observed_geometry_cross_sections"
        )
        self._candidate_frame_pose_preview_mode = (
            active_profile()
            .manifest.get("invariants", {})
            .get("pose_preview_pointcloud_frame")
        )
        self._candidate_frame_pose_preview_visual_mode = (
            active_profile()
            .manifest.get("invariants", {})
            .get(
                "pose_preview_visual_mode",
                "current_candidate_frame_v1",
            )
        )
        self._solved_agentview_anchor_ray_visual_mode = (
            active_profile()
            .manifest.get("invariants", {})
            .get("solved_agentview_anchor_ray_visual_mode")
        )
        self._candidate_frame_pose_preview_views = (
            self._candidate_frame_pose_preview_mode
            in {
                "candidate_gripper_xz_xy_v1",
                "candidate_gripper_jaw_app_lat_v2",
                "candidate_gripper_jaw_app_lat_v3",
            }
        )
        self._candidate_pad_swept_footprint = (
            active_profile()
            .manifest.get("invariants", {})
            .get("pose_preview_closing_sweep_geometry")
            == "pad_collision_box_swept_volume_v1"
        )
        self._candidate_pad_capture_corridor = (
            active_profile()
            .manifest.get("invariants", {})
            .get("pose_preview_closing_corridor_geometry")
            == "open_inner_pad_faces_capture_corridor_v1"
        )
        close_execution_policy = (
            active_profile()
            .manifest.get("invariants", {})
            .get("move_to_close_execution_policy")
        )
        self._close_requires_matching_preview = (
            close_execution_policy
            == "matching_candidate_preview_then_repeat_v1"
        )
        self._close_requires_explicit_preview_commit = (
            close_execution_policy == "frozen_preview_id_commit_v1"
        )
        self._grip_site_position_delta_enabled = (
            operator_invariants.get("move_to_position_delta_frames")
            == "world_or_current_grip_site_jaw_lat_app_v1"
        )
        self._pending_close_confirmation: dict[str, Any] | None = None
        self._agentview_pose_preview_overlay = (
            agentview_pose_preview_mode
            in {
                "hypothetical_candidate_pose_projection",
                "micro_marker_v1",
            }
        )
        self._not_reached_default_view_policy = operator_invariants.get(
            "move_to_not_reached_default_views"
        )
        not_reached_contact_feedback = operator_invariants.get(
            "move_to_not_reached_contact_feedback"
        )
        self._not_reached_current_contact_marker = (
            not_reached_contact_feedback
            in {
                "current_mujoco_contact_micro_marker_v1",
                "current_robot_mujoco_contact_micro_marker_v1",
            }
        )
        self._not_reached_robot_contact_only = (
            not_reached_contact_feedback
            == "current_robot_mujoco_contact_micro_marker_v1"
        )
        self._operator_pointcloud_source = "observation_rgbd:agentview"
        self._operator_pointcloud_metrics: dict[str, Any] = {}
        self._operator_pointcloud_world_bounds: np.ndarray | None = None
        self._pointcloud_contact_sheet: Path | None = None
        self._pointcloud_target_grip_site_xyz: list[float] | None = None
        self._agentview_axis_overlay: Path | None = None
        self._wrist_grip_site_overlay: Path | None = None
        self._agentview_ray_overlay: Path | None = None
        self._point_marks_3d: dict[str, dict[str, Any]] = {}
        self._pending_point_constraints: dict[str, dict[str, Any]] = {}
        self._point_vectors_3d: dict[str, dict[str, dict[str, Any]]] = {}
        self._marked_vectors_3d: dict[str, dict[str, Any]] = {}
        self._vector_view_paths: dict[str, str] = {}
        self._point_view_paths: dict[str, str] = {}
        self._inspection_views: dict[str, dict[str, Any]] = {}
        self._derived_mark_views: dict[str, dict[str, Any]] = {}
        self._active_grip_site_target: dict[str, Any] | None = None
        self._active_grip_site_view_paths: dict[str, str] = {}
        # Solved marks remain immutable in the ledger, but are hidden from
        # fresh Operator images unless explicitly requested.  This prevents a
        # long recovery trace from turning the metric views into a history
        # collage while preserving deliberate historical inspection.
        # ``None`` is the legacy/internal rendering mode used by direct
        # Host-side calls. A real Operator observe sets this to an explicit
        # allowlist (usually empty), which suppresses historical overlays.
        self._visible_history_point_ids: set[str] | None = None
        self._current_authored_point_id: str | None = None
        self._marked_pose: dict[str, Any] | None = None
        self._active_object_reference: dict[str, Any] | None = None
        self._episode_status = "running"
        self._sim_step = 0
        self._capture_index = 0
        self._manual_move_seq = 0
        self._grasp_attempt_seq = 0
        self._active_grasp_attempt_id: str | None = None
        self._active_grasp_target: dict[str, Any] | None = None
        self._pending_grasp_approach: dict[str, Any] | None = None
        self._last_lift_reference: dict[str, Any] | None = None
        self._last_grasp_verification: dict[str, Any] | None = None
        self._world_action_lock = threading.Lock()
        self._world_action_seq = 0
        self._active_world_action: dict[str, Any] | None = None
        self._last_world_action_outcome: dict[str, Any] | None = None
        self._episode_generation = 0
        self._terminal_generation_invalidated = False
        self._terminal_active_action: dict[str, Any] | None = None
        self._closed = False

    @property
    def episode_status(self) -> str:
        return self._episode_status

    @property
    def episode_success(self) -> bool | None:
        if self.failure_case is not None:
            return False
        if self._episode_status == "completed":
            return True
        if self._episode_status in {"aborted", "stopped"}:
            return None
        return self.last_task_success

    # ------------------------------------------------------------------
    # Episode lifecycle and observation projection
    # ------------------------------------------------------------------

    def ensure_started(self) -> None:
        if self._closed:
            raise RuntimeError("Embodied episode is already closed")
        if self.writer is not None:
            return
        if (self.root / "events.jsonl").exists():
            raise RuntimeError(f"Refusing to append to existing episode root: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        operator_root = os.environ.get("OPENETA_OPERATOR_ROOT", "").strip()
        if operator_root:
            # Codex resolves MCP image paths relative to its trusted workspace.
            # Mirror episode media there without duplicating generated files.
            workspace = Path(operator_root).expanduser()
            workspace.mkdir(parents=True, exist_ok=True)
            for name in ("pointcloud_views", "media", "mcp-input"):
                link = workspace / name
                target = self.root / name
                if link.exists() or link.is_symlink():
                    continue
                try:
                    link.symlink_to(target, target_is_directory=True)
                except OSError:
                    pass
        create = self.transport.call_tool(
            "create_env",
            {
                "env_id": self.env_id,
                "render_mode": "rgb_array",
                "seed": self.seed,
                "task": self.task,
                "image_width": self.image_width,
                "image_height": self.image_height,
                "include_objects": self.include_objects,
            },
            timeout_s=180.0,
        )
        if not _payload_ok(create):
            raise RuntimeError(f"Simulator create_env failed: {_payload_error(create)}")
        self.handle = str(create.get("handle") or "")
        self._action_dim = _int_or(create.get("action_dim"), 7) or 7
        self.session_id = str(create.get("session_id") or "")
        if not self.handle:
            raise RuntimeError("Simulator create_env returned no handle")
        self.dashboard_url = mcp_dashboard_url(
            mcp_server_url_from_transport(self.transport),
            self.session_id,
        )

        self.writer = EpisodeObservability(
            self.root,
            episode_id=self.root.name,
            session_id=self.session_id,
            env_id=self.env_id,
            seed=self.seed,
            task=self.task,
            metadata={
                "surface": "clean_operator_gateway",
                "owner": "operator_episode_process",
                "simulator_url": self.simulator_url,
            },
        )
        reset = self.transport.call_tool(
            "reset_env",
            self._session_args({"handle": self.handle, "seed": self.seed}),
            timeout_s=180.0,
        )
        if not _payload_ok(reset):
            raise RuntimeError(f"Simulator reset_env failed: {_payload_error(reset)}")
        settled = reset
        settle_steps = max(0, int(os.environ.get("OPENETA_LIBERO_SETTLE_STEPS", "10")))
        if settle_steps and "libero" in self.env_id.lower():
            try:
                dim = max(1, int(self._action_dim or 7))
                action = [0.0] * dim
                action[-1] = -1.0  # keep the reset gripper open while physics settles
                candidate = self.transport.call_tool(
                    "step_env",
                    self._session_args({
                        "handle": self.handle,
                        "action": action,
                        "num_steps": settle_steps,
                        "render": True,
                        "include_cameras": True,
                    }),
                    timeout_s=180.0,
                )
                if _payload_ok(candidate):
                    settled = candidate
                    self._sim_step += max(0, _int_or(candidate.get("steps_executed"), settle_steps))
            except Exception:
                # Settling is a visual-quality improvement; a simulator that
                # does not expose step_env must still be usable.
                settled = reset
        self._capture_observation(
            settled,
            source="reset_settled" if settled is not reset else "reset",
        )
        self.perception = ObservationBoundPerception(
            artifact_root=self.root,
            sam3=self._sam3_callable,
            sam3_points=self._sam3_points_callable,
            anygrasp=self._anygrasp_callable,
            output_root=self.root / "perception",
        )

    def observe(
        self,
        views: Sequence[str] | None = None,
        inspect: Mapping[str, Any] | None = None,
        history_point_ids: Sequence[str] | None = None,
    ) -> GatewayResult:
        if inspect is not None and not self._operator_observe_inspection_enabled:
            return GatewayResult(False, {
                "kind": "observation",
                "success": False,
                "reason": "inspect_disabled",
                "retryable": True,
                "message": (
                    "Local inspect/crop views are disabled. Request the needed "
                    "individual source views with views=[...]."
                ),
            })
        requested_views, view_error = self._normalize_operator_views(
            views,
            default=self._default_operator_observe_views(),
        )
        if view_error is not None:
            return view_error
        self.ensure_started()
        requested_history = {
            str(point_id).strip()
            for point_id in (history_point_ids or [])
            if str(point_id).strip()
        }
        if self.current_record is not None:
            payload = self.transport.call_tool(
                "render_env",
                self._session_args({"handle": self.handle}),
                timeout_s=180.0,
            )
            if not _payload_ok(payload):
                return GatewayResult(False, {"kind": "observation", "error": _payload_error(payload)})
            self._capture_observation(
                payload,
                source="observe",
                preserve_pending_point_constraints=True,
            )
        # A history request applies only to this returned observation.  A
        # subsequent observe without it goes back to the uncluttered default.
        self._visible_history_point_ids = requested_history
        try:
            result = self._observation_result(views=requested_views)
            if result.success and inspect is not None:
                inspection = self._render_observation_inspection(
                    inspect,
                    source_views=requested_views,
                )
                if not inspection.success:
                    return inspection
                pointcloud_base_names = {
                    "pointcloud_top",
                    "pointcloud_front",
                    "pointcloud_side",
                }
                keep_base_images = [
                    path
                    for name, path in zip(
                        result.text.get("returned_views", []),
                        result.images,
                    )
                    if name not in pointcloud_base_names
                ]
                result.images = [*keep_base_images, *inspection.images]
                result.text["inspection"] = inspection.text.get(
                    "inspection", {}
                )
                result.text["returned_views"] = [
                    *[
                        name
                        for name in result.text.get("returned_views", [])
                        if name not in pointcloud_base_names
                    ],
                    *[
                        item["view"]
                        for item in inspection.text.get(
                            "inspection", {}
                        ).get("views", [])
                    ],
                ]
                result.details["inspection"] = inspection.details
            return result
        finally:
            # The selected image paths remain immutable replay artifacts, but
            # their history allowlist must not leak into a later mark or world
            # action.  An empty set is the normal uncluttered Operator policy.
            self._visible_history_point_ids = set()

    def _render_observation_inspection(
        self,
        inspect: Mapping[str, Any],
        *,
        source_views: Sequence[str] | None = None,
    ) -> GatewayResult:
        """Render clean, observation-bound crops from calibrated point-cloud views."""

        if not isinstance(inspect, Mapping):
            return GatewayResult(False, {
                "kind": "observation",
                "success": False,
                "reason": "invalid_inspect",
                "message": "inspect must be an object containing boxes.",
            })
        boxes = inspect.get("boxes")
        if not isinstance(boxes, Sequence) or isinstance(
            boxes, (str, bytes)
        ) or not 1 <= len(boxes) <= 3:
            return GatewayResult(False, {
                "kind": "observation",
                "success": False,
                "reason": "invalid_inspect",
                "message": "inspect.boxes must contain one to three point-cloud crop boxes.",
            })
        self._ensure_operator_pointcloud_views()
        output_root = self.root / "pointcloud_views" / str(
            self.current_record.get("observation_id")
        ) / "inspections"
        output_root.mkdir(parents=True, exist_ok=True)
        images: list[Path] = []
        image_meta: list[dict[str, Any]] = []
        seen_views: set[str] = set()
        for index, box_spec in enumerate(boxes):
            if not isinstance(box_spec, Mapping):
                return GatewayResult(False, {
                    "kind": "observation",
                    "success": False,
                    "reason": "invalid_inspect_box",
                    "message": f"inspect.boxes[{index}] must be an object.",
                })
            name = str(box_spec.get("view") or "")
            if name not in {
                "pointcloud_top",
                "pointcloud_front",
                "pointcloud_side",
            }:
                return GatewayResult(False, {
                    "kind": "observation",
                    "success": False,
                    "reason": "invalid_inspect_box",
                    "message": (
                        f"inspect.boxes[{index}].view must be pointcloud_top, "
                        "pointcloud_front, or pointcloud_side."
                    ),
                })
            if name in seen_views:
                return GatewayResult(False, {
                    "kind": "observation",
                    "success": False,
                    "reason": "duplicate_inspect_view",
                    "message": f"Only one crop box may be supplied for {name}.",
                })
            seen_views.add(name)
            view = self._pointcloud_views.get(name)
            if view is None:
                continue
            raw_box = box_spec.get("box_xyxy")
            try:
                if (
                    not isinstance(raw_box, Sequence)
                    or isinstance(raw_box, (str, bytes))
                    or len(raw_box) != 4
                ):
                    raise ValueError
                x0, y0, x1, y1 = (int(value) for value in raw_box)
                left = max(0, min(view.width, min(x0, x1)))
                right = max(0, min(view.width, max(x0, x1)))
                top = max(0, min(view.height, min(y0, y1)))
                bottom = max(0, min(view.height, max(y0, y1)))
                if right - left < 16 or bottom - top < 16:
                    raise ValueError
            except (TypeError, ValueError):
                width = int(view.width)
                height = int(view.height)
                return GatewayResult(False, {
                    "kind": "observation",
                    "success": False,
                    "reason": "invalid_inspect_box",
                    "retryable": True,
                    "message": (
                        f"inspect.boxes[{index}].box_xyxy must be a clipped "
                        f"source-pixel box for {name} ({width}x{height}); "
                        f"received {raw_box!r}. Valid coordinates satisfy "
                        f"0 <= x0 < x1 <= {width} and 0 <= y0 < y1 <= {height}. "
                        "Use coordinates from this individual source view, not "
                        "a composite/contact-sheet or pending constraint image."
                    ),
                })
            source = (
                view.clean_image_path
                if view.clean_image_path
                and view.clean_image_path.is_file()
                else view.image_path
            )
            if not source.is_file():
                continue
            try:
                image = Image.open(source).convert("RGB")
                crop = image.crop((left, top, right, bottom))
                scale = max(
                    1,
                    min(8, int(512 / max(1, max(crop.size)))),
                )
                if scale > 1:
                    crop = crop.resize(
                        (crop.width * scale, crop.height * scale),
                        Image.Resampling.NEAREST,
                    )
                crop_id = f"crop{index}"
                inspection_view_id = f"inspection:{crop_id}:{name}"
                path = output_root / f"{name}.{crop_id}.inspection.png"
                crop.save(path)
                images.append(path)
                metadata = {
                    "view": inspection_view_id,
                    "source_view": name,
                    "path": str(path.relative_to(self.root)),
                    "source_observation_id": str(
                        self.current_record.get("observation_id")
                    ),
                    "crop_origin_xy": [left, top],
                    "crop_size_xy": [right - left, bottom - top],
                    "display_scale": scale,
                    "display_size_xy": [crop.width, crop.height],
                    "box_xyxy": [left, top, right, bottom],
                    "clean_source": (
                        view.clean_image_path is not None
                        and source == view.clean_image_path
                    ),
                }
                self._inspection_views[inspection_view_id] = metadata
                image_meta.append(metadata)
            except (OSError, TypeError, ValueError):
                continue
        if not images:
            return GatewayResult(False, {
                "kind": "observation",
                "success": False,
                "reason": "inspection_unavailable",
                "message": "No calibrated point-cloud view was available for this inspection.",
            })
        # Keep the operator-facing contract compact.  The crop origin,
        # display scale, source pixel mapping, and artifact paths remain in
        # details/replay where they are useful for debugging but are not part
        # of the next-action decision.
        compact_images = [
            {
                "view": item["view"],
                "source_view": item["source_view"],
                # The browser needs the artifact path to render the returned
                # crop.  Pixel origin/scale and all other mapping metadata
                # remain details-only.
                "path": item["path"],
            }
            for item in image_meta
        ]
        return GatewayResult(
            True,
            {
                "kind": "observation",
                "success": True,
                "inspection": {
                    "mode": "pixel_boxes",
                    "views": compact_images,
                    "geometric_only": True,
                },
            },
            images=images,
            details={
                "inspection_images": image_meta,
                "inspection_public_result": {
                    "mode": "pixel_boxes",
                    "views": compact_images,
                    "geometric_only": True,
                },
            },
        )

    def inspect_object_reference(self, target: str) -> GatewayResult:
        """Return canonical simulator asset images without locating the scene instance.

        This is deliberately a visual vocabulary lookup, not an oracle: it
        resolves only explicit asset-directory names, returns no world/image
        coordinates, and never changes the active SAM3 prompt or selection.
        """

        self.ensure_started()
        assert self.current_record is not None and self.writer is not None
        if self.failure_case is not None:
            return self._failed_result("object_reference")
        call = self.writer.record_tool_start(
            tool="inspect_object_reference",
            input_frames=self.current_record.get("frame_ids", []),
            parameters={"target": str(target)},
        )
        normalized = re.sub(r"[^a-z0-9]+", "_", str(target).strip().lower()).strip("_")
        self._active_object_reference = None
        if not normalized:
            message = "Object reference target must contain a readable category name."
            self.writer.record_tool_result(
                tool="inspect_object_reference",
                success=False,
                tool_call_id=call.get("tool_call_id"),
                input_frames=self.current_record.get("frame_ids", []),
                result={"reason": "empty_target"},
            )
            return GatewayResult(
                False,
                {"kind": "object_reference", "success": False, "error": message},
            )

        assets_root = self.libero_dir / "libero" / "libero" / "assets"
        groups = (
            assets_root / "stable_hope_objects",
            assets_root / "stable_scanned_objects",
            assets_root / "turbosquid_objects",
        )
        matches: list[tuple[int, Path]] = []
        for group in groups:
            if not group.is_dir():
                continue
            for directory in group.iterdir():
                if not directory.is_dir():
                    continue
                name = directory.name.lower()
                if name == normalized:
                    score = 0
                elif name.endswith(f"_{normalized}") or normalized.endswith(f"_{name}"):
                    score = 1
                else:
                    continue
                matches.append((score, directory))

        matches.sort(key=lambda item: (item[0], item[1].name))
        best_score = matches[0][0] if matches else None
        directories = [path for score, path in matches if score == best_score][:3]
        source_images: list[Path] = []
        for directory in directories:
            images = sorted(
                (
                    path
                    for path in directory.iterdir()
                    if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
                ),
                key=lambda path: (
                    0 if path.name.lower() == "texture_map.png" else 1,
                    0 if "texture" in path.name.lower() else 1,
                    path.name,
                ),
            )
            if images:
                source_images.append(images[0])

        if not source_images:
            message = (
                f"No canonical LIBERO asset image matched {target!r}. "
                "Try the explicit simulator category name or continue from scene RGB."
            )
            self.writer.record_tool_result(
                tool="inspect_object_reference",
                success=False,
                tool_call_id=call.get("tool_call_id"),
                input_frames=self.current_record.get("frame_ids", []),
                result={
                    "reason": "reference_not_found",
                    "normalized_target": normalized,
                    "assets_root": str(assets_root),
                },
            )
            return GatewayResult(
                False,
                {
                    "kind": "object_reference",
                    "success": False,
                    "target": str(target),
                    "error": message,
                    "retryable": True,
                },
                details={"assets_root": str(assets_root)},
            )

        artifact_dir = self.root / "perception" / "object_references" / normalized
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[Path] = []
        render_errors: list[str] = []
        repo_root = Path(__file__).resolve().parents[1]
        renderer = repo_root / "scripts" / "embodied" / "render_libero_asset_reference.py"
        renderer_python = Path(
            os.environ.get("OPENETA_LIBERO_PYTHON")
            or repo_root / "sim" / "venvs" / "libero" / "bin" / "python"
        )
        for directory in directories:
            xml_candidates = sorted(directory.glob("*.xml"))
            preferred_xml = directory / f"{directory.name}.xml"
            xml_path = (
                preferred_xml
                if preferred_xml.is_file()
                else xml_candidates[0]
                if xml_candidates
                else None
            )
            if xml_path is None or not renderer_python.is_file() or not renderer.is_file():
                continue
            destination = artifact_dir / f"{directory.name}-canonical-render.png"
            if not destination.is_file():
                environment = os.environ.copy()
                environment.setdefault("MUJOCO_GL", "egl")
                completed = subprocess.run(
                    [
                        str(renderer_python),
                        str(renderer),
                        "--xml",
                        str(xml_path),
                        "--output",
                        str(destination),
                    ],
                    cwd=repo_root,
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=30.0,
                    check=False,
                )
                if completed.returncode != 0:
                    render_errors.append(
                        (completed.stderr or completed.stdout or "canonical render failed").strip()
                    )
            if destination.is_file():
                artifacts.append(destination)
        for source in source_images:
            destination = artifact_dir / f"{source.parent.name}-{source.name}"
            if not destination.exists():
                shutil.copy2(source, destination)
            artifacts.append(destination)

        self.writer.record_tool_result(
            tool="inspect_object_reference",
            success=True,
            tool_call_id=call.get("tool_call_id"),
            input_frames=self.current_record.get("frame_ids", []),
            artifact_refs=[str(path) for path in artifacts],
            result={
                "target": str(target),
                "normalized_target": normalized,
                "matched_asset_names": [path.parent.name for path in source_images],
                **({"canonical_render_errors": render_errors} if render_errors else {}),
            },
        )
        self._active_object_reference = {
            "target": str(target),
            "normalized_target": normalized,
            "image": str(artifacts[0]),
        }
        return GatewayResult(
            True,
            {
                "kind": "object_reference",
                "success": True,
                "target": str(target),
                "matched_asset_names": [path.parent.name for path in source_images],
                "message": (
                    "Canonical simulator asset reference only. The first image is "
                    "the 3D multi-view render when available; compare its dominant "
                    "colors, shape, and label fragments with current RGB, then mark "
                    "or segment the scene instance yourself. This tool provides no "
                    "scene location."
                ),
            },
            images=artifacts,
            details={
                "source_images": [str(path) for path in source_images],
                **({"canonical_render_errors": render_errors} if render_errors else {}),
            },
        )

    def _mark_failure(
        self,
        *,
        category: str,
        component: str,
        code: str,
        message: str,
        tool: str | None = None,
        action_id: str | None = None,
        input_frames: list[str] | None = None,
        artifact_refs: list[str] | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Mark the episode failed once and retain the first failure as truth."""

        if self.failure_case is not None:
            return self.failure_case
        assert self.writer is not None
        failure = self.writer.record_failure_case(
            category=category,
            component=component,
            code=code,
            message=message,
            tool=tool,
            action_id=action_id,
            input_frames=input_frames or [],
            artifact_refs=artifact_refs or [],
            details=details,
        )
        self.failure_case = dict(failure)
        self._episode_status = "failed"
        self._write_current_projection()
        return self.failure_case

    def _record_issue(
        self,
        *,
        category: str,
        component: str,
        code: str,
        message: str,
        tool: str | None = None,
        action_id: str | None = None,
        input_frames: list[str] | None = None,
        artifact_refs: list[str] | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retain recoverable evidence while keeping the episode runnable."""

        assert self.writer is not None
        issue = self.writer.record_issue(
            category=category,
            component=component,
            code=code,
            message=message,
            tool=tool,
            action_id=action_id,
            input_frames=input_frames or [],
            artifact_refs=artifact_refs or [],
            details=details,
        )
        retained = dict(issue)
        self.issues.append(retained)
        self._write_current_projection()
        return retained

    def _failed_result(self, kind: str, *, message: str | None = None) -> GatewayResult:
        """Return a concise result when a terminal failure already occurred."""

        failure = self.failure_case or {}
        frames = self._operator_frame_paths((self.current_record or {}).get("frames", []))
        return GatewayResult(
            False,
            {
                "kind": kind,
                "success": False,
                "episode_status": "failed",
                "failure_case_id": failure.get("failure_case_id"),
                "failure_code": failure.get("code"),
                "failure_component": failure.get("component"),
                "message": message
                or "Episode is already marked failed; stop and call close_episode.",
            },
            images=frames,
            details={"failure_case": failure},
        )

    @staticmethod
    def _backend_failure_details(details: Mapping[str, Any]) -> dict[str, Any]:
        """Keep failure events debuggable without duplicating raw model output."""

        keys = (
            "reason",
            "backend",
            "backend_tool",
            "model",
            "raw_output_ref",
            "tool_result_ref",
            "diagnostics",
            "metadata",
        )
        return {key: details[key] for key in keys if key in details}

    def _capture_observation(
        self,
        payload: Mapping[str, Any],
        *,
        source: str,
        preserve_pending_point_constraints: bool = False,
    ) -> None:
        assert self.writer is not None
        raw_observation = payload.get("observation", payload)
        if not isinstance(raw_observation, Mapping):
            raise RuntimeError("Simulator response has no observation payload")
        self._capture_index += 1
        materialized = materialize_mcp_images(
            dict(raw_observation),
            output_root=self.root / "mcp-input",
            bundle_id=f"sim-{self._capture_index:06d}",
        )
        observation = materialized.payload
        cameras = observation.get("cameras", [])
        if not isinstance(cameras, list):
            cameras = []
        writer_cameras: list[dict[str, Any]] = []
        for camera in cameras:
            if not isinstance(camera, Mapping):
                continue
            camera_id = str(camera.get("frame_id") or camera.get("camera_id") or "camera")
            intrinsics = camera.get("anygrasp_intrinsics") or camera.get("intrinsics") or {}
            metadata = {
                "intrinsics": dict(intrinsics) if isinstance(intrinsics, Mapping) else {},
                "extrinsics": dict(camera.get("extrinsics") or {})
                if isinstance(camera.get("extrinsics"), Mapping)
                else {},
                "width": camera.get("width"),
                "height": camera.get("height"),
                "depth_unit": "metres_float32",
            }
            writer_cameras.append(
                {
                    "camera_id": camera_id,
                    "rgb_path": camera.get("rgb_path"),
                    "depth_path": camera.get("depth_path"),
                    "metadata": metadata,
                }
            )
        if not writer_cameras:
            raise RuntimeError("Simulator observation contains no RGB-D camera frames")

        task = observation.get("task") or observation.get("task_description")
        if isinstance(task, str) and task.strip():
            self.task = task.strip()
        self.current_sim_payload = dict(observation)
        previous_observation_id = (
            self.current_record.get("observation_id") if self.current_record is not None else None
        )
        self.current_record = self.writer.record_observation(
            writer_cameras,
            source=source,
            sim_step=self._sim_step,
            metadata={
                "task_description": self.task,
                "simulator_observation": dict(observation),
                "robot": observation.get("robot", {}),
                "reset_info": (
                    dict(observation.get("metadata", {}).get("reset_info", {}))
                    if isinstance(observation.get("metadata"), Mapping)
                    and isinstance(observation.get("metadata", {}).get("reset_info"), Mapping)
                    else {}
                ),
            },
        )
        # A world-changing observation invalidates view-bound geometry
        # immediately, but an already solved mark is an explicit immutable
        # world coordinate. Preserve solved marks and their observation
        # provenance so move_to can still execute the authored location after
        # an intermediate move.
        #
        # A pure render_env observe/inspect is different: it does not step
        # physics, so a pending camera ray or orthographic plane remains a
        # valid world-frame constraint. Preserve it explicitly so the intended
        # workflow can be:
        #
        #   first click -> observe(inspect=...) -> complementary click
        #
        # Reset and post-action captures keep the default False and invalidate
        # pending constraints because the world may have changed.
        # Point-cloud construction is intentionally lazy.
        # In live multi-view mode the consensus pass processes roughly 1.8M
        # input points, so RGB-only move_to feedback must not pay that cost.
        # observe materializes geometry below only when the requested visual
        # context contains a pointcloud_* view.
        self._pointcloud_views = {}
        self._operator_pointcloud_points_world = None
        self._operator_pointcloud_colors_rgb = None
        self._pointcloud_contact_sheet = None
        self._pointcloud_target_grip_site_xyz = None
        self._operator_pointcloud_source = "not_materialized"
        self._operator_pointcloud_metrics = {
            "mode": self._operator_pointcloud_mode,
            "source": self._operator_pointcloud_source,
            "observation_id": str(
                self.current_record.get("observation_id") or ""
            ),
            "materialized": False,
        }
        self._operator_pointcloud_world_bounds = None
        self._agentview_axis_overlay = self._render_agentview_world_axes()
        self._wrist_grip_site_overlay = self._render_wrist_grip_site()
        self._agentview_ray_overlay = None
        if not preserve_pending_point_constraints:
            self._pending_point_constraints = {}
        if self._current_authored_point_id not in self._point_marks_3d:
            self._current_authored_point_id = next(
                reversed(self._point_marks_3d),
                None,
            )
        self._point_vectors_3d = {}
        self._marked_vectors_3d = {}
        self._vector_view_paths = {}
        self._point_view_paths = {}
        self._inspection_views = {}
        self._active_grip_site_view_paths = {}
        self._marked_pose = None
        # Perception and grasp choices are valid only for the frame on which
        # they were produced.  Invalidate them whenever a fresh observation
        # enters the gateway, including post-action renders.
        if previous_observation_id is not None:
            self.latest_segmentation = None
            self.selected_detection = None
            self.selected_detection_observation_id = ""
            self.latest_grasp = None
            self.latest_grasp_proposal_id = ""
            self.latest_grasp_result_id = ""
            self.selected_grasp = None
            self.selected_grasp_observation_id = ""
        self.current_visualization = None
        self._manual_mark_image = None
        self._manual_marks = []
        self._write_current_projection()

    def _ensure_operator_pointcloud_views(self) -> None:
        """Materialize and cache geometry for the current observation."""

        assert self.current_record is not None
        observation_id = str(self.current_record.get("observation_id") or "")
        if (
            self._pointcloud_views
            and self._operator_pointcloud_metrics.get("observation_id")
            == observation_id
        ):
            return
        try:
            self._pointcloud_views = self._render_operator_pointcloud_views(
                observation_id=observation_id
            )
            self._pointcloud_contact_sheet = render_pointcloud_contact_sheet(
                self._pointcloud_views,
                output_root=self.root / "pointcloud_views" / observation_id,
            )
            self._render_current_point_views()
        except (OSError, ValueError, KeyError, TypeError):
            if self._operator_pointcloud_mode == "live-multiview-consensus":
                raise
            self._pointcloud_views = {}
            self._operator_pointcloud_points_world = None
            self._operator_pointcloud_colors_rgb = None
            self._pointcloud_contact_sheet = None
        self._write_current_projection()

    def _render_current_point_views(
        self,
        *,
        force_current_agentview_audit: bool = False,
    ) -> dict[str, Path]:
        """Compose actual, active-target, and current authored-point overlays."""

        self._point_view_paths = {}
        self._active_grip_site_view_paths = {}
        if not self._pointcloud_views:
            return {}
        output_root = (
            self.root
            / "pointcloud_views"
            / str(self.current_record.get("observation_id"))
        )
        source_paths: dict[str, Path] = {}
        active = self._active_grip_site_target
        # The base point-cloud renderer already contains the *actual* current
        # gripper.  Candidate/previous targets are history and are composed
        # only when the Operator explicitly asks for their point_id.
        history_filter = self._visible_history_point_ids
        active_visible = (
            history_filter is None
            or active is not None
            and str(active.get("point_id")) in history_filter
        )
        if active and active_visible:
            active_views = annotate_active_grip_site_views(
                self._pointcloud_views,
                output_root=output_root,
                point_id=str(active["point_id"]),
                target_position_xyz_m=active["position_xyz_m"],
                target_pad_contact_centers_world_m=active.get(
                    "pad_contact_centers_world_m"
                ),
            )
            self._active_grip_site_view_paths = {
                name: str(path.relative_to(self.root))
                for name, path in active_views.items()
            }
            source_paths = active_views

        point_id = self._current_authored_point_id
        rendered: dict[str, Path] = {}
        if point_id and point_id in self._pending_point_constraints:
            rendered = annotate_pending_constraint_views(
                self._pointcloud_views,
                self._pending_point_constraints[point_id],
                output_root=output_root,
                point_id=point_id,
                marks={},
                source_paths=source_paths,
                show_visible_surface_marker=(
                    self._agentview_visible_surface_confirmation
                ),
            )
        elif point_id and point_id in self._point_marks_3d:
            # position_point_id is already represented by the orange active
            # grip-site center and pad footprint. Drawing the same point again
            # as a red authored mark creates overlapping labels without adding
            # information.
            if (
                history_filter is None
                or point_id in history_filter
            ) and (
                not active or str(active.get("point_id")) != point_id
            ):
                rendered = annotate_views(
                    self._pointcloud_views,
                    {point_id: self._point_marks_3d[point_id]},
                    output_root=output_root,
                    source_paths=source_paths,
                    solved_agentview_anchor_ray_visual_mode=(
                        self._solved_agentview_anchor_ray_visual_mode
                    ),
                )
            if (
                force_current_agentview_audit
                or history_filter is None
                or point_id in history_filter
            ):
                agentview_projection = self._render_agentview_world_point(
                    point_id=point_id,
                    xyz_m=self._point_marks_3d[point_id]["xyz_m"],
                    output_root=output_root,
                )
                if agentview_projection is not None:
                    rendered["agentview"] = agentview_projection
        if rendered:
            self._point_view_paths = {
                name: str(path.relative_to(self.root))
                for name, path in rendered.items()
            }
            # ``source_paths`` contains the active grip-site footprint while
            # ``rendered`` contains the current authored-point evidence.  They
            # are tracked separately above, but callers need the complete set
            # of visible views.  In particular, when the active target and the
            # authored point share an ID, the orthographic footprint remains
            # the sole geometry overlay and agentview contributes only the
            # semantic audit marker.
            return {**source_paths, **rendered}
        return source_paths

    def _render_agentview_world_point(
        self,
        *,
        point_id: str,
        xyz_m: Sequence[float],
        output_root: Path,
    ) -> Path | None:
        """Project one solved world point into its current calibrated agentview.

        This is a semantic-identity audit overlay only. It neither changes the
        authored point nor infers which object owns the projected pixel.
        """

        if self.current_record is None:
            return None
        frame = _find_frame(self.current_record, "agentview")
        if not isinstance(frame, Mapping):
            return None
        raw_path = frame.get("rgb_path")
        metadata = frame.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        intrinsics = metadata.get("intrinsics")
        extrinsics = metadata.get("extrinsics")
        if (
            not isinstance(raw_path, str)
            or not isinstance(intrinsics, Mapping)
            or not isinstance(extrinsics, Mapping)
        ):
            return None
        image_path = Path(raw_path)
        if not image_path.is_absolute():
            image_path = self.root / image_path
        if not image_path.is_file():
            return None
        try:
            point = np.asarray(xyz_m, dtype=np.float64).reshape(3)
            world_from_camera = camera_to_world_opencv(extrinsics)
            camera_point = (
                world_from_camera[:3, :3].T
                @ (point - world_from_camera[:3, 3])
            )
            if camera_point[2] <= 1e-6:
                return None
            u = int(
                round(
                    float(intrinsics["fx"]) * camera_point[0] / camera_point[2]
                    + float(intrinsics["cx"])
                )
            )
            v = int(
                round(
                    float(intrinsics["fy"]) * camera_point[1] / camera_point[2]
                    + float(intrinsics["cy"])
                )
            )
            image = Image.open(image_path).convert("RGB")
            if not (0 <= u < image.width and 0 <= v < image.height):
                return None
            current_marker = self._current_grip_site_pixel(frame)
            if (
                self._precise_current_grip_site_feedback
                and current_marker is not None
            ):
                self._draw_precise_grip_site_marker(image, current_marker)
            draw = ImageDraw.Draw(image)
            scale = max(1.0, min(image.width, image.height) / 512.0)
            arm = int(round(11 * scale))
            outer = max(4, int(round(3 * scale)))
            inner = max(2, int(round(2 * scale)))
            for color, width in (((0, 0, 0), outer), ((255, 59, 48), inner)):
                draw.line((u - arm, v, u + arm, v), fill=color, width=width)
                draw.line((u, v - arm, u, v + arm), fill=color, width=width)
                draw.ellipse(
                    (
                        u - int(round(4 * scale)),
                        v - int(round(4 * scale)),
                        u + int(round(4 * scale)),
                        v + int(round(4 * scale)),
                    ),
                    outline=color,
                    width=width,
                )
            label = str(point_id)
            font = _gateway_font(max(11, int(round(13 * scale))))
            draw.text(
                (u + int(round(8 * scale)), v - int(round(18 * scale))),
                label,
                fill=(255, 59, 48),
                font=font,
                stroke_width=max(1, int(round(scale))),
                stroke_fill=(0, 0, 0),
            )
            # Show the exact current-orientation pad footprint that would be
            # rigidly translated to this grip-site.  Identity and grasp
            # geometry then share one real RGB audit surface: the crosshair
            # answers "which visible feature?", while the orange pads answer
            # "would the fingers actually straddle it?".  This remains a
            # visualization only and does not validate or change the mark.
            rotation = self._current_grip_site_rotation_for_render()
            if self._solved_point_agentview_pad_footprint and rotation is not None:
                pad_contacts, pad_boxes = self._candidate_pad_geometry_for_pose(
                    position_xyz_m=point,
                    rotation_matrix=rotation,
                )

                def project(candidate_xyz: Sequence[float]) -> tuple[int, int] | None:
                    candidate = np.asarray(candidate_xyz, dtype=np.float64).reshape(3)
                    candidate_camera = (
                        world_from_camera[:3, :3].T
                        @ (candidate - world_from_camera[:3, 3])
                    )
                    if candidate_camera[2] <= 1e-6:
                        return None
                    candidate_u = int(
                        round(
                            float(intrinsics["fx"])
                            * candidate_camera[0]
                            / candidate_camera[2]
                            + float(intrinsics["cx"])
                        )
                    )
                    candidate_v = int(
                        round(
                            float(intrinsics["fy"])
                            * candidate_camera[1]
                            / candidate_camera[2]
                            + float(intrinsics["cy"])
                        )
                    )
                    return candidate_u, candidate_v

                orange = (255, 159, 10)
                projected_contacts = [
                    projected
                    for contact in pad_contacts or ()
                    if (projected := project(contact)) is not None
                ]
                if len(projected_contacts) == 2:
                    draw.line(
                        (*projected_contacts[0], *projected_contacts[1]),
                        fill=orange,
                        width=max(2, int(round(2 * scale))),
                    )
                    pad_radius = max(4, int(round(5 * scale)))
                    for index, (pad_u, pad_v) in enumerate(
                        projected_contacts, start=1
                    ):
                        draw.rectangle(
                            (
                                pad_u - pad_radius,
                                pad_v - pad_radius,
                                pad_u + pad_radius,
                                pad_v + pad_radius,
                            ),
                            outline=orange,
                            width=max(2, int(round(2 * scale))),
                        )
                        draw.text(
                            (
                                pad_u + pad_radius + 2,
                                pad_v - pad_radius - 2,
                            ),
                            f"P{index}",
                            fill=orange,
                            font=_gateway_font(max(9, int(round(10 * scale)))),
                            stroke_width=max(1, int(round(scale))),
                            stroke_fill=(0, 0, 0),
                        )
                for box in pad_boxes or ():
                    center = np.asarray(box["center_world_m"], dtype=np.float64)
                    box_rotation = np.asarray(
                        box["rotation_world"], dtype=np.float64
                    )
                    half_size = np.asarray(box["half_size_m"], dtype=np.float64)
                    corners: list[tuple[int, int]] = []
                    for sx in (-1.0, 1.0):
                        for sy in (-1.0, 1.0):
                            for sz in (-1.0, 1.0):
                                projected = project(
                                    center
                                    + box_rotation
                                    @ (
                                        half_size
                                        * np.asarray(
                                            (sx, sy, sz), dtype=np.float64
                                        )
                                    )
                                )
                                if projected is not None:
                                    corners.append(projected)
                    hull = _convex_hull_2d(corners)
                    if len(hull) >= 3:
                        draw.polygon(
                            hull,
                            outline=orange,
                            width=max(2, int(round(2 * scale))),
                        )
            safe_point_id = re.sub(
                r"[^A-Za-z0-9_.-]+", "_", point_id
            ).strip("_") or "point"
            output_root.mkdir(parents=True, exist_ok=True)
            output = output_root / f"agentview.{safe_point_id}.solved.png"
            image.save(output)
            return output
        except (KeyError, OSError, TypeError, ValueError):
            return None

    def _render_agentview_pose_preview(
        self,
        *,
        target_position_xyz_m: Sequence[float],
        target_rotation_matrix: Sequence[Sequence[float]],
        target_pad_contact_centers_world_m: Sequence[Sequence[float]] | None,
        target_pad_boxes: Sequence[Mapping[str, Any]] | None,
        output_root: Path,
    ) -> Path | None:
        """Project one hypothetical grip-site pose into the real agentview.

        The overlay binds candidate geometry to visible scene identity without
        pretending that the robot has moved. It is a calibrated projection of
        the exact pose resolved by move_to, not a collision, grasp, or
        reachability judgment.
        """

        if self.current_record is None:
            return None
        frame = _find_frame(self.current_record, "agentview")
        if not isinstance(frame, Mapping):
            return None
        raw_path = frame.get("rgb_path")
        metadata = frame.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        intrinsics = metadata.get("intrinsics")
        extrinsics = metadata.get("extrinsics")
        if (
            not isinstance(raw_path, str)
            or not isinstance(intrinsics, Mapping)
            or not isinstance(extrinsics, Mapping)
        ):
            return None
        image_path = Path(raw_path)
        if not image_path.is_absolute():
            image_path = self.root / image_path
        if not image_path.is_file():
            return None
        try:
            target = np.asarray(target_position_xyz_m, dtype=np.float64).reshape(3)
            rotation = np.asarray(
                target_rotation_matrix, dtype=np.float64
            ).reshape(3, 3)
            if not np.isfinite(target).all() or not np.isfinite(rotation).all():
                return None
            world_from_camera = camera_to_world_opencv(extrinsics)
            camera_from_world = world_from_camera[:3, :3].T

            def project(candidate_xyz: Sequence[float]) -> tuple[int, int] | None:
                candidate = np.asarray(candidate_xyz, dtype=np.float64).reshape(3)
                camera_point = camera_from_world @ (
                    candidate - world_from_camera[:3, 3]
                )
                if camera_point[2] <= 1e-6:
                    return None
                return (
                    int(
                        round(
                            float(intrinsics["fx"])
                            * camera_point[0]
                            / camera_point[2]
                            + float(intrinsics["cx"])
                        )
                    ),
                    int(
                        round(
                            float(intrinsics["fy"])
                            * camera_point[1]
                            / camera_point[2]
                            + float(intrinsics["cy"])
                        )
                    ),
                )

            center = project(target)
            if center is None:
                return None
            image = Image.open(image_path).convert("RGB")
            if not (0 <= center[0] < image.width and 0 <= center[1] < image.height):
                return None
            current_marker = self._current_grip_site_pixel(frame)
            if (
                self._precise_current_grip_site_feedback
                and current_marker is not None
            ):
                self._draw_precise_grip_site_marker(image, current_marker)
            draw = ImageDraw.Draw(image)
            scale = max(1.0, min(image.width, image.height) / 512.0)
            orange = (255, 159, 10)
            cyan = (80, 220, 255)
            magenta = (255, 90, 220)
            compact = self._compact_agentview_pose_preview
            font = _gateway_font(max(10, int(round(11 * scale))))
            label_font = _gateway_font(max(11, int(round(13 * scale))))

            if compact:
                outer_radius = max(3, int(round(3 * scale)))
                inner_radius = max(2, outer_radius - 1)
            else:
                radius = max(5, int(round(6 * scale)))
                draw.rectangle(
                    (
                        center[0] - radius,
                        center[1] - radius,
                        center[0] + radius,
                        center[1] + radius,
                    ),
                    outline=orange,
                    width=max(2, int(round(3 * scale))),
                )

            for direction, color, label in (
                (rotation[:, 0], magenta, "JAW"),
                (rotation[:, 2], cyan, "APP"),
            ):
                endpoint = project(
                    target + direction * (0.025 if compact else 0.055)
                )
                if endpoint is None:
                    continue
                draw.line(
                    (*center, *endpoint),
                    fill=color,
                    width=1 if compact else max(2, int(round(3 * scale))),
                )
                if not compact:
                    draw.text(
                        (endpoint[0] + 4, endpoint[1] + 2),
                        label,
                        fill=color,
                        font=font,
                        stroke_width=max(1, int(round(scale))),
                        stroke_fill=(0, 0, 0),
                    )

            projected_contacts = [
                projected
                for contact in target_pad_contact_centers_world_m or ()
                if (projected := project(contact)) is not None
            ]
            if len(projected_contacts) == 2:
                draw.line(
                    (*projected_contacts[0], *projected_contacts[1]),
                    fill=orange,
                    width=1 if compact else max(2, int(round(2 * scale))),
                )
                pad_radius = (
                    max(1, int(round(scale)))
                    if compact
                    else max(4, int(round(5 * scale)))
                )
                for index, (pad_u, pad_v) in enumerate(
                    projected_contacts, start=1
                ):
                    if compact:
                        draw.ellipse(
                            (
                                pad_u - pad_radius,
                                pad_v - pad_radius,
                                pad_u + pad_radius,
                                pad_v + pad_radius,
                            ),
                            fill=orange,
                        )
                    else:
                        draw.rectangle(
                            (
                                pad_u - pad_radius,
                                pad_v - pad_radius,
                                pad_u + pad_radius,
                                pad_v + pad_radius,
                            ),
                            outline=orange,
                            width=max(2, int(round(2 * scale))),
                        )
                        draw.text(
                            (pad_u + pad_radius + 2, pad_v - pad_radius - 2),
                            f"P{index}",
                            fill=orange,
                            font=font,
                            stroke_width=max(1, int(round(scale))),
                            stroke_fill=(0, 0, 0),
                        )

            for box in target_pad_boxes or ():
                box_center = np.asarray(
                    box["center_world_m"], dtype=np.float64
                )
                box_rotation = np.asarray(
                    box["rotation_world"], dtype=np.float64
                )
                half_size = np.asarray(box["half_size_m"], dtype=np.float64)
                corners: list[tuple[int, int]] = []
                for sx in (-1.0, 1.0):
                    for sy in (-1.0, 1.0):
                        for sz in (-1.0, 1.0):
                            projected = project(
                                box_center
                                + box_rotation
                                @ (
                                    half_size
                                    * np.asarray((sx, sy, sz), dtype=np.float64)
                                )
                            )
                            if projected is not None:
                                corners.append(projected)
                hull = _convex_hull_2d(corners)
                if len(hull) >= 3:
                    draw.polygon(
                        hull,
                        outline=orange,
                        width=1 if compact else max(2, int(round(2 * scale))),
                    )

            if compact:
                draw.ellipse(
                    (
                        center[0] - outer_radius,
                        center[1] - outer_radius,
                        center[0] + outer_radius,
                        center[1] + outer_radius,
                    ),
                    outline=(0, 0, 0),
                    width=1,
                )
                draw.ellipse(
                    (
                        center[0] - inner_radius,
                        center[1] - inner_radius,
                        center[0] + inner_radius,
                        center[1] + inner_radius,
                    ),
                    outline=orange,
                    width=1,
                )
                draw.point(center, fill=orange)
            else:
                draw.text(
                    (
                        max(6, center[0] - int(round(85 * scale))),
                        max(6, center[1] - int(round(36 * scale))),
                    ),
                    "HYPOTHETICAL grip_site pose",
                    fill=orange,
                    font=label_font,
                    stroke_width=max(1, int(round(scale))),
                    stroke_fill=(0, 0, 0),
                )
            output_root.mkdir(parents=True, exist_ok=True)
            fingerprint = hashlib.sha256(
                np.concatenate((target, rotation.reshape(-1))).tobytes()
            ).hexdigest()[:10]
            output = output_root / f"agentview.candidate-pose.{fingerprint}.png"
            image.save(output)
            return output
        except (KeyError, OSError, TypeError, ValueError):
            return None

    @staticmethod
    def _views_need_pointcloud(views: Sequence[str]) -> bool:
        return any(str(view).startswith("pointcloud_") for view in views)

    def _render_operator_pointcloud_views(
        self, *, observation_id: str
    ) -> dict[str, PointCloudView]:
        """Render the configured observation-bound point cloud.

        The single-view mode uses only the current agentview RGB-D frame.
        The live multi-view mode asks the simulator that owns ``self.handle``
        for seven synchronized free-camera RGB-D frames without stepping
        physics, then retains only surfaces reproduced by at least two views.
        No static NPZ fallback is allowed in live mode.
        """
        assert self.current_record is not None

        def include_current_grip_site(bounds: np.ndarray) -> np.ndarray:
            if not self._pointcloud_current_reachable_envelope:
                return bounds
            grip_site = self._current_grip_site_xyz_for_render()
            if grip_site is None:
                return bounds
            return world_bounds_including_points(
                bounds,
                [grip_site],
                padding_m=0.10,
            )

        if self._operator_pointcloud_mode == "single-agentview":
            points, colors = world_pointcloud_from_record(
                self.current_record,
                artifact_root=self.root,
                camera_ids=("agentview",),
            )
            # Keep the historical calibrated box for the original tabletop
            # scene, but derive bounds for scenes on another world scale.
            bounds = (
                operator_scene_bounds(points)
                if len(points) and float(np.nanmedian(points[:, 2])) < 0.60
                else None
            )
            if self._pointcloud_current_reachable_envelope:
                bounds = include_current_grip_site(
                    operator_scene_bounds(points)
                )
            points, colors = workspace_pointcloud(points, colors, bounds=bounds)
            self._operator_pointcloud_points_world = points
            self._operator_pointcloud_colors_rgb = colors
            self._operator_pointcloud_world_bounds = bounds
            self._operator_pointcloud_source = "observation_rgbd:agentview"
            self._operator_pointcloud_metrics = {
                "mode": self._operator_pointcloud_mode,
                "source": self._operator_pointcloud_source,
                "observation_id": observation_id,
                "camera_count": 1,
                "workspace_points": int(len(points)),
                "points_after_voxel_fusion": int(len(points)),
                "voxels_5mm": pointcloud_coverage(points, voxel_m=0.005),
                "physics_stepped": False,
            }
            return render_world_pointcloud_views(
                points,
                colors,
                observation_record=self.current_record,
                artifact_root=self.root,
                output_root=self.root / "pointcloud_views",
                grip_site_xyz=self._current_grip_site_xyz_for_render(),
                grip_site_rotation=self._current_grip_site_rotation_for_render(),
                grip_site_aperture_m=self._current_gripper_aperture_for_render(),
                finger_pad_contact_centers_world_m=(
                    self._current_finger_pad_contacts_for_render()
                ),
                target_grip_site_xyz=self._pointcloud_target_grip_site_xyz,
                metric_edge_ticks=self._metric_pointcloud_edge_ticks,
                metric_tick_band_style=(
                    self._metric_pointcloud_tick_band_style
                ),
                compact_grip_site_overlay=(
                    self._precise_current_grip_site_feedback
                ),
                world_bounds=bounds,
            )

        pipeline_started = time.perf_counter()
        stage_started = pipeline_started
        # Center virtual cameras from the current ordinary RGB-D observation.
        # This is host-side geometry only; the Operator still has the same
        # observe / mark_point / move_to public surface.
        try:
            agentview_points, _agentview_colors = world_pointcloud_from_record(
                self.current_record,
                artifact_root=self.root,
                camera_ids=("agentview",),
            )
            scene_bounds = operator_scene_bounds(agentview_points)
            scene_bounds = include_current_grip_site(scene_bounds)
            multiview_lookat = operator_scene_lookat(
                agentview_points,
                bounds=scene_bounds,
            )
        except (OSError, ValueError, KeyError, TypeError):
            scene_bounds = None
            multiview_lookat = None
        response = self.transport.call_tool(
            "render_multiview_env",
            self._session_args(
                {
                    "handle": self.handle,
                    "width": self.image_width,
                    "height": self.image_height,
                    # The calibrated cloud is an environment/object geometry
                    # inspection surface. Robot geometry would occlude held
                    # objects and invite marks on the arm itself; exact
                    # grip-site, pad, jaw, and approach geometry is rendered
                    # separately as an overlay below.
                    "hide_robot": True,
                    **(
                        {"lookat_xyz_m": multiview_lookat.tolist()}
                        if multiview_lookat is not None
                        else {}
                    ),
                }
            ),
            timeout_s=180.0,
        )
        if not _payload_ok(response):
            raise RuntimeError(
                f"live multi-view render failed: {_payload_error(response)}"
            )
        render_multiview_s = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        materialized = materialize_mcp_images(
            dict(response),
            output_root=self.root / "mcp-input",
            bundle_id=f"multiview-{self._capture_index:06d}",
        ).payload
        cameras = materialized.get("cameras", [])
        if not isinstance(cameras, list) or len(cameras) != 7:
            raise RuntimeError(
                f"live multi-view render returned {len(cameras) if isinstance(cameras, list) else 0} cameras"
            )
        materialize_images_s = time.perf_counter() - stage_started
        frames: list[dict[str, Any]] = []
        for camera in cameras:
            if not isinstance(camera, Mapping):
                continue
            intrinsics = camera.get("intrinsics")
            extrinsics = camera.get("extrinsics")
            frames.append(
                {
                    "camera_id": str(camera.get("camera_id") or camera.get("frame_id") or ""),
                    "frame_id": str(camera.get("frame_id") or camera.get("camera_id") or ""),
                    "rgb_path": camera.get("rgb_path"),
                    "depth_path": camera.get("depth_path"),
                    "metadata": {
                        "intrinsics": dict(intrinsics)
                        if isinstance(intrinsics, Mapping)
                        else {},
                        "extrinsics": dict(extrinsics)
                        if isinstance(extrinsics, Mapping)
                        else {},
                        "virtual_camera": dict(camera.get("virtual_camera") or {}),
                    },
                }
            )
        multiview_record = {
            "observation_id": observation_id,
            "frames": frames,
        }
        stage_started = time.perf_counter()
        raw_points, raw_colors = world_pointcloud_from_record(
            multiview_record,
            artifact_root=self.root,
        )
        backproject_s = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        # Reuse the pre-render scene box so pixels, rays, and the fused cloud
        # share the same observation-bound coordinate envelope.
        if scene_bounds is None:
            scene_bounds = operator_scene_bounds(raw_points)
            scene_bounds = include_current_grip_site(scene_bounds)
        workspace_points, workspace_colors = workspace_pointcloud(
            raw_points,
            raw_colors,
            bounds=scene_bounds,
        )
        self._operator_pointcloud_world_bounds = scene_bounds
        workspace_crop_s = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        consensus_points, consensus_colors, consensus_metrics = (
            multiview_consensus_pointcloud(
                workspace_points,
                workspace_colors,
                multiview_record,
                artifact_root=self.root,
                min_support=2,
                tolerance_m=0.015,
            )
        )
        consensus_s = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        fused_points, fused_colors = voxel_fuse_pointcloud(
            consensus_points,
            consensus_colors,
            voxel_m=0.006,
        )
        self._operator_pointcloud_points_world = consensus_points
        self._operator_pointcloud_colors_rgb = consensus_colors
        voxel_fusion_s = time.perf_counter() - stage_started
        if len(fused_points) == 0:
            raise RuntimeError("live multi-view consensus produced no points")
        stage_started = time.perf_counter()
        rendered_views = render_world_pointcloud_views(
            fused_points,
            fused_colors,
            observation_record=self.current_record,
            artifact_root=self.root,
            output_root=self.root / "pointcloud_views",
            grip_site_xyz=self._current_grip_site_xyz_for_render(),
            grip_site_rotation=self._current_grip_site_rotation_for_render(),
            grip_site_aperture_m=self._current_gripper_aperture_for_render(),
            finger_pad_contact_centers_world_m=(
                self._current_finger_pad_contacts_for_render()
            ),
            target_grip_site_xyz=self._pointcloud_target_grip_site_xyz,
            metric_edge_ticks=self._metric_pointcloud_edge_ticks,
            metric_tick_band_style=self._metric_pointcloud_tick_band_style,
            compact_grip_site_overlay=(
                self._precise_current_grip_site_feedback
            ),
            world_bounds=scene_bounds,
        )
        orthographic_render_s = time.perf_counter() - stage_started
        self._operator_pointcloud_source = "live_multiview_consensus"
        self._operator_pointcloud_metrics = {
            "mode": self._operator_pointcloud_mode,
            "source": self._operator_pointcloud_source,
            "observation_id": observation_id,
            "camera_count": len(frames),
            "raw_points": int(len(raw_points)),
            "workspace_points": int(len(workspace_points)),
            "consensus_points": int(len(consensus_points)),
            "points_after_voxel_fusion": int(len(fused_points)),
            "voxels_5mm": pointcloud_coverage(fused_points, voxel_m=0.005),
            "physics_stepped": bool(response.get("physics_stepped", True)),
            "robot_visuals_hidden": bool(
                response.get("robot_visuals_hidden", False)
            ),
            "scene_bounds_m": scene_bounds.tolist(),
            "hidden_robot_visual_geom_count": int(
                response.get("hidden_robot_visual_geom_count", 0) or 0
            ),
            "consensus": consensus_metrics,
            "timings_s": {
                "render_multiview": render_multiview_s,
                "materialize_images": materialize_images_s,
                "backproject": backproject_s,
                "workspace_crop": workspace_crop_s,
                "consensus": consensus_s,
                "voxel_fusion": voxel_fusion_s,
                "orthographic_render": orthographic_render_s,
                "total": time.perf_counter() - pipeline_started,
            },
        }
        metrics_path = (
            self.root / "pointcloud_views" / observation_id / "cloud_metrics.json"
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(self._operator_pointcloud_metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        return rendered_views

    def _current_grip_site_xyz_for_render(self) -> list[float] | None:
        robot = self.current_sim_payload.get("robot", {})
        pose = robot.get("end_effector_pose", {}) if isinstance(robot, Mapping) else {}
        try:
            return _finite_vector3(
                pose.get("xyz") if isinstance(pose, Mapping) else None,
                name="current grip-site translation",
            )
        except (TypeError, ValueError):
            return None

    def _current_grip_site_rotation_for_render(
        self,
    ) -> list[list[float]] | None:
        robot = self.current_sim_payload.get("robot", {})
        pose = robot.get("end_effector_pose", {}) if isinstance(robot, Mapping) else {}
        rotation = _quat_xyzw_to_rotation_matrix(
            pose.get("quat_xyzw") if isinstance(pose, Mapping) else None
        )
        return rotation.tolist() if rotation is not None else None

    def _current_gripper_aperture_for_render(self) -> float | None:
        robot = self.current_sim_payload.get("robot", {})
        state = robot.get("gripper_state", {}) if isinstance(robot, Mapping) else {}
        aperture = state.get("aperture_m") if isinstance(state, Mapping) else None
        if (
            not isinstance(aperture, (int, float))
            or not math.isfinite(float(aperture))
        ):
            return None
        return max(0.0, float(aperture))

    def _current_finger_pad_contacts_for_render(
        self,
    ) -> list[list[float]] | None:
        robot = self.current_sim_payload.get("robot", {})
        state = robot.get("gripper_state", {}) if isinstance(robot, Mapping) else {}
        geometry = state.get("geometry", {}) if isinstance(state, Mapping) else {}
        points = (
            geometry.get("finger_pad_inner_contact_centers_world_m")
            if isinstance(geometry, Mapping)
            else None
        )
        try:
            array = np.asarray(points, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if array.shape != (2, 3) or not np.isfinite(array).all():
            return None
        return array.tolist()

    def _current_finger_pad_boxes_for_render(
        self,
    ) -> list[dict[str, list[Any]]] | None:
        """Return the two live MuJoCo pad collision boxes for visualization."""

        robot = self.current_sim_payload.get("robot", {})
        state = robot.get("gripper_state", {}) if isinstance(robot, Mapping) else {}
        geometry = state.get("geometry", {}) if isinstance(state, Mapping) else {}
        if not isinstance(geometry, Mapping):
            return None
        try:
            centers = np.asarray(
                geometry.get("finger_pad_geom_centers_world_m"),
                dtype=np.float64,
            )
            rotations = np.asarray(
                geometry.get("finger_pad_geom_rotations_world"),
                dtype=np.float64,
            )
            half_sizes = np.asarray(
                geometry.get("finger_pad_geom_half_sizes_m"),
                dtype=np.float64,
            )
        except (TypeError, ValueError):
            return None
        if (
            centers.shape != (2, 3)
            or rotations.shape != (2, 3, 3)
            or half_sizes.shape != (2, 3)
            or not np.isfinite(centers).all()
            or not np.isfinite(rotations).all()
            or not np.isfinite(half_sizes).all()
        ):
            return None
        return [
            {
                "center_world_m": centers[index].tolist(),
                "rotation_world": rotations[index].tolist(),
                "half_size_m": half_sizes[index].tolist(),
            }
            for index in range(2)
        ]

    def _candidate_pad_geometry_for_pose(
        self,
        *,
        position_xyz_m: Sequence[float],
        rotation_matrix: Sequence[Sequence[float]],
        requested_gripper: str | None = None,
    ) -> tuple[list[list[float]] | None, list[dict[str, Any]] | None]:
        """Transfer pad geometry to a candidate pose.

        ``requested_gripper`` only changes the *nominal visual aperture* used
        by a preview.  It never simulates contact, collision, or retention.
        When omitted, the measured live geometry is transferred unchanged.
        """

        actual = self._current_world_from_grip_site()
        candidate_position = np.asarray(position_xyz_m, dtype=np.float64)
        candidate_rotation = np.asarray(rotation_matrix, dtype=np.float64)
        actual_position = actual[:3, 3]
        actual_rotation = actual[:3, :3]
        contacts = self._current_finger_pad_contacts_for_render()
        candidate_contacts = None
        if contacts is not None:
            contact_array = np.asarray(contacts, dtype=np.float64)
            local_contacts = (
                actual_rotation.T
                @ (contact_array - actual_position).T
            ).T
            if requested_gripper in {"open", "close"}:
                # The Panda jaw axis is grip-site +X.  Preserve the pad
                # midpoint and each pad's local Y/Z offsets while replacing
                # only the nominal inner-contact separation.
                nominal_aperture = (
                    PANDA_MAX_APERTURE_M
                    if requested_gripper == "open"
                    else 0.008
                )
                midpoint = local_contacts.mean(axis=0)
                x_signs = np.sign(local_contacts[:, 0] - midpoint[0])
                if not np.all(np.abs(x_signs) > 1e-6):
                    x_signs = np.asarray([-1.0, 1.0])
                local_contacts[:, 0] = (
                    midpoint[0] + x_signs * nominal_aperture / 2.0
                )
            candidate_contacts = (
                candidate_position
                + (candidate_rotation @ local_contacts.T).T
            ).tolist()
        boxes = self._current_finger_pad_boxes_for_render()
        candidate_boxes = None
        if boxes is not None:
            candidate_boxes = []
            live_contacts_local = None
            if contacts is not None:
                live_contacts_local = (
                    actual_rotation.T
                    @ (
                        np.asarray(contacts, dtype=np.float64)
                        - actual_position
                    ).T
                ).T
            for box_index, box in enumerate(boxes):
                center = np.asarray(box["center_world_m"], dtype=np.float64)
                rotation = np.asarray(box["rotation_world"], dtype=np.float64)
                local_center = actual_rotation.T @ (center - actual_position)
                local_rotation = actual_rotation.T @ rotation
                if (
                    requested_gripper in {"open", "close"}
                    and live_contacts_local is not None
                ):
                    nominal_aperture = (
                        PANDA_MAX_APERTURE_M
                        if requested_gripper == "open"
                        else 0.008
                    )
                    local_contact_midpoint = live_contacts_local.mean(axis=0)
                    x_sign = float(
                        np.sign(
                            live_contacts_local[box_index, 0]
                            - local_contact_midpoint[0]
                        )
                    )
                    if abs(x_sign) < 1e-6:
                        x_sign = -1.0 if box_index == 0 else 1.0
                    contact_to_center_x = (
                        local_center[0]
                        - live_contacts_local[box_index, 0]
                    )
                    local_center[0] = (
                        local_contact_midpoint[0]
                        + x_sign * nominal_aperture / 2.0
                        + contact_to_center_x
                    )
                candidate_boxes.append(
                    {
                        "center_world_m": (
                            candidate_position
                            + candidate_rotation @ local_center
                        ).tolist(),
                        "rotation_world": (
                            candidate_rotation @ local_rotation
                        ).tolist(),
                        "half_size_m": list(box["half_size_m"]),
                    }
                )
        return candidate_contacts, candidate_boxes

    def _grip_site_footprint_for_mark(
        self, xyz_m: Sequence[float]
    ) -> dict[str, Any]:
        """Return numeric pad-footprint evidence for an ordinary world mark."""

        pose = self._current_world_from_grip_site()
        contacts, boxes = self._candidate_pad_geometry_for_pose(
            position_xyz_m=xyz_m,
            rotation_matrix=pose[:3, :3],
        )
        result: dict[str, Any] = {
            "frame": "world",
            "eef_frame": "panda_grip_site",
            "orientation_source": "current_grip_site",
            "diagnostic_only": True,
            "pad_contact_centers_world_m": contacts,
        }
        if boxes is not None:
            result["pad_boxes_world"] = boxes
        return result

    def _render_agentview_world_axes(self) -> Path | None:
        """Render calibrated world-axis directions over a copy of agentview.

        The overlay is an orientation aid, not a metric coordinate view. Axis
        directions are projected through the current camera intrinsics and
        extrinsics instead of being hard-coded to screen directions.
        """

        if self.current_record is None:
            return None
        frame = _find_frame(self.current_record, "agentview")
        if not isinstance(frame, Mapping):
            return None
        raw_path = frame.get("rgb_path")
        metadata = frame.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        intrinsics = metadata.get("intrinsics")
        extrinsics = metadata.get("extrinsics")
        if (
            not isinstance(raw_path, str)
            or not isinstance(intrinsics, Mapping)
            or not isinstance(extrinsics, Mapping)
        ):
            return None
        image_path = Path(raw_path)
        if not image_path.is_absolute():
            image_path = self.root / image_path
        if not image_path.is_file():
            return None
        try:
            fx = float(intrinsics["fx"])
            fy = float(intrinsics["fy"])
            world_from_camera = camera_to_world_opencv(extrinsics)
            camera_from_world_rotation = world_from_camera[:3, :3].T
            projected: dict[str, np.ndarray | None] = {}
            for axis, world_vector in {
                "X": np.array([1.0, 0.0, 0.0]),
                "Y": np.array([0.0, 1.0, 0.0]),
                "Z": np.array([0.0, 0.0, 1.0]),
            }.items():
                camera_vector = camera_from_world_rotation @ world_vector
                # Differential pinhole projection at a point on the optical
                # axis. This retains the calibrated image-plane direction.
                direction = np.array(
                    [fx * camera_vector[0], fy * camera_vector[1]],
                    dtype=np.float64,
                )
                norm = float(np.linalg.norm(direction))
                if norm > 1e-8:
                    projected[axis] = direction / norm
                else:
                    projected[axis] = None
            if len(projected) != 3:
                return None

            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            scale = max(0.7, min(image.width, image.height) / 512.0)
            origin = np.array([42.0 * scale, 48.0 * scale])
            length = 30.0 * scale
            colors = {
                "X": (255, 80, 70),
                "Y": (80, 220, 100),
                "Z": (80, 140, 255),
            }
            font = _gateway_font(max(11, int(round(13 * scale))))
            outline_width = max(1, int(round(2 * scale)))
            for axis, direction in projected.items():
                if direction is None:
                    center = tuple(float(value) for value in origin)
                    radius = 7.0 * scale
                    draw.ellipse(
                        (
                            center[0] - radius,
                            center[1] - radius,
                            center[0] + radius,
                            center[1] + radius,
                        ),
                        outline=colors[axis],
                        width=max(2, int(round(3 * scale))),
                    )
                    draw.ellipse(
                        (
                            center[0] - 2 * scale,
                            center[1] - 2 * scale,
                            center[0] + 2 * scale,
                            center[1] + 2 * scale,
                        ),
                        fill=colors[axis],
                    )
                    draw.text(
                        (center[0] + 9 * scale, center[1] - 8 * scale),
                        f"+{axis}",
                        fill=colors[axis],
                        font=font,
                        stroke_width=outline_width,
                        stroke_fill=(0, 0, 0),
                    )
                    continue
                endpoint = origin + direction * length
                start_xy = tuple(float(value) for value in origin)
                end_xy = tuple(float(value) for value in endpoint)
                draw.line(
                    (start_xy, end_xy),
                    fill=colors[axis],
                    width=max(2, int(round(3 * scale))),
                )
                angle = math.atan2(
                    float(endpoint[1] - origin[1]),
                    float(endpoint[0] - origin[0]),
                )
                wing = 8.0 * scale
                for offset in (2.55, -2.55):
                    wing_end = (
                        float(endpoint[0] + math.cos(angle + offset) * wing),
                        float(endpoint[1] + math.sin(angle + offset) * wing),
                    )
                    draw.line(
                        (end_xy, wing_end),
                        fill=colors[axis],
                        width=max(2, int(round(3 * scale))),
                    )
                draw.text(
                    (float(endpoint[0] + 4 * scale), float(endpoint[1] - 8 * scale)),
                    f"+{axis}",
                    fill=colors[axis],
                    font=font,
                    stroke_width=outline_width,
                    stroke_fill=(0, 0, 0),
                )
            draw.ellipse(
                (
                    float(origin[0] - 2 * scale),
                    float(origin[1] - 2 * scale),
                    float(origin[0] + 2 * scale),
                    float(origin[1] + 2 * scale),
                ),
                fill=(255, 255, 255),
                outline=(0, 0, 0),
                width=outline_width,
            )
            marker = self._current_grip_site_pixel(frame)
            if self._precise_current_grip_site_feedback and marker is not None:
                self._draw_precise_grip_site_marker(image, marker)
            output = (
                self.root
                / "pointcloud_views"
                / str(self.current_record.get("observation_id") or "current")
                / "agentview.world_axes.png"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            image.save(output)
            return output
        except (KeyError, OSError, TypeError, ValueError, GraspPoseRefinementError):
            return None

    def _render_wrist_grip_site(self) -> Path | None:
        """Render the current grip-site as a compact calibrated wrist marker."""

        if (
            not self._precise_current_grip_site_feedback
            or self.current_record is None
        ):
            return None
        frame = _find_frame(self.current_record, "wrist")
        if not isinstance(frame, Mapping):
            return None
        raw_path = frame.get("rgb_path")
        if not isinstance(raw_path, str):
            return None
        image_path = Path(raw_path)
        if not image_path.is_absolute():
            image_path = self.root / image_path
        marker = self._current_grip_site_pixel(frame)
        if marker is None or not image_path.is_file():
            return None
        try:
            image = Image.open(image_path).convert("RGB")
            if not (0 <= marker[0] < image.width and 0 <= marker[1] < image.height):
                return None
            self._draw_precise_grip_site_marker(image, marker)
            output = (
                self.root
                / "pointcloud_views"
                / str(self.current_record.get("observation_id") or "current")
                / "wrist.grip_site.png"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            image.save(output)
            return output
        except OSError:
            return None

    def _current_grip_site_pixel(
        self,
        frame: Mapping[str, Any],
    ) -> tuple[int, int] | None:
        """Project the measured grip-site into one calibrated camera frame."""

        grip_site = self._current_grip_site_xyz_for_render()
        if grip_site is None:
            return None
        projection = self._world_point_pixel(frame, grip_site)
        return projection[:2] if projection is not None else None

    @staticmethod
    def _world_point_pixel(
        frame: Mapping[str, Any],
        xyz_m: Sequence[float],
    ) -> tuple[int, int, float] | None:
        """Project one world point into a calibrated camera frame."""

        metadata = frame.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        intrinsics = metadata.get("intrinsics")
        extrinsics = metadata.get("extrinsics")
        if not isinstance(intrinsics, Mapping) or not isinstance(
            extrinsics, Mapping
        ):
            return None
        try:
            world_from_camera = camera_to_world_opencv(extrinsics)
            point = np.asarray(xyz_m, dtype=np.float64).reshape(3)
            camera_point = world_from_camera[:3, :3].T @ (
                point - world_from_camera[:3, 3]
            )
            if camera_point[2] <= 1e-6 or not np.isfinite(camera_point).all():
                return None
            return (
                int(
                    round(
                        float(intrinsics["fx"])
                        * camera_point[0]
                        / camera_point[2]
                        + float(intrinsics["cx"])
                    )
                ),
                int(
                    round(
                        float(intrinsics["fy"])
                        * camera_point[1]
                        / camera_point[2]
                        + float(intrinsics["cy"])
                    )
                ),
                float(camera_point[2]),
            )
        except (KeyError, TypeError, ValueError, GraspPoseRefinementError):
            return None

    def _render_current_mujoco_contact_markers(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Path]:
        """Render only the final physics step's contacts on current RGB views."""

        if (
            not self._not_reached_current_contact_marker
            or self.current_record is None
        ):
            return {}
        contacts = payload.get("mujoco_contacts")
        points = contacts.get("points") if isinstance(contacts, Mapping) else None
        if not isinstance(points, list) or not points:
            return {}
        world_points: list[np.ndarray] = []
        for item in points:
            if self._not_reached_robot_contact_only:
                geom_names = (
                    item.get("geom_names")
                    if isinstance(item, Mapping)
                    else None
                )
                if not (
                    isinstance(geom_names, list)
                    and any(
                        str(name).startswith(("robot0_", "gripper0_"))
                        for name in geom_names
                    )
                ):
                    continue
            xyz = item.get("position_xyz_m") if isinstance(item, Mapping) else None
            try:
                point = np.asarray(xyz, dtype=np.float64).reshape(3)
            except (TypeError, ValueError):
                continue
            if np.isfinite(point).all():
                world_points.append(point)
            if len(world_points) >= 6:
                break
        if not world_points:
            return {}

        observation_id = str(
            self.current_record.get("observation_id") or "current"
        )
        digest = hashlib.sha256(
            json.dumps(
                [point.tolist() for point in world_points],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:10]
        rendered: dict[str, Path] = {}
        for camera_id in ("agentview", "wrist"):
            frame = _find_frame(self.current_record, camera_id)
            if not isinstance(frame, Mapping):
                continue
            source_path = self._operator_view_path(camera_id)
            if source_path is None or not source_path.is_file():
                continue
            try:
                image = Image.open(source_path).convert("RGB")
            except OSError:
                continue
            depth_m: np.ndarray | None = None
            raw_depth_path = frame.get("depth_path")
            if isinstance(raw_depth_path, str):
                depth_path = Path(raw_depth_path)
                if not depth_path.is_absolute():
                    depth_path = self.root / depth_path
                try:
                    depth_m = (
                        np.asarray(Image.open(depth_path), dtype=np.float64)
                        / 1000.0
                    )
                except OSError:
                    depth_m = None

            visible_pixels: list[tuple[int, int]] = []
            for point in world_points:
                projection = self._world_point_pixel(frame, point)
                if projection is None:
                    continue
                u, v, camera_depth_m = projection
                if not (0 <= u < image.width and 0 <= v < image.height):
                    continue
                if depth_m is not None and depth_m.ndim == 2:
                    y0, y1 = max(0, v - 1), min(depth_m.shape[0], v + 2)
                    x0, x1 = max(0, u - 1), min(depth_m.shape[1], u + 2)
                    patch = depth_m[y0:y1, x0:x1]
                    valid = patch[np.isfinite(patch) & (patch > 0.0)]
                    if (
                        valid.size
                        and camera_depth_m
                        > float(np.median(valid)) + 0.025
                    ):
                        continue
                if any(
                    abs(u - previous_u) <= 2 and abs(v - previous_v) <= 2
                    for previous_u, previous_v in visible_pixels
                ):
                    continue
                visible_pixels.append((u, v))

            if not visible_pixels:
                continue
            draw = ImageDraw.Draw(image)
            for u, v in visible_pixels:
                draw.ellipse(
                    (u - 2, v - 2, u + 2, v + 2),
                    outline=(0, 0, 0),
                    width=1,
                )
                draw.ellipse(
                    (u - 1, v - 1, u + 1, v + 1),
                    outline=(0, 255, 255),
                    width=1,
                )
                draw.point((u, v), fill=(0, 255, 255))
            output = (
                self.root
                / "contact_feedback"
                / observation_id
                / f"{camera_id}.mujoco-contact.{digest}.png"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            image.save(output)
            rendered[camera_id] = output
        return rendered

    @staticmethod
    def _draw_precise_grip_site_marker(
        image: Image.Image,
        center: tuple[int, int],
    ) -> None:
        """Draw a one-pixel center with a compact ring for visibility."""

        u, v = center
        draw = ImageDraw.Draw(image)
        scale = max(0.5, min(image.width, image.height) / 512.0)
        outer_radius = max(3, int(round(3 * scale)))
        inner_radius = max(2, outer_radius - 1)
        draw.ellipse(
            (
                u - outer_radius,
                v - outer_radius,
                u + outer_radius,
                v + outer_radius,
            ),
            outline=(0, 0, 0),
            width=1,
        )
        draw.ellipse(
            (
                u - inner_radius,
                v - inner_radius,
                u + inner_radius,
                v + inner_radius,
            ),
            outline=(255, 0, 255),
            width=1,
        )
        draw.point((u, v), fill=(255, 0, 255))

    def _render_agentview_ray_click(
        self,
        *,
        u: int,
        v: int,
        point_id: str,
    ) -> Path | None:
        """Draw the semantic source pixel without altering the raw RGB."""

        if self.current_record is None:
            return None
        frame = _find_frame(self.current_record, "agentview")
        if not isinstance(frame, Mapping):
            return None
        raw_path = frame.get("rgb_path")
        if not isinstance(raw_path, str):
            return None
        image_path = Path(raw_path)
        if not image_path.is_absolute():
            image_path = self.root / image_path
        if not image_path.is_file():
            return None
        try:
            image = Image.open(image_path).convert("RGB")
            current_marker = self._current_grip_site_pixel(frame)
            if (
                self._precise_current_grip_site_feedback
                and current_marker is not None
            ):
                self._draw_precise_grip_site_marker(image, current_marker)
            u = max(0, min(image.width - 1, int(u)))
            v = max(0, min(image.height - 1, int(v)))
            draw = ImageDraw.Draw(image)
            color = (255, 230, 70)
            draw.ellipse((u - 10, v - 10, u + 10, v + 10), outline=color, width=4)
            draw.line((u - 16, v, u + 16, v), fill=(255, 70, 210), width=3)
            draw.line((u, v - 16, u, v + 16), fill=(255, 70, 210), width=3)
            draw.text(
                (u + 13, v + 9),
                f"{point_id} ray source",
                fill=color,
                font=_gateway_font(14),
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
            output = (
                self.root
                / "pointcloud_views"
                / str(self.current_record.get("observation_id") or "current")
                / (
                    "agentview.ray."
                    + hashlib.sha256(
                        json.dumps(
                            {
                                "point_id": point_id,
                                "pixel_xy": [u, v],
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()[:10]
                    + ".png"
                )
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            image.save(output)
            return output
        except OSError:
            return None

    def _write_current_projection(self) -> None:
        if self.current_record is None:
            return
        frames = []
        for frame in self.current_record.get("frames", []):
            if not isinstance(frame, Mapping):
                continue
            frames.append(
                {
                    "camera_id": frame.get("camera_id"),
                    "frame_id": frame.get("frame_id"),
                    "rgb_path": (
                        str(self._agentview_ray_overlay.relative_to(self.root))
                        if frame.get("camera_id") == "agentview"
                        and self._agentview_ray_overlay is not None
                        else str(self._agentview_axis_overlay.relative_to(self.root))
                        if frame.get("camera_id") == "agentview"
                        and self._agentview_axis_overlay is not None
                        else str(self._wrist_grip_site_overlay.relative_to(self.root))
                        if frame.get("camera_id") == "wrist"
                        and self._wrist_grip_site_overlay is not None
                        else frame.get("rgb_path")
                    ),
                    **(
                        {"raw_rgb_path": frame.get("rgb_path")}
                        if (
                            frame.get("camera_id") == "agentview"
                            and (
                                self._agentview_axis_overlay is not None
                                or self._agentview_ray_overlay is not None
                            )
                        )
                        or (
                            frame.get("camera_id") == "wrist"
                            and self._wrist_grip_site_overlay is not None
                        )
                        else {}
                    ),
                }
            )
        pointcloud_views = [
            view.public_dict(self.root) for view in self._pointcloud_views.values()
        ]
        payload = {
            "schema_version": "openeta.operator_projection.v1",
            "status": self._episode_status,
            "failure_case": self.failure_case,
            "latest_issue": self.issues[-1] if self.issues else None,
            "issue_count": len(self.issues),
            "task": self.task,
            "sim_step": self._sim_step,
            "viewer_url": self.dashboard_url,
            "visualization": self.current_visualization,
            "pointcloud_views": pointcloud_views,
            "pointcloud_source": self._operator_pointcloud_source,
            "pointcloud_mode": self._operator_pointcloud_mode,
            "pointcloud_observation_id": self._operator_pointcloud_metrics.get(
                "observation_id"
            ),
            "pointcloud_metrics": self._operator_pointcloud_metrics,
            "point_marks_3d": self._point_marks_3d,
            "pending_point_constraints": self._pending_point_constraints,
            "point_view_paths": self._point_view_paths,
            "active_grip_site_target": self._active_grip_site_target,
            "active_grip_site_view_paths": self._active_grip_site_view_paths,
            "current_authored_point_id": self._current_authored_point_id,
            "record": self.current_record,
            "operator": {
                "observation_id": self.current_record.get("observation_id"),
                "frames": frames,
                "numeric_state_hidden": True,
            },
        }
        path = self.root / "current.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        temporary.replace(path)
        operator_payload = {
            "schema_version": "openeta.operator_view.v1",
            "status": self._episode_status,
            "failure_case": (
                {
                    "failure_case_id": self.failure_case.get("failure_case_id"),
                    "component": self.failure_case.get("component"),
                    "code": self.failure_case.get("code"),
                    "message": self.failure_case.get("message"),
                }
                if self.failure_case
                else None
            ),
            "latest_issue": (
                {
                    "issue_id": self.issues[-1].get("issue_id"),
                    "component": self.issues[-1].get("component"),
                    "code": self.issues[-1].get("code"),
                    "message": self.issues[-1].get("message"),
                }
                if self.issues
                else None
            ),
            "task": self.task,
            "sim_step": self._sim_step,
            "viewer_url": self.dashboard_url,
            "observation_id": self.current_record.get("observation_id"),
            "frames": frames,
            "numeric_state_hidden": True,
        }
        operator_path = self.root / "operator.json"
        operator_temporary = operator_path.with_suffix(".tmp")
        operator_temporary.write_text(
            json.dumps(operator_payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        operator_temporary.replace(operator_path)

    def _observation_result(
        self,
        *,
        views: Sequence[str] | None = None,
    ) -> GatewayResult:
        assert self.current_record is not None
        frames = self.current_record.get("frames", [])
        requested_views, view_error = self._normalize_operator_views(
            views,
            default=self._default_operator_observe_views(),
        )
        if view_error is not None:
            return view_error
        if self._views_need_pointcloud(requested_views):
            self._ensure_operator_pointcloud_views()
        images = self._operator_view_paths(requested_views)
        view_contract = self._operator_view_contract(requested_views)
        full_text = {
            "kind": "observation",
            "success": True,
            "task": self.task,
            "observation_id": self.current_record.get("observation_id"),
            "sim_step": self._sim_step,
            "viewer_url": self.dashboard_url,
            "cameras": [
                {
                    "camera_id": frame.get("camera_id"),
                    "frame_id": frame.get("frame_id"),
                    "visual_label": str(frame.get("camera_id") or "camera"),
                }
                for frame in frames
                if isinstance(frame, Mapping)
            ],
            "pointcloud_views": [
                view.public_dict(self.root, absolute_paths=True)
                for view in self._pointcloud_views.values()
            ],
            "pointcloud_source": self._operator_pointcloud_source,
            "pointcloud_mode": self._operator_pointcloud_mode,
            "pointcloud_metrics": self._operator_pointcloud_metrics,
            "pointcloud_contact_sheet": (
                str(self._pointcloud_contact_sheet.resolve())
                if self._pointcloud_contact_sheet is not None
                else None
            ),
            "requested_views": requested_views,
            "returned_views": self._returned_operator_view_names(
                requested_views
            ),
            "available_views": list(self._operator_view_names()),
            "view_contract": view_contract,
        }
        actual_position = self._current_grip_site_xyz_for_render()
        actual_rotation = self._current_grip_site_rotation_for_render()
        aperture_m = self._current_gripper_aperture_for_render()
        actual_jaw_direction = None
        actual_approach_direction = None
        if actual_rotation is not None:
            rotation = np.asarray(actual_rotation, dtype=np.float64)
            if rotation.shape == (3, 3) and np.isfinite(rotation).all():
                actual_jaw_direction = _finite_unit_direction(rotation[:, 0])
                actual_approach_direction = _finite_unit_direction(
                    rotation[:, 2]
                )
        text = {
            "success": True,
            "observation_id": self.current_record.get("observation_id"),
            "returned_views": full_text["returned_views"],
            "actual_grip_site_xyz_m": (
                [round(float(value), 4) for value in actual_position]
                if actual_position is not None
                else None
            ),
            "gripper_aperture_mm": (
                round(float(aperture_m) * 1000.0, 2)
                if aperture_m is not None
                else None
            ),
        }
        if self._operator_compact_result:
            if not self._operator_gripper_reference_sent:
                references = {
                    "open": round(PANDA_MAX_APERTURE_M * 1000.0, 2),
                    "near_closed": round(
                        PANDA_NEAR_CLOSED_THRESHOLD_M * 1000.0,
                        2,
                    ),
                }
                if (
                    active_profile()
                    .manifest.get("invariants", {})
                    .get("move_to_aperture_reference_result")
                    == "compact_v1"
                ):
                    references["closed"] = round(
                        PANDA_CLOSED_APERTURE_M * 1000.0,
                        2,
                    )
                text["gripper_reference_mm"] = references
                self._operator_gripper_reference_sent = True
        else:
            text["kind"] = "observation"
        full_text["spatial"] = {
            "frame": "world",
            "eef_frame": "panda_grip_site",
            "actual_grip_site_position_xyz_m": actual_position,
            "actual_approach_direction_world": actual_approach_direction,
            "actual_jaw_direction_world": actual_jaw_direction,
            "gripper_aperture_mm": (
                round(float(aperture_m) * 1000.0, 3)
                if aperture_m is not None
                else None
            ),
        }
        full_text["gripper"] = {
            "aperture_mm": (
                round(float(aperture_m) * 1000.0, 3)
                if aperture_m is not None
                else None
            ),
            "open_aperture_mm": 80.0,
            # This is the Operator-facing near-closed threshold, matching the
            # adaptive LIBERO primitive's 2.5 mm endpoint + 1.0 mm tolerance.
            "closed_aperture_mm": round(
                PANDA_NEAR_CLOSED_THRESHOLD_M * 1000.0,
                2,
            ),
        }
        return GatewayResult(
            True,
            text,
            images=images,
            details={
                "record": self.current_record,
                "observe_full_result": full_text,
                "pointcloud_views": full_text["pointcloud_views"],
                "pointcloud_contact_sheet": full_text["pointcloud_contact_sheet"],
                "point_marks_3d": self._point_marks_3d,
                "point_view_paths": self._point_view_paths,
            },
        )

    def _default_operator_observe_views(self) -> tuple[str, ...]:
        if self._operator_source_view_only_images:
            return ("agentview", "wrist")
        return ("agentview", "pointcloud_contact_sheet")

    def _operator_view_names(self) -> tuple[str, ...]:
        source_views = (
            "agentview",
            "wrist",
            "pointcloud_top",
            "pointcloud_front",
            "pointcloud_side",
        )
        if self._operator_source_view_only_images:
            return source_views
        return (*source_views, "pointcloud_contact_sheet")

    @staticmethod
    def _remaining_point_mark_view_names(
        pending_constraint: Mapping[str, Any] | None,
    ) -> list[str]:
        """Return only source views that can complete a pending point."""

        ordered = [
            "pointcloud_top",
            "pointcloud_front",
            "pointcloud_side",
        ]
        if not isinstance(pending_constraint, Mapping):
            return []
        if (
            str(pending_constraint.get("source_kind") or "")
            == "agentview_camera_ray"
        ):
            return ordered
        source_view = str(pending_constraint.get("view") or "")
        return [name for name in ordered if name != source_view]

    def _remaining_point_mark_images(
        self,
        pending_views: Mapping[str, Path],
        pending_constraint: Mapping[str, Any] | None,
    ) -> tuple[list[str], list[Path]]:
        names = self._remaining_point_mark_view_names(pending_constraint)
        return names, [
            pending_views[name]
            for name in names
            if name in pending_views
        ]

    def _operator_view_contract(
        self,
        views: Sequence[str],
    ) -> dict[str, Any]:
        """Describe the coordinate role and source pixel bounds of returned views.

        The Operator may receive several image attachments for one tool call.
        This contract makes it explicit which images are authoritative source
        views for pixel coordinates and which are explanatory composites.
        """

        pointcloud_names = {
            "pointcloud_top",
            "pointcloud_front",
            "pointcloud_side",
        }
        contract: dict[str, Any] = {}
        for name in views:
            if name in pointcloud_names:
                view = self._pointcloud_views.get(name)
                if view is None:
                    continue
                contract[name] = {
                    "image_role": "source_view",
                    "pixel_size": [int(view.width), int(view.height)],
                    "pixel_origin": [0, 0],
                    "valid_box_xyxy_exclusive": [
                        0,
                        0,
                        int(view.width),
                        int(view.height),
                    ],
                    "coordinate_system": "source_pixels_zero_based",
                    "crop_allowed": (
                        self._operator_observe_inspection_enabled
                    ),
                }
            elif name == "pointcloud_contact_sheet":
                contract[name] = {
                    "image_role": "composite_explanation",
                    "crop_allowed": False,
                    "coordinate_system": "display_only",
                }
            elif name.startswith("inspection:"):
                metadata = self._inspection_views.get(name)
                if metadata is None:
                    continue
                contract[name] = {
                    "image_role": "inspection_view",
                    "source_view": metadata["source_view"],
                    "pixel_size": [
                        int(metadata["display_size_xy"][0]),
                        int(metadata["display_size_xy"][1]),
                    ],
                    "coordinate_system": "inspection_display_pixels",
                    "mark_point_allowed": True,
                    "crop_allowed": False,
                }
            else:
                contract[name] = {
                    "image_role": "camera_or_display_view",
                    "crop_allowed": False,
                }
        return contract

    def _normalize_operator_views(
        self,
        views: Sequence[str] | None,
        *,
        default: Sequence[str],
    ) -> tuple[list[str], GatewayResult | None]:
        if views is None:
            normalized = list(default)
        elif isinstance(views, (str, bytes)) or not isinstance(views, Sequence):
            normalized = []
            return normalized, GatewayResult(False, {
                "kind": "observation",
                "success": False,
                "reason": "invalid_views",
                "message": "views must be a list of supported view names.",
                "available_views": list(self._operator_view_names()),
            })
        else:
            normalized = []
            for raw in views:
                name = str(raw).strip()
                if name and name not in normalized:
                    normalized.append(name)
        unknown = [
            name for name in normalized
            if name not in self._operator_view_names()
        ]
        if unknown:
            return normalized, GatewayResult(False, {
                "kind": "observation",
                "success": False,
                "reason": "unknown_views",
                "unknown_views": unknown,
                "available_views": list(self._operator_view_names()),
                "message": "Choose views only from available_views.",
            })
        if not normalized:
            return normalized, GatewayResult(False, {
                "kind": "observation",
                "success": False,
                "reason": "empty_views",
                "available_views": list(self._operator_view_names()),
                "message": "Choose at least one view.",
            })
        return normalized, None

    def _operator_view_path(self, name: str) -> Path | None:
        assert self.current_record is not None
        if name in {"agentview", "wrist"}:
            marked_ref = self._point_view_paths.get(name)
            if marked_ref:
                marked = self.root / marked_ref
                if marked.is_file():
                    return marked
            if (
                name == "agentview"
                and self._agentview_axis_overlay is not None
                and self._agentview_axis_overlay.is_file()
            ):
                return self._agentview_axis_overlay
            if (
                name == "wrist"
                and self._wrist_grip_site_overlay is not None
                and self._wrist_grip_site_overlay.is_file()
            ):
                return self._wrist_grip_site_overlay
            frame = _find_frame(self.current_record, name)
            paths = self._frame_paths([frame] if frame is not None else [])
            return paths[0] if paths else None
        if name == "pointcloud_contact_sheet":
            return (
                self._pointcloud_contact_sheet
                if self._pointcloud_contact_sheet is not None
                and self._pointcloud_contact_sheet.is_file()
                else None
            )
        view = self._pointcloud_views.get(name)
        if view is None:
            return None
        marked_ref = self._point_view_paths.get(name)
        if marked_ref:
            marked = self.root / marked_ref
            if marked.is_file():
                return marked
        active_ref = self._active_grip_site_view_paths.get(name)
        if active_ref:
            active = self.root / active_ref
            if active.is_file():
                return active
        # The persistent planning surface intentionally retains the current
        # actual grip-site. Historical marks stay out of the image; only the
        # active target and current authored point are composed above it.
        if view.image_path.is_file():
            return view.image_path
        return view.image_path if view.image_path.is_file() else None

    def _operator_view_paths(self, views: Sequence[str]) -> list[Path]:
        paths: list[Path] = []
        for name in views:
            path = self._operator_view_path(name)
            if path is not None and path not in paths:
                paths.append(path)
        return paths

    def _solved_mark_confirmation_source(self, view: str) -> Path | None:
        """Return the clean image underlying one just-completed mark click."""

        derived = self._derived_mark_views.get(view)
        if isinstance(derived, Mapping):
            path = Path(str(derived.get("path") or ""))
            return path if path.is_file() else None
        pointcloud_view = self._pointcloud_views.get(view)
        if pointcloud_view is not None and pointcloud_view.image_path.is_file():
            return pointcloud_view.image_path
        if view == "agentview":
            if (
                self._agentview_axis_overlay is not None
                and self._agentview_axis_overlay.is_file()
            ):
                return self._agentview_axis_overlay
        if view in {"agentview", "wrist"} and self.current_record is not None:
            if (
                view == "wrist"
                and self._wrist_grip_site_overlay is not None
                and self._wrist_grip_site_overlay.is_file()
            ):
                return self._wrist_grip_site_overlay
            frame = _find_frame(self.current_record, view)
            paths = self._frame_paths([frame] if frame is not None else [])
            return paths[0] if paths else None
        return None

    def _markable_view_image(self, view: str) -> Path | None:
        """Return the exact current image whose native pixels mark_point uses."""

        derived = self._derived_mark_views.get(view)
        if isinstance(derived, Mapping):
            path = Path(str(derived.get("path") or ""))
            if not path.is_absolute():
                path = self.root / path
            return path if path.is_file() else None
        inspection = self._inspection_views.get(view)
        if isinstance(inspection, Mapping):
            path = Path(str(inspection.get("path") or ""))
            if not path.is_absolute():
                path = self.root / path
            return path if path.is_file() else None
        return self._operator_view_path(view)

    def _invalid_mark_pixel_result(
        self,
        *,
        view: str,
        u: int,
        v: int,
        point_id: str,
    ) -> GatewayResult | None:
        """Reject, rather than silently clamp, pixels outside a returned image."""

        source = self._markable_view_image(view)
        if source is None:
            return None
        try:
            with Image.open(source) as image:
                width, height = image.size
        except OSError:
            return None
        if 0 <= int(u) < width and 0 <= int(v) < height:
            return None
        return GatewayResult(
            False,
            {
                "success": False,
                "reason": "pixel_out_of_bounds",
                "retryable": True,
                "point_id": point_id,
                "next_views": [view],
                "message": (
                    f"{view} is {width}x{height}; use "
                    f"0 <= x < {width} and 0 <= y < {height}."
                ),
            },
            images=[source],
        )

    def _render_solved_mark_confirmations(
        self,
        *,
        point_id: str,
        pixel_by_view: Mapping[str, Sequence[int]],
    ) -> tuple[list[str], list[Path]]:
        """Render only the newly solved point on its contributing source views."""

        if (
            not self._solved_point_current_only_feedback
            or self.current_record is None
        ):
            return [], []
        output_root = (
            self.root
            / "pointcloud_views"
            / str(self.current_record.get("observation_id"))
            / "solved_mark_confirmations"
        )
        output_root.mkdir(parents=True, exist_ok=True)
        safe_point_id = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", point_id
        ).strip("_") or "point"
        views: list[str] = []
        images: list[Path] = []
        for view, raw_pixel in pixel_by_view.items():
            if len(raw_pixel) != 2:
                continue
            source = self._solved_mark_confirmation_source(str(view))
            if source is None:
                continue
            try:
                image = Image.open(source).convert("RGB")
                u = max(0, min(image.width - 1, int(raw_pixel[0])))
                v = max(0, min(image.height - 1, int(raw_pixel[1])))
                if self._precise_solved_mark_feedback:
                    self._draw_precise_point_marker(image, (u, v))
                else:
                    draw = ImageDraw.Draw(image)
                    radius = 9
                    draw.ellipse(
                        (u - radius, v - radius, u + radius, v + radius),
                        outline=(0, 0, 0),
                        width=6,
                    )
                    draw.ellipse(
                        (u - radius, v - radius, u + radius, v + radius),
                        outline=(255, 32, 32),
                        width=3,
                    )
                    draw.line(
                        (u - radius - 4, v, u + radius + 4, v),
                        fill=(255, 32, 32),
                        width=3,
                    )
                    draw.line(
                        (u, v - radius - 4, u, v + radius + 4),
                        fill=(255, 32, 32),
                        width=3,
                    )
                digest = hashlib.sha256(
                    json.dumps(
                        {
                            "view": str(view),
                            "pixel_xy": [u, v],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()[:10]
                output = output_root / (
                    f"{safe_point_id}.{digest}.current-only.png"
                )
                image.save(output)
            except (OSError, TypeError, ValueError):
                continue
            views.append(str(view))
            images.append(output)
        return views, images

    @staticmethod
    def _draw_precise_point_marker(
        image: Image.Image,
        center: tuple[int, int],
    ) -> None:
        """Draw an exact one-pixel point without covering local appearance."""

        u, v = center
        draw = ImageDraw.Draw(image)
        scale = max(0.5, min(image.width, image.height) / 512.0)
        outer_radius = max(3, int(round(3 * scale)))
        inner_radius = max(2, outer_radius - 1)
        draw.ellipse(
            (
                u - outer_radius,
                v - outer_radius,
                u + outer_radius,
                v + outer_radius,
            ),
            outline=(0, 0, 0),
            width=1,
        )
        draw.ellipse(
            (
                u - inner_radius,
                v - inner_radius,
                u + inner_radius,
                v + inner_radius,
            ),
            outline=(255, 32, 32),
            width=1,
        )
        draw.point((u, v), fill=(255, 32, 32))

    def _returned_operator_view_names(
        self,
        views: Sequence[str],
    ) -> list[str]:
        return [
            name for name in views
            if self._operator_view_path(name) is not None
        ]

    def _point_mark_zoom_paths(
        self,
        rendered_views: Mapping[str, Path],
        *,
        pixel_by_view: Mapping[str, Sequence[int]],
        point_id: str,
        status: str,
        grip_site_preview_xyz_m: Sequence[float] | None = None,
    ) -> list[Path]:
        """Create deterministic, uncluttered inspection crops around marks.

        Full orthographic views are necessary for global identity and axis
        reasoning, but their dense cloud makes a several-pixel center/edge
        mistake hard to notice.  The full marked views deliberately contain
        pose arrows and labels, but those overlays can cover the local surface
        being inspected.  Crop the clean calibrated view instead and add only
        a compact crosshair.  This exposes the exact authored pixel without
        snapping it or changing the authoritative world coordinate.
        """

        zoom_paths: list[Path] = []
        safe_point_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", point_id).strip("_")
        safe_point_id = safe_point_id or "point"
        for view_name, pixel_xy in pixel_by_view.items():
            source = None
            pointcloud_view = self._pointcloud_views.get(view_name)
            if pointcloud_view is not None:
                clean_source = pointcloud_view.clean_image_path
                if clean_source is not None and clean_source.is_file():
                    source = clean_source
                elif pointcloud_view.image_path.is_file():
                    source = pointcloud_view.image_path
            elif view_name == "agentview" and self.current_record is not None:
                frame = _find_frame(self.current_record, "agentview")
                paths = self._frame_paths([frame] if frame is not None else [])
                source = paths[0] if paths else None
            if source is None:
                source = rendered_views.get(view_name)
            if source is None or not source.is_file() or len(pixel_xy) != 2:
                continue
            try:
                image = Image.open(source).convert("RGB")
                u, v = int(pixel_xy[0]), int(pixel_xy[1])
                radius = 54
                left = max(0, u - radius)
                top = max(0, v - radius)
                right = min(image.width, u + radius + 1)
                bottom = min(image.height, v + radius + 1)
                zoom = image.crop((left, top, right, bottom))
                scale = max(1, min(4, int(384 / max(1, max(zoom.size)))))
                if scale > 1:
                    zoom = zoom.resize(
                        (zoom.width * scale, zoom.height * scale),
                        Image.Resampling.NEAREST,
                    )
                draw = ImageDraw.Draw(zoom)
                cx = (u - left) * scale
                cy = (v - top) * scale
                arm = max(8, 7 * scale)
                width = max(2, scale)
                outline_width = width + 2
                for color, stroke in (("black", outline_width), ("#ff3b30", width)):
                    draw.line((cx - arm, cy, cx + arm, cy), fill=color, width=stroke)
                    draw.line((cx, cy - arm, cx, cy + arm), fill=color, width=stroke)
                    draw.ellipse(
                        (
                            cx - max(2, scale),
                            cy - max(2, scale),
                            cx + max(2, scale),
                            cy + max(2, scale),
                        ),
                        outline=color,
                        width=stroke,
                    )
                if grip_site_preview_xyz_m is not None and view_name in self._pointcloud_views:
                    center = np.asarray(
                        grip_site_preview_xyz_m, dtype=np.float64
                    ).reshape(3)
                    rotation = np.asarray(
                        self._current_grip_site_rotation_for_render(),
                        dtype=np.float64,
                    )
                    aperture_m = self._current_gripper_aperture_for_render()
                    actual_grip = np.asarray(
                        self._current_grip_site_xyz_for_render(),
                        dtype=np.float64,
                    )
                    live_pad_contacts = self._current_finger_pad_contacts_for_render()
                    live_pad_boxes = self._current_finger_pad_boxes_for_render()
                    if (
                        rotation.shape == (3, 3)
                        and np.isfinite(rotation).all()
                        and actual_grip.shape == (3,)
                        and np.isfinite(actual_grip).all()
                    ):
                        preview_color = "#ff9f0a"
                        pad_pixels: list[tuple[int, int]] = []
                        pad_points = None
                        try:
                            candidate = np.asarray(
                                live_pad_contacts, dtype=np.float64
                            )
                            if candidate.shape == (2, 3) and np.isfinite(
                                candidate
                            ).all():
                                pad_points = candidate + (center - actual_grip)
                        except (TypeError, ValueError):
                            pad_points = None
                        if pad_points is None and isinstance(
                            aperture_m, (int, float)
                        ) and math.isfinite(float(aperture_m)):
                            half_aperture = (
                                max(0.0, float(aperture_m)) / 2.0
                            )
                            pad_points = np.asarray(
                                (
                                    center - rotation[:, 0] * half_aperture,
                                    center + rotation[:, 0] * half_aperture,
                                ),
                                dtype=np.float64,
                            )
                        if pad_points is None:
                            continue
                        for pad_xyz in pad_points:
                            pad_u, pad_v = project_world_to_view(
                                self._pointcloud_views[view_name],
                                pad_xyz,
                            )
                            pad_pixels.append(
                                (
                                    int((pad_u - left) * scale),
                                    int((pad_v - top) * scale),
                                )
                            )
                        draw.line(
                            (
                                pad_pixels[0][0],
                                pad_pixels[0][1],
                                pad_pixels[1][0],
                                pad_pixels[1][1],
                            ),
                            fill=preview_color,
                            width=max(2, scale),
                        )
                        pad_radius = max(4, 3 * scale)
                        for pad_x, pad_y in pad_pixels:
                            draw.rectangle(
                                (
                                    pad_x - pad_radius,
                                    pad_y - pad_radius,
                                    pad_x + pad_radius,
                                    pad_y + pad_radius,
                                ),
                                outline=preview_color,
                                width=max(2, scale),
                            )
                        if live_pad_boxes is not None:
                            translation = center - actual_grip
                            for box in live_pad_boxes:
                                box_center = np.asarray(
                                    box["center_world_m"], dtype=np.float64
                                ) + translation
                                box_rotation = np.asarray(
                                    box["rotation_world"], dtype=np.float64
                                )
                                half_size = np.asarray(
                                    box["half_size_m"], dtype=np.float64
                                )
                                corners = []
                                for sx in (-1.0, 1.0):
                                    for sy in (-1.0, 1.0):
                                        for sz in (-1.0, 1.0):
                                            local = half_size * np.asarray(
                                                [sx, sy, sz], dtype=np.float64
                                            )
                                            corner = (
                                                box_center + box_rotation @ local
                                            )
                                            corner_u, corner_v = project_world_to_view(
                                                self._pointcloud_views[view_name],
                                                corner,
                                            )
                                            corners.append(
                                                (
                                                    int((corner_u - left) * scale),
                                                    int((corner_v - top) * scale),
                                                )
                                            )
                                hull = _convex_hull_2d(corners)
                                if len(hull) >= 3:
                                    draw.polygon(
                                        hull,
                                        outline=preview_color,
                                        width=max(2, scale),
                                    )
                        preview_caption = (
                            "orange = IF USED AS GRIP_SITE: pad collision volumes"
                        )
                        preview_box = draw.textbbox((0, 0), preview_caption)
                        preview_y = max(
                            0,
                            zoom.height
                            - int(preview_box[3] - preview_box[1] + 8),
                        )
                        draw.rectangle(
                            (0, preview_y, zoom.width, zoom.height),
                            fill=(0, 0, 0),
                        )
                        draw.text(
                            (5, preview_y + 4),
                            preview_caption,
                            fill=preview_color,
                        )
                caption = f"{view_name}  {point_id}  pixel=({u},{v})"
                box = draw.textbbox((0, 0), caption)
                caption_height = int(box[3] - box[1] + 8)
                draw.rectangle(
                    (0, 0, min(zoom.width, int(box[2] - box[0] + 10)), caption_height),
                    fill=(0, 0, 0),
                )
                draw.text((5, 4), caption, fill=(255, 255, 255))
                output = source.with_name(
                    f"{view_name}.{safe_point_id}.{status}.zoom.png"
                )
                zoom.save(output)
                zoom_paths.append(output)
            except (OSError, TypeError, ValueError):
                continue
        return zoom_paths

    def _pending_point_inspection_sheet(
        self,
        rendered_views: Mapping[str, Path],
        *,
        local_zoom_path: Path | None,
        point_id: str,
        source_view: str,
    ) -> Path | None:
        """Combine local precision and global constraints without inference.

        The pending point remains exactly the authored pixel constraint.  This
        sheet only assembles the already-rendered local crop and calibrated
        cross-view projections so their relationship is visible at once.
        """

        if local_zoom_path is None or not local_zoom_path.is_file():
            return None
        panels: list[tuple[str, Path]] = [
            ("LOCAL CLICK — revise before solving", local_zoom_path)
        ]
        for view_name in (
            "pointcloud_top",
            "pointcloud_front",
            "pointcloud_side",
        ):
            path = rendered_views.get(view_name)
            if path is not None and path.is_file():
                panels.append(
                    (view_name.removeprefix("pointcloud_").upper(), path)
                )
        if len(panels) < 2:
            return None
        try:
            panel_size = 512
            header_height = 34
            canvas = Image.new(
                "RGB",
                (panel_size * 2, (panel_size + header_height) * 2),
                (18, 18, 18),
            )
            draw = ImageDraw.Draw(canvas)
            for index, (title, path) in enumerate(panels[:4]):
                image = Image.open(path).convert("RGB")
                image.thumbnail(
                    (panel_size, panel_size),
                    Image.Resampling.NEAREST,
                )
                cell_x = (index % 2) * panel_size
                cell_y = (index // 2) * (panel_size + header_height)
                paste_x = cell_x + (panel_size - image.width) // 2
                paste_y = (
                    cell_y
                    + header_height
                    + (panel_size - image.height) // 2
                )
                canvas.paste(image, (paste_x, paste_y))
                draw.rectangle(
                    (
                        cell_x,
                        cell_y,
                        cell_x + panel_size - 1,
                        cell_y + header_height - 1,
                    ),
                    fill=(0, 0, 0),
                )
                draw.text(
                    (cell_x + 8, cell_y + 8),
                    title,
                    fill=(255, 255, 255),
                    font=ImageFont.load_default(),
                )
            safe_point_id = re.sub(
                r"[^A-Za-z0-9_.-]+", "_", point_id
            ).strip("_") or "point"
            safe_source_view = re.sub(
                r"[^A-Za-z0-9_.-]+", "_", source_view
            ).strip("_") or "view"
            output = local_zoom_path.with_name(
                f"{safe_source_view}.{safe_point_id}.pending.inspection.png"
            )
            canvas.save(output)
            return output
        except (OSError, TypeError, ValueError):
            return None

    def _commit_derived_world_mark(
        self,
        *,
        point_id: str,
        label: str,
        view: str,
        xyz_m: Sequence[float],
        source: Mapping[str, Any],
    ) -> GatewayResult:
        """Commit a world point solved by a non-world-orthographic image."""

        assert self.current_record is not None and self.writer is not None
        xyz = np.asarray(xyz_m, dtype=np.float64).reshape(3)
        if not np.isfinite(xyz).all():
            return GatewayResult(False, {
                "kind": "marked_point",
                "success": False,
                "reason": "invalid_derived_world_point",
                "view": view,
                "point_id": point_id,
            })
        self._pending_point_constraints.pop(point_id, None)
        item = {
            "point_id": point_id,
            "label": str(label),
            "view": view,
            "observation_id": self.current_record.get("observation_id"),
            "xyz_m": xyz.tolist(),
            **dict(source),
        }
        self._point_marks_3d[point_id] = item
        self._current_authored_point_id = point_id
        self._marked_pose = None
        marked_views = self._render_current_point_views()
        confirmation_pixels = source.get("confirmation_pixels_by_view")
        confirmation_views, confirmation_images = (
            self._render_solved_mark_confirmations(
                point_id=point_id,
                pixel_by_view=(
                    confirmation_pixels
                    if isinstance(confirmation_pixels, Mapping)
                    else {}
                ),
            )
        )
        artifact = self.root / "perception" / "marked_points.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            json.dumps(
                {
                    "observation_id": self.current_record.get(
                        "observation_id"
                    ),
                    "marks": self._point_marks_3d,
                    "marked_views": {
                        key: str(value)
                        for key, value in marked_views.items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self._write_current_projection()
        self.writer.record_tool_result(
            tool="mark_point",
            success=True,
            input_frames=self.current_record.get("frame_ids", []),
            artifact_refs=[
                str(artifact),
                *[str(path) for path in marked_views.values()],
                *[str(path) for path in confirmation_images],
            ],
            result=item,
        )
        text = {
            "success": True,
            "status": "solved",
            "observation_id": self.current_record.get("observation_id"),
            "point_id": point_id,
            "xyz_m": item["xyz_m"],
        }
        if confirmation_views:
            text["views"] = confirmation_views
        if not self._operator_compact_result:
            text.update(
                {
                    "kind": "marked_point",
                    "frame": "world",
                    "source_views": sorted(
                        str(name)
                        for name in (
                            source.get("view_contributions_m") or {view: {}}
                        )
                    ),
                    "measurement_semantics": str(
                        source.get("measurement_semantics")
                        or "explicit_image_world_point"
                    ),
                }
            )
        return GatewayResult(
            True,
            text,
            images=(
                confirmation_images
                if self._solved_point_current_only_feedback
                else (
                    []
                    if self._operator_source_view_only_images
                    else list(marked_views.values())
                )
            ),
            details={
                "mark": item,
                "operator_result": item,
                "point_view_paths": dict(self._point_view_paths),
                "solved_mark_confirmation_views": confirmation_views,
                "solved_mark_confirmation_paths": [
                    str(path.relative_to(self.root))
                    for path in confirmation_images
                ],
            },
        )

    def _mark_rgbd_surface_view(
        self,
        *,
        view_ref: str,
        camera_id: str,
        u: int,
        v: int,
        point_id: str,
        label: str,
    ) -> GatewayResult:
        """Solve the aligned RGB-D surface behind one returned camera image."""

        assert self.current_record is not None
        existing = self._point_marks_3d.get(point_id)
        if existing is not None:
            return GatewayResult(False, {
                "kind": "marked_point",
                "success": False,
                "reason": "point_already_solved",
                "retryable": True,
                "point_id": point_id,
                "view": view_ref,
                "message": (
                    f"{point_id} is already solved; use a new point_id to "
                    "author another image point."
                ),
            })
        pending = self._pending_point_constraints.get(point_id)
        if pending is not None:
            return GatewayResult(False, {
                "kind": "marked_point",
                "success": False,
                "reason": "pending_view_mode_conflict",
                "retryable": True,
                "point_id": point_id,
                "view": view_ref,
                "message": (
                    f"{point_id} already has a pending constraint from "
                    f"{pending.get('view')}."
                ),
            })
        try:
            ray = camera_ray_from_image_click(
                self.current_record,
                camera_id=camera_id,
                u=u,
                v=v,
                artifact_root=self.root,
                world_bounds=self._operator_pointcloud_world_bounds,
            )
            visible_surface = ray.get("visible_surface")
            if (
                not isinstance(visible_surface, Mapping)
                or not isinstance(
                    visible_surface.get("xyz_m"),
                    (list, tuple),
                )
                or len(visible_surface["xyz_m"]) != 3
            ):
                return GatewayResult(False, {
                    "kind": "marked_point",
                    "success": False,
                    "reason": "rgbd_surface_unavailable",
                    "retryable": True,
                    "point_id": point_id,
                    "view": view_ref,
                    "message": (
                        f"{view_ref} has no valid aligned depth at this pixel."
                    ),
                })
            self._ensure_operator_pointcloud_views()
            xyz = [float(value) for value in visible_surface["xyz_m"]]
            return self._commit_derived_world_mark(
                point_id=point_id,
                label=label,
                view=view_ref,
                xyz_m=xyz,
                source={
                    "status": "solved",
                    "source_kind": "rgbd_first_visible_surface",
                    "source_camera_view": camera_id,
                    "requested_pixel_xy": [int(u), int(v)],
                    "confirmation_pixels_by_view": {
                        view_ref: [int(u), int(v)],
                    },
                    "measurement_semantics": "rgbd_visible_surface_anchor",
                    "view_contributions_m": {
                        view_ref: {
                            "surface_xyz_m": xyz,
                            "pixel_xy": list(ray["pixel_xy"]),
                        }
                    },
                    "visible_surface": dict(visible_surface),
                    "camera_source_ray": {
                        "camera_id": camera_id,
                        "pixel_xy": list(ray["pixel_xy"]),
                        "origin_xyz_m": list(ray["origin_xyz_m"]),
                        "direction_xyz": list(ray["direction_xyz"]),
                    },
                },
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            return GatewayResult(False, {
                "kind": "marked_point",
                "success": False,
                "reason": "rgbd_surface_unavailable",
                "retryable": True,
                "point_id": point_id,
                "view": view_ref,
                "message": str(exc),
            })

    def _mark_candidate_frame_view(
        self,
        *,
        view_ref: str,
        metadata: Mapping[str, Any],
        u: int,
        v: int,
        point_id: str,
        label: str,
    ) -> GatewayResult:
        """Solve two clicks expressed in one preview's candidate frame."""

        assert self.current_record is not None and self.writer is not None
        if str(metadata.get("source_observation_id")) != str(
            self.current_record.get("observation_id")
        ):
            return GatewayResult(False, {
                "kind": "marked_point",
                "success": False,
                "reason": "stale_image_view",
                "retryable": True,
                "point_id": point_id,
                "view": view_ref,
                "message": "This image belongs to an older observation.",
            })
        if point_id in self._point_marks_3d:
            return GatewayResult(False, {
                "kind": "marked_point",
                "success": False,
                "reason": "point_already_solved",
                "retryable": True,
                "point_id": point_id,
                "view": view_ref,
            })
        try:
            width, height = (
                int(value) for value in metadata["image_size_xy"]
            )
            radius = float(metadata["extent_m"])
            base_view = str(metadata["base_view"])
            preview_id = str(metadata["preview_id"])
            rotation = np.asarray(
                metadata["candidate_rotation_world"],
                dtype=np.float64,
            ).reshape(3, 3)
            origin = np.asarray(
                metadata["candidate_origin_world_m"],
                dtype=np.float64,
            ).reshape(3)
            u = max(0, min(width - 1, int(u)))
            v = max(0, min(height - 1, int(v)))
            jaw = -radius + (2.0 * radius * u / max(1, width - 1))
            if base_view == "pointcloud_front":
                vertical_sign = int(metadata["vertical_screen_sign"])
                vertical = vertical_sign * (
                    radius
                    - 2.0 * radius * v / max(1, height - 1)
                )
                components = {"JAW": jaw, "APP": vertical}
                hidden_axis = "LAT"
            elif base_view == "pointcloud_side":
                lateral = (
                    radius
                    - 2.0 * radius * v / max(1, height - 1)
                )
                components = {"JAW": jaw, "LAT": lateral}
                hidden_axis = "APP"
            else:
                raise ValueError("unknown candidate-frame view")
            if (
                not np.isfinite(rotation).all()
                or not np.isfinite(origin).all()
                or not all(math.isfinite(value) for value in components.values())
            ):
                raise ValueError("candidate-frame mapping is not finite")
        except (KeyError, TypeError, ValueError) as exc:
            return GatewayResult(False, {
                "kind": "marked_point",
                "success": False,
                "reason": "invalid_image_view",
                "point_id": point_id,
                "view": view_ref,
                "message": str(exc),
            })
        click = {
            "source_kind": "candidate_frame_plane_coordinate",
            "view": view_ref,
            "base_view": base_view,
            "preview_id": preview_id,
            "pixel_xy": [u, v],
            "components_m": components,
            "hidden_axis": hidden_axis,
        }
        pending = self._pending_point_constraints.get(point_id)
        if pending is not None and (
            str(pending.get("source_kind") or "")
            != "candidate_frame_plane_coordinate"
            or str(pending.get("preview_id") or "") != preview_id
        ):
            return GatewayResult(False, {
                "kind": "marked_point",
                "success": False,
                "reason": "pending_view_mode_conflict",
                "retryable": True,
                "point_id": point_id,
                "view": view_ref,
                "message": (
                    f"{point_id} already has a pending constraint from "
                    f"{pending.get('view')}."
                ),
            })
        sibling_refs = [
            str(value) for value in metadata.get("sibling_view_refs", [])
        ]
        complementary_refs = [
            ref for ref in sibling_refs if ref != view_ref
        ]
        if pending is None or str(pending.get("view") or "") == view_ref:
            self._pending_point_constraints[point_id] = click
            self._current_authored_point_id = point_id
            images = [
                Path(self._derived_mark_views[ref]["path"])
                for ref in complementary_refs
                if ref in self._derived_mark_views
            ]
            text = {
                "success": True,
                "status": "pending",
                "observation_id": self.current_record.get("observation_id"),
                "point_id": point_id,
                "next_views": complementary_refs,
            }
            self.writer.record_tool_result(
                tool="mark_point",
                success=True,
                input_frames=self.current_record.get("frame_ids", []),
                artifact_refs=[str(path) for path in images],
                result={
                    **text,
                    "view": view_ref,
                    "pending_constraint": click,
                },
            )
            return GatewayResult(
                True,
                text,
                images=images,
                details={"pending_constraint": click},
            )
        shared_residual = abs(
            float(pending["components_m"]["JAW"]) - jaw
        )
        pixel_size_m = 2.0 * radius / max(1, width - 1)
        tolerance_m = max(0.002, 3.0 * pixel_size_m)
        if shared_residual > tolerance_m:
            retry_path = self._derived_mark_views.get(view_ref, {}).get("path")
            text = {
                "success": False,
                "reason": "inconsistent_views",
                "retryable": True,
                "observation_id": self.current_record.get("observation_id"),
                "point_id": point_id,
                "next_views": [view_ref],
                "message": (
                    "The candidate-frame clicks disagree on their shared "
                    "JAW coordinate."
                ),
            }
            self.writer.record_tool_result(
                tool="mark_point",
                success=False,
                input_frames=self.current_record.get("frame_ids", []),
                artifact_refs=[str(retry_path)] if retry_path else [],
                result={
                    **text,
                    "view": view_ref,
                    "shared_axis_residual_m": shared_residual,
                    "consistency_tolerance_m": tolerance_m,
                    "pending_constraint": pending,
                },
            )
            return GatewayResult(
                False,
                text,
                images=[Path(retry_path)] if retry_path else [],
                details={"pending_constraint": pending},
            )
        local_components = {
            **dict(pending["components_m"]),
            **components,
        }
        local = np.asarray(
            [
                local_components["JAW"],
                local_components["LAT"],
                local_components["APP"],
            ],
            dtype=np.float64,
        )
        xyz = origin + rotation @ local
        return self._commit_derived_world_mark(
            point_id=point_id,
            label=label,
            view=view_ref,
            xyz_m=xyz,
            source={
                "status": "solved",
                "source_kind": "candidate_frame_multiview_point",
                "preview_id": preview_id,
                "candidate_local_xyz_m": local.tolist(),
                "measurement_semantics":
                    "explicit_candidate_frame_multiview_world_point",
                "shared_axis": "JAW",
                "shared_axis_residual_m": shared_residual,
                "consistency_tolerance_m": tolerance_m,
                "view_contributions_m": {
                    str(pending["view"]): dict(pending["components_m"]),
                    view_ref: components,
                },
                "confirmation_pixels_by_view": {
                    str(pending["view"]): list(pending["pixel_xy"]),
                    view_ref: [int(u), int(v)],
                },
                "uncertainty_m": {
                    "JAW": pixel_size_m,
                    "LAT": pixel_size_m,
                    "APP": pixel_size_m,
                },
            },
        )

    def mark_point_3d(
        self,
        *,
        view: str,
        u: int,
        v: int,
        point_id: str = "P0",
        label: str = "",
        _skip_agentview_surface_shortcut: bool = False,
    ) -> GatewayResult:
        """Record one ordinary point using two complementary calibrated views."""

        self.ensure_started()
        assert self.current_record is not None and self.writer is not None
        point_id = str(point_id).strip()
        if not point_id:
            return GatewayResult(False, {
                "kind": "marked_point",
                "success": False,
                "reason": "invalid_point_id",
                "message": "point_id must be a non-empty string.",
            })
        if self.failure_case is not None:
            return self._failed_result("mark_point")
        derived_view = self._derived_mark_views.get(view)
        if derived_view is not None:
            if str(derived_view.get("source_observation_id")) != str(
                self.current_record.get("observation_id")
            ):
                return GatewayResult(False, {
                    "kind": "marked_point",
                    "success": False,
                    "reason": "stale_image_view",
                    "retryable": True,
                    "point_id": point_id,
                    "view": view,
                    "message": "This image belongs to an older observation.",
                })
            invalid_pixel = self._invalid_mark_pixel_result(
                view=view,
                u=u,
                v=v,
                point_id=point_id,
            )
            if invalid_pixel is not None:
                return invalid_pixel
            derived_kind = str(derived_view.get("kind") or "")
            if derived_kind == "rgbd_camera_overlay":
                return self._mark_rgbd_surface_view(
                    view_ref=view,
                    camera_id=str(derived_view["camera_id"]),
                    u=u,
                    v=v,
                    point_id=point_id,
                    label=label,
                )
            if derived_kind == "candidate_frame":
                return self._mark_candidate_frame_view(
                    view_ref=view,
                    metadata=derived_view,
                    u=u,
                    v=v,
                    point_id=point_id,
                    label=label,
                )
        if not view.startswith("inspection:"):
            invalid_pixel = self._invalid_mark_pixel_result(
                view=view,
                u=u,
                v=v,
                point_id=point_id,
            )
            if invalid_pixel is not None:
                return invalid_pixel
        if (
            self._all_returned_images_markable
            and view in self._rgbd_surface_mark_views
        ):
            return self._mark_rgbd_surface_view(
                view_ref=view,
                camera_id=view,
                u=u,
                v=v,
                point_id=point_id,
                label=label,
            )
        existing_mark = self._point_marks_3d.get(point_id)
        pending_mark = self._pending_point_constraints.get(point_id)
        if view == "agentview":
            pending_source_kind = (
                str(pending_mark.get("source_kind") or "")
                if isinstance(pending_mark, Mapping)
                else ""
            )
            if existing_mark is not None and pending_mark is None:
                return GatewayResult(False, {
                    "kind": "marked_point",
                    "success": False,
                    "reason": "point_already_solved",
                    "retryable": True,
                    "observation_id": self.current_record.get("observation_id"),
                    "point_id": point_id,
                    "view": view,
                    "message": (
                        f"{point_id} is already solved. Keep its world point; "
                        "use a new point_id to author a different agentview ray."
                    ),
                })
            if (
                pending_mark is not None
                and pending_source_kind != "agentview_camera_ray"
            ):
                return GatewayResult(False, {
                    "kind": "marked_point",
                    "success": False,
                    "reason": "pending_view_mode_conflict",
                    "retryable": True,
                    "observation_id": self.current_record.get("observation_id"),
                    "point_id": point_id,
                    "view": view,
                    "pending_view": pending_mark.get("view"),
                    "message": (
                        f"{point_id} already has a pending point-cloud "
                        f"constraint from {pending_mark.get('view')}. Complete "
                        "or revise that constraint there; use a new point_id "
                        "to start an agentview-ray workflow."
                    ),
                })
            revised_agentview_ray = (
                pending_mark is not None
                and pending_source_kind == "agentview_camera_ray"
            )
            revised_from_agentview_pixel_xy = (
                list(pending_mark.get("pixel_xy", []))
                if revised_agentview_ray
                and isinstance(pending_mark.get("pixel_xy"), list)
                else None
            )
        else:
            revised_agentview_ray = False
            revised_from_agentview_pixel_xy = None
        if view.startswith("inspection:"):
            if not self._operator_observe_inspection_enabled:
                return GatewayResult(False, {
                    "kind": "marked_point",
                    "success": False,
                    "reason": "inspection_disabled",
                    "retryable": True,
                    "observation_id": self.current_record.get("observation_id"),
                    "point_id": point_id,
                    "view": view,
                    "message": (
                        "Inspection views are disabled. Mark an individual "
                        "agentview or pointcloud_top/front/side image."
                    ),
                })
            metadata = self._inspection_views.get(view)
            if metadata is None:
                return GatewayResult(False, {
                    "kind": "marked_point",
                    "success": False,
                    "reason": "unknown_inspection_view",
                    "view": view,
                    "available_views": sorted(self._inspection_views),
                    "message": "Inspection view is unknown or belongs to an older observation.",
                })
            if str(metadata.get("source_observation_id")) != str(
                self.current_record.get("observation_id")
            ):
                return GatewayResult(False, {
                    "kind": "marked_point",
                    "success": False,
                    "reason": "stale_inspection_view",
                    "view": view,
                    "message": "Inspection view belongs to an older observation; call observe(inspect=...) again.",
                })
            invalid_pixel = self._invalid_mark_pixel_result(
                view=view,
                u=u,
                v=v,
                point_id=point_id,
            )
            if invalid_pixel is not None:
                return invalid_pixel
            try:
                display_scale = float(metadata["display_scale"])
                origin = metadata["crop_origin_xy"]
                display_width, display_height = metadata["display_size_xy"]
                if display_scale <= 0 or len(origin) != 2:
                    raise ValueError
                local_u = float(u) / display_scale
                local_v = float(v) / display_scale
                source_u = int(round(float(origin[0]) + local_u))
                source_v = int(round(float(origin[1]) + local_v))
                source_view = str(metadata["source_view"])
            except (KeyError, TypeError, ValueError, IndexError):
                return GatewayResult(False, {
                    "kind": "marked_point",
                    "success": False,
                    "reason": "invalid_inspection_view",
                    "view": view,
                    "message": "Inspection view metadata is malformed.",
                })
            result = self.mark_point_3d(
                view=source_view,
                u=source_u,
                v=source_v,
                point_id=point_id,
                label=label,
            )
            inspection_source = {
                "inspection_view": view,
                "source_view": source_view,
                "clicked_pixel_xy": [int(u), int(v)],
                "source_pixel_xy": [source_u, source_v],
                "display_scale": display_scale,
            }
            # Keep the crop-to-source mapping in host-side details only.
            # It is useful for replay/audit, but is not part of the compact
            # Operator contract and must never leak into result.text.
            result.details.setdefault("inspection_source", inspection_source)
            return result
        if (
            isinstance(pending_mark, Mapping)
            and str(pending_mark.get("source_kind") or "")
            == "candidate_frame_plane_coordinate"
        ):
            return GatewayResult(False, {
                "kind": "marked_point",
                "success": False,
                "reason": "pending_view_mode_conflict",
                "retryable": True,
                "observation_id": self.current_record.get("observation_id"),
                "point_id": point_id,
                "view": view,
                "pending_view": pending_mark.get("view"),
                "message": (
                    f"{point_id} is pending in candidate-frame image "
                    f"{pending_mark.get('view')}; use its returned "
                    "complementary image reference or a new point_id."
                ),
            })
        if view == "agentview":
            try:
                self._ensure_operator_pointcloud_views()
                pending = camera_ray_from_image_click(
                    self.current_record,
                    camera_id="agentview",
                    u=u,
                    v=v,
                    artifact_root=self.root,
                    world_bounds=self._operator_pointcloud_world_bounds,
                )
                visible_surface = pending.get("visible_surface")
                if (
                    self._agentview_first_visible_surface_shortcut
                    and not _skip_agentview_surface_shortcut
                    and isinstance(visible_surface, Mapping)
                    and isinstance(visible_surface.get("xyz_m"), (list, tuple))
                    and len(visible_surface["xyz_m"]) == 3
                    and self._pointcloud_views
                ):
                    # Reuse the ordinary calibrated point-cloud commit path
                    # with a projected pixel for the RGB-D visible surface.
                    # This keeps storage, overlays, replay, and the compact
                    # public result identical to the two-view path while the
                    # depth source remains the clicked Agentview pixel.
                    source_view = next(
                        (
                            name
                            for name in (
                                "pointcloud_front",
                                "pointcloud_top",
                                "pointcloud_side",
                            )
                            if name in self._pointcloud_views
                        ),
                        None,
                    )
                    if source_view is not None:
                        surface_xyz = np.asarray(
                            visible_surface["xyz_m"],
                            dtype=np.float64,
                        )
                        source_u, source_v = project_world_to_view(
                            self._pointcloud_views[source_view],
                            surface_xyz,
                        )
                        pending["shortcut_mode"] = (
                            "agentview_first_visible_surface"
                        )
                        self._pending_point_constraints[point_id] = pending
                        self._current_authored_point_id = point_id
                        solved = self.mark_point_3d(
                            view=source_view,
                            u=source_u,
                            v=source_v,
                            point_id=point_id,
                            label=label,
                            _skip_agentview_surface_shortcut=True,
                        )
                        if solved.success:
                            solved.details.setdefault(
                                "agentview_surface_shortcut",
                                {
                                    "source_pixel_xy": pending["pixel_xy"],
                                    "surface_xyz_m": list(
                                        visible_surface["xyz_m"]
                                    ),
                                    "source_view": source_view,
                                    "source_pixel_xy_projected": [
                                        source_u,
                                        source_v,
                                    ],
                                },
                            )
                        return solved
                self._pending_point_constraints[point_id] = pending
                self._current_authored_point_id = point_id
                pending_views = self._render_current_point_views()
                ray_segments = projected_camera_ray_segments(
                    self._pointcloud_views,
                    pending,
                )
                agentview_ray = self._render_agentview_ray_click(
                    u=u,
                    v=v,
                    point_id=point_id,
                )
                self._agentview_ray_overlay = agentview_ray
                if self._operator_source_view_only_images:
                    point_zoom_paths = []
                    inspection_sheet = None
                else:
                    point_zoom_paths = self._point_mark_zoom_paths(
                        pending_views,
                        pixel_by_view={view: [int(u), int(v)]},
                        point_id=point_id,
                        status="pending",
                    )
                    inspection_sheet = self._pending_point_inspection_sheet(
                        pending_views,
                        local_zoom_path=(
                            point_zoom_paths[0] if point_zoom_paths else None
                        ),
                        point_id=point_id,
                        source_view=view,
                    )
                self._write_current_projection()
                rich_text_full = {
                    "kind": "marked_point",
                    "success": True,
                    "status": "pending",
                    "source_kind": "agentview_camera_ray",
                    "observation_id": self.current_record.get("observation_id"),
                    "point_id": point_id,
                    "label": label,
                    "view": view,
                    "clicked_pixel_xy": [int(u), int(v)],
                    "needs_complementary_view": sorted(self._pointcloud_views),
                    "projected_ray_segments": ray_segments,
                    **(
                        {
                            "revised_from_pixel_xy": (
                                revised_from_agentview_pixel_xy
                            ),
                            "revised_to_pixel_xy": [int(u), int(v)],
                        }
                        if revised_agentview_ray
                        else {}
                    ),
                    "message": (
                        f"{point_id} semantic ray "
                        + (
                            "was revised from the previous agentview click. "
                            if revised_agentview_ray
                            else "was recorded from agentview. "
                        )
                        + (
                            "The projected ray is shown in top/front/side. When "
                            "aligned Agentview depth is available, the cyan ring "
                            "on that ray is the same visible RGB-D surface. Mark "
                            f"the same {point_id} near that ring in one "
                            "point-cloud view to confirm and solve the visible "
                            "feature. "
                            if self._agentview_visible_surface_confirmation
                            else
                            "The projected ray is shown in top/front/side. Mark "
                            f"the same {point_id} on that ray in one point-cloud "
                            "view to choose depth and solve the world point. "
                        )
                        + "This ray remains bound to the clicked surface "
                        "feature; use a new point_id in two complementary "
                        "point-cloud views for a hidden or free-space grip-site."
                    ),
                }
                text = {
                    "success": True,
                    "status": "pending",
                    "observation_id": self.current_record.get("observation_id"),
                    "point_id": point_id,
                    "next_views": [
                        "pointcloud_top",
                        "pointcloud_front",
                        "pointcloud_side",
                    ],
                    **(
                        {
                            "revised_from_pixel_xy": (
                                revised_from_agentview_pixel_xy
                            )
                        }
                        if revised_agentview_ray
                        else {}
                    ),
                }
                if not self._operator_compact_result:
                    text.update(
                        {
                            "kind": "marked_point",
                            "source_view": "agentview",
                            "measurement_semantics":
                                "agentview_visible_feature_ray",
                            "ray_rendered": True,
                            "clickable_views": [
                                {
                                    "image_number": image_number,
                                    "view": source_view,
                                    "pixel_size": [384, 384],
                                    "coordinate_system":
                                        "source_pixels_zero_based",
                                }
                                for image_number, source_view in enumerate(
                                    (
                                        "pointcloud_top",
                                        "pointcloud_front",
                                        "pointcloud_side",
                                    ),
                                    start=1,
                                )
                            ],
                        }
                    )
                self.writer.record_tool_result(
                    tool="mark_point",
                    success=True,
                    input_frames=self.current_record.get("frame_ids", []),
                    artifact_refs=[
                        *(
                            [str(agentview_ray)]
                            if agentview_ray is not None
                            else []
                        ),
                        *[str(path) for path in pending_views.values()],
                        *[str(path) for path in point_zoom_paths],
                        *(
                            [str(inspection_sheet)]
                            if inspection_sheet is not None
                            else []
                        ),
                    ],
                    result={**rich_text_full, "pending_constraint": pending},
                )
                return GatewayResult(
                    True,
                    text,
                    images=[
                        pending_views[source_view]
                        for source_view in (
                            "pointcloud_top",
                            "pointcloud_front",
                            "pointcloud_side",
                        )
                    ],
                    details={
                        "pending_constraint": pending,
                        "projected_ray_segments": ray_segments,
                        "operator_result": rich_text_full,
                        "point_view_paths": dict(self._point_view_paths),
                        "point_zoom_paths": [
                            str(path.relative_to(self.root))
                            for path in point_zoom_paths
                        ],
                        "inspection_sheet_path": (
                            str(inspection_sheet.relative_to(self.root))
                            if inspection_sheet is not None
                            else None
                        ),
                    },
                )
            except (
                GraspPoseRefinementError,
                KeyError,
                OSError,
                TypeError,
                ValueError,
            ) as exc:
                return GatewayResult(False, {
                    "kind": "marked_point",
                    "success": False,
                    "reason": "agentview_ray_unavailable",
                    "view": view,
                    "message": str(exc),
                })
        # A preceding move_to may intentionally return RGB-only feedback. That
        # invalidates the old observation-bound orthographic cache, but it
        # should not force the caller to discover this implementation detail
        # through unknown_pointcloud_view and repeat the same click after an
        # explicit observe. Materialize geometry lazily for the current
        # observation when mark_point itself requests a calibrated view.
        if view.startswith("pointcloud_") and view not in self._pointcloud_views:
            self._ensure_operator_pointcloud_views()
            invalid_pixel = self._invalid_mark_pixel_result(
                view=view,
                u=u,
                v=v,
                point_id=point_id,
            )
            if invalid_pixel is not None:
                return invalid_pixel
        if view not in self._pointcloud_views:
            return GatewayResult(False, {
                "kind": "marked_point",
                "success": False,
                "reason": "unknown_pointcloud_view",
                "view": view,
                "available_views": ["agentview", *sorted(self._pointcloud_views)],
            })
        existing = self._point_marks_3d.get(point_id)
        if existing is not None and point_id not in self._pending_point_constraints:
            try:
                solved_xyz = np.asarray(existing["xyz_m"], dtype=np.float64)
                solved_u, solved_v = project_world_to_view(
                    self._pointcloud_views[view],
                    solved_xyz,
                )
                view_spec = self._pointcloud_views[view].spec
                h0, h1 = (
                    float(value)
                    for value in view_spec["horizontal_range_m"]
                )
                v0, v1 = (
                    float(value)
                    for value in view_spec["vertical_range_m"]
                )
                h_error_m = (
                    abs(int(u) - solved_u)
                    * (h1 - h0)
                    / max(1, self._pointcloud_views[view].width - 1)
                )
                v_error_m = (
                    abs(int(v) - solved_v)
                    * (v1 - v0)
                    / max(1, self._pointcloud_views[view].height - 1)
                )
                tolerance_m = 0.015
                if max(h_error_m, v_error_m) > tolerance_m:
                    return GatewayResult(False, {
                        "kind": "marked_point",
                        "success": False,
                        "reason": "inconsistent_with_solved_point",
                        "retryable": True,
                        "observation_id": self.current_record.get("observation_id"),
                        "point_id": point_id,
                        "label": label,
                        "view": view,
                        "solved_projection_pixel_xy": [solved_u, solved_v],
                        "clicked_pixel_xy": [int(u), int(v)],
                        "display_axis_error_m": [h_error_m, v_error_m],
                        "consistency_tolerance_m": tolerance_m,
                        "message": (
                            f"{point_id} is already solved and this extra {view} "
                            "click disagrees with its projection. Keep the solved "
                            "point, or use a new point_id to define another point."
                        ),
                    })
                self._current_authored_point_id = point_id
                marked_views = self._render_current_point_views(
                    force_current_agentview_audit=(
                        self._solved_point_agentview_audit_return
                    )
                )
                point_zoom_paths = (
                    []
                    if self._operator_source_view_only_images
                    else self._point_mark_zoom_paths(
                        marked_views,
                        pixel_by_view={
                            name: project_world_to_view(
                                point_view,
                                solved_xyz,
                            )
                            for name, point_view in self._pointcloud_views.items()
                        },
                        point_id=point_id,
                        status="verified",
                        grip_site_preview_xyz_m=solved_xyz,
                    )
                )
                self._write_current_projection()
                rich_text_full = {
                    "kind": "marked_point",
                    "success": True,
                    "status": "verified",
                    "observation_id": self.current_record.get("observation_id"),
                    "point_id": point_id,
                    "label": label,
                    "view": view,
                    "frame": "world",
                    "xyz_m": existing["xyz_m"],
                    "solved_projection_pixel_xy": [solved_u, solved_v],
                    "clicked_pixel_xy": [int(u), int(v)],
                    "display_axis_error_m": [h_error_m, v_error_m],
                    "consistency_tolerance_m": tolerance_m,
                    "grip_site_footprint_preview": {
                        "shown_in_zoom_images": True,
                        "color": "orange",
                        "semantics": (
                            "Hypothetical current open-pad centers if this ordinary "
                            "point were used as the Panda grip_site. Diagnostic only; "
                            "it does not change the point or validate a grasp."
                        ),
                    },
                    **(
                        {
                            "grip_site_footprint_world": (
                                self._grip_site_footprint_for_mark(
                                    existing["xyz_m"]
                                )
                            )
                        }
                        if self._mark_point_numeric_footprint
                        else {}
                    ),
                    "message": (
                        f"{point_id} was already solved. The extra {view} click "
                        "is consistent, so the authoritative world point is "
                        "unchanged and no new pending constraint was created."
                    ),
                }
                text = {
                    "success": True,
                    "status": "verified",
                    "observation_id": self.current_record.get("observation_id"),
                    "point_id": point_id,
                    "xyz_m": existing["xyz_m"],
                }
                if not self._operator_compact_result:
                    text.update(
                        {
                            "kind": "marked_point",
                            "frame": "world",
                            "source_view": view,
                        }
                    )
                self.writer.record_tool_result(
                    tool="mark_point",
                    success=True,
                    input_frames=self.current_record.get("frame_ids", []),
                    artifact_refs=[
                        *[str(path) for path in marked_views.values()],
                        *[str(path) for path in point_zoom_paths],
                    ],
                    result=rich_text_full,
                )
                return GatewayResult(
                    True,
                    text,
                    images=(
                        []
                        if self._operator_source_view_only_images
                        else [*point_zoom_paths, *marked_views.values()]
                    ),
                    details={
                        "mark": existing,
                        "operator_result": rich_text_full,
                        "point_view_paths": dict(self._point_view_paths),
                        "point_zoom_paths": [
                            str(path.relative_to(self.root))
                            for path in point_zoom_paths
                        ],
                    },
                )
            except (KeyError, TypeError, ValueError):
                # Let the normal two-view path below report malformed state.
                pass
        try:
            pending_before = self._pending_point_constraints.get(point_id)
            revised_pending_click = bool(
                isinstance(pending_before, Mapping)
                and str(pending_before.get("source_kind") or "")
                != "agentview_camera_ray"
                and str(pending_before.get("view") or "") == view
            )
            revised_from_pixel_xy = (
                list(pending_before.get("pixel_xy", []))
                if revised_pending_click
                and isinstance(pending_before.get("pixel_xy"), list)
                else None
            )
            xyz, source = mark_world_point(
                self._pointcloud_views[view],
                u=u,
                v=v,
                # A second click in the same orthographic view revises the
                # still-pending two-axis constraint. This is an explicit
                # author correction after inspecting the returned zoom, not
                # snapping or depth inference. Camera rays remain immutable
                # semantic bindings and still require a point-cloud view.
                pending_constraint=(
                    None
                    if revised_pending_click
                    else pending_before
                ),
                enforce_visible_surface_layer=(
                    self._agentview_visible_surface_confirmation
                ),
            )
            status = str(source.get("status") or "solved")
            if status != "solved":
                pending_views: dict[str, Path] = {}
                pending = source.get("pending_constraint")
                if isinstance(pending, Mapping):
                    # A complementary click is an observation used to solve
                    # the original constraint.  If it disagrees, reject that
                    # latest click and keep the original anchor.  Only a
                    # second click in the same source view is an explicit
                    # revision of the pending constraint.
                    if (
                        status == "inconsistent_views"
                        and isinstance(pending_before, Mapping)
                        and str(
                            pending_before.get("source_kind") or ""
                        ) != "agentview_camera_ray"
                        and str(pending_before.get("view") or "") != view
                    ):
                        self._pending_point_constraints[point_id] = dict(
                            pending_before
                        )
                    else:
                        self._pending_point_constraints[point_id] = dict(pending)
                else:
                    self._pending_point_constraints.pop(point_id, None)
                if point_id in self._pending_point_constraints:
                    self._current_authored_point_id = point_id
                    pending_views = self._render_current_point_views()
                    if (
                        self._rejected_mark_current_click_feedback
                        and status in {
                            "inconsistent_views",
                            "different_visible_depth_layer",
                            "outside_ray_workspace",
                        }
                        and view in pending_views
                    ):
                        pending_views[view] = annotate_rejected_point_click(
                            pending_views[view],
                            output_root=(
                                self.root
                                / "pointcloud_views"
                                / str(
                                    self.current_record.get(
                                        "observation_id",
                                        "unknown",
                                    )
                                )
                                / "rejected_mark_feedback"
                            ),
                            point_id=point_id,
                            view=view,
                            pixel_xy=[int(u), int(v)],
                        )
                current_pending = self._pending_point_constraints.get(point_id)
                if self._operator_source_view_only_images:
                    point_zoom_paths = []
                    inspection_sheet = None
                    remaining_view_names, remaining_view_images = (
                        self._remaining_point_mark_images(
                            pending_views,
                            current_pending,
                        )
                    )
                else:
                    point_zoom_paths = self._point_mark_zoom_paths(
                        pending_views,
                        pixel_by_view={view: [int(u), int(v)]},
                        point_id=point_id,
                        status="pending",
                    )
                    inspection_sheet = self._pending_point_inspection_sheet(
                        pending_views,
                        local_zoom_path=(
                            point_zoom_paths[0] if point_zoom_paths else None
                        ),
                        point_id=point_id,
                        source_view=view,
                    )
                    remaining_view_names = list(
                        source.get("needs_complementary_view", [])
                    )
                    remaining_view_images = (
                        [inspection_sheet]
                        if inspection_sheet is not None
                        else point_zoom_paths
                    )
                self._write_current_projection()
                if status == "pending":
                    revision_fields = (
                        {
                            "revised": True,
                            "revised_from_pixel_xy": revised_from_pixel_xy,
                            "revised_to_pixel_xy": [int(u), int(v)],
                        }
                        if revised_pending_click
                        else {}
                    )
                    rich_text_full = {
                        "kind": "marked_point",
                        "success": True,
                        "status": "pending",
                        "observation_id": self.current_record.get("observation_id"),
                        "point_id": point_id,
                        "label": label,
                        "view": view,
                        "needs_complementary_view": source.get("needs_complementary_view", []),
                        **revision_fields,
                        "message": (
                            (
                                f"{point_id} pending {view} constraint revised "
                                f"from pixel {revised_from_pixel_xy} to "
                                f"[{int(u)}, {int(v)}]. "
                                if revised_pending_click
                                else f"{point_id} constraint recorded from {view}. "
                            )
                            + (
                                "Use one returned individual source view to "
                                f"mark the same {point_id} and solve the 3D "
                                "position."
                                if self._operator_source_view_only_images
                                else
                                "Inspect the returned local zoom; you may revise "
                                f"the same pending {view} click before solving. "
                                f"Mark the same {point_id} once in a "
                                "complementary view to solve the 3D position."
                            )
                        ),
                    }
                    text = {
                        "success": True,
                        "status": "pending",
                        "observation_id": self.current_record.get("observation_id"),
                        "point_id": point_id,
                        "next_views": remaining_view_names,
                    }
                    if not self._operator_compact_result:
                        text.update(
                            {
                                "kind": "marked_point",
                                "source_view": view,
                            }
                        )
                    self.writer.record_tool_result(
                        tool="mark_point",
                        success=True,
                        input_frames=self.current_record.get("frame_ids", []),
                        artifact_refs=[
                            *[str(path) for path in pending_views.values()],
                            *[str(path) for path in point_zoom_paths],
                            *(
                                [str(inspection_sheet)]
                                if inspection_sheet is not None
                                else []
                            ),
                        ],
                        result={
                            **rich_text_full,
                            "pending_constraint": self._pending_point_constraints.get(
                                point_id
                            ),
                        },
                    )
                    return GatewayResult(
                        True,
                        text,
                        images=remaining_view_images,
                        details={
                            "pending_constraint": self._pending_point_constraints.get(
                                point_id
                            ),
                            "operator_result": rich_text_full,
                            "point_view_paths": dict(self._point_view_paths),
                            "point_zoom_paths": [
                                str(path.relative_to(self.root))
                                for path in point_zoom_paths
                            ],
                            "inspection_sheet_path": (
                                str(inspection_sheet.relative_to(self.root))
                                if inspection_sheet is not None
                                else None
                            ),
                        },
                    )
                hints = {
                    "complementary_view_required": (
                        f"{point_id} already has a pending click in {view}. "
                        "Mark it in a complementary view "
                        f"({', '.join(source.get('needs_complementary_view', []))}) instead."
                    ),
                    "inconsistent_views": (
                        "The two views disagree on the shared "
                        f"{source.get('shared_axis')} axis by "
                        f"{float(source.get('shared_axis_residual_m') or 0.0) * 1000:.1f} mm "
                        f"(tolerance {float(source.get('consistency_tolerance_m') or 0.0) * 1000:.1f} mm). "
                        + (
                            "The original agentview semantic ray is unchanged; "
                            "click on that rendered ray in a point-cloud view to retry."
                            if str(
                                self._pending_point_constraints.get(
                                    point_id, {}
                                ).get("source_kind")
                                or ""
                            )
                            == "agentview_camera_ray"
                            else
                            "The latest complementary click was rejected; "
                            "the original pending click is kept. Click the same "
                            "world point again in a complementary view to retry."
                        )
                        + (
                            f" The visible Agentview surface projects to "
                            f"{source.get('visible_surface_ray_pixel_xy')} in "
                            f"{view}; use the cyan surface marker there."
                            if isinstance(
                                source.get("visible_surface_ray_pixel_xy"),
                                list,
                            )
                            else ""
                        )
                    ),
                    "outside_ray_workspace": (
                        "The click is outside the rendered workspace segment of "
                        "the Agentview ray. "
                        + (
                            f"The visible Agentview surface projects to "
                            f"{source.get('visible_surface_ray_pixel_xy')} in "
                            f"{view}; use the cyan surface marker there."
                            if isinstance(
                                source.get("visible_surface_ray_pixel_xy"),
                                list,
                            )
                            else "Choose a point on the colored workspace ray."
                        )
                    ),
                    "different_visible_depth_layer": (
                        "The selected point is "
                        f"{abs(float(source.get('visible_surface_delta_m') or 0.0)) * 1000:.1f} mm "
                        + (
                            "behind"
                            if float(source.get("visible_surface_delta_m") or 0.0) > 0
                            else "in front of"
                        )
                        + " the surface visible at the clicked agentview pixel. "
                        "That would bind the semantic mark to another occlusion "
                        "layer. The original ray is retained. "
                        + (
                            f"Use the cyan visible-surface marker at "
                            f"{source.get('visible_surface_ray_pixel_xy')} in "
                            f"{view}, or use a separate two-view point_id for an "
                            "intentionally hidden execution point."
                            if isinstance(
                                source.get("visible_surface_ray_pixel_xy"),
                                list,
                            )
                            else
                            "Choose the visible surface depth on the ray, or use "
                            "a separate two-view point_id for an intentionally "
                            "hidden execution point."
                        )
                    ),
                }
                rich_text_full = {
                    "kind": "marked_point",
                    "success": False,
                    "reason": status,
                    "retryable": True,
                    "observation_id": self.current_record.get("observation_id"),
                    "point_id": point_id,
                    "label": label,
                    "view": view,
                    "shared_axis": source.get("shared_axis"),
                    "shared_axis_residual_m": source.get("shared_axis_residual_m"),
                    "consistency_tolerance_m": source.get("consistency_tolerance_m"),
                    **(
                        {"nearest_ray_pixel_xy": source["nearest_ray_pixel_xy"]}
                        if isinstance(source.get("nearest_ray_pixel_xy"), list)
                        else {}
                    ),
                    **(
                        {
                            "visible_surface_ray_pixel_xy":
                                source["visible_surface_ray_pixel_xy"]
                        }
                        if isinstance(
                            source.get("visible_surface_ray_pixel_xy"),
                            list,
                        )
                        else {}
                    ),
                    "message": hints.get(status, status),
                }
                text = {
                    "success": False,
                    "reason": status,
                    "retryable": True,
                    "observation_id": self.current_record.get("observation_id"),
                    "point_id": point_id,
                    "source_view": view,
                    "message": {
                        "complementary_view_required":
                            "Mark the same point_id in a complementary view.",
                        "inconsistent_views":
                            "The two clicks disagree; choose the same 3D feature in a complementary view.",
                        "different_visible_depth_layer":
                            "This is another depth layer. Choose the cyan visible-surface marker, or use a new point_id.",
                        "outside_ray_workspace":
                            "Choose the cyan visible-surface marker on the rendered workspace ray.",
                    }.get(status, "Retry the mark on a complementary view."),
                }
                suggested_pixel = None
                if isinstance(
                    source.get("visible_surface_ray_pixel_xy"),
                    list,
                ):
                    suggested_pixel = source["visible_surface_ray_pixel_xy"]
                    text["message"] = (
                        f"{text['message']} In {view}, that surface projects to "
                        f"{source['visible_surface_ray_pixel_xy']}."
                    )
                elif (
                    status == "inconsistent_views"
                    and isinstance(source.get("nearest_ray_pixel_xy"), list)
                ):
                    suggested_pixel = source["nearest_ray_pixel_xy"]
                    text["message"] = (
                        f"{text['message']} In {view}, the nearest pixel on the "
                        f"rendered ray is {source['nearest_ray_pixel_xy']}."
                    )
                # A rejected click is a correction in the view just authored,
                # not a request to resend every pending image.  In source-view
                # mode, keep the authored ray/marker legible by returning only
                # that view.  A fresh agentview ray still returns all three
                # views on the initial pending result; this narrowing applies
                # only after a concrete complementary click was rejected.
                if (
                    self._operator_mark_point_failure_feedback
                    == "single_correction_source_view_with_optional_suggested_pixel"
                    and
                    self._operator_source_view_only_images
                    and status in {
                        "inconsistent_views",
                        "different_visible_depth_layer",
                        "outside_ray_workspace",
                    }
                    and view in pending_views
                ):
                    remaining_view_names = [view]
                    remaining_view_images = [pending_views[view]]
                if (
                    self._operator_mark_point_failure_feedback
                    == "single_correction_source_view_with_optional_suggested_pixel"
                    and self._operator_source_view_only_images
                    and remaining_view_names
                ):
                    text["message"] = (
                        f"{text['message']} Retry in one returned individual "
                        f"source view: {', '.join(remaining_view_names)}."
                    )
                if self._operator_compact_result:
                    if (
                        self._operator_mark_point_failure_feedback
                        == "single_correction_source_view_with_optional_suggested_pixel"
                    ):
                        text["next_views"] = remaining_view_names
                    if (
                        self._operator_mark_point_failure_feedback
                        == "single_correction_source_view_with_optional_suggested_pixel"
                        and suggested_pixel is not None
                    ):
                        text["suggested_pixel_xy"] = [
                            int(value) for value in suggested_pixel
                        ]
                        text["message"] = (
                            f"Retry {view} at pixel "
                            f"{[int(value) for value in suggested_pixel]}."
                        )
                    elif (
                        self._operator_mark_point_failure_feedback
                        == "single_correction_source_view_with_optional_suggested_pixel"
                        and
                        status in {
                            "inconsistent_views",
                            "different_visible_depth_layer",
                            "outside_ray_workspace",
                        }
                        and remaining_view_names
                    ):
                        text["message"] = (
                            f"Retry in {remaining_view_names[0]}."
                        )
                else:
                    text["kind"] = "marked_point"
                self.writer.record_tool_result(
                    tool="mark_point",
                    success=False,
                    input_frames=self.current_record.get("frame_ids", []),
                    artifact_refs=[str(path) for path in pending_views.values()],
                    result={
                        **rich_text_full,
                        "pending_constraint": self._pending_point_constraints.get(
                            point_id
                        ),
                    },
                )
                return GatewayResult(
                    False,
                    text,
                    images=(
                        remaining_view_images
                        if self._operator_source_view_only_images
                        else (
                            [inspection_sheet]
                            if inspection_sheet is not None
                            else list(pending_views.values())
                        )
                    ),
                    details={
                        "pending_constraint": self._pending_point_constraints.get(
                            point_id
                        ),
                        "point_view_paths": dict(self._point_view_paths),
                        "operator_result": rich_text_full,
                    },
                )
            confirmation_pixels: dict[str, list[int]] = {
                str(view): [int(u), int(v)],
            }
            if (
                isinstance(pending_before, Mapping)
                and isinstance(pending_before.get("pixel_xy"), list)
                and len(pending_before["pixel_xy"]) == 2
            ):
                confirmation_pixels[str(pending_before.get("view") or view)] = [
                    int(value) for value in pending_before["pixel_xy"]
                ]
            self._pending_point_constraints.pop(point_id, None)
            assert xyz is not None
            source_kind = str(source.get("source_kind") or "")
            authored_view = (
                "agentview"
                if source_kind == "agentview_first_visible_surface"
                else view
            )
            item = {
                "point_id": point_id,
                "label": str(label),
                "view": authored_view,
                "observation_id": self.current_record.get("observation_id"),
                "xyz_m": [float(value) for value in xyz],
                **source,
            }
            self._point_marks_3d[point_id] = item
            self._current_authored_point_id = point_id
            self._marked_pose = None
            marked_views = self._render_current_point_views(
                force_current_agentview_audit=(
                    self._solved_point_agentview_audit_return
                )
            )
            confirmation_views, confirmation_images = (
                self._render_solved_mark_confirmations(
                    point_id=point_id,
                    pixel_by_view=confirmation_pixels,
                )
            )
            point_zoom_paths = (
                []
                if self._operator_source_view_only_images
                else self._point_mark_zoom_paths(
                    marked_views,
                    pixel_by_view={
                        name: project_world_to_view(point_view, xyz)
                        for name, point_view in self._pointcloud_views.items()
                    },
                    point_id=point_id,
                    status="solved",
                    grip_site_preview_xyz_m=xyz,
                )
            )
            artifact = self.root / "perception" / "marked_points.json"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(json.dumps({
                "observation_id": self.current_record.get("observation_id"),
                "marks": self._point_marks_3d,
                "marked_views": {key: str(value) for key, value in marked_views.items()},
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self._write_current_projection()
            self.writer.record_tool_result(
                tool="mark_point",
                success=True,
                input_frames=self.current_record.get("frame_ids", []),
                artifact_refs=[
                    str(artifact),
                    *[str(path) for path in marked_views.values()],
                    *[str(path) for path in point_zoom_paths],
                    *[str(path) for path in confirmation_images],
                ],
                result=item,
            )
            source_view_names = set(
                str(name)
                for name in (source.get("view_contributions_m") or {})
            )
            if not source_view_names:
                source_view_names = {
                    str(self._pending_point_constraints.get(point_id, {}).get("view") or view)
                }
            operator_zoom_paths = [
                path
                for path in point_zoom_paths
                if path.name.split(".", 1)[0] in source_view_names
            ]
            if not operator_zoom_paths:
                operator_zoom_paths = point_zoom_paths[:2]
            operator_audit_paths = (
                [marked_views["agentview"]]
                if self._solved_point_agentview_audit_return
                and "agentview" in marked_views
                else []
            )
            source_view_names = set(
                str(name)
                for name in (source.get("view_contributions_m") or {})
            )
            if not source_view_names:
                source_view_names = {str(view)}
            measurement_semantics = (
                "agentview_visible_feature_anchor"
                if source_kind in {
                    "agentview_ray_orthographic_intersection",
                    "agentview_first_visible_surface",
                }
                else "explicit_multiview_world_point"
            )
            solved_text = {
                "success": True,
                "status": "solved",
                "observation_id": self.current_record.get("observation_id"),
                "point_id": point_id,
                "xyz_m": item["xyz_m"],
            }
            if confirmation_views:
                solved_text["views"] = confirmation_views
            if not self._operator_compact_result:
                solved_text.update(
                    {
                        "kind": "marked_point",
                        "frame": "world",
                        "source_views": sorted(source_view_names),
                        "measurement_semantics": measurement_semantics,
                    }
                )
            return GatewayResult(True, solved_text, images=(
                confirmation_images
                if self._solved_point_current_only_feedback
                else (
                    []
                    if self._operator_source_view_only_images
                    else [*operator_zoom_paths, *operator_audit_paths]
                )
            ), details={
                "mark": item,
                "operator_result": {
                    "kind": "marked_point",
                    "success": True,
                    "status": "solved",
                    "observation_id": self.current_record.get("observation_id"),
                    "point_id": point_id,
                    "label": label,
                    "view": authored_view,
                    "frame": "world",
                    "xyz_m": item["xyz_m"],
                    "shared_axis_residual_m": item.get("shared_axis_residual_m"),
                    "uncertainty_m": item.get("uncertainty_m"),
                    "solved_point_ids": sorted(self._point_marks_3d),
                    "grip_site_footprint_preview": {
                        "shown_in_zoom_images": True,
                        "color": "orange",
                        "semantics": (
                            "Hypothetical current open-pad centers if this ordinary "
                            "point were used as the Panda grip_site. Diagnostic only."
                        ),
                    },
                },
                "point_zoom_paths": [
                    str(path.relative_to(self.root))
                    for path in point_zoom_paths
                ],
                "operator_returned_point_zoom_paths": [
                    str(path.relative_to(self.root))
                    for path in operator_zoom_paths
                ],
                "operator_returned_agentview_audit_path": (
                    str(operator_audit_paths[0].relative_to(self.root))
                    if operator_audit_paths
                    else None
                ),
                "solved_mark_confirmation_views": confirmation_views,
                "solved_mark_confirmation_paths": [
                    str(path.relative_to(self.root))
                    for path in confirmation_images
                ],
            })
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return GatewayResult(False, {
                "kind": "marked_point",
                "success": False,
                "reason": "invalid_3d_mark",
                "view": view,
                "point_id": point_id,
                "label": label,
                "message": str(exc),
            })

    def mark_vector_3d(
        self,
        *,
        view: str,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        role: str = "approach",
    ) -> GatewayResult:
        """Record one 2D vector projection and merge complementary views."""

        self.ensure_started()
        assert self.current_record is not None and self.writer is not None
        if self.failure_case is not None:
            return self._failed_result("mark_vector")
        if view not in self._pointcloud_views:
            return GatewayResult(False, {
                "kind": "marked_vector",
                "success": False,
                "reason": "unknown_pointcloud_view",
                "view": view,
                "available_views": sorted(self._pointcloud_views),
            })
        try:
            projection = world_vector_from_view_delta(
                self._pointcloud_views[view],
                start_xy=(int(start_x), int(start_y)),
                end_xy=(int(end_x), int(end_y)),
            )
            if projection["length_m"] < 0.01:
                raise ValueError("marked vector is shorter than 1 cm")
            role_projections = self._point_vectors_3d.setdefault(role, {})
            role_projections[view] = projection
            merged = merge_world_vector_projections(role_projections)
            self._marked_vectors_3d[role] = {
                "role": role,
                "observation_id": self.current_record.get("observation_id"),
                "projections": role_projections,
                "vector": merged,
            }
            vector_views = annotate_vector_views(
                self._pointcloud_views,
                self._point_vectors_3d,
                output_root=self.root / "pointcloud_views" / str(self.current_record.get("observation_id")),
            )
            self._vector_view_paths = {
                key: str(path.relative_to(self.root))
                for key, path in vector_views.items()
            }
            artifact = self.root / "perception" / "marked_vectors.json"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(json.dumps({
                "observation_id": self.current_record.get("observation_id"),
                "vectors": self._marked_vectors_3d,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self._write_current_projection()
            self.writer.record_tool_result(
                tool="mark_vector",
                success=True,
                input_frames=self.current_record.get("frame_ids", []),
                artifact_refs=[str(artifact), *[str(path) for path in vector_views.values()]],
                result=self._marked_vectors_3d[role],
            )
            complete = merged is not None
            return GatewayResult(True, {
                "kind": "marked_vector",
                "success": True,
                "observation_id": self.current_record.get("observation_id"),
                "role": role,
                "view": view,
                "projection": projection,
                "marked_vector": self._marked_vectors_3d[role],
                "vector_view_paths": self._vector_view_paths,
                "complete_3d": complete,
                "message": (
                    "Two complementary view projections merged into a world-frame 3D vector."
                    if complete
                    else "2D vector recorded; mark the same role in a complementary view to complete the 3D vector."
                ),
            }, images=list(vector_views.values()), details={"projection": projection, "marked_vector": self._marked_vectors_3d[role]})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return GatewayResult(False, {
                "kind": "marked_vector",
                "success": False,
                "reason": "invalid_vector_mark",
                "view": view,
                "role": role,
                "message": str(exc),
            })

    # ------------------------------------------------------------------
    # Perception and visual selection
    # ------------------------------------------------------------------

    def python_mark_object(self, code: str) -> GatewayResult:
        """Run a visual-only Python marking snippet on the current agentview.

        The snippet receives the current RGB image and two small helpers:
        ``mark_box`` and ``mark_point``.  It can inspect pixels or use any
        local Python image logic, but the gateway only carries the resulting
        marked image into the next explicit ``segment_object`` call.
        """

        self.ensure_started()
        assert self.current_record is not None and self.writer is not None
        if self.failure_case is not None:
            return self._failed_result("python_mark_object")
        call = self.writer.record_tool_start(
            tool="python_mark_object",
            input_frames=self.current_record.get("frame_ids", []),
        )
        frame = _find_frame(self.current_record, "agentview")
        if frame is None:
            message = "Current observation has no agentview image."
            self.writer.record_tool_result(
                tool="python_mark_object",
                success=False,
                tool_call_id=call.get("tool_call_id"),
                input_frames=self.current_record.get("frame_ids", []),
                result={"reason": "missing_agentview"},
            )
            return GatewayResult(False, {"kind": "python_mark_object", "success": False, "error": message})
        source = _frame_rgb_path(self.root, frame)
        if source is None:
            message = "Current observation has no readable agentview image."
            self.writer.record_tool_result(
                tool="python_mark_object",
                success=False,
                tool_call_id=call.get("tool_call_id"),
                input_frames=self.current_record.get("frame_ids", []),
                result={"reason": "missing_rgb"},
            )
            return GatewayResult(False, {"kind": "python_mark_object", "success": False, "error": message})

        try:
            from PIL import Image, ImageColor, ImageDraw

            image = Image.open(source).convert("RGB")
            marks: list[dict[str, Any]] = []

            def _coords(values: Any, *, normalized: bool) -> tuple[float, ...]:
                if not isinstance(values, (list, tuple)):
                    raise ValueError("mark coordinates must be a list or tuple")
                numbers = tuple(float(value) for value in values)
                if normalized:
                    numbers = tuple(
                        number * (image.width if index % 2 == 0 else image.height)
                        for index, number in enumerate(numbers)
                    )
                return numbers

            def mark_box(
                box_or_x0: Any,
                y0: Any = None,
                x1: Any = None,
                y1: Any = None,
                label: str = "object",
                normalized: bool = False,
                color: Any = "red",
                width: int | None = None,
                **_style: Any,
            ) -> dict[str, Any]:
                # Accept both intuitive forms:
                #   mark_box((x0, y0, x1, y1))
                #   mark_box(x0, y0, x1, y1)
                box = (
                    (box_or_x0, y0, x1, y1)
                    if y0 is not None or x1 is not None or y1 is not None
                    else box_or_x0
                )
                values = _coords(box, normalized=normalized)
                if len(values) != 4:
                    raise ValueError("mark_box requires x0, y0, x1, y1")
                x0, y0, x1, y1 = values
                item = {
                    "kind": "box",
                    "label": str(label),
                    "bbox_xyxy": [
                        max(0.0, min(float(image.width), min(x0, x1))),
                        max(0.0, min(float(image.height), min(y0, y1))),
                        max(0.0, min(float(image.width), max(x0, x1))),
                        max(0.0, min(float(image.height), max(y0, y1))),
                    ],
                    "style": {
                        "color": str(color),
                        "width": int(width) if isinstance(width, (int, float)) else None,
                    },
                }
                marks.append(item)
                return item

            def mark_point(
                point_or_x: Any,
                y: Any = None,
                label: str = "object",
                normalized: bool = False,
                color: Any = "red",
                radius: int | None = None,
                **_style: Any,
            ) -> dict[str, Any]:
                # Accept both mark_point((x, y)) and mark_point(x, y).
                point = (point_or_x, y) if y is not None else point_or_x
                values = _coords(point, normalized=normalized)
                if len(values) != 2:
                    raise ValueError("mark_point requires x, y")
                x, y = values
                item = {
                    "kind": "point",
                    "label": str(label),
                    "point_xy": [
                        max(0.0, min(float(image.width), x)),
                        max(0.0, min(float(image.height), y)),
                    ],
                    "style": {
                        "color": str(color),
                        "radius": int(radius) if isinstance(radius, (int, float)) else None,
                    },
                }
                marks.append(item)
                return item

            namespace: dict[str, Any] = {
                "image": image,
                "width": image.width,
                "height": image.height,
                "mark_box": mark_box,
                "mark_point": mark_point,
                "marks": marks,
            }
            exec(str(code), namespace, namespace)
            if not marks:
                raise ValueError("Python snippet produced no mark; call mark_box(...) or mark_point(...)")

            draw = ImageDraw.Draw(image)
            for item in marks:
                label = item["label"]
                style = item.get("style", {})
                try:
                    mark_color = ImageColor.getrgb(str(style.get("color") or "red"))
                except ValueError:
                    mark_color = (255, 64, 32)
                if item["kind"] == "box":
                    box = item["bbox_xyxy"]
                    line_width = style.get("width")
                    line_width = (
                        max(1, int(line_width))
                        if isinstance(line_width, int)
                        else max(2, image.width // 128)
                    )
                    draw.rectangle(box, outline=mark_color, width=line_width)
                    draw.text((box[0] + 3, box[1] + 3), label, fill=(255, 255, 0))
                else:
                    x, y = item["point_xy"]
                    requested_radius = style.get("radius")
                    radius = (
                        max(2, int(requested_radius))
                        if isinstance(requested_radius, int)
                        else max(5, image.width // 64)
                    )
                    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=mark_color, width=3)
                    draw.line((x - radius * 2, y, x + radius * 2, y), fill=mark_color, width=2)
                    draw.line((x, y - radius * 2, x, y + radius * 2), fill=mark_color, width=2)
                    draw.text((x + radius + 2, y + 2), label, fill=(255, 255, 0))

            output = self.root / "perception" / "manual_marks" / f"{self.current_record.get('observation_id', 'observation')}-{int(time.time() * 1000)}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            image.save(output)

            # A 256 px scene view is often enough to locate an instance but not
            # enough to verify small packaging text or distinguish nearby cans.
            # Return a deterministic local zoom around the mark as additional
            # evidence; it never changes the SAM3 prompt coordinates.
            x_values: list[float] = []
            y_values: list[float] = []
            for item in marks:
                if item["kind"] == "box":
                    x0, y0, x1, y1 = item["bbox_xyxy"]
                else:
                    x, y = item["point_xy"]
                    radius = max(36.0, min(image.width, image.height) * 0.18)
                    x0, y0, x1, y1 = x - radius, y - radius, x + radius, y + radius
                x_values.extend((float(x0), float(x1)))
                y_values.extend((float(y0), float(y1)))
            x0, x1 = min(x_values), max(x_values)
            y0, y1 = min(y_values), max(y_values)
            margin = max(12.0, 0.25 * max(x1 - x0, y1 - y0))
            crop_box = (
                max(0, int(math.floor(x0 - margin))),
                max(0, int(math.floor(y0 - margin))),
                min(image.width, int(math.ceil(x1 + margin))),
                min(image.height, int(math.ceil(y1 + margin))),
            )
            zoom = image.crop(crop_box)
            scale = max(1, min(6, int(384 / max(1, max(zoom.size)))))
            if scale > 1:
                zoom = zoom.resize(
                    (zoom.width * scale, zoom.height * scale),
                    Image.Resampling.LANCZOS,
                )
            zoom_output = output.with_name(f"{output.stem}.zoom.png")
            zoom.save(zoom_output)
        except Exception as exc:  # noqa: BLE001 - operator can correct and retry.
            message = f"Python marking failed: {type(exc).__name__}: {exc}"
            self.writer.record_tool_result(
                tool="python_mark_object",
                success=False,
                tool_call_id=call.get("tool_call_id"),
                input_frames=self.current_record.get("frame_ids", []),
                result={"reason": "python_error", "message": message},
            )
            return GatewayResult(False, {"kind": "python_mark_object", "success": False, "error": message})

        self._manual_mark_image = output
        self._manual_marks = marks
        self.current_visualization = {
            "kind": "manual_mark",
            "image_path": str(output),
            "observation_id": self.current_record.get("observation_id"),
            "marks": marks,
        }
        self._write_current_projection()
        self.writer.record_tool_result(
            tool="python_mark_object",
            success=True,
            tool_call_id=call.get("tool_call_id"),
            input_frames=self.current_record.get("frame_ids", []),
            artifact_refs=[str(output), str(zoom_output)],
            result={"mark_count": len(marks), "marks": marks, "crop_box_xyxy": crop_box},
        )
        return GatewayResult(
            True,
            {
                "kind": "python_mark_object",
                "success": True,
                "observation_id": self.current_record.get("observation_id"),
                "mark_count": len(marks),
                "message": "Inspect both the full marked image and the local zoom. Verify instance identity before calling segment_object explicitly.",
            },
            images=[output, zoom_output],
            details={
                "marks": marks,
                "source_image": str(source),
                "marked_image": str(output),
                "zoom_image": str(zoom_output),
                "crop_box_xyxy": crop_box,
            },
        )

    def segment_object(self, target: str) -> GatewayResult:
        self.ensure_started()
        assert self.current_record is not None and self.perception is not None and self.writer is not None
        if self.failure_case is not None:
            return self._failed_result("segmentation")
        call = self.writer.record_tool_start(
            tool="segment_object",
            input_frames=self.current_record.get("frame_ids", []),
            parameters={"target": target},
        )
        point_prompts: list[dict[str, float | int]] | None = None
        if self._manual_marks:
            if len(self._manual_marks) != 1:
                message = "Manual SAM3 prompting requires exactly one point or box; draw one mark and retry."
                self.writer.record_tool_result(
                    tool="segment_object",
                    success=False,
                    tool_call_id=call.get("tool_call_id"),
                    input_frames=self.current_record.get("frame_ids", []),
                    result={"reason": "single_manual_mark_required", "retryable": True},
                )
                return GatewayResult(
                    False,
                    {
                        "kind": "segmentation",
                        "success": False,
                        "reason": "single_manual_mark_required",
                        "retryable": True,
                        "message": message,
                    },
                    images=[self._manual_mark_image] if self._manual_mark_image is not None else [],
                )
            mark = self._manual_marks[0]
            if mark.get("kind") == "point" and isinstance(mark.get("point_xy"), list):
                point = mark["point_xy"]
                x, y = float(point[0]), float(point[1])
            elif mark.get("kind") == "box" and isinstance(mark.get("bbox_xyxy"), list):
                x0, y0, x1, y1 = (float(value) for value in mark["bbox_xyxy"])
                x, y = (x0 + x1) * 0.5, (y0 + y1) * 0.5
            else:
                message = "Manual SAM3 prompting requires one valid point or box."
                self.writer.record_tool_result(
                    tool="segment_object",
                    success=False,
                    tool_call_id=call.get("tool_call_id"),
                    input_frames=self.current_record.get("frame_ids", []),
                    result={"reason": "invalid_manual_mark", "retryable": True},
                )
                return GatewayResult(
                    False,
                    {
                        "kind": "segmentation",
                        "success": False,
                        "reason": "invalid_manual_mark",
                        "retryable": True,
                        "message": message,
                    },
                    images=[self._manual_mark_image] if self._manual_mark_image is not None else [],
                )
            point_prompts = [{"x": x, "y": y, "label": 1}]
        try:
            result = self.perception.segment_object(
                self.current_record,
                target,
                point_prompts=point_prompts,
            )
        except Exception as exc:  # noqa: BLE001
            result = ToolResult(False, f"segment_object failed: {exc}", {"reason": "exception"})
        details = result.details
        if self._manual_mark_image is not None:
            details = dict(details)
            details["manual_mark_image"] = str(self._manual_mark_image)
            details["manual_marks"] = self._manual_marks
        images = self._segmentation_images(details)
        if result.success and not images:
            details = dict(details)
            detections = details.get("detections")
            detection_count = details.get("detection_count")
            no_detections = not (
                isinstance(detections, list)
                and detections
                or isinstance(detection_count, int)
                and detection_count > 0
            )
            details.setdefault("diagnostics", []).append(
                {
                    "code": "sam3_no_detections" if no_detections else "sam3_visualization_missing",
                    "message": (
                        "SAM3 returned no detections."
                        if no_detections
                        else "SAM3 returned no renderable mask overlay."
                    ),
                }
            )
            details["reason"] = "no_detections" if no_detections else "visualization_failed"
            result = ToolResult(
                False,
                "SAM3 segmentation failed: no detections."
                if no_detections
                else "SAM3 segmentation failed: no renderable mask overlay.",
                details,
            )
        self.latest_segmentation = result
        self.selected_detection = None
        self.selected_detection_observation_id = ""
        detections = details.get("detections", []) if isinstance(details, Mapping) else []
        detection_ids = [
            str(item.get("id"))
            for item in detections
            if isinstance(item, Mapping) and item.get("id") is not None
        ]
        summary = {
            "observation": details.get("observation"),
            "detection_ids": detection_ids,
            "selection_required": bool(details.get("selection_required", len(detection_ids) > 1)),
        }
        artifact_refs = _artifact_refs(details)
        self.writer.record_tool_result(
            tool="segment_object",
            success=result.success,
            tool_call_id=call.get("tool_call_id"),
            input_frames=self.current_record.get("frame_ids", []),
            artifact_refs=artifact_refs,
            result=summary,
        )
        failure = None
        issue = None
        if not result.success and details.get("reason") == "no_detections":
            issue = self._record_issue(
                category="perception",
                component="sam3",
                code="no_detections",
                message=result.content,
                tool="segment_object",
                input_frames=list(self.current_record.get("frame_ids", [])),
                artifact_refs=artifact_refs,
                details={"target": target, "retry_hint": "Try a shorter visual concept or explicit mark."},
            )
        elif not result.success and (
            "backend" in details
            or details.get("reason") in {"mask_not_found", "visualization_failed", "invalid_segmentation"}
        ):
            failure = self._mark_failure(
                category="perception",
                component="sam3",
                code=str(details.get("reason") or "sam3_failed"),
                message=result.content,
                tool="segment_object",
                input_frames=list(self.current_record.get("frame_ids", [])),
                artifact_refs=artifact_refs,
                details=self._backend_failure_details(details),
            )
        if images:
            self.current_visualization = {
                "kind": "segmentation",
                "image_path": str(images[0]),
                "observation_id": self.current_record.get("observation_id"),
                "frame_id": details.get("observation", {}).get("frame_id")
                if isinstance(details.get("observation"), Mapping)
                else None,
            }
            self._write_current_projection()
        response_images = list(images)
        reference = self._active_object_reference
        reference_attached = False
        if (
            result.success
            and isinstance(reference, Mapping)
            and _semantic_targets_overlap(
                str(reference.get("normalized_target") or ""),
                str(target),
            )
        ):
            reference_path = Path(str(reference.get("image") or ""))
            if reference_path.is_file() and response_images:
                comparison_path = (
                    self.root
                    / "perception"
                    / "reference_comparisons"
                    / (
                        f"{_safe_name(str(self.current_record.get('observation_id') or 'obs'))}-"
                        f"{_safe_name(str(target))}.png"
                    )
                )
                _write_reference_detection_board(
                    reference_path,
                    response_images[0],
                    comparison_path,
                )
                response_images.insert(0, comparison_path)
                reference_attached = True
        return GatewayResult(
            result.success,
            {
                "kind": "segmentation",
                "success": result.success,
                "target": target,
                "observation_id": self.current_record.get("observation_id"),
                "detection_ids": detection_ids,
                "selection_required": summary["selection_required"],
                "message": (
                    "The first image places the active canonical reference on the "
                    "left and current detections on the right. Compare them, inspect "
                    "the following contact sheet and overlays, then select one "
                    "detection_id."
                    if reference_attached and summary["selection_required"]
                    else "The first image places the active canonical reference on the "
                    "left and the current mask on the right; compare them before "
                    "proposing grasps."
                    if reference_attached
                    else "Inspect the mask image and select one detection_id."
                    if summary["selection_required"]
                    else "Inspect the mask image before proposing grasps."
                ),
                **(
                    {
                        "canonical_reference_target": reference.get("target"),
                        "canonical_reference_attached": True,
                    }
                    if reference_attached and isinstance(reference, Mapping)
                    else {}
                ),
                **({"error": result.content} if not result.success else {}),
                **(
                    {
                        "episode_status": "failed",
                        "failure_case_id": failure.get("failure_case_id"),
                        "failure_code": failure.get("code"),
                        "failure_component": failure.get("component"),
                    }
                    if failure
                    else {}
                ),
                **(
                    {
                        "episode_status": "running",
                        "retryable": True,
                        "issue_id": issue.get("issue_id"),
                        "issue_code": issue.get("code"),
                        "retry_hint": "Try a shorter visual concept or use python_mark_object after inspecting alternatives.",
                    }
                    if issue
                    else {}
                ),
            },
            images=response_images,
            details=details,
        )

    def select_detection(self, detection_id: str, reason: str = "") -> GatewayResult:
        """Record the Operator's visual SAM3 detection choice."""

        self.ensure_started()
        assert self.current_record is not None and self.writer is not None
        if self.failure_case is not None:
            return self._failed_result("detection_selection")
        if self.latest_segmentation is None or not self.latest_segmentation.success:
            return GatewayResult(
                False,
                {
                    "kind": "detection_selection",
                    "success": False,
                    "error": "Call segment_object first.",
                },
            )
        details = self.latest_segmentation.details
        binding = details.get("observation") if isinstance(details, Mapping) else None
        if not isinstance(binding, Mapping) or binding.get("observation_id") != self.current_record.get("observation_id"):
            return GatewayResult(
                False,
                {
                    "kind": "detection_selection",
                    "success": False,
                    "reason": "stale_segmentation",
                    "error": "The SAM3 result belongs to an older observation; segment again.",
                },
            )
        detections = details.get("detections", []) if isinstance(details, Mapping) else []
        selected = next(
            (
                item for item in detections
                if isinstance(item, Mapping) and item.get("id") == detection_id
            ),
            None,
        )
        if selected is None:
            return GatewayResult(
                False,
                {
                    "kind": "detection_selection",
                    "success": False,
                    "error": f"Unknown detection_id: {detection_id}",
                    "detection_ids": [
                        str(item.get("id")) for item in detections
                        if isinstance(item, Mapping) and item.get("id") is not None
                    ],
                },
            )
        self.selected_detection = dict(selected)
        self.selected_detection_observation_id = str(self.current_record.get("observation_id") or "")
        self.writer.record_operator_choice(
            choice_type="detection",
            choice_id=detection_id,
            input_frames=self.current_record.get("frame_ids", []),
            metadata={"reason": reason},
        )
        return GatewayResult(
            True,
            {
                "kind": "detection_selection",
                "success": True,
                "observation_id": self.selected_detection_observation_id,
                "selected_detection_id": detection_id,
                "message": "Detection selected. Call propose_grasps next.",
            },
            images=self._segmentation_images(details),
            details={"selected_detection": self.selected_detection},
        )

    def propose_grasps(self, detection_id: str | None = None) -> GatewayResult:
        self.ensure_started()
        assert self.current_record is not None and self.perception is not None and self.writer is not None
        if self.failure_case is not None:
            return self._failed_result("grasp_candidates")
        # Never leave an older successful result selectable after a failed or
        # retried request.
        self.latest_grasp = None
        self.latest_grasp_proposal_id = ""
        self.latest_grasp_result_id = ""
        self.selected_grasp = None
        self.selected_grasp_observation_id = ""
        effective_detection_id = detection_id
        if effective_detection_id is None and self.selected_detection_observation_id == self.current_record.get("observation_id"):
            effective_detection_id = str(self.selected_detection.get("id")) if self.selected_detection else None
        call = self.writer.record_tool_start(
            tool="propose_grasps",
            input_frames=self.current_record.get("frame_ids", []),
            parameters={"detection_id": effective_detection_id},
        )
        try:
            result = self.perception.propose_grasps(self.current_record, detection_id=effective_detection_id)
        except Exception as exc:  # noqa: BLE001
            result = ToolResult(False, f"propose_grasps failed: {exc}", {"reason": "exception"})
        details = result.details
        coherent_detection = (
            details.get("selected_detection")
            if isinstance(details, Mapping)
            else None
        )
        proposal_id = ""
        result_id = ""
        if result.success:
            result_id = str(details.get("result_id") or "").strip()
            identity_source = result_id or str(call.get("tool_call_id") or "").strip()
            proposal_id = f"proposal-{identity_source}"
            details["proposal_id"] = proposal_id
            details["result_id"] = result_id or identity_source
            self.latest_grasp = result
            self.latest_grasp_proposal_id = proposal_id
            self.latest_grasp_result_id = str(details["result_id"])
            self.selected_grasp = None
            self.selected_grasp_observation_id = ""
            if isinstance(coherent_detection, Mapping):
                # Replace the complete detection binding in one assignment.
                # In particular, the depth-coherent mask returned by propose
                # becomes the only mask authority for later refinements.
                self.selected_detection = dict(coherent_detection)
                self.selected_detection_observation_id = str(
                    self.current_record.get("observation_id") or ""
                )
            else:
                self.selected_detection = None
                self.selected_detection_observation_id = ""
        else:
            self.latest_grasp = None
            self.latest_grasp_proposal_id = ""
            self.latest_grasp_result_id = ""
            self.selected_grasp = None
            self.selected_grasp_observation_id = ""
        candidates = details.get("grasp_candidates", []) if isinstance(details, Mapping) else []
        candidate_ids = [
            str(item.get("id"))
            for item in candidates
            if isinstance(item, Mapping) and item.get("id") is not None
        ]
        summary = {
            "observation": details.get("observation"),
            "candidate_ids": candidate_ids,
            "selected_detection": details.get("selected_detection"),
            "proposal_id": proposal_id or None,
            "result_id": self.latest_grasp_result_id or None,
            "canonical_grasp_candidates_ref": details.get(
                "canonical_grasp_candidates_ref"
            ),
            "inspection_surface": "viser",
        }
        self.writer.record_tool_result(
            tool="propose_grasps",
            success=result.success,
            tool_call_id=call.get("tool_call_id"),
            input_frames=self.current_record.get("frame_ids", []),
            artifact_refs=_artifact_refs(details),
            result=summary,
        )
        failure = None
        issue = None
        failure_reason = str(details.get("reason") or "")
        fatal_reasons = {
            "service_unavailable",
            "backend_unavailable",
            "model_load_failed",
            "model_inference_failed",
            "transport_error",
            "exception",
        }
        if not result.success and failure_reason in fatal_reasons:
            failure = self._mark_failure(
                category="perception",
                component="anygrasp",
                code=failure_reason or "anygrasp_failed",
                message=result.content,
                tool="propose_grasps",
                input_frames=list(self.current_record.get("frame_ids", [])),
                artifact_refs=_artifact_refs(details),
                details=self._backend_failure_details(details),
            )
        elif not result.success:
            issue = self._record_issue(
                category="perception",
                component="anygrasp",
                code=failure_reason or "grasp_proposal_failed",
                message=result.content,
                tool="propose_grasps",
                input_frames=list(self.current_record.get("frame_ids", [])),
                artifact_refs=_artifact_refs(details),
                details={
                    **self._backend_failure_details(details),
                    "retry_hint": (
                        "Re-observe or change the selected target region/view, then "
                        "retry. A destination region should use placement semantics, "
                        "not grasp proposal semantics."
                    ),
                },
            )
        return GatewayResult(
            result.success,
            {
                "kind": "grasp_candidates",
                "success": result.success,
                "observation_id": self.current_record.get("observation_id"),
                "candidate_ids": candidate_ids,
                **({"proposal_id": proposal_id, "result_id": self.latest_grasp_result_id} if result.success else {}),
                "inspection_surface": "viser",
                "message": "Grasp candidates retained. Wait for the matching Viser proposal scene, inspect poses from useful views, then select_grasp(grasp_id) to commit the exact candidate."
                if result.success
                else result.content,
                **(
                    {
                        "episode_status": "failed",
                        "failure_case_id": failure.get("failure_case_id"),
                        "failure_code": failure.get("code"),
                        "failure_component": failure.get("component"),
                    }
                    if failure
                    else {}
                ),
                **(
                    {
                        "episode_status": "running",
                        "retryable": True,
                        "issue_id": issue.get("issue_id"),
                        "issue_code": issue.get("code"),
                    }
                    if issue
                    else {}
                ),
            },
            details=details,
        )

    def inspect_grasp(self, grasp_id: str) -> GatewayResult:
        """Focus and capture one Viser candidate without committing it."""

        self.ensure_started()
        assert self.current_record is not None and self.writer is not None
        if self.failure_case is not None:
            return self._failed_result("grasp_inspection")
        call = self.writer.record_tool_start(
            tool="inspect_grasp",
            input_frames=self.current_record.get("frame_ids", []),
            parameters={"grasp_id": grasp_id},
        )
        if self.latest_grasp is None or not self.latest_grasp.success:
            result = GatewayResult(
                False,
                {"kind": "grasp_inspection", "success": False, "error": "Call propose_grasps first."},
            )
            self.writer.record_tool_result(
                tool="inspect_grasp",
                success=False,
                tool_call_id=call.get("tool_call_id"),
                input_frames=self.current_record.get("frame_ids", []),
                result={"grasp_id": grasp_id, "reason": "grasps_required"},
            )
            return result
        if not self._latest_grasp_matches_current_observation():
            result = GatewayResult(
                False,
                {
                    "kind": "grasp_inspection",
                    "success": False,
                    "reason": "stale_grasp",
                    "error": "The grasp candidates belong to an older observation; propose grasps again.",
                },
            )
            self.writer.record_tool_result(
                tool="inspect_grasp",
                success=False,
                tool_call_id=call.get("tool_call_id"),
                input_frames=self.current_record.get("frame_ids", []),
                result={"grasp_id": grasp_id, "reason": "stale_grasp"},
            )
            return result

        candidates = self.latest_grasp.details.get("grasp_candidates", [])
        candidate = next(
            (item for item in candidates if isinstance(item, Mapping) and item.get("id") == grasp_id),
            None,
        )
        if candidate is None:
            result = GatewayResult(
                False,
                {
                    "kind": "grasp_inspection",
                    "success": False,
                    "error": f"Unknown grasp_id: {grasp_id}",
                    "candidate_ids": [
                        str(item.get("id"))
                        for item in candidates
                        if isinstance(item, Mapping) and item.get("id") is not None
                    ],
                },
            )
            self.writer.record_tool_result(
                tool="inspect_grasp",
                success=False,
                tool_call_id=call.get("tool_call_id"),
                input_frames=self.current_record.get("frame_ids", []),
                result={"grasp_id": grasp_id, "reason": "unknown_grasp"},
            )
            return result

        inspector_state = self._validated_grasp_inspector_state(
            expected_scene_mode="proposal"
        )
        inspector_pose_id: str | None = None
        support_clearance: dict[str, Any] | None = None
        if inspector_state.get("success"):
            inspector_pose = next(
                (
                    item
                    for item in inspector_state.get("poses", [])
                    if isinstance(item, Mapping)
                    and (
                        item.get("source_id") == grasp_id
                        or item.get("pose_id") == grasp_id
                    )
                ),
                None,
            )
            if isinstance(inspector_pose, Mapping):
                inspector_pose_id = str(inspector_pose.get("pose_id") or "") or None
                support_clearance = self._grasp_support_clearance_contract(
                    grasp_id, inspector_state=inspector_state
                )
        if inspector_pose_id is None:
            code = str(inspector_state.get("code") or "pose_not_in_viser_scene")
            message = str(
                inspector_state.get("message")
                or "The matching grasp pose is not present in the current Viser scene yet."
            )
            self.writer.record_tool_result(
                tool="inspect_grasp",
                success=False,
                tool_call_id=call.get("tool_call_id"),
                input_frames=self.current_record.get("frame_ids", []),
                result={"grasp_id": grasp_id, "reason": code},
            )
            return GatewayResult(
                False,
                {
                    "kind": "grasp_inspection",
                    "success": False,
                    "code": code,
                    "retryable": True,
                    "observation_id": self.current_record.get("observation_id"),
                    "inspected_grasp_id": grasp_id,
                    "message": message,
                },
                details={"inspected_grasp": dict(candidate), "inspector": inspector_state},
            )
        viewer_ids = sorted(
            str(item.get("viewer_id"))
            for item in inspector_state.get("viewer_clients", [])
            if isinstance(item, Mapping)
            and item.get("connected") is not False
            and item.get("viewer_id") is not None
        )
        viewer_id = viewer_ids[0] if viewer_ids else None
        view_result = self._grasp_inspector.configure(
            show_anygrasp=True,
            show_graspgenx=True,
            show_refined=True,
            pose_scope="focus",
            focus_pose_id=inspector_pose_id,
            camera_preset="pose_jaws",
            **({"viewer_id": viewer_id} if viewer_id is not None else {}),
        )
        capture_result: dict[str, Any] = {}
        image: Path | None = None
        if view_result.get("success"):
            capture_result, image = self._grasp_inspector.capture_image(
                camera="current",
                **({"viewer_id": viewer_id} if viewer_id is not None else {}),
            )
        success = bool(view_result.get("success") and capture_result.get("success") and image)
        artifact_refs = [str(image)] if image is not None else []
        self.writer.record_tool_result(
            tool="inspect_grasp",
            success=success,
            tool_call_id=call.get("tool_call_id"),
            input_frames=self.current_record.get("frame_ids", []),
            artifact_refs=artifact_refs,
            result={
                "grasp_id": grasp_id,
                "inspector_pose_id": inspector_pose_id,
                "view_success": bool(view_result.get("success")),
                "capture_success": bool(capture_result.get("success")),
                **(support_clearance or {}),
            },
        )
        return GatewayResult(
            success,
            {
                "kind": "grasp_inspection",
                "success": success,
                "observation_id": self.current_record.get("observation_id"),
                "inspected_grasp_id": grasp_id,
                "inspector_pose_id": inspector_pose_id,
                **(support_clearance or {}),
                "message": "Viser pose focused and captured. This is read-only; call select_grasp(grasp_id) only after judging the geometry."
                if success
                else str(
                    capture_result.get("message")
                    or view_result.get("message")
                    or "Viser inspection failed."
                ),
            },
            images=[image] if image is not None else [],
            details={
                "inspected_grasp": dict(candidate),
                "support_clearance": support_clearance,
                "inspector_view": view_result,
                "inspector_capture": capture_result,
            },
        )

    def get_grasp_inspector(self) -> GatewayResult:
        """Return persistent 3D inspector state without changing the scene."""

        result = self._validated_grasp_inspector_state()
        deadline = time.monotonic() + GRASP_INSPECTOR_SCENE_WAIT_S
        while (
            not result.get("success")
            and result.get("code")
            in {"stale_scene", "stale_proposal", "wrong_scene_mode"}
            and result.get("retryable") is not False
            and time.monotonic() < deadline
        ):
            time.sleep(GRASP_INSPECTOR_SCENE_POLL_S)
            result = self._validated_grasp_inspector_state()
        return GatewayResult(
            bool(result.get("success")),
            {"kind": "grasp_inspector", **result},
            details=dict(result),
        )

    def configure_grasp_view(
        self,
        *,
        show_anygrasp: bool = True,
        show_graspgenx: bool = True,
        show_refined: bool = True,
        pose_scope: str = "default",
        focus_pose_id: str | None = None,
        camera_preset: str = "keep",
        viewer_id: str | None = None,
        orbit_azimuth_deg: float = 0.0,
        orbit_elevation_deg: float = 0.0,
        zoom_scale: float = 1.0,
    ) -> GatewayResult:
        """Set an explicit Viser view; this never selects or executes a pose."""

        state = self._validated_grasp_inspector_state()
        if not state.get("success"):
            return GatewayResult(
                False,
                {"kind": "grasp_inspector_view", **state},
                details=dict(state),
            )
        arguments: dict[str, Any] = {
            "show_anygrasp": bool(show_anygrasp),
            "show_graspgenx": bool(show_graspgenx),
            "show_refined": bool(show_refined),
            "pose_scope": str(pose_scope),
            "camera_preset": str(camera_preset),
        }
        if focus_pose_id is not None:
            arguments["focus_pose_id"] = str(focus_pose_id)
        if viewer_id is not None:
            arguments["viewer_id"] = str(viewer_id)
        if any(
            abs(float(value)) > 1e-12
            for value in (orbit_azimuth_deg, orbit_elevation_deg)
        ) or not math.isclose(float(zoom_scale), 1.0):
            arguments.update(
                {
                    "orbit_azimuth_deg": float(orbit_azimuth_deg),
                    "orbit_elevation_deg": float(orbit_elevation_deg),
                    "zoom_scale": float(zoom_scale),
                }
            )
        result = self._grasp_inspector.configure(**arguments)
        return GatewayResult(
            bool(result.get("success")),
            {"kind": "grasp_inspector_view", **result},
            details=dict(result),
        )

    def refine_mask_side_grasp(
        self,
        *,
        base_grasp_id: str,
        side: str,
        insertion_depth_m: float,
        surface_floor_quantile: float = 0.05,
        top_quantile: float = 0.99,
        reason: str = "",
    ) -> GatewayResult:
        """Create a world-frame Panda pose from selected-mask geometry."""

        self.ensure_started()
        assert self.current_record is not None and self.writer is not None
        call = self.writer.record_tool_start(
            tool="refine_mask_side_grasp",
            input_frames=self.current_record.get("frame_ids", []),
            parameters={
                "base_grasp_id": base_grasp_id,
                "side": side,
                "insertion_depth_m": insertion_depth_m,
                "surface_floor_quantile": surface_floor_quantile,
                "top_quantile": top_quantile,
                "reason": reason,
            },
        )
        try:
            proposal_detection = self._validate_current_proposal_binding()
            base = self._current_grasp_candidate(base_grasp_id)
            if base is None:
                raise GraspPoseRefinementError(
                    f"unknown current base_grasp_id: {base_grasp_id}"
                )
            mask_ref = proposal_detection.get("mask_ref")
            if not isinstance(mask_ref, str) or not Path(mask_ref).is_file():
                raise GraspPoseRefinementError("selected detection mask is missing")
            frame = resolve_observation_frame(
                self.current_record,
                artifact_root=self.root,
                camera_id="agentview",
            )
            refined = derive_mask_side_pose(
                frame,
                mask_path=mask_ref,
                detection_id=str(proposal_detection.get("id") or ""),
                side=side,  # type: ignore[arg-type]
                insertion_depth_m=float(insertion_depth_m),
                surface_floor_quantile=float(surface_floor_quantile),
                top_quantile=float(top_quantile),
            )
            registration = self._register_world_grasp_pose(
                pose_id=refined.pose_id,
                base_grasp_id=base_grasp_id,
                transform=refined.world_from_grip_site,
                method="mask_side_top_down_v1",
                reason=reason,
                provenance=refined.as_dict(),
            )
        except (GraspPoseRefinementError, ValueError, TypeError) as exc:
            inspector_wait = isinstance(exc, GraspInspectorSceneNotReady)
            self.writer.record_tool_result(
                tool="refine_mask_side_grasp",
                success=False,
                tool_call_id=call.get("tool_call_id"),
                input_frames=self.current_record.get("frame_ids", []),
                result={
                    "reason": "inspector_not_ready" if inspector_wait else "invalid_refinement",
                    "message": str(exc),
                    "retryable": inspector_wait,
                },
            )
            return GatewayResult(
                False,
                {
                    "kind": "grasp_pose_refinement",
                    "success": False,
                    "reason": "inspector_not_ready" if inspector_wait else "invalid_refinement",
                    "retryable": inspector_wait,
                    "message": str(exc),
                },
            )
        self.writer.record_tool_result(
            tool="refine_mask_side_grasp",
            success=True,
            tool_call_id=call.get("tool_call_id"),
            input_frames=self.current_record.get("frame_ids", []),
            artifact_refs=[registration["artifact_ref"]],
            result={
                "pose_id": registration["candidate"]["id"],
                "base_grasp_id": base_grasp_id,
                "method": "mask_side_top_down_v1",
                **registration["support_clearance"],
                "inspector_registered": registration["inspector_registered"],
                "inspector_capture_success": registration["inspector_capture_success"],
            },
        )
        refinement_diagnostics = refined.diagnostics
        engagement = _mask_side_engagement_diagnostics(
            float(insertion_depth_m),
            target_z_m=float(refined.world_from_grip_site[2, 3]),
            mask_low_quantile_z_m=(
                float(refinement_diagnostics["mask_low_quantile_z_m"])
                if isinstance(
                    refinement_diagnostics.get("mask_low_quantile_z_m"),
                    (int, float),
                )
                else None
            ),
        )
        return GatewayResult(
            True,
            {
                "kind": "grasp_pose_refinement",
                "success": True,
                "observation_id": self.current_record.get("observation_id"),
                "pose_id": registration["candidate"]["id"],
                "base_grasp_id": base_grasp_id,
                "side": side,
                "insertion_depth_mm": round(float(insertion_depth_m) * 1000.0, 3),
                "geometry_diagnostics": engagement,
                **registration["support_clearance"],
                "inspector_registered": registration["inspector_registered"],
                "inspector_capture_success": registration["inspector_capture_success"],
                "message": (
                    "Refined pose registered and shown in green. The returned "
                    "3D jaws-side image and geometry_diagnostics are required evidence "
                    "when available. Adjust contact depth in either direction as "
                    "the geometry warrants; selection is enabled only when signed "
                    "collision-mesh support clearance is eligible."
                ),
            },
            images=self._registration_images(registration),
            details=registration,
        )

    def register_grasp_pose(
        self,
        *,
        pose_id: str,
        base_grasp_id: str,
        transform_world_from_grip_site: list[list[float]],
        method: str,
        reason: str = "",
    ) -> GatewayResult:
        """Register an Operator-computed world-frame Panda grip-site pose."""

        self.ensure_started()
        assert self.current_record is not None and self.writer is not None
        call = self.writer.record_tool_start(
            tool="register_grasp_pose",
            input_frames=self.current_record.get("frame_ids", []),
            parameters={
                "pose_id": pose_id,
                "base_grasp_id": base_grasp_id,
                "transform_world_from_grip_site": transform_world_from_grip_site,
                "method": method,
                "reason": reason,
            },
        )
        try:
            transform = rigid_transform(
                transform_world_from_grip_site,
                name="transform_world_from_grip_site",
            )
            registration = self._register_world_grasp_pose(
                pose_id=pose_id,
                base_grasp_id=base_grasp_id,
                transform=transform,
                method=method,
                reason=reason,
                provenance={"operator_supplied_transform": True},
            )
        except (GraspPoseRefinementError, ValueError, TypeError) as exc:
            inspector_wait = isinstance(exc, GraspInspectorSceneNotReady)
            self.writer.record_tool_result(
                tool="register_grasp_pose",
                success=False,
                tool_call_id=call.get("tool_call_id"),
                input_frames=self.current_record.get("frame_ids", []),
                result={
                    "reason": "inspector_not_ready" if inspector_wait else "invalid_pose",
                    "message": str(exc),
                    "retryable": inspector_wait,
                },
            )
            return GatewayResult(
                False,
                {
                    "kind": "grasp_pose_registration",
                    "success": False,
                    "reason": "inspector_not_ready" if inspector_wait else "invalid_pose",
                    "retryable": inspector_wait,
                    "message": str(exc),
                },
            )
        self.writer.record_tool_result(
            tool="register_grasp_pose",
            success=True,
            tool_call_id=call.get("tool_call_id"),
            input_frames=self.current_record.get("frame_ids", []),
            artifact_refs=[registration["artifact_ref"]],
            result={
                "pose_id": pose_id,
                "base_grasp_id": base_grasp_id,
                **registration["support_clearance"],
                "inspector_registered": registration["inspector_registered"],
                "inspector_capture_success": registration["inspector_capture_success"],
            },
        )
        return GatewayResult(
            True,
            {
                "kind": "grasp_pose_registration",
                "success": True,
                "observation_id": self.current_record.get("observation_id"),
                "pose_id": pose_id,
                "base_grasp_id": base_grasp_id,
                **registration["support_clearance"],
                "inspector_registered": registration["inspector_registered"],
                "inspector_capture_success": registration["inspector_capture_success"],
                "message": (
                    "Exact world-frame Panda grip-site pose registered; no pose "
                    "values were modified. Registration and inspection remain "
                    "available even when support clearance is unsafe or unknown."
                ),
            },
            images=self._registration_images(registration),
            details=registration,
        )

    def refine_grasp_pose(
        self,
        *,
        base_grasp_id: str,
        translation_delta_mm: list[float],
        rotation_delta_deg: list[float],
        frame: str = "grasp_local",
        reason: str = "",
    ) -> GatewayResult:
        """Apply one explicit SE(3) delta and register an immutable child pose."""

        self.ensure_started()
        assert self.current_record is not None and self.writer is not None
        call = self.writer.record_tool_start(
            tool="refine_grasp_pose",
            input_frames=self.current_record.get("frame_ids", []),
            parameters={
                "base_grasp_id": base_grasp_id,
                "translation_delta_mm": translation_delta_mm,
                "rotation_delta_deg": rotation_delta_deg,
                "frame": frame,
                "reason": reason,
            },
        )
        try:
            self._validate_current_proposal_binding()
            if frame not in {"grasp_local", "world"}:
                raise GraspPoseRefinementError(
                    "frame must be 'grasp_local' or 'world'"
                )
            translation_delta = _finite_vector3(
                translation_delta_mm, name="translation_delta_mm"
            )
            rotation_delta = _finite_vector3(
                rotation_delta_deg, name="rotation_delta_deg"
            )
            if any(abs(value) > 200.0 for value in translation_delta):
                raise GraspPoseRefinementError(
                    "each translation delta must be within +/-200 mm"
                )
            if any(abs(value) > 180.0 for value in rotation_delta):
                raise GraspPoseRefinementError(
                    "each rotation delta must be within +/-180 degrees"
                )
            base = self._current_grasp_candidate(base_grasp_id)
            if base is None:
                raise GraspPoseRefinementError(
                    f"unknown current base_grasp_id: {base_grasp_id}"
                )
            base_transform = self._candidate_world_from_grip_site(base)
            delta_rotation = _euler_xyz_matrix_degrees(rotation_delta)
            refined_transform = base_transform.copy()
            delta_metres = np.asarray(translation_delta, dtype=np.float64) / 1000.0
            if frame == "grasp_local":
                refined_transform[:3, 3] = (
                    base_transform[:3, 3]
                    + base_transform[:3, :3] @ delta_metres
                )
                refined_transform[:3, :3] = (
                    base_transform[:3, :3] @ delta_rotation
                )
            else:
                refined_transform[:3, 3] = base_transform[:3, 3] + delta_metres
                refined_transform[:3, :3] = (
                    delta_rotation @ base_transform[:3, :3]
                )
            refined_transform = rigid_transform(
                refined_transform, name="refined_world_from_grip_site"
            )
            candidates = self.latest_grasp.details.get("grasp_candidates", [])
            refinement_index = sum(
                1
                for item in candidates
                if isinstance(item, Mapping)
                and item.get("source_backend") == "refined"
            )
            pose_id = f"refined_{refinement_index:03d}"
            registration = self._register_world_grasp_pose(
                pose_id=pose_id,
                base_grasp_id=base_grasp_id,
                transform=refined_transform,
                method="explicit_se3_delta_v1",
                reason=reason,
                provenance={
                    "origin": "model_refinement",
                    "translation_delta_mm": translation_delta,
                    "rotation_delta_deg": rotation_delta,
                    "delta_frame": frame,
                },
            )
        except (GraspPoseRefinementError, ValueError, TypeError) as exc:
            inspector_wait = isinstance(exc, GraspInspectorSceneNotReady)
            self.writer.record_tool_result(
                tool="refine_grasp_pose",
                success=False,
                tool_call_id=call.get("tool_call_id"),
                input_frames=self.current_record.get("frame_ids", []),
                result={
                    "reason": "inspector_not_ready" if inspector_wait else "invalid_refinement",
                    "message": str(exc),
                    "retryable": inspector_wait,
                },
            )
            return GatewayResult(
                False,
                {
                    "kind": "grasp_pose_refinement",
                    "success": False,
                    "reason": "inspector_not_ready" if inspector_wait else "invalid_refinement",
                    "retryable": inspector_wait,
                    "message": str(exc),
                },
            )
        self.writer.record_tool_result(
            tool="refine_grasp_pose",
            success=True,
            tool_call_id=call.get("tool_call_id"),
            input_frames=self.current_record.get("frame_ids", []),
            artifact_refs=[registration["artifact_ref"]],
            result={
                "pose_id": pose_id,
                "base_grasp_id": base_grasp_id,
                "translation_delta_mm": translation_delta,
                "rotation_delta_deg": rotation_delta,
                "delta_frame": frame,
                **registration["support_clearance"],
                "inspector_registered": registration["inspector_registered"],
                "inspector_capture_success": registration["inspector_capture_success"],
            },
        )
        return GatewayResult(
            True,
            {
                "kind": "grasp_pose_refinement",
                "success": True,
                "observation_id": self.current_record.get("observation_id"),
                "pose_id": pose_id,
                "base_grasp_id": base_grasp_id,
                "delta_frame": frame,
                "translation_delta_mm": translation_delta,
                "rotation_delta_deg": rotation_delta,
                **registration["support_clearance"],
                "inspector_registered": registration["inspector_registered"],
                "inspector_capture_success": registration["inspector_capture_success"],
                "message": (
                    "Immutable refined pose registered. Use the returned 3D jaws-side "
                    "image as required evidence when capture succeeded; refine it "
                    "again as needed. select_grasp(pose_id) is enabled only when "
                    "signed collision-mesh support clearance is eligible."
                ),
            },
            images=self._registration_images(registration),
            details=registration,
        )

    def _validate_current_proposal_binding(self) -> Mapping[str, Any]:
        if (
            self.current_record is None
            or self.latest_grasp is None
            or not self.latest_grasp.success
            or not self._latest_grasp_matches_current_observation()
        ):
            raise GraspPoseRefinementError(
                "grasp proposal does not match the current observation"
            )
        details = self.latest_grasp.details
        if (
            not self.latest_grasp_proposal_id
            or details.get("proposal_id") != self.latest_grasp_proposal_id
            or not self.latest_grasp_result_id
            or details.get("result_id") != self.latest_grasp_result_id
        ):
            raise GraspPoseRefinementError(
                "grasp proposal identity is stale or internally inconsistent"
            )
        proposal_detection = details.get("selected_detection")
        if not isinstance(proposal_detection, Mapping):
            raise GraspPoseRefinementError(
                "grasp proposal has no selected detection binding"
            )
        if (
            self.selected_detection is None
            or self.selected_detection_observation_id
            != self.current_record.get("observation_id")
        ):
            raise GraspPoseRefinementError(
                "selected detection does not match the current observation"
            )
        for key in ("id", "mask_ref", "original_mask_ref"):
            if proposal_detection.get(key) != self.selected_detection.get(key):
                raise GraspPoseRefinementError(
                    "selected detection does not match the active grasp proposal"
                )
        return proposal_detection

    def _current_grasp_candidate(self, grasp_id: str) -> Mapping[str, Any] | None:
        if (
            self.latest_grasp is None
            or not self.latest_grasp.success
            or not self._latest_grasp_matches_current_observation()
        ):
            return None
        candidates = self.latest_grasp.details.get("grasp_candidates", [])
        return next(
            (
                item
                for item in candidates
                if isinstance(item, Mapping) and item.get("id") == grasp_id
            ),
            None,
        )

    def rank_grasp_candidates_by_geometry(
        self,
        *,
        target_xyz: Sequence[float] | None = None,
        approach_axis_world: Sequence[float] = (0.0, 0.0, -1.0),
        expected_approach_offset_m: float = 0.025,
        expected_width_m: float = 0.07,
    ) -> list[dict[str, Any]]:
        """Score current grasp candidates by top-down alignment and target centering.

        The score favours candidates whose Panda grip_site approach axis points
        straight down (``-Z`` world), whose jaw centre is close to the target
        centroid in the gripper's lateral plane, and whose width is near a
        typical soup-can diameter.  It is a deterministic filter, not a learned
        model: candidates that are kinematically implausible for the arm (for
        example near-inverted side grasps) are ranked below reachable top-down
        ones regardless of the AnyGrasp confidence score.

        Returns a list of dicts sorted by descending ``geometry_score``.  Each
        entry carries the original candidate id plus the diagnostic features so
        the caller can inspect why a candidate won or lost.
        """

        if self.latest_grasp is None or not self.latest_grasp.success:
            return []
        candidates = self.latest_grasp.details.get("grasp_candidates", [])
        target = (
            np.asarray(target_xyz, dtype=np.float64).reshape(3)
            if target_xyz is not None
            else None
        )
        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            grasp_id = str(candidate.get("id") or "")
            try:
                transform = self._candidate_world_from_grip_site(candidate)
            except Exception as exc:  # noqa: BLE001 - skip malformed candidates.
                scored.append(
                    {
                        "id": grasp_id,
                        "geometry_score": float("-inf"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            rotation = transform[:3, :3]
            translation = transform[:3, 3]
            width = float(candidate.get("width") or 0.08)
            features = grasp_geometry_score(
                rotation=rotation,
                translation=translation,
                width=width,
                target_xyz=target,
                approach_axis_world=approach_axis_world,
                expected_approach_offset_m=expected_approach_offset_m,
                expected_width_m=expected_width_m,
            )
            entry: dict[str, Any] = {
                "id": grasp_id,
                "translation_xyz": translation.tolist(),
                "anygrasp_score": candidate.get("score"),
                "anygrasp_rank": candidate.get("rank"),
                **features,
            }
            scored.append(entry)
        scored.sort(key=lambda item: item["geometry_score"], reverse=True)
        return scored

    def _candidate_world_from_grip_site(
        self, candidate: Mapping[str, Any]
    ) -> np.ndarray:
        """Normalize one current candidate to world-frame Panda grip_site."""

        if (
            candidate.get("pose_frame") == "world"
            and candidate.get("eef_frame") == "panda_grip_site"
        ):
            return rigid_transform(
                candidate.get("transform_world_from_grip_site"),
                name="transform_world_from_grip_site",
            )
        assert self.current_record is not None
        frame = _find_frame(self.current_record, "agentview")
        if frame is None:
            raise GraspPoseRefinementError(
                "current observation has no agentview frame"
            )
        metadata = frame.get("metadata", {})
        extrinsics = metadata.get("extrinsics") if isinstance(metadata, Mapping) else None
        if not isinstance(extrinsics, Mapping):
            raise GraspPoseRefinementError(
                "current agentview has no camera-to-world extrinsics"
            )
        context = ToolExecutionContext(
            name="camera_pose_to_world",
            spec=build_default_tool_registry().get("camera_pose_to_world"),
            parameters={
                "camera_frame_id": "agentview",
                "camera_pose": dict(candidate),
                "camera_extrinsics": dict(extrinsics),
            },
        )
        transformed = _camera_pose_to_world_handler(context)
        if not transformed.success:
            raise GraspPoseRefinementError(transformed.content)
        world_pose = transformed.details.get("outputs", {}).get("world_pose", {})
        translation = _finite_vector3(
            world_pose.get("translation_xyz"), name="world translation"
        )
        panda_rotation = _anygrasp_rotation_to_panda_grip_site(
            world_pose.get("rotation_matrix")
        )
        if panda_rotation is None:
            raise GraspPoseRefinementError(
                "candidate has no usable AnyGrasp rotation"
            )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(panda_rotation, dtype=np.float64)
        transform[:3, 3] = np.asarray(translation, dtype=np.float64)
        return rigid_transform(transform, name="world_from_grip_site")

    def _register_world_grasp_pose(
        self,
        *,
        pose_id: str,
        base_grasp_id: str,
        transform: Any,
        method: str,
        reason: str,
        provenance: Mapping[str, Any],
    ) -> dict[str, Any]:
        assert self.current_record is not None and self.latest_grasp is not None
        self._validate_current_proposal_binding()
        pose_id = str(pose_id).strip()
        method = str(method).strip()
        if not pose_id:
            raise GraspPoseRefinementError("pose_id is required")
        if not method:
            raise GraspPoseRefinementError("method is required")
        base = self._current_grasp_candidate(base_grasp_id)
        if base is None:
            raise GraspPoseRefinementError(
                f"unknown current base_grasp_id: {base_grasp_id}"
            )
        candidates = self.latest_grasp.details.get("grasp_candidates", [])
        if any(
            isinstance(item, Mapping) and item.get("id") == pose_id
            for item in candidates
        ):
            raise GraspPoseRefinementError(f"pose_id already exists: {pose_id}")
        inspector_state = self._validated_grasp_inspector_state(
            expected_scene_mode="proposal"
        )
        if not inspector_state.get("success"):
            raise GraspInspectorSceneNotReady(inspector_state)
        rigid = rigid_transform(transform, name="transform_world_from_grip_site")
        numeric_ranks = [
            int(item["rank"])
            for item in candidates
            if isinstance(item, Mapping) and isinstance(item.get("rank"), int)
        ]
        candidate = {
            "id": pose_id,
            "rank": max(numeric_ranks, default=-1) + 1,
            "pose_frame": "world",
            "eef_frame": "panda_grip_site",
            "transform_world_from_grip_site": rigid.tolist(),
            "translation_xyz": rigid[:3, 3].tolist(),
            "rotation_matrix": rigid[:3, :3].tolist(),
            "width": float(base.get("width") or 0.08),
            "score": None,
            "parent_score": base.get("score"),
            "source_backend": "refined",
            "parent_grasp_id": base_grasp_id,
            "method": method,
            "reason": reason,
            "observation": {
                "observation_id": self.current_record.get("observation_id"),
                "frame_id": _find_frame(self.current_record, "agentview").get("frame_id")
                if _find_frame(self.current_record, "agentview")
                else None,
            },
            "provenance": dict(provenance),
        }
        viewer_ids = sorted(
            str(item.get("viewer_id"))
            for item in inspector_state.get("viewer_clients", [])
            if isinstance(item, Mapping)
            and item.get("connected") is not False
            and item.get("viewer_id") is not None
        )
        viewer_id = viewer_ids[0] if viewer_ids else None
        inspector_result = self._grasp_inspector.add_pose(
            pose_id=pose_id,
            label=f"R{sum(1 for item in candidates if isinstance(item, Mapping) and item.get('source_backend') == 'refined')}",
            observation_id=self.current_record.get("observation_id"),
            transform_world_from_grip_site=rigid.tolist(),
            source_id=base_grasp_id,
            method=_pose_display_method(method, provenance),
            parent_score=base.get("score"),
        )
        if not inspector_result.get("success"):
            raise GraspInspectorSceneNotReady(inspector_result)
        support_clearance = self._grasp_support_clearance_contract(
            pose_id, inspector_state=inspector_result
        )
        candidate.update(
            {
                key: support_clearance[key]
                for key in (
                    "support_clearance_mm",
                    "support_clearance_required_mm",
                    "support_clearance_status",
                )
            }
        )
        candidates.append(candidate)
        self.latest_grasp.details["grasp_candidates"] = candidates
        artifact = self.root / "perception" / "registered_poses" / f"{_safe_name(pose_id)}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            json.dumps(candidate, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        inspector_view_result: dict[str, Any] = {}
        inspector_capture_result: dict[str, Any] = {}
        inspector_image: Path | None = None
        inspector_view_result = self._grasp_inspector.configure(
            show_anygrasp=True,
            show_graspgenx=True,
            show_refined=True,
            pose_scope="focus",
            focus_pose_id=pose_id,
            camera_preset="pose_jaws",
            **({"viewer_id": viewer_id} if viewer_id is not None else {}),
        )
        if inspector_view_result.get("success"):
            inspector_capture_result, inspector_image = (
                self._grasp_inspector.capture_image(
                    camera="current",
                    **({"viewer_id": viewer_id} if viewer_id is not None else {}),
                )
            )
        return {
            "candidate": candidate,
            "artifact_ref": str(artifact),
            "inspector_registered": bool(inspector_result.get("success")),
            "inspector_result": inspector_result,
            "inspector_view_result": inspector_view_result,
            "inspector_capture_result": inspector_capture_result,
            "inspector_image": inspector_image,
            "inspector_capture_success": bool(
                inspector_capture_result.get("success") and inspector_image
            ),
            "support_clearance": support_clearance,
        }

    def _registration_images(self, registration: Mapping[str, Any]) -> list[Path]:
        images: list[Path] = []
        inspector_image = registration.get("inspector_image")
        if isinstance(inspector_image, Path) and inspector_image.is_file():
            images.append(inspector_image)
        return images

    def capture_grasp_view(
        self,
        *,
        camera: str = "current",
        viewer_id: str | None = None,
    ) -> GatewayResult:
        """Capture the connected browser's current or deterministic Viser view."""

        state = self._validated_grasp_inspector_state()
        if not state.get("success"):
            return GatewayResult(
                False,
                {"kind": "grasp_inspector_capture", **state},
                details=dict(state),
            )
        arguments: dict[str, Any] = {"camera": str(camera)}
        if viewer_id is not None:
            arguments["viewer_id"] = str(viewer_id)
        result, image = self._grasp_inspector.capture_image(**arguments)
        return GatewayResult(
            bool(result.get("success")),
            {"kind": "grasp_inspector_capture", **result},
            images=[image] if image is not None else [],
            details=dict(result),
        )

    def _validated_grasp_inspector_state(
        self, *, expected_scene_mode: str | None = None
    ) -> dict[str, Any]:
        """Reject a viewer that is not bound to this exact current scene."""

        if self.current_record is None:
            return {
                "success": False,
                "code": "observation_required",
                "retryable": True,
                "message": "Call observe before using the 3D grasp inspector.",
            }
        state = self._grasp_inspector.state()
        if not state.get("success"):
            return state
        expected_root = str(self.root)
        expected_observation = str(
            self.current_record.get("observation_id") or ""
        )
        actual_root = str(state.get("episode_root") or "")
        actual_observation = str(state.get("observation_id") or "")
        if (
            actual_root != expected_root
            or actual_observation != expected_observation
        ):
            same_episode = actual_root == expected_root
            return {
                "success": False,
                "code": "stale_scene",
                "retryable": same_episode,
                "message": (
                    "The 3D viewer is still switching from an older observation; "
                    "wait for the current scene and retry."
                    if same_episode
                    else "The 3D viewer is bound to a different episode; open a "
                    "new inspector for the current scene."
                ),
                "expected_episode_root": expected_root,
                "expected_observation_id": expected_observation,
                "viewer_episode_root": actual_root,
                "viewer_observation_id": actual_observation,
            }
        actual_scene_mode = str(state.get("scene_mode") or "")
        if expected_scene_mode is not None and actual_scene_mode != expected_scene_mode:
            return {
                "success": False,
                "code": "wrong_scene_mode",
                "retryable": True,
                "message": (
                    f"The 3D viewer is still in {actual_scene_mode or 'unknown'} mode; "
                    f"wait for the {expected_scene_mode} scene for this observation."
                ),
                "expected_episode_root": expected_root,
                "expected_observation_id": expected_observation,
                "expected_scene_mode": expected_scene_mode,
                "viewer_scene_mode": actual_scene_mode,
            }
        if actual_scene_mode == "proposal" or expected_scene_mode == "proposal":
            actual_proposal_id = str(state.get("proposal_id") or "")
            actual_result_id = str(state.get("result_id") or "")
            if (
                not self.latest_grasp_proposal_id
                or not self.latest_grasp_result_id
                or actual_proposal_id != self.latest_grasp_proposal_id
                or actual_result_id != self.latest_grasp_result_id
            ):
                return {
                    "success": False,
                    "code": "stale_proposal",
                    "retryable": True,
                    "message": (
                        "The 3D viewer is bound to a different grasp proposal "
                        "result for this observation; wait for the exact current "
                        "proposal scene to reload."
                    ),
                    "expected_observation_id": expected_observation,
                    "expected_proposal_id": self.latest_grasp_proposal_id,
                    "expected_result_id": self.latest_grasp_result_id,
                    "viewer_observation_id": actual_observation,
                    "viewer_proposal_id": actual_proposal_id,
                    "viewer_result_id": actual_result_id,
                }
        return state

    def _grasp_support_clearance_contract(
        self,
        grasp_id: str,
        *,
        inspector_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Read one pose's fail-closed support-clearance contract."""

        state = (
            dict(inspector_state)
            if inspector_state is not None
            else self._validated_grasp_inspector_state(
                expected_scene_mode="proposal"
            )
        )
        unknown = {
            "support_clearance_mm": None,
            "support_clearance_required_mm": SUPPORT_CLEARANCE_REQUIRED_MM,
            "support_clearance_status": "unknown",
        }
        if not state.get("success"):
            return {
                **unknown,
                "message": str(
                    state.get("message")
                    or "The current proposal viewer cannot establish support clearance."
                ),
                "inspector_code": state.get("code"),
            }
        pose = next(
            (
                item
                for item in state.get("poses", [])
                if isinstance(item, Mapping)
                and (
                    item.get("source_id") == grasp_id
                    or item.get("pose_id") == grasp_id
                )
            ),
            None,
        )
        if not isinstance(pose, Mapping):
            return {
                **unknown,
                "message": (
                    "The current proposal viewer has no support-clearance result "
                    f"for grasp {grasp_id!r}."
                ),
            }
        try:
            clearance = float(pose.get("support_clearance_mm"))
            required = float(pose.get("support_clearance_required_mm"))
        except (TypeError, ValueError):
            return {
                **unknown,
                "inspector_pose_id": pose.get("pose_id"),
                "message": "Support clearance is unknown; autonomous execution is disabled.",
            }
        if (
            not math.isfinite(clearance)
            or not math.isfinite(required)
            or not math.isclose(
                required, SUPPORT_CLEARANCE_REQUIRED_MM, abs_tol=1e-6
            )
        ):
            return {
                **unknown,
                "inspector_pose_id": pose.get("pose_id"),
                "message": "Support-clearance contract is invalid or unsupported.",
            }
        computed_status = (
            "eligible" if clearance >= SUPPORT_CLEARANCE_REQUIRED_MM else "unsafe"
        )
        if pose.get("support_clearance_status") != computed_status:
            return {
                **unknown,
                "inspector_pose_id": pose.get("pose_id"),
                "message": "Support-clearance status disagrees with the signed distance.",
            }
        return {
            "support_clearance_mm": round(clearance, 3),
            "support_clearance_required_mm": SUPPORT_CLEARANCE_REQUIRED_MM,
            "support_clearance_status": computed_status,
            "inspector_pose_id": pose.get("pose_id"),
        }

    @staticmethod
    def _unsafe_grasp_clearance_result(
        *,
        kind: str,
        grasp_id: str,
        clearance: Mapping[str, Any],
        phase: str,
    ) -> GatewayResult:
        status = str(clearance.get("support_clearance_status") or "unknown")
        measured = clearance.get("support_clearance_mm")
        required = clearance.get("support_clearance_required_mm")
        measurement = (
            f"signed collision-mesh clearance {float(measured):.3f} mm is below "
            f"the required {float(required):.1f} mm"
            if isinstance(measured, (int, float))
            and isinstance(required, (int, float))
            else "signed collision-mesh support clearance is unknown"
        )
        return GatewayResult(
            False,
            {
                "kind": kind,
                "success": False,
                "issue_code": "unsafe_grasp_clearance",
                "retryable": status == "unknown",
                "grasp_id": grasp_id,
                "support_clearance_mm": measured,
                "support_clearance_required_mm": required,
                "support_clearance_status": status,
                "message": (
                    f"Grasp execution is fail-closed during {phase}: {measurement}. "
                    "The pose remains available for registration, display, and inspection."
                ),
            },
            details={"support_clearance": dict(clearance)},
        )

    def select_grasp(self, grasp_id: str, reason: str = "") -> GatewayResult:
        self.ensure_started()
        assert self.current_record is not None and self.writer is not None
        if self.failure_case is not None:
            return self._failed_result("grasp_selection")
        if self.latest_grasp is None or not self.latest_grasp.success:
            return GatewayResult(False, {"kind": "grasp_selection", "success": False, "error": "Call propose_grasps first."})
        if not self._latest_grasp_matches_current_observation():
            return GatewayResult(
                False,
                {
                    "kind": "grasp_selection",
                    "success": False,
                    "reason": "stale_grasp",
                    "error": "The grasp candidates belong to an older observation; propose grasps again.",
                },
            )
        candidates = self.latest_grasp.details.get("grasp_candidates", [])
        selected = next(
            (item for item in candidates if isinstance(item, Mapping) and item.get("id") == grasp_id),
            None,
        )
        if selected is None:
            return GatewayResult(False, {"kind": "grasp_selection", "success": False, "error": f"Unknown grasp_id: {grasp_id}"})
        clearance = self._grasp_support_clearance_contract(grasp_id)
        if clearance["support_clearance_status"] != "eligible":
            return self._unsafe_grasp_clearance_result(
                kind="grasp_selection",
                grasp_id=grasp_id,
                clearance=clearance,
                phase="select_grasp",
            )
        # A checkpoint is bound to one immutable selected pose.  Selecting a
        # different candidate invalidates that checkpoint even when the new
        # selection happens to share the same observation.
        self._pending_grasp_approach = None
        self.selected_grasp = dict(selected)
        self.selected_grasp.update(
            {
                key: clearance[key]
                for key in (
                    "support_clearance_mm",
                    "support_clearance_required_mm",
                    "support_clearance_status",
                )
            }
        )
        self.selected_grasp_observation_id = str(self.current_record.get("observation_id") or "")
        self.writer.record_operator_choice(
            choice_type="grasp",
            choice_id=grasp_id,
            input_frames=self.current_record.get("frame_ids", []),
            metadata={"reason": reason},
        )
        return GatewayResult(
            True,
            {
                "kind": "grasp_selection",
                "success": True,
                "observation_id": self.selected_grasp_observation_id,
                "selected_grasp_id": grasp_id,
                "support_clearance_mm": clearance["support_clearance_mm"],
                "support_clearance_required_mm": clearance[
                    "support_clearance_required_mm"
                ],
                "support_clearance_status": clearance[
                    "support_clearance_status"
                ],
                "message": (
                    "Exact grasp id selected after the current proposal viewer "
                    "reported eligible signed collision-mesh support clearance. "
                    "move_to_selected_grasp will check the same gate again."
                ),
            },
            details={"selected_grasp": self.selected_grasp},
        )

    def _latest_grasp_matches_current_observation(self) -> bool:
        if self.latest_grasp is None or self.current_record is None:
            return False
        binding = self.latest_grasp.details.get("observation", {})
        frame = _find_frame(self.current_record, "agentview")
        return (
            isinstance(binding, Mapping)
            and frame is not None
            and binding.get("observation_id") == self.current_record.get("observation_id")
            and binding.get("frame_id") == frame.get("frame_id")
        )

    # ------------------------------------------------------------------
    # World-changing actions and native task verification
    # ------------------------------------------------------------------

    def _begin_world_action(
        self, *, action: str, stage: str
    ) -> tuple[int | None, GatewayResult | None]:
        """Acquire the episode-wide robot-action slot without blocking."""

        now_monotonic = time.monotonic()
        with self._world_action_lock:
            if self._task_success_latch is not None:
                latch = dict(self._task_success_latch)
                return None, GatewayResult(
                    False,
                    {
                        "kind": action,
                        "success": False,
                        "retryable": False,
                        "reason": "task_already_completed",
                        "issue_code": "task_success_latched",
                        "episode_status": self._episode_status,
                        "message": (
                            "Native task success is already confirmed. No further "
                            "world action was executed; call finish_episode with "
                            "outcome='success'."
                        ),
                    },
                    details={"task_success_latch": latch},
                )
            active = self._active_world_action
            if active is not None:
                duration_s = max(
                    0.0,
                    now_monotonic - float(active["started_monotonic"]),
                )
                active_details = {
                    "active_action": active["action"],
                    "active_stage": active["stage"],
                    "active_started_at_unix_s": active["started_at_unix_s"],
                    "active_duration_s": round(duration_s, 3),
                }
                return None, GatewayResult(
                    False,
                    {
                        "kind": action,
                        "success": False,
                        "retryable": True,
                        "issue_code": "action_in_progress",
                        **active_details,
                        "message": (
                            "A robot world action is already in progress. No second "
                            "world action was started; retry after the active call returns."
                        ),
                    },
                    details={"world_action_guard": active_details},
                )
            self._world_action_seq += 1
            token = self._world_action_seq
            self._active_world_action = {
                "token": token,
                "action": str(action),
                "stage": str(stage),
                "episode_generation": self._episode_generation,
                "started_at_unix_s": time.time(),
                "started_monotonic": now_monotonic,
            }
            return token, None

    def _world_action_generation(self, token: int) -> int:
        with self._world_action_lock:
            active = self._active_world_action
            if active is None or active.get("token") != token:
                raise RuntimeError("world action guard ownership was lost")
            return int(active["episode_generation"])

    def _end_world_action(self, token: int) -> None:
        with self._world_action_lock:
            active = self._active_world_action
            if active is not None and active.get("token") == token:
                self._active_world_action = None

    def _bind_active_world_action(
        self,
        *,
        action_generation: int,
        action_id: str,
        tool_call_id: str,
        grasp_attempt_id: str | None,
    ) -> None:
        """Attach durable action identifiers once the writer has allocated them."""

        with self._world_action_lock:
            active = self._active_world_action
            if (
                active is not None
                and active.get("episode_generation") == action_generation
            ):
                active.update(
                    {
                        "action_id": action_id,
                        "tool_call_id": tool_call_id,
                        "grasp_attempt_id": grasp_attempt_id,
                    }
                )

    def _publish_world_action_outcome(
        self,
        *,
        action_generation: int,
        stage: str,
        action_id: str,
        tool_call_id: str,
        grasp_attempt_id: str | None,
        result: GatewayResult,
    ) -> None:
        """Publish the committed action result before the active slot is released."""

        with self._world_action_lock:
            active = self._active_world_action
            action_name = (
                str(active.get("action"))
                if active is not None
                and active.get("episode_generation") == action_generation
                else "step"
            )
            self._last_world_action_outcome = {
                "status": "completed",
                "action": action_name,
                "stage": stage,
                "episode_generation": action_generation,
                "action_id": action_id,
                "tool_call_id": tool_call_id,
                "grasp_attempt_id": grasp_attempt_id,
                "success": bool(result.success),
                "issue_code": result.text.get("issue_code"),
                "observation_id": result.text.get("observation_id"),
                "remote_completion_unknown": bool(
                    result.text.get("remote_completion_unknown")
                ),
                "completed_at_unix_s": time.time(),
            }

    def _world_action_timeout_report_authority(
        self, *, component: str, code: str
    ) -> dict[str, Any] | None:
        """Return Gateway-owned action state for stale Operator timeout claims."""

        if code not in WORLD_ACTION_TIMEOUT_REPORT_CODES:
            return None
        component = str(component).strip()
        now_monotonic = time.monotonic()
        with self._world_action_lock:
            active = (
                dict(self._active_world_action)
                if self._active_world_action is not None
                else None
            )
            completed = (
                dict(self._last_world_action_outcome)
                if self._last_world_action_outcome is not None
                else None
            )

        def matches(state: Mapping[str, Any] | None) -> bool:
            if state is None:
                return False
            return component in {
                str(state.get("action") or ""),
                str(state.get("stage") or ""),
            }

        if component not in WORLD_ACTION_COMPONENTS:
            return None
        active_match = matches(active)
        completed_match = matches(completed)
        if active is not None and active_match:
            active.pop("token", None)
            started_monotonic = active.pop("started_monotonic", None)
            active["status"] = "in_progress"
            active["issue_code"] = "action_in_progress"
            if isinstance(started_monotonic, (int, float)):
                active["active_duration_s"] = round(
                    max(0.0, now_monotonic - float(started_monotonic)),
                    3,
                )
            return {
                "reason": "world_action_in_progress",
                "authoritative_action": active,
                **(
                    {"last_completed_action": completed}
                    if completed is not None and completed_match
                    else {}
                ),
            }
        if completed is not None and completed_match:
            return {
                "reason": "stale_action_context",
                "authoritative_action": completed,
            }
        return None

    def _invalidate_episode_generation_for_terminal(self) -> dict[str, Any] | None:
        """Fence all in-flight completions before terminal state is written."""

        with self._world_action_lock:
            if not self._terminal_generation_invalidated:
                self._episode_generation += 1
                self._terminal_generation_invalidated = True
                active = self._active_world_action
                if active is not None:
                    self._terminal_active_action = {
                        "active_action": active["action"],
                        "active_stage": active["stage"],
                        "active_started_at_unix_s": active["started_at_unix_s"],
                        "active_duration_s": round(
                            max(
                                0.0,
                                time.monotonic()
                                - float(active["started_monotonic"]),
                            ),
                            3,
                        ),
                        "remote_completion_unknown": True,
                        "canceled": False,
                    }
            return (
                dict(self._terminal_active_action)
                if self._terminal_active_action is not None
                else None
            )

    def _world_action_completion_is_current(self, generation: int) -> bool:
        with self._world_action_lock:
            return (
                generation == self._episode_generation
                and not self._terminal_generation_invalidated
                and not self._closed
                and self._episode_status == "running"
            )

    def _ignored_late_world_action_completion(
        self,
        *,
        stage: str,
        action_id: str,
        tool_call_id: str,
        action_generation: int,
        remote_name: str | None,
        remote: Mapping[str, Any],
    ) -> GatewayResult:
        """Retain host-only evidence without mutating terminal episode state."""

        diagnostic = {
            "schema_version": "openeta.ignored_late_completion.v1",
            "stage": stage,
            "action_id": action_id,
            "tool_call_id": tool_call_id,
            "action_generation": action_generation,
            "terminal_generation": self._episode_generation,
            "episode_status": self._episode_status,
            "remote_tool": remote_name,
            "remote_result": dict(remote),
            "ignored_late_completion": True,
        }
        path = (
            self.root
            / "diagnostics"
            / "ignored_late_completions"
            / f"{_safe_name(action_id or stage)}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(diagnostic, ensure_ascii=False, indent=2, default=str)
            + "\n",
            encoding="utf-8",
        )
        return GatewayResult(
            False,
            {
                "kind": "step",
                "success": False,
                "reason": "episode_terminated",
                "issue_code": "ignored_late_completion",
                "episode_status": self._episode_status,
                "stage": stage,
                "message": (
                    "The remote world action returned after the episode became "
                    "terminal. Its completion was ignored and did not update "
                    "episode state or create a post-action observation."
                ),
            },
            details={
                "ignored_late_completion": diagnostic,
                "diagnostic_ref": str(path),
            },
        )

    def move_to_pose(
        self,
        position_xyz_m: list[float],
        rotation_rpy_deg: list[float],
        reason: str = "",
        tolerance_mm: float = 3.0,
        max_steps: int = 320,
    ) -> GatewayResult:
        """Move directly to a world-frame Panda grip-site pose.

        This is an explicit escape hatch around grasp-candidate conversion. It
        still goes through the normal action log, post-action observation, and
        exact TARGET/ACTUAL execution record consumed by Viser.
        """

        self.ensure_started()
        if self.failure_case is not None:
            return self._failed_result("move_to_pose")
        token, blocked = self._begin_world_action(
            action="move_to_pose", stage="move_to_selected_grasp"
        )
        if blocked is not None:
            return blocked
        assert token is not None
        action_generation = self._world_action_generation(token)
        try:
            try:
                position = _finite_vector3(position_xyz_m, name="position_xyz_m")
                rpy = _finite_vector3(rotation_rpy_deg, name="rotation_rpy_deg")
                transform = np.eye(4, dtype=np.float64)
                transform[:3, :3] = _euler_xyz_matrix_degrees(rpy)
                transform[:3, 3] = np.asarray(position, dtype=np.float64)
                controller = _direct_move_controller_contract(
                    tolerance_mm=tolerance_mm,
                    max_steps=max_steps,
                )
                return self._move_to_world_grip_site(
                    transform,
                    control_mode="direct_world_pose",
                    reason=reason,
                    provenance={"controller": controller},
                    action_generation=action_generation,
                )
            except (GraspPoseRefinementError, ValueError) as exc:
                return GatewayResult(
                    False,
                    {
                        "kind": "move_to_pose",
                        "success": False,
                        "retryable": True,
                        "error": str(exc),
                    },
                )
        finally:
            self._end_world_action(token)

    def move_to_target(
        self,
        target: Mapping[str, Any],
        *,
        reason: str = "",
        views: Sequence[str] | None = None,
    ) -> GatewayResult:
        """Resolve and execute one observation-bound Operator target.

        Point IDs resolve only geometry explicitly marked on the current
        observation. Omitted position, orientation, or gripper state is
        preserved. The low-level controller receives one exact world-frame
        Panda grip-site transform; no object-relative state is introduced.
        """

        explicit_preview_only = bool(
            isinstance(target, Mapping) and target.get("preview_only")
        )
        preview_only = explicit_preview_only
        implicit_close_preview_possible = bool(
            (
                self._close_requires_matching_preview
                or self._close_requires_explicit_preview_commit
            )
            and isinstance(target, Mapping)
            and str(target.get("gripper") or "").strip().lower() == "close"
            and not explicit_preview_only
        )
        default_views_requested = views is None
        requested_views, view_error = self._normalize_operator_views(
            views,
            default=(
                (
                    "agentview",
                    "pointcloud_front",
                    "pointcloud_side",
                )
                if preview_only or implicit_close_preview_possible
                else ("agentview", "wrist")
            ),
        )
        if view_error is not None:
            view_error.text["kind"] = "move_to"
            return view_error
        self.ensure_started()
        if not isinstance(target, Mapping):
            return GatewayResult(False, {
                "kind": "move_to",
                "success": False,
                "reason": "invalid_target",
                "message": "target must be an object with world-frame geometry.",
            })
        # Preview is read-only and remains useful after native success, but
        # every executable move_to variant (including gripper-only calls) must
        # stop at the same one-way success latch before resolving or dispatching
        # any simulator action.
        if (
            self._task_success_latch is not None
            and not preview_only
            and not (
                self._close_requires_explicit_preview_commit
                and implicit_close_preview_possible
            )
        ):
            return GatewayResult(
                False,
                {
                    "kind": "move_to",
                    "success": False,
                    "retryable": False,
                    "reason": "task_already_completed",
                    "issue_code": "task_success_latched",
                    "episode_status": self._episode_status,
                    "message": (
                        "Native task success is already confirmed. No further "
                        "world action was executed; call finish_episode with "
                        "outcome='success'."
                    ),
                },
                details={"task_success_latch": dict(self._task_success_latch)},
            )
        allowed_target_fields = {
            "position_xyz_m",
            "position_point_id",
            "position_delta_mm",
            "position_delta_from_point_id",
            "position_delta_to_point_id",
            "position_from_point_id",
            "position_to_point_id",
            "approach_from_point_id",
            "jaw_toward_point_id",
            "gripper",
            "observation_id",
            "delta_frame",
            "preview_only",
            "rotation_matrix",
            "approach_direction_world",
            "jaw_direction_world",
            "orientation_resolution_policy",
            "pose_id",
            "execute_preview_id",
        }
        unknown_target_fields = sorted(set(target) - allowed_target_fields)
        if unknown_target_fields:
            return GatewayResult(False, {
                "kind": "move_to",
                "success": False,
                "reason": "invalid_target",
                "unknown_target_fields": unknown_target_fields,
                "message": (
                    f"Unknown target fields: {unknown_target_fields}. "
                    "For a gripper-only action use exactly "
                    '{"gripper":"open"} or {"gripper":"close"}.'
                ),
                "retryable": True,
            })
        execute_preview_id = str(
            target.get("execute_preview_id") or ""
        ).strip()
        consumed_preview_id: str | None = None
        if execute_preview_id:
            mixed_fields = sorted(
                key
                for key, value in target.items()
                if key != "execute_preview_id"
                and value is not None
                and value is not False
                and value != ""
            )
            if mixed_fields:
                return GatewayResult(False, {
                    "kind": "move_to",
                    "success": False,
                    "reason": "invalid_target",
                    "retryable": True,
                    "message": (
                        "execute_preview_id commits one frozen candidate and "
                        "cannot be combined with target changes."
                    ),
                })
            if not self._close_requires_explicit_preview_commit:
                return GatewayResult(False, {
                    "kind": "move_to",
                    "success": False,
                    "reason": "invalid_target",
                    "retryable": True,
                    "message": (
                        "execute_preview_id is not enabled by the active "
                        "move_to contract."
                    ),
                })
            pending = self._pending_close_confirmation
            if (
                not isinstance(pending, Mapping)
                or str(pending.get("preview_id") or "") != execute_preview_id
            ):
                return GatewayResult(False, {
                    "kind": "move_to",
                    "success": False,
                    "reason": "stale_preview",
                    "retryable": True,
                    "message": (
                        f"Preview {execute_preview_id!r} is not pending. "
                        "Request a close candidate again before execution."
                    ),
                })
            try:
                pending_base = rigid_transform(
                    pending.get("base_world_from_grip_site"),
                    name="pending close preview base",
                )
                current_base = self._current_world_from_grip_site()
            except (RuntimeError, TypeError, ValueError) as exc:
                self._pending_close_confirmation = None
                return GatewayResult(False, {
                    "kind": "move_to",
                    "success": False,
                    "reason": "stale_preview",
                    "retryable": True,
                    "message": (
                        "The pending close preview cannot be bound to the "
                        f"current grip-site state: {exc}"
                    ),
                })
            if not _move_to_preview_base_matches(
                pending_base,
                current_base,
            ):
                self._pending_close_confirmation = None
                return GatewayResult(False, {
                    "kind": "move_to",
                    "success": False,
                    "reason": "stale_preview",
                    "retryable": True,
                    "message": (
                        f"Preview {execute_preview_id!r} no longer matches "
                        "the current grip-site pose. Preview the close again."
                    ),
                })
            frozen_target = pending.get("target")
            if not isinstance(frozen_target, Mapping):
                self._pending_close_confirmation = None
                return GatewayResult(False, {
                    "kind": "move_to",
                    "success": False,
                    "reason": "stale_preview",
                    "retryable": True,
                    "message": "The pending close preview has no executable target.",
                })
            target = dict(frozen_target)
            consumed_preview_id = execute_preview_id
            # Consume before dispatch so a failed or interrupted physical
            # action cannot be replayed by reusing the same decision token.
            self._pending_close_confirmation = None
            if default_views_requested:
                requested_views, view_error = self._normalize_operator_views(
                    None,
                    default=("agentview", "wrist"),
                )
                if view_error is not None:
                    view_error.text["kind"] = "move_to"
                    return view_error
            if self._task_success_latch is not None:
                return GatewayResult(
                    False,
                    {
                        "kind": "move_to",
                        "success": False,
                        "retryable": False,
                        "reason": "task_already_completed",
                        "issue_code": "task_success_latched",
                        "episode_status": self._episode_status,
                        "message": (
                            "Native task success is already confirmed. No "
                            "previewed action was executed; call finish_episode "
                            "with outcome='success'."
                        ),
                    },
                    details={
                        "task_success_latch": dict(self._task_success_latch)
                    },
                )
        expected = target.get("observation_id")
        actual = self.current_record.get("observation_id") if self.current_record else None
        if expected is not None and str(expected) != str(actual):
            return GatewayResult(False, {
                "kind": "move_to",
                "success": False,
                "reason": "stale_target",
                "expected_observation_id": str(expected),
                "current_observation_id": str(actual or ""),
                "retryable": True,
            })
        try:
            position = target.get("position_xyz_m")
            position_point_id = str(target.get("position_point_id") or "").strip()
            position_delta_raw = target.get("position_delta_mm")
            legacy_point_pair = bool(
                target.get("position_delta_from_point_id")
                or target.get("position_delta_to_point_id")
            )
            natural_point_pair = bool(
                target.get("position_from_point_id")
                or target.get("position_to_point_id")
            )
            if legacy_point_pair and natural_point_pair:
                raise ValueError(
                    "use one point-pair naming contract, not both legacy "
                    "position_delta_* and position_from/position_to fields"
                )
            position_delta_from_point_id = str(
                target.get("position_from_point_id")
                or target.get("position_delta_from_point_id")
                or ""
            ).strip()
            position_delta_to_point_id = str(
                target.get("position_to_point_id")
                or target.get("position_delta_to_point_id")
                or ""
            ).strip()
            has_point_delta = bool(
                position_delta_from_point_id or position_delta_to_point_id
            )
            if has_point_delta and not (
                position_delta_from_point_id and position_delta_to_point_id
            ):
                raise ValueError(
                    "point-defined translation requires both "
                    "position_from_point_id and position_to_point_id"
                )
            position_delta = None
            if position_delta_raw is not None:
                delta_frame = str(target.get("delta_frame") or "world").strip().lower()
                supported_delta_frames = {"world"}
                if self._grip_site_position_delta_enabled:
                    supported_delta_frames.add("grip_site")
                if delta_frame not in supported_delta_frames:
                    supported = ", ".join(
                        repr(frame) for frame in sorted(supported_delta_frames)
                    )
                    raise ValueError(
                        f"target.delta_frame must be one of {supported}"
                    )
                position_delta = _finite_vector3(
                    position_delta_raw,
                    name="target.position_delta_mm",
                )
            elif has_point_delta:
                delta_frame = "world"
            else:
                delta_frame = None
            position_sources = sum(
                (
                    bool(position_point_id),
                    position is not None,
                    position_delta is not None,
                    has_point_delta,
                )
            )
            if position_sources > 1:
                raise ValueError(
                    "use exactly one of position_point_id, position_xyz_m, or "
                    "position_delta_mm, or the paired "
                    "position_from_point_id/position_to_point_id"
                )
            approach_point_id = str(
                target.get("approach_from_point_id") or ""
            ).strip()
            jaw_point_id = str(target.get("jaw_toward_point_id") or "").strip()
            point_ids = [
                point_id
                for point_id in (
                    position_point_id,
                    position_delta_from_point_id,
                    position_delta_to_point_id,
                    approach_point_id,
                    jaw_point_id,
                )
                if point_id
            ]
            missing_point_ids = [
                point_id
                for point_id in point_ids
                if point_id not in self._point_marks_3d
            ]
            if missing_point_ids:
                raise ValueError(
                    "missing marked point ids: "
                    + ", ".join(sorted(set(missing_point_ids)))
                )
            if position_point_id:
                position = self._point_marks_3d[position_point_id]["xyz_m"]
            if position is None:
                pose_id = target.get("pose_id")
                if pose_id and self.latest_grasp is not None:
                    candidate = self._current_grasp_candidate(str(pose_id))
                    if candidate is not None:
                        transform = self._candidate_world_from_grip_site(candidate)
                        position = transform[:3, 3].tolist()
                        target = {**dict(target), "rotation_matrix": transform[:3, :3].tolist()}
            requested_gripper = target.get("gripper")
            if requested_gripper is not None:
                requested_gripper = str(requested_gripper).strip().lower()
                if requested_gripper not in {"open", "close"}:
                    raise ValueError("target.gripper must be 'open' or 'close'")
            has_orientation_ids = bool(approach_point_id or jaw_point_id)
            if has_orientation_ids and not (
                approach_point_id and jaw_point_id
            ):
                raise ValueError(
                    "point-defined orientation requires both "
                    "approach_from_point_id and jaw_toward_point_id; position "
                    "may be omitted to rotate around the current grip_site"
                )
            has_direct_orientation = (
                target.get("rotation_matrix") is not None
                or target.get("approach_direction_world") is not None
                or target.get("jaw_direction_world") is not None
            )
            if (
                position is None
                and position_delta is None
                and not has_point_delta
                and not has_orientation_ids
                and not has_direct_orientation
                and requested_gripper is None
            ):
                raise ValueError(
                    "move_to requires at least one position, orientation, or gripper change"
                )
            current_transform = self._current_world_from_grip_site()
            base_actual_position = current_transform[:3, 3].tolist()
            position_delta_world = None
            if has_point_delta:
                from_xyz = np.asarray(
                    self._point_marks_3d[position_delta_from_point_id]["xyz_m"],
                    dtype=np.float64,
                )
                to_xyz = np.asarray(
                    self._point_marks_3d[position_delta_to_point_id]["xyz_m"],
                    dtype=np.float64,
                )
                position_delta = ((to_xyz - from_xyz) * 1000.0).tolist()
            if position_delta is not None:
                position_delta_world = np.asarray(
                    position_delta,
                    dtype=np.float64,
                )
                if delta_frame == "grip_site":
                    position_delta_world = (
                        current_transform[:3, :3] @ position_delta_world
                    )
                position = (
                    current_transform[:3, 3]
                    + position_delta_world / 1000.0
                ).tolist()
            if position is None:
                position = current_transform[:3, 3].tolist()
            position = _finite_vector3(position, name="target.position_xyz_m")
            resolution = str(target.get("orientation_resolution_policy") or "")
            rotation = target.get("rotation_matrix")
            constraints = (
                ["position_point_id"]
                if position_point_id
                else (
                    ["position_xyz_m"]
                    if target.get("position_xyz_m") is not None
                    else (
                        [
                            (
                                "position_delta_from_point_id"
                                if legacy_point_pair
                                else "position_from_point_id"
                            ),
                            (
                                "position_delta_to_point_id"
                                if legacy_point_pair
                                else "position_to_point_id"
                            ),
                        ]
                        if has_point_delta
                        else (
                            ["position_delta_mm"]
                            if position_delta is not None
                            else []
                        )
                    )
                )
            )
            inherited: list[str] = []
            if has_orientation_ids:
                if rotation is not None or has_direct_orientation:
                    raise ValueError(
                        "use either three point IDs or direct orientation fields, not both"
                    )
                pose_marks = self._point_marks_3d
                pose_position_point_id = position_point_id
                if not pose_position_point_id:
                    pose_position_point_id = "__current_grip_site__"
                    pose_marks = {
                        **self._point_marks_3d,
                        pose_position_point_id: {
                            "point_id": pose_position_point_id,
                            "observation_id": actual,
                            "xyz_m": list(position),
                        },
                    }
                marked_pose = pose_from_points(
                    pose_marks,
                    position_point_id=pose_position_point_id,
                    approach_from_point_id=approach_point_id,
                    jaw_toward_point_id=jaw_point_id,
                )
                rotation = marked_pose["rotation_matrix"]
                constraints.extend(
                    [
                        "approach_from_point_id",
                        "jaw_toward_point_id",
                    ]
                )
                if position_point_id:
                    resolution = resolution or "three_marked_points"
                else:
                    inherited.append("current_position")
                    resolution = (
                        resolution
                        or "two_marked_direction_anchors_current_position"
                    )
            direct_approach = target.get("approach_direction_world")
            direct_jaw = target.get("jaw_direction_world")
            if rotation is None and (
                direct_approach is not None or direct_jaw is not None
            ):
                current_rotation = current_transform[:3, :3]
                if direct_approach is not None:
                    approach = np.asarray(
                        _finite_vector3(
                            direct_approach,
                            name="approach_direction_world",
                        ),
                        dtype=np.float64,
                    )
                    z_axis = approach / np.linalg.norm(approach)
                    constraints.append("approach_direction")
                else:
                    z_axis = np.asarray(
                        current_rotation[:, 2],
                        dtype=np.float64,
                    )
                    inherited.append("approach_axis_from_current")

                if direct_jaw is not None:
                    x_hint = np.asarray(
                        _finite_vector3(
                            direct_jaw,
                            name="jaw_direction_world",
                        ),
                        dtype=np.float64,
                    )
                    constraints.append("jaw_direction")
                else:
                    x_hint = np.asarray(
                        current_rotation[:, 0],
                        dtype=np.float64,
                    )
                    inherited.append("jaw_roll_from_current")

                x_projected = (
                    x_hint - z_axis * float(np.dot(x_hint, z_axis))
                )
                if np.linalg.norm(x_projected) < 1e-6:
                    if direct_jaw is not None:
                        raise ValueError(
                            "jaw_direction_world must not be parallel to the "
                            "resolved approach direction"
                        )
                    x_projected = (
                        current_rotation[:, 1]
                        - z_axis
                        * float(np.dot(current_rotation[:, 1], z_axis))
                    )
                x_axis = x_projected / np.linalg.norm(x_projected)
                y_axis = np.cross(z_axis, x_axis)
                y_axis /= np.linalg.norm(y_axis)
                x_axis = np.cross(y_axis, z_axis)
                rotation = np.column_stack(
                    [x_axis, y_axis, z_axis]
                ).tolist()
                if direct_approach is not None and direct_jaw is not None:
                    resolution = resolution or "semantic_directions"
                elif direct_approach is not None:
                    resolution = (
                        resolution or "approach_nearest_current_roll"
                    )
                else:
                    resolution = (
                        resolution or "jaw_preserve_current_approach"
                    )
            if rotation is None:
                rotation = current_transform[:3, :3].tolist()
                inherited.append("current_orientation")
                resolution = resolution or "preserve_current"
            rigid = rigid_transform(
                np.block([
                    [np.asarray(rotation, dtype=np.float64), np.asarray(position, dtype=np.float64).reshape(3, 1)],
                    [np.zeros((1, 3)), np.ones((1, 1))],
                ]),
                name="resolved_move_target",
            )
            rpy = _rotation_to_euler_degrees(rigid[:3, :3].tolist())
            if rpy is None:
                raise ValueError("resolved target orientation is not convertible")
            motion_requested = bool(
                target.get("position_xyz_m") is not None
                or position_point_id
                or position_delta is not None
                or has_point_delta
                or has_orientation_ids
                or has_direct_orientation
            )
            close_confirmation_key = None
            close_confirmation_consumed = False
            implicit_close_preview = False
            preview_id = None
            if (
                (
                    self._close_requires_matching_preview
                    or self._close_requires_explicit_preview_commit
                )
                and requested_gripper == "close"
            ):
                close_confirmation_key = _move_to_close_confirmation_key(
                    observation_id=str(actual or ""),
                    target_position_xyz_m=position,
                    target_rotation_matrix=rigid[:3, :3],
                    current_world_from_grip_site=current_transform,
                )
                if self._close_requires_explicit_preview_commit:
                    if consumed_preview_id is not None:
                        close_confirmation_consumed = True
                        preview_id = consumed_preview_id
                    else:
                        pending_key = (
                            str(
                                self._pending_close_confirmation.get("key")
                                or ""
                            )
                            if self._pending_close_confirmation is not None
                            else ""
                        )
                        if (
                            not explicit_preview_only
                            and pending_key == close_confirmation_key
                        ):
                            return GatewayResult(
                                False,
                                {
                                    "kind": "move_to",
                                    "success": False,
                                    "reason": "preview_decision_required",
                                    "retryable": True,
                                    "message": (
                                        "The unchanged close candidate is "
                                        "already previewed. Execute its prior "
                                        "preview_id or issue a corrected "
                                        "move_to."
                                    ),
                                },
                            )
                        preview_id = _move_to_close_preview_id(
                            target_position_xyz_m=position,
                            target_rotation_matrix=rigid[:3, :3],
                            current_world_from_grip_site=current_transform,
                        )
                        preview_only = True
                        implicit_close_preview = not explicit_preview_only
                        frozen_target: dict[str, Any] = {
                            "gripper": "close",
                        }
                        if motion_requested:
                            frozen_target["position_xyz_m"] = list(position)
                        if has_orientation_ids or has_direct_orientation:
                            frozen_target["rotation_matrix"] = (
                                rigid[:3, :3].tolist()
                            )
                        self._pending_close_confirmation = {
                            "key": close_confirmation_key,
                            "preview_id": preview_id,
                            "observation_id": str(actual or ""),
                            "base_world_from_grip_site": (
                                current_transform.tolist()
                            ),
                            "target": frozen_target,
                            "resolved_target_pose": {
                                "position_xyz_m": list(position),
                                "rotation_matrix": rigid[:3, :3].tolist(),
                            },
                        }
                else:
                    pending_key = (
                        str(self._pending_close_confirmation.get("key") or "")
                        if self._pending_close_confirmation is not None
                        else ""
                    )
                    if explicit_preview_only:
                        self._pending_close_confirmation = {
                            "key": close_confirmation_key,
                            "observation_id": str(actual or ""),
                        }
                    elif pending_key == close_confirmation_key:
                        close_confirmation_consumed = True
                        self._pending_close_confirmation = None
                        if default_views_requested:
                            requested_views, view_error = (
                                self._normalize_operator_views(
                                    None,
                                    default=("agentview", "wrist"),
                                )
                            )
                            if view_error is not None:
                                view_error.text["kind"] = "move_to"
                                return view_error
                    else:
                        preview_only = True
                        implicit_close_preview = True
                        self._pending_close_confirmation = {
                            "key": close_confirmation_key,
                            "observation_id": str(actual or ""),
                        }
            if preview_only:
                candidate_contacts, candidate_boxes = (
                    self._candidate_pad_geometry_for_pose(
                        position_xyz_m=position,
                        rotation_matrix=rigid[:3, :3],
                        requested_gripper=requested_gripper,
                    )
                )
                candidate_sweep_start_contacts = None
                candidate_sweep_start_boxes = None
                candidate_capture_corridor_boxes = None
                if requested_gripper == "close":
                    (
                        candidate_sweep_start_contacts,
                        open_candidate_boxes,
                    ) = (
                        self._candidate_pad_geometry_for_pose(
                            position_xyz_m=position,
                            rotation_matrix=rigid[:3, :3],
                            requested_gripper="open",
                        )
                    )
                    if self._candidate_pad_swept_footprint:
                        candidate_sweep_start_boxes = open_candidate_boxes
                    if self._candidate_pad_capture_corridor:
                        candidate_capture_corridor_boxes = open_candidate_boxes
                self._ensure_operator_pointcloud_views()
                preview_artifact_id = _pose_preview_artifact_id(
                    observation_id=str(actual or ""),
                    target_position_xyz_m=position,
                    target_rotation_matrix=rigid[:3, :3],
                    actual_world_from_grip_site=current_transform,
                    requested_gripper=requested_gripper,
                    target_pad_contact_centers_world_m=candidate_contacts,
                    target_pad_sweep_start_centers_world_m=(
                        candidate_sweep_start_contacts
                    ),
                    target_pad_sweep_start_boxes=(
                        candidate_sweep_start_boxes
                    ),
                    target_pad_capture_corridor_boxes=(
                        candidate_capture_corridor_boxes
                    ),
                    identifier_prefix=(
                        "candidate"
                        if self._pose_candidate_identifier_namespace
                        else "preview"
                    ),
                )
                preview_root = (
                    self.root
                    / "pointcloud_views"
                    / str(actual)
                    / "pose_previews"
                    / preview_artifact_id
                )
                if (
                    self._candidate_frame_pose_preview_views
                    and self._operator_pointcloud_points_world is not None
                    and self._operator_pointcloud_colors_rgb is not None
                ):
                    preview_views = render_pose_candidate_frame_preview_views(
                        self._operator_pointcloud_points_world,
                        self._operator_pointcloud_colors_rgb,
                        output_root=preview_root,
                        target_position_xyz_m=position,
                        target_rotation_matrix=rigid[:3, :3],
                        target_pad_contact_centers_world_m=candidate_contacts,
                        target_pad_sweep_start_centers_world_m=(
                            candidate_sweep_start_contacts
                        ),
                        target_pad_boxes=candidate_boxes,
                        target_pad_sweep_start_boxes=(
                            candidate_sweep_start_boxes
                        ),
                        target_pad_capture_corridor_boxes=(
                            candidate_capture_corridor_boxes
                        ),
                        axis_label_semantics=(
                            (
                                "jaw_lat_app_v3"
                                if self._candidate_frame_pose_preview_mode
                                == "candidate_gripper_jaw_app_lat_v3"
                                else "jaw_lat_app_v2"
                            )
                            if self._candidate_frame_pose_preview_mode
                            in {
                                "candidate_gripper_jaw_app_lat_v2",
                                "candidate_gripper_jaw_app_lat_v3",
                            }
                            else "local_xyz_v1"
                        ),
                        visual_mode=(
                            self._candidate_frame_pose_preview_visual_mode
                        ),
                        markable=self._all_returned_images_markable,
                    )
                else:
                    preview_views = annotate_pose_preview_views(
                        self._pointcloud_views,
                        output_root=preview_root,
                        target_position_xyz_m=position,
                        target_rotation_matrix=rigid[:3, :3],
                        target_pad_contact_centers_world_m=candidate_contacts,
                        target_pad_sweep_start_centers_world_m=(
                            candidate_sweep_start_contacts
                        ),
                        target_pad_boxes=candidate_boxes,
                        actual_position_xyz_m=current_transform[:3, 3],
                        actual_rotation_matrix=current_transform[:3, :3],
                        compact_labels=self._compact_pose_preview_overlay,
                    )
                preview_substituted_from = [
                    name for name in requested_views if name not in preview_views
                ]
                preview_returned_views = [
                    name for name in requested_views if name in preview_views
                ]
                preview_images = [
                    preview_views[name] for name in preview_returned_views
                ]
                agentview_preview_path = None
                if (
                    self._agentview_pose_preview_overlay
                    and "agentview" in requested_views
                ):
                    agentview_preview_path = self._render_agentview_pose_preview(
                        target_position_xyz_m=position,
                        target_rotation_matrix=rigid[:3, :3],
                        target_pad_contact_centers_world_m=candidate_contacts,
                        target_pad_boxes=candidate_boxes,
                        output_root=preview_root,
                    )
                    if agentview_preview_path is not None:
                        preview_substituted_from = [
                            name
                            for name in preview_substituted_from
                            if name != "agentview"
                        ]
                        if self._operator_source_view_only_images:
                            preview_returned_views.insert(
                                0,
                                "agentview_candidate_pose_overlay"
                            )
                            preview_images.insert(0, agentview_preview_path)
                local_inspection_path = None
                candidate_frame_inspection_path = None
                if (
                    self._candidate_frame_pose_inspection
                    and not self._operator_source_view_only_images
                ):
                    candidate_frame_inspection_path = (
                        render_pose_candidate_frame_inspection(
                            self._pointcloud_views,
                            output_path=(
                                preview_root
                                / "candidate_frame_inspection.png"
                            ),
                            target_position_xyz_m=position,
                            target_rotation_matrix=rigid[:3, :3],
                            target_pad_boxes=candidate_boxes,
                        )
                    )
                    if candidate_frame_inspection_path is not None:
                        preview_images.insert(0, candidate_frame_inspection_path)
                if (
                    not self._operator_source_view_only_images
                    and (
                        preview_substituted_from
                        or agentview_preview_path is not None
                    )
                ):
                    preview_contact_sheet = _render_named_image_contact_sheet(
                        preview_views,
                        output_path=(
                            preview_root
                            / "pointcloud_contact_sheet.png"
                        ),
                        order=(
                            "pointcloud_top",
                            "pointcloud_front",
                            "pointcloud_side",
                        ),
                    )
                    preview_returned_views = []
                    preview_images = []
                    if agentview_preview_path is not None:
                        preview_returned_views.append(
                            "agentview_candidate_pose_overlay"
                        )
                        preview_images.append(agentview_preview_path)
                    if preview_substituted_from:
                        preview_returned_views.append("pointcloud_contact_sheet")
                        preview_images.append(preview_contact_sheet)
                if self._all_returned_images_markable:
                    view_namespace = (
                        "candidate"
                        if self._pose_candidate_identifier_namespace
                        else "preview"
                    )
                    stable_refs = {
                        name: f"{view_namespace}:{preview_artifact_id}:{name}"
                        for name in preview_returned_views
                    }
                    candidate_refs = [
                        stable_refs[name]
                        for name in (
                            "pointcloud_front",
                            "pointcloud_side",
                        )
                        if name in stable_refs
                    ]
                    candidate_radius_m = (
                        0.05
                        if self._candidate_frame_pose_preview_visual_mode
                        == "candidate_corridor_local_v2"
                        else 0.06
                    )
                    for name, path in zip(
                        preview_returned_views,
                        preview_images,
                    ):
                        ref = stable_refs[name]
                        if name == "agentview_candidate_pose_overlay":
                            self._derived_mark_views[ref] = {
                                "kind": "rgbd_camera_overlay",
                                "camera_id": "agentview",
                                "path": str(path),
                                "source_observation_id": str(actual),
                                "preview_id": preview_artifact_id,
                            }
                        elif (
                            self._candidate_frame_pose_preview_views
                            and name
                            in {"pointcloud_front", "pointcloud_side"}
                        ):
                            self._derived_mark_views[ref] = {
                                "kind": "candidate_frame",
                                "base_view": name,
                                "path": str(path),
                                "source_observation_id": str(actual),
                                "preview_id": preview_artifact_id,
                                "candidate_origin_world_m": list(position),
                                "candidate_rotation_world":
                                    rigid[:3, :3].tolist(),
                                "extent_m": candidate_radius_m,
                                "image_size_xy": [384, 384],
                                "vertical_screen_sign": (
                                    -1
                                    if (
                                        name == "pointcloud_front"
                                        and self._candidate_frame_pose_preview_mode
                                        == "candidate_gripper_jaw_app_lat_v3"
                                    )
                                    else 1
                                ),
                                "sibling_view_refs": candidate_refs,
                            }
                    preview_returned_views = [
                        stable_refs[name] for name in preview_returned_views
                    ]
                if position_point_id:
                    self._active_grip_site_target = {
                        "point_id": position_point_id,
                        "position_xyz_m": list(position),
                        "rotation_matrix": rigid[:3, :3].tolist(),
                        "pad_contact_centers_world_m": candidate_contacts,
                    }
                    self._render_current_point_views()
                result = GatewayResult(
                    True,
                    {
                        "kind": "move_to",
                        "success": True,
                        "preview_only": True,
                        "message": (
                            "Candidate pose resolved and rendered without "
                            "moving the robot."
                        ),
                    },
                    images=preview_images,
                    details={
                        "preview_artifact_id": preview_artifact_id,
                        "pose_preview_pointcloud_frame": (
                            self._candidate_frame_pose_preview_mode
                            if self._candidate_frame_pose_preview_views
                            else "world_orthographic_v1"
                        ),
                        "pose_preview_paths": {
                            name: str(path.relative_to(self.root))
                            for name, path in preview_views.items()
                        },
                        "local_inspection_path": (
                            str(local_inspection_path.relative_to(self.root))
                            if local_inspection_path is not None
                            else None
                        ),
                        "candidate_frame_inspection_path": (
                            str(
                                candidate_frame_inspection_path.relative_to(
                                    self.root
                                )
                            )
                            if candidate_frame_inspection_path is not None
                            else None
                        ),
                        "agentview_candidate_pose_overlay_path": (
                            str(agentview_preview_path.relative_to(self.root))
                            if agentview_preview_path is not None
                            else None
                        ),
                        "preview_view_substitution": {
                            "requested_unavailable_views": (
                                preview_substituted_from
                            ),
                            "returned_view": (
                                "pointcloud_contact_sheet"
                                if preview_substituted_from
                                else None
                            ),
                        },
                        "gripper_preview": {
                            "requested_state": requested_gripper,
                            "geometry_mode": (
                                f"nominal_{requested_gripper}_footprint"
                                if requested_gripper in {"open", "close"}
                                else "measured_live_footprint"
                            ),
                            "action_executed": False,
                            "contact_predicted": False,
                            "retention_predicted": False,
                            "closing_sweep_rendered": requested_gripper == "close",
                            "sweep_is_geometric_only": requested_gripper == "close",
                        },
                        "close_confirmation": {
                            "policy": (
                                "frozen_preview_id_commit_v1"
                                if self._close_requires_explicit_preview_commit
                                else (
                                    "matching_candidate_preview_then_repeat_v1"
                                    if self._close_requires_matching_preview
                                    else "not_required"
                                )
                            ),
                            "implicit_preview": implicit_close_preview,
                            "confirmation_key": close_confirmation_key,
                            "preview_id": preview_id,
                        },
                    },
                )
                if preview_id is not None:
                    result.text["preview_id"] = preview_id
                result.text["gripper_preview"] = dict(
                    result.details.get("gripper_preview", {})
                )
                if requested_gripper is not None:
                    # Preview accepts the exact execution target but never
                    # performs the gripper action.
                    result.text["gripper_requested"] = requested_gripper
                    result.text["gripper_executed"] = False
                    result.text["gripper_skip_reason"] = "preview_only"
                    result.text["gripper_preview_ignored"] = False
            elif motion_requested:
                if not close_confirmation_consumed:
                    self._pending_close_confirmation = None
                result = self.move_to_pose(position, list(rpy), reason=reason)
                # A latched native success is a terminal episode fact, not a
                # controller miss. Preserve that explicit reason through the
                # compact Operator result instead of reinterpreting the guard
                # response as ``motion_status=failed``.
                if (
                    not result.success
                    and result.text.get("reason") == "task_already_completed"
                ):
                    result.text["kind"] = "move_to"
                    result.text["requested_views"] = requested_views
                    result.text["returned_views"] = []
                    return result
            else:
                if requested_gripper != "close":
                    self._pending_close_confirmation = None
                result = GatewayResult(
                    True,
                    {
                        "kind": "move_to",
                        "success": True,
                        "motion_skipped": True,
                        "message": "Pose preserved; executing gripper-only move_to.",
                    },
                )
            if requested_gripper is not None and not preview_only:
                result.text["gripper_requested"] = requested_gripper
                if result.success:
                    gripper_result = self.step(
                        "open_gripper"
                        if requested_gripper == "open"
                        else "close_gripper"
                    )
                    result.text["gripper_executed"] = True
                    if close_confirmation_consumed:
                        result.details["close_confirmation"] = {
                            "policy": (
                                "frozen_preview_id_commit_v1"
                                if self._close_requires_explicit_preview_commit
                                else "matching_candidate_preview_then_repeat_v1"
                            ),
                            "consumed": True,
                            "confirmation_key": close_confirmation_key,
                            "preview_id": preview_id,
                        }
                    result.text["gripper_result"] = dict(gripper_result.text)
                    result.details["gripper_result"] = gripper_result.details
                    # The outer move_to result must represent the final state after
                    # every requested sub-action. For position+gripper this replaces
                    # the intermediate post-motion images with post-gripper images;
                    # for gripper-only it fixes the historical empty response.
                    result.images = list(gripper_result.images)
                    if not gripper_result.success:
                        # A position move can itself satisfy the native LIBERO
                        # predicate.  In that case the one-way success latch
                        # intentionally rejects the second sub-action, but that
                        # rejection must not turn the already-completed
                        # combination into a failed public move_to result.
                        # Preserve the terminal action outcome and retain the
                        # skipped gripper result only in host details.
                        if (
                            self._task_success_latch is not None
                            and gripper_result.text.get("reason")
                            == "task_already_completed"
                        ):
                            result.text["gripper_executed"] = False
                            result.text["gripper_skip_reason"] = (
                                "task_already_completed"
                            )
                            result.details["gripper_result"] = (
                                gripper_result.details
                            )
                        else:
                            result.success = False
                            result.text["success"] = False
                            result.text["issue_code"] = (
                                gripper_result.text.get("issue_code")
                                or gripper_result.text.get("reason")
                                or "gripper_action_failed"
                            )
                            gripper_message = (
                                gripper_result.text.get("message")
                                or gripper_result.text.get("error")
                            )
                            if gripper_message:
                                result.text["message"] = str(gripper_message)
                            if gripper_result.text.get("retryable") is not None:
                                result.text["retryable"] = bool(
                                    gripper_result.text["retryable"]
                                )
                else:
                    result.text["gripper_executed"] = False
                    result.text["gripper_skip_reason"] = "motion_failed"
                    result.text["message"] = (
                        f"{result.text.get('message', 'Motion failed.')} "
                        f"The requested gripper {requested_gripper} action was "
                        "not executed because the preceding motion failed."
                    )
            contact_overlay_paths: dict[str, Path] = {}
            if (
                motion_requested
                and not preview_only
                and not result.success
            ):
                remote_payload = result.details.get("remote")
                if isinstance(remote_payload, Mapping):
                    contact_overlay_paths = (
                        self._render_current_mujoco_contact_markers(
                            remote_payload
                        )
                    )
                    if contact_overlay_paths:
                        result.details["mujoco_contact_overlay_paths"] = {
                            name: str(path.relative_to(self.root))
                            for name, path in contact_overlay_paths.items()
                        }
            # Bind the exact commanded grip-site center to the final
            # post-action observation. A combined move+gripper call captures a
            # second observation, so this must happen after every requested
            # sub-action rather than immediately after the motion primitive.
            # The orthographic views then show TARGET, ACTUAL, their residual
            # line, and measured finger-pad centers. This visualizes explicit
            # command/state only; it does not infer collision or reachability.
            if motion_requested and not preview_only:
                self._pointcloud_target_grip_site_xyz = list(position)
                if position_point_id:
                    candidate_contacts, _candidate_boxes = (
                        self._candidate_pad_geometry_for_pose(
                            position_xyz_m=position,
                            rotation_matrix=rigid[:3, :3],
                        )
                    )
                    self._active_grip_site_target = {
                        "point_id": position_point_id,
                        "position_xyz_m": list(position),
                        "rotation_matrix": rigid[:3, :3].tolist(),
                        "pad_contact_centers_world_m": candidate_contacts,
                    }
            # Select the requested views from the final observation, regardless
            # of whether the action was pose-only, gripper-only, or combined.
            execution_return_views = list(requested_views)
            move_feedback_view: str | None = None
            execution_returned_views_override: list[str] | None = None
            if (
                not preview_only
                and default_views_requested
                and motion_requested
                and self._not_reached_default_view_policy
                in {
                    "agentview_plus_residual_projection_v1",
                    "agentview_plus_residual_crop_v1",
                    "agentview_plus_actual_pad_crop_v1",
                    "agentview_plus_actual_pad_contact_corridor_v1",
                    "agentview_plus_actual_pad_closing_corridor_v1",
                }
                and _move_to_motion_status(
                    result.text,
                    success=result.success,
                    motion_requested=motion_requested,
                )
                in {"stalled", "not_converged", "failed"}
            ):
                try:
                    actual_pose_for_views = (
                        self._current_world_from_grip_site()
                    )
                    residual_for_views = (
                        np.asarray(position, dtype=np.float64)
                        - actual_pose_for_views[:3, 3]
                    )
                    pad_contacts_for_views = (
                        self._current_finger_pad_contacts_for_render()
                    )
                    pad_separation_xy = None
                    if (
                        self._not_reached_default_view_policy
                        in {
                            "agentview_plus_actual_pad_crop_v1",
                            "agentview_plus_actual_pad_contact_corridor_v1",
                            "agentview_plus_actual_pad_closing_corridor_v1",
                        }
                        and pad_contacts_for_views is not None
                    ):
                        pad_separation = np.asarray(
                            pad_contacts_for_views[1],
                            dtype=np.float64,
                        ) - np.asarray(
                            pad_contacts_for_views[0],
                            dtype=np.float64,
                        )
                        pad_separation_xy = np.abs(pad_separation[:2])
                    # Both front and side expose world Z. Choose the view that
                    # separates the two live pads most clearly.
                    if (
                        pad_separation_xy is not None
                        and float(np.max(pad_separation_xy)) > 1e-6
                    ):
                        residual_view = (
                            "pointcloud_front"
                            if float(pad_separation_xy[0])
                            >= float(pad_separation_xy[1])
                            else "pointcloud_side"
                        )
                    else:
                        residual_view = (
                            "pointcloud_front"
                            if abs(float(residual_for_views[0]))
                            >= abs(float(residual_for_views[1]))
                            else "pointcloud_side"
                        )
                    execution_return_views = [
                        "agentview",
                        residual_view,
                    ]
                    if (
                        self._not_reached_default_view_policy
                        in {
                            "agentview_plus_residual_crop_v1",
                            "agentview_plus_actual_pad_crop_v1",
                            "agentview_plus_actual_pad_contact_corridor_v1",
                            "agentview_plus_actual_pad_closing_corridor_v1",
                        }
                    ):
                        move_feedback_view = residual_view
                    result.details["default_view_selection"] = {
                        "policy": self._not_reached_default_view_policy,
                        "replaced_view": "wrist",
                        "selected_view": residual_view,
                    }
                except (RuntimeError, TypeError, ValueError):
                    # Visual evidence selection must never alter action
                    # outcome. Fall back to the ordinary agentview+wrist pair
                    # if the measured pose is unavailable.
                    execution_return_views = list(requested_views)
            if (
                not preview_only
                and self._views_need_pointcloud(execution_return_views)
            ):
                self._ensure_operator_pointcloud_views()
            if not preview_only:
                if move_feedback_view is not None:
                    source_view = self._pointcloud_views.get(
                        move_feedback_view
                    )
                    agentview_path = (
                        contact_overlay_paths.get("agentview")
                        or self._operator_view_path("agentview")
                    )
                    actual_pose_for_feedback = (
                        self._current_world_from_grip_site()
                    )
                    feedback_path = None
                    if source_view is not None:
                        suffix = move_feedback_view.removeprefix(
                            "pointcloud_"
                        )
                        feedback_output_path = (
                            self.root
                            / "pointcloud_views"
                            / str(
                                self.current_record.get(
                                    "observation_id"
                                )
                            )
                            / f"move_feedback_{suffix}.png"
                        )
                        if (
                            self._not_reached_default_view_policy
                            in {
                                "agentview_plus_actual_pad_crop_v1",
                                "agentview_plus_actual_pad_contact_corridor_v1",
                                "agentview_plus_actual_pad_closing_corridor_v1",
                            }
                            and self._operator_pointcloud_points_world
                            is not None
                            and self._operator_pointcloud_colors_rgb
                            is not None
                        ):
                            feedback_path = render_move_feedback_local_cloud(
                                source_view,
                                output_path=feedback_output_path,
                                world_points=(
                                    self._operator_pointcloud_points_world
                                ),
                                colors_rgb=(
                                    self._operator_pointcloud_colors_rgb
                                ),
                                target_position_xyz_m=position,
                                actual_position_xyz_m=(
                                    actual_pose_for_feedback[:3, 3]
                                ),
                                actual_pad_contact_centers_world_m=(
                                    self._current_finger_pad_contacts_for_render()
                                ),
                                actual_pad_boxes=(
                                    self._current_finger_pad_boxes_for_render()
                                ),
                                render_actual_pad_contact_corridor=(
                                    self._not_reached_default_view_policy
                                    in {
                                        "agentview_plus_actual_pad_contact_corridor_v1",
                                        "agentview_plus_actual_pad_closing_corridor_v1",
                                    }
                                ),
                                render_actual_pad_closing_cues=(
                                    self._not_reached_default_view_policy
                                    == "agentview_plus_actual_pad_closing_corridor_v1"
                                ),
                            )
                        else:
                            feedback_path = render_move_feedback_crop(
                                source_view,
                                output_path=feedback_output_path,
                                target_position_xyz_m=position,
                                actual_position_xyz_m=(
                                    actual_pose_for_feedback[:3, 3]
                                ),
                                actual_pad_contact_centers_world_m=(
                                    self._current_finger_pad_contacts_for_render()
                                ),
                                actual_pad_boxes=(
                                    self._current_finger_pad_boxes_for_render()
                                ),
                                compact_actual_pad_geometry=(
                                    self._not_reached_default_view_policy
                                    in {
                                        "agentview_plus_actual_pad_crop_v1",
                                        "agentview_plus_actual_pad_contact_corridor_v1",
                                        "agentview_plus_actual_pad_closing_corridor_v1",
                                    }
                                ),
                                render_actual_pad_contact_corridor=(
                                    self._not_reached_default_view_policy
                                    in {
                                        "agentview_plus_actual_pad_contact_corridor_v1",
                                        "agentview_plus_actual_pad_closing_corridor_v1",
                                    }
                                ),
                                render_actual_pad_closing_cues=(
                                    self._not_reached_default_view_policy
                                    == "agentview_plus_actual_pad_closing_corridor_v1"
                                ),
                            )
                    result.images = [
                        path
                        for path in (agentview_path, feedback_path)
                        if path is not None
                    ]
                    execution_return_views = [
                        *(["agentview"] if agentview_path is not None else []),
                        *(
                            [
                                "move_feedback_"
                                + move_feedback_view.removeprefix(
                                    "pointcloud_"
                                )
                            ]
                            if feedback_path is not None
                            else []
                        ),
                    ]
                    execution_returned_views_override = list(
                        execution_return_views
                    )
                    if feedback_path is not None:
                        result.details["default_view_selection"][
                            "feedback_artifact"
                        ] = str(feedback_path.relative_to(self.root))
                else:
                    result.images = [
                        path
                        for name in execution_return_views
                        for path in [
                            contact_overlay_paths.get(name)
                            or self._operator_view_path(name)
                        ]
                        if path is not None
                    ]
            result.text["requested_views"] = requested_views
            result.text["returned_views"] = (
                preview_returned_views
                if preview_only
                else (
                    execution_returned_views_override
                    if execution_returned_views_override is not None
                    else self._returned_operator_view_names(
                        execution_return_views
                    )
                )
            )
            if (
                preview_only
                and preview_substituted_from
                and not self._operator_source_view_only_images
            ):
                substitution_reason = (
                    "A hypothetical pose cannot appear in live agentview or "
                    "wrist imagery; the returned contact sheet contains the "
                    "exact candidate PAD/JAW/APP geometry."
                )
                if agentview_preview_path is not None:
                    substitution_reason = (
                        "A hypothetical pose cannot appear in live wrist "
                        "imagery. The returned contact sheet contains the exact "
                        "candidate PAD/JAW/APP geometry; agentview_candidate_pose_overlay "
                        "is a calibrated hypothetical projection over the real RGB frame."
                    )
                result.text["view_substitution"] = {
                    "requested_unavailable_views": preview_substituted_from,
                    "returned_view": "pointcloud_contact_sheet",
                    "reason": substitution_reason,
                }
            result.text.update({
                "target_resolution": {
                    "observation_id": actual,
                    "constraints_used": constraints,
                    "position_source": (
                        "marked_point"
                        if position_point_id
                        else "direct_numeric"
                        if target.get("position_xyz_m") is not None
                        else "measured_point_pair_delta"
                        if has_point_delta
                        else "grip_site_delta"
                        if position_delta is not None
                        and delta_frame == "grip_site"
                        else "world_delta"
                        if position_delta is not None
                        else "preserved_current"
                    ),
                    "inherited_constraints": inherited,
                    "orientation_resolution_policy": resolution,
                    "delta_frame": delta_frame,
                    "base_actual_position_xyz_m": (
                        base_actual_position if position_delta is not None else None
                    ),
                    "position_delta_mm": position_delta,
                    "position_delta_mm_world": (
                        position_delta_world.tolist()
                        if position_delta_world is not None
                        else None
                    ),
                    "point_ids": {
                        "position": position_point_id or None,
                        "position_delta_from": (
                            position_delta_from_point_id or None
                        ),
                        "position_delta_to": (
                            position_delta_to_point_id or None
                        ),
                        "approach_from": approach_point_id or None,
                        "jaw_toward": jaw_point_id or None,
                    },
                    "point_provenance_observation_ids": {
                        point_id: self._point_marks_3d[point_id].get(
                            "observation_id"
                        )
                        for point_id in point_ids
                    },
                    "point_world_xyz_m": {
                        point_id: self._point_marks_3d[point_id].get("xyz_m")
                        for point_id in point_ids
                    },
                    "resolved_target_pose": {
                        "frame": "world",
                        "eef_frame": "panda_grip_site",
                        "position_xyz_m": position,
                        "rotation_matrix": rigid[:3, :3].tolist(),
                        "jaw_axis_world": rigid[:3, 0].tolist(),
                        "approach_axis_world": rigid[:3, 2].tolist(),
                    },
                }
            })
            try:
                actual_pose = self._current_world_from_grip_site()
                residual_mm = (
                    np.asarray(position, dtype=np.float64)
                    - actual_pose[:3, 3]
                ) * 1000.0
                result.text["actual_grip_site_pose"] = {
                    "frame": "world",
                    "eef_frame": "panda_grip_site",
                    "position_xyz_m": actual_pose[:3, 3].tolist(),
                    "rotation_matrix": actual_pose[:3, :3].tolist(),
                    "jaw_axis_world": actual_pose[:3, 0].tolist(),
                    "approach_axis_world": actual_pose[:3, 2].tolist(),
                }
                result.text["target_minus_actual_mm_world"] = (
                    residual_mm.tolist()
                )
                result.text["actual_position_delta_mm_world"] = (
                    (
                        actual_pose[:3, 3]
                        - np.asarray(base_actual_position, dtype=np.float64)
                    )
                    * 1000.0
                ).tolist()
                if preview_only:
                    result.text["actual_position_delta_mm_world"] = [
                        0.0,
                        0.0,
                        0.0,
                    ]
                result.text["convergence_interpretation"] = (
                    "This reports only commanded-versus-actual pose error. "
                    "A motion_not_converged result does not by itself prove "
                    "collision, table contact, kinematic infeasibility, or "
                    "that motion in the remaining-error direction is blocked."
                )
            except (RuntimeError, TypeError, ValueError):
                pass
            # Keep the Operator-facing contract small and stable. The inner
            # motion/gripper implementations intentionally retain rich control
            # evidence, but exposing their state-machine names, complete poses,
            # controller windows, and provenance on every call makes the next
            # physical decision harder to see. Preserve that full payload in
            # host details and return only action outcome, endpoint status,
            # measured gripper state, and the requested visual evidence.
            full_result = dict(result.text)
            current_gripper = _operator_gripper_feedback(
                self.current_sim_payload
            )
            if current_gripper:
                full_result["current_gripper"] = current_gripper
            result.details["move_to_full_result"] = full_result
            result.text = _move_to_operator_result(
                full_result,
                success=result.success,
                motion_requested=motion_requested,
                requested_gripper=requested_gripper,
                returned_views=result.text["returned_views"],
                observation_id=(
                    self.current_record.get("observation_id")
                    if self.current_record is not None
                    else None
                ),
                orientation_requested=any(
                    target.get(field) is not None
                    for field in (
                        "rotation_matrix",
                        "approach_direction_world",
                        "jaw_direction_world",
                        "approach_from_point_id",
                        "jaw_toward_point_id",
                    )
                ),
            )
            return result
        except (TypeError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            return GatewayResult(False, {
                "kind": "move_to",
                "success": False,
                "reason": "invalid_target",
                "message": str(exc),
                "retryable": True,
            })

    def nudge_end_effector(
        self,
        translation_delta_mm: list[float],
        rotation_delta_deg: list[float] | None = None,
        frame: str = "world",
        reason: str = "",
        tolerance_mm: float = 2.0,
        max_steps: int = 320,
    ) -> GatewayResult:
        """Move incrementally from the simulator-reported current grip site."""

        self.ensure_started()
        if self.failure_case is not None:
            return self._failed_result("nudge_end_effector")
        token, blocked = self._begin_world_action(
            action="nudge_end_effector", stage="move_to_selected_grasp"
        )
        if blocked is not None:
            return blocked
        assert token is not None
        action_generation = self._world_action_generation(token)
        try:
            try:
                translation = np.asarray(
                    _finite_vector3(
                        translation_delta_mm, name="translation_delta_mm"
                    ),
                    dtype=np.float64,
                ) / 1000.0
                rotation_delta = _finite_vector3(
                    rotation_delta_deg or [0.0, 0.0, 0.0],
                    name="rotation_delta_deg",
                )
                frame = str(frame).strip().lower()
                if frame not in {"world", "eef_local", "wrist_camera"}:
                    raise GraspPoseRefinementError(
                        "frame must be 'world', 'eef_local', or 'wrist_camera'"
                    )
                current = self._current_world_from_grip_site()
                controller = _direct_move_controller_contract(
                    tolerance_mm=tolerance_mm,
                    max_steps=max_steps,
                )
                delta_rotation = _euler_xyz_matrix_degrees(rotation_delta)
                target = current.copy()
                if frame == "world":
                    target[:3, 3] = current[:3, 3] + translation
                    target[:3, :3] = delta_rotation @ current[:3, :3]
                elif frame == "eef_local":
                    target[:3, 3] = current[:3, 3] + current[:3, :3] @ translation
                    target[:3, :3] = current[:3, :3] @ delta_rotation
                else:
                    if self.current_record is None:
                        raise RuntimeError(
                            "Current observation is required for wrist-camera motion"
                        )
                    wrist = resolve_observation_frame(
                        self.current_record,
                        artifact_root=self.root,
                        camera_id="wrist",
                    )
                    world_from_wrist = camera_to_world_opencv(wrist.extrinsics)
                    wrist_rotation = world_from_wrist[:3, :3]
                    target[:3, 3] = (
                        current[:3, 3] + wrist_rotation @ translation
                    )
                    world_delta_rotation = (
                        wrist_rotation
                        @ delta_rotation
                        @ wrist_rotation.T
                    )
                    target[:3, :3] = world_delta_rotation @ current[:3, :3]
                return self._move_to_world_grip_site(
                    target,
                    control_mode=f"nudge_{frame}",
                    reason=reason,
                    provenance={
                        "translation_delta_mm": translation_delta_mm,
                        "rotation_delta_deg": rotation_delta,
                        "delta_frame": frame,
                        **(
                            {
                                "wrist_camera_axes": {
                                    "x": "image_right",
                                    "y": "image_down",
                                    "z": "camera_forward",
                                }
                            }
                            if frame == "wrist_camera"
                            else {}
                        ),
                        "controller": controller,
                    },
                    action_generation=action_generation,
                )
            except (GraspPoseRefinementError, RuntimeError, ValueError) as exc:
                return GatewayResult(
                    False,
                    {
                        "kind": "nudge_end_effector",
                        "success": False,
                        "retryable": True,
                        "error": str(exc),
                    },
                )
        finally:
            self._end_world_action(token)

    def _current_world_from_grip_site(self) -> np.ndarray:
        robot = self.current_sim_payload.get("robot", {})
        pose = robot.get("end_effector_pose", {}) if isinstance(robot, Mapping) else {}
        xyz = _finite_vector3(
            pose.get("xyz") if isinstance(pose, Mapping) else None,
            name="current grip-site translation",
        )
        rotation = _quat_xyzw_to_rotation_matrix(
            pose.get("quat_xyzw") if isinstance(pose, Mapping) else None
        )
        if rotation is None:
            raise RuntimeError(
                "Current observation has no frame-consistent grip-site orientation"
            )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.asarray(xyz, dtype=np.float64)
        return rigid_transform(transform, name="current_world_from_grip_site")

    def _capture_active_grasp_target(self) -> dict[str, Any] | None:
        """Retain the semantic target and pre-grasp metric centroid."""

        if self.current_record is None or not isinstance(self.selected_detection, Mapping):
            return None
        detection = dict(self.selected_detection)
        target: dict[str, Any] = {
            "label": str(detection.get("label") or "").strip(),
            "detection_id": detection.get("id"),
            "observation_id": self.current_record.get("observation_id"),
            "mask_ref": detection.get("mask_ref"),
        }
        bbox = detection.get("bbox_xyxy")
        if (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(value, (int, float)) for value in bbox)
        ):
            target["baseline_point_xy"] = [
                (float(bbox[0]) + float(bbox[2])) * 0.5,
                (float(bbox[1]) + float(bbox[3])) * 0.5,
            ]
        try:
            frame = resolve_observation_frame(
                self.current_record,
                artifact_root=self.root,
                camera_id="agentview",
            )
            mask_path = Path(str(detection.get("mask_ref") or ""))
            if not mask_path.is_absolute():
                mask_path = self.root / mask_path
            points = backproject_masked_world(frame, mask_path).points_world
            target["baseline_centroid_world_m"] = np.median(points, axis=0).tolist()
        except Exception as exc:  # noqa: BLE001 - later verification will return unknown.
            target["baseline_geometry_error"] = _exception_message(exc)
        return target

    def _move_to_world_grip_site(
        self,
        transform: Any,
        *,
        control_mode: str,
        reason: str,
        provenance: Mapping[str, Any] | None = None,
        action_generation: int,
    ) -> GatewayResult:
        assert self.current_record is not None and self.writer is not None
        rigid = rigid_transform(transform, name="direct_world_from_grip_site")
        # Direct visual-servo motion changes the physical pregrasp relation, so
        # a previously approved contact target must never remain executable.
        self._pending_grasp_approach = None
        self._manual_move_seq += 1
        candidate_id = f"M{self._manual_move_seq - 1}"
        candidate = {
            "id": candidate_id,
            "display_label": candidate_id,
            "rank": 0,
            "width": 0.08,
            "pose_frame": "world",
            "eef_frame": "panda_grip_site",
            "transform_world_from_grip_site": rigid.tolist(),
            "source_backend": "operator_direct_move",
            "provenance": {
                "control_mode": control_mode,
                "reason": str(reason),
                **dict(provenance or {}),
            },
        }
        self.selected_grasp = candidate
        self.selected_grasp_observation_id = str(
            self.current_record.get("observation_id") or ""
        )
        self.writer.record_operator_choice(
            choice_type="direct_move",
            choice_id=candidate_id,
            input_frames=self.current_record.get("frame_ids", []),
            metadata={"control_mode": control_mode, "reason": str(reason)},
        )
        result = self._step_world_action(
            "move_to_selected_grasp", action_generation=action_generation
        )
        result.text["kind"] = (
            "nudge_end_effector"
            if control_mode.startswith("nudge_")
            else "move_to_pose"
        )
        result.text["control_mode"] = control_mode
        result.text["direct_move_id"] = candidate_id
        controller = candidate["provenance"].get("controller")
        if isinstance(controller, Mapping):
            result.text["controller"] = {
                "tolerance_mm": controller.get("tolerance_mm"),
                "max_steps": controller.get("max_steps"),
            }
        result.text["message"] = (
            f"{result.text.get('message', '')} Direct move {candidate_id} used the "
            "requested world-frame Panda grip-site target without AG/GX/R conversion."
        ).strip()
        result.details["direct_move"] = candidate
        # Direct pose and nudge tools are visual-servo recovery surfaces. The
        # close-range wrist result is needed to judge centering and clearance
        # before choosing the next small correction.
        if self.current_record is not None and not result.text.get(
            "remote_completion_unknown"
        ):
            result.images = self._operator_approach_frame_paths(
                self.current_record.get("frames", [])
            )
        return result

    def step(self, stage: str) -> GatewayResult:
        if stage == "move_to_selected_grasp":
            return GatewayResult(
                False,
                {
                    "kind": "step",
                    "success": False,
                    "retryable": True,
                    "reason": "wrist_checkpoint_required",
                    "required_next_tool": "move_to_selected_pregrasp",
                    "error": (
                        "Normal selected-grasp execution requires a wrist-visible "
                        "pregrasp checkpoint before final approach."
                        ),
                    },
                )
        self.ensure_started()
        token, blocked = self._begin_world_action(action="step", stage=stage)
        if blocked is not None:
            return blocked
        assert token is not None
        action_generation = self._world_action_generation(token)
        try:
            return self._step_world_action(
                stage, action_generation=action_generation
            )
        finally:
            self._end_world_action(token)

    def _step_world_action(
        self, stage: str, *, action_generation: int
    ) -> GatewayResult:
        assert self.current_record is not None and self.writer is not None
        if self.failure_case is not None:
            return self._failed_result("step")
        allowed = {
            "move_to_selected_grasp",
            "move_to_selected_pregrasp",
            "approach_selected_grasp",
            "lift_grasp",
            "open_gripper",
            "close_gripper",
        }
        if stage not in allowed:
            return GatewayResult(False, {"kind": "step", "success": False, "error": f"Unknown stage: {stage}", "allowed": sorted(allowed)})
        if stage in {"lift_grasp", "open_gripper", "close_gripper"}:
            # Any other robot action makes the wrist-visible checkpoint stale.
            self._pending_grasp_approach = None
        grasp_start_stages = {
            "move_to_selected_grasp",
            "move_to_selected_pregrasp",
        }
        if stage in grasp_start_stages:
            if self.selected_grasp is None or self.selected_grasp_observation_id != self.current_record.get("observation_id"):
                return GatewayResult(False, {"kind": "step", "success": False, "error": "Select a grasp from the current observation first."})
        if stage == "approach_selected_grasp" and self._pending_grasp_approach is None:
            return GatewayResult(
                False,
                {
                    "kind": "step",
                    "success": False,
                    "retryable": True,
                    "error": (
                        "No selected-grasp pregrasp checkpoint is pending. Call "
                        "move_to_selected_pregrasp first."
                    ),
                },
            )

        pending_approach_snapshot = (
            dict(self._pending_grasp_approach)
            if stage == "approach_selected_grasp"
            and self._pending_grasp_approach is not None
            else None
        )
        executed_grasp = (
            dict(pending_approach_snapshot["selected_grasp"])
            if pending_approach_snapshot is not None
            else dict(self.selected_grasp)
            if self.selected_grasp
            else None
        )
        direct_move = bool(
            executed_grasp
            and executed_grasp.get("source_backend") == "operator_direct_move"
        )
        if stage in grasp_start_stages and not direct_move:
            assert executed_grasp is not None
            grasp_id = str(executed_grasp.get("id") or "")
            clearance = self._grasp_support_clearance_contract(grasp_id)
            if clearance["support_clearance_status"] != "eligible":
                return self._unsafe_grasp_clearance_result(
                    kind="step",
                    grasp_id=grasp_id,
                    clearance=clearance,
                    phase="move_to_selected_grasp preflight",
                )
            self.selected_grasp.update(
                {
                    key: clearance[key]
                    for key in (
                        "support_clearance_mm",
                        "support_clearance_required_mm",
                        "support_clearance_status",
                    )
                }
            )
            executed_grasp = dict(self.selected_grasp)
        if stage in grasp_start_stages and not direct_move:
            self._pending_grasp_approach = None
            self._grasp_attempt_seq += 1
            self._active_grasp_attempt_id = (
                f"grasp-attempt-{self._grasp_attempt_seq:03d}"
            )
            self._active_grasp_target = self._capture_active_grasp_target()
            self._last_lift_reference = None
            self._last_grasp_verification = None
        grasp_attempt_id = self._active_grasp_attempt_id
        input_frames = list(self.current_record.get("frame_ids", []))
        action = self.writer.record_action(
            request={
                "stage": stage,
                "grasp_id": executed_grasp.get("id") if executed_grasp else None,
                "grasp_attempt_id": grasp_attempt_id,
            },
            input_frames=input_frames,
        )
        tool_call = self.writer.record_tool_start(
            tool=stage,
            action_id=action.get("action_id"),
            input_frames=input_frames,
            parameters={"stage": stage, "grasp_attempt_id": grasp_attempt_id},
        )
        self._bind_active_world_action(
            action_generation=action_generation,
            action_id=str(action.get("action_id") or ""),
            tool_call_id=str(tool_call.get("tool_call_id") or ""),
            grasp_attempt_id=grasp_attempt_id,
        )
        motion_stage = stage in {
            "move_to_selected_grasp",
            "move_to_selected_pregrasp",
            "approach_selected_grasp",
            "lift_grasp",
        }
        pre_gripper = _operator_gripper_feedback(
            {"observation": self.current_sim_payload}
        )
        gripper_state_error: str | None = None
        remote_name = {"open_gripper": "gripper_open", "close_gripper": "gripper_close"}.get(stage)
        remote_args: dict[str, Any] = {}
        approach_open_args: dict[str, Any] | None = None
        approach_open_result: dict[str, Any] | None = None
        approach_open_feedback: dict[str, Any] | None = None
        control_artifact: Path | None = None
        world_pose: dict[str, Any] | None = None
        pregrasp_args: dict[str, Any] | None = None
        pending_approach: dict[str, Any] | None = None
        pending_lift_reference: dict[str, Any] | None = None
        remote: dict[str, Any]
        success = False
        content = ""
        remote_task_success = False
        try:
            if stage in grasp_start_stages:
                world_pose, remote_args = self._selected_move_arguments()
                remote_name = "move_to"
                if direct_move:
                    # Direct move_to is the general Operator motion surface,
                    # including contact-rich placement and recovery. Give it
                    # the same measured tracking guard as other Cartesian
                    # motion so a constrained target returns actual stall
                    # evidence instead of pushing for the whole horizon. This
                    # does not alter the exact requested target or claim
                    # success. Do not apply the short contact phase's
                    # translation cap to every direct move: long free-space
                    # moves otherwise exhaust max_steps while still making
                    # healthy progress.
                    remote_args.update(
                        {
                            "tracking_stall_steps": (
                                PREGRASP_TRACKING_STALL_STEPS
                            ),
                            "tracking_min_aligned_progress_m": (
                                PREGRASP_TRACKING_MIN_ALIGNED_PROGRESS_M
                            ),
                            "tracking_cross_track_tolerance_m": (
                                PREGRASP_TRACKING_CROSS_TRACK_TOLERANCE_M
                            ),
                            "tracking_min_error_improvement_m": (
                                PREGRASP_TRACKING_MIN_ERROR_IMPROVEMENT_M
                            ),
                        }
                    )
                else:
                    approach_open_args = self._session_args({"handle": self.handle})
                    pregrasp_args = _pregrasp_move_arguments(
                        world_pose,
                        remote_args,
                        standoff_m=PREGRASP_STANDOFF_M,
                    )
                    remote_args["num_steps"] = min(
                        int(remote_args.get("num_steps", 220)), 120
                    )
                    # The short contact phase uses conservative Cartesian
                    # increments.  Both phases use tracking guards, but the
                    # free-space pregrasp window is longer so normal arm
                    # transients do not consume an entire LIBERO horizon when
                    # a target pose is unreachable or the controller plateaus.
                    remote_args.update(
                        {
                            "max_translation_step_m": (
                                CONTACT_MAX_TRANSLATION_STEP_M
                            ),
                            "tracking_stall_steps": (
                                CONTACT_TRACKING_STALL_STEPS
                            ),
                            "tracking_min_aligned_progress_m": (
                                CONTACT_TRACKING_MIN_ALIGNED_PROGRESS_M
                            ),
                            "tracking_cross_track_tolerance_m": (
                                CONTACT_TRACKING_CROSS_TRACK_TOLERANCE_M
                            ),
                            "tracking_min_error_improvement_m": (
                                CONTACT_TRACKING_MIN_ERROR_IMPROVEMENT_M
                            ),
                        }
                    )
                    if stage == "move_to_selected_pregrasp":
                        pending_approach = {
                            "selected_grasp": executed_grasp,
                            "control_pose": world_pose,
                            "contact_arguments": dict(remote_args),
                            "grasp_attempt_id": grasp_attempt_id,
                        }
                control_artifact = self._write_control_artifact(
                    action_id=action.get("action_id", "action"),
                    payload={
                        "stage": stage,
                        "grasp_attempt_id": grasp_attempt_id,
                        "selected_grasp": executed_grasp,
                        "control_pose": world_pose,
                        "target_semantics": world_pose.get("target_semantics"),
                        **(
                            {"approach_open_arguments": approach_open_args}
                            if approach_open_args is not None
                            else {}
                        ),
                        **(
                            {"pregrasp_arguments": pregrasp_args}
                            if pregrasp_args is not None
                            else {}
                        ),
                        "remote_arguments": remote_args,
                    },
                )
            elif stage == "approach_selected_grasp":
                assert pending_approach_snapshot is not None
                world_pose = dict(pending_approach_snapshot["control_pose"])
                remote_args = dict(
                    pending_approach_snapshot["contact_arguments"]
                )
                executed_grasp = dict(
                    pending_approach_snapshot["selected_grasp"]
                )
                remote_name = "move_to"
                control_artifact = self._write_control_artifact(
                    action_id=action.get("action_id", "action"),
                    payload={
                        "stage": stage,
                        "grasp_attempt_id": grasp_attempt_id,
                        "selected_grasp": executed_grasp,
                        "control_pose": world_pose,
                        "target_semantics": world_pose.get("target_semantics"),
                        "checkpoint_observation_id": (
                            pending_approach_snapshot.get(
                                "checkpoint_observation_id"
                            )
                        ),
                        "remote_arguments": remote_args,
                    },
                )
            elif stage == "lift_grasp":
                remote_name = "move_to"
                remote_args = self._lift_move_arguments()
                pending_lift_reference = {
                    "grasp_attempt_id": grasp_attempt_id,
                    "start_observation_id": self.current_record.get("observation_id"),
                    "start_eef_world_m": self._current_world_from_grip_site()[:3, 3].tolist(),
                }
                robot = self.current_sim_payload.get("robot", {})
                pose = (
                    robot.get("end_effector_pose", {})
                    if isinstance(robot, Mapping)
                    else {}
                )
                lift_rotation = _quat_xyzw_to_rotation_matrix(
                    pose.get("quat_xyzw") if isinstance(pose, Mapping) else None
                )
                if lift_rotation is None:
                    raise RuntimeError(
                        "Current observation has no EEF orientation for lift_grasp"
                    )
                world_pose = {
                    "frame": "world",
                    "eef_frame": "panda_grip_site",
                    "translation_xyz": [
                        float(remote_args[key]) for key in ("x", "y", "z")
                    ],
                    "rotation_matrix": lift_rotation.tolist(),
                    "source_backend": "operator_lift",
                    "source_grasp_id": grasp_attempt_id or "lift",
                    "target_semantics": "explicit_world_panda_grip_site_lift",
                    "rotation_adapter": "none_world_panda_grip_site",
                }
                aperture_mm = pre_gripper.get("aperture_mm")
                executed_grasp = {
                    "id": f"{grasp_attempt_id or 'unbound'}-lift",
                    "width": (
                        float(aperture_mm) / 1000.0
                        if isinstance(aperture_mm, (int, float))
                        else 0.08
                    ),
                    "source_backend": "operator_lift",
                }
                control_artifact = self._write_control_artifact(
                    action_id=action.get("action_id", "action"),
                    payload={
                        "stage": stage,
                        "grasp_attempt_id": grasp_attempt_id,
                        "selected_grasp": executed_grasp,
                        "control_pose": world_pose,
                        "target_semantics": world_pose.get("target_semantics"),
                        "remote_arguments": remote_args,
                    },
                )
            else:
                remote_args = self._session_args({"handle": self.handle})
            assert remote_name is not None
            if stage == "approach_selected_grasp":
                # The checkpoint authorizes exactly one contact attempt.  Keep
                # the snapshot locally for observability, but consume the live
                # token before issuing motion so timeout/failure cannot be
                # retried accidentally against stale wrist evidence.
                self._pending_grasp_approach = None
            if pregrasp_args is not None:
                assert approach_open_args is not None
                remote_name = "gripper_open"
                approach_open_result = self.transport.call_tool(
                    remote_name,
                    approach_open_args,
                    timeout_s=_remote_world_action_timeout_s(
                        remote_name, approach_open_args
                    ),
                )
                approach_open_feedback = _operator_gripper_feedback(
                    approach_open_result
                )
                approach_open_ok, approach_open_message = (
                    _approach_gripper_open_payload_ok(approach_open_result)
                )
                if not _payload_ok(approach_open_result) or not approach_open_ok:
                    remote = dict(approach_open_result)
                    remote.update(
                        {
                            "success": False,
                            "motion_phase": "approach_open",
                            "motion_plan": (
                                "open_staged_pregrasp_checkpoint"
                                if stage == "move_to_selected_pregrasp"
                                else "open_staged_pregrasp_contact"
                            ),
                            "controller_error": (
                                _payload_error(approach_open_result)
                                if not _payload_ok(approach_open_result)
                                else approach_open_message
                            ),
                            "approach_gripper_opened": False,
                            "approach_gripper": approach_open_feedback,
                            "approach_open_result": dict(approach_open_result),
                        }
                    )
                elif _task_terminal_success(approach_open_result):
                    remote = dict(approach_open_result)
                    remote.update(
                        {
                            "motion_phase": "approach_open",
                            "motion_plan": (
                                "open_staged_pregrasp_checkpoint"
                                if stage == "move_to_selected_pregrasp"
                                else "open_staged_pregrasp_contact"
                            ),
                            "approach_gripper_opened": True,
                            "approach_gripper": approach_open_feedback,
                            "approach_open_result": dict(approach_open_result),
                        }
                    )
                else:
                    remote_name = "move_to"
                    pregrasp_plan = _pregrasp_waypoint_arguments(
                        approach_open_result,
                        pregrasp_args,
                    )
                    pregrasp_results: list[dict[str, Any]] = []
                    for phase, waypoint_args in pregrasp_plan:
                        waypoint_result = self.transport.call_tool(
                            remote_name,
                            waypoint_args,
                            timeout_s=_remote_world_action_timeout_s(
                                remote_name, waypoint_args
                            ),
                        )
                        pregrasp_results.append(
                            {
                                "phase": phase,
                                "arguments": dict(waypoint_args),
                                "result": dict(waypoint_result),
                            }
                        )
                        waypoint_ok, _waypoint_message = _motion_payload_ok(
                            waypoint_result
                        )
                        if (
                            not _payload_ok(waypoint_result)
                            or not waypoint_ok
                            or _task_terminal_success(waypoint_result)
                        ):
                            break
                    pregrasp_remote = _combine_pregrasp_waypoint_results(
                        pregrasp_results
                    )
                    pregrasp_ok, _pregrasp_message = _motion_payload_ok(
                        pregrasp_remote
                    )
                    if (
                        stage == "move_to_selected_pregrasp"
                        and _payload_ok(pregrasp_remote)
                        and pregrasp_ok
                        and not _task_terminal_success(pregrasp_remote)
                    ):
                        remote = dict(pregrasp_remote)
                        remote["motion_phase"] = "pregrasp_checkpoint"
                        remote["motion_plan"] = "open_staged_pregrasp_checkpoint"
                    elif _payload_ok(pregrasp_remote) and pregrasp_ok and not _task_terminal_success(pregrasp_remote):
                        contact_remote = self.transport.call_tool(
                            remote_name,
                            remote_args,
                            timeout_s=_remote_world_action_timeout_s(
                                remote_name, remote_args
                            ),
                        )
                        remote = _combine_staged_motion_results(
                            pregrasp_remote,
                            contact_remote,
                        )
                    else:
                        remote = dict(pregrasp_remote)
                        remote.setdefault("motion_phase", "pregrasp")
                        remote["motion_plan"] = (
                            "open_staged_pregrasp_checkpoint"
                            if stage == "move_to_selected_pregrasp"
                            else "open_staged_pregrasp_contact"
                        )
                    remote["steps_executed"] = max(
                        0, _int_or(approach_open_result.get("steps_executed"), 0)
                    ) + max(0, _int_or(remote.get("steps_executed"), 0))
                    remote["motion_plan"] = (
                        "open_staged_pregrasp_checkpoint"
                        if stage == "move_to_selected_pregrasp"
                        else "open_staged_pregrasp_contact"
                    )
                    remote["approach_gripper_opened"] = True
                    remote["approach_gripper"] = approach_open_feedback
                    remote["approach_open_result"] = dict(approach_open_result)
            else:
                remote = self.transport.call_tool(
                    remote_name,
                    remote_args,
                    timeout_s=_remote_world_action_timeout_s(remote_name, remote_args),
                )
            success = _payload_ok(remote)
            content = _payload_error(remote) if not success else f"{stage} completed"
            # LIBERO can satisfy the native predicate during a gripper
            # primitive (for example, the release that completes placement),
            # not only during Cartesian motion. Detect terminal success once
            # for every world-changing backend response so a subsequent
            # gripper-state mismatch cannot hide a completed episode.
            remote_task_success = _task_terminal_success(remote)
            if motion_stage:
                motion_ok, motion_message = _motion_payload_ok(remote)
                if not motion_ok:
                    success = False
                    phase = str(remote.get("motion_phase") or "").strip()
                    content = (
                        f"{phase} phase {motion_message}"
                        if phase
                        else motion_message
                    )
                else:
                    if remote_task_success:
                        success = True
                        content = (
                            "Native task completed during motion; the simulator "
                            "stopped at its terminal success state."
                        )
                    else:
                        content = _motion_feedback_message(stage, remote)
                        if stage == "lift_grasp":
                            content += (
                                " This confirms arm motion only; grasp contact is "
                                "unverified. Call verify_grasp before transport."
                            )
            elif stage in {"open_gripper", "close_gripper"} and success:
                gripper_feedback = _operator_gripper_feedback(remote)
                gripper_ok, gripper_message = _gripper_action_payload_ok(
                    stage,
                    before=pre_gripper,
                    after=gripper_feedback,
                )
                if remote_task_success:
                    # The native terminal predicate is authoritative for the
                    # episode. Preserve the gripper measurement in host
                    # details, but do not turn a successful release/close
                    # primitive into a public action failure.
                    success = True
                    gripper_state_error = None
                    content = (
                        "Native task completed during the gripper action; the "
                        "simulator stopped at its terminal success state."
                    )
                elif not gripper_ok:
                    success = False
                    gripper_state_error = gripper_message
                    content = gripper_message
                aperture = gripper_feedback.get("aperture_mm")
                if (
                    success
                    and not remote_task_success
                    and isinstance(aperture, (int, float))
                ):
                    content = f"{stage} completed; gripper aperture {aperture:.1f} mm."
                if stage == "close_gripper":
                    gripper_feedback["grasp_evidence"] = (
                        _gripper_close_evidence(gripper_feedback)
                    )
        except Exception as exc:  # noqa: BLE001
            error_message = _exception_message(exc)
            remote_completion_unknown = (
                remote_name == "move_to" and _is_timeout_exception(exc)
            )
            remote = {
                "error": error_message,
                "error_type": type(exc).__name__,
                **(
                    {
                        "completion_state": "unknown",
                        "canceled": False,
                    }
                    if remote_completion_unknown
                    else {}
                ),
            }
            success = False
            content = (
                f"{stage} timed out while waiting for the remote simulator. "
                "The remote action was not canceled and may still complete; "
                "remote completion is unknown."
                if remote_completion_unknown
                else f"{stage} failed: {error_message}"
            )

        remote_completion_unknown = bool(
            remote.get("completion_state") == "unknown"
        )

        if not self._world_action_completion_is_current(action_generation):
            return self._ignored_late_world_action_completion(
                stage=stage,
                action_id=str(action.get("action_id") or ""),
                tool_call_id=str(tool_call.get("tool_call_id") or ""),
                action_generation=action_generation,
                remote_name=remote_name,
                remote=remote,
            )

        if remote_task_success:
            self.last_task_success = True
            self._task_success_latch = {
                "source": "native_world_action_termination",
                "observation_id": (
                    self.current_record.get("observation_id")
                    if self.current_record is not None
                    else None
                ),
                "action_id": action.get("action_id"),
                "stage": stage,
                "reward": remote.get("reward"),
                "terminated": remote.get("terminated"),
            }
        if stage == "lift_grasp" and success and pending_lift_reference is not None:
            self._last_lift_reference = pending_lift_reference
            self._last_grasp_verification = None
        self._sim_step += max(1, _int_or(remote.get("steps_executed"), 1))
        # A world-changing call must publish what the simulator looks like
        # afterwards even when the controller reports failure.  Otherwise a
        # terminal motion failure leaves Kitty and the Operator looking at the
        # pre-action grasp board, which hides where the arm actually stopped.
        # This render is evidence only: it never retries or recovers the action.
        post_frames: list[str] = []
        post_observation_error: str | None = None
        execution_comparison: Path | None = None
        execution_comparison_error: str | None = None
        if remote_completion_unknown:
            post_observation_error = (
                "Post-action render was not queued because the timed-out remote "
                "world action may still be running."
            )
        else:
            try:
                rendered = self.transport.call_tool(
                    "render_env",
                    self._session_args({"handle": self.handle}),
                    timeout_s=180.0,
                )
                if not _payload_ok(rendered):
                    post_observation_error = _payload_error(rendered)
                else:
                    self._capture_observation(
                        rendered,
                        source=f"post_{stage}",
                    )
                    post_frames = list(self.current_record.get("frame_ids", [])) if self.current_record else []
                    if (
                        motion_stage
                        and executed_grasp is not None
                        and world_pose is not None
                    ):
                        try:
                            comparison_control_pose = dict(world_pose)
                            if (
                                stage == "move_to_selected_pregrasp"
                                and pregrasp_args is not None
                            ):
                                comparison_control_pose.update(
                                    {
                                        "translation_xyz": [
                                            float(pregrasp_args[key])
                                            for key in ("x", "y", "z")
                                        ],
                                        "target_semantics": (
                                            "selected_grasp_pregrasp_standoff"
                                        ),
                                    }
                                )
                            execution_comparison = self._render_execution_comparison(
                                executed_grasp=executed_grasp,
                                control_pose=comparison_control_pose,
                                action_id=str(action.get("action_id") or "action"),
                                execution_stage=stage,
                            )
                        except Exception as exc:  # noqa: BLE001 - diagnostic failure is recoverable.
                            execution_comparison_error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # noqa: BLE001 - post-action evidence is mandatory.
                post_observation_error = f"{type(exc).__name__}: {exc}"

        checkpoint_candidate = bool(
            stage == "move_to_selected_pregrasp"
            and success
            and not remote_task_success
            and post_observation_error is None
            and pending_approach is not None
            and self.current_record is not None
        )
        checkpoint_ready = False
        checkpoint_evidence_error: str | None = None
        wrist_checkpoint: dict[str, Any] | None = None
        if checkpoint_candidate:
            assert self.current_record is not None
            assert pending_approach is not None
            evidence_observation_id = str(
                self.current_record.get("observation_id") or ""
            )
            agentview_frame = _find_frame(self.current_record, "agentview")
            wrist_frame = _find_frame(self.current_record, "wrist")
            post_gripper = _operator_gripper_feedback(self.current_sim_payload)
            measured_aperture_mm = post_gripper.get("aperture_mm")
            evidence_failures: list[str] = []
            if agentview_frame is None:
                evidence_failures.append("post-pregrasp agentview frame is missing")
            if wrist_frame is None:
                evidence_failures.append("post-pregrasp wrist frame is missing")
            if not isinstance(measured_aperture_mm, (int, float)):
                evidence_failures.append(
                    "post-pregrasp measured gripper aperture is missing"
                )
            elif float(measured_aperture_mm) < APPROACH_OPEN_MIN_APERTURE_MM:
                evidence_failures.append(
                    "post-pregrasp measured gripper aperture is insufficient "
                    f"({float(measured_aperture_mm):.1f} mm < "
                    f"{APPROACH_OPEN_MIN_APERTURE_MM:.1f} mm)"
                )
            checkpoint_id = (
                f"{grasp_attempt_id or 'grasp-attempt'}:"
                f"{evidence_observation_id or 'unknown-observation'}"
            )
            selected_grasp_id = str(
                pending_approach["selected_grasp"].get("id") or ""
            )
            wrist_checkpoint = {
                "checkpoint_id": checkpoint_id,
                "evidence_observation_id": evidence_observation_id,
                "selected_grasp_id": selected_grasp_id,
                "grasp_attempt_id": grasp_attempt_id,
                "frames": {
                    "agentview": {
                        "camera_id": "agentview",
                        "frame_id": (
                            agentview_frame.get("frame_id")
                            if agentview_frame is not None
                            else None
                        ),
                    },
                    "wrist": {
                        "camera_id": "wrist",
                        "frame_id": (
                            wrist_frame.get("frame_id")
                            if wrist_frame is not None
                            else None
                        ),
                    },
                },
                "gripper": {
                    **post_gripper,
                    "measured_observation_id": evidence_observation_id,
                },
                "robot_max_aperture_mm": PANDA_MAX_APERTURE_M * 1000.0,
                "minimum_automatic_approach_aperture_mm": (
                    APPROACH_OPEN_MIN_APERTURE_MM
                ),
                "visual_checks": [
                    "target visibly lies between the two fingers",
                    "visible target span leaves clearance inside the measured aperture",
                    "short final approach is not blocked by table or nearby geometry",
                ],
                "decision": (
                    "operator_visual_assessment_required"
                    if not evidence_failures
                    else "blocked_missing_atomic_evidence"
                ),
                "if_uncertain": (
                    "Do not approach blindly; nudge, refine, select another pose, "
                    "or re-open and create a fresh checkpoint."
                ),
                **(
                    {"evidence_failures": evidence_failures}
                    if evidence_failures
                    else {}
                ),
            }
            checkpoint_ready = not evidence_failures
        if checkpoint_ready:
            assert self.current_record is not None
            assert pending_approach is not None
            assert wrist_checkpoint is not None
            self._pending_grasp_approach = {
                **pending_approach,
                "checkpoint_observation_id": self.current_record.get(
                    "observation_id"
                ),
                "checkpoint_id": wrist_checkpoint["checkpoint_id"],
                "wrist_checkpoint": dict(wrist_checkpoint),
            }
            content = (
                "Pregrasp checkpoint reached. Inspect the returned agentview and "
                "wrist images; confirm the target is centered between the fingers, "
                "the aperture is sufficient, and the final approach is clear. Then "
                "call approach_selected_grasp, or refine/nudge and create a new "
                "checkpoint."
            )
        elif checkpoint_candidate:
            self._pending_grasp_approach = None
            success = False
            assert wrist_checkpoint is not None
            failures = wrist_checkpoint.get("evidence_failures", [])
            checkpoint_evidence_error = "; ".join(str(item) for item in failures)
            content = (
                "Pregrasp motion completed, but the atomic wrist checkpoint is "
                f"not safe to use: {checkpoint_evidence_error}. No final-approach "
                "token was issued. Inspect the returned evidence, correct the "
                "state, and create a fresh checkpoint."
            )

        issue: dict[str, Any] | None = None
        if not success:
            failure_code = (
                "pregrasp_checkpoint_evidence_incomplete"
                if checkpoint_evidence_error is not None
                else "remote_completion_unknown"
                if remote_completion_unknown
                else "cartesian_tracking_stalled"
                if motion_stage
                and remote.get("controller_error")
                == "cartesian_tracking_stalled"
                else "motion_not_converged"
                if motion_stage
                and ("reached_target" in remote or "motion_converged" in remote)
                else "gripper_state_mismatch"
                if gripper_state_error is not None
                else "remote_action_failed"
            )
            issue = self._record_issue(
                category=(
                    "observation"
                    if checkpoint_evidence_error is not None
                    else "action"
                ),
                component=(
                    "pregrasp_checkpoint"
                    if checkpoint_evidence_error is not None
                    else stage
                ),
                code=failure_code,
                message=content,
                tool=stage,
                action_id=action.get("action_id"),
                input_frames=input_frames,
                artifact_refs=[str(control_artifact)] if control_artifact is not None else [],
                details={
                    "grasp_attempt_id": grasp_attempt_id,
                    "remote_tool": remote_name,
                    "remote": remote,
                    **(
                        {"pre_gripper": pre_gripper, "gripper_state_error": gripper_state_error}
                        if gripper_state_error is not None
                        else {}
                    ),
                    **(
                        {"post_observation_error": post_observation_error}
                        if post_observation_error is not None
                        else {}
                    ),
                    **(
                        {
                            "checkpoint_evidence_error": checkpoint_evidence_error,
                            "wrist_checkpoint": wrist_checkpoint,
                        }
                        if checkpoint_evidence_error is not None
                        else {}
                    ),
                },
            )
        elif post_observation_error is not None:
            success = False
            content = f"{stage} completed, but post-action observation failed: {post_observation_error}"
            issue = self._record_issue(
                category="observation",
                component="simulator_renderer",
                code="post_action_observation_failed",
                message=content,
                tool=stage,
                action_id=action.get("action_id"),
                input_frames=input_frames,
                details={
                    "grasp_attempt_id": grasp_attempt_id,
                    "remote_tool": remote_name,
                    "error": post_observation_error,
                },
            )
        if execution_comparison_error is not None:
            comparison_issue = self._record_issue(
                category="observability",
                component="execution_pose_record",
                code="execution_comparison_failed",
                message=(
                    "Robot action completed, but the TARGET/ACTUAL execution record "
                    f"could not be written for Viser: {execution_comparison_error}"
                ),
                tool=stage,
                action_id=action.get("action_id"),
                input_frames=post_frames,
                details={
                    "grasp_attempt_id": grasp_attempt_id,
                    "error": execution_comparison_error,
                },
            )
            if issue is None:
                issue = comparison_issue
        artifact_refs = [str(control_artifact)] if control_artifact is not None else []
        if execution_comparison is not None:
            artifact_refs.append(str(execution_comparison))
        motion = _motion_details(remote) if motion_stage else {}
        operator_motion = _operator_motion_feedback(remote) if motion_stage else {}
        operator_gripper = (
            _operator_gripper_feedback(remote)
            if stage in {"open_gripper", "close_gripper"}
            else {}
        )
        if operator_gripper:
            operator_gripper.update(
                _gripper_action_outcome(
                    stage,
                    before=pre_gripper,
                    after=operator_gripper,
                    action_completed=bool(_payload_ok(remote)),
                )
            )
            if stage == "close_gripper":
                operator_gripper["grasp_evidence"] = _gripper_close_evidence(
                    operator_gripper
                )
        if control_artifact is not None:
            self._write_control_artifact(
                action_id=action.get("action_id", "action"),
                payload={
                    "stage": stage,
                    "grasp_attempt_id": grasp_attempt_id,
                    "selected_grasp": executed_grasp,
                    **(
                        {"control_pose": world_pose}
                        if world_pose is not None
                        else {}
                    ),
                    **(
                        {"target_semantics": world_pose.get("target_semantics")}
                        if world_pose is not None
                        else {}
                    ),
                    "remote_arguments": remote_args,
                    **(
                        {"approach_open_arguments": approach_open_args}
                        if approach_open_args is not None
                        else {}
                    ),
                    **(
                        {"approach_open_result": approach_open_result}
                        if approach_open_result is not None
                        else {}
                    ),
                    **(
                        {"approach_gripper": approach_open_feedback}
                        if approach_open_feedback is not None
                        else {}
                    ),
                    **(
                        {"pregrasp_arguments": pregrasp_args}
                        if pregrasp_args is not None
                        else {}
                    ),
                    "remote_result": remote,
                    "motion": motion,
                },
            )
        self.writer.record_tool_result(
            tool=stage,
            success=success,
            action_id=action.get("action_id"),
            tool_call_id=tool_call.get("tool_call_id"),
            input_frames=input_frames,
            post_frames=post_frames,
            artifact_refs=artifact_refs,
            result={
                "remote_tool": remote_name,
                "remote_success": success,
                "grasp_attempt_id": grasp_attempt_id,
                **({"motion": motion} if motion else {}),
                **(
                    {"motion_plan": remote.get("motion_plan")}
                    if remote.get("motion_plan")
                    else {}
                ),
                **(
                    {"approach_gripper": remote.get("approach_gripper")}
                    if isinstance(remote.get("approach_gripper"), Mapping)
                    else {}
                ),
                **({"gripper": operator_gripper} if operator_gripper else {}),
                **(
                    {
                        "contact_status": "unverified",
                        "required_next_tool": "verify_grasp",
                    }
                    if stage == "lift_grasp" and success and not remote_task_success
                    else {}
                ),
                **(
                    {
                        "checkpoint_status": "ready",
                        "required_next_tool": "approach_selected_grasp",
                        "checkpoint_observation_id": self.current_record.get(
                            "observation_id"
                        ),
                    }
                    if checkpoint_ready and self.current_record is not None
                    else {}
                ),
                **(
                    {
                        "checkpoint_status": "blocked",
                        "wrist_checkpoint": wrist_checkpoint,
                    }
                    if checkpoint_evidence_error is not None
                    else {}
                ),
                **(
                    {"wrist_checkpoint": wrist_checkpoint}
                    if checkpoint_ready and wrist_checkpoint is not None
                    else {}
                ),
                **({"post_observation_error": post_observation_error} if post_observation_error else {}),
                **({"execution_comparison_error": execution_comparison_error} if execution_comparison_error else {}),
                **({"remote_completion_unknown": True, "remote_action_canceled": False} if remote_completion_unknown else {}),
            },
        )
        images = (
            self._operator_approach_frame_paths(
                self.current_record.get("frames", [])
            )
            if (
                self._active_grasp_attempt_id is not None
                and bool(post_frames)
            )
            and self.current_record
            else self._operator_frame_paths(self.current_record.get("frames", []))
            if self.current_record
            else []
        )
        result = GatewayResult(
            success,
            {
                "kind": "step",
                "success": success,
                "stage": stage,
                "action_id": str(action.get("action_id") or ""),
                "grasp_attempt_id": grasp_attempt_id,
                "observation_id": self.current_record.get("observation_id") if self.current_record else None,
                "message": content,
                **({"motion": operator_motion} if operator_motion else {}),
                **(
                    {"motion_plan": remote.get("motion_plan")}
                    if remote.get("motion_plan")
                    else {}
                ),
                **(
                    {"approach_gripper": remote.get("approach_gripper")}
                    if isinstance(remote.get("approach_gripper"), Mapping)
                    else {}
                ),
                **({"gripper": operator_gripper} if operator_gripper else {}),
                **(
                    {
                        "contact_status": "unverified",
                        "required_next_tool": "verify_grasp",
                    }
                    if stage == "lift_grasp" and success and not remote_task_success
                    else {}
                ),
                **(
                    {
                        "checkpoint_status": "ready",
                        "required_next_tool": "approach_selected_grasp",
                        "checkpoint_observation_id": self.current_record.get(
                            "observation_id"
                        ),
                    }
                    if checkpoint_ready and self.current_record is not None
                    else {}
                ),
                **(
                    {
                        "checkpoint_status": "blocked",
                        "wrist_checkpoint": wrist_checkpoint,
                    }
                    if checkpoint_evidence_error is not None
                    else {}
                ),
                **(
                    {"wrist_checkpoint": wrist_checkpoint}
                    if checkpoint_ready and wrist_checkpoint is not None
                    else {}
                ),
                **(
                    {
                        "viser_execution_scene": (
                            "A PREGRASP TARGET/ACTUAL scene is being bound to this exact "
                            "post-action observation. TARGET is the standoff pose, not the "
                            "contact pose. Use get_grasp_inspector, orbit the focused poses, "
                            "and capture several current views before judging recovery."
                            if stage == "move_to_selected_pregrasp"
                            else "A TARGET/ACTUAL execution scene is being bound to this exact post-action observation. Use get_grasp_inspector, orbit the focused poses, and capture several current views before judging fine geometry."
                        )
                    }
                    if execution_comparison is not None
                    else {}
                ),
                **(
                    {
                        "viser_execution_scene": (
                            "Gripper-only action: no new TARGET/ACTUAL pose scene is "
                            "created. Inspect the returned RGB now and verify contact "
                            "with the next lift motion."
                        )
                    }
                    if stage in {"open_gripper", "close_gripper"}
                    else {}
                ),
                **(
                    {
                        "episode_status": "running",
                        "retryable": not remote_completion_unknown,
                        "issue_id": issue.get("issue_id"),
                        "issue_code": issue.get("code"),
                        "issue_component": issue.get("component"),
                    }
                    if issue
                    else {}
                ),
            },
            images=images,
            details={"remote": remote, **({"issue": issue} if issue else {})},
        )
        if stage == "open_gripper":
            self._active_grasp_attempt_id = None
            self._active_grasp_target = None
            self._last_lift_reference = None
            self._last_grasp_verification = None
        self._publish_world_action_outcome(
            action_generation=action_generation,
            stage=stage,
            action_id=str(action.get("action_id") or ""),
            tool_call_id=str(tool_call.get("tool_call_id") or ""),
            grasp_attempt_id=grasp_attempt_id,
            result=result,
        )
        return result

    def verify_grasp(self, detection_id: str | None = None) -> GatewayResult:
        """Verify that the grasp target moved with the end effector after lift.

        This is deliberately observation-only.  It re-segments the semantic
        target retained at grasp execution time and compares metric RGB-D
        geometry before and after the lift.  Ambiguous or occluded evidence is
        reported as ``unknown`` rather than guessed as successful contact.
        """

        self.ensure_started()
        assert self.current_record is not None and self.perception is not None and self.writer is not None
        if self.failure_case is not None:
            return self._failed_result("grasp_verification")
        target = self._active_grasp_target
        lift_reference = self._last_lift_reference
        input_frames = list(self.current_record.get("frame_ids", []))
        call = self.writer.record_tool_start(
            tool="verify_grasp",
            input_frames=input_frames,
            parameters={"detection_id": detection_id},
        )
        if not target or not lift_reference:
            message = (
                "No completed lift is bound to a retained grasp target. Execute "
                "move_to_selected_grasp, close_gripper, and lift_grasp first."
            )
            self.writer.record_tool_result(
                tool="verify_grasp",
                success=False,
                tool_call_id=call.get("tool_call_id"),
                input_frames=input_frames,
                result={"contact_status": "unknown", "reason": "lift_reference_required"},
            )
            return GatewayResult(
                False,
                {
                    "kind": "grasp_verification",
                    "success": False,
                    "contact_status": "unknown",
                    "reason": "lift_reference_required",
                    "message": message,
                },
                images=self._operator_frame_paths(self.current_record.get("frames", [])),
            )

        label = str(target.get("label") or "").strip()
        if not label:
            return GatewayResult(
                False,
                {
                    "kind": "grasp_verification",
                    "success": False,
                    "contact_status": "unknown",
                    "reason": "target_label_missing",
                    "message": "The executed grasp has no retained semantic target label.",
                },
                images=self._operator_frame_paths(self.current_record.get("frames", [])),
            )
        baseline_point = target.get("baseline_point_xy")
        point_prompts = None
        if (
            isinstance(baseline_point, list)
            and len(baseline_point) == 2
            and all(isinstance(value, (int, float)) for value in baseline_point)
        ):
            point_prompts = [
                {
                    "x": float(baseline_point[0]),
                    "y": float(baseline_point[1]),
                    "label": 1,
                }
            ]
        try:
            segmented = self.perception.segment_object(
                self.current_record,
                label,
                point_prompts=point_prompts,
            )
        except Exception as exc:  # noqa: BLE001 - verification must remain recoverable.
            segmented = ToolResult(
                False,
                f"verify_grasp segmentation failed: {exc}",
                {"reason": "exception"},
            )
        details = segmented.details if isinstance(segmented.details, Mapping) else {}
        images = self._segmentation_images(details)
        near_field_images = self._operator_approach_frame_paths(
            self.current_record.get("frames", [])
        )
        if not images:
            images = near_field_images
        elif self._active_grasp_attempt_id is not None:
            for path in near_field_images:
                if "wrist" in path.name and path not in images:
                    images.append(path)
        detections = details.get("detections")
        detections = detections if isinstance(detections, list) else []
        selected: Mapping[str, Any] | None = None
        current_centroid: np.ndarray | None = None
        frame = None
        if detection_id is not None:
            selected = next(
                (
                    item
                    for item in detections
                    if isinstance(item, Mapping)
                    and str(item.get("id") or "") == detection_id
                ),
                None,
            )
        elif len(detections) == 1 and isinstance(detections[0], Mapping):
            selected = detections[0]
        elif segmented.success and target.get("baseline_centroid_world_m") is not None:
            try:
                frame = resolve_observation_frame(
                    self.current_record,
                    artifact_root=self.root,
                    camera_id="agentview",
                )
                baseline = np.asarray(
                    target.get("baseline_centroid_world_m"), dtype=np.float64
                )
                nearest: tuple[float, Mapping[str, Any], np.ndarray] | None = None
                for detection in detections:
                    if not isinstance(detection, Mapping):
                        continue
                    mask_path = Path(str(detection.get("mask_ref") or ""))
                    if not mask_path.is_absolute():
                        mask_path = self.root / mask_path
                    centroid = np.median(
                        backproject_masked_world(frame, mask_path).points_world,
                        axis=0,
                    )
                    distance = float(np.linalg.norm(centroid - baseline))
                    if nearest is None or distance < nearest[0]:
                        nearest = (distance, detection, centroid)
                if nearest is not None and nearest[0] <= 0.05:
                    selected = nearest[1]
                    current_centroid = nearest[2]
            except Exception:
                pass

        status = "unknown"
        reason = "verification_segmentation_failed"
        metrics: dict[str, Any] = {}
        if segmented.success and selected is None:
            reason = "verification_detection_selection_required"
        elif segmented.success and selected is not None:
            mask_ref = selected.get("mask_ref")
            try:
                if frame is None:
                    frame = resolve_observation_frame(
                        self.current_record,
                        artifact_root=self.root,
                        camera_id="agentview",
                    )
                if current_centroid is None:
                    mask_path = Path(str(mask_ref))
                    if not mask_path.is_absolute():
                        mask_path = self.root / mask_path
                    current_points = backproject_masked_world(frame, mask_path).points_world
                    current_centroid = np.median(current_points, axis=0)
                baseline_centroid = np.asarray(
                    target.get("baseline_centroid_world_m"), dtype=np.float64
                )
                lift_start_eef = np.asarray(
                    lift_reference.get("start_eef_world_m"), dtype=np.float64
                )
                current_eef = self._current_world_from_grip_site()[:3, 3]
                status, reason, metrics = _classify_grasp_contact(
                    baseline_centroid_world_m=baseline_centroid,
                    current_centroid_world_m=current_centroid,
                    lift_start_eef_world_m=lift_start_eef,
                    current_eef_world_m=current_eef,
                )
            except Exception as exc:  # noqa: BLE001 - uncertain evidence is not terminal.
                reason = "verification_geometry_unavailable"
                metrics = {"error": _exception_message(exc)}

        verification = {
            "schema_version": "openeta.grasp_contact_verification.v1",
            "grasp_attempt_id": self._active_grasp_attempt_id,
            "target_label": label,
            "baseline_observation_id": target.get("observation_id"),
            "current_observation_id": self.current_record.get("observation_id"),
            "detection_id": selected.get("id") if isinstance(selected, Mapping) else detection_id,
            "contact_status": status,
            "reason": reason,
            "metrics": metrics,
            "probe": "baseline_positive_point" if point_prompts else "semantic_text",
        }
        artifact = (
            self.root
            / "control"
            / "grasp-verification"
            / f"{_safe_name(self._active_grasp_attempt_id or 'unbound')}.json"
        )
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            json.dumps(verification, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        self._last_grasp_verification = verification
        issue = None
        if status == "not_grasped":
            issue = self._record_issue(
                category="action",
                component="grasp_contact",
                code="grasp_contact_failure",
                message=(
                    "The target remained near its pre-grasp position while the "
                    "end effector lifted. Contact was not retained."
                ),
                tool="verify_grasp",
                input_frames=input_frames,
                artifact_refs=[str(artifact), *_artifact_refs(details)],
                details=verification,
            )
        result_success = status == "confirmed"
        self.writer.record_tool_result(
            tool="verify_grasp",
            success=result_success,
            tool_call_id=call.get("tool_call_id"),
            input_frames=input_frames,
            artifact_refs=[str(artifact), *_artifact_refs(details)],
            result={
                "contact_status": status,
                "reason": reason,
                "detection_ids": [
                    str(item.get("id"))
                    for item in detections
                    if isinstance(item, Mapping) and item.get("id") is not None
                ],
            },
        )
        if status == "confirmed":
            message = "Target motion is consistent with retained gripper contact."
        elif status == "not_grasped":
            message = (
                "Target did not follow the lift. Reopen the gripper, return to the "
                "target, and change one grasp assumption before retrying."
            )
        elif segmented.success and len(detections) > 1 and detection_id is None:
            message = (
                "Multiple current target detections are visible. Inspect the mask "
                "image and call verify_grasp(detection_id) for the executed instance."
            )
        else:
            message = (
                "Contact evidence is inconclusive. Do not begin transport; inspect "
                "the returned image or obtain another observation and retry verification."
            )
        return GatewayResult(
            result_success,
            {
                "kind": "grasp_verification",
                "success": result_success,
                "contact_status": status,
                "reason": reason,
                "target": label,
                "observation_id": self.current_record.get("observation_id"),
                "detection_ids": [
                    str(item.get("id"))
                    for item in detections
                    if isinstance(item, Mapping) and item.get("id") is not None
                ],
                "message": message,
                **(
                    {"issue_id": issue.get("issue_id"), "issue_code": issue.get("code")}
                    if issue
                    else {}
                ),
            },
            images=images,
            details={"verification": verification, "segmentation": dict(details)},
        )

    def check_task(self) -> GatewayResult:
        self.ensure_started()
        assert self.writer is not None and self.current_record is not None
        if self.failure_case is not None:
            return self._failed_result("task_check")
        call = self.writer.record_tool_start(
            tool="check_task",
            input_frames=self.current_record.get("frame_ids", []),
        )
        try:
            if self._task_success_latch is not None:
                payload = {
                    "available": True,
                    "success": True,
                    "latched": True,
                    "source": self._task_success_latch.get("source"),
                }
                task_success = True
                checker_call_success = True
                success = True
                message = (
                    "Native task success was already confirmed during motion. "
                    "No simulator checker call was needed; call finish_episode "
                    "with outcome='success' to finalize it."
                )
            else:
                payload = self.transport.call_tool(
                    "check_task",
                    self._session_args({"handle": self.handle}),
                    timeout_s=60.0,
                )
                available = bool(payload.get("available", False))
                task_success = payload.get("success") if available else None
                if not isinstance(task_success, bool):
                    task_success = None
                if task_success is True:
                    self.last_task_success = True
                    self._task_success_latch = {
                        "source": "native_task_checker",
                        "observation_id": self.current_record.get("observation_id"),
                    }
                elif self.last_task_success is not True:
                    self.last_task_success = task_success
                checker_call_success = available and task_success is not None
                success = checker_call_success
                if not checker_call_success:
                    message = "Native task checker unavailable."
                elif task_success is False:
                    message = (
                        "Native task checker reports the task is not satisfied yet; "
                        "the episode remains running, so inspect, recover, and retry."
                    )
                else:
                    message = (
                        "Native task checker reports task success. Call finish_episode "
                        "with outcome='success' to finalize it."
                    )
        except Exception as exc:  # noqa: BLE001
            payload = {"available": False, "error": str(exc)}
            task_success = None
            if self._task_success_latch is None:
                self.last_task_success = None
            checker_call_success = False
            success = False
            message = f"Task checker failed: {exc}"
        issue = None
        if not checker_call_success:
            issue = self._record_issue(
                category="verification",
                component="simulator_checker",
                code="task_checker_unavailable",
                message=message,
                tool="check_task",
                input_frames=list(self.current_record.get("frame_ids", [])),
                details={"checker": payload},
            )
        self.writer.record_tool_result(
            tool="check_task",
            success=checker_call_success,
            tool_call_id=call.get("tool_call_id"),
            input_frames=self.current_record.get("frame_ids", []),
            result={
                "checker_call_success": checker_call_success,
                "task_success": task_success,
                "available": bool(payload.get("available", False)),
                "builder_diagnostics": payload.get("diagnostics"),
            },
        )
        return GatewayResult(
            success,
            {
                "kind": "task_check",
                "success": success,
                "task_success": task_success,
                "authority": "native_simulator_checker",
                "episode_status": self._episode_status,
                "observation_id": self.current_record.get("observation_id"),
                "message": message,
                **(
                    {
                        "episode_status": "running",
                        "retryable": True,
                        "issue_id": issue.get("issue_id"),
                        "issue_code": issue.get("code"),
                    }
                    if issue
                    else {}
                ),
            },
            images=self._operator_frame_paths(self.current_record.get("frames", [])),
            details={"checker": payload},
        )

    def report_issue(self, component: str, code: str, message: str) -> GatewayResult:
        self.ensure_started()
        assert self.writer is not None and self.current_record is not None
        if self.failure_case is not None:
            return self._failed_result("report_issue")
        component = str(component).strip()
        code = str(code).strip()
        message = str(message).strip()
        if not component or not code or not message:
            return GatewayResult(
                False,
                {
                    "kind": "report_issue",
                    "success": False,
                    "retryable": True,
                    "error": "component, code, and message must all be non-empty.",
                },
            )
        frames = list(self.current_record.get("frame_ids", []))
        call = self.writer.record_tool_start(
            tool="report_issue",
            input_frames=frames,
            parameters={"component": component, "code": code},
        )
        action_authority = self._world_action_timeout_report_authority(
            component=component,
            code=code,
        )
        if action_authority is not None:
            guarded_result = {
                "issue_recorded": False,
                "requested_issue_code": code,
                **action_authority,
            }
            self.writer.record_tool_result(
                tool="report_issue",
                success=False,
                tool_call_id=call.get("tool_call_id"),
                input_frames=frames,
                result=guarded_result,
            )
            return GatewayResult(
                False,
                {
                    "kind": "report_issue",
                    "success": False,
                    "issue_recorded": False,
                    "retryable": False,
                    "requested_issue_code": code,
                    "message": (
                        "World-action lifecycle is owned by the original action "
                        "call. No timeout issue was recorded; use the authoritative "
                        "action state returned here."
                    ),
                    **action_authority,
                },
                images=(
                    self._operator_approach_frame_paths(
                        self.current_record.get("frames", [])
                    )
                    if self._active_grasp_attempt_id is not None
                    else self._operator_frame_paths(
                        self.current_record.get("frames", [])
                    )
                ),
                details={"action_authority": action_authority},
            )
        issue = self._record_issue(
            category="operator_observed",
            component=component,
            code=code,
            message=message,
            tool="report_issue",
            input_frames=frames,
            details={
                "authority": "operator_visual_observation",
                "grasp_attempt_id": self._active_grasp_attempt_id,
            },
        )
        self.writer.record_tool_result(
            tool="report_issue",
            success=True,
            tool_call_id=call.get("tool_call_id"),
            input_frames=frames,
            result={
                "issue_id": issue.get("issue_id"),
                "component": component,
                "code": code,
                "grasp_attempt_id": self._active_grasp_attempt_id,
            },
        )
        return GatewayResult(
            True,
            {
                "kind": "report_issue",
                "success": True,
                "recorded": True,
                "episode_status": "running",
                "next_step": "continue_recovery",
                "message": (
                    "Observed attempt recorded. This does not establish a tool "
                    "failure. Continue with a changed, evidence-based pose and "
                    "verify the intended object is retained before transport."
                ),
            },
            images=self._operator_frame_paths(self.current_record.get("frames", [])),
        )

    def report_failure(self, component: str, code: str, message: str) -> GatewayResult:
        """Backward-compatible nonterminal alias for attempt failure reports."""

        result = self.report_issue(component, code, message)
        if result.text.get("kind") == "report_issue":
            result.text["kind"] = "report_failure"
            result.text["compatibility_alias"] = "report_issue_nonterminal"
        return result

    def finish_episode(
        self,
        outcome: str,
        reason: str = "",
        failure_postmortem: Mapping[str, Any] | None = None,
        operator_feedback: Mapping[str, Any] | None = None,
    ) -> GatewayResult:
        """Explicitly finalize success, failure, or operator abort."""

        self.ensure_started()
        assert self.writer is not None and self.current_record is not None
        outcome = str(outcome).strip().lower()
        reason = str(reason).strip()
        if outcome not in {"success", "failure", "abort"}:
            return GatewayResult(
                False,
                {
                    "kind": "episode_finish",
                    "success": False,
                    "retryable": True,
                    "error": "outcome must be success, failure, or abort.",
                },
            )
        if self._task_success_latch is not None and outcome != "success":
            return GatewayResult(
                False,
                {
                    "kind": "episode_finish",
                    "success": False,
                    "retryable": False,
                    "episode_status": "running",
                    "reason": "task_already_completed",
                    "issue_code": "task_success_latched",
                    "message": (
                        "Native task success is already confirmed. Finalize with "
                        "outcome='success'; failure or abort would discard a "
                        "completed task."
                    ),
                },
                details={"task_success_latch": dict(self._task_success_latch)},
            )
        normalized_feedback: dict[str, Any] | None = None
        if operator_feedback is not None:
            if not isinstance(operator_feedback, Mapping):
                return GatewayResult(
                    False,
                    {
                        "kind": "episode_finish",
                        "success": False,
                        "retryable": True,
                        "episode_status": "running",
                        "reason": "invalid_operator_feedback",
                        "missing_or_invalid_fields": ["operator_feedback"],
                    },
                )
            feedback_list_fields = (
                "tool_contract_issues",
                "context_issues",
                "redundant_information",
                "missing_capabilities",
                "helpful_evidence",
            )
            invalid_feedback_fields: list[str] = []
            normalized_feedback = {}
            for field in feedback_list_fields:
                values = operator_feedback.get(field, [])
                if (
                    not isinstance(values, list)
                    or any(not str(value).strip() for value in values)
                ):
                    invalid_feedback_fields.append(f"operator_feedback.{field}")
                else:
                    normalized_feedback[field] = [
                        str(value).strip() for value in values
                    ]
            blocked_step = operator_feedback.get("blocked_step", "")
            if not isinstance(blocked_step, str):
                invalid_feedback_fields.append("operator_feedback.blocked_step")
            else:
                normalized_feedback["blocked_step"] = blocked_step.strip()
            confidence = operator_feedback.get("confidence", "medium")
            if confidence not in {"low", "medium", "high"}:
                invalid_feedback_fields.append("operator_feedback.confidence")
            else:
                normalized_feedback["confidence"] = str(confidence)
            if invalid_feedback_fields:
                return GatewayResult(
                    False,
                    {
                        "kind": "episode_finish",
                        "success": False,
                        "retryable": True,
                        "episode_status": "running",
                        "reason": "invalid_operator_feedback",
                        "missing_or_invalid_fields": sorted(set(invalid_feedback_fields)),
                    },
                )
        normalized_postmortem: dict[str, Any] | None = None
        if outcome == "failure":
            allowed_stages = {
                "target_identification",
                "point_authoring",
                "pose_or_orientation",
                "approach",
                "grasp_contact",
                "retention",
                "transport",
                "placement",
                "task_verification",
                "recovery",
                "unknown",
            }
            allowed_layers = {
                "operator_strategy",
                "prompt_or_context",
                "tool_contract",
                "visualization",
                "perception_geometry",
                "motion_controller",
                "gripper_or_contact",
                "simulator_or_task",
                "unknown",
            }
            supplied = (
                dict(failure_postmortem)
                if isinstance(failure_postmortem, Mapping)
                else {}
            )
            invalid_fields = []
            if supplied.get("progress_stopped_at") not in allowed_stages:
                invalid_fields.append("progress_stopped_at")
            for field in ("expected_observation", "actual_observation"):
                if not str(supplied.get(field) or "").strip():
                    invalid_fields.append(field)
            for field in ("evidence_refs", "recovery_attempts"):
                values = supplied.get(field)
                if (
                    not isinstance(values, list)
                    or not values
                    or any(not str(value).strip() for value in values)
                ):
                    invalid_fields.append(field)
            hypotheses = supplied.get("diagnostic_hypotheses")
            if hypotheses is not None and (
                not isinstance(hypotheses, list) or not hypotheses
            ):
                invalid_fields.append("diagnostic_hypotheses")
            elif hypotheses is not None:
                for index, hypothesis in enumerate(hypotheses):
                    prefix = f"diagnostic_hypotheses[{index}]"
                    if not isinstance(hypothesis, Mapping):
                        invalid_fields.append(prefix)
                        continue
                    if hypothesis.get("suspected_layer") not in allowed_layers:
                        invalid_fields.append(f"{prefix}.suspected_layer")
                    if hypothesis.get("confidence") not in {
                        "low",
                        "medium",
                        "high",
                    }:
                        invalid_fields.append(f"{prefix}.confidence")
                    for field in (
                        "explanation",
                        "supporting_evidence",
                        "missing_or_conflicting_evidence",
                    ):
                        if not str(hypothesis.get(field) or "").strip():
                            invalid_fields.append(f"{prefix}.{field}")
            intervention = supplied.get("proposed_intervention")
            intervention_text_fields = (
                "independent_variable",
                "control_condition",
                "treatment_condition",
                "predicted_effect",
                "primary_metric",
                "adoption_criterion",
            )
            allowed_execution_scopes = {
                "current_tools",
                "system_change",
                "external_or_unavailable",
            }
            if intervention is not None and not isinstance(intervention, Mapping):
                invalid_fields.append("proposed_intervention")
            elif isinstance(intervention, Mapping):
                for field in intervention_text_fields:
                    if not str(intervention.get(field) or "").strip():
                        invalid_fields.append(f"proposed_intervention.{field}")
                held_constant = intervention.get("held_constant")
                if (
                    not isinstance(held_constant, list)
                    or not held_constant
                    or any(not str(value).strip() for value in held_constant)
                ):
                    invalid_fields.append(
                        "proposed_intervention.held_constant"
                    )
                execution_scope = intervention.get("execution_scope")
                if execution_scope not in allowed_execution_scopes:
                    invalid_fields.append(
                        "proposed_intervention.execution_scope"
                    )
                current_tool_plan = intervention.get("current_tool_plan")
                if not isinstance(current_tool_plan, list) or any(
                    not str(value).strip() for value in current_tool_plan
                ):
                    invalid_fields.append(
                        "proposed_intervention.current_tool_plan"
                    )
                elif execution_scope == "current_tools" and not current_tool_plan:
                    invalid_fields.append(
                        "proposed_intervention.current_tool_plan"
                    )
                attempted_in_episode = intervention.get(
                    "attempted_in_episode"
                )
                if not isinstance(attempted_in_episode, bool):
                    invalid_fields.append(
                        "proposed_intervention.attempted_in_episode"
                    )
                attempt_evidence_refs = intervention.get(
                    "attempt_evidence_refs"
                )
                if not isinstance(attempt_evidence_refs, list) or any(
                    not str(value).strip()
                    for value in attempt_evidence_refs
                ):
                    invalid_fields.append(
                        "proposed_intervention.attempt_evidence_refs"
                    )
                elif attempted_in_episode and not attempt_evidence_refs:
                    invalid_fields.append(
                        "proposed_intervention.attempt_evidence_refs"
                    )
                planned_trials = intervention.get(
                    "planned_current_tool_trials"
                )
                completed_trials = intervention.get(
                    "completed_current_tool_trials"
                )
                if (
                    not isinstance(planned_trials, int)
                    or isinstance(planned_trials, bool)
                    or planned_trials < 0
                ):
                    invalid_fields.append(
                        "proposed_intervention."
                        "planned_current_tool_trials"
                    )
                elif execution_scope == "current_tools" and planned_trials < 1:
                    invalid_fields.append(
                        "proposed_intervention."
                        "planned_current_tool_trials"
                    )
                if (
                    not isinstance(completed_trials, int)
                    or isinstance(completed_trials, bool)
                    or completed_trials < 0
                ):
                    invalid_fields.append(
                        "proposed_intervention."
                        "completed_current_tool_trials"
                    )
                remaining_actions = intervention.get(
                    "remaining_current_tool_actions"
                )
                if not isinstance(remaining_actions, list) or any(
                    not str(value).strip() for value in remaining_actions
                ):
                    invalid_fields.append(
                        "proposed_intervention."
                        "remaining_current_tool_actions"
                    )
                contradicting_attempts = intervention.get(
                    "contradicting_attempts"
                )
                if not isinstance(contradicting_attempts, list) or any(
                    not str(value).strip()
                    for value in contradicting_attempts
                ):
                    invalid_fields.append(
                        "proposed_intervention.contradicting_attempts"
                    )
                exhaustion_reason = str(
                    intervention.get(
                        "operator_claimed_exhaustion_reason"
                    )
                    or ""
                ).strip()
                if execution_scope in {
                    "current_tools",
                    "external_or_unavailable",
                } and attempted_in_episode and not exhaustion_reason:
                    invalid_fields.append(
                        "proposed_intervention."
                        "operator_claimed_exhaustion_reason"
                    )
                if (
                    execution_scope == "external_or_unavailable"
                    and not exhaustion_reason
                ):
                    invalid_fields.append(
                        "proposed_intervention."
                        "operator_claimed_exhaustion_reason"
                    )
            if invalid_fields:
                return GatewayResult(
                    False,
                    {
                        "kind": "episode_finish",
                        "success": False,
                        "retryable": True,
                        "episode_status": "running",
                        "reason": "failure_postmortem_required",
                        "missing_or_invalid_fields": sorted(set(invalid_fields)),
                        "questionnaire": {
                            "progress_stopped_at": sorted(allowed_stages),
                            "expected_observation": (
                                "What observable result should have happened?"
                            ),
                            "actual_observation": (
                                "What was observed instead, without assigning cause?"
                            ),
                            "evidence_refs": (
                                "Which observation IDs, views, coordinates, motion, "
                                "gripper, or checker results support this?"
                            ),
                            "recovery_attempts": (
                                "Which useful recoveries were tried and what happened?"
                            ),
                            "diagnostic_hypotheses": {
                                "suspected_layer": sorted(allowed_layers),
                                "explanation": "What falsifiable explanation fits?",
                                "supporting_evidence": "What evidence supports it?",
                                "missing_or_conflicting_evidence": (
                                    "What evidence is missing or conflicts?"
                                ),
                                "confidence": ["low", "medium", "high"],
                            },
                            "proposed_intervention": {
                                "independent_variable": (
                                    "What one factor should treatment change?"
                                ),
                                "control_condition": "What is the baseline?",
                                "treatment_condition": (
                                    "What differs only in that factor?"
                                ),
                                "held_constant": (
                                    "What model, seeds, task, tools, controller, "
                                    "and perception settings stay fixed?"
                                ),
                                "predicted_effect": (
                                    "What behavior should change if the hypothesis is true?"
                                ),
                                "primary_metric": (
                                    "What causal metric is declared before the run?"
                                ),
                                "adoption_criterion": (
                                    "What repeatable A/B result is required to adopt?"
                                ),
                                "execution_scope": sorted(
                                    allowed_execution_scopes
                                ),
                                "current_tool_plan": (
                                    "If current_tools, which exact current-tool "
                                    "actions execute it now?"
                                ),
                                "attempted_in_episode": (
                                    "Was this exact intervention attempted now?"
                                ),
                                "attempt_evidence_refs": (
                                    "Which trace or observation evidence proves "
                                    "that exact attempt?"
                                ),
                                "planned_current_tool_trials": (
                                    "How many matched trials are required before "
                                    "this current-tool intervention is exhausted?"
                                ),
                                "completed_current_tool_trials": (
                                    "How many of those matched trials were "
                                    "completed in this episode?"
                                ),
                                "remaining_current_tool_actions": (
                                    "Which declared current-tool actions remain?"
                                ),
                                "contradicting_attempts": (
                                    "Which attempts conflict with the hypothesis "
                                    "or predicted effect?"
                                ),
                                "operator_claimed_exhaustion_reason": (
                                    "Why is it exhausted or unavailable now? "
                                    "This remains an Operator claim."
                                ),
                            },
                        },
                        "message": (
                            "Failure is not finalized yet. Complete the structured "
                            "postmortem after recovery is exhausted. Hypotheses remain "
                            "unconfirmed, and the proposed change remains adoption-pending "
                            "until a controlled A/B meets its predeclared criterion."
                        ),
                    },
                    images=self._operator_frame_paths(
                        self.current_record.get("frames", [])
                    ),
                )
            if (
                isinstance(intervention, Mapping)
                and intervention["execution_scope"] == "current_tools"
                and (
                    intervention["attempted_in_episode"] is False
                    or intervention["completed_current_tool_trials"]
                    < intervention["planned_current_tool_trials"]
                    or bool(intervention["remaining_current_tool_actions"])
                )
            ):
                return GatewayResult(
                    False,
                    {
                        "kind": "episode_finish",
                        "success": False,
                        "retryable": True,
                        "episode_status": "running",
                        "reason": "executable_recovery_remaining",
                        "proposed_intervention": {
                            key: (
                                [str(value).strip() for value in raw_value]
                                if isinstance(raw_value, list)
                                else raw_value
                            )
                            for key, raw_value in intervention.items()
                        },
                        "message": (
                            "The proposed intervention is executable with current "
                            "tools and its declared plan is not exhausted. Complete "
                            "the remaining matched trials/actions and inspect their "
                            "evidence before finalizing failure."
                        ),
                    },
                    images=self._operator_frame_paths(
                        self.current_record.get("frames", [])
                    ),
                )
            normalized_postmortem = {
                "progress_stopped_at": str(supplied["progress_stopped_at"]),
                "expected_observation": str(
                    supplied["expected_observation"]
                ).strip(),
                "actual_observation": str(
                    supplied["actual_observation"]
                ).strip(),
                "evidence_refs": [
                    str(value).strip() for value in supplied["evidence_refs"]
                ],
                "recovery_attempts": [
                    str(value).strip()
                    for value in supplied["recovery_attempts"]
                ],
            }
            if hypotheses is not None:
                normalized_postmortem["diagnostic_hypotheses"] = [
                    {
                        "suspected_layer": str(hypothesis["suspected_layer"]),
                        "explanation": str(hypothesis["explanation"]).strip(),
                        "supporting_evidence": str(
                            hypothesis["supporting_evidence"]
                        ).strip(),
                        "missing_or_conflicting_evidence": str(
                            hypothesis["missing_or_conflicting_evidence"]
                        ).strip(),
                        "confidence": str(hypothesis["confidence"]),
                    }
                    for hypothesis in hypotheses
                ]
            if isinstance(intervention, Mapping):
                normalized_postmortem["proposed_intervention"] = {
                    **{
                        field: str(intervention[field]).strip()
                        for field in intervention_text_fields
                    },
                    "held_constant": [
                        str(value).strip()
                        for value in intervention["held_constant"]
                    ],
                    "execution_scope": str(intervention["execution_scope"]),
                    "current_tool_plan": [
                        str(value).strip()
                        for value in intervention["current_tool_plan"]
                    ],
                    "attempted_in_episode": bool(
                        intervention["attempted_in_episode"]
                    ),
                    "attempt_evidence_refs": [
                        str(value).strip()
                        for value in intervention["attempt_evidence_refs"]
                    ],
                    "planned_current_tool_trials": int(
                        intervention["planned_current_tool_trials"]
                    ),
                    "completed_current_tool_trials": int(
                        intervention["completed_current_tool_trials"]
                    ),
                    "remaining_current_tool_actions": [
                        str(value).strip()
                        for value in intervention["remaining_current_tool_actions"]
                    ],
                    "contradicting_attempts": [
                        str(value).strip()
                        for value in intervention["contradicting_attempts"]
                    ],
                    "operator_claimed_exhaustion_reason": str(
                        intervention.get("operator_claimed_exhaustion_reason") or ""
                    ).strip(),
                    "evidence_status": "hypothesis_only",
                    "adoption_status": "pending_controlled_ab",
                }
        if outcome == "success" and self.last_task_success is not True:
            return GatewayResult(
                False,
                {
                    "kind": "episode_finish",
                    "success": False,
                    "retryable": True,
                    "episode_status": "running",
                    "error": (
                        "Native task success has not been confirmed. Call check_task "
                        "and continue recovering if it reports false."
                    ),
                },
                images=self._operator_frame_paths(self.current_record.get("frames", [])),
            )
        active_action_unknown = self._invalidate_episode_generation_for_terminal()
        if outcome == "failure":
            self._mark_failure(
                category="episode_outcome",
                component="operator",
                code="operator_finished_failure",
                message=reason or "Operator explicitly finalized the episode as failure.",
                tool="finish_episode",
                input_frames=list(self.current_record.get("frame_ids", [])),
                details={
                    "issue_count": len(self.issues),
                    "operator_postmortem": normalized_postmortem,
                    **(
                        {"operator_feedback": normalized_feedback}
                        if normalized_feedback is not None
                        else {}
                    ),
                    **(
                        {"active_action_unknown": active_action_unknown}
                        if active_action_unknown is not None
                        else {}
                    ),
                },
            )
        self._requested_outcome = outcome
        self.close()
        return GatewayResult(
            self.close_error is None,
            {
                "kind": "episode_finish",
                "success": self.close_error is None,
                "outcome": outcome,
                "episode_status": self.episode_status,
                "episode_success": self.episode_success,
                "issue_count": len(self.issues),
                **(
                    {"failure_postmortem": normalized_postmortem}
                    if normalized_postmortem is not None
                    else {}
                ),
                **(
                    {"operator_feedback": normalized_feedback}
                    if normalized_feedback is not None
                    else {}
                ),
                "message": reason or f"Episode explicitly finalized as {outcome}.",
                **(
                    {
                        "issue_code": "active_action_unknown",
                        "active_action_unknown": active_action_unknown,
                    }
                    if active_action_unknown is not None
                    else {}
                ),
                **({"cleanup_error": self.close_error} if self.close_error else {}),
            },
            images=self._operator_frame_paths(self.current_record.get("frames", [])),
            details={
                **(
                    {"failure_postmortem": normalized_postmortem}
                    if normalized_postmortem is not None
                    else {}
                ),
                **(
                    {"active_action_unknown": active_action_unknown}
                    if active_action_unknown is not None
                    else {}
                ),
                **(
                    {"operator_feedback": normalized_feedback}
                    if normalized_feedback is not None
                    else {}
                ),
            },
        )

    def close(self) -> None:
        if self.writer is None or self._closed:
            return
        active_action_unknown = self._invalidate_episode_generation_for_terminal()
        self._episode_status = (
            "failed"
            if self.failure_case is not None
            else "completed"
            if self._requested_outcome == "success" or self.last_task_success is True
            else "aborted"
            if self._requested_outcome == "abort"
            else "stopped"
        )
        self.close_error = None
        try:
            if self.handle and active_action_unknown is None:
                self.transport.call_tool(
                    "close_env",
                    self._session_args({"handle": self.handle}),
                    timeout_s=15.0,
                )
        except Exception as exc:  # noqa: BLE001 - retain terminal evidence on cleanup failure.
            self.close_error = f"{type(exc).__name__}: {exc}"
            self._mark_failure(
                category="cleanup",
                component="simulator",
                code="cleanup_failed",
                message=f"Simulator cleanup failed: {self.close_error}",
                tool="close_episode",
                input_frames=list((self.current_record or {}).get("frame_ids", [])),
                details={"error": self.close_error},
            )
        finally:
            self._closed = True
            if self.failure_case is not None:
                self._episode_status = "failed"
            self.writer.finish(
                status=self._episode_status,
                success=(
                    False
                    if self.failure_case is not None
                    else True
                    if self._episode_status == "completed"
                    else None
                ),
                final_frames=(self.current_record or {}).get("frame_ids", []),
                result={
                    "task": self.task,
                    "sim_step": self._sim_step,
                    "issue_count": len(self.issues),
                    **({"requested_outcome": self._requested_outcome} if self._requested_outcome else {}),
                    **({"failure_case_id": self.failure_case.get("failure_case_id")} if self.failure_case else {}),
                    **({"cleanup_error": self.close_error} if self.close_error else {}),
                    **(
                        {
                            "active_action_unknown": active_action_unknown,
                            "remote_close_skipped": True,
                        }
                        if active_action_unknown is not None
                        else {}
                    ),
                },
            )
            self._write_current_projection()
            self.handle = ""

    # ------------------------------------------------------------------
    # Host-only helpers
    # ------------------------------------------------------------------

    def _session_args(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.session_id:
            payload["session_id"] = self.session_id
        return payload

    def _frame_paths(self, frames: Any) -> list[Path]:
        if not isinstance(frames, list):
            return []
        ordered = sorted(
            [frame for frame in frames if isinstance(frame, Mapping)],
            key=lambda frame: (0 if frame.get("camera_id") == "agentview" else 1, str(frame.get("camera_id", ""))),
        )
        paths: list[Path] = []
        for frame in ordered:
            if (
                frame.get("camera_id") == "agentview"
                and self._agentview_axis_overlay is not None
                and self._agentview_axis_overlay.is_file()
            ):
                paths.append(self._agentview_axis_overlay)
                continue
            if (
                frame.get("camera_id") == "wrist"
                and self._wrist_grip_site_overlay is not None
                and self._wrist_grip_site_overlay.is_file()
            ):
                paths.append(self._wrist_grip_site_overlay)
                continue
            raw = frame.get("rgb_path")
            if not isinstance(raw, str):
                continue
            path = self.root / raw if not Path(raw).is_absolute() else Path(raw)
            if path.is_file():
                paths.append(path)
        return paths

    def _operator_frame_paths(self, frames: Any) -> list[Path]:
        """Return the smallest visual payload for the normal Operator turn.

        Retained artifacts include every camera.  The Operator receives the
        primary agentview image by default; this keeps MCP image responses
        small enough for a Codex turn while preserving wrist/render frames for
        diagnostics and explicit future multi-view tools.
        """
        if not isinstance(frames, list):
            return []
        primary = [
            frame for frame in frames
            if isinstance(frame, Mapping) and frame.get("camera_id") == "agentview"
        ]
        return self._frame_paths(primary or frames[:1])

    def _operator_approach_frame_paths(self, frames: Any) -> list[Path]:
        """Return agentview plus wrist while a contact approach is pending."""

        if not isinstance(frames, list):
            return []
        close_range = [
            frame
            for frame in frames
            if isinstance(frame, Mapping)
            and frame.get("camera_id") in {"agentview", "wrist"}
        ]
        return self._frame_paths(close_range or frames[:1])

    def _segmentation_images(self, details: Mapping[str, Any]) -> list[Path]:
        bundle = details.get("selection_bundle")
        refs: list[str] = []
        if isinstance(bundle, Mapping):
            ref = bundle.get("contact_sheet_ref")
            if isinstance(ref, str):
                refs.append(ref)
        refs.extend(
            str(item.get("path"))
            for item in details.get("artifacts", [])
            if isinstance(item, Mapping) and item.get("type") == "sam3_candidate_overlay" and item.get("path")
        )
        return _existing_paths(refs)

    def _selected_move_arguments(self) -> tuple[dict[str, Any], dict[str, Any]]:
        assert self.selected_grasp is not None and self.current_record is not None
        transform = self._candidate_world_from_grip_site(self.selected_grasp)
        target = transform[:3, 3].tolist()
        control_rotation = transform[:3, :3].tolist()
        args: dict[str, Any] = {
            "handle": self.handle,
            "x": float(target[0]),
            "y": float(target[1]),
            "z": float(target[2]),
            "num_steps": 220,
            "tolerance": 0.01,
            "ori_tolerance": 0.08,
            "velocity_tolerance": 0.05,
            "settle_steps": 5,
            "enable_collision_check": True,
        }
        provenance = self.selected_grasp.get("provenance")
        controller = (
            provenance.get("controller")
            if isinstance(provenance, Mapping)
            else None
        )
        if isinstance(controller, Mapping):
            args["num_steps"] = int(controller.get("max_steps", args["num_steps"]))
            args["tolerance"] = float(
                controller.get("tolerance_m", args["tolerance"])
            )
        rpy = _rotation_to_euler_degrees(control_rotation)
        if rpy is None:
            raise RuntimeError("Panda grip-site rotation could not be converted to Euler angles")
        args.update({"roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2]})
        self._session_args(args)
        refined = (
            self.selected_grasp.get("pose_frame") == "world"
            and self.selected_grasp.get("eef_frame") == "panda_grip_site"
        )
        control_pose = {
            "frame": "world",
            "eef_frame": "panda_grip_site",
            "translation_xyz": [float(item) for item in target],
            "rotation_matrix": control_rotation,
            "source_backend": self.selected_grasp.get("source_backend")
            or "anygrasp",
            "source_grasp_id": self.selected_grasp.get("id"),
            "target_semantics": (
                "explicit_world_panda_grip_site"
                if refined
                else "anygrasp_jaw_center_to_panda_grip_site"
            ),
            "rotation_adapter": (
                "none_world_panda_grip_site"
                if refined
                else "anygrasp_to_panda_grip_site"
            ),
        }
        return control_pose, args

    def _lift_move_arguments(self) -> dict[str, Any]:
        robot = self.current_sim_payload.get("robot", {})
        pose = robot.get("end_effector_pose", {}) if isinstance(robot, Mapping) else {}
        xyz = pose.get("xyz") if isinstance(pose, Mapping) else None
        if not isinstance(xyz, list) or len(xyz) != 3:
            raise RuntimeError("Current observation has no EEF position for lift_grasp")
        args: dict[str, Any] = {
            "handle": self.handle,
            "x": float(xyz[0]),
            "y": float(xyz[1]),
            "z": float(xyz[2]) + 0.10,
            "num_steps": 120,
            "tolerance": 0.01,
            "velocity_tolerance": 0.05,
            "settle_steps": 5,
            "enable_collision_check": True,
        }
        self._session_args(args)
        return args

    def _write_control_artifact(self, *, action_id: str, payload: Mapping[str, Any]) -> Path:
        path = self.root / "control" / f"{action_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return path

    def _render_execution_comparison(
        self,
        *,
        executed_grasp: Mapping[str, Any],
        control_pose: Mapping[str, Any],
        action_id: str,
        execution_stage: str = "motion",
    ) -> Path | None:
        """Write exact commanded/observed grip-site transforms for Viser.

        No static pose image is rendered.  The retained post-action RGB-D frame
        and this comparison record are consumed by the observation-bound Viser
        supervisor, where the Operator can orbit and zoom before acting.
        """

        if self.current_record is None:
            return None
        agentview = _find_frame(self.current_record, "agentview")
        if agentview is None:
            return None
        metadata = agentview.get("metadata", {})
        extrinsics = metadata.get("extrinsics") if isinstance(metadata, Mapping) else None
        if not isinstance(extrinsics, Mapping):
            return None

        target_xyz = _finite_vector3(
            control_pose.get("translation_xyz"), name="control translation"
        )
        target_rotation = control_pose.get("rotation_matrix")
        target_rigid = np.eye(4, dtype=np.float64)
        target_rigid[:3, :3] = np.asarray(target_rotation, dtype=np.float64)
        target_rigid[:3, 3] = np.asarray(target_xyz, dtype=np.float64)
        target_rigid = rigid_transform(target_rigid, name="target_world_from_grip_site")

        robot = self.current_sim_payload.get("robot", {})
        pose = robot.get("end_effector_pose", {}) if isinstance(robot, Mapping) else {}
        actual_xyz = _finite_vector3(
            pose.get("xyz") if isinstance(pose, Mapping) else None,
            name="actual grip-site translation",
        )
        actual_rotation = _quat_xyzw_to_rotation_matrix(
            pose.get("quat_xyzw") if isinstance(pose, Mapping) else None
        )
        if actual_rotation is None:
            return None
        actual_rigid = np.eye(4, dtype=np.float64)
        actual_rigid[:3, :3] = actual_rotation
        actual_rigid[:3, 3] = np.asarray(actual_xyz, dtype=np.float64)
        actual_rigid = rigid_transform(actual_rigid, name="actual_world_from_grip_site")

        proposed_aperture = executed_grasp.get("width")
        if not isinstance(proposed_aperture, (int, float)) or not math.isfinite(
            float(proposed_aperture)
        ):
            proposed_aperture = None
        else:
            proposed_aperture = max(0.0, float(proposed_aperture))
        gripper_state = (
            robot.get("gripper_state", {}) if isinstance(robot, Mapping) else {}
        )
        measured_aperture = (
            gripper_state.get("aperture_m")
            if isinstance(gripper_state, Mapping)
            else None
        )
        if not isinstance(measured_aperture, (int, float)) or not math.isfinite(
            float(measured_aperture)
        ):
            measured_aperture = None
        else:
            measured_aperture = max(0.0, float(measured_aperture))
        compatibility_aperture = min(
            PANDA_MAX_APERTURE_M,
            proposed_aperture
            if proposed_aperture is not None
            else PANDA_MAX_APERTURE_M,
        )
        output_dir = (
            self.root
            / "control"
            / "execution-comparison"
            / _safe_name(action_id)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        position_error_m = float(
            np.linalg.norm(actual_rigid[:3, 3] - target_rigid[:3, 3])
        )
        rotation_delta = target_rigid[:3, :3].T @ actual_rigid[:3, :3]
        orientation_error_rad = float(
            math.acos(
                max(-1.0, min(1.0, (float(np.trace(rotation_delta)) - 1.0) / 2.0))
            )
        )
        diagnostics = {
            "schema_version": "openeta.execution_pose_comparison.v2",
            "action_id": action_id,
            "execution_stage": execution_stage,
            "target_semantics": control_pose.get("target_semantics"),
            "observation_id": self.current_record.get("observation_id"),
            "source_frame_id": agentview.get("frame_id"),
            "source_grasp_id": executed_grasp.get("id"),
            "eef_frame": "panda_grip_site",
            "jaw_width_m": compatibility_aperture,
            "jaw_width_semantics": "robot_clamped_compatibility",
            "proposed_jaw_width_m": proposed_aperture,
            "proposed_aperture_status": (
                "exceeds_robot_limit"
                if proposed_aperture is not None
                and proposed_aperture > PANDA_MAX_APERTURE_M + 1e-6
                else "within_robot_limit"
                if proposed_aperture is not None
                else "unknown"
            ),
            "robot_max_aperture_m": PANDA_MAX_APERTURE_M,
            "viewer_mesh_aperture_m": PANDA_MAX_APERTURE_M,
            "viewer_mesh_aperture_semantics": "physical_open_geometry",
            "measured_aperture_m": measured_aperture,
            "target_world_from_grip_site": target_rigid.tolist(),
            "actual_world_from_grip_site": actual_rigid.tolist(),
            "position_error_m": position_error_m,
            "orientation_error_rad": orientation_error_rad,
            "inspection_surface": "viser",
            "static_pose_rendering": False,
        }
        diagnostics_path = output_dir / "comparison.json"
        diagnostics_path.write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return diagnostics_path


def _build_anygrasp_depth_horizon_fallback(
    detect_grasps: AnyGraspDetectCallable,
    *,
    artifact_root: str | Path,
    backend_horizon_m: float = 1.0,
    retry_target_depth_m: float = 0.90,
) -> AnyGraspDetectCallable:
    """Retry an explicit AnyGrasp depth-horizon rejection by metric scaling.

    Some deployed AnyGrasp services retain a 1 m preprocessing horizon even
    though LIBERO's fixed agentview can place valid target objects at 1.1--1.3
    m.  The fallback runs only after the backend returns
    ``empty_target_mask`` and every valid target pixel is beyond that horizon.
    It uniformly scales the depth-derived point cloud into range, then maps the
    returned camera translation and jaw width back to the original metric
    scene.  Rotation and physical gripper depth/height remain unchanged.
    """

    root = Path(artifact_root).expanduser().resolve()

    def wrapped(request: dict[str, Any]) -> dict[str, Any]:
        first = detect_grasps(request)
        details = first.get("details") if isinstance(first, Mapping) else None
        reason = details.get("reason") if isinstance(details, Mapping) else None
        if first.get("success") is True or reason != "empty_target_mask":
            return first

        depth_payload = request.get("depth")
        mask_payload = request.get("target_mask")
        intrinsics = request.get("intrinsics")
        if not (
            isinstance(depth_payload, Mapping)
            and isinstance(mask_payload, Mapping)
            and isinstance(intrinsics, Mapping)
        ):
            return first
        try:
            depth_image = _decode_payload_image(depth_payload)
            mask_image = _decode_payload_image(mask_payload)
            depth_array = np.asarray(depth_image)
            mask_array = np.asarray(mask_image) > 0
            scale = float(intrinsics.get("scale"))
        except Exception:  # noqa: BLE001 - retain the original backend result.
            return first
        if (
            depth_array.ndim != 2
            or mask_array.shape != depth_array.shape
            or not math.isfinite(scale)
            or scale <= 0
        ):
            return first
        target_raw = depth_array[mask_array]
        target_raw = target_raw[target_raw > 0]
        if target_raw.size == 0:
            return first
        target_metric = target_raw.astype(np.float64) / scale
        target_min = float(np.min(target_metric))
        target_p95 = float(np.quantile(target_metric, 0.95))
        if target_min < backend_horizon_m:
            return first
        depth_scale_factor = min(1.0, retry_target_depth_m / target_p95)
        if not 0.0 < depth_scale_factor < 0.999:
            return first

        integer_max = np.iinfo(depth_array.dtype).max if depth_array.dtype.kind in "ui" else None
        scaled = depth_array.astype(np.float64) * depth_scale_factor
        if integer_max is not None:
            scaled = np.clip(np.rint(scaled), 0, integer_max).astype(depth_array.dtype)
        else:
            scaled = scaled.astype(depth_array.dtype)
        encoded_depth = _encode_payload_image(scaled, format_hint=str(depth_payload.get("format") or "png"))

        root.mkdir(parents=True, exist_ok=True)
        stamp = f"{int(time.time() * 1_000_000)}"
        adapted_path = root / f"{stamp}-scaled-depth.png"
        _save_array_png(scaled, adapted_path)
        retry_request = dict(request)
        retry_request["depth"] = encoded_depth
        retried = detect_grasps(retry_request)
        retried_details = retried.get("details") if isinstance(retried, Mapping) else None
        if retried.get("success") is not True or not isinstance(retried_details, dict):
            return first

        inverse = 1.0 / depth_scale_factor
        candidates = retried_details.get("grasp_candidates")
        if not isinstance(candidates, list):
            return first
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            translation = candidate.get("translation_xyz")
            rotation = candidate.get("rotation_matrix")
            depth = candidate.get("depth")
            if isinstance(translation, list) and len(translation) == 3:
                translation = [float(value) * inverse for value in translation]
                candidate["translation_xyz"] = translation
            width = candidate.get("width")
            if isinstance(width, (int, float)):
                candidate["width"] = min(0.1, float(width) * inverse)
            if (
                isinstance(translation, list)
                and len(translation) == 3
                and isinstance(rotation, list)
                and len(rotation) == 3
                and isinstance(depth, (int, float))
            ):
                try:
                    approach = [float(rotation[index][0]) for index in range(3)]
                    candidate["gripper_tip_position_xyz"] = [
                        float(translation[index]) + approach[index] * float(depth)
                        for index in range(3)
                    ]
                except (IndexError, TypeError, ValueError):
                    pass
        metadata = retried_details.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            retried_details["metadata"] = metadata
        metadata["client_depth_horizon_fallback"] = {
            "backend_horizon_m": backend_horizon_m,
            "target_min_depth_m": target_min,
            "target_p95_depth_m": target_p95,
            "depth_scale_factor": depth_scale_factor,
            "metric_inverse_scale": inverse,
            "adapted_depth_ref": str(adapted_path),
        }
        retried["content"] = (
            str(retried.get("content") or "AnyGrasp grasp detection completed.")
            + " Client retried after the deployed depth horizon rejected the target mask."
        )
        return retried

    return wrapped


def _decode_payload_image(payload: Mapping[str, Any]):
    from PIL import Image

    encoded = payload.get("base64")
    if not isinstance(encoded, str):
        raise ValueError("missing image payload")
    return Image.open(BytesIO(base64.b64decode(encoded))).copy()


def _encode_payload_image(array: np.ndarray, *, format_hint: str) -> dict[str, str]:
    from PIL import Image

    buffer = BytesIO()
    image_format = "PNG" if format_hint.lower() == "png" else format_hint.upper()
    Image.fromarray(array).save(buffer, format=image_format)
    return {
        "format": format_hint.lower(),
        "base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def _save_array_png(array: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG")


def _find_frame(record: Mapping[str, Any], camera_id: str) -> Mapping[str, Any] | None:
    frames = record.get("frames")
    if not isinstance(frames, list):
        return None
    return next((frame for frame in frames if isinstance(frame, Mapping) and frame.get("camera_id") == camera_id), None)


def _frame_rgb_path(root: Path, frame: Mapping[str, Any]) -> Path | None:
    raw = frame.get("rgb_path")
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return path if path.is_file() else None


def _frame_depth_path(root: Path, frame: Mapping[str, Any]) -> Path | None:
    raw = frame.get("depth_path")
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return path if path.is_file() else None


def _existing_paths(refs: list[str]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for ref in refs:
        path = Path(ref)
        if path.is_file() and path not in seen:
            result.append(path)
            seen.add(path)
    return result


def _artifact_refs(details: Mapping[str, Any]) -> list[str]:
    refs: list[str] = []
    for item in details.get("artifacts", []) if isinstance(details.get("artifacts"), list) else []:
        if isinstance(item, Mapping) and isinstance(item.get("path"), str):
            refs.append(item["path"])
    for key in ("canonical_grasp_candidates_ref", "raw_output_ref"):
        if isinstance(details.get(key), str):
            refs.append(details[key])
    visualization = details.get("visualization")
    if isinstance(visualization, Mapping) and isinstance(visualization.get("overlay_ref"), str):
        refs.append(visualization["overlay_ref"])
    return refs


def _payload_ok(payload: Mapping[str, Any]) -> bool:
    return isinstance(payload, Mapping) and "error" not in payload and payload.get("success", True) is not False


def _semantic_targets_overlap(left: str, right: str) -> bool:
    """Return whether two visual labels share a meaningful token."""

    def tokens(value: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", value.lower())
            if len(token) >= 3 and token not in {"the", "object", "item"}
        }

    return bool(tokens(left) & tokens(right))


def _write_reference_detection_board(
    reference_path: Path,
    detection_sheet_path: Path,
    output_path: Path,
) -> None:
    """Put canonical reference and current detections in one labeled image."""

    from PIL import Image, ImageDraw

    with Image.open(reference_path) as source_reference:
        reference = source_reference.convert("RGB")
    with Image.open(detection_sheet_path) as source_detections:
        detections = source_detections.convert("RGB")
    reference.thumbnail((560, 760), Image.Resampling.LANCZOS)
    detections.thumbnail((1040, 900), Image.Resampling.LANCZOS)
    margin = 24
    gap = 32
    header = 52
    width = margin * 2 + reference.width + gap + detections.width
    height = margin * 2 + header + max(reference.height, detections.height)
    board = Image.new("RGB", (width, height), (18, 20, 24))
    draw = ImageDraw.Draw(board)
    reference_x = margin
    detections_x = margin + reference.width + gap
    draw.text((reference_x, margin), "CANONICAL REFERENCE", fill=(80, 220, 255))
    draw.text((detections_x, margin), "CURRENT DETECTIONS", fill=(255, 210, 80))
    board.paste(reference, (reference_x, margin + header))
    board.paste(detections, (detections_x, margin + header))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.png")
    board.save(temporary)
    temporary.replace(output_path)


def _remote_world_action_timeout_s(
    remote_name: str, arguments: Mapping[str, Any]
) -> float:
    """Allow slow local simulator control loops to return their real result."""

    if remote_name != "move_to":
        return 240.0
    steps = max(1, _int_or(arguments.get("num_steps"), 220))
    return max(300.0, 90.0 + 1.5 * steps)


def _pregrasp_move_arguments(
    control_pose: Mapping[str, Any],
    contact_arguments: Mapping[str, Any],
    *,
    standoff_m: float,
) -> dict[str, Any]:
    """Move opposite the selected pose's approach axis before contact."""

    translation = np.asarray(
        _finite_vector3(
            control_pose.get("translation_xyz"),
            name="selected grasp translation",
        ),
        dtype=np.float64,
    )
    rotation = rigid_transform(
        [
            [*control_pose["rotation_matrix"][0], translation[0]],
            [*control_pose["rotation_matrix"][1], translation[1]],
            [*control_pose["rotation_matrix"][2], translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        name="selected_grasp_world_from_grip_site",
    )[:3, :3]
    if not math.isfinite(float(standoff_m)) or float(standoff_m) <= 0.0:
        raise GraspPoseRefinementError("pregrasp standoff must be positive and finite")
    pregrasp = translation - rotation[:, 2] * float(standoff_m)
    arguments = dict(contact_arguments)
    arguments.update(
        {
            "x": float(pregrasp[0]),
            "y": float(pregrasp[1]),
            "z": float(pregrasp[2]),
            "num_steps": max(220, _int_or(contact_arguments.get("num_steps"), 220)),
            "tolerance": max(
                0.015,
                float(contact_arguments.get("tolerance", 0.01)),
            ),
            "tracking_stall_steps": PREGRASP_TRACKING_STALL_STEPS,
            "tracking_min_aligned_progress_m": (
                PREGRASP_TRACKING_MIN_ALIGNED_PROGRESS_M
            ),
            "tracking_cross_track_tolerance_m": (
                PREGRASP_TRACKING_CROSS_TRACK_TOLERANCE_M
            ),
            "tracking_min_error_improvement_m": (
                PREGRASP_TRACKING_MIN_ERROR_IMPROVEMENT_M
            ),
        }
    )
    return arguments


def _pregrasp_waypoint_arguments(
    open_result: Mapping[str, Any],
    pregrasp_arguments: Mapping[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Return a single direct move to the pregrasp standoff.

    Earlier versions split the pregrasp into clearance lift, translate, orient,
    and descend waypoints to avoid OSC-coupled Cartesian drift.  Live runs
    showed that the vertical lift phase stalls with suspected contact because
    the arm starts low and the long lift drives the gripper into the table or
    object.  A single move to the pregrasp standoff (already backed off the
    approach axis) keeps the path short and uses the pregrasp tracking stall
    guard to stop unreachable poses early without consuming the episode horizon.
    """

    observation = open_result.get("observation", open_result)
    robot = observation.get("robot", {}) if isinstance(observation, Mapping) else {}
    pose = robot.get("end_effector_pose", {}) if isinstance(robot, Mapping) else {}
    current_xyz = pose.get("xyz") if isinstance(pose, Mapping) else None
    if (
        not isinstance(current_xyz, list)
        or len(current_xyz) != 3
        or not all(isinstance(value, (int, float)) for value in current_xyz)
    ):
        raise RuntimeError(
            "gripper_open result has no current Panda grip-site position for "
            "the staged pregrasp path"
        )

    return [("pregrasp", dict(pregrasp_arguments))]


def _combine_pregrasp_waypoint_results(
    waypoint_results: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if not waypoint_results:
        raise RuntimeError("staged pregrasp produced no waypoint result")
    final_entry = waypoint_results[-1]
    final_result = final_entry.get("result")
    if not isinstance(final_result, Mapping):
        raise RuntimeError("staged pregrasp waypoint returned no result")
    result = dict(final_result)
    first_result = waypoint_results[0].get("result")
    if isinstance(first_result, Mapping) and isinstance(
        first_result.get("start"), Mapping
    ):
        result["start"] = dict(first_result["start"])
    result["steps_executed"] = sum(
        max(
            0,
            _int_or(
                entry.get("result", {}).get("steps_executed")
                if isinstance(entry.get("result"), Mapping)
                else 0,
                0,
            ),
        )
        for entry in waypoint_results
    )
    result["motion_phase"] = str(final_entry.get("phase") or "pregrasp")
    result["pregrasp_path"] = "direct_standoff"
    result["pregrasp_waypoints"] = [dict(entry) for entry in waypoint_results]
    return result


def _combine_staged_motion_results(
    pregrasp: Mapping[str, Any],
    contact: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve final contact convergence plus the full staged motion trace."""

    result = dict(contact)
    if isinstance(pregrasp.get("start"), Mapping):
        result["start"] = dict(pregrasp["start"])
    result["steps_executed"] = max(
        0, _int_or(pregrasp.get("steps_executed"), 0)
    ) + max(0, _int_or(contact.get("steps_executed"), 0))
    result["motion_phase"] = "contact"
    result["motion_plan"] = "staged_pregrasp_contact"
    result["pregrasp_result"] = dict(pregrasp)
    return result


def _is_timeout_exception(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return "timeout" in name or "timed out" in message or "timeout" in message




def _motion_details(payload: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "reached_target",
        "motion_converged",
        "steps_executed",
        "position_error_xyz",
        "position_error_m",
        "orientation_error_rad",
        "joint_velocity_max_abs",
        "pose_converged",
        "velocity_converged",
        "settling_converged",
        "settle_steps_required",
        "settle_steps_completed",
        "controller_error",
        "suspected_contact_constraint",
        "tracking",
    )
    return {key: payload[key] for key in keys if key in payload}


def _motion_payload_ok(payload: Mapping[str, Any]) -> tuple[bool, str]:
    """Validate actual motion completion independently of transport success."""
    if _task_terminal_success(payload):
        return True, "task completed during motion"
    if payload.get("motion_phase") == "approach_open":
        detail = str(payload.get("controller_error") or "").strip()
        return False, detail or "gripper did not reach the required approach aperture."
    if payload.get("controller_error") == "cartesian_tracking_stalled":
        tracking = payload.get("tracking")
        window = (
            tracking.get("latest_window")
            if isinstance(tracking, Mapping)
            else None
        )
        measurements: list[str] = []
        if isinstance(window, Mapping):
            aligned = window.get("command_aligned_progress_m")
            cross_track = window.get("cross_track_drift_m")
            improvement = window.get("position_error_improvement_m")
            if isinstance(aligned, (int, float)):
                measurements.append(
                    f"aligned progress={float(aligned) * 1000.0:.1f}mm"
                )
            if isinstance(cross_track, (int, float)):
                measurements.append(
                    f"cross-track drift={float(cross_track) * 1000.0:.1f}mm"
                )
            if isinstance(improvement, (int, float)):
                measurements.append(
                    f"error improvement={float(improvement) * 1000.0:.1f}mm"
                )
        detail = ", ".join(measurements)
        return False, (
            "Cartesian tracking stalled during the guarded contact approach"
            + (f" ({detail})" if detail else "")
            + ". This is consistent with a physical constraint but does not "
            "confirm object contact."
        )
    end = payload.get("end")
    end_xyz = end.get("xyz") if isinstance(end, Mapping) else None
    if payload.get("pose_feedback_available") is False or not (
        isinstance(end_xyz, list) and len(end_xyz) == 3
    ):
        detail = str(payload.get("controller_error") or "").strip()
        return False, (
            "move_to returned no final EEF pose"
            + (f" ({detail})" if detail else ".")
        )
    reached = payload.get("reached_target")
    if reached is False or payload.get("motion_converged") is False:
        position_error = payload.get("position_error_m")
        steps = payload.get("steps_executed")
        suffix = []
        if isinstance(position_error, (int, float)):
            suffix.append(f"position error={float(position_error):.3f}m")
        if isinstance(steps, int):
            suffix.append(f"steps={steps}")
        detail = ", ".join(suffix)
        return False, "move_to did not converge" + (f" ({detail})" if detail else ".")
    if reached is None and payload.get("motion_converged") is None:
        return False, "move_to returned no convergence result."
    return True, "move_to converged"


def _task_terminal_success(payload: Mapping[str, Any]) -> bool:
    reward = payload.get("reward")
    return (
        payload.get("terminated") is True
        and isinstance(reward, (int, float))
        and not isinstance(reward, bool)
        and float(reward) > 0.0
    )


def _operator_motion_feedback(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a small, human-readable convergence summary without raw poses."""

    feedback: dict[str, Any] = {}
    position_error = payload.get("position_error_m")
    if isinstance(position_error, (int, float)):
        feedback["position_error_mm"] = round(float(position_error) * 1000.0, 1)
    orientation_error = payload.get("orientation_error_rad")
    if isinstance(orientation_error, (int, float)):
        feedback["orientation_error_deg"] = round(math.degrees(float(orientation_error)), 1)
    steps = payload.get("steps_executed")
    if isinstance(steps, int):
        feedback["steps_executed"] = steps
    if "settling_converged" in payload:
        feedback["settled"] = bool(payload.get("settling_converged"))
    elif "velocity_converged" in payload:
        feedback["settled"] = bool(payload.get("velocity_converged"))
    controller_error = payload.get("controller_error")
    if isinstance(controller_error, str) and controller_error:
        feedback["controller_error"] = controller_error
    if payload.get("suspected_contact_constraint") is True:
        feedback["suspected_contact_constraint"] = True
    tracking = payload.get("tracking")
    window = (
        tracking.get("latest_window")
        if isinstance(tracking, Mapping)
        else None
    )
    if isinstance(window, Mapping):
        aligned = window.get("command_aligned_progress_m")
        cross_track = window.get("cross_track_drift_m")
        improvement = window.get("position_error_improvement_m")
        feedback["tracking"] = {
            **(
                {"aligned_progress_mm": round(float(aligned) * 1000.0, 1)}
                if isinstance(aligned, (int, float))
                else {}
            ),
            **(
                {"cross_track_drift_mm": round(float(cross_track) * 1000.0, 1)}
                if isinstance(cross_track, (int, float))
                else {}
            ),
            **(
                {"error_improvement_mm": round(float(improvement) * 1000.0, 1)}
                if isinstance(improvement, (int, float))
                else {}
            ),
        }
    return feedback


def _operator_gripper_feedback(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return measured finger state without guessing whether contact occurred."""

    observation = payload.get("observation", payload)
    if not isinstance(observation, Mapping):
        return {}
    robot = observation.get("robot", {})
    if not isinstance(robot, Mapping):
        return {}
    state = robot.get("gripper_state", {})
    if not isinstance(state, Mapping):
        return {}
    feedback: dict[str, Any] = {}
    if isinstance(state.get("open"), bool):
        feedback["open"] = bool(state["open"])
    aperture = state.get("aperture_m")
    if isinstance(aperture, (int, float)) and math.isfinite(float(aperture)):
        feedback["aperture_mm"] = round(float(aperture) * 1000.0, 2)
    velocities = state.get("finger_qvel")
    if isinstance(velocities, list):
        finite = [abs(float(value)) for value in velocities if isinstance(value, (int, float)) and math.isfinite(float(value))]
        if finite:
            feedback["finger_speed_max_m_s"] = round(max(finite), 6)
    return feedback


def _approach_gripper_open_payload_ok(
    payload: Mapping[str, Any],
) -> tuple[bool, str]:
    """Require measured jaw clearance before a selected-grasp approach.

    The simulator's coarse ``open`` flag flips near the middle of the Panda
    finger range, so it cannot establish that a wide object can pass between
    the fingers.  The selected-grasp primitive therefore gates motion on the
    measured aperture itself.
    """

    feedback = _operator_gripper_feedback(payload)
    aperture = feedback.get("aperture_mm")
    if not isinstance(aperture, (int, float)):
        return False, "gripper_open returned no measured aperture for approach."
    if float(aperture) < APPROACH_OPEN_MIN_APERTURE_MM:
        return (
            False,
            "gripper_open did not establish enough approach clearance "
            f"({float(aperture):.1f} mm < {APPROACH_OPEN_MIN_APERTURE_MM:.1f} mm).",
        )
    return True, "approach aperture established"


def _classify_grasp_contact(
    *,
    baseline_centroid_world_m: Any,
    current_centroid_world_m: Any,
    lift_start_eef_world_m: Any,
    current_eef_world_m: Any,
) -> tuple[str, str, dict[str, Any]]:
    """Classify generic post-lift contact from metric target/EEF displacement."""

    vectors = [
        np.asarray(value, dtype=np.float64)
        for value in (
            baseline_centroid_world_m,
            current_centroid_world_m,
            lift_start_eef_world_m,
            current_eef_world_m,
        )
    ]
    if any(vector.shape != (3,) or not np.isfinite(vector).all() for vector in vectors):
        raise ValueError("grasp verification requires four finite xyz vectors")
    baseline, current, lift_start, eef = vectors
    object_delta = current - baseline
    eef_delta = eef - lift_start
    object_displacement = float(np.linalg.norm(object_delta))
    object_lift = float(object_delta[2])
    eef_lift = float(eef_delta[2])
    object_to_eef = float(np.linalg.norm(current - eef))
    metrics = {
        "object_displacement_mm": round(object_displacement * 1000.0, 1),
        "object_lift_mm": round(object_lift * 1000.0, 1),
        "eef_lift_mm": round(eef_lift * 1000.0, 1),
        "object_to_eef_mm": round(object_to_eef * 1000.0, 1),
    }
    if eef_lift < 0.015:
        return "unknown", "insufficient_eef_lift", metrics
    if object_lift <= 0.012 and object_displacement <= 0.025:
        return "not_grasped", "target_remained_at_baseline", metrics
    required_object_lift = max(0.020, 0.35 * eef_lift)
    if object_lift >= required_object_lift and object_to_eef <= 0.18:
        return "confirmed", "target_followed_eef_lift", metrics
    return "unknown", "target_motion_inconclusive", metrics


def _gripper_action_payload_ok(
    stage: str,
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[bool, str]:
    """Reject measured gripper no-ops without guessing grasp contact."""

    if stage not in {"open_gripper", "close_gripper"}:
        return True, ""
    if not after:
        return False, f"{stage} returned no measured gripper state."
    before_aperture = before.get("aperture_mm")
    after_aperture = after.get("aperture_mm")
    before_value = (
        float(before_aperture)
        if isinstance(before_aperture, (int, float))
        else None
    )
    after_value = (
        float(after_aperture)
        if isinstance(after_aperture, (int, float))
        else None
    )
    after_open = after.get("open")
    if stage == "close_gripper":
        if after_open is False:
            return True, ""
        if after_value is not None and after_value <= 10.0:
            return True, ""
        if (
            before_value is not None
            and after_value is not None
            and before_value - after_value >= 2.0
        ):
            # The fingers moved toward closure and may have contacted a wide
            # object. Attachment is verified separately by lift_grasp.
            return True, ""
        detail = (
            f" ({before_value:.1f} -> {after_value:.1f} mm)"
            if before_value is not None and after_value is not None
            else ""
        )
        return (
            False,
            "close_gripper produced no effective measured closure"
            f"{detail}; gripper remains open.",
        )
    if after_open is True:
        return True, ""
    if after_value is not None and after_value >= 65.0:
        return True, ""
    if (
        before_value is not None
        and after_value is not None
        and after_value - before_value >= 2.0
    ):
        return True, ""
    detail = (
        f" ({before_value:.1f} -> {after_value:.1f} mm)"
        if before_value is not None and after_value is not None
        else ""
    )
    return False, f"open_gripper produced no effective measured opening{detail}."


def _gripper_action_outcome(
    stage: str,
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    action_completed: bool,
) -> dict[str, Any]:
    """Separate command completion from the measured requested state."""

    requested_state = (
        "open"
        if stage == "open_gripper"
        else "close"
        if stage == "close_gripper"
        else None
    )
    if requested_state is None:
        return {}
    before_aperture = before.get("aperture_mm")
    after_aperture = after.get("aperture_mm")
    before_value = (
        float(before_aperture)
        if isinstance(before_aperture, (int, float))
        else None
    )
    after_value = (
        float(after_aperture)
        if isinstance(after_aperture, (int, float))
        else None
    )
    measured_open = after.get("open")
    if requested_state == "open":
        # LIBERO's coarse ``open`` boolean flips near the middle of the finger
        # range.  When aperture is available it is the authoritative state:
        # an intermediate opening is useful motion, not full-open completion.
        reached = bool(
            after_value >= APPROACH_OPEN_MIN_APERTURE_MM
            if after_value is not None
            else measured_open is True
        )
        moved_toward_requested = bool(
            before_value is not None
            and after_value is not None
            and after_value > before_value + 0.5
        )
    else:
        # Likewise, ``open=False`` only means the fingers crossed the backend's
        # midpoint threshold.  It must not turn a wide contact aperture into a
        # claim that the requested near-closed endpoint was reached.
        reached = bool(
            after_value
            <= PANDA_NEAR_CLOSED_THRESHOLD_M * 1000.0
            if after_value is not None
            else measured_open is False
        )
        moved_toward_requested = bool(
            before_value is not None
            and after_value is not None
            and after_value < before_value - 0.5
        )
    return {
        "requested_state": requested_state,
        "action_completed": bool(action_completed),
        "requested_state_reached": reached,
        "moved_toward_requested_state": moved_toward_requested,
        "blocked_or_in_contact": bool(
            action_completed and moved_toward_requested and not reached
        ),
    }


def _gripper_close_evidence(after: Mapping[str, Any]) -> dict[str, Any]:
    """Describe close evidence without conflating contact with target retention."""

    aperture = after.get("aperture_mm")
    blocked = after.get("blocked_or_in_contact")
    if blocked is True:
        state = "closure_obstructed"
        closed_reference_mm = PANDA_NEAR_CLOSED_THRESHOLD_M * 1000.0
        if isinstance(aperture, (int, float)) and float(aperture) > closed_reference_mm:
            interpretation = (
                f"The measured aperture ({float(aperture):.1f} mm) stayed above "
                f"the configured closed reference ({closed_reference_mm:.1f} mm), "
                "so the fingers were physically obstructed. This is strong "
                "evidence of contact and is compatible with contact from an "
                "object between "
                "the pads, but it does not identify what was contacted or prove "
                "that the intended object is retained; verify target retention "
                "from a lift and object co-motion."
            )
        else:
            interpretation = (
                "The fingers stopped before the requested closed state, but the "
                "measured aperture provides no positive width evidence for an "
                "object wider than the closed reference. This is compatible "
                "with contact, but does not identify what was contacted or prove "
                "that an object is retained."
            )
    elif (
        isinstance(aperture, (int, float))
        and float(aperture)
        <= PANDA_NEAR_CLOSED_THRESHOLD_M * 1000.0
    ):
        state = "near_full_closure"
        interpretation = (
            "The fingers reached near-full closure with no measured obstruction. "
            "This provides no positive grasp evidence. It does not reveal whether "
            "the missed contact is above, below, or to either side; infer any "
            "correction direction from the returned geometry, then verify retention "
            "from object motion in a subsequent lift."
        )
    else:
        state = "closure_state_only"
        interpretation = (
            "Only finger state was measured. It does not determine a correction "
            "direction. Verify contact and retention from the returned geometry "
            "and object motion in a subsequent lift."
        )
    return {
        "state": state,
        "semantic_grasp_confirmed": False,
        "interpretation": interpretation,
    }


def _move_to_motion_status(
    payload: Mapping[str, Any],
    *,
    success: bool,
    motion_requested: bool,
) -> str:
    """Reduce controller evidence to a neutral endpoint-status vocabulary."""

    message = str(payload.get("message") or "").lower()
    gripper_result = payload.get("gripper_result")
    gripper_message = (
        str(gripper_result.get("message") or "").lower()
        if isinstance(gripper_result, Mapping)
        else ""
    )
    if (
        "native task completed during" in message
        or "native task completed during" in gripper_message
    ):
        return "task_completed"
    if payload.get("preview_only") is True:
        return "previewed"
    if not motion_requested:
        return "skipped"
    motion = payload.get("motion")
    motion_payload = motion if isinstance(motion, Mapping) else {}
    controller_error = str(
        motion_payload.get("controller_error")
        or payload.get("issue_code")
        or ""
    )
    if controller_error == "cartesian_tracking_stalled":
        return "stalled"
    if success:
        return "reached"
    if controller_error in {
        "motion_not_converged",
        "post_action_observation_failed",
    } or payload.get("issue_code") == "motion_not_converged":
        return "not_converged"
    return "failed"


def _move_to_gripper_summary(
    payload: Mapping[str, Any],
    *,
    requested_gripper: str | None,
    expose_current_aperture_after_skipped_request: bool = False,
) -> dict[str, Any] | None:
    """Return the small gripper contract exposed to the Operator.

    The detailed command state machine remains in ``move_to_full_result``.
    The public result intentionally stays small: it reports only the measured
    aperture, nominal open endpoint, physical closed endpoint, and near-closed
    recognition threshold.
    Contact and retention remain visual questions, not fields inferred by
    this summary.
    """

    def references() -> dict[str, float]:
        return {
            "open_aperture_mm": round(PANDA_MAX_APERTURE_M * 1000.0, 2),
            "closed_aperture_mm": round(
                PANDA_CLOSED_APERTURE_M * 1000.0,
                2,
            ),
            "near_closed_aperture_mm": round(
                PANDA_NEAR_CLOSED_THRESHOLD_M * 1000.0,
                2,
            ),
        }

    if requested_gripper is None:
        current = payload.get("current_gripper")
        measured_current = current if isinstance(current, Mapping) else {}
        aperture = measured_current.get("aperture_mm")
        if isinstance(aperture, (int, float)):
            return {
                "aperture_mm": round(float(aperture), 2),
                **references(),
            }
        return None
    if (
        payload.get("preview_only") is True
        or payload.get("gripper_skip_reason") == "preview_only"
    ):
        summary = {
            "requested_state": requested_gripper,
            **references(),
        }
        preview = payload.get("gripper_preview")
        if isinstance(preview, Mapping):
            summary["geometry_mode"] = preview.get("geometry_mode")
        return summary
    executed = bool(payload.get("gripper_executed"))
    nested = payload.get("gripper_result")
    nested_payload = nested if isinstance(nested, Mapping) else {}
    gripper = nested_payload.get("gripper")
    measured = gripper if isinstance(gripper, Mapping) else {}
    summary: dict[str, Any] = {
        "requested_state": requested_gripper,
    }
    aperture = measured.get("aperture_mm")
    if (
        not isinstance(aperture, (int, float))
        and not executed
        and expose_current_aperture_after_skipped_request
        and payload.get("gripper_skip_reason") == "motion_failed"
    ):
        current = payload.get("current_gripper")
        measured_current = current if isinstance(current, Mapping) else {}
        aperture = measured_current.get("aperture_mm")
    if isinstance(aperture, (int, float)):
        summary["aperture_mm"] = round(float(aperture), 2)
    summary.update(references())
    return summary


def _move_to_operator_result(
    payload: Mapping[str, Any],
    *,
    success: bool,
    motion_requested: bool,
    requested_gripper: str | None,
    returned_views: Sequence[str],
    observation_id: Any,
    orientation_requested: bool = False,
) -> dict[str, Any]:
    """Build the concise public result while full diagnostics stay in details."""

    invariants = active_profile().manifest.get("invariants", {})
    internal_status = _move_to_motion_status(
        payload,
        success=success,
        motion_requested=motion_requested,
    )
    compact_result = (
        invariants.get("operator_result_schema") == "compact_v1"
    )
    status = internal_status
    if compact_result:
        if internal_status == "skipped":
            status = "not_requested"
        elif internal_status in {"stalled", "not_converged", "failed"}:
            status = "not_reached"
    if (
        compact_result
        and not success
        and internal_status == "skipped"
        and requested_gripper is not None
        and payload.get("preview_only") is not True
    ):
        result: dict[str, Any] = {
            "success": False,
            "reason": str(
                payload.get("issue_code")
                or payload.get("reason")
                or "gripper_action_failed"
            ),
        }
        if payload.get("retryable") is not None:
            result["retryable"] = bool(payload["retryable"])
        if payload.get("message"):
            result["message"] = str(payload["message"])
        return result
    gripper = _move_to_gripper_summary(
        payload,
        requested_gripper=requested_gripper,
        expose_current_aperture_after_skipped_request=(
            invariants.get("move_to_failed_gripper_request_feedback")
            == "actual_aperture_v1"
        ),
    )
    spatial = _move_to_spatial_summary(
        payload,
        motion_requested=motion_requested,
    )

    result = {
        "success": bool(success),
        "observation_id": observation_id,
        "motion_status": status,
    }
    if compact_result:
        result["returned_views"] = list(returned_views)
    else:
        result["kind"] = "move_to"
    if (
        compact_result
        and internal_status == "previewed"
        and invariants.get("pose_preview_public_result") == "views_only_v1"
    ):
        # The candidate target is explicit in the call arguments and rendered
        # in the returned images. Mixing the current measured grip-site pose
        # into this result would combine actual-state text with candidate-state
        # imagery. Full diagnostics remain available in details and replay.
        if payload.get("preview_id"):
            result["preview_id"] = str(payload["preview_id"])
        return result
    actual_xyz = spatial.get("actual_grip_site_position_xyz_m")
    if actual_xyz is not None:
        result["actual_grip_site_xyz_m"] = actual_xyz
    residual = spatial.get("remaining_error_to_target_mm")
    if (
        internal_status in {"stalled", "not_converged", "failed"}
        and residual is not None
    ):
        result["remaining_target_delta_mm"] = residual
    if orientation_requested:
        approach = spatial.get("actual_approach_direction_world")
        jaw = spatial.get("actual_jaw_direction_world")
        if approach is not None and jaw is not None:
            result["actual_directions_world"] = {
                "approach": approach,
                "jaw": jaw,
            }
    if (
        gripper is not None
        and payload.get("preview_only") is not True
        and isinstance(gripper.get("aperture_mm"), (int, float))
    ):
        if compact_result:
            result["gripper_aperture_mm"] = gripper["aperture_mm"]
            if (
                invariants.get("move_to_aperture_reference_result")
                == "compact_v1"
            ):
                result["gripper_reference_mm"] = {
                    "open": gripper["open_aperture_mm"],
                    "closed": gripper["closed_aperture_mm"],
                    "near_closed": gripper["near_closed_aperture_mm"],
                }
        else:
            result["gripper"] = {
                "aperture_mm": gripper["aperture_mm"],
                "open_reference_mm": gripper["open_aperture_mm"],
                "closed_reference_mm": gripper["near_closed_aperture_mm"],
            }
    if (
        internal_status in {"stalled", "not_converged", "failed"}
        and not compact_result
    ):
        result["message"] = (
            "Endpoint not reached. Use the remaining world-frame delta and "
            "returned images to choose the next correction."
            if residual is not None
            else str(payload.get("message") or "move_to failed.")
        )
    if payload.get("retryable") is not None and not compact_result:
        result["retryable"] = bool(payload.get("retryable"))
    if payload.get("episode_status") is not None and not compact_result:
        result["episode_status"] = payload.get("episode_status")
    return result


def _pose_preview_artifact_id(
    *,
    observation_id: str,
    target_position_xyz_m: Sequence[float],
    target_rotation_matrix: Sequence[Sequence[float]],
    actual_world_from_grip_site: Sequence[Sequence[float]],
    requested_gripper: str | None,
    target_pad_contact_centers_world_m: Sequence[Sequence[float]] | None,
    target_pad_sweep_start_centers_world_m: Sequence[Sequence[float]] | None,
    target_pad_sweep_start_boxes: Sequence[Mapping[str, Any]] | None = None,
    target_pad_capture_corridor_boxes: (
        Sequence[Mapping[str, Any]] | None
    ) = None,
    identifier_prefix: str = "preview",
) -> str:
    """Return an immutable identity for one exact pose-preview rendering.

    Preview artifacts are replay evidence. Their identity must include inputs
    that change the rendered geometry, rather than sharing fixed filenames
    that a later preview can overwrite.
    """

    def finite_list(value: Any) -> list[Any] | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("pose preview artifact input must be finite")
        return array.tolist()

    payload = {
        "observation_id": str(observation_id),
        "target_position_xyz_m": finite_list(target_position_xyz_m),
        "target_rotation_matrix": finite_list(target_rotation_matrix),
        "actual_world_from_grip_site": finite_list(
            actual_world_from_grip_site
        ),
        "requested_gripper": requested_gripper,
        "target_pad_contact_centers_world_m": finite_list(
            target_pad_contact_centers_world_m
        ),
        "target_pad_sweep_start_centers_world_m": finite_list(
            target_pad_sweep_start_centers_world_m
        ),
    }
    if target_pad_sweep_start_boxes is not None:
        payload["target_pad_sweep_start_boxes"] = [
            {
                "center_world_m": finite_list(box.get("center_world_m")),
                "rotation_world": finite_list(box.get("rotation_world")),
                "half_size_m": finite_list(box.get("half_size_m")),
            }
            for box in target_pad_sweep_start_boxes
        ]
    if target_pad_capture_corridor_boxes is not None:
        payload["target_pad_capture_corridor_boxes"] = [
            {
                "center_world_m": finite_list(box.get("center_world_m")),
                "rotation_world": finite_list(box.get("rotation_world")),
                "half_size_m": finite_list(box.get("half_size_m")),
            }
            for box in target_pad_capture_corridor_boxes
        ]
    if identifier_prefix not in {"preview", "candidate"}:
        raise ValueError(
            "pose preview identifier prefix must be preview or candidate"
        )
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:12]
    return f"{identifier_prefix}-{digest}"


def _move_to_close_confirmation_key(
    *,
    observation_id: str,
    target_position_xyz_m: Sequence[float],
    target_rotation_matrix: Sequence[Sequence[float]],
    current_world_from_grip_site: Sequence[Sequence[float]],
) -> str:
    """Bind one close confirmation to the exact observed and resolved pose."""

    payload = {
        "observation_id": str(observation_id),
        "target_position_xyz_m": np.asarray(
            target_position_xyz_m,
            dtype=np.float64,
        ).round(9).tolist(),
        "target_rotation_matrix": np.asarray(
            target_rotation_matrix,
            dtype=np.float64,
        ).round(9).tolist(),
        "current_world_from_grip_site": np.asarray(
            current_world_from_grip_site,
            dtype=np.float64,
        ).round(9).tolist(),
        "gripper": "close",
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _move_to_close_preview_id(
    *,
    target_position_xyz_m: Sequence[float],
    target_rotation_matrix: Sequence[Sequence[float]],
    current_world_from_grip_site: Sequence[Sequence[float]],
) -> str:
    """Identify one frozen close candidate independently of observations."""

    payload = {
        "target_position_xyz_m": np.asarray(
            target_position_xyz_m,
            dtype=np.float64,
        ).round(9).tolist(),
        "target_rotation_matrix": np.asarray(
            target_rotation_matrix,
            dtype=np.float64,
        ).round(9).tolist(),
        "current_world_from_grip_site": np.asarray(
            current_world_from_grip_site,
            dtype=np.float64,
        ).round(9).tolist(),
        "gripper": "close",
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:12]
    return f"P-{digest}"


def _move_to_preview_base_matches(
    pending: Sequence[Sequence[float]],
    current: Sequence[Sequence[float]],
) -> bool:
    """Allow read-only observations but reject a physically changed base."""

    pending_transform = rigid_transform(
        pending,
        name="pending close preview base",
    )
    current_transform = rigid_transform(
        current,
        name="current close preview base",
    )
    translation_error_m = float(
        np.linalg.norm(
            pending_transform[:3, 3] - current_transform[:3, 3]
        )
    )
    relative_rotation = (
        pending_transform[:3, :3].T @ current_transform[:3, :3]
    )
    cosine = float(
        np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
    )
    rotation_error_deg = float(np.degrees(np.arccos(cosine)))
    return translation_error_m <= 0.0005 and rotation_error_deg <= 0.25


def _render_named_image_contact_sheet(
    images: Mapping[str, Path],
    *,
    output_path: Path,
    order: Sequence[str],
) -> Path:
    """Combine named inspection images without changing their geometry."""

    selected = [
        (name, Path(images[name]))
        for name in order
        if name in images and Path(images[name]).is_file()
    ]
    if not selected:
        raise ValueError("no pose preview images available for contact sheet")
    opened = [(name, Image.open(path).convert("RGB")) for name, path in selected]
    width = max(image.width for _name, image in opened)
    height = max(image.height for _name, image in opened)
    sheet = Image.new("RGB", (width * 2, height * 2), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for index, (name, image) in enumerate(opened):
        x = (index % 2) * width
        y = (index // 2) * height
        sheet.paste(image, (x, y))
        draw.text(
            (x + 8, y + 8),
            name,
            fill=(255, 255, 255),
            font=ImageFont.load_default(),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def _finite_xyz(value: Any) -> list[float] | None:
    try:
        xyz = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        return None
    return [round(float(component), 6) for component in xyz]


def _finite_unit_direction(value: Any) -> list[float] | None:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if vector.shape != (3,) or not np.isfinite(vector).all():
        return None
    length = float(np.linalg.norm(vector))
    if length < 1e-9:
        return None
    return [
        round(float(component), 6)
        for component in vector / length
    ]


def _move_to_spatial_summary(
    payload: Mapping[str, Any],
    *,
    motion_requested: bool,
) -> dict[str, Any]:
    """Return only the world-frame positions needed for closed-loop correction."""

    actual_pose = payload.get("actual_grip_site_pose")
    actual_pose = actual_pose if isinstance(actual_pose, Mapping) else {}
    actual_xyz = _finite_xyz(actual_pose.get("position_xyz_m"))
    actual_approach = _finite_unit_direction(
        actual_pose.get("approach_axis_world")
    )
    actual_jaw = _finite_unit_direction(actual_pose.get("jaw_axis_world"))

    target_xyz: list[float] | None = None
    target_approach: list[float] | None = None
    target_jaw: list[float] | None = None
    residual_mm: list[float] | None = None
    actual_delta_mm: list[float] | None = None
    point_references: list[dict[str, Any]] = []
    stale_point_ids: list[str] = []
    if motion_requested:
        resolution = payload.get("target_resolution")
        resolution = resolution if isinstance(resolution, Mapping) else {}
        target_pose = resolution.get("resolved_target_pose")
        target_pose = target_pose if isinstance(target_pose, Mapping) else {}
        target_xyz = _finite_xyz(target_pose.get("position_xyz_m"))
        target_approach = _finite_unit_direction(
            target_pose.get("approach_axis_world")
        )
        target_jaw = _finite_unit_direction(
            target_pose.get("jaw_axis_world")
        )
        residual_mm = _finite_xyz(payload.get("target_minus_actual_mm_world"))
        actual_delta_mm = _finite_xyz(
            payload.get("actual_position_delta_mm_world")
        )
        resolution_observation_id = resolution.get("observation_id")
        point_ids = resolution.get("point_ids")
        point_ids = point_ids if isinstance(point_ids, Mapping) else {}
        provenance = resolution.get("point_provenance_observation_ids")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        point_xyz = resolution.get("point_world_xyz_m")
        point_xyz = point_xyz if isinstance(point_xyz, Mapping) else {}
        for role, raw_point_id in point_ids.items():
            point_id = str(raw_point_id or "").strip()
            if not point_id:
                continue
            marked_observation_id = provenance.get(point_id)
            is_current = (
                marked_observation_id is not None
                and resolution_observation_id is not None
                and str(marked_observation_id) == str(resolution_observation_id)
            )
            reference: dict[str, Any] = {
                "role": str(role),
                "point_id": point_id,
                "marked_observation_id": marked_observation_id,
                "current_at_resolution": is_current,
            }
            xyz = _finite_xyz(point_xyz.get(point_id))
            if xyz is not None:
                reference["xyz_m"] = xyz
            point_references.append(reference)
            if not is_current:
                stale_point_ids.append(point_id)

    if (
        actual_xyz is None
        and target_xyz is None
        and residual_mm is None
        and actual_delta_mm is None
    ):
        return {}

    summary: dict[str, Any] = {
        "frame": "world",
        "eef_frame": "panda_grip_site",
    }
    if actual_xyz is not None:
        summary["actual_grip_site_position_xyz_m"] = actual_xyz
    if actual_approach is not None:
        summary["actual_approach_direction_world"] = actual_approach
    if actual_jaw is not None:
        summary["actual_jaw_direction_world"] = actual_jaw
    if target_xyz is not None:
        summary["target_grip_site_position_xyz_m"] = target_xyz
        source = resolution.get("position_source")
        if isinstance(source, str) and source:
            summary["target_position_source"] = source
    if target_approach is not None:
        summary["target_approach_direction_world"] = target_approach
    if target_jaw is not None:
        summary["target_jaw_direction_world"] = target_jaw
    if residual_mm is not None:
        summary["remaining_error_to_target_mm"] = residual_mm
    if actual_delta_mm is not None:
        summary["actual_displacement_mm"] = actual_delta_mm
    if point_references:
        summary["point_references"] = point_references
    if stale_point_ids:
        summary["stale_reference_warning"] = {
            "point_ids": sorted(set(stale_point_ids)),
            "message": (
                "These marks are immutable world points authored from an older "
                "observation. They remain valid coordinates, but they are not "
                "evidence of any object's current position. Re-measure from the "
                "current observation if the referenced scene geometry may have moved."
            ),
        }
    return summary


def _motion_feedback_message(stage: str, payload: Mapping[str, Any]) -> str:
    feedback = _operator_motion_feedback(payload)
    parts: list[str] = []
    if "position_error_mm" in feedback:
        parts.append(f"position error {feedback['position_error_mm']:.1f} mm")
    if "orientation_error_deg" in feedback:
        parts.append(f"orientation error {feedback['orientation_error_deg']:.1f}°")
    if feedback.get("settled") is True:
        parts.append("settled")
    if "steps_executed" in feedback:
        parts.append(f"{feedback['steps_executed']} steps")
    return f"{stage} completed" + (f"; {', '.join(parts)}." if parts else ".")


def _payload_error(payload: Mapping[str, Any]) -> str:
    return str(payload.get("error") or payload.get("content") or "remote tool failed")


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _gateway_font(size: int) -> Any:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _finite_vector3(value: Any, *, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise GraspPoseRefinementError(f"{name} must contain exactly 3 values")
    try:
        output = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise GraspPoseRefinementError(f"{name} must be numeric") from exc
    if not all(math.isfinite(item) for item in output):
        raise GraspPoseRefinementError(f"{name} must be finite")
    return output


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in str(value)
    )


def _mask_side_engagement_diagnostics(
    insertion_depth_m: float,
    *,
    target_z_m: float | None = None,
    mask_low_quantile_z_m: float | None = None,
) -> dict[str, Any]:
    ratio = float(insertion_depth_m) / PANDA_GRIP_SITE_REFERENCE_SPAN_M
    warnings: list[dict[str, str]] = []
    if ratio < 0.5:
        warnings.append(
            {
                "code": "low_visible_surface_engagement",
                "message": (
                    "The visible top surface is near the grip-site/fingertip end of "
                    "the Panda reference span. This may close above or only skim "
                    "the intended contact region. Inspect object engagement and signed "
                    "support clearance together, then adjust shallower or deeper only "
                    "as the geometry warrants."
                ),
            }
        )
    if ratio > 1.15:
        warnings.append(
            {
                "code": "deep_visible_surface_engagement",
                "message": (
                    "The grip site is deeper than the Panda base-to-grip-site "
                    "reference span. Check table and object collision before selection."
                ),
            }
        )
    mask_low_quantile_clearance_mm: float | None = None
    if (
        isinstance(target_z_m, (int, float))
        and isinstance(mask_low_quantile_z_m, (int, float))
        and math.isfinite(float(target_z_m))
        and math.isfinite(float(mask_low_quantile_z_m))
    ):
        mask_low_quantile_clearance_mm = (
            float(target_z_m) - float(mask_low_quantile_z_m)
        ) * 1000.0
        if mask_low_quantile_clearance_mm < 0.0:
            warnings.append(
                {
                    "code": "target_below_mask_low_quantile",
                    "message": (
                        "The proposed Panda grip_site is below the selected mask's "
                        "low depth-sample quantile. This mask statistic is not table "
                        "or robot collision clearance; inspect the local geometry and "
                        "signed collision-mesh support clearance before adjusting depth."
                    ),
                }
            )
    return {
        "visible_top_to_grip_site_mm": round(float(insertion_depth_m) * 1000.0, 3),
        "panda_base_to_grip_site_reference_mm": round(
            PANDA_GRIP_SITE_REFERENCE_SPAN_M * 1000.0, 3
        ),
        "engagement_ratio": round(ratio, 4),
        **(
            {
                "target_z_m": round(float(target_z_m), 6),
                "mask_low_quantile_z_m": round(
                    float(mask_low_quantile_z_m), 6
                ),
                "grip_site_minus_mask_low_quantile_mm": round(
                    float(mask_low_quantile_clearance_mm), 3
                ),
                "mask_low_quantile_clearance_semantics": (
                    "grip_site_z minus selected-mask low quantile; not collision clearance"
                ),
            }
            if mask_low_quantile_clearance_mm is not None
            else {}
        ),
        "warnings": warnings,
    }


def _direct_move_controller_contract(
    *, tolerance_mm: float, max_steps: int
) -> dict[str, Any]:
    try:
        tolerance = float(tolerance_mm)
        steps = int(max_steps)
    except (TypeError, ValueError) as exc:
        raise GraspPoseRefinementError(
            "tolerance_mm and max_steps must be numeric"
        ) from exc
    if not math.isfinite(tolerance) or not 0.5 <= tolerance <= 20.0:
        raise GraspPoseRefinementError("tolerance_mm must be in [0.5, 20.0]")
    if not 1 <= steps <= 2000:
        raise GraspPoseRefinementError("max_steps must be in [1, 2000]")
    return {
        "tolerance_mm": tolerance,
        "tolerance_m": tolerance / 1000.0,
        "max_steps": steps,
    }


def _pose_display_method(method: str, provenance: Mapping[str, Any]) -> str:
    diagnostics = provenance.get("diagnostics")
    if method == "mask_side_top_down_v1" and isinstance(diagnostics, Mapping):
        side = diagnostics.get("side")
        insertion = diagnostics.get("insertion_depth_m")
        if isinstance(side, str) and isinstance(insertion, (int, float)):
            return f"mask-side {side} {float(insertion) * 1000.0:.0f}mm"
    return method


def _euler_xyz_matrix_degrees(value: list[float]) -> np.ndarray:
    """Return Rz(yaw) @ Ry(pitch) @ Rx(roll), matching simulator Euler input."""

    roll, pitch, yaw = (math.radians(float(item)) for item in value)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _rotation_to_euler_degrees(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, list) or len(value) != 3 or not all(isinstance(row, list) and len(row) == 3 for row in value):
        return None
    try:
        r = [[float(item) for item in row] for row in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for row in r for item in row):
        return None
    # Intrinsic XYZ (the convention used by the simulator MCP move_to API).
    pitch = math.asin(max(-1.0, min(1.0, -r[2][0])))
    if abs(math.cos(pitch)) > 1e-7:
        roll = math.atan2(r[2][1], r[2][2])
        yaw = math.atan2(r[1][0], r[0][0])
    else:
        roll = math.atan2(-r[1][2], r[1][1])
        yaw = 0.0
    return tuple(math.degrees(item) for item in (roll, pitch, yaw))


def _quat_xyzw_to_rotation_matrix(value: Any) -> np.ndarray | None:
    """Convert a finite XYZW quaternion to a normalized rotation matrix."""

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x, y, z, w = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x, y, z, w)):
        return None
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return None
    x, y, z, w = (item / norm for item in (x, y, z, w))
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def grasp_geometry_score(
    *,
    rotation: np.ndarray,
    translation: np.ndarray,
    width: float,
    target_xyz: np.ndarray | Sequence[float] | None = None,
    approach_axis_world: Sequence[float] = (0.0, 0.0, -1.0),
    expected_approach_offset_m: float = 0.025,
    expected_width_m: float = 0.07,
) -> dict[str, float]:
    """Score a world-frame Panda grip_site grasp pose against a task geometry.

    The score rewards alignment of the gripper's approach axis with the task's
    desired approach direction (defaults to world-down for a tabletop pick),
    penalises lateral/approach-plane offset of the target centroid from the jaw
    centre, and mildly prefers a gripper width near the expected object size.
    It is deterministic and unit-testable; reachability is still verified by
    the actual ``move_to`` call afterwards.

    The defaults are tuned for a top-down tabletop pick of a small rigid
    object.  Other task families (e.g. pulling a drawer handle) should pass
    their own ``approach_axis_world`` / ``expected_approach_offset_m`` /
    ``expected_width_m`` rather than reusing these defaults.
    """

    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    width = float(width)
    approach_axis = np.asarray(approach_axis_world, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(approach_axis))
    if axis_norm < 1e-9:
        raise ValueError("approach_axis_world must be non-zero")
    approach_axis = approach_axis / axis_norm
    grip_site_approach = rotation[:, 2]
    approach_z = float(grip_site_approach[2])
    alignment = float(np.dot(grip_site_approach, approach_axis))
    features: dict[str, float] = {
        "approach_z": approach_z,
        "approach_alignment": alignment,
        "closing_axis_z": float(rotation[2, 0]),
        "width_m": width,
    }
    score = 3.0 * alignment - 2.0 * abs(width - expected_width_m)
    if target_xyz is not None:
        target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        offset_local = rotation.T @ (target - translation)
        lateral = abs(float(offset_local[1]))
        closing = abs(float(offset_local[0]))
        approach_offset = float(offset_local[2])
        features.update(
            {
                "target_distance_m": float(np.linalg.norm(target - translation)),
                "lateral_offset_m": lateral,
                "closing_offset_m": closing,
                "approach_offset_m": approach_offset,
            }
        )
        score += (
            -20.0 * (lateral + 0.5 * closing)
            - 10.0 * abs(approach_offset - expected_approach_offset_m)
        )
    features["geometry_score"] = float(score)
    return features


def _anygrasp_rotation_to_panda_grip_site(value: Any) -> list[list[float]] | None:
    """Relabel AnyGrasp local axes as robosuite Panda grip-site axes.

    AnyGrasp / GraspNet uses local ``+X`` for approach and ``+Y`` for jaw
    closing. Robosuite Panda's ``grip_site`` uses local ``+Z`` for approach
    and local ``+X`` for jaw closing. For an AnyGrasp rotation ``R`` whose
    columns are its local axes in world coordinates, the Panda target is::

        R_panda = R @ [[0, 0, 1],
                       [1, 0, 0],
                       [0, 1, 0]]

    so Panda ``(X, Y, Z)`` becomes AnyGrasp ``(Y, Z, X)``. The translation
    remains the shared jaw-center / contact-plane point.
    """

    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(isinstance(row, list) and len(row) == 3 for row in value)
    ):
        return None
    try:
        rotation = [[float(item) for item in row] for row in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for row in rotation for item in row):
        return None
    axis_map = (
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    return [
        [
            sum(
                rotation[row][index] * axis_map[index][column]
                for index in range(3)
            )
            for column in range(3)
        ]
        for row in range(3)
    ]


__all__ = [
    "EmbodiedGateway",
    "GatewayResult",
    "grasp_geometry_score",
    "_anygrasp_rotation_to_panda_grip_site",
    "_operator_gripper_feedback",
    "_motion_payload_ok",
    "_task_terminal_success",
]
