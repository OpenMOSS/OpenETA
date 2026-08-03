"""Observation-bound semantic perception tools for the embodied Operator.

The existing :mod:`agent.tools.handlers` module deliberately exposes the
low-level SAM3 and AnyGrasp contracts used by host-side diagnostics.  This
module is the smaller Operator-facing layer: it resolves one retained camera
frame, invokes those backends, and carries the observation binding through the
visual result.

The class is intentionally backend-agnostic.  Real MCP callables can be
injected in production; tests can inject deterministic fakes without making a
fake result look like real model evidence.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import numpy as np

from agent.tools.handlers import (
    AnyGraspDetectCallable,
    Sam3SegmentCallable,
    build_anygrasp_handler,
    build_sam3_handler,
)
from agent.tools.registry import ToolExecutionContext, ToolResult, build_default_tool_registry


class ObservationBindingError(ValueError):
    """Raised when a perception request cannot resolve one retained frame."""


DEFAULT_GRASP_TOP_K = 1


@dataclass(frozen=True, slots=True)
class ObservationFrame:
    """A concrete RGB-D frame selected from one observation record."""

    artifact_root: Path
    observation_id: str
    frame_id: str
    camera_id: str
    rgb_path: Path
    depth_path: Path
    intrinsics: dict[str, Any]
    extrinsics: dict[str, Any]

    @property
    def binding(self) -> dict[str, str]:
        return {
            "observation_id": self.observation_id,
            "frame_id": self.frame_id,
            "camera_id": self.camera_id,
        }


@dataclass(frozen=True, slots=True)
class _SegmentationState:
    observation_id: str
    frame_id: str
    camera_id: str
    result: ToolResult


def resolve_observation_frame(
    observation_record: Mapping[str, Any],
    *,
    artifact_root: str | Path,
    camera_id: str = "agentview",
    frame_id: str | None = None,
) -> ObservationFrame:
    """Resolve one camera frame and its retained RGB-D artifacts.

    ``LiberoObservationFacade`` returns paths relative to its episode root.
    The Operator never has to supply those paths; this function is the single
    Host-side resolution point for the semantic tools.
    """

    observation_id = _required_text(observation_record.get("observation_id"), "observation_id")
    raw_frames = observation_record.get("frames")
    if not isinstance(raw_frames, list):
        raise ObservationBindingError("observation has no retained camera frames")

    selected: Mapping[str, Any] | None = None
    for candidate in raw_frames:
        if not isinstance(candidate, Mapping):
            continue
        if frame_id is not None and candidate.get("frame_id") == frame_id:
            selected = candidate
            break
        if frame_id is None and candidate.get("camera_id") == camera_id:
            selected = candidate
            break
    if selected is None:
        selector = f"frame_id={frame_id!r}" if frame_id else f"camera_id={camera_id!r}"
        raise ObservationBindingError(f"observation frame not found for {selector}")

    resolved_camera_id = _required_text(selected.get("camera_id"), "camera_id")
    resolved_frame_id = _required_text(selected.get("frame_id"), "frame_id")
    rgb_path = _resolve_artifact_path(artifact_root, selected.get("rgb_path"), "rgb_path")
    depth_path = _resolve_artifact_path(artifact_root, selected.get("depth_path"), "depth_path")
    metadata = selected.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    raw_intrinsics = metadata.get("anygrasp_intrinsics") or metadata.get("intrinsics")
    if not isinstance(raw_intrinsics, Mapping):
        raise ObservationBindingError(f"frame {resolved_frame_id} has no camera intrinsics")
    intrinsics = dict(raw_intrinsics)
    # EpisodeObservability stores metric depth as uint16 millimetres.  The
    # backend contract calls this conversion factor ``scale``.
    intrinsics.setdefault("scale", 1000.0)
    extrinsics = metadata.get("extrinsics")
    return ObservationFrame(
        artifact_root=Path(artifact_root),
        observation_id=observation_id,
        frame_id=resolved_frame_id,
        camera_id=resolved_camera_id,
        rgb_path=rgb_path,
        depth_path=depth_path,
        intrinsics=intrinsics,
        extrinsics=dict(extrinsics) if isinstance(extrinsics, Mapping) else {},
    )


class ObservationBoundPerception:
    """Expose semantic perception calls bound to the current observation.

    Public methods:

    ``segment_object(observation, target)``
        Returns mask overlays and a stable detection id.  Multiple detections
        remain an explicit selection obligation.

    ``propose_grasps(observation, detection_id=...)``
        Uses the selected SAM3 mask from the same frame and returns labeled
        candidate imagery.  It rejects a newer observation until segmentation
        is run again.
    """

    def __init__(
        self,
        *,
        artifact_root: str | Path,
        sam3: Sam3SegmentCallable,
        sam3_points: Sam3SegmentCallable | None = None,
        anygrasp: AnyGraspDetectCallable,
        output_root: str | Path | None = None,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.output_root = Path(output_root) if output_root is not None else self.artifact_root / "perception"
        self._sam3 = build_sam3_handler(
            sam3,
            segment_points=sam3_points,
            output_root=self.output_root / "sam3" / "images",
            result_output_root=self.output_root / "sam3" / "results",
        )
        self._anygrasp = build_anygrasp_handler(
            anygrasp,
            output_root=self.output_root / "anygrasp" / "results",
        )
        self._latest_segmentation: _SegmentationState | None = None

    def segment_object(
        self,
        observation_record: Mapping[str, Any],
        target: str,
        *,
        camera_id: str = "agentview",
        frame_id: str | None = None,
        image_path: str | Path | None = None,
        box_prompt_xyxy: list[float] | None = None,
        point_prompts: list[dict[str, float | int]] | None = None,
    ) -> ToolResult:
        """Segment the requested object in one retained observation frame."""

        target = target.strip()
        if not target:
            return _failure("segment_object", "missing_target", "Object target is required.")
        try:
            frame = resolve_observation_frame(
                observation_record,
                artifact_root=self.artifact_root,
                camera_id=camera_id,
                frame_id=frame_id,
            )
        except ObservationBindingError as exc:
            return _failure("segment_object", "invalid_observation", str(exc))

        source_image = Path(image_path) if image_path is not None else frame.rgb_path
        if not source_image.is_file():
            return _failure("segment_object", "invalid_source_image", f"SAM3 source image not found: {source_image}")
        result = self._sam3(
            _context(
                "sam3",
                {
                    "image": str(source_image),
                    "prompt": target,
                    **(
                        {"box_prompt_xyxy": box_prompt_xyxy}
                        if box_prompt_xyxy is not None
                        else {}
                    ),
                    **(
                        {"point_prompts": point_prompts}
                        if point_prompts is not None
                        else {}
                    ),
                },
            )
        )
        details = dict(result.details)
        details.update(
            {
                "tool": "segment_object",
                "backend_tool": "sam3",
                "target": target,
                "source_image": str(source_image),
                "observation": frame.binding,
            }
        )
        result.details = details
        if result.success:
            self._latest_segmentation = _SegmentationState(
                observation_id=frame.observation_id,
                frame_id=frame.frame_id,
                camera_id=frame.camera_id,
                result=result,
            )
        return result

    def propose_grasps(
        self,
        observation_record: Mapping[str, Any],
        *,
        detection_id: str | None = None,
        camera_id: str = "agentview",
        frame_id: str | None = None,
        collision_detection: bool = True,
        dense_grasp: bool = False,
    ) -> ToolResult:
        """Propose labeled grasp candidates for the selected SAM3 object."""

        try:
            frame = resolve_observation_frame(
                observation_record,
                artifact_root=self.artifact_root,
                camera_id=camera_id,
                frame_id=frame_id,
            )
        except ObservationBindingError as exc:
            return _failure("propose_grasps", "invalid_observation", str(exc))

        state = self._latest_segmentation
        if state is None:
            return _failure(
                "propose_grasps",
                "segmentation_required",
                "Call segment_object on the current observation first.",
                observation=frame.binding,
            )
        if state.observation_id != frame.observation_id or state.frame_id != frame.frame_id:
            return _failure(
                "propose_grasps",
                "stale_segmentation",
                "The SAM3 result belongs to an older observation frame; segment again.",
                observation=frame.binding,
                segmentation_observation={
                    "observation_id": state.observation_id,
                    "frame_id": state.frame_id,
                    "camera_id": state.camera_id,
                },
            )
        if not state.result.success:
            return _failure(
                "propose_grasps",
                "segmentation_failed",
                "The current SAM3 result is not usable.",
                observation=frame.binding,
            )

        segmentation = state.result.details
        detections = segmentation.get("detections")
        if not isinstance(detections, list):
            return _failure("propose_grasps", "invalid_segmentation", "SAM3 returned no detection list.", observation=frame.binding)
        selected = _select_detection(detections, detection_id)
        if selected is None:
            reason = "object_selection_required" if detection_id is None else "unknown_detection"
            return _failure(
                "propose_grasps",
                reason,
                "Select one visible SAM3 detection before requesting grasps."
                if reason == "object_selection_required"
                else f"SAM3 detection {detection_id!r} was not found.",
                observation=frame.binding,
                detections=detections,
                selection_bundle=segmentation.get("selection_bundle", {}),
            )
        mask_ref = selected.get("mask_ref")
        if not isinstance(mask_ref, str) or not Path(mask_ref).is_file():
            return _failure(
                "propose_grasps",
                "mask_not_found",
                "The selected SAM3 mask artifact is missing.",
                observation=frame.binding,
            )

        original_mask_ref = mask_ref
        try:
            coherent_mask_ref, depth_coherence = _write_depth_coherent_mask(
                depth_path=frame.depth_path,
                mask_path=Path(mask_ref),
                depth_scale=float(frame.intrinsics.get("scale", 1000.0)),
                output_path=(
                    self.output_root
                    / "depth_coherent_masks"
                    / frame.observation_id
                    / f"{selected.get('id') or 'selected'}.png"
                ),
            )
            mask_ref = str(coherent_mask_ref)
        except (OSError, ValueError, TypeError) as exc:
            depth_coherence = {
                "applied": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        result = self._anygrasp(
            _context(
                "anygrasp",
                {
                    "mode": "targeted",
                    "rgb": str(frame.rgb_path),
                    "depth": str(frame.depth_path),
                    "target_mask": mask_ref,
                    "intrinsics": dict(frame.intrinsics),
                    "collision_detection": collision_detection,
                    "dense_grasp": dense_grasp,
                },
            )
        )
        details = dict(result.details)
        details.update(
            {
                "tool": "propose_grasps",
                "backend_tool": "anygrasp",
                "observation": frame.binding,
                "selected_detection": {
                    key: selected.get(key)
                    for key in ("id", "label", "score", "rank")
                    if selected.get(key) is not None
                },
                "depth_coherence": depth_coherence,
            }
        )
        details["selected_detection"].update(
            {
                "mask_ref": mask_ref,
                "original_mask_ref": original_mask_ref,
            }
        )
        artifacts = details.get("artifacts")
        if not isinstance(artifacts, list):
            artifacts = []
            details["artifacts"] = artifacts
        if mask_ref != original_mask_ref:
            artifacts.append(
                {
                    "type": "depth_coherent_target_mask",
                    "path": mask_ref,
                }
            )
        if result.success:
            details["inspection_surface"] = "viser"
            details["static_pose_rendering"] = False
        result.details = details
        _write_semantic_result(details, result, self.output_root)
        return result


def render_anygrasp_candidates(
    rgb_path: str | Path,
    candidates: list[Mapping[str, Any]],
    *,
    intrinsics: Mapping[str, Any],
    output_dir: str | Path,
    camera_extrinsics: Mapping[str, Any] | None = None,
    wrist_rgb_path: str | Path | None = None,
    wrist_intrinsics: Mapping[str, Any] | None = None,
    wrist_extrinsics: Mapping[str, Any] | None = None,
    top_k: int = DEFAULT_GRASP_TOP_K,
    selected_id: str | None = None,
    focus_id: str | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Render a visual grasp decision board.

    The full candidate list remains in the structured AnyGrasp result.  The
    human/Operator-facing board intentionally shows only the highest-ranked
    ``top_k`` candidates, plus an explicitly selected candidate if it falls
    outside that set.  ``focus_id`` is a read-only inspection mode: it renders
    only that one stable candidate, which keeps the board useful when many
    candidates overlap.  The primary agentview and wrist view are rendered
    from the same camera-frame pose, with the pose transformed through the
    retained camera extrinsics before wrist projection.
    """

    from PIL import Image, ImageDraw, ImageFont, ImageOps

    image = Image.open(rgb_path).convert("RGB")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_list = [item for item in candidates if isinstance(item, Mapping)]
    ranked = sorted(enumerate(candidate_list), key=_candidate_order_key)
    visible: list[dict[str, Any]] = []
    if focus_id is not None:
        focused = next((candidate for candidate in candidate_list if candidate.get("id") == focus_id), None)
        if focused is not None:
            item = dict(focused)
            item["_display_index"] = int(_number(focused.get("rank"), default=0))
            visible.append(item)
    else:
        for display_index, (_index, candidate) in enumerate(ranked[: max(1, int(top_k))]):
            item = dict(candidate)
            item["_display_index"] = display_index
            visible.append(item)
    if selected_id is not None and not any(item.get("id") == selected_id for item in visible):
        selected = next((item for item in candidate_list if item.get("id") == selected_id), None)
        if selected is not None:
            item = dict(selected)
            item["_display_index"] = int(_number(selected.get("rank"), default=len(visible)))
            item["_selected_outside_top_k"] = True
            visible.append(item)

    colors = (
        (0, 220, 255),
        (255, 210, 0),
        (255, 125, 45),
        (180, 110, 255),
        (70, 220, 140),
        (255, 90, 150),
        (90, 160, 255),
    )
    primary_overlay, primary_visuals = _render_pose_view(
        image,
        visible,
        intrinsics=intrinsics,
        source_extrinsics=camera_extrinsics,
        target_extrinsics=None,
        highlight_id=selected_id or focus_id,
        colors=colors,
        label_prefix="G",
    )
    primary_ref = output_root / "agentview_topk.png"
    primary_overlay.save(primary_ref, format="PNG")

    wrist_overlay: Image.Image | None = None
    wrist_visuals: list[dict[str, Any]] = []
    wrist_ref: Path | None = None
    if wrist_rgb_path is not None and wrist_intrinsics is not None:
        wrist_image = Image.open(wrist_rgb_path).convert("RGB")
        wrist_overlay, wrist_visuals = _render_pose_view(
            wrist_image,
            visible,
            intrinsics=wrist_intrinsics,
            source_extrinsics=camera_extrinsics,
            target_extrinsics=wrist_extrinsics,
            highlight_id=selected_id or focus_id,
            colors=colors,
            label_prefix="G",
        )
        wrist_ref = output_root / "wrist_topk.png"

    font = _load_font(16)
    small_font = _load_font(13)
    wrist_by_id = {item.get("id"): item for item in wrist_visuals}
    if wrist_overlay is not None and visible and not any(item.get("in_frame") for item in wrist_visuals):
        wrist_draw = ImageDraw.Draw(wrist_overlay)
        warning = "top-k poses outside wrist FOV"
        warning_box = wrist_draw.textbbox((0, 0), warning, font=small_font)
        warning_width = warning_box[2] - warning_box[0] + 12
        warning_height = warning_box[3] - warning_box[1] + 8
        wrist_draw.rounded_rectangle(
            (4, 4, min(wrist_overlay.width - 4, 4 + warning_width), 4 + warning_height),
            radius=3,
            fill=(25, 28, 34),
            outline=(255, 190, 60),
            width=1,
        )
        wrist_draw.text((10, 8), warning, fill=(255, 210, 100), font=small_font)
    if wrist_overlay is not None and wrist_ref is not None:
        wrist_overlay.save(wrist_ref, format="PNG")
    primary_display = ImageOps.contain(primary_overlay, (512, 512))
    if wrist_overlay is not None:
        wrist_display = ImageOps.contain(wrist_overlay, (320, 320))
    else:
        wrist_display = None

    panel_width = 360
    table_floor = 365 if wrist_display is not None else 72
    canvas_height = max(
        552,
        primary_display.height + 40,
        table_floor + 28 + len(visible) * 52 + 44,
    )
    canvas = Image.new("RGB", (512 + panel_width + 16, canvas_height), (24, 27, 32))
    canvas_draw = ImageDraw.Draw(canvas)
    canvas.paste(primary_display, (0, 32))
    if focus_id is not None and visible:
        focus_label = f"G{int(_number(visible[0].get('_display_index'), default=0))}"
        primary_title = f"AGENTVIEW  INSPECT {focus_label} / {len(candidate_list)}"
    else:
        primary_title = f"AGENTVIEW  top {min(len(visible), max(1, int(top_k)))} / {len(candidate_list)}"
    canvas_draw.text((8, 8), primary_title, fill=(245, 245, 245), font=font)

    panel_x = 528
    if wrist_display is not None:
        wrist_in_frame_count = sum(1 for item in wrist_visuals if item.get("in_frame"))
        wrist_title = "WRIST VIEW"
        if focus_id is not None and visible:
            wrist_title += f"  INSPECT G{int(_number(visible[0].get('_display_index'), default=0))}"
        wrist_title += f"  {wrist_in_frame_count}/{len(visible)} in frame"
        canvas_draw.text(
            (panel_x, 8),
            wrist_title,
            fill=(245, 245, 245),
            font=font,
        )
        canvas.paste(wrist_display, (panel_x, 32))
        table_y = 365
    else:
        canvas_draw.text((panel_x, 8), "WRIST VIEW  unavailable", fill=(170, 175, 185), font=font)
        canvas_draw.text((panel_x, 34), "No wrist RGB frame in this observation.", fill=(170, 175, 185), font=small_font)
        table_y = 72

    canvas_draw.text((panel_x, table_y), "VISIBLE CANDIDATES", fill=(245, 245, 245), font=font)
    row_y = table_y + 28
    for index, candidate in enumerate(visible):
        rank = int(_number(candidate.get("rank"), default=index))
        candidate_id = str(candidate.get("id") or f"grasp_{rank:03d}")
        display_index = int(_number(candidate.get("_display_index"), default=index))
        selected = candidate_id == selected_id
        color = (70, 245, 120) if selected else colors[rank % len(colors)]
        score = candidate.get("score")
        score_text = f"{float(score):.4f}" if _is_number(score) else "n/a"
        width = _number(candidate.get("width"), default=0.0) * 1000.0
        focused = candidate_id == focus_id
        marker = "SELECTED " if selected else ("INSPECTING " if focused else "")
        wrist_visual = wrist_by_id.get(candidate_id, {})
        if wrist_display is None:
            wrist_status = "wrist unavailable"
        elif wrist_visual.get("in_frame"):
            wrist_status = "wrist in view"
        elif wrist_visual.get("projected"):
            wrist_status = "wrist off-frame"
        else:
            wrist_status = "wrist not projectable"
        display_label = str(candidate.get("display_label") or f"G{display_index}")
        label = f"{marker}{display_label}  {candidate_id}  score {score_text}"
        canvas_draw.rounded_rectangle(
            (panel_x - 4, row_y - 3, panel_x + panel_width - 8, row_y + 22),
            radius=4,
            fill=(42, 54, 48) if selected else (34, 38, 46),
            outline=color,
            width=2 if selected else 1,
        )
        canvas_draw.text((panel_x + 4, row_y + 2), label, fill=color, font=small_font)
        canvas_draw.text(
            (panel_x + 4, row_y + 25),
            f"jaw width {width:.1f} mm  ·  {wrist_status}",
            fill=(165, 170, 180),
            font=small_font,
        )
        row_y += 52
    if len(candidate_list) > len(visible):
        canvas_draw.text(
            (panel_x, min(canvas_height - 24, row_y + 4)),
            f"+ {len(candidate_list) - len(visible)} more retained in artifact",
            fill=(165, 170, 180),
            font=small_font,
        )
    canvas_draw.text(
        (8, canvas_height - 22),
        "jaw line = gripper opening   arrow = approach   RGB axes = orientation",
        fill=(185, 190, 200),
        font=small_font,
    )

    overlay_ref = output_root / "candidate_overlay.png"
    canvas.save(overlay_ref, format="PNG")
    visuals = []
    primary_by_id = {item.get("id"): item for item in primary_visuals}
    visible_ids = {item.get("id") for item in visible}
    for candidate in candidate_list:
        candidate_id = candidate.get("id")
        rank = int(_number(candidate.get("rank"), default=candidate_list.index(candidate)))
        primary = primary_by_id.get(candidate_id, {})
        wrist = wrist_by_id.get(candidate_id, {})
        visuals.append(
            {
                "id": candidate_id,
                "rank": rank,
                "display_label": primary.get("label"),
                "score": candidate.get("score"),
                "visible": candidate_id in visible_ids,
                "selected": candidate_id == selected_id,
                "focused": candidate_id == focus_id,
                "projected": bool(primary.get("projected")),
                "in_frame": bool(primary.get("in_frame")),
                "wrist_projected": bool(wrist.get("projected")),
                "wrist_in_frame": bool(wrist.get("in_frame")),
                "center_px": primary.get("center_px"),
                "wrist_center_px": wrist.get("center_px"),
            }
        )
    return overlay_ref, visuals


def _render_pose_view(
    image: Any,
    candidates: list[Mapping[str, Any]],
    *,
    intrinsics: Mapping[str, Any],
    source_extrinsics: Mapping[str, Any] | None,
    target_extrinsics: Mapping[str, Any] | None,
    highlight_id: str | None,
    colors: tuple[tuple[int, int, int], ...],
    label_prefix: str,
) -> tuple[Any, list[dict[str, Any]]]:
    from PIL import ImageDraw

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    visuals: list[dict[str, Any]] = []
    occupied: list[tuple[float, float, float, float]] = []
    for index, candidate in enumerate(candidates):
        rank = int(_number(candidate.get("rank"), default=index))
        candidate_id = str(candidate.get("id") or f"grasp_{rank:03d}")
        display_index = int(_number(candidate.get("_display_index"), default=index))
        label = str(candidate.get("display_label") or f"{label_prefix}{display_index}")
        color = (70, 245, 120) if candidate_id == highlight_id else colors[rank % len(colors)]
        geometry = _candidate_geometry(
            candidate,
            source_extrinsics=source_extrinsics,
            target_extrinsics=target_extrinsics,
        )
        center = geometry.get("center")
        if center is None:
            visuals.append({"id": candidate_id, "rank": rank, "projected": False})
            continue
        center_px = _project(center, intrinsics)
        if center_px is None:
            visuals.append(
                {
                    "id": candidate_id,
                    "rank": rank,
                    "label": label,
                    "projected": False,
                    "in_frame": False,
                }
            )
            continue
        in_frame = _point_in_frame(center_px, overlay.size)
        if not in_frame:
            # Keep the numerical projection for diagnostics, but do not clamp
            # the pose label into the image: that made several off-screen
            # wrist candidates look as if they were piled on the border.
            edge_x = min(max(center_px[0], 4.0), float(overlay.width - 5))
            edge_y = min(max(center_px[1], 4.0), float(overlay.height - 5))
            draw.ellipse(
                (edge_x - 3, edge_y - 3, edge_x + 3, edge_y + 3),
                outline=color,
                width=2,
            )
            visuals.append(
                {
                    "id": candidate_id,
                    "rank": rank,
                    "label": label,
                    "projected": True,
                    "in_frame": False,
                    "center_px": [int(center_px[0]), int(center_px[1])],
                }
            )
            continue
        highlighted = candidate_id == highlight_id
        line_width = 5 if highlighted else (4 if rank == 0 else 3)
        tip_px = _project(geometry["tip"], intrinsics) if geometry.get("tip") is not None else None
        if tip_px is not None:
            _draw_arrow(draw, center_px, tip_px, fill=color, width=line_width)
        jaw_left = _project(geometry["jaw_left"], intrinsics) if geometry.get("jaw_left") is not None else None
        jaw_right = _project(geometry["jaw_right"], intrinsics) if geometry.get("jaw_right") is not None else None
        if jaw_left is not None and jaw_right is not None:
            draw.line((jaw_left, jaw_right), fill=color, width=line_width + 1)
        for axis_index, axis_color in enumerate(((235, 85, 85), (80, 220, 110), (80, 145, 245))):
            axis_px = _project(geometry["axes"][axis_index], intrinsics) if geometry.get("axes") else None
            if axis_px is not None:
                draw.line((center_px, axis_px), fill=axis_color, width=2 if highlighted else 1)
        radius = 8 if highlighted else 5
        draw.ellipse(
            (center_px[0] - radius, center_px[1] - radius, center_px[0] + radius, center_px[1] + radius),
            outline=color,
            width=3 if highlighted else 2,
        )
        text_box = draw.textbbox((0, 0), label)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_x, text_y = _label_position(
            center_px,
            width=text_width + 10,
            height=text_height + 6,
            image_size=overlay.size,
            occupied=occupied,
        )
        occupied.append((text_x, text_y, text_x + text_width + 10, text_y + text_height + 6))
        draw.rounded_rectangle(
            (text_x, text_y, text_x + text_width + 10, text_y + text_height + 6),
            radius=3,
            fill=(10, 12, 15),
            outline=color,
            width=2 if highlighted else 1,
        )
        draw.text((text_x + 5, text_y + 3), label, fill=color, stroke_width=0)
        visuals.append(
            {
                "id": candidate_id,
                "rank": rank,
                "label": label,
                "score": candidate.get("score"),
                "projected": True,
                "in_frame": True,
                "center_px": [int(center_px[0]), int(center_px[1])],
            }
        )
    return overlay, visuals


def _candidate_geometry(
    candidate: Mapping[str, Any],
    *,
    source_extrinsics: Mapping[str, Any] | None,
    target_extrinsics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if (
        candidate.get("pose_frame") == "world"
        and candidate.get("eef_frame") == "panda_grip_site"
    ):
        raw_transform = candidate.get("transform_world_from_grip_site")
        try:
            transform = [[float(item) for item in row] for row in raw_transform]
        except (TypeError, ValueError):
            return {}
        if len(transform) != 4 or any(len(row) != 4 for row in transform):
            return {}
        camera_extrinsics = target_extrinsics or source_extrinsics
        if camera_extrinsics is None:
            return {}
        center_world = [transform[index][3] for index in range(3)]
        closing_world = [transform[index][0] for index in range(3)]
        approach_world = [transform[index][2] for index in range(3)]
        axes_world = [
            [transform[row][column] for row in range(3)]
            for column in range(3)
        ]
        center = _world_point_to_camera(center_world, camera_extrinsics)
        side = _world_vector_to_camera(closing_world, camera_extrinsics)
        approach = _world_vector_to_camera(approach_world, camera_extrinsics)
        axes = [
            _world_vector_to_camera(axis, camera_extrinsics)
            for axis in axes_world
        ]
        width = _number(candidate.get("width"), default=0.08)
        axis_length = max(width, 0.04) * 0.8
        tip = _add(center, _scale(approach, 0.06))
        return {
            "center": center,
            "tip": tip,
            "jaw_left": _add(center, _scale(side, width / 2.0)),
            "jaw_right": _add(center, _scale(side, -width / 2.0)),
            "axes": [_add(center, _scale(axis, axis_length)) for axis in axes],
        }

    center = _vector(candidate.get("translation_xyz"))
    tip = _vector(candidate.get("gripper_tip_position_xyz")) or center
    rotation = candidate.get("rotation_matrix")
    side = _matrix_column(rotation, 1)
    axes = [_matrix_column(rotation, index) for index in range(3)] if rotation is not None else []
    width = _number(candidate.get("width"), default=0.04)
    if center is None:
        return {}
    axis_length = max(width, 0.04) * 0.8
    if target_extrinsics is not None:
        if source_extrinsics is None:
            return {}
        transform_point = lambda value: _camera_point_between_frames(value, source_extrinsics, target_extrinsics)
        transform_vector = lambda value: _camera_vector_between_frames(value, source_extrinsics, target_extrinsics)
        center = transform_point(center)
        tip = transform_point(tip) if tip is not None else None
        side = transform_vector(side) if side is not None else None
        axes = [transform_vector(axis) if axis is not None else None for axis in axes]
    return {
        "center": center,
        "tip": tip,
        "jaw_left": _add(center, _scale(side, width / 2.0)) if side is not None else None,
        "jaw_right": _add(center, _scale(side, -width / 2.0)) if side is not None else None,
        "axes": [_add(center, _scale(axis, axis_length)) if axis is not None else None for axis in axes],
    }


def _world_point_to_camera(
    point: list[float], camera_extrinsics: Mapping[str, Any]
) -> list[float]:
    rotation, position, camera_frame = _parse_render_extrinsics(camera_extrinsics)
    native = _mat3_vec(_transpose3(rotation), _subtract(point, position))
    return _convert_render_frame(native, camera_frame, "opencv")


def _world_vector_to_camera(
    vector: list[float], camera_extrinsics: Mapping[str, Any]
) -> list[float]:
    rotation, _position, camera_frame = _parse_render_extrinsics(camera_extrinsics)
    native = _mat3_vec(_transpose3(rotation), vector)
    return _convert_render_frame(native, camera_frame, "opencv")


def _camera_point_between_frames(
    point: list[float],
    source_extrinsics: Mapping[str, Any],
    target_extrinsics: Mapping[str, Any],
) -> list[float]:
    source_rotation, source_position, source_frame = _parse_render_extrinsics(source_extrinsics)
    target_rotation, target_position, target_frame = _parse_render_extrinsics(target_extrinsics)
    source_point = _convert_render_frame(point, "opencv", source_frame)
    world = _add(_mat3_vec(source_rotation, source_point), source_position)
    target_point = _mat3_vec(_transpose3(target_rotation), _subtract(world, target_position))
    return _convert_render_frame(target_point, target_frame, "opencv")


def _camera_vector_between_frames(
    vector: list[float],
    source_extrinsics: Mapping[str, Any],
    target_extrinsics: Mapping[str, Any],
) -> list[float]:
    source_rotation, _source_position, source_frame = _parse_render_extrinsics(source_extrinsics)
    target_rotation, _target_position, target_frame = _parse_render_extrinsics(target_extrinsics)
    source_vector = _convert_render_frame(vector, "opencv", source_frame)
    world = _mat3_vec(source_rotation, source_vector)
    target_vector = _mat3_vec(_transpose3(target_rotation), world)
    return _convert_render_frame(target_vector, target_frame, "opencv")


def _parse_render_extrinsics(value: Mapping[str, Any]) -> tuple[list[list[float]], list[float], str]:
    position = _vector(value.get("pos")) or [0.0, 0.0, 0.0]
    raw_matrix = value.get("mat")
    if isinstance(raw_matrix, list) and len(raw_matrix) == 9:
        flat = [float(item) for item in raw_matrix]
        if str(value.get("matrix_layout", "row_major")) == "column_major":
            rotation = [[flat[0], flat[3], flat[6]], [flat[1], flat[4], flat[7]], [flat[2], flat[5], flat[8]]]
        else:
            rotation = [flat[0:3], flat[3:6], flat[6:9]]
    elif isinstance(raw_matrix, list) and len(raw_matrix) == 3:
        rotation = [[float(item) for item in row] for row in raw_matrix]
    else:
        rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    raw_frame = str(value.get("camera_frame") or value.get("frame_convention") or "opengl").lower()
    frame = "opencv" if raw_frame in {"opencv", "cv", "pinhole"} else "opengl"
    return rotation, position, frame


def _convert_render_frame(value: list[float], source: str, target: str) -> list[float]:
    if source == target:
        return list(value)
    if {source, target} == {"opencv", "opengl"}:
        return [value[0], -value[1], -value[2]]
    return list(value)


def _mat3_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3)]


def _transpose3(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[row][col] for row in range(3)] for col in range(3)]


def _subtract(left: list[float], right: list[float]) -> list[float]:
    return [left[index] - right[index] for index in range(3)]


def _draw_arrow(draw: Any, start: tuple[float, float], end: tuple[float, float], *, fill: tuple[int, int, int], width: int) -> None:
    draw.line((start, end), fill=fill, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return
    ux, uy = dx / length, dy / length
    size = 7 + width
    left = (end[0] - ux * size - uy * size * 0.55, end[1] - uy * size + ux * size * 0.55)
    right = (end[0] - ux * size + uy * size * 0.55, end[1] - uy * size - ux * size * 0.55)
    draw.polygon((end, left, right), fill=fill)


def _label_position(
    center: tuple[float, float],
    *,
    width: float,
    height: float,
    image_size: tuple[int, int],
    occupied: list[tuple[float, float, float, float]],
) -> tuple[float, float]:
    image_width, image_height = image_size
    offsets = ((8, -height - 4), (8, 8), (-width - 8, -height - 4), (-width - 8, 8), (0, 0))
    for dx, dy in offsets:
        x = max(0.0, min(float(image_width) - width, center[0] + dx))
        y = max(0.0, min(float(image_height) - height, center[1] + dy))
        box = (x, y, x + width, y + height)
        if not any(_boxes_overlap(box, other) for other in occupied):
            return x, y
    return max(0.0, min(float(image_width) - width, center[0] + 8)), max(0.0, min(float(image_height) - height, center[1] + 8))


def _boxes_overlap(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> bool:
    return not (left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1])


def _load_font(size: int) -> Any:
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _context(name: str, parameters: dict[str, Any]) -> ToolExecutionContext:
    registry = build_default_tool_registry()
    return ToolExecutionContext(
        name=name,
        spec=registry.get(name),
        parameters=parameters,
    )


def _select_detection(detections: list[Any], detection_id: str | None) -> Mapping[str, Any] | None:
    valid = [item for item in detections if isinstance(item, Mapping)]
    if detection_id is not None:
        return next((item for item in valid if item.get("id") == detection_id), None)
    return valid[0] if len(valid) == 1 else None


def _write_depth_coherent_mask(
    *,
    depth_path: Path,
    mask_path: Path,
    depth_scale: float,
    output_path: Path,
    split_gap_m: float = 0.04,
) -> tuple[Path, dict[str, Any]]:
    """Keep the dominant contiguous depth cluster inside a semantic mask.

    SAM-style masks can include a few pixels from an occluding robot or the
    background. Those pixels are harmless in 2D but can be tens of centimetres
    away in 3D and attract a targeted grasp proposal. This filter is purely
    geometric: it splits sorted target depths at large metric gaps and retains
    the largest cluster without using object-category knowledge.
    """

    from PIL import Image

    if not math.isfinite(depth_scale) or depth_scale <= 0.0:
        raise ValueError("depth_scale must be positive")
    depth = np.asarray(Image.open(depth_path), dtype=np.float64) / depth_scale
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("depth and mask dimensions differ")
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    values = np.sort(depth[valid])
    if values.size == 0:
        raise ValueError("selected mask has no valid depth samples")

    split_indices = np.flatnonzero(np.diff(values) > float(split_gap_m)) + 1
    clusters = np.split(values, split_indices)
    dominant = max(clusters, key=lambda cluster: int(cluster.size))
    lower = float(dominant[0])
    upper = float(dominant[-1])
    coherent = valid & (depth >= lower) & (depth <= upper)
    retained = int(np.count_nonzero(coherent))
    original = int(np.count_nonzero(valid))
    if retained == 0:
        raise ValueError("depth coherence filter removed every target pixel")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((coherent.astype(np.uint8) * 255), mode="L").save(output_path)
    return output_path, {
        "applied": True,
        "method": "dominant_contiguous_depth_cluster_v1",
        "split_gap_m": float(split_gap_m),
        "original_pixel_count": original,
        "retained_pixel_count": retained,
        "removed_pixel_count": original - retained,
        "retained_fraction": retained / original,
        "cluster_count": len(clusters),
        "retained_depth_min_m": lower,
        "retained_depth_max_m": upper,
    }


def _resolve_artifact_path(root: str | Path, value: Any, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ObservationBindingError(f"frame has no {name}")
    path = Path(value)
    if not path.is_absolute():
        path = Path(root) / path
    if not path.is_file():
        raise ObservationBindingError(f"{name} artifact does not exist: {path}")
    return path


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ObservationBindingError(f"observation has no {name}")
    return value


def _failure(tool: str, reason: str, content: str, **extra: Any) -> ToolResult:
    details = {"tool": tool, "success": False, "reason": reason}
    details.update(extra)
    return ToolResult(False, content=content, details=details)


def _result_output_dir(details: Mapping[str, Any], fallback: Path) -> Path:
    raw = details.get("raw_output_ref")
    if isinstance(raw, str) and raw:
        return Path(raw).parent
    return fallback


def _write_semantic_result(details: Mapping[str, Any], result: ToolResult, output_root: Path) -> None:
    raw = details.get("raw_output_ref")
    if isinstance(raw, str) and raw:
        target = Path(raw).parent / "semantic_tool_result.json"
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        target = output_root / "semantic_tool_result.json"
    target.write_text(
        json.dumps({"success": result.success, "content": result.content, "details": details}, ensure_ascii=False, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )


def _number(value: Any, *, default: float) -> float:
    return float(value) if _is_number(value) else default


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _vector(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    if not all(_is_number(item) for item in value):
        return None
    return [float(item) for item in value]


def _matrix_column(value: Any, index: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        column = [float(value[row][index]) for row in range(3)]
    except (IndexError, TypeError, ValueError):
        return None
    return column if all(math.isfinite(item) for item in column) else None


def _add(left: list[float], right: list[float]) -> list[float]:
    return [left[index] + right[index] for index in range(3)]


def _scale(value: list[float], amount: float) -> list[float]:
    return [item * amount for item in value]


def _project(point: list[float], intrinsics: Mapping[str, Any]) -> tuple[float, float] | None:
    x, y, z = point
    if z <= 1e-6:
        return None
    try:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(item) and item > 0 for item in (fx, fy)):
        return None
    return fx * x / z + cx, fy * y / z + cy


def _point_in_frame(point: tuple[float, float], image_size: tuple[int, int]) -> bool:
    x, y = point
    width, height = image_size
    return 0.0 <= x < float(width) and 0.0 <= y < float(height)


def _candidate_order_key(pair: tuple[int, Mapping[str, Any]]) -> tuple[int, float, float, int]:
    index, candidate = pair
    rank = candidate.get("rank")
    if _is_number(rank):
        return (0, float(rank), 0.0, index)
    score = _number(candidate.get("score"), default=float("-inf"))
    return (1, 0.0, -score, index)


__all__ = [
    "ObservationBindingError",
    "ObservationBoundPerception",
    "ObservationFrame",
    "render_anygrasp_candidates",
    "resolve_observation_frame",
]
