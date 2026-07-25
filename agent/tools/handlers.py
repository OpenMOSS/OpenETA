"""Reusable tool handlers for OpenETA runtime tests, CLI, and MCP-backed tools."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import time
from io import BytesIO
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from adapter.protocol import JsonDict
from agent.runtime.artifact_paths import artifact_session_id, artifact_session_root
from agent.tools.attachment_probe import build_prepare_attachment_probe_handler
from agent.tools.grasp_geometry import (
    build_compile_grasp_seed_handler,
    build_wrist_alignment_handler,
)
from agent.tools.registry import (
    ToolExecutionContext,
    ToolHandler,
    ToolRegistry,
    ToolResult,
    make_tool_result,
)
from agent.tools.sim_mcp import SseSimulatorMcpTransport


ApprovalCallback = Callable[[ToolExecutionContext], bool]
Sam3SegmentCallable = Callable[[JsonDict], JsonDict]
Sam3PointSegmentCallable = Callable[[JsonDict], JsonDict]
AnyGraspDetectCallable = Callable[[JsonDict], JsonDict]
ContactGraspNetPredictCallable = Callable[[JsonDict], JsonDict]
AnyPlacePredictCallable = Callable[[JsonDict], JsonDict]
MolmoPointCallable = Callable[[JsonDict], JsonDict]
GraspGenXPredictCallable = Callable[[JsonDict], JsonDict]
GraspGenXListCallable = Callable[[], JsonDict]
DepthPriorCallable = Callable[[JsonDict], JsonDict]
DepthPriorPrefetchCallable = Callable[[ToolExecutionContext, str], JsonDict]
DEFAULT_SAM3_IMAGE_OUTPUT_ROOT = Path("tmp") / "image" / "sam3"
DEFAULT_SAM3_RESULT_OUTPUT_ROOT = Path("tmp") / "tool_result" / "sam3"
DEFAULT_ANYGRASP_OUTPUT_ROOT = Path("tmp") / "tool_result" / "anygrasp"
DEFAULT_CONTACT_GRASPNET_OUTPUT_ROOT = Path("tmp") / "tool_result" / "contact_graspnet"
DEFAULT_SAM3_SELECTION_VISUAL_LIMIT = 8
DEFAULT_SAM3_ROI_PADDING_RATIO = 0.12
DEFAULT_SAM3_ROI_FALLBACK_PROMPT = "foreground object"
SAM3_MAX_POINT_COUNT = 64
DEFAULT_ANYPLACE_OUTPUT_ROOT = Path("tmp") / "tool_result" / "anyplace"
DEFAULT_MOLMOPOINT_OUTPUT_ROOT = Path("tmp") / "tool_result" / "molmopoint"
DEFAULT_GRASPGENX_OUTPUT_ROOT = Path("tmp") / "tool_result" / "graspgenx"
DEFAULT_DEPTH_PRIOR_OUTPUT_ROOT = Path("tmp") / "tool_result" / "depth_prior"
GRASP_POSE_ESTIMATE_SCHEMA = "openeta.grasp_pose_estimate.v1"
DEFAULT_GRASP_POSE_BACKEND_ORDER = (
    "anygrasp",
    "contact_graspnet",
    "graspgenx",
)
GRASP_POSE_FALLBACK_REASONS = {
    "backend_unavailable",
    "inconsistent_grasp_outputs",
    "mcp_call_failed",
    "model_inference_failed",
    "model_load_failed",
    "no_grasp_candidates",
    "unknown_error",
}

CONTACT_GRASPNET_MODEL = "contact_graspnet_pytorch_unofficial"
CONTACT_GRASPNET_GRIPPER_DEPTH = 0.1034
CONTACT_GRASPNET_MAX_CANDIDATES = 20
MOLMOPOINT_MAX_IMAGE_COUNT = 4
MOLMOPOINT_MAX_IMAGE_SIDE = 8192
MOLMOPOINT_MAX_IMAGE_PIXELS = 32_000_000
MOLMOPOINT_MAX_TOTAL_IMAGE_PIXELS = 64_000_000
MOLMOPOINT_MAX_PROMPT_CHARS = 1024
MOLMOPOINT_COORDINATE_CONVENTION = {
    "origin": "top_left",
    "x_direction": "right",
    "y_direction": "down",
    "units": "pixels",
}
GRASPGENX_MODEL = "graspgenx"
GRASPGENX_BACKEND = "graspgenx_mcp"
GRASPGENX_PLANNER = "graspmoe"
GRASPGENX_MAX_CANDIDATES = 20
GRASPGENX_POSE_CONVENTION = "p_camera = R @ p_grasp + t"
GRASPGENX_VISUAL_LIMIT = 10
GRASPGENX_BOX_EDGES = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)
GRASPGENX_PALETTE = (
    (0, 255, 160, 255),
    (255, 210, 0, 255),
    (40, 180, 255, 255),
    (255, 70, 180, 255),
    (180, 255, 60, 255),
    (255, 130, 40, 255),
    (170, 100, 255, 255),
    (0, 235, 235, 255),
    (255, 80, 80, 255),
    (210, 210, 210, 255),
)


def bind_dummy_tool_handlers(
    tools: ToolRegistry,
    *,
    replace: bool = False,
    approve_world_mutating: ApprovalCallback | None = None,
    include_dummy_safety: bool = True,
) -> ToolRegistry:
    """Bind deterministic dummy handlers for common perception/control tools."""

    handlers = {
        "observe": _observe_handler,
        "scene_detector": _scene_detector_handler,
        "sam3": _sam3_handler,
        "anygrasp": _anygrasp_handler,
        "camera_pose_to_world": _camera_pose_to_world_handler,
        "compile_grasp_seed": build_compile_grasp_seed_handler(),
        "prepare_attachment_probe": build_prepare_attachment_probe_handler(),
        "compute_wrist_alignment": build_wrist_alignment_handler(),
        "hand_pose_database": _hand_pose_handler,
        "move_to": _approval_control_handler(approve_world_mutating),
        "follow_eef_trajectory": _approval_control_handler(approve_world_mutating),
        "gripper_control": _approval_control_handler(approve_world_mutating),
        "lower_body_control_policy": _approval_control_handler(approve_world_mutating),
    }
    if include_dummy_safety:
        handlers.update(
            {
                "ik_preview_check": _ik_preview_handler,
                "obstacle_avoidance": _obstacle_avoidance_handler,
            }
        )
    for name, handler in handlers.items():
        if tools.can_execute(name) and not replace:
            continue
        tools.bind_handler(name, handler, replace=replace)
    return tools


def build_sam3_handler(
    segment: Sam3SegmentCallable,
    *,
    segment_points: Sam3PointSegmentCallable | None = None,
    depth_prior_prefetch: DepthPriorPrefetchCallable | None = None,
    output_root: str | Path | None = None,
    result_output_root: str | Path | None = None,
) -> ToolHandler:
    """Build a SAM3 handler backed by text and optional point MCP callables."""

    image_output_root = (
        Path(output_root) if output_root is not None else DEFAULT_SAM3_IMAGE_OUTPUT_ROOT
    )
    json_output_root = (
        Path(result_output_root)
        if result_output_root is not None
        else DEFAULT_SAM3_RESULT_OUTPUT_ROOT
        if output_root is None
        else image_output_root
    )

    def handler(context: ToolExecutionContext) -> ToolResult:
        session_id = artifact_session_id(context.metadata)
        run_dir = _create_run_dir(artifact_session_root(json_output_root, session_id))
        artifacts_dir = artifact_session_root(image_output_root, session_id) / run_dir.name
        request_ref = run_dir / "request.json"
        raw_output_ref = run_dir / "response.raw.json"
        tool_result_ref = run_dir / "tool_result.json"
        explicit_mode = _string_param(context.parameters.get("mode")).lower()
        positive_points_value = context.parameters.get("positive_points")
        legacy_point_prompt = positive_points_value is not None
        raw_points = context.parameters.get("points")
        if legacy_point_prompt:
            mode = "points"
            raw_points = positive_points_value
        else:
            mode = explicit_mode or "text"
        requested_image = _string_param(context.parameters.get("image"))
        image = _resolve_current_observation_rgb_path(requested_image, context.observation)
        prompt = _string_param(context.parameters.get("prompt"))
        roi_bbox_value = context.parameters.get("roi_bbox_xyxy")
        points: list[JsonDict] = []
        request: JsonDict = {"mode": mode, "image": image}
        if mode == "text":
            request["prompt"] = prompt
        elif mode == "points":
            request["points"] = raw_points if isinstance(raw_points, list) else []
        else:
            if prompt:
                request["prompt"] = prompt
            if raw_points is not None:
                request["points"] = raw_points
        if roi_bbox_value is not None:
            request["roi_bbox_xyxy"] = roi_bbox_value
        if context.parameters.get("positive_points") is not None:
            request["positive_points"] = raw_points
        context.parameters = dict(request)

        def finish(
            result: ToolResult,
            *,
            mcp_called: bool,
            response: Any = None,
            reason: str = "",
        ) -> ToolResult:
            _write_json(request_ref, dict(context.parameters))
            details = dict(result.details)
            details["raw_output_ref"] = str(raw_output_ref)
            result.details = details
            raw_record: JsonDict = {"mcp_called": mcp_called}
            if isinstance(response, Mapping):
                raw_record["response"] = _scrub_sam3_response(
                    dict(response),
                    detections=[
                        item
                        for item in _list_or_empty(details.get("detections"))
                        if isinstance(item, dict)
                    ],
                    artifacts=[
                        item
                        for item in _list_or_empty(details.get("artifacts"))
                        if isinstance(item, dict)
                    ],
                )
            if reason:
                raw_record["reason"] = reason
            _write_json(raw_output_ref, raw_record)
            _write_json(
                tool_result_ref,
                {
                    "success": result.success,
                    "content": result.content,
                    "details": result.details,
                },
            )
            return result

        if mode not in {"text", "points"}:
            return finish(
                _sam3_failure(
                    mode=mode,
                    prompt=prompt,
                    points=[],
                    source_image=image,
                    reason="invalid_mode",
                    content="SAM3 segmentation failed: invalid mode.",
                ),
                mcp_called=False,
                reason="invalid_mode",
            )
        if not image:
            return finish(
                _sam3_failure(
                    mode=mode,
                    prompt=prompt,
                    points=[],
                    source_image=image,
                    reason="missing_image",
                    content="SAM3 segmentation failed: missing image.",
                ),
                mcp_called=False,
                reason="missing_image",
            )
        if (mode == "text" and raw_points) or (mode == "points" and prompt):
            return finish(
                _sam3_failure(
                    mode=mode,
                    prompt=prompt,
                    points=[],
                    source_image=image,
                    reason="conflicting_prompt_inputs",
                    content="SAM3 segmentation failed: prompt inputs conflict with mode.",
                ),
                mcp_called=False,
                reason="conflicting_prompt_inputs",
            )
        if mode == "text" and not prompt:
            return finish(
                _sam3_failure(
                    mode=mode,
                    prompt=prompt,
                    points=[],
                    source_image=image,
                    reason="missing_prompt",
                    content="SAM3 segmentation failed: missing prompt.",
                ),
                mcp_called=False,
                reason="missing_prompt",
            )
        if mode == "points" and segment_points is None:
            unavailable_reason = (
                "point_prompt_not_configured"
                if legacy_point_prompt
                else "point_backend_unavailable"
            )
            return finish(
                _sam3_failure(
                    mode=mode,
                    prompt="",
                    points=[],
                    source_image=image,
                    reason=unavailable_reason,
                    content="SAM3 point segmentation failed: point backend unavailable.",
                ),
                mcp_called=False,
                reason=unavailable_reason,
            )

        if roi_bbox_value is not None and mode == "points":
            return finish(
                _sam3_failure(
                    mode=mode,
                    prompt=prompt,
                    points=[],
                    source_image=image,
                    reason="conflicting_prompts",
                    content=(
                        "SAM3 segmentation failed: ROI and point prompts cannot be "
                        "combined."
                    ),
                ),
                mcp_called=False,
                reason="conflicting_prompts",
            )
        roi_metadata: JsonDict = {}
        roi_artifact: JsonDict | None = None
        positive_points: list[JsonDict] | None = None
        try:
            if legacy_point_prompt:
                positive_points = _validate_sam3_prompt_points(
                    positive_points_value,
                    image_size=_sam3_image_size(Path(image)),
                )
                image_base64, image_format = _encode_image_path(image)
            elif roi_bbox_value is None:
                image_base64, image_format = _encode_image_path(image)
            else:
                (
                    image_base64,
                    image_format,
                    roi_metadata,
                    roi_artifact,
                ) = _prepare_sam3_roi_attention_image(
                    source_image=Path(image),
                    roi_bbox_xyxy=roi_bbox_value,
                    output_root=artifact_session_root(image_output_root, session_id),
                )
        except FileNotFoundError:
            return finish(
                _sam3_failure(
                    mode=mode,
                    prompt=prompt,
                    points=[],
                    source_image=image,
                    reason="image_not_found",
                    content="SAM3 segmentation failed: image not found.",
                ),
                mcp_called=False,
                reason="image_not_found",
            )
        except ValueError as exc:
            reason = (
                "invalid_positive_points"
                if legacy_point_prompt
                else "invalid_roi_bbox"
            )
            return finish(
                _sam3_failure(
                    mode=mode,
                    prompt=prompt,
                    points=[],
                    source_image=image,
                    reason=reason,
                    content=(
                        "SAM3 segmentation failed: invalid prompt geometry: "
                        f"{exc}"
                    ),
                ),
                mcp_called=False,
                reason=reason,
            )
        except Exception as exc:  # noqa: BLE001 - image IO must stay structured.
            return finish(
                _sam3_failure(
                    mode=mode,
                    prompt=prompt,
                    points=[],
                    source_image=image,
                    reason="image_encode_failed",
                    content=f"SAM3 segmentation failed: image encode failed: {exc}",
                    metadata={"error_type": type(exc).__name__},
                ),
                mcp_called=False,
                reason="image_encode_failed",
            )

        if mode == "points":
            try:
                points = _normalise_sam3_tool_points(raw_points, image_path=Path(image))
            except _Sam3ToolInputError as exc:
                return finish(
                    _sam3_failure(
                        mode=mode,
                        prompt="",
                        points=[],
                        source_image=image,
                        reason=exc.reason,
                        content=f"SAM3 point segmentation failed: {exc}.",
                    ),
                    mcp_called=False,
                    reason=exc.reason,
                )
            context.parameters["points"] = points
            if legacy_point_prompt:
                context.parameters["positive_points"] = points
            positive_points = points

        mcp_request: JsonDict = {
            "image_base64": image_base64,
            "image_format": image_format,
        }
        if mode == "text":
            mcp_request["prompt"] = prompt
            call = segment
        else:
            mcp_request["points"] = points
            call = segment_points
        fallback_attempted = False
        sam_prompt_used = prompt or "point_prompt"
        depth_prefetch_metadata: JsonDict = {}
        if depth_prior_prefetch is not None:
            try:
                depth_prefetch_metadata = depth_prior_prefetch(context, image)
            except Exception as exc:  # noqa: BLE001 - prefetch never blocks SAM3.
                depth_prefetch_metadata = {
                    "status": "failed",
                    "reason": "prefetch_start_failed",
                    "error_type": type(exc).__name__,
                }
        try:
            response = call(mcp_request)
            if roi_metadata and _sam3_response_has_no_detections(response):
                fallback_attempted = True
                sam_prompt_used = DEFAULT_SAM3_ROI_FALLBACK_PROMPT
                response = segment(
                    {
                        "image_base64": image_base64,
                        "image_format": image_format,
                        "prompt": sam_prompt_used,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - tool failures must stay structured.
            return finish(
                _sam3_failure(
                    mode=mode,
                    prompt=prompt,
                    points=points,
                    source_image=image,
                    reason="mcp_call_failed",
                    content=f"SAM3 segmentation failed: MCP call failed: {exc}",
                    metadata={"error_type": type(exc).__name__},
                ),
                mcp_called=True,
                reason="mcp_call_failed",
            )
        if (
            mode == "points"
            and not legacy_point_prompt
            and not _is_consistent_sam3_point_response(
            response,
            points=points,
            source_image=Path(image),
            )
        ):
            return finish(
                _sam3_failure(
                    mode=mode,
                    prompt="",
                    points=points,
                    source_image=image,
                    reason="inconsistent_detection_outputs",
                    content="SAM3 returned inconsistent point-prompt outputs.",
                ),
                mcp_called=True,
                response=response,
                reason="inconsistent_detection_outputs",
            )
        result = _normalise_sam3_response(
            response,
            mode=mode,
            prompt=prompt,
            points=points,
            source_image=image,
            request={
                "image": image,
                "mode": mode,
                "image_format": image_format,
                "prompt": prompt,
                **roi_metadata,
                **({"positive_points": positive_points} if positive_points is not None else {}),
                "sam_prompt_used": sam_prompt_used,
                "fallback_attempted": fallback_attempted,
            },
            run_dir=run_dir,
            artifacts_dir=artifacts_dir,
            roi_bbox_xyxy=roi_metadata.get("effective_roi_bbox_xyxy"),
            extra_artifacts=[roi_artifact] if roi_artifact is not None else [],
            output_metadata={
                **roi_metadata,
                **({"positive_points": positive_points} if positive_points is not None else {}),
                "segmentation_mode": (
                    "point_prompt"
                    if positive_points is not None
                    else "roi_attention"
                    if roi_metadata
                    else "full_frame"
                ),
                "sam_prompt_used": sam_prompt_used,
                "fallback_attempted": fallback_attempted,
                "fallback_prompt": (
                    DEFAULT_SAM3_ROI_FALLBACK_PROMPT if fallback_attempted else None
                ),
                **(
                    {"depth_prior_prefetch": depth_prefetch_metadata}
                    if depth_prefetch_metadata
                    else {}
                ),
            },
        )
        reason = "" if result.success else _string_param(result.details.get("reason"))
        return finish(
            result,
            mcp_called=True,
            response=response,
            reason=reason,
        )

    return handler


def _resolve_current_observation_rgb_path(image: str, observation: Any) -> str:
    """Resolve a frame id only from the current observation's RGB artifacts."""

    return _resolve_current_observation_artifact_path(
        image,
        observation,
        kind="rgb",
        allow_frame_alias=True,
    )


def _resolve_current_observation_artifact_path(
    requested: str,
    observation: Any,
    *,
    kind: str,
    allow_frame_alias: bool = False,
) -> str:
    """Repair one missing path only from a unique current-observation artifact."""

    if not requested or Path(requested).is_file() or observation is None:
        return requested
    metadata = getattr(observation, "metadata", None)
    if not isinstance(metadata, dict):
        return requested
    artifacts = metadata.get("image_artifacts")
    if not isinstance(artifacts, list):
        return requested
    requested_name = Path(requested).name
    basename_matches: list[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        if artifact.get("kind") != kind:
            continue
        path = artifact.get("path")
        if not isinstance(path, str) or not path:
            continue
        if allow_frame_alias and artifact.get("frame_id") == requested:
            return path
        if Path(path).name == requested_name and Path(path).is_file():
            basename_matches.append(path)
    if len(basename_matches) == 1:
        return basename_matches[0]
    return requested


def build_stdio_sam3_mcp_segmenter(
    *,
    command: str,
    args: Sequence[str] | None = None,
    cwd: str | Path | None = None,
    tool_name: str = "segment",
    timeout_seconds: float = 600.0,
) -> Sam3SegmentCallable:
    """Build a synchronous callable that invokes one stdio MCP call per request."""

    def segment(request: JsonDict) -> JsonDict:
        return asyncio.run(
            _call_stdio_mcp_tool(
                command=command,
                args=list(args or ()),
                cwd=str(cwd) if cwd is not None else None,
                tool_name=tool_name,
                arguments=request,
                timeout_seconds=timeout_seconds,
            )
        )

    return segment


def build_sse_sam3_mcp_segmenter(
    *,
    url: str,
    tool_name: str = "segment",
    timeout_seconds: float = 600.0,
) -> Sam3SegmentCallable:
    """Build a synchronous SAM3 callable for an already-running SSE MCP server."""

    transport = SseSimulatorMcpTransport(url)

    def segment(request: JsonDict) -> JsonDict:
        return transport.call_tool(tool_name, request, timeout_s=timeout_seconds)

    return segment


def build_depth_prior_handler(
    estimate: DepthPriorCallable,
    *,
    output_root: str | Path = DEFAULT_DEPTH_PRIOR_OUTPUT_ROOT,
) -> ToolHandler:
    """Build a metric monocular depth-prior handler backed by an MCP callable."""

    result_output_root = Path(output_root)

    def handler(context: ToolExecutionContext) -> ToolResult:
        session_id = artifact_session_id(context.metadata)
        run_dir = _create_run_dir(artifact_session_root(result_output_root, session_id))
        request_ref = run_dir / "request.json"
        raw_output_ref = run_dir / "response.raw.json"
        tool_result_ref = run_dir / "tool_result.json"
        rgb = _string_param(context.parameters.get("rgb"))
        intrinsics = context.parameters.get("intrinsics")
        camera_id = _string_param(context.parameters.get("camera_id")) or "camera"
        camera_model = _string_param(context.parameters.get("camera_model")) or "pinhole"
        calibration_profile_id = _string_param(
            context.parameters.get("calibration_profile_id")
        )
        bundle_id = _string_param(context.parameters.get("bundle_id")) or run_dir.name
        if not rgb or not isinstance(intrinsics, Mapping):
            return _depth_prior_failure(
                context,
                reason="invalid_request",
                content="Depth prior estimation failed: rgb and intrinsics are required.",
            )
        try:
            rgb_payload = _encode_file_payload(rgb)
        except FileNotFoundError:
            return _depth_prior_failure(
                context,
                reason="rgb_not_found",
                content="Depth prior estimation failed: rgb file not found.",
            )
        request: JsonDict = {
            "rgb": rgb_payload,
            "intrinsics": dict(intrinsics),
            "camera_id": camera_id,
            "camera_model": camera_model,
            "calibration_profile_id": calibration_profile_id,
            "bundle_id": bundle_id,
        }
        if context.parameters.get("resolution_level") is not None:
            request["resolution_level"] = context.parameters["resolution_level"]
        _write_json(request_ref, _scrub_depth_prior_payload(request))
        try:
            response = estimate(request)
        except Exception as exc:  # noqa: BLE001 - MCP failures are structured.
            return _depth_prior_failure(
                context,
                reason="mcp_call_failed",
                content=f"Depth prior estimation failed: MCP call failed: {exc}",
                metadata={"error_type": type(exc).__name__},
            )
        _write_json(raw_output_ref, _scrub_depth_prior_payload(response))
        try:
            normalized = _normalise_depth_prior_response(
                response,
                run_dir=run_dir,
                camera_id=camera_id,
                source_rgb=rgb,
                request_ref=request_ref,
                raw_output_ref=raw_output_ref,
            )
        except Exception as exc:  # noqa: BLE001 - malformed model payload.
            return _depth_prior_failure(
                context,
                reason="invalid_mcp_response",
                content=(
                    "Depth prior estimation failed: invalid model payload: "
                    f"{type(exc).__name__}: {exc}"
                ),
                metadata={"error_type": type(exc).__name__},
            )
        normalized.details["tool_result_ref"] = str(tool_result_ref)
        normalized.details.setdefault("artifacts", []).append(
            {
                "type": "depth_prior_tool_result",
                "kind": "json",
                "tool": "estimate_depth_prior",
                "index": "tool_result",
                "path": str(tool_result_ref),
            }
        )
        _write_json(
            tool_result_ref,
            {
                "success": normalized.success,
                "content": normalized.content,
                "details": normalized.details,
            },
        )
        return normalized

    return handler


def build_sse_depth_prior_mcp_estimator(
    *,
    url: str,
    tool_name: str = "estimate_depth",
    timeout_seconds: float = 600.0,
) -> DepthPriorCallable:
    """Build a synchronous depth-prior estimator for an SSE MCP server."""

    transport = SseSimulatorMcpTransport(url)

    def estimate(request: JsonDict) -> JsonDict:
        return transport.call_tool(tool_name, request, timeout_s=timeout_seconds)

    return estimate


def build_anygrasp_handler(
    detect_grasps: AnyGraspDetectCallable,
    *,
    output_root: str | Path = DEFAULT_ANYGRASP_OUTPUT_ROOT,
) -> ToolHandler:
    """Build an AnyGrasp ToolRegistry handler backed by an injected MCP callable."""

    def handler(context: ToolExecutionContext) -> ToolResult:
        session_id = artifact_session_id(context.metadata)
        mode = _string_param(context.parameters.get("mode")) or "targeted"
        rgb = _string_param(context.parameters.get("rgb"))
        depth = _string_param(context.parameters.get("depth"))
        rgb = _resolve_current_observation_artifact_path(
            rgb,
            context.observation,
            kind="rgb",
        )
        depth = _resolve_current_observation_artifact_path(
            depth,
            context.observation,
            kind="depth",
        )
        target_mask = _string_param(context.parameters.get("target_mask"))
        intrinsics = context.parameters.get("intrinsics")
        try:
            depth_cutoff_factor = float(context.parameters.get("depth_cutoff_factor", 1.0))
        except (TypeError, ValueError):
            depth_cutoff_factor = math.nan
        if not rgb:
            return _anygrasp_failure(
                mode=mode,
                source_rgb=rgb,
                source_depth=depth,
                target_mask=target_mask,
                reason="missing_rgb",
                content="AnyGrasp grasp detection failed: missing rgb.",
            )
        if not depth:
            return _anygrasp_failure(
                mode=mode,
                source_rgb=rgb,
                source_depth=depth,
                target_mask=target_mask,
                reason="missing_depth",
                content="AnyGrasp grasp detection failed: missing depth.",
            )
        if not isinstance(intrinsics, dict):
            return _anygrasp_failure(
                mode=mode,
                source_rgb=rgb,
                source_depth=depth,
                target_mask=target_mask,
                reason="missing_intrinsics",
                content="AnyGrasp grasp detection failed: missing intrinsics.",
            )
        if mode not in {"targeted", "scene"}:
            return _anygrasp_failure(
                mode=mode,
                source_rgb=rgb,
                source_depth=depth,
                target_mask=target_mask,
                reason="invalid_mode",
                content="AnyGrasp grasp detection failed: invalid mode.",
            )
        if (
            not math.isfinite(depth_cutoff_factor)
            or depth_cutoff_factor < 1.0
            or depth_cutoff_factor > 4.0
        ):
            return _anygrasp_failure(
                mode=mode,
                source_rgb=rgb,
                source_depth=depth,
                target_mask=target_mask,
                reason="invalid_depth_cutoff_factor",
                content=(
                    "AnyGrasp grasp detection failed: depth_cutoff_factor "
                    "must be finite and in [1, 4]."
                ),
            )
        if mode == "targeted" and not target_mask:
            return _anygrasp_failure(
                mode=mode,
                source_rgb=rgb,
                source_depth=depth,
                target_mask=target_mask,
                reason="missing_target_mask",
                content="AnyGrasp grasp detection failed: missing target mask.",
            )
        if mode == "scene" and target_mask:
            return _anygrasp_failure(
                mode=mode,
                source_rgb=rgb,
                source_depth=depth,
                target_mask=target_mask,
                reason="target_mask_not_allowed_in_scene_mode",
                content="AnyGrasp grasp detection failed: target mask is not allowed in scene mode.",
            )

        try:
            rgb_payload = _encode_file_payload(rgb)
        except FileNotFoundError:
            return _anygrasp_failure(
                mode=mode,
                source_rgb=rgb,
                source_depth=depth,
                target_mask=target_mask,
                reason="rgb_not_found",
                content="AnyGrasp grasp detection failed: rgb file not found.",
            )
        try:
            depth_payload = _encode_file_payload(depth)
        except FileNotFoundError:
            return _anygrasp_failure(
                mode=mode,
                source_rgb=rgb,
                source_depth=depth,
                target_mask=target_mask,
                reason="depth_not_found",
                content="AnyGrasp grasp detection failed: depth file not found.",
            )
        target_mask_payload = None
        if target_mask:
            try:
                target_mask_payload = _encode_file_payload(target_mask)
            except FileNotFoundError:
                return _anygrasp_failure(
                    mode=mode,
                    source_rgb=rgb,
                    source_depth=depth,
                    target_mask=target_mask,
                    reason="target_mask_not_found",
                    content="AnyGrasp grasp detection failed: target mask file not found.",
                )

        request = {
            "mode": mode,
            "rgb": rgb,
            "depth": depth,
            "target_mask": target_mask or None,
            "intrinsics": dict(intrinsics),
            "approach_steering": context.parameters.get("approach_steering"),
            "approach_thresh": context.parameters.get("approach_thresh"),
            "collision_detection": context.parameters.get("collision_detection", True),
            "dense_grasp": context.parameters.get("dense_grasp", False),
            "depth_cutoff_factor": depth_cutoff_factor,
        }
        service_intrinsics = dict(intrinsics)
        try:
            service_intrinsics["scale"] = (
                float(service_intrinsics["scale"]) * depth_cutoff_factor
            )
        except (KeyError, TypeError, ValueError):
            pass
        mcp_request = {
            **request,
            "rgb": rgb_payload,
            "depth": depth_payload,
            "target_mask": target_mask_payload,
            "intrinsics": service_intrinsics,
        }
        if target_mask_payload is None:
            mcp_request.pop("target_mask", None)

        try:
            response = detect_grasps(mcp_request)
        except Exception as exc:  # noqa: BLE001 - tool failures must stay structured.
            return _anygrasp_failure(
                mode=mode,
                source_rgb=rgb,
                source_depth=depth,
                target_mask=target_mask,
                reason="mcp_call_failed",
                content=f"AnyGrasp grasp detection failed: MCP call failed: {exc}",
                metadata={"error_type": type(exc).__name__},
            )
        response = _restore_anygrasp_length_scale(response, depth_cutoff_factor)
        return _normalise_anygrasp_response(
            response,
            mode=mode,
            source_rgb=rgb,
            source_depth=depth,
            target_mask=target_mask,
            request=request,
            output_root=artifact_session_root(output_root, session_id),
        )

    return handler


def build_grasp_pose_estimate_handler(
    backend_handlers: Mapping[str, ToolHandler],
    *,
    backend_order: Sequence[str] = DEFAULT_GRASP_POSE_BACKEND_ORDER,
    graspgenx_gripper_name: str = "franka_panda",
    graspgenx_up_direction_camera: Sequence[float] = (0.0, 0.0, -1.0),
) -> ToolHandler:
    """Build one agent-facing grasp estimator over independent backend handlers."""

    handlers = {
        str(name): handler
        for name, handler in backend_handlers.items()
        if str(name) in DEFAULT_GRASP_POSE_BACKEND_ORDER
    }
    order = tuple(
        name
        for name in (str(value) for value in backend_order)
        if name in DEFAULT_GRASP_POSE_BACKEND_ORDER
    )

    def handler(context: ToolExecutionContext) -> ToolResult:
        parameters = context.parameters
        mode = _string_param(parameters.get("mode")) or "targeted"
        rgb = _string_param(parameters.get("rgb"))
        depth = _string_param(parameters.get("depth"))
        object_mask_value = parameters.get("object_mask")
        intrinsics_value = parameters.get("intrinsics")
        camera_frame_id = _string_param(parameters.get("camera_frame_id"))
        scene_epoch = parameters.get("scene_epoch")
        hints_value = parameters.get("hints")
        hints = dict(hints_value) if isinstance(hints_value, Mapping) else {}
        excluded_value = hints.get("excluded_backends")
        excluded_backends = {
            str(value)
            for value in excluded_value
            if isinstance(value, str) and value in DEFAULT_GRASP_POSE_BACKEND_ORDER
        } if isinstance(excluded_value, list | tuple | set) else set()

        invalid_reason = _grasp_pose_common_input_error(
            mode=mode,
            rgb=rgb,
            depth=depth,
            object_mask=object_mask_value,
            intrinsics=intrinsics_value,
            camera_frame_id=camera_frame_id,
            scene_epoch=scene_epoch,
        )
        if invalid_reason:
            return _grasp_pose_estimate_failure(
                invalid_reason,
                attempts=[],
                retryable=False,
            )

        assert isinstance(intrinsics_value, Mapping)
        intrinsics = dict(intrinsics_value)
        object_mask = (
            dict(object_mask_value) if isinstance(object_mask_value, Mapping) else None
        )
        attempts: list[JsonDict] = []
        for backend in order:
            if backend in excluded_backends:
                attempts.append(
                    {
                        "backend": backend,
                        "status": "skipped",
                        "reason": "excluded_by_host_fallback",
                        "candidate_count": 0,
                    }
                )
                continue
            backend_handler = handlers.get(backend)
            if backend_handler is None:
                attempts.append(
                    {
                        "backend": backend,
                        "status": "unavailable",
                        "reason": "backend_unavailable",
                    }
                )
                continue
            backend_parameters = _grasp_pose_backend_parameters(
                backend,
                mode=mode,
                rgb=rgb,
                depth=depth,
                object_mask=object_mask,
                intrinsics=intrinsics,
                hints=hints,
                graspgenx_gripper_name=graspgenx_gripper_name,
                graspgenx_up_direction_camera=graspgenx_up_direction_camera,
            )
            if backend_parameters is None:
                attempts.append(
                    {
                        "backend": backend,
                        "status": "ineligible",
                        "reason": "backend_incompatible",
                    }
                )
                continue
            started = time.monotonic()
            backend_result = backend_handler(
                ToolExecutionContext(
                    name=backend,
                    spec=context.spec,
                    parameters=backend_parameters,
                    observation=context.observation,
                    metadata=dict(context.metadata),
                )
            )
            elapsed_ms = round((time.monotonic() - started) * 1000.0, 3)
            reason = _grasp_backend_failure_reason(backend_result)
            attempt = {
                "backend": backend,
                "status": "success" if backend_result.success else "failed",
                "reason": "" if backend_result.success else reason,
                "candidate_count": _grasp_backend_candidate_count(backend_result),
                "elapsed_ms": elapsed_ms,
            }
            attempts.append(attempt)
            if backend_result.success:
                normalized = _normalise_grasp_pose_estimate_result(
                    backend_result,
                    backend=backend,
                    attempts=attempts,
                    mode=mode,
                    rgb=rgb,
                    depth=depth,
                    object_mask=object_mask,
                    intrinsics=intrinsics,
                    camera_frame_id=camera_frame_id,
                    scene_epoch=int(scene_epoch),
                    hints=hints,
                )
                if normalized.success:
                    return normalized
                reason = _grasp_backend_failure_reason(normalized)
                attempt["status"] = "failed"
                attempt["reason"] = reason
                if reason in GRASP_POSE_FALLBACK_REASONS:
                    continue
                return normalized
            if reason not in GRASP_POSE_FALLBACK_REASONS:
                return _grasp_pose_estimate_failure(
                    reason,
                    attempts=attempts,
                    retryable=False,
                    content=backend_result.content,
                )

        reason = (
            "all_backends_failed"
            if any(attempt["status"] == "failed" for attempt in attempts)
            else "no_compatible_backend"
        )
        return _grasp_pose_estimate_failure(
            reason,
            attempts=attempts,
            retryable=reason == "all_backends_failed",
        )

    return handler


def build_contact_graspnet_handler(
    predict_grasps: ContactGraspNetPredictCallable,
    *,
    output_root: str | Path = DEFAULT_CONTACT_GRASPNET_OUTPUT_ROOT,
) -> ToolHandler:
    """Build a targeted Contact-GraspNet handler backed by an MCP callable."""

    def handler(context: ToolExecutionContext) -> ToolResult:
        root = artifact_session_root(
            output_root,
            artifact_session_id(context.metadata),
        )
        run_dir = _create_run_dir(root)
        request_ref = run_dir / "request.json"
        raw_output_ref = run_dir / "response.raw.json"
        tool_result_ref = run_dir / "tool_result.json"

        rgb = _string_param(context.parameters.get("rgb"))
        depth = _string_param(context.parameters.get("depth"))
        object_mask_value = context.parameters.get("object_mask")
        intrinsics_value = context.parameters.get("intrinsics")
        object_mask_request = (
            dict(object_mask_value) if isinstance(object_mask_value, Mapping) else object_mask_value
        )
        request: JsonDict = {
            "rgb": rgb,
            "depth": depth,
            "object_mask": object_mask_request,
            "intrinsics": (
                dict(intrinsics_value)
                if isinstance(intrinsics_value, Mapping)
                else intrinsics_value
            ),
        }
        _write_json(request_ref, _scrub_contact_graspnet_payload(request))

        def finish(result: ToolResult, *, response: JsonDict | None = None) -> ToolResult:
            details = dict(result.details)
            details["request_ref"] = str(request_ref)
            details["raw_output_ref"] = str(raw_output_ref) if response is not None else None
            details["tool_result_ref"] = str(tool_result_ref)
            result.details = details
            if response is not None:
                _write_json(raw_output_ref, _scrub_contact_graspnet_payload(response))
            _write_json(
                tool_result_ref,
                {"success": result.success, "content": result.content, "details": details},
            )
            return result

        if not rgb:
            return finish(_contact_graspnet_failure("missing_rgb"))
        if not depth:
            return finish(_contact_graspnet_failure("missing_depth"))
        if not isinstance(object_mask_value, Mapping):
            return finish(_contact_graspnet_failure("invalid_object_mask"))
        mask_ref = _string_param(object_mask_value.get("mask_ref"))
        mask_source_image = _string_param(object_mask_value.get("source_image"))
        if not mask_ref or not mask_source_image:
            return finish(_contact_graspnet_failure("invalid_object_mask"))
        if intrinsics_value is None:
            return finish(_contact_graspnet_failure("missing_intrinsics"))
        intrinsics = _normalise_camera_intrinsics(intrinsics_value)
        if intrinsics is None:
            return finish(_contact_graspnet_failure("invalid_intrinsics"))

        if not Path(rgb).expanduser().is_file():
            return finish(_contact_graspnet_failure("rgb_not_found"))
        if not Path(depth).expanduser().is_file():
            return finish(_contact_graspnet_failure("depth_not_found"))
        if not Path(mask_ref).expanduser().is_file():
            return finish(_contact_graspnet_failure("object_mask_not_found"))
        if not _same_resolved_path(rgb, mask_source_image):
            return finish(_contact_graspnet_failure("object_mask_source_mismatch"))

        try:
            depth_payload = _encode_file_payload(depth)
            mask_payload = _encode_file_payload(mask_ref)
        except OSError as exc:
            return finish(
                _contact_graspnet_failure(
                    "input_encode_failed",
                    metadata={"error_type": type(exc).__name__},
                )
            )

        try:
            response = predict_grasps(
                {
                    "depth": depth_payload,
                    "object_mask": mask_payload,
                    "intrinsics": intrinsics,
                }
            )
        except Exception as exc:  # noqa: BLE001 - transport failures stay structured.
            return finish(
                _contact_graspnet_failure(
                    "mcp_call_failed",
                    metadata={"error_type": type(exc).__name__},
                )
            )
        result = _normalise_contact_graspnet_response(
            response,
            source_rgb=rgb,
            source_depth=depth,
            object_mask=mask_ref,
            intrinsics=intrinsics,
        )
        return finish(result, response=response if isinstance(response, dict) else None)

    return handler


def build_graspgenx_handler(
    predict_grasps: GraspGenXPredictCallable,
    list_grippers: GraspGenXListCallable,
    *,
    output_root: str | Path = DEFAULT_GRASPGENX_OUTPUT_ROOT,
) -> ToolHandler:
    """Build a targeted GraspGenX handler backed by MCP callables."""

    root = Path(output_root)

    def handler(context: ToolExecutionContext) -> ToolResult:
        run_dir = _create_run_dir(root)
        request_ref = run_dir / "request.json"
        raw_output_ref = run_dir / "response.raw.json"
        tool_result_ref = run_dir / "tool_result.json"

        rgb = _string_param(context.parameters.get("rgb"))
        depth = _string_param(context.parameters.get("depth"))
        object_mask_value = context.parameters.get("object_mask")
        intrinsics_value = context.parameters.get("intrinsics")
        gripper_name = _string_param(context.parameters.get("gripper_name"))
        up_value = context.parameters.get("up_direction_camera")
        object_mask_request = (
            dict(object_mask_value)
            if isinstance(object_mask_value, Mapping)
            else object_mask_value
        )
        request: JsonDict = {
            "rgb": rgb,
            "depth": depth,
            "object_mask": object_mask_request,
            "intrinsics": (
                dict(intrinsics_value)
                if isinstance(intrinsics_value, Mapping)
                else intrinsics_value
            ),
            "gripper_name": gripper_name,
            "up_direction_camera": up_value,
        }

        def finish(
            result: ToolResult,
            *,
            mcp_called: bool,
            response: Any = None,
            reason: str = "",
        ) -> ToolResult:
            _write_json(request_ref, _scrub_graspgenx_payload(request))
            raw_record: JsonDict = {"mcp_called": mcp_called}
            if isinstance(response, Mapping):
                raw_record["response"] = _scrub_graspgenx_payload(response)
            if reason:
                raw_record["reason"] = reason
            _write_json(raw_output_ref, raw_record)

            details = dict(result.details)
            details["request_ref"] = str(request_ref)
            details["raw_output_ref"] = str(raw_output_ref)
            details["tool_result_ref"] = str(tool_result_ref)
            details.setdefault("result_id", run_dir.name)
            artifacts = [
                dict(artifact)
                for artifact in _list_or_empty(details.get("artifacts"))
                if isinstance(artifact, Mapping)
            ]
            artifacts.extend(
                [
                    {
                        "type": "graspgenx_request",
                        "kind": "json",
                        "tool": "graspgenx",
                        "path": str(request_ref),
                    },
                    {
                        "type": "graspgenx_raw_response",
                        "kind": "json",
                        "tool": "graspgenx",
                        "path": str(raw_output_ref),
                    },
                    {
                        "type": "graspgenx_tool_result",
                        "kind": "json",
                        "tool": "graspgenx",
                        "path": str(tool_result_ref),
                    },
                ]
            )
            details["artifacts"] = artifacts
            result.details = details
            _write_json(
                tool_result_ref,
                {"success": result.success, "content": result.content, "details": details},
            )
            return result

        def fail(reason: str, *, metadata: JsonDict | None = None) -> ToolResult:
            return finish(
                _graspgenx_failure(reason, metadata=metadata),
                mcp_called=False,
                reason=reason,
            )

        if not rgb:
            return fail("missing_rgb")
        if not depth:
            return fail("missing_depth")
        if not isinstance(object_mask_value, Mapping):
            return fail("invalid_object_mask")
        mask_ref = _string_param(object_mask_value.get("mask_ref"))
        mask_source_image = _string_param(object_mask_value.get("source_image"))
        if not mask_ref or not mask_source_image:
            return fail("invalid_object_mask")
        if intrinsics_value is None:
            return fail("missing_intrinsics")
        intrinsics = _normalise_camera_intrinsics(intrinsics_value)
        if intrinsics is None:
            return fail("invalid_intrinsics")
        if not gripper_name:
            return fail("missing_gripper_name")
        up_direction = _normalise_graspgenx_up_direction(up_value)
        if up_direction is None:
            return fail("invalid_up_direction_camera")

        request["intrinsics"] = intrinsics
        request["up_direction_camera"] = up_direction
        for path_value, reason in (
            (rgb, "rgb_not_found"),
            (depth, "depth_not_found"),
            (mask_ref, "object_mask_not_found"),
        ):
            if not Path(path_value).expanduser().is_file():
                return fail(reason)
        if not _same_resolved_path(rgb, mask_source_image):
            return fail("object_mask_source_mismatch")

        try:
            sizes = [_image_size(path) for path in (rgb, depth, mask_ref)]
        except Exception as exc:  # noqa: BLE001 - Pillow decoding boundary.
            return fail(
                "image_decode_failed",
                metadata={"error_type": type(exc).__name__},
            )
        if len(set(sizes)) != 1:
            return fail(
                "image_shape_mismatch",
                metadata={
                    "rgb_size": list(sizes[0]),
                    "depth_size": list(sizes[1]),
                    "object_mask_size": list(sizes[2]),
                },
            )

        try:
            depth_payload = _encode_file_payload(depth)
            mask_payload = _encode_file_payload(mask_ref)
        except OSError as exc:
            return fail(
                "input_encode_failed",
                metadata={"error_type": type(exc).__name__},
            )

        try:
            response = predict_grasps(
                {
                    "depth": depth_payload,
                    "object_mask": mask_payload,
                    "intrinsics": intrinsics,
                    "gripper_name": gripper_name,
                    "up_direction_camera": up_direction,
                }
            )
        except Exception as exc:  # noqa: BLE001 - transport failures stay structured.
            return finish(
                _graspgenx_failure(
                    "mcp_call_failed",
                    metadata={"error_type": type(exc).__name__},
                ),
                mcp_called=True,
                reason="mcp_call_failed",
            )

        result = _normalise_graspgenx_response(
            response,
            source_rgb=rgb,
            source_depth=depth,
            object_mask=mask_ref,
            intrinsics=intrinsics,
            gripper_name=gripper_name,
            up_direction_camera=up_direction,
        )
        if result.success:
            try:
                listing = list_grippers()
                grippers = _normalise_graspgenx_gripper_listing(listing)
                geometry = next(
                    item for item in grippers if item["name"] == gripper_name
                )
                raw_candidates = _graspgenx_raw_candidates(response)
                result.details["artifacts"] = _build_graspgenx_visual_artifacts(
                    rgb_path=rgb,
                    mask_path=mask_ref,
                    candidates=raw_candidates,
                    gripper_geometry=geometry,
                    intrinsics=intrinsics,
                    output_dir=run_dir,
                )
            except Exception as exc:  # noqa: BLE001 - visualization is auxiliary.
                diagnostics = [
                    dict(item)
                    for item in _list_or_empty(result.details.get("diagnostics"))
                    if isinstance(item, Mapping)
                ]
                diagnostics.append(
                    {
                        "code": "visualization_failed",
                        "error_type": type(exc).__name__,
                    }
                )
                result.details["diagnostics"] = diagnostics
        reason = "" if result.success else _string_param(result.details.get("reason"))
        return finish(
            result,
            mcp_called=True,
            response=response,
            reason=reason,
        )

    return handler


def build_graspgenx_gripper_list_handler(
    list_grippers: GraspGenXListCallable,
) -> ToolHandler:
    """Build the read-only agent capability-discovery handler."""

    def handler(_context: ToolExecutionContext) -> ToolResult:
        try:
            response = list_grippers()
        except Exception as exc:  # noqa: BLE001 - transport failures stay structured.
            return _graspgenx_gripper_list_failure(
                "mcp_call_failed",
                metadata={"error_type": type(exc).__name__},
            )
        if not isinstance(response, Mapping):
            return _graspgenx_gripper_list_failure("inconsistent_gripper_outputs")
        if not bool(response.get("success", False)):
            details = response.get("details")
            reason = (
                _string_param(details.get("reason"))
                if isinstance(details, Mapping)
                else "unknown_error"
            ) or "unknown_error"
            return _graspgenx_gripper_list_failure(reason)
        try:
            grippers = _normalise_graspgenx_gripper_listing(response)
        except ValueError:
            return _graspgenx_gripper_list_failure("inconsistent_gripper_outputs")
        details = response.get("details")
        assert isinstance(details, Mapping)
        return ToolResult(
            True,
            content=_string_param(response.get("content"))
            or "GraspGenX gripper listing completed.",
            details={
                "tool": "list_graspgenx_grippers",
                "backend": GRASPGENX_BACKEND,
                "model": GRASPGENX_MODEL,
                "gripper_count": len(grippers),
                "grippers": grippers,
                "model_loaded": bool(details.get("model_loaded", False)),
            },
        )

    return handler


def build_molmopoint_handler(
    point_images: MolmoPointCallable,
    *,
    output_root: str | Path = DEFAULT_MOLMOPOINT_OUTPUT_ROOT,
) -> ToolHandler:
    """Build a MolmoPoint handler backed by an injected MCP callable."""

    root = Path(output_root)

    def handler(context: ToolExecutionContext) -> ToolResult:
        run_dir = _create_run_dir(root)
        request_ref = run_dir / "request.json"
        raw_output_ref = run_dir / "response.raw.json"
        tool_result_ref = run_dir / "tool_result.json"
        prompt = _string_param(context.parameters.get("prompt"))
        raw_images = context.parameters.get("images")
        normalized_images = _best_effort_molmopoint_paths(raw_images)
        context.parameters = {"images": normalized_images, "prompt": prompt}

        def finish(
            result: ToolResult,
            *,
            mcp_called: bool,
            response: Any = None,
            reason: str = "",
        ) -> ToolResult:
            _write_json(request_ref, dict(context.parameters))
            raw_record: JsonDict = {"mcp_called": mcp_called}
            if isinstance(response, Mapping):
                raw_record["response"] = _scrub_molmopoint_payload(response)
            if reason:
                raw_record["reason"] = reason
            _write_json(raw_output_ref, raw_record)

            details = dict(result.details)
            artifacts = [
                artifact
                for artifact in _list_or_empty(details.get("artifacts"))
                if isinstance(artifact, dict)
            ]
            artifacts.extend(
                [
                    {
                        "type": "molmopoint_request",
                        "kind": "json",
                        "tool": "molmopoint",
                        "path": str(request_ref),
                    },
                    {
                        "type": "molmopoint_raw_response",
                        "kind": "json",
                        "tool": "molmopoint",
                        "path": str(raw_output_ref),
                    },
                    {
                        "type": "molmopoint_tool_result",
                        "kind": "json",
                        "tool": "molmopoint",
                        "path": str(tool_result_ref),
                    },
                ]
            )
            details["artifacts"] = artifacts
            result.details = details
            _write_json(
                tool_result_ref,
                {"success": result.success, "content": result.content, "details": details},
            )
            return result

        if not isinstance(raw_images, list) or not 1 <= len(raw_images) <= MOLMOPOINT_MAX_IMAGE_COUNT:
            return finish(
                _molmopoint_failure(context, "invalid_image_count"),
                mcp_called=False,
                reason="invalid_image_count",
            )
        if not prompt or len(prompt) > MOLMOPOINT_MAX_PROMPT_CHARS:
            return finish(
                _molmopoint_failure(context, "invalid_prompt"),
                mcp_called=False,
                reason="invalid_prompt",
            )

        try:
            normalized_images, payloads, image_metadata = _prepare_molmopoint_images(raw_images)
        except _MolmoPointHandlerError as exc:
            context.parameters = {"images": normalized_images, "prompt": prompt}
            return finish(
                _molmopoint_failure(context, exc.reason),
                mcp_called=False,
                reason=exc.reason,
            )
        context.parameters = {"images": normalized_images, "prompt": prompt}
        _write_json(request_ref, dict(context.parameters))
        try:
            response = point_images({"images": payloads, "prompt": prompt})
        except Exception:  # noqa: BLE001 - transport failures stay structured.
            return finish(
                _molmopoint_failure(context, "mcp_call_failed"),
                mcp_called=True,
                reason="mcp_call_failed",
            )

        result = _normalise_molmopoint_response(
            context,
            response,
            image_metadata=image_metadata,
        )
        if result.success:
            try:
                visual_artifacts = _build_molmopoint_visual_artifacts(
                    image_paths=normalized_images,
                    points=_list_or_empty(result.details.get("outputs", {}).get("points")),
                    output_dir=run_dir,
                )
            except Exception:  # noqa: BLE001 - invalid visuals invalidate trusted output.
                result = _molmopoint_failure(context, "visualization_failed")
            else:
                result.details["artifacts"] = visual_artifacts
        reason = ""
        if not result.success:
            diagnostics = _list_or_empty(result.details.get("diagnostics"))
            if diagnostics and isinstance(diagnostics[0], Mapping):
                reason = _string_param(diagnostics[0].get("code"))
        return finish(
            result,
            mcp_called=True,
            response=response,
            reason=reason,
        )

    return handler


def build_anyplace_handler(
    predict_placement: AnyPlacePredictCallable,
    *,
    output_root: str | Path = DEFAULT_ANYPLACE_OUTPUT_ROOT,
) -> ToolHandler:
    """Build an AnyPlace ToolRegistry handler backed by an injected MCP callable."""

    def handler(context: ToolExecutionContext) -> ToolResult:
        session_id = artifact_session_id(context.metadata)
        rgb = _string_param(context.parameters.get("rgb"))
        depth = _string_param(context.parameters.get("depth"))
        object_mask = _string_param(context.parameters.get("object_mask"))
        placement_region_mask = context.parameters.get("placement_region_mask")
        selected_grasp = context.parameters.get("selected_grasp")
        if not rgb:
            return _anyplace_failure(
                "missing_rgb", "AnyPlace placement prediction failed: missing rgb."
            )
        if not depth:
            return _anyplace_failure(
                "missing_depth", "AnyPlace placement prediction failed: missing depth."
            )
        if not object_mask:
            return _anyplace_failure(
                "missing_object_mask",
                "AnyPlace placement prediction failed: missing object mask.",
            )
        if not isinstance(placement_region_mask, Mapping):
            return _anyplace_failure(
                "invalid_placement_region_mask",
                "AnyPlace placement prediction failed: invalid placement region mask artifact.",
            )
        placement_mask_ref = _string_param(placement_region_mask.get("mask_ref"))
        placement_source_image = _string_param(placement_region_mask.get("source_image"))
        if not placement_mask_ref or not placement_source_image:
            return _anyplace_failure(
                "invalid_placement_region_mask",
                "AnyPlace placement prediction failed: placement mask requires "
                "mask_ref and source_image.",
            )
        if not isinstance(selected_grasp, Mapping):
            return _anyplace_failure(
                "selected_grasp_requires_targeted_source",
                "AnyPlace placement prediction failed: selected grasp requires "
                "targeted source provenance.",
            )
        candidate_value = selected_grasp.get("candidate")
        source_value = selected_grasp.get("source")
        if (
            not isinstance(source_value, Mapping)
            or _string_param(source_value.get("mode")) != "targeted"
        ):
            return _anyplace_failure(
                "selected_grasp_requires_targeted_source",
                "AnyPlace placement prediction failed: selected grasp requires "
                "targeted source provenance.",
            )
        source_rgb = _string_param(source_value.get("rgb"))
        source_depth = _string_param(source_value.get("depth"))
        source_object_mask = _string_param(source_value.get("object_mask"))
        try:
            depth_cutoff_factor = float(source_value.get("depth_cutoff_factor", 1.0))
        except (TypeError, ValueError):
            depth_cutoff_factor = math.nan
        raw_source_tool = _string_param(source_value.get("source_tool"))
        source_tool = raw_source_tool or "anygrasp"
        if source_tool not in {"grasp_pose_estimate", "anygrasp", "graspgenx"}:
            return _anyplace_failure(
                "selected_grasp_requires_targeted_source",
                "AnyPlace placement prediction failed: unsupported grasp source.",
            )
        source_backend = _string_param(source_value.get("source_backend")) or source_tool
        if not source_rgb or not source_depth or not source_object_mask:
            return _anyplace_failure(
                "selected_grasp_requires_targeted_source",
                "AnyPlace placement prediction failed: selected grasp source is incomplete.",
            )
        if (
            not math.isfinite(depth_cutoff_factor)
            or depth_cutoff_factor < 1.0
            or depth_cutoff_factor > 4.0
        ):
            return _anyplace_failure(
                "invalid_depth_cutoff_factor",
                "AnyPlace placement prediction failed: depth cutoff factor "
                "must be finite and in [1, 4].",
            )

        normalized_candidate = _normalise_anygrasp_candidate(candidate_value)
        if (
            not isinstance(candidate_value, dict)
            or normalized_candidate is None
            or normalized_candidate.get("camera_frame") != "opencv"
            or not _is_rotation_matrix3(normalized_candidate.get("rotation_matrix"))
            or any(normalized_candidate[key] < 0 for key in ("depth", "width", "height"))
        ):
            return _anyplace_failure(
                "invalid_selected_grasp",
                "AnyPlace placement prediction failed: invalid selected grasp candidate.",
            )
        source_gripper_name: str | None = None
        source_up_direction: list[float] | None = None
        if source_backend == "graspgenx":
            source_gripper_name = _string_param(source_value.get("gripper_name")) or None
            source_up_direction = _normalise_graspgenx_up_direction(
                source_value.get("up_direction_camera")
            )
            if (
                source_gripper_name is None
                or source_up_direction is None
                or normalized_candidate.get("gripper_name") != source_gripper_name
            ):
                return _anyplace_failure(
                    "selected_grasp_requires_targeted_source",
                    "AnyPlace placement prediction failed: incomplete GraspGenX source.",
                )
        normalized_candidate["source_tool"] = source_tool
        if source_gripper_name is not None:
            normalized_candidate["gripper_name"] = source_gripper_name
        intrinsics = _normalise_camera_intrinsics(context.parameters.get("intrinsics"))
        source_intrinsics = _normalise_camera_intrinsics(source_value.get("intrinsics"))
        if intrinsics is None:
            return _anyplace_failure(
                "invalid_intrinsics",
                "AnyPlace placement prediction failed: invalid intrinsics.",
            )
        if source_intrinsics is None or not _intrinsics_equal(intrinsics, source_intrinsics):
            return _anyplace_failure(
                "source_intrinsics_mismatch",
                "AnyPlace placement prediction failed: selected grasp intrinsics do not match.",
            )

        for left, right, reason, label in (
            (rgb, source_rgb, "source_rgb_mismatch", "rgb"),
            (depth, source_depth, "source_depth_mismatch", "depth"),
            (object_mask, source_object_mask, "source_object_mask_mismatch", "object mask"),
            (
                rgb,
                placement_source_image,
                "placement_mask_source_mismatch",
                "placement mask source",
            ),
        ):
            if not _same_resolved_path(left, right):
                return _anyplace_failure(
                    reason,
                    f"AnyPlace placement prediction failed: {label} provenance does not match.",
                )

        payloads: dict[str, JsonDict] = {}
        for key, path_value, reason in (
            ("rgb", rgb, "rgb_not_found"),
            ("depth", depth, "depth_not_found"),
            ("object_mask", object_mask, "object_mask_not_found"),
            ("placement_region_mask", placement_mask_ref, "placement_region_mask_not_found"),
        ):
            try:
                payloads[key] = _encode_file_payload(path_value)
            except FileNotFoundError:
                return _anyplace_failure(
                    reason,
                    f"AnyPlace placement prediction failed: {key} file not found.",
                )
            except OSError as exc:
                return _anyplace_failure(
                    "input_encode_failed",
                    f"AnyPlace placement prediction failed: input encode failed: {exc}",
                    metadata={"error_type": type(exc).__name__},
                )

        request: JsonDict = {
            "rgb": rgb,
            "depth": depth,
            "object_mask": object_mask,
            "placement_region_mask": dict(placement_region_mask),
            "intrinsics": intrinsics,
            "selected_grasp": {
                "candidate": dict(candidate_value),
                "source": {
                    "source_tool": source_tool,
                    "source_backend": source_backend,
                    "mode": "targeted",
                    "rgb": source_rgb,
                    "depth": source_depth,
                    "object_mask": source_object_mask,
                    "intrinsics": source_intrinsics,
                },
            },
        }
        if source_gripper_name is not None:
            request["selected_grasp"]["source"]["gripper_name"] = source_gripper_name
        if source_up_direction is not None:
            request["selected_grasp"]["source"][
                "up_direction_camera"
            ] = source_up_direction
        request = _scrub_anyplace_response(request)
        service_intrinsics = dict(intrinsics)
        service_intrinsics["scale"] = (
            float(service_intrinsics["scale"]) * depth_cutoff_factor
        )
        service_candidate = _scale_anyplace_grasp_candidate(
            normalized_candidate,
            depth_cutoff_factor,
        )
        mcp_request: JsonDict = {
            **payloads,
            "intrinsics": service_intrinsics,
            "selected_grasp": service_candidate,
        }
        try:
            response = predict_placement(mcp_request)
        except Exception as exc:  # noqa: BLE001 - tool failures must stay structured.
            return _anyplace_failure(
                "mcp_call_failed",
                f"AnyPlace placement prediction failed: MCP call failed: {exc}",
                metadata={"error_type": type(exc).__name__},
            )
        response = _restore_anyplace_length_scale(response, depth_cutoff_factor)
        return _normalise_anyplace_response(
            response,
            selected_grasp=normalized_candidate,
            selected_grasp_source=request["selected_grasp"]["source"],
            request=request,
            output_root=artifact_session_root(output_root, session_id),
        )

    return handler


def _observe_handler(context: ToolExecutionContext) -> ToolResult:
    observation = context.observation
    return make_tool_result(
        context,
        success=True,
        content="latest observation summarized",
        outputs={
            "camera_ids": [camera.frame_id for camera in observation.cameras]
            if observation
            else [],
            "objects": observation.objects if observation else [],
            "metadata": observation.metadata if observation else {},
        },
    )


def _scene_detector_handler(context: ToolExecutionContext) -> ToolResult:
    observation = context.observation
    object_names = []
    if observation is not None:
        object_names = [
            str(obj.get("name"))
            for obj in observation.objects
            if isinstance(obj, dict) and obj.get("name")
        ]
    objects = object_names or ["cube"]
    return make_tool_result(
        context,
        success=True,
        content="dummy scene objects detected",
        outputs={
            "image": context.parameters.get("image"),
            "objects": objects,
            "detections": [{"label": name, "source": context.name} for name in objects],
        },
        artifacts=[
            {
                "type": "object_list",
                "id": "dummy-scene-objects",
                "count": len(objects),
            }
        ],
    )


def _sam3_handler(context: ToolExecutionContext) -> ToolResult:
    prompt = context.parameters.get("prompt", "object")
    mask_id = f"mask-{str(prompt).replace(' ', '-')}-001"
    return make_tool_result(
        context,
        success=True,
        content="dummy segmentation mask generated",
        outputs={
            "image": context.parameters.get("image"),
            "prompt": prompt,
            "masks": [
                {
                    "mask_id": mask_id,
                    "label": prompt,
                    "score": 0.99,
                }
            ],
        },
        artifacts=[{"type": "segmentation_mask", "id": mask_id, "label": prompt}],
    )


def _anygrasp_handler(context: ToolExecutionContext) -> ToolResult:
    target_mask = context.parameters.get("target_mask")
    candidate = {
        "id": "grasp-1",
        "frame": "camera",
        "score": 0.91,
        "translation_xyz": [0.45, 0.0, 0.18],
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "depth": 0.03,
        "width": 0.06,
        "height": 0.03,
        "gripper_tip_position_xyz": [0.48, 0.0, 0.18],
    }
    return make_tool_result(
        context,
        success=True,
        content="dummy grasp candidates generated",
        outputs={
            "mode": context.parameters.get("mode", "targeted"),
            "rgb": context.parameters.get("rgb"),
            "depth": context.parameters.get("depth"),
            "target_mask": target_mask,
            "intrinsics": context.parameters.get("intrinsics"),
            "candidate_count": 1,
            "grasp_candidates": [candidate],
        },
        artifacts=[{"type": "grasp_candidates", "id": "dummy-grasps-001", "count": 1}],
    )


def _camera_pose_to_world_handler(context: ToolExecutionContext) -> ToolResult:
    camera_pose = context.parameters.get("camera_pose")
    camera_to_world = context.parameters.get("camera_to_world")
    extrinsics = (
        camera_to_world
        if camera_to_world is not None
        else context.parameters.get("camera_extrinsics")
    )
    camera_frame_id = _string_param(context.parameters.get("camera_frame_id")) or None
    convention = (
        _string_param(context.parameters.get("matrix_convention"))
        or _string_param(context.parameters.get("convention"))
        or "camera_to_world_row_major"
    )
    if not isinstance(camera_pose, Mapping):
        return _camera_pose_transform_failure(
            context,
            reason="missing_camera_pose",
            content="camera_pose_to_world failed: missing camera_pose.",
        )
    if not isinstance(extrinsics, Mapping) and not isinstance(extrinsics, list):
        return _camera_pose_transform_failure(
            context,
            reason="missing_camera_to_world",
            content="camera_pose_to_world failed: missing camera_to_world.",
        )
    frame = _string_param(camera_pose.get("frame"))
    if frame and frame != "camera":
        return _camera_pose_transform_failure(
            context,
            reason="invalid_frame",
            content="camera_pose_to_world failed: input pose is not in camera frame.",
            metadata={"frame": frame},
        )
    if convention not in {"camera_to_world_row_major", "world_to_camera_row_major"}:
        return _camera_pose_transform_failure(
            context,
            reason="invalid_convention",
            content="camera_pose_to_world failed: unsupported convention.",
            metadata={"convention": convention},
        )
    input_camera_frame = _canonical_camera_frame(
        context.parameters.get("input_camera_frame")
        or camera_pose.get("camera_frame")
        or camera_pose.get("camera_convention"),
        default="opencv",
    )
    if input_camera_frame is None:
        return _camera_pose_transform_failure(
            context,
            reason="invalid_input_camera_frame",
            content="camera_pose_to_world failed: unsupported input_camera_frame.",
            metadata={
                "input_camera_frame": context.parameters.get("input_camera_frame")
                or camera_pose.get("camera_frame")
                or camera_pose.get("camera_convention")
            },
        )
    translation = _finite_vector(camera_pose.get("translation_xyz"), length=3)
    if translation is None:
        return _camera_pose_transform_failure(
            context,
            reason="invalid_translation",
            content="camera_pose_to_world failed: camera_pose.translation_xyz must be 3 finite floats.",
        )

    rotation_value = camera_pose.get("rotation_matrix")
    rotation = None
    if rotation_value is not None:
        rotation = _finite_matrix3(rotation_value)
        if rotation is None:
            return _camera_pose_transform_failure(
                context,
                reason="invalid_rotation",
                content="camera_pose_to_world failed: camera_pose.rotation_matrix must be 3x3 finite floats.",
            )

    tip_value = camera_pose.get("gripper_tip_position_xyz")
    tip = None
    if tip_value is not None:
        tip = _finite_vector(tip_value, length=3)
        if tip is None:
            return _camera_pose_transform_failure(
                context,
                reason="invalid_gripper_tip",
                content=(
                    "camera_pose_to_world failed: "
                    "camera_pose.gripper_tip_position_xyz must be 3 finite floats."
                ),
            )

    parsed_extrinsics = _parse_camera_extrinsics(extrinsics)
    if parsed_extrinsics is None:
        return _camera_pose_transform_failure(
            context,
            reason="invalid_camera_to_world",
            content=(
                "camera_pose_to_world failed: camera_to_world must contain "
                "{pos: [x,y,z], mat: 9 floats}, a 4x4 matrix, or a "
                "camera_to_world/pose_mat matrix field."
            ),
        )
    camera_rotation, camera_position, source_format, default_camera_frame = parsed_extrinsics
    camera_to_world_frame = _canonical_camera_frame(
        context.parameters.get("camera_to_world_frame")
        or _camera_frame_from_extrinsics(extrinsics),
        default=default_camera_frame,
    )
    if camera_to_world_frame is None:
        return _camera_pose_transform_failure(
            context,
            reason="invalid_camera_to_world_frame",
            content="camera_pose_to_world failed: unsupported camera_to_world_frame.",
            metadata={
                "camera_to_world_frame": context.parameters.get("camera_to_world_frame")
                or _camera_frame_from_extrinsics(extrinsics)
            },
        )
    translation = _convert_camera_vector(
        translation,
        source_frame=input_camera_frame,
        target_frame=camera_to_world_frame,
    )
    if rotation is not None:
        rotation = _convert_camera_rotation(
            rotation,
            source_frame=input_camera_frame,
            target_frame=camera_to_world_frame,
        )
    if tip is not None:
        tip = _convert_camera_vector(
            tip,
            source_frame=input_camera_frame,
            target_frame=camera_to_world_frame,
        )
    if convention == "camera_to_world_row_major":
        world_translation = _mat3_vec3(camera_rotation, translation)
        world_translation = _vec3_add(world_translation, camera_position)
        world_rotation = _mat3_mat3(camera_rotation, rotation) if rotation is not None else None
        world_tip = _vec3_add(_mat3_vec3(camera_rotation, tip), camera_position) if tip else None
    else:
        inverse_rotation = _transpose3(camera_rotation)
        offset_translation = (
            translation if source_format == "pos_mat" else _vec3_sub(translation, camera_position)
        )
        world_translation = _mat3_vec3(inverse_rotation, offset_translation)
        if source_format == "pos_mat":
            world_translation = _vec3_add(world_translation, camera_position)
        world_rotation = _mat3_mat3(inverse_rotation, rotation) if rotation is not None else None
        if tip is not None:
            offset_tip = tip if source_format == "pos_mat" else _vec3_sub(tip, camera_position)
            world_tip = _mat3_vec3(inverse_rotation, offset_tip)
            if source_format == "pos_mat":
                world_tip = _vec3_add(world_tip, camera_position)
        else:
            world_tip = None

    world_pose: JsonDict = dict(camera_pose)
    world_pose["frame"] = "world"
    world_pose.pop("camera_frame", None)
    world_pose.pop("camera_convention", None)
    world_pose["translation_xyz"] = _round_vector(world_translation)
    if world_rotation is not None:
        world_pose["rotation_matrix"] = _round_matrix(world_rotation)
    if world_tip is not None:
        world_pose["gripper_tip_position_xyz"] = _round_vector(world_tip)

    return make_tool_result(
        context,
        success=True,
        content="camera-frame pose transformed to world frame",
        outputs={
            "frame": "world",
            "camera_frame_id": camera_frame_id,
            "matrix_convention": convention,
            "input_camera_frame": input_camera_frame,
            "camera_to_world_frame": camera_to_world_frame,
            "camera_to_world_format": source_format,
            "camera_to_world_matrix_layout": _matrix_layout_from_extrinsics(extrinsics),
            "world_pose": world_pose,
            "translation_xyz": world_pose["translation_xyz"],
            "rotation_matrix": world_pose.get("rotation_matrix"),
            "gripper_tip_position_xyz": world_pose.get("gripper_tip_position_xyz"),
        },
    )


def _hand_pose_handler(context: ToolExecutionContext) -> ToolResult:
    return make_tool_result(
        context,
        success=True,
        content="dummy hand pose retrieved",
        outputs={
            "object": context.parameters.get("object"),
            "pose_id": "hand-pose-cube-001",
        },
        artifacts=[{"type": "hand_pose", "id": "hand-pose-cube-001"}],
    )


def _ik_preview_handler(context: ToolExecutionContext) -> ToolResult:
    return make_tool_result(
        context,
        success=True,
        content="IK preview feasible in dummy scene",
        outputs={
            "feasible": True,
            "target_pose": context.parameters.get("target_pose"),
        },
    )


def _obstacle_avoidance_handler(context: ToolExecutionContext) -> ToolResult:
    return make_tool_result(
        context,
        success=True,
        content="No blocking obstacles in dummy scene",
        outputs={
            "clear": True,
            "path": context.parameters.get("path"),
        },
    )


def _approval_control_handler(
    approve_world_mutating: ApprovalCallback | None,
) -> Callable[[ToolExecutionContext], ToolResult]:
    def handler(context: ToolExecutionContext) -> ToolResult:
        approved = approve_world_mutating(context) if approve_world_mutating else True
        if not approved:
            return make_tool_result(
                context,
                success=False,
                content="User denied world-mutating command.",
                outputs={"approved": False},
                diagnostics=[{"code": "operator_denied"}],
            )
        return make_tool_result(
            context,
            success=True,
            content="Dummy world-mutating command approved and executed.",
            outputs={"approved": True},
            state_delta=_dummy_state_delta(context),
        )

    return handler


def build_stdio_anygrasp_mcp_grasper(
    *,
    command: str,
    args: Sequence[str] | None = None,
    cwd: str | Path | None = None,
    tool_name: str = "detect_grasps",
    timeout_seconds: float = 600.0,
) -> AnyGraspDetectCallable:
    """Build a synchronous callable that invokes one AnyGrasp stdio MCP call."""

    def detect_grasps(request: JsonDict) -> JsonDict:
        return asyncio.run(
            _call_stdio_mcp_tool(
                command=command,
                args=list(args or ()),
                cwd=str(cwd) if cwd is not None else None,
                tool_name=tool_name,
                arguments=request,
                timeout_seconds=timeout_seconds,
                invalid_payload=_invalid_mcp_payload(
                    tool="anygrasp",
                    backend="anygrasp_mcp",
                    model="anygrasp_sdk",
                    content="AnyGrasp grasp detection failed: invalid MCP response.",
                ),
            )
        )

    return detect_grasps


def build_sse_anygrasp_mcp_grasper(
    *,
    url: str,
    tool_name: str = "detect_grasps",
    timeout_seconds: float = 600.0,
) -> AnyGraspDetectCallable:
    """Build a synchronous AnyGrasp callable for an already-running SSE MCP server."""

    transport = SseSimulatorMcpTransport(url)

    def detect_grasps(request: JsonDict) -> JsonDict:
        return transport.call_tool(tool_name, request, timeout_s=timeout_seconds)

    return detect_grasps


def build_stdio_contact_graspnet_mcp_predictor(
    *,
    command: str,
    args: Sequence[str] | None = None,
    cwd: str | Path | None = None,
    tool_name: str = "predict_grasps",
    timeout_seconds: float = 600.0,
) -> ContactGraspNetPredictCallable:
    """Build a synchronous callable for one Contact-GraspNet stdio MCP call."""

    def predict_grasps(request: JsonDict) -> JsonDict:
        return asyncio.run(
            _call_stdio_mcp_tool(
                command=command,
                args=list(args or ()),
                cwd=str(cwd) if cwd is not None else None,
                tool_name=tool_name,
                arguments=request,
                timeout_seconds=timeout_seconds,
                invalid_payload=_invalid_mcp_payload(
                    tool="contact_graspnet",
                    backend="contact_graspnet_mcp",
                    model=CONTACT_GRASPNET_MODEL,
                    content="Contact-GraspNet grasp prediction failed: invalid MCP response.",
                ),
            )
        )

    return predict_grasps


def build_sse_contact_graspnet_mcp_predictor(
    *,
    url: str,
    tool_name: str = "predict_grasps",
    timeout_seconds: float = 600.0,
) -> ContactGraspNetPredictCallable:
    """Build a synchronous Contact-GraspNet callable for an SSE MCP server."""

    transport = SseSimulatorMcpTransport(url)

    def predict_grasps(request: JsonDict) -> JsonDict:
        return transport.call_tool(tool_name, request, timeout_s=timeout_seconds)

    return predict_grasps


def build_stdio_graspgenx_mcp_predictor(
    *,
    command: str,
    args: Sequence[str] | None = None,
    cwd: str | Path | None = None,
    tool_name: str = "predict_grasps",
    timeout_seconds: float = 600.0,
) -> GraspGenXPredictCallable:
    """Build a synchronous callable for one GraspGenX stdio prediction."""

    def predict_grasps(request: JsonDict) -> JsonDict:
        return asyncio.run(
            _call_stdio_mcp_tool(
                command=command,
                args=list(args or ()),
                cwd=str(cwd) if cwd is not None else None,
                tool_name=tool_name,
                arguments=request,
                timeout_seconds=timeout_seconds,
                invalid_payload=_invalid_mcp_payload(
                    tool="graspgenx",
                    backend=GRASPGENX_BACKEND,
                    model=GRASPGENX_MODEL,
                    content="GraspGenX grasp prediction failed: invalid MCP response.",
                ),
            )
        )

    return predict_grasps


def build_sse_graspgenx_mcp_predictor(
    *,
    url: str,
    tool_name: str = "predict_grasps",
    timeout_seconds: float = 600.0,
) -> GraspGenXPredictCallable:
    """Build a synchronous GraspGenX predictor for an SSE MCP server."""

    transport = SseSimulatorMcpTransport(url)

    def predict_grasps(request: JsonDict) -> JsonDict:
        return transport.call_tool(tool_name, request, timeout_s=timeout_seconds)

    return predict_grasps


def build_stdio_graspgenx_mcp_gripper_lister(
    *,
    command: str,
    args: Sequence[str] | None = None,
    cwd: str | Path | None = None,
    tool_name: str = "list_grippers",
    timeout_seconds: float = 600.0,
) -> GraspGenXListCallable:
    """Build a synchronous callable for one GraspGenX stdio capability query."""

    def list_grippers() -> JsonDict:
        return asyncio.run(
            _call_stdio_mcp_tool(
                command=command,
                args=list(args or ()),
                cwd=str(cwd) if cwd is not None else None,
                tool_name=tool_name,
                arguments={},
                timeout_seconds=timeout_seconds,
                invalid_payload=_invalid_mcp_payload(
                    tool="list_graspgenx_grippers",
                    backend=GRASPGENX_BACKEND,
                    model=GRASPGENX_MODEL,
                    content="GraspGenX gripper listing failed: invalid MCP response.",
                ),
            )
        )

    return list_grippers


def build_sse_graspgenx_mcp_gripper_lister(
    *,
    url: str,
    tool_name: str = "list_grippers",
    timeout_seconds: float = 600.0,
) -> GraspGenXListCallable:
    """Build a synchronous GraspGenX gripper lister for an SSE MCP server."""

    transport = SseSimulatorMcpTransport(url)

    def list_grippers() -> JsonDict:
        return transport.call_tool(tool_name, {}, timeout_s=timeout_seconds)

    return list_grippers


def build_stdio_molmopoint_mcp_pointer(
    *,
    command: str,
    args: Sequence[str] | None = None,
    cwd: str | Path | None = None,
    tool_name: str = "point_image",
    timeout_seconds: float = 600.0,
) -> MolmoPointCallable:
    """Build a synchronous callable for one MolmoPoint stdio MCP call."""

    def point_images(request: JsonDict) -> JsonDict:
        return asyncio.run(
            _call_stdio_mcp_tool(
                command=command,
                args=list(args or ()),
                cwd=str(cwd) if cwd is not None else None,
                tool_name=tool_name,
                arguments=request,
                timeout_seconds=timeout_seconds,
                invalid_payload=_invalid_mcp_payload(
                    tool="molmopoint",
                    backend="molmopoint_mcp",
                    model="unknown",
                    content="MolmoPoint image pointing failed: invalid MCP response.",
                ),
            )
        )

    return point_images


def build_sse_molmopoint_mcp_pointer(
    *,
    url: str,
    tool_name: str = "point_image",
    timeout_seconds: float = 600.0,
) -> MolmoPointCallable:
    """Build a synchronous MolmoPoint callable for an SSE MCP server."""

    transport = SseSimulatorMcpTransport(url)

    def point_images(request: JsonDict) -> JsonDict:
        return transport.call_tool(tool_name, request, timeout_s=timeout_seconds)

    return point_images


def build_stdio_anyplace_mcp_placer(
    *,
    command: str,
    args: Sequence[str] | None = None,
    cwd: str | Path | None = None,
    tool_name: str = "predict_placement",
    timeout_seconds: float = 600.0,
) -> AnyPlacePredictCallable:
    """Build a synchronous callable that invokes one AnyPlace stdio MCP call."""

    def predict_placement(request: JsonDict) -> JsonDict:
        return asyncio.run(
            _call_stdio_mcp_tool(
                command=command,
                args=list(args or ()),
                cwd=str(cwd) if cwd is not None else None,
                tool_name=tool_name,
                arguments=request,
                timeout_seconds=timeout_seconds,
                invalid_payload=_invalid_mcp_payload(
                    tool="anyplace",
                    backend="anyplace_mcp",
                    model="anyplace_multitask",
                    content="AnyPlace placement prediction failed: invalid MCP response.",
                ),
            )
        )

    return predict_placement


def build_sse_anyplace_mcp_placer(
    *,
    url: str,
    tool_name: str = "predict_placement",
    timeout_seconds: float = 600.0,
) -> AnyPlacePredictCallable:
    """Build a synchronous AnyPlace callable for an SSE MCP server."""

    transport = SseSimulatorMcpTransport(url)

    def predict_placement(request: JsonDict) -> JsonDict:
        return transport.call_tool(tool_name, request, timeout_s=timeout_seconds)

    return predict_placement


def _dummy_state_delta(context: ToolExecutionContext) -> JsonDict:
    if context.name == "gripper_control":
        return {"gripper_state": {"position": context.parameters.get("position")}}
    if context.name == "follow_eef_trajectory":
        return {"eef_trajectory_executed": context.parameters.get("trajectory", [])}
    if context.name == "lower_body_control_policy":
        return {"base_command": context.parameters.get("command")}
    return {"eef_pose": context.parameters.get("target_pose")}


async def _call_stdio_mcp_tool(
    *,
    command: str,
    args: list[str],
    cwd: str | None,
    tool_name: str,
    arguments: JsonDict,
    timeout_seconds: float,
    invalid_payload: JsonDict | None = None,
) -> JsonDict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=command, args=args, cwd=cwd)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                tool_name,
                arguments,
                read_timeout_seconds=timedelta(seconds=timeout_seconds),
            )
    payload = _parse_mcp_tool_result(result, invalid_payload=invalid_payload)
    if result.isError and payload.get("success", False):
        payload["success"] = False
    return payload


def _parse_mcp_tool_result(result: Any, *, invalid_payload: JsonDict | None = None) -> JsonDict:
    if isinstance(result, Mapping):
        return dict(result)

    for attr in ("structuredContent", "structured_content"):
        structured = getattr(result, attr, None)
        if isinstance(structured, Mapping):
            return dict(structured)

    if hasattr(result, "model_dump"):
        dumped = result.model_dump()
        if isinstance(dumped, Mapping):
            for key in ("structuredContent", "structured_content"):
                structured = dumped.get(key)
                if isinstance(structured, Mapping):
                    return dict(structured)
            parsed = _parse_mcp_content_items(dumped.get("content", []))
            if parsed is not None:
                return parsed

    parsed = _parse_mcp_content_items(getattr(result, "content", []) or [])
    if parsed is not None:
        return parsed

    text = str(result)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        return payload

    return invalid_payload or _invalid_mcp_payload(
        tool="sam3",
        backend="sam3_mcp",
        model="sam3",
        content="SAM3 segmentation failed: invalid MCP response.",
        extra={
            "prompt": "",
            "source_image": "",
            "raw_output_ref": None,
            "detection_count": 0,
            "detections": [],
        },
    )


def _parse_mcp_content_items(items: Any) -> JsonDict | None:
    if not isinstance(items, (list, tuple)):
        return None
    for item in items:
        if isinstance(item, Mapping):
            if isinstance(item.get("json"), Mapping):
                return dict(item["json"])
            if isinstance(item.get("data"), Mapping):
                return dict(item["data"])
            text = item.get("text", "")
        else:
            if isinstance(getattr(item, "json", None), Mapping):
                return dict(getattr(item, "json"))
            if isinstance(getattr(item, "data", None), Mapping):
                return dict(getattr(item, "data"))
            text = getattr(item, "text", "")
        if isinstance(text, Mapping):
            return dict(text)
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


class _Sam3ToolInputError(ValueError):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _normalise_sam3_tool_points(
    value: Any,
    *,
    image_path: Path,
) -> list[JsonDict]:
    if value is None or value == []:
        raise _Sam3ToolInputError("missing_points", "at least one point is required")
    if not isinstance(value, list) or len(value) > SAM3_MAX_POINT_COUNT:
        raise _Sam3ToolInputError(
            "invalid_points",
            f"points must contain between 1 and {SAM3_MAX_POINT_COUNT} items",
        )
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
    except Exception as exc:  # noqa: BLE001 - image diagnostics stay structured.
        raise _Sam3ToolInputError("image_encode_failed", "image decode failed") from exc

    points: list[JsonDict] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"x", "y", "label"}:
            raise _Sam3ToolInputError(
                "invalid_points",
                "each point must contain exactly x, y, and label",
            )
        x = _finite_float(item.get("x"))
        y = _finite_float(item.get("y"))
        label = item.get("label")
        if (
            isinstance(item.get("x"), bool)
            or isinstance(item.get("y"), bool)
            or x is None
            or y is None
            or isinstance(label, bool)
            or not isinstance(label, int)
            or label not in {0, 1}
        ):
            raise _Sam3ToolInputError(
                "invalid_points",
                "point coordinates must be finite and label must be 0 or 1",
            )
        if not 0 <= x < width or not 0 <= y < height:
            raise _Sam3ToolInputError(
                "point_out_of_bounds",
                "point is outside the source image",
            )
        points.append({"x": x, "y": y, "label": label})
    if not any(point["label"] == 1 for point in points):
        raise _Sam3ToolInputError(
            "invalid_points",
            "at least one foreground point with label=1 is required",
        )
    return points


def _is_consistent_sam3_point_response(
    response: Any,
    *,
    points: list[JsonDict],
    source_image: Path,
) -> bool:
    if not isinstance(response, Mapping):
        return True
    success = response.get("success")
    if not isinstance(success, bool):
        return False
    if success is False:
        return True
    details = response.get("details")
    if not isinstance(details, Mapping):
        return False
    if (
        details.get("tool") != "sam3"
        or details.get("backend") != "sam3_mcp"
        or details.get("model") != "sam3"
        or details.get("prompt_type") != "points"
        or not _sam3_point_echo_matches(details.get("points"), points)
        or details.get("ranking") != "score_descending"
        or details.get("detection_count") != 3
    ):
        return False
    try:
        from PIL import Image

        with Image.open(source_image) as source:
            image_size = source.size
    except Exception:  # noqa: BLE001 - invalid source invalidates trusted output.
        return False

    detections = details.get("detections")
    if not isinstance(detections, list) or len(detections) != 3:
        return False
    scores: list[float] = []
    backend_indices: set[int] = set()
    for rank, detection in enumerate(detections):
        if not isinstance(detection, Mapping):
            return False
        score = _finite_float(detection.get("score"))
        backend_index = detection.get("backend_index")
        if (
            detection.get("label") != "point_prompt"
            or isinstance(detection.get("rank"), bool)
            or detection.get("rank") != rank
            or isinstance(detection.get("score"), bool)
            or score is None
            or isinstance(backend_index, bool)
            or not isinstance(backend_index, int)
            or backend_index not in {0, 1, 2}
        ):
            return False
        mask = _decode_valid_sam3_point_image(detection.get("mask"), image_size=image_size)
        if mask is None:
            return False
        mask_values = set(mask.getdata())
        if not mask_values or not mask_values.issubset({0, 255}) or 255 not in mask_values:
            return False
        area_px = detection.get("area_px")
        expected_area = sum(value == 255 for value in mask.getdata())
        expected_bbox = list(mask.getbbox() or ())
        bbox = detection.get("bbox_xyxy")
        if (
            isinstance(area_px, bool)
            or not isinstance(area_px, int)
            or area_px != expected_area
            or not isinstance(bbox, list)
            or len(bbox) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in bbox)
            or bbox != expected_bbox
        ):
            return False
        scores.append(score)
        backend_indices.add(backend_index)
    if backend_indices != {0, 1, 2} or scores != sorted(scores, reverse=True):
        return False

    artifacts = details.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 3:
        return False
    artifact_bindings: set[tuple[int, int]] = set()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            return False
        rank = artifact.get("rank")
        backend_index = artifact.get("backend_index")
        if (
            artifact.get("artifact_type") != "candidate_overlay"
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or isinstance(backend_index, bool)
            or not isinstance(backend_index, int)
            or not 0 <= rank < 3
            or not 0 <= backend_index < 3
            or detections[rank].get("backend_index") != backend_index
            or _decode_valid_sam3_point_image(artifact, image_size=image_size) is None
        ):
            return False
        artifact_bindings.add((rank, backend_index))
    if len(artifact_bindings) != 3:
        return False

    metadata = details.get("metadata")
    foreground_count = sum(point["label"] == 1 for point in points)
    count_keys = (
        "point_count",
        "foreground_point_count",
        "background_point_count",
        "candidate_count",
    )
    if (
        not isinstance(metadata, Mapping)
        or any(
            isinstance(metadata.get(key), bool)
            or not isinstance(metadata.get(key), int)
            for key in count_keys
        )
        or any(
            (
                metadata.get("prompt_type") != "points",
                metadata.get("coordinate_units") != "pixels",
                metadata.get("coordinate_origin") != "top_left",
                metadata.get("multimask_output") is not True,
                metadata.get("point_count") != len(points),
                metadata.get("foreground_point_count") != foreground_count,
                metadata.get("background_point_count") != len(points) - foreground_count,
                metadata.get("candidate_count") != 3,
                metadata.get("image_size") != list(image_size),
            )
        )
    ):
        return False
    return True


def _sam3_point_echo_matches(value: Any, expected: list[JsonDict]) -> bool:
    if not isinstance(value, list) or len(value) != len(expected):
        return False
    for returned, requested in zip(value, expected):
        if (
            not isinstance(returned, Mapping)
            or set(returned) != {"x", "y", "label"}
            or type(returned.get("x")) is not float
            or type(returned.get("y")) is not float
            or type(returned.get("label")) is not int
            or returned != requested
        ):
            return False
    return True


def _decode_valid_sam3_point_image(
    value: Any,
    *,
    image_size: tuple[int, int],
) -> Any | None:
    if (
        not isinstance(value, Mapping)
        or value.get("format") != "png"
        or not isinstance(value.get("base64"), str)
    ):
        return None
    try:
        from PIL import Image

        raw = base64.b64decode(value["base64"], validate=True)
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception:  # noqa: BLE001 - untrusted image payloads fail closed.
        return None
    if image.format != "PNG" or image.size != image_size:
        return None
    return image.convert("L")


def _normalise_sam3_response(
    response: JsonDict,
    *,
    mode: str,
    prompt: str,
    points: list[JsonDict],
    source_image: str,
    request: JsonDict,
    run_dir: Path,
    artifacts_dir: Path,
    roi_bbox_xyxy: object = None,
    extra_artifacts: Sequence[JsonDict] = (),
    output_metadata: JsonDict | None = None,
) -> ToolResult:
    if not isinstance(response, dict):
        return _sam3_failure(
            mode=mode,
            prompt=prompt,
            points=points,
            source_image=source_image,
            reason="mcp_call_failed",
            content="SAM3 segmentation failed: invalid MCP response.",
        )
    details = response.get("details")
    success = response.get("success")
    if not isinstance(success, bool):
        return _sam3_failure(
            mode=mode,
            prompt=prompt,
            points=points,
            source_image=source_image,
            reason="inconsistent_detection_outputs",
            content="SAM3 returned inconsistent detection outputs.",
        )
    if success and not isinstance(details, dict):
        return _sam3_failure(
            mode=mode,
            prompt=prompt,
            points=points,
            source_image=source_image,
            reason="inconsistent_detection_outputs",
            content="SAM3 returned inconsistent detection outputs.",
        )
    if not isinstance(details, dict):
        details = {}
    if not success:
        reason = _string_param(details.get("reason")) or "unknown_error"
        content = _string_param(response.get("content")) or f"SAM3 segmentation failed: {reason}."
        return _sam3_failure(
            mode=mode,
            prompt=prompt,
            points=points,
            source_image=source_image,
            reason=reason,
            content=content,
            raw_output_ref=details.get("raw_output_ref"),
            metadata=_dict_or_empty(details.get("metadata")),
        )

    if "detections" not in details or "detection_count" not in details:
        return _sam3_failure(
            mode=mode,
            prompt=prompt,
            points=points,
            source_image=source_image,
            reason="inconsistent_detection_outputs",
            content="SAM3 returned inconsistent detection outputs.",
            raw_output_ref=details.get("raw_output_ref"),
            metadata=_dict_or_empty(details.get("metadata")),
        )

    detections_value = details.get("detections")
    if not isinstance(detections_value, list):
        return _sam3_failure(
            mode=mode,
            prompt=prompt,
            points=points,
            source_image=source_image,
            reason="inconsistent_detection_outputs",
            content="SAM3 returned inconsistent detection outputs.",
            raw_output_ref=details.get("raw_output_ref"),
            metadata=_dict_or_empty(details.get("metadata")),
        )
    _write_json(run_dir / "request.json", request)

    ranked_detections = sorted(
        enumerate(detections_value),
        key=_sam3_response_detection_sort_key,
    )
    detections: list[JsonDict] = []
    mask_artifacts: list[JsonDict] = []
    for rank, (response_index, detection) in enumerate(ranked_detections):
        if not isinstance(detection, dict):
            return _sam3_failure(
                mode=mode,
                prompt=prompt,
                points=points,
                source_image=source_image,
                reason="inconsistent_detection_outputs",
                content="SAM3 returned inconsistent detection outputs.",
                raw_output_ref=details.get("raw_output_ref"),
                metadata=_dict_or_empty(details.get("metadata")),
            )
        mask = detection.get("mask")
        if not isinstance(mask, dict) or not mask.get("base64"):
            return _sam3_failure(
                mode=mode,
                prompt=prompt,
                points=points,
                source_image=source_image,
                reason="inconsistent_detection_outputs",
                content="SAM3 returned inconsistent detection outputs.",
                metadata=_dict_or_empty(details.get("metadata")),
            )
        try:
            if roi_bbox_xyxy is None:
                mask_ref = _write_base64_artifact(
                    mask["base64"],
                    artifacts_dir / f"mask_{rank:03d}.{_safe_extension(mask.get('format'))}",
                )
                mask_area_px = detection.get("area_px")
                mask_bbox_xyxy = detection.get("bbox_xyxy")
            else:
                mask_ref, mask_area_px, mask_bbox_xyxy = _write_sam3_mask_artifact(
                    mask["base64"],
                    artifacts_dir / f"mask_{rank:03d}.png",
                    source_image=Path(source_image),
                    roi_bbox_xyxy=roi_bbox_xyxy,
                )
        except Exception as exc:  # noqa: BLE001
            return _sam3_failure(
                mode=mode,
                prompt=prompt,
                points=points,
                source_image=source_image,
                reason="artifact_write_failed",
                content=f"SAM3 segmentation failed: artifact write failed: {exc}",
                metadata={"error_type": type(exc).__name__},
            )
        label = _string_param(detection.get("label")) or prompt
        score = _finite_float(detection.get("score"))
        bbox_xyxy = mask_bbox_xyxy
        area_px = mask_area_px
        detection_id = f"detection_{rank:03d}"
        backend_index = _sam3_backend_index(detection, fallback=response_index)
        detections.append(
            {
                "id": detection_id,
                "label": label,
                "score": score,
                "rank": rank,
                "backend_index": backend_index,
                "bbox_xyxy": bbox_xyxy,
                "mask_ref": str(mask_ref),
                "area_px": area_px,
            }
        )
        mask_artifacts.append(
            {
                "type": "segmentation_mask",
                "kind": "mask",
                "tool": "sam3",
                "index": detection_id,
                "label": label,
                "mode": mode,
                "prompt": prompt,
                "points": [dict(point) for point in points],
                "path": str(mask_ref),
                "mask_ref": str(mask_ref),
                "source_image": source_image,
                "score": score,
                "bbox_xyxy": bbox_xyxy,
                "area_px": area_px,
            }
        )

    declared_count = details.get("detection_count")
    try:
        parsed_count = int(declared_count)
    except (TypeError, ValueError):
        parsed_count = -1
    if parsed_count != len(detections):
        return _sam3_failure(
            mode=mode,
            prompt=prompt,
            points=points,
            source_image=source_image,
            reason="inconsistent_detection_outputs",
            content="SAM3 returned inconsistent detection outputs.",
            raw_output_ref=details.get("raw_output_ref"),
            metadata=_dict_or_empty(details.get("metadata")),
        )

    artifacts_value = details.get("artifacts", [])
    if not isinstance(artifacts_value, list):
        artifacts_value = []
    artifacts: list[JsonDict] = [dict(artifact) for artifact in extra_artifacts]
    for idx, artifact in enumerate(artifacts_value):
        if not isinstance(artifact, dict) or not artifact.get("base64"):
            continue
        artifact_type = _string_param(artifact.get("artifact_type")) or f"artifact_{idx:03d}"
        fmt = _safe_extension(artifact.get("format"))
        suffix = artifact.get("rank")
        if not isinstance(suffix, int) or isinstance(suffix, bool):
            suffix = idx
        try:
            artifact_ref = _write_base64_artifact(
                artifact["base64"],
                artifacts_dir / f"{artifact_type}_{suffix:03d}.{fmt}",
            )
        except Exception as exc:  # noqa: BLE001
            return _sam3_failure(
                mode=mode,
                prompt=prompt,
                points=points,
                source_image=source_image,
                reason="artifact_write_failed",
                content=f"SAM3 segmentation failed: artifact write failed: {exc}",
                metadata={"error_type": type(exc).__name__},
            )
        materialized = dict(artifact)
        materialized.pop("base64", None)
        materialized["artifact_ref"] = str(artifact_ref)
        artifacts.append(materialized)
    artifacts.extend(mask_artifacts)

    selection_bundle: JsonDict = {}
    visualization_diagnostics: list[JsonDict] = []
    if detections:
        try:
            selection_bundle, selection_artifacts = _build_sam3_selection_artifacts(
                source_image=Path(source_image),
                detections=detections,
                output_dir=artifacts_dir,
                prompt=prompt or "point_prompt",
            )
            artifacts.extend(selection_artifacts)
        except Exception as exc:  # noqa: BLE001 - required selection evidence failed.
            return _sam3_failure(
                mode=mode,
                prompt=prompt,
                points=points,
                source_image=source_image,
                reason="artifact_write_failed",
                content=(
                    "SAM3 segmentation failed: selection visualization "
                    f"failed: {exc}"
                ),
                metadata={"error_type": type(exc).__name__},
            )

    content = _string_param(response.get("content"))
    if not content:
        content = (
            "SAM3 segmentation completed."
            if detections
            else "SAM3 segmentation completed with no detections."
        )
    result = ToolResult(
        True,
        content=content,
        details={
            "tool": "sam3",
            "backend": _string_param(details.get("backend")) or "sam3_mcp",
            "model": _string_param(details.get("model")) or "sam3",
            "mode": mode,
            "prompt_type": _string_param(details.get("prompt_type")) or mode,
            "prompt": prompt,
            "points": [dict(point) for point in points],
            "source_image": source_image,
            "raw_output_ref": None,
            "result_id": artifacts_dir.name,
            "detection_count": len(detections),
            "detections": detections,
            "ranking": "score_descending",
            "selection_required": len(detections) > 1,
            "selected_detection": detections[0] if len(detections) == 1 else None,
            "selection_bundle": selection_bundle,
            "artifacts": artifacts,
            "diagnostics": visualization_diagnostics,
            "metadata": _dict_or_empty(details.get("metadata")),
            **dict(output_metadata or {}),
        },
    )
    return result


def _prepare_sam3_roi_attention_image(
    *,
    source_image: Path,
    roi_bbox_xyxy: object,
    output_root: Path,
) -> tuple[str, str, JsonDict, JsonDict]:
    from PIL import Image

    if not source_image.is_file():
        raise FileNotFoundError(source_image)
    with Image.open(source_image) as loaded:
        original = loaded.convert("RGB")
    requested_bbox = _validate_sam3_roi_bbox(roi_bbox_xyxy, image_size=original.size)
    effective_bbox = _pad_sam3_roi_bbox(
        requested_bbox,
        image_size=original.size,
        padding_ratio=DEFAULT_SAM3_ROI_PADDING_RATIO,
    )
    attention = Image.new("RGB", original.size, color=(0, 0, 0))
    attention.paste(original.crop(effective_bbox), effective_bbox[:2])
    roi_dir = output_root / f"roi-{uuid4().hex[:12]}"
    roi_dir.mkdir(parents=True, exist_ok=False)
    attention_ref = roi_dir / "attention.png"
    attention.save(attention_ref, format="PNG")
    requested = list(requested_bbox)
    effective = list(effective_bbox)
    metadata = {
        "requested_roi_bbox_xyxy": requested,
        "effective_roi_bbox_xyxy": effective,
        "roi_padding_ratio": DEFAULT_SAM3_ROI_PADDING_RATIO,
        "roi_attention_image_ref": str(attention_ref),
        "source_image_size": [original.width, original.height],
    }
    artifact = {
        "type": "sam3_roi_attention_image",
        "kind": "rgb",
        "tool": "sam3",
        "path": str(attention_ref),
        "source_image": str(source_image),
        "requested_roi_bbox_xyxy": requested,
        "effective_roi_bbox_xyxy": effective,
        "width": original.width,
        "height": original.height,
    }
    encoded = base64.b64encode(attention_ref.read_bytes()).decode("ascii")
    return encoded, "png", metadata, artifact


def _validate_sam3_roi_bbox(
    value: object,
    *,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("roi_bbox_xyxy must contain [left, top, right, bottom]")
    coordinates: list[int] = []
    for coordinate in value:
        number = _finite_float(coordinate)
        if number is None:
            raise ValueError("roi_bbox_xyxy coordinates must be finite numbers")
        coordinates.append(int(round(number)))
    left, top, right, bottom = coordinates
    width, height = image_size
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("roi_bbox_xyxy must be non-empty and inside the original image bounds")
    return left, top, right, bottom


def _sam3_image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def _validate_sam3_prompt_points(
    value: object,
    *,
    image_size: tuple[int, int],
) -> list[JsonDict]:
    if not isinstance(value, list) or not 1 <= len(value) <= 64:
        raise ValueError("positive_points must contain between one and 64 points")
    width, height = image_size
    points: list[JsonDict] = []
    foreground_count = 0
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"positive_points[{index}] must be an object")
        x = _finite_float(item.get("x"))
        y = _finite_float(item.get("y"))
        if x is None or y is None:
            raise ValueError(f"positive_points[{index}] requires finite x/y pixels")
        label = item.get("label", 1)
        if isinstance(label, bool):
            raise ValueError(f"positive_points[{index}].label must be 0 or 1")
        try:
            parsed_label = int(label)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"positive_points[{index}].label must be 0 or 1") from exc
        if parsed_label not in {0, 1}:
            raise ValueError(f"positive_points[{index}].label must be 0 or 1")
        if not 0 <= x < width or not 0 <= y < height:
            raise ValueError(f"positive_points[{index}] is outside the source image")
        foreground_count += parsed_label
        points.append({"x": x, "y": y, "label": parsed_label})
    if foreground_count == 0:
        raise ValueError("positive_points requires at least one foreground point")
    return points


def _pad_sam3_roi_bbox(
    bbox: tuple[int, int, int, int],
    *,
    image_size: tuple[int, int],
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width, height = image_size
    pad_x = max(2, int(round((right - left) * padding_ratio)))
    pad_y = max(2, int(round((bottom - top) * padding_ratio)))
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )


def _sam3_response_has_no_detections(response: object) -> bool:
    if not isinstance(response, dict) or not bool(response.get("success")):
        return False
    details = response.get("details")
    if not isinstance(details, dict):
        return False
    detections = details.get("detections")
    return isinstance(detections, list) and not detections


def _write_sam3_mask_artifact(
    encoded: str,
    path: Path,
    *,
    source_image: Path,
    roi_bbox_xyxy: object,
) -> tuple[Path, int, list[int] | None]:
    from PIL import Image

    raw = base64.b64decode(encoded, validate=True)
    with Image.open(BytesIO(raw)) as loaded_mask:
        mask = loaded_mask.convert("L")
    with Image.open(source_image) as loaded_source:
        source_size = loaded_source.size
    if mask.size != source_size:
        raise ValueError(
            f"mask size {mask.size!r} does not match source image size {source_size!r}"
        )
    if roi_bbox_xyxy is not None:
        bbox = _validate_sam3_roi_bbox(roi_bbox_xyxy, image_size=source_size)
        clamped = Image.new("L", source_size, color=0)
        clamped.paste(mask.crop(bbox), bbox[:2])
        mask = clamped
    binary = mask.point(lambda pixel: 255 if pixel else 0, mode="L")
    path.parent.mkdir(parents=True, exist_ok=True)
    binary.save(path, format="PNG")
    histogram = binary.histogram()
    area_px = sum(histogram[1:])
    mask_bbox = binary.getbbox()
    return path, area_px, list(mask_bbox) if mask_bbox is not None else None


def _sam3_response_detection_sort_key(
    item: tuple[int, object],
) -> tuple[int, float, int]:
    response_index, detection = item
    if not isinstance(detection, dict):
        return 1, 0.0, response_index
    score = _finite_float(detection.get("score"))
    if score is None:
        return 1, 0.0, response_index
    return 0, -score, response_index


def _sam3_backend_index(detection: JsonDict, *, fallback: int) -> int:
    value = detection.get("backend_index")
    if isinstance(value, bool):
        return fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed >= 0 else fallback


def _build_sam3_selection_artifacts(
    *,
    source_image: Path,
    detections: list[JsonDict],
    output_dir: Path,
    prompt: str,
    visual_limit: int = DEFAULT_SAM3_SELECTION_VISUAL_LIMIT,
) -> tuple[JsonDict, list[JsonDict]]:
    from PIL import Image, ImageDraw, ImageEnhance, ImageOps

    original = Image.open(source_image).convert("RGB")
    visualized = detections[: max(1, visual_limit)]
    artifacts: list[JsonDict] = []
    panels: list[tuple[str, Any]] = [("original", original.copy())]
    bundle_candidates: list[JsonDict] = []
    colors = (
        (0, 220, 255, 190),
        (255, 80, 120, 190),
        (255, 210, 0, 190),
        (90, 235, 120, 190),
        (180, 110, 255, 190),
        (255, 145, 45, 190),
        (70, 145, 255, 190),
        (235, 90, 220, 190),
    )

    for detection in visualized:
        detection_id = str(detection.get("id") or "detection")
        rank = int(detection.get("rank") or 0)
        mask = Image.open(str(detection["mask_ref"])).convert("L")
        if mask.size != original.size:
            mask = mask.resize(original.size, resample=Image.Resampling.NEAREST)
        dimmed = ImageEnhance.Brightness(original).enhance(0.45).convert("RGBA")
        base = original.convert("RGBA")
        tint = Image.new("RGBA", original.size, colors[rank % len(colors)])
        highlighted = Image.blend(base, tint, 0.45)
        overlay = Image.composite(highlighted, dimmed, mask).convert("RGB")
        draw = ImageDraw.Draw(overlay)
        bbox = _sam3_visual_bbox(detection.get("bbox_xyxy"), mask=mask)
        if bbox is not None:
            draw.rectangle(bbox, outline=colors[rank % len(colors)][:3], width=4)
        score = detection.get("score")
        score_text = f" score={score:.3f}" if isinstance(score, float) else ""
        label = f"{detection_id}{score_text}"
        draw.rectangle((0, 0, min(overlay.width, 280), 24), fill=(0, 0, 0))
        draw.text((6, 5), label, fill=(255, 255, 255))

        overlay_ref = output_dir / f"{detection_id}.overlay.png"
        overlay.save(overlay_ref, format="PNG")
        crop_box = _sam3_padded_crop_box(bbox, image_size=original.size)
        crop = overlay.crop(crop_box) if crop_box is not None else overlay.copy()
        crop_ref = output_dir / f"{detection_id}.crop.png"
        crop.save(crop_ref, format="PNG")
        detection["overlay_ref"] = str(overlay_ref)
        detection["crop_ref"] = str(crop_ref)

        artifacts.extend(
            [
                {
                    "type": "sam3_candidate_overlay",
                    "kind": "image",
                    "tool": "sam3",
                    "index": detection_id,
                    "path": str(overlay_ref),
                },
                {
                    "type": "sam3_candidate_crop",
                    "kind": "image",
                    "tool": "sam3",
                    "index": detection_id,
                    "path": str(crop_ref),
                },
            ]
        )
        candidate_panel = Image.new("RGB", (360, 430), "white")
        full_panel = ImageOps.contain(overlay, (340, 270))
        crop_panel = ImageOps.contain(crop, (340, 120))
        candidate_panel.paste(full_panel, ((360 - full_panel.width) // 2, 30))
        candidate_panel.paste(crop_panel, ((360 - crop_panel.width) // 2, 300))
        panel_draw = ImageDraw.Draw(candidate_panel)
        panel_draw.text((10, 8), label, fill=(0, 0, 0))
        panels.append((detection_id, candidate_panel))
        bundle_candidates.append(
            {
                key: detection.get(key)
                for key in (
                    "id",
                    "label",
                    "score",
                    "rank",
                    "backend_index",
                    "bbox_xyxy",
                    "area_px",
                    "mask_ref",
                    "overlay_ref",
                    "crop_ref",
                )
            }
        )

    original_panel = Image.new("RGB", (360, 430), "white")
    original_thumb = ImageOps.contain(original, (340, 390))
    original_panel.paste(original_thumb, ((360 - original_thumb.width) // 2, 30))
    ImageDraw.Draw(original_panel).text((10, 8), f"original: {prompt}", fill=(0, 0, 0))
    panels[0] = ("original", original_panel)
    columns = min(3, len(panels))
    rows = math.ceil(len(panels) / columns)
    sheet = Image.new("RGB", (columns * 360, rows * 430), (238, 238, 238))
    for index, (_name, panel) in enumerate(panels):
        sheet.paste(panel, ((index % columns) * 360, (index // columns) * 430))
    contact_sheet_ref = output_dir / "selection.contact_sheet.png"
    sheet.save(contact_sheet_ref, format="PNG")
    artifacts.append(
        {
            "type": "sam3_selection_contact_sheet",
            "kind": "image",
            "tool": "sam3",
            "index": "selection",
            "path": str(contact_sheet_ref),
        }
    )
    return (
        {
            "target_prompt": prompt,
            "original_image_ref": str(source_image),
            "contact_sheet_ref": str(contact_sheet_ref),
            "candidate_count": len(detections),
            "visualized_candidate_count": len(visualized),
            "visuals_truncated": len(visualized) < len(detections),
            "candidates": bundle_candidates,
        },
        artifacts,
    )


def _sam3_visual_bbox(value: object, *, mask: Any) -> tuple[int, int, int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            coords = tuple(int(round(float(item))) for item in value)
        except (TypeError, ValueError):
            coords = ()
        if len(coords) == 4:
            left, top, right, bottom = coords
            left = min(max(0, left), mask.width - 1)
            top = min(max(0, top), mask.height - 1)
            right = min(max(left + 1, right), mask.width)
            bottom = min(max(top + 1, bottom), mask.height)
            return left, top, right, bottom
    bbox = mask.getbbox()
    return tuple(int(item) for item in bbox) if bbox is not None else None


def _sam3_padded_crop_box(
    bbox: tuple[int, int, int, int] | None,
    *,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    padding = max(8, int(max(right - left, bottom - top) * 0.2))
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(image_size[0], right + padding),
        min(image_size[1], bottom + padding),
    )


def _sam3_failure(
    *,
    mode: str = "text",
    prompt: str,
    points: list[JsonDict] | None = None,
    source_image: str,
    reason: str,
    content: str,
    raw_output_ref: Any = None,
    metadata: JsonDict | None = None,
) -> ToolResult:
    return ToolResult(
        False,
        content=content,
        details={
            "tool": "sam3",
            "backend": "sam3_mcp",
            "model": "sam3",
            "mode": mode,
            "prompt_type": mode,
            "prompt": prompt,
            "points": [dict(point) for point in points or []],
            "source_image": source_image,
            "raw_output_ref": raw_output_ref,
            "detection_count": 0,
            "detections": [],
            "artifacts": [],
            "reason": reason,
            "metadata": dict(metadata or {}),
        },
    )


def _normalise_anygrasp_response(
    response: JsonDict,
    *,
    mode: str,
    source_rgb: str,
    source_depth: str,
    target_mask: str,
    request: JsonDict,
    output_root: Path,
) -> ToolResult:
    if not isinstance(response, dict):
        return _anygrasp_failure(
            mode=mode,
            source_rgb=source_rgb,
            source_depth=source_depth,
            target_mask=target_mask,
            reason="mcp_call_failed",
            content="AnyGrasp grasp detection failed: invalid MCP response.",
        )
    details = response.get("details")
    success = bool(response.get("success", False))
    if success and not isinstance(details, dict):
        return _anygrasp_failure(
            mode=mode,
            source_rgb=source_rgb,
            source_depth=source_depth,
            target_mask=target_mask,
            reason="inconsistent_grasp_outputs",
            content="AnyGrasp returned inconsistent grasp outputs.",
        )
    if not isinstance(details, dict):
        details = {}
    if not success:
        reason = _string_param(details.get("reason")) or "unknown_error"
        content = (
            _string_param(response.get("content")) or f"AnyGrasp grasp detection failed: {reason}."
        )
        return _anygrasp_failure(
            mode=mode,
            source_rgb=source_rgb,
            source_depth=source_depth,
            target_mask=target_mask,
            reason=reason,
            content=content,
            raw_output_ref=details.get("raw_output_ref"),
            metadata=_dict_or_empty(details.get("metadata")),
        )

    candidates_value = details.get("grasp_candidates")
    candidate_count = details.get("candidate_count")
    if not isinstance(candidates_value, list):
        return _anygrasp_inconsistent(
            mode=mode,
            source_rgb=source_rgb,
            source_depth=source_depth,
            target_mask=target_mask,
            metadata=_dict_or_empty(details.get("metadata")),
        )
    try:
        parsed_count = int(candidate_count)
    except (TypeError, ValueError):
        parsed_count = -1
    if parsed_count != len(candidates_value) or parsed_count <= 0:
        return _anygrasp_inconsistent(
            mode=mode,
            source_rgb=source_rgb,
            source_depth=source_depth,
            target_mask=target_mask,
            metadata=_dict_or_empty(details.get("metadata")),
        )

    candidates: list[JsonDict] = []
    depth_cutoff_factor = float(request.get("depth_cutoff_factor") or 1.0)
    for backend_index, candidate in enumerate(candidates_value):
        normalized = _normalise_anygrasp_candidate(candidate)
        if normalized is None:
            return _anygrasp_inconsistent(
                mode=mode,
                source_rgb=source_rgb,
                source_depth=source_depth,
                target_mask=target_mask,
                metadata=_dict_or_empty(details.get("metadata")),
            )
        normalized["backend_index"] = _nonnegative_int(
            candidate.get("backend_index"),
            default=backend_index,
        )
        normalized["depth_cutoff_factor"] = depth_cutoff_factor
        normalized["length_scale_restored"] = True
        candidates.append(normalized)

    candidates.sort(key=lambda candidate: -float(candidate["score"]))
    for rank, candidate in enumerate(candidates):
        candidate["rank"] = rank
        candidate["id"] = f"grasp_{rank:03d}"

    source: JsonDict | None = None
    if mode == "targeted":
        normalized_intrinsics = _normalise_camera_intrinsics(request.get("intrinsics"))
        if normalized_intrinsics is None or not target_mask:
            return _anygrasp_inconsistent(
                mode=mode,
                source_rgb=source_rgb,
                source_depth=source_depth,
                target_mask=target_mask,
                metadata=_dict_or_empty(details.get("metadata")),
            )
        source = {
            "source_tool": "anygrasp",
            "mode": "targeted",
            "rgb": source_rgb,
            "depth": source_depth,
            "object_mask": target_mask,
            "intrinsics": normalized_intrinsics,
            "depth_cutoff_factor": depth_cutoff_factor,
        }

    run_dir = _new_run_dir(output_root)
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        run_dir = _new_run_dir(output_root)
        run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "request.json", request)
    raw_output_ref = run_dir / "response.raw.json"
    artifacts = _scrub_anygrasp_artifacts(details.get("artifacts"), mark_omitted=False)
    _write_json(raw_output_ref, _scrub_anygrasp_response(response))
    content = _string_param(response.get("content")) or "AnyGrasp grasp detection completed."
    result_details: JsonDict = {
        "tool": "anygrasp",
        "backend": _string_param(details.get("backend")) or "anygrasp_mcp",
        "model": _string_param(details.get("model")) or "anygrasp_sdk",
        "mode": _string_param(details.get("mode")) or mode,
        "source_rgb": source_rgb,
        "source_depth": source_depth,
        "target_mask": target_mask or None,
        "raw_output_ref": str(raw_output_ref),
        "result_id": run_dir.name,
        "candidate_count": len(candidates),
        "grasp_candidates": candidates,
        "best_grasp_candidate": candidates[0],
        "active_grasp_candidate": candidates[0],
        "ranking": "score_descending",
        "artifacts": artifacts,
        "metadata": _dict_or_empty(details.get("metadata")),
    }
    if source is not None:
        result_details["source"] = source
    result = ToolResult(
        True,
        content=content,
        details=result_details,
    )
    _write_json(
        run_dir / "tool_result.json",
        {"success": result.success, "content": result.content, "details": result.details},
    )
    return result


def _grasp_pose_common_input_error(
    *,
    mode: str,
    rgb: str,
    depth: str,
    object_mask: object,
    intrinsics: object,
    camera_frame_id: str,
    scene_epoch: object,
) -> str:
    if mode not in {"targeted", "scene"}:
        return "invalid_mode"
    if not rgb:
        return "missing_rgb"
    if not depth:
        return "missing_depth"
    if not Path(rgb).expanduser().is_file():
        return "rgb_not_found"
    if not Path(depth).expanduser().is_file():
        return "depth_not_found"
    if _normalise_camera_intrinsics(intrinsics) is None:
        return "invalid_intrinsics"
    if not camera_frame_id:
        return "missing_camera_frame_id"
    if isinstance(scene_epoch, bool) or not isinstance(scene_epoch, int) or scene_epoch < 0:
        return "invalid_scene_epoch"
    if mode == "scene":
        return "object_mask_not_allowed_in_scene_mode" if object_mask is not None else ""
    if not isinstance(object_mask, Mapping):
        return "invalid_object_mask"
    mask_ref = _string_param(object_mask.get("mask_ref"))
    source_image = _string_param(object_mask.get("source_image"))
    if not mask_ref or not source_image:
        return "invalid_object_mask"
    if not Path(mask_ref).expanduser().is_file():
        return "object_mask_not_found"
    if not _same_resolved_path(rgb, source_image):
        return "object_mask_source_mismatch"
    return ""


def _grasp_pose_backend_parameters(
    backend: str,
    *,
    mode: str,
    rgb: str,
    depth: str,
    object_mask: JsonDict | None,
    intrinsics: JsonDict,
    hints: JsonDict,
    graspgenx_gripper_name: str,
    graspgenx_up_direction_camera: Sequence[float],
) -> JsonDict | None:
    enhancement = hints.get("depth_enhancement")
    enhanced_candidate_only = (
        isinstance(enhancement, Mapping)
        and enhancement.get("candidate_generation_only") is True
    )
    if enhanced_candidate_only and backend != "anygrasp":
        return None
    if backend == "anygrasp":
        parameters: JsonDict = {
            "mode": mode,
            "rgb": rgb,
            "depth": depth,
            "intrinsics": intrinsics,
            "collision_detection": (
                False
                if enhanced_candidate_only
                else hints.get("collision_check", True)
            ),
            "dense_grasp": hints.get("dense_sampling", False),
            "depth_cutoff_factor": hints.get("depth_cutoff_factor", 1.0),
        }
        if object_mask is not None:
            parameters["target_mask"] = object_mask["mask_ref"]
        if "approach_direction_camera" in hints:
            parameters["approach_steering"] = hints["approach_direction_camera"]
        if "approach_threshold_rad" in hints:
            parameters["approach_thresh"] = hints["approach_threshold_rad"]
        return parameters
    if mode != "targeted" or object_mask is None:
        return None
    common = {
        "rgb": rgb,
        "depth": depth,
        "object_mask": object_mask,
        "intrinsics": intrinsics,
    }
    if backend == "contact_graspnet":
        return common
    if backend == "graspgenx":
        up_direction = _normalise_graspgenx_up_direction(
            list(graspgenx_up_direction_camera)
        )
        if not graspgenx_gripper_name or up_direction is None:
            return None
        return {
            **common,
            "gripper_name": graspgenx_gripper_name,
            "up_direction_camera": up_direction,
        }
    return None


def _grasp_backend_failure_reason(result: ToolResult) -> str:
    details = result.details if isinstance(result.details, dict) else {}
    outputs = details.get("outputs")
    for source in (details, outputs):
        if not isinstance(source, Mapping):
            continue
        reason = _string_param(source.get("reason"))
        if reason:
            return reason
    return "unknown_error"


def _grasp_backend_candidate_count(result: ToolResult) -> int:
    details = result.details if isinstance(result.details, dict) else {}
    outputs = details.get("outputs")
    source = outputs if isinstance(outputs, Mapping) else details
    try:
        return max(0, int(source.get("candidate_count") or 0))
    except (TypeError, ValueError):
        return 0


def _normalise_grasp_pose_estimate_result(
    result: ToolResult,
    *,
    backend: str,
    attempts: list[JsonDict],
    mode: str,
    rgb: str,
    depth: str,
    object_mask: JsonDict | None,
    intrinsics: JsonDict,
    camera_frame_id: str,
    scene_epoch: int,
    hints: JsonDict,
) -> ToolResult:
    details = result.details if isinstance(result.details, dict) else {}
    outputs = details.get("outputs")
    source_details = outputs if isinstance(outputs, Mapping) else details
    raw_candidates = source_details.get("grasp_candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        return _grasp_pose_estimate_failure(
            "inconsistent_grasp_outputs",
            attempts=attempts,
            retryable=True,
        )
    result_id = f"gpe-{uuid4().hex[:16]}"
    candidates: list[JsonDict] = []
    for backend_index, value in enumerate(raw_candidates):
        if not isinstance(value, Mapping):
            return _grasp_pose_estimate_failure(
                "inconsistent_grasp_outputs",
                attempts=attempts,
                retryable=True,
            )
        candidate = dict(value)
        backend_candidate_id = _string_param(candidate.get("id"))
        if not backend_candidate_id:
            return _grasp_pose_estimate_failure(
                "inconsistent_grasp_outputs",
                attempts=attempts,
                retryable=True,
            )
        candidate.update(
            {
                "id": f"{result_id}-{backend_index:03d}",
                "rank": backend_index,
                "backend_index": _nonnegative_int(
                    candidate.get("backend_index"),
                    default=backend_index,
                ),
                "backend_candidate_id": backend_candidate_id,
                "source_tool": "grasp_pose_estimate",
                "source_backend": backend,
                "score_scope": "backend_local",
            }
        )
        if "depth" not in candidate and "gripper_depth" in candidate:
            candidate["depth"] = candidate["gripper_depth"]
        candidates.append(candidate)
    candidates.sort(key=lambda candidate: -float(candidate.get("score") or 0.0))
    for rank, candidate in enumerate(candidates):
        candidate["rank"] = rank
        candidate["id"] = f"{result_id}-{rank:03d}"

    source: JsonDict = {
        "source_tool": "grasp_pose_estimate",
        "source_backend": backend,
        "mode": mode,
        "rgb": rgb,
        "depth": depth,
        "object_mask": object_mask.get("mask_ref") if object_mask else None,
        "intrinsics": dict(intrinsics),
        "camera_frame_id": camera_frame_id,
        "scene_epoch": scene_epoch,
    }
    enhancement = hints.get("depth_enhancement")
    if isinstance(enhancement, Mapping):
        source["depth_enhancement"] = dict(enhancement)
        source["requires_sensor_safety_check"] = bool(
            enhancement.get("requires_sensor_safety_check", True)
        )
    backend_source = source_details.get("source")
    if isinstance(backend_source, Mapping):
        for key in ("depth_cutoff_factor", "gripper_name", "up_direction_camera"):
            if key in backend_source:
                source[key] = backend_source[key]
    artifacts = [
        dict(value)
        for value in source_details.get("artifacts", [])
        if isinstance(value, Mapping)
    ]
    return ToolResult(
        True,
        content=(
            f"Grasp pose estimation completed with {backend}; "
            f"{len(candidates)} candidates."
        ),
        details={
            "schema_version": GRASP_POSE_ESTIMATE_SCHEMA,
            "tool": "grasp_pose_estimate",
            "selected_backend": backend,
            "mode": mode,
            "frame": "camera",
            "camera_frame": "opencv",
            "grasp_frame": "graspnet",
            "result_id": result_id,
            "source": source,
            "source_rgb": rgb,
            "source_depth": depth,
            "object_mask": source.get("object_mask"),
            "camera_frame_id": camera_frame_id,
            "scene_epoch": scene_epoch,
            "candidate_count": len(candidates),
            "grasp_candidates": candidates,
            "best_grasp_candidate": candidates[0],
            "active_grasp_candidate": candidates[0],
            "ranking": "score_descending_backend_local",
            "backend_attempts": [dict(value) for value in attempts],
            "artifacts": artifacts,
            "diagnostics": [
                {
                    "code": "grasp_backend_fallback",
                    "backend": attempt["backend"],
                    "reason": attempt["reason"],
                }
                for attempt in attempts[:-1]
                if attempt["status"] in {"failed", "unavailable"}
            ],
        },
    )


def _grasp_pose_estimate_failure(
    reason: str,
    *,
    attempts: list[JsonDict],
    retryable: bool,
    content: str = "",
) -> ToolResult:
    return ToolResult(
        False,
        content=content or f"Grasp pose estimation failed: {reason}.",
        details={
            "schema_version": GRASP_POSE_ESTIMATE_SCHEMA,
            "tool": "grasp_pose_estimate",
            "reason": reason,
            "retryable": retryable,
            "candidate_count": 0,
            "grasp_candidates": [],
            "backend_attempts": [dict(value) for value in attempts],
            "artifacts": [],
            "diagnostics": [
                {
                    "code": "grasp_pose_estimate_failed",
                    "reason": reason,
                    "retryable": retryable,
                }
            ],
        },
    )


def _restore_anygrasp_length_scale(response: JsonDict, factor: float) -> JsonDict:
    """Restore candidate lengths after the fixed-cutoff service scale workaround."""

    if factor == 1.0:
        return response
    details = response.get("details")
    if not isinstance(details, dict):
        return response
    candidates = details.get("grasp_candidates")
    if not isinstance(candidates, list):
        return response
    for value in candidates:
        if not isinstance(value, dict):
            continue
        valid = True
        for key in ("depth", "width", "height"):
            if key not in value:
                continue
            try:
                number = float(value[key])
            except (TypeError, ValueError):
                valid = False
                continue
            if not math.isfinite(number):
                valid = False
                continue
            value[key] = number * factor
        for key in ("translation_xyz", "gripper_tip_position_xyz"):
            if key not in value:
                continue
            vector = _finite_vector(value[key], length=3)
            if vector is None:
                valid = False
                continue
            value[key] = [component * factor for component in vector]
        value["depth_cutoff_factor"] = factor
        value["length_scale_restored"] = valid
    metadata = details.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        details["metadata"] = metadata
    metadata.update(
        {
            "depth_cutoff_factor": factor,
            "length_scale_correction": factor,
            "lengths_restored_to_metres": True,
            "compatibility_warning": (
                "service-internal collision and gripper geometry are not scale-invariant"
            ),
        }
    )
    return response


def _normalise_contact_graspnet_response(
    response: Any,
    *,
    source_rgb: str,
    source_depth: str,
    object_mask: str,
    intrinsics: JsonDict,
) -> ToolResult:
    if not isinstance(response, dict):
        return _contact_graspnet_failure("mcp_call_failed")
    details = response.get("details")
    success = bool(response.get("success", False))
    if not isinstance(details, dict):
        return _contact_graspnet_failure(
            "inconsistent_grasp_outputs" if success else "mcp_call_failed"
        )
    if not success:
        reason = _string_param(details.get("reason")) or "unknown_error"
        content = _string_param(response.get("content")) or (
            f"Contact-GraspNet grasp prediction failed: {reason}."
        )
        return _contact_graspnet_failure(
            reason,
            content=content,
            metadata=_scrub_contact_graspnet_payload(_dict_or_empty(details.get("metadata"))),
        )

    if (
        details.get("tool") != "contact_graspnet"
        or details.get("backend") != "contact_graspnet_mcp"
        or details.get("model") != CONTACT_GRASPNET_MODEL
        or details.get("mode") != "targeted"
        or details.get("frame") != "camera"
        or details.get("camera_frame") != "opencv"
        or details.get("grasp_frame") != "graspnet"
    ):
        return _contact_graspnet_failure("inconsistent_grasp_outputs")

    candidates_value = details.get("grasp_candidates")
    try:
        candidate_count = int(details.get("candidate_count"))
    except (TypeError, ValueError):
        candidate_count = -1
    if (
        not isinstance(candidates_value, list)
        or candidate_count != len(candidates_value)
        or not 1 <= candidate_count <= CONTACT_GRASPNET_MAX_CANDIDATES
    ):
        return _contact_graspnet_failure("inconsistent_grasp_outputs")

    metadata = _scrub_contact_graspnet_payload(_dict_or_empty(details.get("metadata")))
    max_gripper_width = _finite_float(metadata.get("max_gripper_width"))
    if max_gripper_width is None or max_gripper_width <= 0:
        return _contact_graspnet_failure("inconsistent_grasp_outputs")

    candidates: list[JsonDict] = []
    candidate_ids: set[str] = set()
    previous_score = math.inf
    for value in candidates_value:
        candidate = _normalise_contact_graspnet_candidate(
            value,
            max_gripper_width=max_gripper_width,
        )
        if (
            candidate is None
            or candidate["id"] in candidate_ids
            or candidate["score"] > previous_score
        ):
            return _contact_graspnet_failure("inconsistent_grasp_outputs")
        candidate_ids.add(candidate["id"])
        previous_score = candidate["score"]
        candidates.append(candidate)

    return ToolResult(
        True,
        content=_string_param(response.get("content"))
        or "Contact-GraspNet grasp prediction completed.",
        details={
            "tool": "contact_graspnet",
            "backend": "contact_graspnet_mcp",
            "model": CONTACT_GRASPNET_MODEL,
            "mode": "targeted",
            "frame": "camera",
            "camera_frame": "opencv",
            "grasp_frame": "graspnet",
            "source_rgb": source_rgb,
            "source_depth": source_depth,
            "object_mask": object_mask,
            "source": {
                "mode": "targeted",
                "rgb": source_rgb,
                "depth": source_depth,
                "object_mask": object_mask,
                "intrinsics": dict(intrinsics),
            },
            "candidate_count": len(candidates),
            "grasp_candidates": candidates,
            "artifacts": [],
            "metadata": metadata,
        },
    )


def _normalise_contact_graspnet_candidate(
    value: Any,
    *,
    max_gripper_width: float,
) -> JsonDict | None:
    if not isinstance(value, Mapping):
        return None
    if (
        value.get("frame") != "camera"
        or value.get("camera_frame") != "opencv"
        or value.get("grasp_frame") != "graspnet"
        or value.get("source_model") != "contact_graspnet"
        or value.get("gripper_model") != "panda"
    ):
        return None
    candidate_id = _string_param(value.get("id"))
    score = _finite_float(value.get("score"))
    translation = _finite_vector(value.get("translation_xyz"), length=3)
    rotation = _finite_matrix3(value.get("rotation_matrix"))
    gripper_depth = _finite_float(value.get("gripper_depth"))
    width = _finite_float(value.get("width"))
    tip = _finite_vector(value.get("gripper_tip_position_xyz"), length=3)
    contact = _finite_vector(value.get("contact_point_xyz"), length=3)
    if (
        not candidate_id
        or score is None
        or translation is None
        or rotation is None
        or gripper_depth is None
        or width is None
        or tip is None
        or contact is None
        or not _is_rotation_matrix3(rotation)
        or not math.isclose(
            gripper_depth,
            CONTACT_GRASPNET_GRIPPER_DEPTH,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or width < 0
        or width > max_gripper_width + 1e-6
    ):
        return None
    expected_tip = [translation[row] + gripper_depth * rotation[row][0] for row in range(3)]
    if any(
        not math.isclose(tip[row], expected_tip[row], rel_tol=0.0, abs_tol=1e-5) for row in range(3)
    ):
        return None
    return {
        "id": candidate_id,
        "frame": "camera",
        "camera_frame": "opencv",
        "grasp_frame": "graspnet",
        "source_model": "contact_graspnet",
        "gripper_model": "panda",
        "score": score,
        "translation_xyz": translation,
        "rotation_matrix": rotation,
        "gripper_depth": gripper_depth,
        "width": width,
        "gripper_tip_position_xyz": tip,
        "contact_point_xyz": contact,
    }


def _contact_graspnet_failure(
    reason: str,
    *,
    content: str | None = None,
    metadata: JsonDict | None = None,
) -> ToolResult:
    return ToolResult(
        False,
        content=content or f"Contact-GraspNet grasp prediction failed: {reason}.",
        details={
            "tool": "contact_graspnet",
            "backend": "contact_graspnet_mcp",
            "model": CONTACT_GRASPNET_MODEL,
            "mode": "targeted",
            "frame": "camera",
            "camera_frame": "opencv",
            "grasp_frame": "graspnet",
            "candidate_count": 0,
            "grasp_candidates": [],
            "artifacts": [],
            "reason": reason,
            "metadata": dict(metadata or {}),
        },
    )


def _normalise_graspgenx_response(
    response: Any,
    *,
    source_rgb: str,
    source_depth: str,
    object_mask: str,
    intrinsics: JsonDict,
    gripper_name: str,
    up_direction_camera: list[float],
) -> ToolResult:
    if not isinstance(response, Mapping):
        return _graspgenx_failure("mcp_call_failed")
    details = response.get("details")
    success = bool(response.get("success", False))
    if not isinstance(details, Mapping):
        return _graspgenx_failure(
            "inconsistent_grasp_outputs" if success else "mcp_call_failed"
        )
    if not success:
        reason = _string_param(details.get("reason")) or "unknown_error"
        content = _string_param(response.get("content")) or (
            f"GraspGenX grasp prediction failed: {reason}."
        )
        return _graspgenx_failure(
            reason,
            content=content,
            metadata=_scrub_graspgenx_payload(
                _dict_or_empty(details.get("metadata"))
            ),
        )
    if (
        details.get("tool") != "predict_grasps"
        or details.get("backend") != GRASPGENX_BACKEND
        or details.get("model") != GRASPGENX_MODEL
        or details.get("planner") != GRASPGENX_PLANNER
        or details.get("deterministic") is not False
        or details.get("frame") != "camera"
        or details.get("camera_frame") != "opencv"
        or details.get("grasp_frame") != "graspnet"
        or details.get("gripper_name") != gripper_name
        or details.get("ranking") != "score_descending"
    ):
        return _graspgenx_failure("inconsistent_grasp_outputs")

    candidates_value = details.get("grasp_candidates")
    candidate_count = details.get("candidate_count")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or not isinstance(candidates_value, list)
        or candidate_count != len(candidates_value)
        or not 1 <= candidate_count <= GRASPGENX_MAX_CANDIDATES
    ):
        return _graspgenx_failure("inconsistent_grasp_outputs")

    candidates: list[JsonDict] = []
    previous_score = math.inf
    candidate_ids: set[str] = set()
    for rank, value in enumerate(candidates_value):
        candidate = _normalise_graspgenx_candidate(
            value,
            expected_rank=rank,
            gripper_name=gripper_name,
        )
        if (
            candidate is None
            or candidate["id"] in candidate_ids
            or float(candidate["score"]) > previous_score
        ):
            return _graspgenx_failure("inconsistent_grasp_outputs")
        candidate_ids.add(candidate["id"])
        previous_score = float(candidate["score"])
        candidates.append(candidate)

    metadata_value = details.get("metadata")
    if not isinstance(metadata_value, Mapping):
        return _graspgenx_failure("inconsistent_grasp_outputs")
    metadata = _scrub_graspgenx_payload(metadata_value)
    return ToolResult(
        True,
        content=_string_param(response.get("content"))
        or "GraspGenX grasp prediction completed.",
        details={
            "tool": "graspgenx",
            "backend": GRASPGENX_BACKEND,
            "model": GRASPGENX_MODEL,
            "planner": GRASPGENX_PLANNER,
            "deterministic": False,
            "mode": "targeted",
            "frame": "camera",
            "camera_frame": "opencv",
            "grasp_frame": "graspnet",
            "gripper_name": gripper_name,
            "source_rgb": source_rgb,
            "source_depth": source_depth,
            "object_mask": object_mask,
            "source": {
                "source_tool": "graspgenx",
                "mode": "targeted",
                "rgb": source_rgb,
                "depth": source_depth,
                "object_mask": object_mask,
                "intrinsics": dict(intrinsics),
                "gripper_name": gripper_name,
                "up_direction_camera": list(up_direction_camera),
            },
            "candidate_count": len(candidates),
            "grasp_candidates": candidates,
            "best_grasp_candidate": candidates[0],
            "active_grasp_candidate": candidates[0],
            "ranking": "score_descending",
            "artifacts": [],
            "diagnostics": [],
            "metadata": metadata,
        },
    )


def _normalise_graspgenx_candidate(
    value: Any,
    *,
    expected_rank: int,
    gripper_name: str,
) -> JsonDict | None:
    if not isinstance(value, Mapping):
        return None
    rank = value.get("rank")
    backend_index = value.get("backend_index")
    if (
        value.get("id") != f"graspgenx_{expected_rank:03d}"
        or isinstance(rank, bool)
        or rank != expected_rank
        or isinstance(backend_index, bool)
        or not isinstance(backend_index, int)
        or backend_index < 0
        or value.get("source_model") != GRASPGENX_MODEL
        or value.get("gripper_name") != gripper_name
        or value.get("candidate_source") not in {"diffusion", "obb"}
        or value.get("frame") != "camera"
        or value.get("camera_frame") != "opencv"
        or value.get("grasp_frame") != "graspnet"
        or value.get("convention") != GRASPGENX_POSE_CONVENTION
    ):
        return None
    score = _finite_float(value.get("score"))
    translation = _finite_vector(value.get("translation_xyz"), length=3)
    rotation = _finite_matrix3(value.get("rotation_matrix"))
    transform = _finite_matrix4(value.get("transform_matrix"))
    tip = _finite_vector(value.get("gripper_tip_position_xyz"), length=3)
    depth = _finite_float(value.get("depth"))
    width = _finite_float(value.get("width"))
    height = _finite_float(value.get("height"))
    if (
        score is None
        or translation is None
        or rotation is None
        or transform is None
        or tip is None
        or depth is None
        or width is None
        or height is None
        or not _is_rotation_matrix3(rotation)
        or not _is_rigid_transform4(transform)
        or depth <= 0
        or width <= 0
        or height <= 0
    ):
        return None
    for row in range(3):
        if not math.isclose(
            transform[row][3], translation[row], rel_tol=0.0, abs_tol=1e-6
        ):
            return None
        for column in range(3):
            if not math.isclose(
                transform[row][column],
                rotation[row][column],
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                return None
    return {
        "id": f"graspgenx_{expected_rank:03d}",
        "rank": expected_rank,
        "backend_index": backend_index,
        "source_model": GRASPGENX_MODEL,
        "gripper_name": gripper_name,
        "candidate_source": value["candidate_source"],
        "frame": "camera",
        "camera_frame": "opencv",
        "grasp_frame": "graspnet",
        "convention": GRASPGENX_POSE_CONVENTION,
        "score": score,
        "translation_xyz": translation,
        "rotation_matrix": rotation,
        "transform_matrix": transform,
        "gripper_tip_position_xyz": tip,
        "depth": depth,
        "width": width,
        "height": height,
    }


def _graspgenx_failure(
    reason: str,
    *,
    content: str | None = None,
    metadata: JsonDict | None = None,
) -> ToolResult:
    return ToolResult(
        False,
        content=content or f"GraspGenX grasp prediction failed: {reason}.",
        details={
            "tool": "graspgenx",
            "backend": GRASPGENX_BACKEND,
            "model": GRASPGENX_MODEL,
            "planner": GRASPGENX_PLANNER,
            "deterministic": False,
            "mode": "targeted",
            "frame": "camera",
            "camera_frame": "opencv",
            "grasp_frame": "graspnet",
            "candidate_count": 0,
            "grasp_candidates": [],
            "artifacts": [],
            "diagnostics": [],
            "reason": reason,
            "metadata": dict(metadata or {}),
        },
    )


def _normalise_graspgenx_gripper_listing(response: Any) -> list[JsonDict]:
    if not isinstance(response, Mapping) or not bool(response.get("success", False)):
        raise ValueError("invalid GraspGenX gripper listing")
    details = response.get("details")
    if not isinstance(details, Mapping) or details.get("tool") != "list_grippers":
        raise ValueError("invalid GraspGenX gripper listing")
    values = details.get("grippers")
    count = details.get("gripper_count")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not isinstance(values, list)
        or count != len(values)
        or count < 1
    ):
        raise ValueError("invalid GraspGenX gripper listing")
    grippers: list[JsonDict] = []
    names: set[str] = set()
    for value in values:
        gripper = _normalise_graspgenx_gripper(value)
        if gripper is None or gripper["name"] in names:
            raise ValueError("invalid GraspGenX gripper listing")
        names.add(gripper["name"])
        grippers.append(gripper)
    if [item["name"] for item in grippers] != sorted(names):
        raise ValueError("invalid GraspGenX gripper listing")
    return grippers


def _normalise_graspgenx_gripper(value: Any) -> JsonDict | None:
    if not isinstance(value, Mapping):
        return None
    name = _string_param(value.get("name"))
    gripper_type = _string_param(value.get("gripper_type"))
    asset_family = _string_param(value.get("asset_family"))
    fingertip_depth = _finite_float(value.get("fingertip_depth"))
    open_sweep = value.get("sweep_volume_open")
    mid_sweep = value.get("sweep_volume_mid")
    if (
        not name
        or not gripper_type
        or not asset_family
        or fingertip_depth is None
        or fingertip_depth <= 0
        or not isinstance(open_sweep, Mapping)
        or not isinstance(mid_sweep, Mapping)
    ):
        return None
    open_extents = _finite_vector(open_sweep.get("extents_xyz"), length=3)
    open_offset = _finite_vector(open_sweep.get("offset_xyz"), length=3)
    mid_extents = _finite_vector(mid_sweep.get("extents_xyz"), length=3)
    mid_offset = _finite_vector(mid_sweep.get("offset_xyz"), length=3)
    if (
        open_extents is None
        or open_offset is None
        or mid_extents is None
        or mid_offset is None
        or any(item <= 0 for item in (*open_extents, *mid_extents))
    ):
        return None
    return {
        "name": name,
        "gripper_type": gripper_type,
        "fingertip_depth": fingertip_depth,
        "sweep_volume_open": {
            "extents_xyz": open_extents,
            "offset_xyz": open_offset,
        },
        "sweep_volume_mid": {
            "extents_xyz": mid_extents,
            "offset_xyz": mid_offset,
        },
        "asset_family": asset_family,
    }


def _graspgenx_gripper_list_failure(
    reason: str,
    *,
    metadata: JsonDict | None = None,
) -> ToolResult:
    return ToolResult(
        False,
        content=f"GraspGenX gripper listing failed: {reason}.",
        details={
            "tool": "list_graspgenx_grippers",
            "backend": GRASPGENX_BACKEND,
            "model": GRASPGENX_MODEL,
            "gripper_count": 0,
            "grippers": [],
            "reason": reason,
            "metadata": dict(metadata or {}),
        },
    )


class _MolmoPointHandlerError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _best_effort_molmopoint_paths(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    paths: list[Any] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            paths.append(item)
            continue
        try:
            paths.append(str(Path(item).expanduser().resolve()))
        except OSError:
            paths.append(item.strip())
    return paths


def _prepare_molmopoint_images(
    values: list[Any],
) -> tuple[list[str], list[JsonDict], list[JsonDict]]:
    from PIL import Image

    normalized_paths: list[str] = []
    payloads: list[JsonDict] = []
    metadata: list[JsonDict] = []
    total_pixels = 0
    for image_index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise _MolmoPointHandlerError("invalid_image_path")
        try:
            path = Path(value).expanduser().resolve()
        except OSError as exc:
            raise _MolmoPointHandlerError("invalid_image_path") from exc
        normalized_paths.append(str(path))
        if not path.is_file():
            raise _MolmoPointHandlerError("image_not_found")
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise _MolmoPointHandlerError("unsupported_image_format")
        try:
            raw = path.read_bytes()
            image = Image.open(io.BytesIO(raw))
        except OSError as exc:
            raise _MolmoPointHandlerError("decode_failed") from exc
        try:
            image_format = _string_param(image.format).lower()
            if image_format == "jpg":
                image_format = "jpeg"
            if image_format not in {"png", "jpeg"}:
                raise _MolmoPointHandlerError("unsupported_image_format")
            if int(getattr(image, "n_frames", 1)) != 1:
                raise _MolmoPointHandlerError("decode_failed")
            if image.getexif().get(274, 1) not in (None, 1):
                raise _MolmoPointHandlerError("unsupported_image_orientation")
            width, height = image.size
            pixels = width * height
            total_pixels += pixels
            if (
                width <= 0
                or height <= 0
                or max(width, height) > MOLMOPOINT_MAX_IMAGE_SIDE
                or pixels > MOLMOPOINT_MAX_IMAGE_PIXELS
                or total_pixels > MOLMOPOINT_MAX_TOTAL_IMAGE_PIXELS
            ):
                raise _MolmoPointHandlerError("image_too_large")
            source_mode = _string_param(image.mode)
            image.load()
        except _MolmoPointHandlerError:
            raise
        except Exception as exc:  # noqa: BLE001 - Pillow decoding boundary.
            raise _MolmoPointHandlerError("decode_failed") from exc
        finally:
            image.close()
        payloads.append(
            {
                "format": image_format,
                "base64": base64.b64encode(raw).decode("ascii"),
            }
        )
        metadata.append(
            {
                "image_index": image_index,
                "format": image_format,
                "width": width,
                "height": height,
                "source_image_mode": source_mode,
                "model_image_mode": "RGB",
            }
        )
    return normalized_paths, payloads, metadata


def _normalise_molmopoint_response(
    context: ToolExecutionContext,
    response: Any,
    *,
    image_metadata: list[JsonDict],
) -> ToolResult:
    if not isinstance(response, Mapping):
        return _molmopoint_failure(context, "mcp_call_failed")
    details = response.get("details")
    success = response.get("success") is True
    if not isinstance(details, Mapping):
        return _molmopoint_failure(
            context,
            "inconsistent_point_outputs" if success else "mcp_call_failed",
        )
    if not success:
        reason = _string_param(details.get("reason")) or "inference_failed"
        return _molmopoint_failure(context, reason)
    if _molmopoint_contains_blocked_output(response):
        return _molmopoint_failure(context, "inconsistent_point_outputs")
    model = _string_param(details.get("model"))
    if (
        details.get("tool") != "molmopoint"
        or details.get("backend") != "molmopoint_mcp"
        or not model
        or details.get("coordinate_convention") != MOLMOPOINT_COORDINATE_CONVENTION
    ):
        return _molmopoint_failure(context, "inconsistent_point_outputs")

    metadata_value = details.get("metadata")
    if not isinstance(metadata_value, Mapping):
        return _molmopoint_failure(context, "inconsistent_point_outputs")
    revision = _string_param(metadata_value.get("model_revision"))
    returned_images = metadata_value.get("images")
    if (
        re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None
        or returned_images != image_metadata
    ):
        return _molmopoint_failure(context, "inconsistent_point_outputs")
    metadata: JsonDict = {
        "model": model,
        "model_revision": revision,
        "images": [dict(item) for item in image_metadata],
    }
    for key in ("model_load_seconds", "inference_seconds", "total_seconds"):
        if key not in metadata_value:
            continue
        number = _finite_float(metadata_value.get(key))
        if number is None or number < 0:
            return _molmopoint_failure(context, "inconsistent_point_outputs")
        metadata[key] = number

    values = details.get("points")
    point_count = details.get("point_count")
    if (
        isinstance(point_count, bool)
        or not isinstance(point_count, int)
        or not isinstance(values, list)
        or point_count != len(values)
    ):
        return _molmopoint_failure(context, "inconsistent_point_outputs")
    points: list[JsonDict] = []
    for point_index, value in enumerate(values):
        if not isinstance(value, Mapping):
            return _molmopoint_failure(context, "inconsistent_point_outputs")
        image_index = value.get("image_index")
        pixel_x = _finite_float(value.get("pixel_x"))
        pixel_y = _finite_float(value.get("pixel_y"))
        if (
            value.get("id") != f"point_{point_index:03d}"
            or isinstance(image_index, bool)
            or not isinstance(image_index, int)
            or not 0 <= image_index < len(image_metadata)
            or pixel_x is None
            or pixel_y is None
        ):
            return _molmopoint_failure(context, "inconsistent_point_outputs")
        image = image_metadata[image_index]
        if not (0 <= pixel_x < image["width"] and 0 <= pixel_y < image["height"]):
            return _molmopoint_failure(context, "inconsistent_point_outputs")
        if set(value) != {"id", "image_index", "pixel_x", "pixel_y"}:
            return _molmopoint_failure(context, "inconsistent_point_outputs")
        points.append(
            {
                "id": f"point_{point_index:03d}",
                "image_index": image_index,
                "pixel_x": pixel_x,
                "pixel_y": pixel_y,
            }
        )
    return make_tool_result(
        context,
        success=True,
        content=_string_param(response.get("content"))
        or "MolmoPoint image pointing completed.",
        outputs={
            "image_count": len(image_metadata),
            "point_count": len(points),
            "points": points,
            "image_sources": list(context.parameters["images"]),
            "coordinate_convention": dict(MOLMOPOINT_COORDINATE_CONVENTION),
            "metadata": metadata,
        },
    )


def _molmopoint_failure(context: ToolExecutionContext, reason: str) -> ToolResult:
    image_count = len(context.parameters.get("images", [])) if isinstance(
        context.parameters.get("images"), list
    ) else 0
    return make_tool_result(
        context,
        success=False,
        content=f"MolmoPoint image pointing failed: {reason}.",
        outputs={
            "image_count": image_count,
            "point_count": 0,
            "points": [],
            "image_sources": [],
            "coordinate_convention": {},
            "metadata": {},
        },
        diagnostics=[{"code": reason}],
    )


def _molmopoint_contains_blocked_output(value: Any) -> bool:
    blocked = {
        "base64",
        "raw_generation",
        "generated_text",
        "object_id",
        "object_ids",
    }
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in blocked or _molmopoint_contains_blocked_output(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_molmopoint_contains_blocked_output(child) for child in value)
    return False


def _scrub_molmopoint_payload(value: Any) -> Any:
    blocked = {
        "base64",
        "raw_generation",
        "generated_text",
        "object_id",
        "object_ids",
        "hf_home",
        "cache_dir",
        "model_path",
        "python_path",
    }
    if isinstance(value, Mapping):
        return {
            str(key): _scrub_molmopoint_payload(child)
            for key, child in value.items()
            if str(key).lower() not in blocked
        }
    if isinstance(value, list):
        return [_scrub_molmopoint_payload(child) for child in value]
    return value


def _build_molmopoint_visual_artifacts(
    *,
    image_paths: list[str],
    points: list[Any],
    output_dir: Path,
) -> list[JsonDict]:
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    panels: list[Any] = []
    artifacts: list[JsonDict] = []
    for image_index, image_path in enumerate(image_paths):
        with Image.open(image_path) as source:
            canvas = source.convert("RGB")
        image_points = [
            point
            for point in points
            if isinstance(point, Mapping) and point.get("image_index") == image_index
        ]
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        radius = max(8, min(canvas.size) // 25)
        stroke = max(3, radius // 4)
        for point in image_points:
            x = float(point["pixel_x"])
            y = float(point["pixel_y"])
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=(255, 35, 35),
                width=stroke,
            )
            draw.line(
                (x - radius * 1.4, y, x + radius * 1.4, y),
                fill=(255, 255, 0),
                width=stroke,
            )
            draw.line(
                (x, y - radius * 1.4, x, y + radius * 1.4),
                fill=(255, 255, 0),
                width=stroke,
            )
            draw.text(
                (x + radius + 4, y - radius),
                str(point["id"]),
                fill=(255, 255, 0),
                font=font,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
        overlay = canvas
        title_draw = ImageDraw.Draw(overlay)
        title_draw.rectangle((0, 0, min(overlay.width, 420), 30), fill=(0, 0, 0))
        title_draw.text(
            (8, 10),
            f"Image {image_index + 1} / image_index={image_index} / {len(image_points)} point(s)",
            fill=(255, 255, 255),
            font=font,
        )
        overlay_ref = output_dir / f"image_{image_index}_overlay.png"
        overlay.save(overlay_ref, format="PNG")
        artifacts.append(
            {
                "type": "molmopoint_point_overlay",
                "kind": "image",
                "tool": "molmopoint",
                "image_index": image_index,
                "path": str(overlay_ref),
            }
        )
        panels.append(ImageOps.contain(overlay, (2048, 2048)))

    columns = min(2, len(panels))
    rows = math.ceil(len(panels) / columns)
    cell_width = max(panel.width for panel in panels)
    cell_height = max(panel.height for panel in panels)
    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        (32, 32, 32),
    )
    for panel_index, panel in enumerate(panels):
        x = (panel_index % columns) * cell_width + (cell_width - panel.width) // 2
        y = (panel_index // columns) * cell_height + (cell_height - panel.height) // 2
        sheet.paste(panel, (x, y))
    if max(sheet.size) > 4096:
        sheet.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
    sheet_ref = output_dir / "contact_sheet.png"
    sheet.save(sheet_ref, format="PNG")
    artifacts.append(
        {
            "type": "molmopoint_contact_sheet",
            "kind": "image",
            "tool": "molmopoint",
            "path": str(sheet_ref),
        }
    )
    return artifacts


def _graspgenx_raw_candidates(response: Any) -> list[JsonDict]:
    if not isinstance(response, Mapping):
        raise ValueError("invalid GraspGenX response")
    details = response.get("details")
    values = details.get("grasp_candidates") if isinstance(details, Mapping) else None
    if not isinstance(values, list) or not values:
        raise ValueError("missing GraspGenX candidates")
    return [dict(value) for value in values if isinstance(value, Mapping)]


def _build_graspgenx_visual_artifacts(
    *,
    rgb_path: str,
    mask_path: str,
    candidates: list[JsonDict],
    gripper_geometry: JsonDict,
    intrinsics: JsonDict,
    output_dir: Path,
) -> list[JsonDict]:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    if not candidates:
        raise ValueError("missing GraspGenX candidates")
    sweep = gripper_geometry.get("sweep_volume_open")
    if not isinstance(sweep, Mapping):
        raise ValueError("missing GraspGenX sweep geometry")
    extents = np.asarray(sweep.get("extents_xyz"), dtype=np.float64)
    offset = np.asarray(sweep.get("offset_xyz"), dtype=np.float64)
    if (
        extents.shape != (3,)
        or offset.shape != (3,)
        or not np.isfinite(extents).all()
        or not np.isfinite(offset).all()
        or np.any(extents <= 0)
    ):
        raise ValueError("invalid GraspGenX sweep geometry")
    low = offset - extents / 2.0
    high = offset + extents / 2.0
    x0, y0, z0 = low
    x1, y1, z1 = high
    local_box = np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x0, y1, z0],
            [x1, y1, z0],
            [x0, y1, z1],
            [x1, y1, z1],
        ],
        dtype=np.float64,
    )

    def project(points: Any) -> Any:
        points_array = np.asarray(points, dtype=np.float64)
        pixels = np.full((len(points_array), 2), np.nan, dtype=np.float64)
        valid = points_array[:, 2] > 1e-6
        pixels[valid, 0] = (
            float(intrinsics["fx"])
            * points_array[valid, 0]
            / points_array[valid, 2]
            + float(intrinsics["cx"])
        )
        pixels[valid, 1] = (
            float(intrinsics["fy"])
            * points_array[valid, 1]
            / points_array[valid, 2]
            + float(intrinsics["cy"])
        )
        return pixels

    def render(selected: list[JsonDict], output_path: Path, title: str) -> None:
        with Image.open(rgb_path) as source_rgb:
            rgb = source_rgb.convert("RGBA")
        with Image.open(mask_path) as source_mask:
            mask = np.asarray(source_mask.convert("L")) > 0
        if mask.shape != (rgb.height, rgb.width):
            raise ValueError("RGB and object mask dimensions differ")
        mask_layer = np.zeros((rgb.height, rgb.width, 4), dtype=np.uint8)
        mask_layer[mask] = [255, 45, 25, 42]
        canvas = Image.alpha_composite(rgb, Image.fromarray(mask_layer, mode="RGBA"))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        for index in reversed(range(len(selected))):
            candidate = selected[index]
            native = candidate.get("model_native_grasp_pose")
            if not isinstance(native, Mapping):
                raise ValueError("missing model-native GraspGenX pose")
            transform_value = _finite_matrix4(native.get("transform_matrix"))
            if transform_value is None or not _is_rigid_transform4(transform_value):
                raise ValueError("invalid model-native GraspGenX pose")
            transform = np.asarray(transform_value, dtype=np.float64)
            color = GRASPGENX_PALETTE[index % len(GRASPGENX_PALETTE)]
            line_width = 5 if len(selected) == 1 else (4 if index == 0 else 2)
            box_camera = local_box @ transform[:3, :3].T + transform[:3, 3]
            pixels = project(box_camera)
            for start, end in GRASPGENX_BOX_EDGES:
                if np.isfinite(pixels[[start, end]]).all():
                    draw.line(
                        [tuple(pixels[start]), tuple(pixels[end])],
                        fill=color,
                        width=line_width,
                    )

            base_camera = transform[:3, 3]
            sweep_center = offset @ transform[:3, :3].T + transform[:3, 3]
            approach_pixels = project(np.stack([base_camera, sweep_center], axis=0))
            if np.isfinite(approach_pixels).all():
                draw.line(
                    [tuple(approach_pixels[0]), tuple(approach_pixels[1])],
                    fill=color,
                    width=max(2, line_width - 1),
                )

            tip = _finite_vector(candidate.get("gripper_tip_position_xyz"), length=3)
            if tip is None:
                raise ValueError("invalid GraspGenX fingertip")
            tip_pixel = project(np.asarray([tip], dtype=np.float64))[0]
            if np.isfinite(tip_pixel).all():
                x, y = tip_pixel
                radius = 7 if index == 0 else 4
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=color,
                    outline=(0, 0, 0, 255),
                    width=2,
                )
            center_pixel = project(np.asarray([sweep_center], dtype=np.float64))[0]
            score = _finite_float(candidate.get("score"))
            if score is None:
                raise ValueError("invalid GraspGenX score")
            if np.isfinite(center_pixel).all() and (len(selected) == 1 or index == 0):
                x, y = center_pixel
                draw.text(
                    (x + 8, y - 19),
                    f"#{index + 1} {score:.3f}",
                    fill=color,
                    font=font,
                    stroke_width=2,
                    stroke_fill=(0, 0, 0, 230),
                )

        draw.rectangle((0, 0, rgb.width, 54), fill=(0, 0, 0, 180))
        draw.text((14, 8), title, fill=(255, 255, 255, 255), font=font)
        draw.text(
            (14, 31),
            "native pose + advertised sweep volume; dot = fingertip; red = mask",
            fill=(220, 220, 220, 255),
            font=font,
        )
        canvas.convert("RGB").save(output_path, format="PNG")

    selected = candidates[: min(GRASPGENX_VISUAL_LIMIT, len(candidates))]
    top_1_ref = output_dir / "top_1_overlay.png"
    top_10_ref = output_dir / "top_10_overlay.png"
    render(
        selected[:1],
        top_1_ref,
        "GraspGenX: top-1 native grasp + sweep volume",
    )
    render(
        selected,
        top_10_ref,
        f"GraspGenX: top {len(selected)} native grasp volumes",
    )
    return [
        {
            "type": "graspgenx_grasp_overlay",
            "kind": "image",
            "tool": "graspgenx",
            "selection": "top_1",
            "candidate_ids": [str(selected[0].get("id") or "")],
            "path": str(top_1_ref),
        },
        {
            "type": "graspgenx_grasp_overlay",
            "kind": "image",
            "tool": "graspgenx",
            "selection": "top_10",
            "candidate_ids": [str(item.get("id") or "") for item in selected],
            "path": str(top_10_ref),
        },
    ]


def _anygrasp_inconsistent(
    *,
    mode: str,
    source_rgb: str,
    source_depth: str,
    target_mask: str,
    metadata: JsonDict | None = None,
) -> ToolResult:
    return _anygrasp_failure(
        mode=mode,
        source_rgb=source_rgb,
        source_depth=source_depth,
        target_mask=target_mask,
        reason="inconsistent_grasp_outputs",
        content="AnyGrasp returned inconsistent grasp outputs.",
        metadata=metadata,
    )


def _normalise_anyplace_response(
    response: JsonDict,
    *,
    selected_grasp: JsonDict,
    selected_grasp_source: JsonDict,
    request: JsonDict,
    output_root: Path,
) -> ToolResult:
    if not isinstance(response, dict):
        return _anyplace_failure(
            "mcp_call_failed",
            "AnyPlace placement prediction failed: invalid MCP response.",
        )
    details = response.get("details")
    success = bool(response.get("success", False))
    if not isinstance(details, dict):
        if success:
            return _anyplace_inconsistent()
        details = {}
    if not success:
        reason = _string_param(details.get("reason")) or "unknown_error"
        content = _string_param(response.get("content")) or (
            f"AnyPlace placement prediction failed: {reason}."
        )
        return _anyplace_failure(
            reason,
            content,
            metadata=_scrub_anyplace_response(_dict_or_empty(details.get("metadata"))),
        )
    if details.get("frame") != "camera" or details.get("camera_frame") != "opencv":
        return _anyplace_inconsistent()
    candidates_value = details.get("placement_candidates")
    try:
        candidate_count = int(details.get("candidate_count"))
    except (TypeError, ValueError):
        candidate_count = -1
    if not isinstance(candidates_value, list) or candidate_count != 5 or len(candidates_value) != 5:
        return _anyplace_inconsistent()

    candidates: list[JsonDict] = []
    candidate_ids: set[str] = set()
    for candidate in candidates_value:
        normalized = _normalise_anyplace_candidate(
            candidate,
            selected_grasp=selected_grasp,
            selected_grasp_source=selected_grasp_source,
        )
        if normalized is None or normalized["id"] in candidate_ids:
            return _anyplace_inconsistent()
        candidate_ids.add(normalized["id"])
        candidates.append(normalized)

    run_dir = _new_run_dir(output_root)
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        run_dir = _new_run_dir(output_root)
        run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "request.json", request)
    raw_output_ref = run_dir / "response.raw.json"
    _write_json(raw_output_ref, _scrub_anyplace_response(response))
    result = ToolResult(
        True,
        content=_string_param(response.get("content"))
        or "AnyPlace placement prediction completed.",
        details={
            "tool": "anyplace",
            "backend": _string_param(details.get("backend")) or "anyplace_mcp",
            "model": _string_param(details.get("model")) or "anyplace_multitask",
            "frame": "camera",
            "camera_frame": "opencv",
            "source": {
                "rgb": request["rgb"],
                "depth": request["depth"],
                "object_mask": request["object_mask"],
                "placement_region_mask": request["placement_region_mask"],
                "intrinsics": request["intrinsics"],
            },
            "selected_grasp_source": dict(selected_grasp_source),
            "selected_grasp_id": selected_grasp["id"],
            "raw_output_ref": str(raw_output_ref),
            "candidate_count": 5,
            "placement_candidates": candidates,
            "metadata": _scrub_anyplace_response(_dict_or_empty(details.get("metadata"))),
        },
    )
    _write_json(
        run_dir / "tool_result.json",
        {"success": result.success, "content": result.content, "details": result.details},
    )
    return result


def _scale_anyplace_grasp_candidate(candidate: JsonDict, factor: float) -> JsonDict:
    """Match AnyPlace's fixed depth cutoff without changing camera geometry."""

    if factor == 1.0:
        return dict(candidate)
    scaled = dict(candidate)
    for key in ("depth", "width", "height"):
        scaled[key] = float(scaled[key]) / factor
    for key in ("translation_xyz", "gripper_tip_position_xyz"):
        scaled[key] = [float(component) / factor for component in scaled[key]]
    return scaled


def _restore_anyplace_length_scale(response: JsonDict, factor: float) -> JsonDict:
    """Restore placement transforms produced from uniformly scaled RGB-D."""

    if factor == 1.0 or not isinstance(response, dict):
        return response
    details = response.get("details")
    if not isinstance(details, dict):
        return response
    candidates = details.get("placement_candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            transform = candidate.get("object_placement_transform")
            matrix = (
                transform.get("transform_matrix")
                if isinstance(transform, dict)
                else None
            )
            if isinstance(matrix, list) and len(matrix) == 4:
                for row_index in range(3):
                    row = matrix[row_index]
                    if isinstance(row, list) and len(row) == 4:
                        row[3] = float(row[3]) * factor
            place = candidate.get("place_grasp_pose")
            if not isinstance(place, dict):
                continue
            for key in ("depth", "width", "height"):
                if key in place:
                    place[key] = float(place[key]) * factor
            for key in ("translation_xyz", "gripper_tip_position_xyz"):
                vector = place.get(key)
                if isinstance(vector, list) and len(vector) == 3:
                    place[key] = [float(component) * factor for component in vector]
    metadata = details.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        details["metadata"] = metadata
    metadata.update(
        {
            "depth_cutoff_factor": factor,
            "length_scale_correction": factor,
            "lengths_restored_to_metres": True,
        }
    )
    return response


def _normalise_anyplace_candidate(
    candidate: Any,
    *,
    selected_grasp: JsonDict,
    selected_grasp_source: JsonDict,
) -> JsonDict | None:
    if not isinstance(candidate, Mapping):
        return None
    candidate_id = _string_param(candidate.get("id"))
    if not candidate_id or _string_param(candidate.get("source_grasp_id")) != selected_grasp["id"]:
        return None
    transform_value = candidate.get("object_placement_transform")
    place_value = candidate.get("place_grasp_pose")
    if not isinstance(transform_value, Mapping) or not isinstance(place_value, Mapping):
        return None
    transform = _finite_matrix4(transform_value.get("transform_matrix"))
    if (
        transform is None
        or transform_value.get("frame") != "camera"
        or transform_value.get("camera_frame") != "opencv"
        or transform_value.get("convention") != "p_placed = R @ p_current + t"
        or not _is_rigid_transform4(transform)
    ):
        return None
    place = _normalise_anygrasp_candidate(place_value)
    if (
        place is None
        or place.get("camera_frame") != "opencv"
        or not _is_rotation_matrix3(place.get("rotation_matrix"))
        or not _same_gripper_shape(place, selected_grasp)
    ):
        return None
    source_tool = (
        _string_param(selected_grasp_source.get("source_tool"))
        or "anygrasp"
    )
    source_backend = (
        _string_param(selected_grasp_source.get("source_backend"))
        or source_tool
    )
    place["source_tool"] = source_tool
    place["source_backend"] = source_backend
    if source_backend == "graspgenx":
        gripper_name = _string_param(selected_grasp_source.get("gripper_name"))
        if not gripper_name or selected_grasp.get("gripper_name") != gripper_name:
            return None
        place["gripper_name"] = gripper_name
    return {
        "id": candidate_id,
        "source_grasp_id": selected_grasp["id"],
        "object_placement_transform": {
            "frame": "camera",
            "camera_frame": "opencv",
            "convention": "p_placed = R @ p_current + t",
            "transform_matrix": transform,
        },
        "place_grasp_pose": {
            **place,
            "source_grasp_id": selected_grasp["id"],
        },
    }


def _anyplace_inconsistent() -> ToolResult:
    return _anyplace_failure(
        "inconsistent_placement_outputs",
        "AnyPlace returned inconsistent placement outputs.",
    )


def _anyplace_failure(
    reason: str,
    content: str,
    *,
    metadata: JsonDict | None = None,
) -> ToolResult:
    return ToolResult(
        False,
        content=content,
        details={
            "tool": "anyplace",
            "backend": "anyplace_mcp",
            "model": "anyplace_multitask",
            "frame": "camera",
            "camera_frame": "opencv",
            "candidate_count": 0,
            "placement_candidates": [],
            "reason": reason,
            "metadata": dict(metadata or {}),
        },
    )


def _normalise_anygrasp_candidate(candidate: Any) -> JsonDict | None:
    if not isinstance(candidate, dict):
        return None
    if _string_param(candidate.get("frame")) != "camera":
        return None
    translation = _finite_vector(candidate.get("translation_xyz"), length=3)
    tip = _finite_vector(candidate.get("gripper_tip_position_xyz"), length=3)
    rotation_value = candidate.get("rotation_matrix")
    if not isinstance(rotation_value, list) or len(rotation_value) != 3:
        return None
    rotation: list[list[float]] = []
    for row in rotation_value:
        parsed = _finite_vector(row, length=3)
        if parsed is None:
            return None
        rotation.append(parsed)
    score = _finite_float(candidate.get("score"))
    depth = _finite_float(candidate.get("depth"))
    width = _finite_float(candidate.get("width"))
    height = _finite_float(candidate.get("height"))
    if None in {score, depth, width, height}:
        return None
    candidate_id = _string_param(candidate.get("id"))
    if not candidate_id:
        return None
    normalized: JsonDict = {
        "id": candidate_id,
        "frame": "camera",
        "camera_frame": _string_param(candidate.get("camera_frame")) or "opencv",
        "score": score,
        "translation_xyz": translation,
        "rotation_matrix": rotation,
        "depth": depth,
        "width": width,
        "height": height,
        "gripper_tip_position_xyz": tip,
    }
    grasp_frame = _string_param(candidate.get("grasp_frame"))
    if grasp_frame:
        if grasp_frame != "graspnet":
            return None
        normalized["grasp_frame"] = grasp_frame
    transform_value = candidate.get("transform_matrix")
    if transform_value is not None:
        transform = _finite_matrix4(transform_value)
        if transform is None or not _is_rigid_transform4(transform):
            return None
        for row in range(3):
            if not math.isclose(
                transform[row][3], translation[row], rel_tol=0.0, abs_tol=1e-6
            ):
                return None
            for column in range(3):
                if not math.isclose(
                    transform[row][column],
                    rotation[row][column],
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ):
                    return None
        normalized["transform_matrix"] = transform
    gripper_name = _string_param(candidate.get("gripper_name"))
    if "gripper_name" in candidate:
        if not gripper_name:
            return None
        normalized["gripper_name"] = gripper_name
    source_tool = _string_param(candidate.get("source_tool"))
    if source_tool:
        if source_tool not in {"grasp_pose_estimate", "anygrasp", "graspgenx"}:
            return None
        normalized["source_tool"] = source_tool
    source_backend = _string_param(candidate.get("source_backend"))
    if source_backend:
        if source_backend not in DEFAULT_GRASP_POSE_BACKEND_ORDER:
            return None
        normalized["source_backend"] = source_backend
    for key in ("source_model", "candidate_source", "convention"):
        value = _string_param(candidate.get(key))
        if value:
            normalized[key] = value
    for key in ("rank", "backend_index"):
        value = candidate.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        normalized[key] = value
    return normalized


def _nonnegative_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _anygrasp_failure(
    *,
    mode: str,
    source_rgb: str,
    source_depth: str,
    target_mask: str,
    reason: str,
    content: str,
    raw_output_ref: Any = None,
    metadata: JsonDict | None = None,
) -> ToolResult:
    return ToolResult(
        False,
        content=content,
        details={
            "tool": "anygrasp",
            "backend": "anygrasp_mcp",
            "model": "anygrasp_sdk",
            "mode": mode,
            "source_rgb": source_rgb,
            "source_depth": source_depth,
            "target_mask": target_mask or None,
            "raw_output_ref": raw_output_ref,
            "candidate_count": 0,
            "grasp_candidates": [],
            "artifacts": [],
            "reason": reason,
            "metadata": dict(metadata or {}),
        },
    )


def _scrub_anygrasp_artifacts(value: Any, *, mark_omitted: bool) -> list[JsonDict]:
    artifacts: list[JsonDict] = []
    if not isinstance(value, list):
        return artifacts
    for artifact in value:
        if not isinstance(artifact, dict):
            continue
        scrubbed = dict(artifact)
        if "base64" in scrubbed:
            scrubbed.pop("base64", None)
            if mark_omitted:
                scrubbed["base64_omitted"] = True
        artifacts.append(scrubbed)
    return artifacts


def _string_param(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _dict_or_empty(value: Any) -> JsonDict:
    return value if isinstance(value, dict) else {}


def _list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _finite_vector(value: Any, *, length: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != length:
        return None
    parsed: list[float] = []
    for item in value:
        number = _finite_float(item)
        if number is None:
            return None
        parsed.append(number)
    return parsed


def _normalise_camera_intrinsics(value: Any) -> JsonDict | None:
    if not isinstance(value, Mapping):
        return None
    normalized: JsonDict = {}
    for key in ("fx", "fy", "cx", "cy", "scale"):
        parsed = _finite_float(value.get(key))
        if parsed is None:
            return None
        normalized[key] = parsed
    if normalized["fx"] <= 0 or normalized["fy"] <= 0 or normalized["scale"] <= 0:
        return None
    return normalized


def _intrinsics_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        math.isclose(float(left[key]), float(right[key]), rel_tol=1e-9, abs_tol=1e-9)
        for key in ("fx", "fy", "cx", "cy", "scale")
    )


def _same_resolved_path(left: str, right: str) -> bool:
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def _is_rotation_matrix3(value: Any) -> bool:
    rotation = _finite_matrix3(value)
    if rotation is None:
        return False
    transpose_product = _mat3_mat3(_transpose3(rotation), rotation)
    for row in range(3):
        for col in range(3):
            expected = 1.0 if row == col else 0.0
            if not math.isclose(transpose_product[row][col], expected, rel_tol=0.0, abs_tol=1e-5):
                return False
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    return math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1e-5)


def _is_rigid_transform4(value: Any) -> bool:
    matrix = _finite_matrix4(value)
    if matrix is None:
        return False
    if any(
        not math.isclose(matrix[3][index], expected, rel_tol=0.0, abs_tol=1e-6)
        for index, expected in enumerate((0.0, 0.0, 0.0, 1.0))
    ):
        return False
    return _is_rotation_matrix3([row[:3] for row in matrix[:3]])


def _same_gripper_shape(place: Mapping[str, Any], selected: Mapping[str, Any]) -> bool:
    for key in ("score", "depth", "width", "height"):
        if not math.isclose(float(place[key]), float(selected[key]), rel_tol=1e-9, abs_tol=1e-9):
            return False
    return all(float(place[key]) >= 0 for key in ("depth", "width", "height"))


def _finite_matrix3(value: Any, *, flat_layout: str = "row_major") -> list[list[float]] | None:
    if isinstance(value, list) and len(value) == 9:
        vector = _finite_vector(value, length=9)
        if vector is None:
            return None
        if flat_layout == "column_major":
            return [
                [vector[0], vector[3], vector[6]],
                [vector[1], vector[4], vector[7]],
                [vector[2], vector[5], vector[8]],
            ]
        return [vector[0:3], vector[3:6], vector[6:9]]
    if not isinstance(value, list) or len(value) != 3:
        return None
    matrix: list[list[float]] = []
    for row in value:
        parsed = _finite_vector(row, length=3)
        if parsed is None:
            return None
        matrix.append(parsed)
    return matrix


def _parse_camera_extrinsics(value: Any) -> tuple[list[list[float]], list[float], str, str] | None:
    if isinstance(value, Mapping):
        pos = _finite_vector(value.get("pos"), length=3)
        mat = _finite_matrix3(
            value.get("mat"),
            flat_layout=_matrix_layout_from_extrinsics(value),
        )
        if pos is not None and mat is not None:
            return mat, pos, "pos_mat", "opengl"
        matrix_value = None
        source_format = "matrix4"
        for key in ("camera_to_world", "pose_mat", "matrix", "transform", "T"):
            if key in value:
                matrix_value = value.get(key)
                source_format = key
                break
    else:
        matrix_value = value
        source_format = "matrix4"
    matrix4 = _finite_matrix4(matrix_value)
    if matrix4 is None:
        return None
    rotation = [row[:3] for row in matrix4[:3]]
    translation = [matrix4[0][3], matrix4[1][3], matrix4[2][3]]
    return rotation, translation, source_format, "opencv"


def _matrix_layout_from_extrinsics(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "row_major"
    for key in ("matrix_layout", "mat_layout", "layout"):
        raw = _string_param(value.get(key)).lower().replace("-", "_").replace(" ", "_")
        if raw in {"row_major", "row"}:
            return "row_major"
        if raw in {"column_major", "col_major", "column", "col"}:
            return "column_major"
    return "row_major"


def _camera_frame_from_extrinsics(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    for key in ("camera_frame", "frame_convention", "camera_convention"):
        parsed = _string_param(value.get(key))
        if parsed:
            return parsed
    metadata = value.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("camera_frame", "frame_convention", "camera_convention"):
            parsed = _string_param(metadata.get(key))
            if parsed:
                return parsed
    return ""


def _canonical_camera_frame(value: Any, *, default: str | None = None) -> str | None:
    raw = _string_param(value).lower().replace("-", "_").replace(" ", "_")
    if not raw:
        return default
    aliases = {
        "opencv": "opencv",
        "opencv_optical": "opencv",
        "cv": "opencv",
        "pinhole": "opencv",
        "opengl": "opengl",
        "gl": "opengl",
        "mujoco": "opengl",
        "renderer": "opengl",
    }
    return aliases.get(raw)


def _convert_camera_vector(
    vector: list[float],
    *,
    source_frame: str,
    target_frame: str,
) -> list[float]:
    if source_frame == target_frame:
        return vector
    if {source_frame, target_frame} == {"opencv", "opengl"}:
        return [vector[0], -vector[1], -vector[2]]
    raise ValueError(f"unsupported camera frame conversion: {source_frame} -> {target_frame}")


def _convert_camera_rotation(
    rotation: list[list[float]],
    *,
    source_frame: str,
    target_frame: str,
) -> list[list[float]]:
    if source_frame == target_frame:
        return rotation
    if {source_frame, target_frame} != {"opencv", "opengl"}:
        raise ValueError(f"unsupported camera frame conversion: {source_frame} -> {target_frame}")
    basis = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    return _mat3_mat3(_mat3_mat3(basis, rotation), basis)


def _finite_matrix4(value: Any) -> list[list[float]] | None:
    if isinstance(value, list) and len(value) == 16:
        vector = _finite_vector(value, length=16)
        if vector is None:
            return None
        return [vector[0:4], vector[4:8], vector[8:12], vector[12:16]]
    if not isinstance(value, list) or len(value) != 4:
        return None
    matrix: list[list[float]] = []
    for row in value:
        parsed = _finite_vector(row, length=4)
        if parsed is None:
            return None
        matrix.append(parsed)
    return matrix


def _normalise_graspgenx_up_direction(value: Any) -> list[float] | None:
    vector = _finite_vector(value, length=3)
    if vector is None:
        return None
    norm = math.sqrt(sum(component * component for component in vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    return [component / norm for component in vector]


def _mat3_vec3(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [
        matrix[row][0] * vector[0] + matrix[row][1] * vector[1] + matrix[row][2] * vector[2]
        for row in range(3)
    ]


def _mat3_mat3(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [
            left[row][0] * right[0][col]
            + left[row][1] * right[1][col]
            + left[row][2] * right[2][col]
            for col in range(3)
        ]
        for row in range(3)
    ]


def _transpose3(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[row][col] for row in range(3)] for col in range(3)]


def _vec3_add(left: list[float], right: list[float]) -> list[float]:
    return [left[index] + right[index] for index in range(3)]


def _vec3_sub(left: list[float], right: list[float]) -> list[float]:
    return [left[index] - right[index] for index in range(3)]


def _round_vector(vector: list[float]) -> list[float]:
    return [0.0 if abs(value) < 1e-12 else round(value, 12) for value in vector]


def _round_matrix(matrix: list[list[float]]) -> list[list[float]]:
    return [_round_vector(row) for row in matrix]


def _camera_pose_transform_failure(
    context: ToolExecutionContext,
    *,
    reason: str,
    content: str,
    metadata: JsonDict | None = None,
) -> ToolResult:
    return make_tool_result(
        context,
        success=False,
        content=content,
        outputs={"reason": reason, "metadata": dict(metadata or {})},
        diagnostics=[{"code": reason, **dict(metadata or {})}],
    )


def _encode_image_path(image: str) -> tuple[str, str]:
    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(image)
    suffix = path.suffix.lower().lstrip(".")
    image_format = suffix or "png"
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii"), image_format


def _encode_file_payload(path_value: str) -> JsonDict:
    encoded, file_format = _encode_image_path(path_value)
    return {"format": file_format, "base64": encoded}


def _normalise_depth_prior_response(
    response: JsonDict,
    *,
    run_dir: Path,
    camera_id: str,
    source_rgb: str,
    request_ref: Path,
    raw_output_ref: Path,
) -> ToolResult:
    if not isinstance(response, dict):
        return ToolResult(
            False,
            content="Depth prior estimation failed: invalid MCP response.",
            details=_depth_prior_details(
                success=False,
                reason="invalid_mcp_response",
                camera_id=camera_id,
                source_rgb=source_rgb,
                request_ref=request_ref,
                raw_output_ref=raw_output_ref,
            ),
        )
    success_value = response.get("success")
    if success_value is False:
        reason = _string_param(response.get("reason")) or "mcp_call_failed"
        return ToolResult(
            False,
            content=_string_param(response.get("content")) or "Depth prior estimation failed.",
            details=_depth_prior_details(
                success=False,
                reason=reason,
                camera_id=camera_id,
                source_rgb=source_rgb,
                request_ref=request_ref,
                raw_output_ref=raw_output_ref,
            ),
        )
    details = response.get("details")
    if not isinstance(details, Mapping):
        details = response
    depth_path = _materialize_depth_prior_array(
        details,
        run_dir=run_dir,
        stem=f"{camera_id}-depth",
        npy_keys=("depth_npy_base64",),
        array_keys=("depth_m", "depth", "metric_depth"),
    )
    if depth_path is None:
        return ToolResult(
            False,
            content="Depth prior estimation failed: response did not contain metric depth.",
            details=_depth_prior_details(
                success=False,
                reason="missing_depth",
                camera_id=camera_id,
                source_rgb=source_rgb,
                request_ref=request_ref,
                raw_output_ref=raw_output_ref,
            ),
        )
    confidence_path = _materialize_depth_prior_array(
        details,
        run_dir=run_dir,
        stem=f"{camera_id}-confidence",
        npy_keys=("confidence_npy_base64",),
        array_keys=("confidence",),
    )
    confidence_semantics = _string_param(details.get("confidence_semantics"))
    if confidence_path is not None:
        confidence_semantics = confidence_semantics or "higher_is_better"
    else:
        confidence_path = _materialize_depth_prior_array(
            details,
            run_dir=run_dir,
            stem=f"{camera_id}-uncertainty",
            npy_keys=("uncertainty_npy_base64",),
            array_keys=("uncertainty",),
        )
        if confidence_path is not None:
            confidence_semantics = confidence_semantics or "lower_is_better"
    if confidence_path is None:
        confidence_semantics = "none"
    if confidence_semantics not in {
        "higher_is_better",
        "lower_is_better",
        "none",
    }:
        return ToolResult(
            False,
            content="Depth prior estimation failed: invalid confidence semantics.",
            details=_depth_prior_details(
                success=False,
                reason="invalid_confidence_semantics",
                camera_id=camera_id,
                source_rgb=source_rgb,
                request_ref=request_ref,
                raw_output_ref=raw_output_ref,
            ),
        )
    artifacts = [
        {
            "type": "depth_prior",
            "kind": "depth",
            "tool": "estimate_depth_prior",
            "index": camera_id,
            "path": str(depth_path),
            "camera_id": camera_id,
            "source_rgb": source_rgb,
        }
    ]
    if confidence_path is not None:
        artifacts.append(
            {
                "type": "depth_prior_confidence",
                "kind": "confidence",
                "tool": "estimate_depth_prior",
                "index": camera_id,
                "path": str(confidence_path),
                "camera_id": camera_id,
                "source_rgb": source_rgb,
                "confidence_semantics": confidence_semantics,
            }
        )
    outputs = {
        "camera_id": camera_id,
        "source_rgb": source_rgb,
        "prior_depth": str(depth_path),
        "prior_confidence": str(confidence_path) if confidence_path is not None else "",
        "prior_confidence_semantics": confidence_semantics,
        "request_ref": str(request_ref),
        "raw_output_ref": str(raw_output_ref),
        "backend": _string_param(details.get("backend")) or "depth_prior_mcp",
        "model": _string_param(details.get("model")) or "metric_depth_prior",
        "next_tool_hint": (
            "Call enhance_depth with the same rgb/depth/intrinsics and these "
            "prior_depth/prior_confidence paths."
        ),
    }
    return ToolResult(
        True,
        content="Depth prior estimation completed.",
        details={
            **_depth_prior_details(
                success=True,
                reason="",
                camera_id=camera_id,
                source_rgb=source_rgb,
                request_ref=request_ref,
                raw_output_ref=raw_output_ref,
            ),
            "outputs": outputs,
            "artifacts": artifacts,
        },
    )


def _materialize_depth_prior_array(
    value: Mapping[str, Any],
    *,
    run_dir: Path,
    stem: str,
    npy_keys: tuple[str, ...],
    array_keys: tuple[str, ...],
) -> Path | None:
    import numpy as np

    for key in npy_keys:
        encoded = value.get(key)
        if isinstance(encoded, str) and encoded.strip():
            payload = base64.b64decode(encoded, validate=True)
            array = np.load(io.BytesIO(payload), allow_pickle=False)
            if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
                raise ValueError(f"{key} must encode one numeric HxW NPY array")
            path = run_dir / f"{stem}.npy"
            np.save(path, np.asarray(array, dtype=np.float32))
            return path
    for key in array_keys:
        array_value = value.get(key)
        if array_value is not None:
            path = run_dir / f"{stem}.npy"
            np.save(path, np.asarray(array_value, dtype=np.float32))
            return path
    return None


def _depth_prior_details(
    *,
    success: bool,
    reason: str,
    camera_id: str,
    source_rgb: str,
    request_ref: Path,
    raw_output_ref: Path,
) -> JsonDict:
    details: JsonDict = {
        "tool": "estimate_depth_prior",
        "backend": "depth_prior_mcp",
        "model": "metric_depth_prior",
        "success": success,
        "camera_id": camera_id,
        "source_rgb": source_rgb,
        "request_ref": str(request_ref),
        "raw_output_ref": str(raw_output_ref),
        "artifacts": [],
    }
    if reason:
        details["reason"] = reason
    return details


def _depth_prior_failure(
    context: ToolExecutionContext,
    *,
    reason: str,
    content: str,
    metadata: JsonDict | None = None,
) -> ToolResult:
    return make_tool_result(
        context,
        success=False,
        content=content,
        outputs={"reason": reason, "metadata": dict(metadata or {})},
        diagnostics=[{"code": reason, **dict(metadata or {})}],
    )


def _scrub_depth_prior_payload(value: Any) -> Any:
    if isinstance(value, dict):
        scrubbed: JsonDict = {}
        for key, child in value.items():
            key_str = str(key)
            if key_str.endswith("_base64") or key_str in {"rgb"}:
                scrubbed[key_str] = {"base64_omitted": True}
            else:
                scrubbed[key_str] = _scrub_depth_prior_payload(child)
        return scrubbed
    if isinstance(value, list):
        if len(value) > 16:
            return {"array_omitted": True, "length": len(value)}
        return [_scrub_depth_prior_payload(item) for item in value]
    return value


def _invalid_mcp_payload(
    *,
    tool: str,
    backend: str,
    model: str,
    content: str,
    extra: JsonDict | None = None,
) -> JsonDict:
    details = {
        "tool": tool,
        "backend": backend,
        "model": model,
        "artifacts": [],
        "reason": "mcp_call_failed",
        "metadata": {},
    }
    if tool == "anygrasp":
        details.update({"candidate_count": 0, "grasp_candidates": []})
    if tool == "contact_graspnet":
        details.update(
            {
                "mode": "targeted",
                "frame": "camera",
                "camera_frame": "opencv",
                "grasp_frame": "graspnet",
                "candidate_count": 0,
                "grasp_candidates": [],
            }
        )
    if tool == "anyplace":
        details.update(
            {
                "frame": "camera",
                "camera_frame": "opencv",
                "candidate_count": 0,
                "placement_candidates": [],
            }
        )
    if tool == "molmopoint":
        details.update(
            {
                "point_count": 0,
                "points": [],
                "coordinate_convention": dict(MOLMOPOINT_COORDINATE_CONVENTION),
            }
        )
    details.update(extra or {})
    return {"success": False, "content": content, "details": details}


def _new_run_dir(output_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return output_root / f"{stamp}-{uuid4().hex[:8]}"


def _create_run_dir(output_root: Path) -> Path:
    run_dir = _new_run_dir(output_root)
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        run_dir = _new_run_dir(output_root)
        run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _image_size(path: str | Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(Path(path).expanduser()) as image:
        image.load()
        return int(image.width), int(image.height)


def _safe_extension(value: Any) -> str:
    fmt = _string_param(value).lower().lstrip(".") or "png"
    if fmt in {"jpg", "jpeg"}:
        return "jpg"
    if fmt == "png":
        return "png"
    return "bin"


def _write_base64_artifact(encoded: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(encoded, validate=True))
    return path


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _scrub_anygrasp_response(response: JsonDict) -> JsonDict:
    scrubbed = json.loads(json.dumps(response))
    details = scrubbed.get("details")
    if not isinstance(details, dict):
        return scrubbed
    details["artifacts"] = _scrub_anygrasp_artifacts(
        details.get("artifacts"),
        mark_omitted=True,
    )
    return scrubbed


def _scrub_anyplace_response(response: JsonDict) -> JsonDict:
    blocked_keys = {
        "base64",
        "pointcloud",
        "point_cloud",
        "point_clouds",
        "object_points",
        "placement_points",
    }

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: scrub(item)
                for key, item in value.items()
                if str(key).lower() not in blocked_keys
            }
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    return scrub(json.loads(json.dumps(response)))


def _scrub_contact_graspnet_payload(value: Any) -> Any:
    blocked_keys = {
        "base64",
        "pointcloud",
        "point_cloud",
        "point_clouds",
        "scene_points",
        "object_points",
        "contact_graspnet_root",
        "backend_root",
        "checkpoint_dir",
        "checkpoint_path",
    }

    def scrub(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): scrub(child)
                for key, child in item.items()
                if str(key).lower() not in blocked_keys
            }
        if isinstance(item, list):
            return [scrub(child) for child in item]
        return item

    return scrub(json.loads(json.dumps(value)))


def _scrub_graspgenx_payload(value: Any) -> Any:
    blocked_keys = {
        "base64",
        "pointcloud",
        "point_cloud",
        "point_clouds",
        "scene_points",
        "object_points",
        "graspgenx_root",
        "backend_root",
        "checkpoint_root",
        "checkpoint_dir",
        "checkpoint_path",
        "gripper_descriptions_root",
        "model_path",
        "python_path",
    }

    def scrub(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): scrub(child)
                for key, child in item.items()
                if str(key).lower() not in blocked_keys
            }
        if isinstance(item, list):
            return [scrub(child) for child in item]
        if isinstance(item, tuple):
            return [scrub(child) for child in item]
        return item

    return scrub(json.loads(json.dumps(value)))


def _scrub_sam3_response(
    response: JsonDict,
    *,
    detections: list[JsonDict],
    artifacts: list[JsonDict],
) -> JsonDict:
    def scrub(value: Any) -> Any:
        if isinstance(value, Mapping):
            cleaned: JsonDict = {}
            for key, child in value.items():
                key_text = str(key)
                key_lower = key_text.lower()
                if "base64" in key_lower:
                    marker = (
                        key_text
                        if key_lower.endswith("_omitted")
                        else "base64_omitted"
                        if key_lower == "base64"
                        else f"{key_text}_omitted"
                    )
                    cleaned[marker] = True
                    continue
                cleaned[key_text] = scrub(child)
            return cleaned
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    scrubbed = scrub(json.loads(json.dumps(response)))
    details = scrubbed.get("details")
    if not isinstance(details, dict):
        return scrubbed
    raw_detections = details.get("detections", [])
    if isinstance(raw_detections, list):
        for idx, detection in enumerate(raw_detections):
            if not isinstance(detection, dict):
                continue
            mask = detection.get("mask")
            if isinstance(mask, dict) and mask.get("base64_omitted") is True:
                raw_backend_index = _sam3_backend_index(detection, fallback=idx)
                matches = [
                    item
                    for item in detections
                    if item.get("backend_index") == raw_backend_index
                ]
                if len(matches) == 1:
                    mask["artifact_ref"] = matches[0].get("mask_ref")
    raw_artifacts = details.get("artifacts", [])
    if isinstance(raw_artifacts, list):
        for idx, artifact in enumerate(raw_artifacts):
            if (
                not isinstance(artifact, dict)
                or artifact.get("base64_omitted") is not True
            ):
                continue
            if idx < len(artifacts):
                artifact["artifact_ref"] = artifacts[idx].get("artifact_ref")
    return scrubbed
