"""Reusable tool handlers for OpenETA runtime tests, CLI, and MCP-backed tools."""

from __future__ import annotations

import asyncio
import base64
import json
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from adapter.protocol import JsonDict
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
AnyGraspDetectCallable = Callable[[JsonDict], JsonDict]
ContactGraspNetPredictCallable = Callable[[JsonDict], JsonDict]
AnyPlacePredictCallable = Callable[[JsonDict], JsonDict]
DEFAULT_SAM3_IMAGE_OUTPUT_ROOT = Path("tmp") / "image" / "sam3"
DEFAULT_SAM3_RESULT_OUTPUT_ROOT = Path("tmp") / "tool_result" / "sam3"
DEFAULT_ANYGRASP_OUTPUT_ROOT = Path("tmp") / "tool_result" / "anygrasp"
DEFAULT_CONTACT_GRASPNET_OUTPUT_ROOT = Path("tmp") / "tool_result" / "contact_graspnet"
DEFAULT_SAM3_SELECTION_VISUAL_LIMIT = 8
DEFAULT_ANYPLACE_OUTPUT_ROOT = Path("tmp") / "tool_result" / "anyplace"

CONTACT_GRASPNET_MODEL = "contact_graspnet_pytorch_unofficial"
CONTACT_GRASPNET_GRIPPER_DEPTH = 0.1034
CONTACT_GRASPNET_MAX_CANDIDATES = 20


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
    segment_points: Sam3SegmentCallable | None = None,
    output_root: str | Path | None = None,
    result_output_root: str | Path | None = None,
) -> ToolHandler:
    """Build a SAM3 ToolRegistry handler backed by an injected segment callable."""

    image_output_root = Path(output_root) if output_root is not None else DEFAULT_SAM3_IMAGE_OUTPUT_ROOT
    json_output_root = (
        Path(result_output_root)
        if result_output_root is not None
        else DEFAULT_SAM3_RESULT_OUTPUT_ROOT
        if output_root is None
        else image_output_root
    )

    def handler(context: ToolExecutionContext) -> ToolResult:
        image = _string_param(context.parameters.get("image"))
        prompt = _string_param(context.parameters.get("prompt"))
        box_xyxy = context.parameters.get("box_prompt_xyxy")
        point_prompts = context.parameters.get("point_prompts")
        if not image:
            return _sam3_failure(
                prompt=prompt,
                source_image=image,
                reason="missing_image",
                content="SAM3 segmentation failed: missing image.",
            )
        if not prompt and box_xyxy is None and point_prompts is None:
            return _sam3_failure(
                prompt=prompt,
                source_image=image,
                reason="missing_prompt",
                content="SAM3 segmentation failed: missing prompt.",
            )

        try:
            image_base64, image_format = _encode_image_path(image)
        except FileNotFoundError:
            return _sam3_failure(
                prompt=prompt,
                source_image=image,
                reason="image_not_found",
                content="SAM3 segmentation failed: image not found.",
            )
        except Exception as exc:  # noqa: BLE001 - image IO must stay structured.
            return _sam3_failure(
                prompt=prompt,
                source_image=image,
                reason="image_encode_failed",
                content=f"SAM3 segmentation failed: image encode failed: {exc}",
                metadata={"error_type": type(exc).__name__},
            )

        try:
            wire_request = {
                "image_base64": image_base64,
                "image_format": image_format,
                "prompt": prompt,
            }
            selected_segment = segment
            if point_prompts is not None:
                if segment_points is None:
                    return _sam3_failure(
                        prompt=prompt,
                        source_image=image,
                        reason="geometric_prompt_unavailable",
                        content="SAM3 segmentation failed: point-prompt service is unavailable.",
                    )
                wire_request = {
                    "image_base64": image_base64,
                    "image_format": image_format,
                    "points": point_prompts,
                }
                selected_segment = segment_points
            elif box_xyxy is not None:
                wire_request["box_xyxy"] = box_xyxy
            response = selected_segment(wire_request)
        except Exception as exc:  # noqa: BLE001 - tool failures must stay structured.
            return _sam3_failure(
                prompt=prompt,
                source_image=image,
                reason="mcp_call_failed",
                content=f"SAM3 segmentation failed: MCP call failed: {exc}",
                metadata={"error_type": type(exc).__name__},
            )
        return _normalise_sam3_response(
            response,
            prompt=prompt,
            source_image=image,
            request={
                "image": image,
                "image_format": image_format,
                "prompt": prompt,
                **({"box_xyxy": box_xyxy} if box_xyxy is not None else {}),
                **({"points": point_prompts} if point_prompts is not None else {}),
            },
            image_output_root=image_output_root,
            result_output_root=json_output_root,
        )

    return handler


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


def build_anygrasp_handler(
    detect_grasps: AnyGraspDetectCallable,
    *,
    output_root: str | Path = DEFAULT_ANYGRASP_OUTPUT_ROOT,
) -> ToolHandler:
    """Build an AnyGrasp ToolRegistry handler backed by an injected MCP callable."""

    def handler(context: ToolExecutionContext) -> ToolResult:
        mode = _string_param(context.parameters.get("mode")) or "targeted"
        rgb = _string_param(context.parameters.get("rgb"))
        depth = _string_param(context.parameters.get("depth"))
        target_mask = _string_param(context.parameters.get("target_mask"))
        intrinsics = context.parameters.get("intrinsics")
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
        }
        mcp_request = {
            **request,
            "rgb": rgb_payload,
            "depth": depth_payload,
            "target_mask": target_mask_payload,
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
        return _normalise_anygrasp_response(
            response,
            mode=mode,
            source_rgb=rgb,
            source_depth=depth,
            target_mask=target_mask,
            request=request,
            output_root=Path(output_root),
        )

    return handler


def build_contact_graspnet_handler(
    predict_grasps: ContactGraspNetPredictCallable,
    *,
    output_root: str | Path = DEFAULT_CONTACT_GRASPNET_OUTPUT_ROOT,
) -> ToolHandler:
    """Build a targeted Contact-GraspNet handler backed by an MCP callable."""

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


def build_anyplace_handler(
    predict_placement: AnyPlacePredictCallable,
    *,
    output_root: str | Path = DEFAULT_ANYPLACE_OUTPUT_ROOT,
) -> ToolHandler:
    """Build an AnyPlace ToolRegistry handler backed by an injected MCP callable."""

    def handler(context: ToolExecutionContext) -> ToolResult:
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
        if not source_rgb or not source_depth or not source_object_mask:
            return _anyplace_failure(
                "selected_grasp_requires_targeted_source",
                "AnyPlace placement prediction failed: selected grasp source is incomplete.",
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
                    "mode": "targeted",
                    "rgb": source_rgb,
                    "depth": source_depth,
                    "object_mask": source_object_mask,
                    "intrinsics": source_intrinsics,
                },
            },
        }
        request = _scrub_anyplace_response(request)
        mcp_request: JsonDict = {
            **payloads,
            "intrinsics": intrinsics,
            "selected_grasp": normalized_candidate,
        }
        try:
            response = predict_placement(mcp_request)
        except Exception as exc:  # noqa: BLE001 - tool failures must stay structured.
            return _anyplace_failure(
                "mcp_call_failed",
                f"AnyPlace placement prediction failed: MCP call failed: {exc}",
                metadata={"error_type": type(exc).__name__},
            )
        return _normalise_anyplace_response(
            response,
            selected_grasp=normalized_candidate,
            request=request,
            output_root=Path(output_root),
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
            translation
            if source_format == "pos_mat"
            else _vec3_sub(translation, camera_position)
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


def _normalise_sam3_response(
    response: JsonDict,
    *,
    prompt: str,
    source_image: str,
    request: JsonDict,
    image_output_root: Path,
    result_output_root: Path,
) -> ToolResult:
    if not isinstance(response, dict):
        return _sam3_failure(
            prompt=prompt,
            source_image=source_image,
            reason="mcp_call_failed",
            content="SAM3 segmentation failed: invalid MCP response.",
        )
    details = response.get("details")
    success = bool(response.get("success", False))
    if success and not isinstance(details, dict):
        return _sam3_failure(
            prompt=prompt,
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
            prompt=prompt,
            source_image=source_image,
            reason=reason,
            content=content,
            raw_output_ref=details.get("raw_output_ref"),
            metadata=_dict_or_empty(details.get("metadata")),
        )

    if "detections" not in details or "detection_count" not in details:
        return _sam3_failure(
            prompt=prompt,
            source_image=source_image,
            reason="inconsistent_detection_outputs",
            content="SAM3 returned inconsistent detection outputs.",
            raw_output_ref=details.get("raw_output_ref"),
            metadata=_dict_or_empty(details.get("metadata")),
        )

    detections_value = details.get("detections")
    if not isinstance(detections_value, list):
        return _sam3_failure(
            prompt=prompt,
            source_image=source_image,
            reason="inconsistent_detection_outputs",
            content="SAM3 returned inconsistent detection outputs.",
            raw_output_ref=details.get("raw_output_ref"),
            metadata=_dict_or_empty(details.get("metadata")),
        )

    run_dir = _new_run_dir(result_output_root)
    artifacts_dir = image_output_root / run_dir.name
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        run_dir = _new_run_dir(result_output_root)
        artifacts_dir = image_output_root / run_dir.name
        run_dir.mkdir(parents=True, exist_ok=False)
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
                prompt=prompt,
                source_image=source_image,
                reason="inconsistent_detection_outputs",
                content="SAM3 returned inconsistent detection outputs.",
                raw_output_ref=details.get("raw_output_ref"),
                metadata=_dict_or_empty(details.get("metadata")),
            )
        mask = detection.get("mask")
        if not isinstance(mask, dict) or not mask.get("base64"):
            return _sam3_failure(
                prompt=prompt,
                source_image=source_image,
                reason="inconsistent_detection_outputs",
                content="SAM3 returned inconsistent detection outputs.",
                metadata=_dict_or_empty(details.get("metadata")),
            )
        try:
            mask_ref = _write_base64_artifact(
                mask["base64"],
                artifacts_dir / f"mask_{rank:03d}.{_safe_extension(mask.get('format'))}",
            )
        except Exception as exc:  # noqa: BLE001
            return _sam3_failure(
                prompt=prompt,
                source_image=source_image,
                reason="artifact_write_failed",
                content=f"SAM3 segmentation failed: artifact write failed: {exc}",
                metadata={"error_type": type(exc).__name__},
            )
        label = (
            prompt
            if request.get("points") is not None and prompt
            else _string_param(detection.get("label")) or prompt
        )
        score = _finite_float(detection.get("score"))
        bbox_xyxy = detection.get("bbox_xyxy")
        area_px = detection.get("area_px")
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
                "prompt": prompt,
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
            prompt=prompt,
            source_image=source_image,
            reason="inconsistent_detection_outputs",
            content="SAM3 returned inconsistent detection outputs.",
            raw_output_ref=details.get("raw_output_ref"),
            metadata=_dict_or_empty(details.get("metadata")),
        )

    artifacts_value = details.get("artifacts", [])
    if not isinstance(artifacts_value, list):
        artifacts_value = []
    artifacts: list[JsonDict] = []
    for idx, artifact in enumerate(artifacts_value):
        if not isinstance(artifact, dict) or not artifact.get("base64"):
            continue
        artifact_type = _string_param(artifact.get("artifact_type")) or f"artifact_{idx:03d}"
        fmt = _safe_extension(artifact.get("format"))
        try:
            artifact_ref = _write_base64_artifact(
                artifact["base64"],
                artifacts_dir / f"{artifact_type}.{fmt}",
            )
        except Exception as exc:  # noqa: BLE001
            return _sam3_failure(
                prompt=prompt,
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
                prompt=prompt,
            )
            artifacts.extend(selection_artifacts)
        except Exception as exc:  # noqa: BLE001 - selection visuals are best effort.
            visualization_diagnostics.append(
                {
                    "code": "sam3_selection_visualization_failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    raw_output_ref = run_dir / "response.raw.json"
    scrubbed_response = _scrub_sam3_response(response, detections=detections, artifacts=artifacts)
    _write_json(raw_output_ref, scrubbed_response)
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
            "prompt": prompt,
            "source_image": source_image,
            "raw_output_ref": str(raw_output_ref),
            "result_id": run_dir.name,
            "detection_count": len(detections),
            "detections": detections,
            "ranking": "score_descending",
            "selection_required": len(detections) > 1,
            "selected_detection": detections[0] if len(detections) == 1 else None,
            "selection_bundle": selection_bundle,
            "artifacts": artifacts,
            "diagnostics": visualization_diagnostics,
            "metadata": _dict_or_empty(details.get("metadata")),
        },
    )
    _write_json(
        run_dir / "tool_result.json",
        {"success": result.success, "content": result.content, "details": result.details},
    )
    return result


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
    prompt: str,
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
            "prompt": prompt,
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
        content = _string_param(response.get("content")) or f"AnyGrasp grasp detection failed: {reason}."
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
            "mode": "targeted",
            "rgb": source_rgb,
            "depth": source_depth,
            "object_mask": target_mask,
            "intrinsics": normalized_intrinsics,
        }

    run_dir = _new_run_dir(output_root)
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        run_dir = _new_run_dir(output_root)
        run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "request.json", request)
    raw_output_ref = run_dir / "response.raw.json"
    canonical_candidates_ref = run_dir / "grasp_candidates.canonical.json"
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
        "canonical_grasp_candidates_ref": str(canonical_candidates_ref),
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
    _write_json(
        canonical_candidates_ref,
        {
            "schema_version": "openeta.canonical_grasp_candidates.v1",
            "result_id": run_dir.name,
            "ranking": "score_descending",
            "candidate_count": len(candidates),
            "grasp_candidates": candidates,
        },
    )
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
            metadata=_scrub_contact_graspnet_payload(
                _dict_or_empty(details.get("metadata"))
            ),
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
    expected_tip = [
        translation[row] + gripper_depth * rotation[row][0] for row in range(3)
    ]
    if any(
        not math.isclose(tip[row], expected_tip[row], rel_tol=0.0, abs_tol=1e-5)
        for row in range(3)
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
        normalized = _normalise_anyplace_candidate(candidate, selected_grasp=selected_grasp)
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


def _normalise_anyplace_candidate(
    candidate: Any,
    *,
    selected_grasp: JsonDict,
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
    return {
        "id": candidate_id,
        "source_grasp_id": selected_grasp["id"],
        "object_placement_transform": {
            "frame": "camera",
            "camera_frame": "opencv",
            "convention": "p_placed = R @ p_current + t",
            "transform_matrix": transform,
        },
        "place_grasp_pose": place,
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
    return {
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
            if not math.isclose(
                transpose_product[row][col], expected, rel_tol=0.0, abs_tol=1e-5
            ):
                return False
    determinant = (
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
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
        if not math.isclose(
            float(place[key]), float(selected[key]), rel_tol=1e-9, abs_tol=1e-9
        ):
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
    # AnyGrasp's rotation columns are the gripper's local axes expressed in
    # camera coordinates.  Changing the camera coordinate basis therefore
    # transforms each column exactly like a vector: B @ R.  Right-multiplying
    # by B would also relabel the gripper-local axes and add an unintended
    # 180-degree roll around the approach axis.
    return _mat3_mat3(basis, rotation)


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


def _scrub_sam3_response(
    response: JsonDict,
    *,
    detections: list[JsonDict],
    artifacts: list[JsonDict],
) -> JsonDict:
    scrubbed = json.loads(json.dumps(response))
    details = scrubbed.get("details")
    if not isinstance(details, dict):
        return scrubbed
    raw_detections = details.get("detections", [])
    if isinstance(raw_detections, list):
        for idx, detection in enumerate(raw_detections):
            if not isinstance(detection, dict):
                continue
            mask = detection.get("mask")
            if isinstance(mask, dict) and "base64" in mask:
                mask.pop("base64", None)
                mask["base64_omitted"] = True
                if idx < len(detections):
                    mask["artifact_ref"] = detections[idx].get("mask_ref")
    raw_artifacts = details.get("artifacts", [])
    if isinstance(raw_artifacts, list):
        for idx, artifact in enumerate(raw_artifacts):
            if not isinstance(artifact, dict) or "base64" not in artifact:
                continue
            artifact.pop("base64", None)
            artifact["base64_omitted"] = True
            if idx < len(artifacts):
                artifact["artifact_ref"] = artifacts[idx].get("artifact_ref")
    return scrubbed
