"""Host-owned preparation for articulated-handle attachment probes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from adapter.protocol import JsonDict
from agent.backends.planner import PlannerBackend, PlannerBackendRequest
from agent.tools.registry import ToolExecutionContext, ToolHandler, make_tool_result


ARTICULATED_ATTACHMENT_PROBE_SCHEMA = "openeta.articulated_attachment_probe.v1"
ARTICULATED_ATTACHMENT_PROBE_DISTANCE_M = 0.05
ARTICULATED_ATTACHMENT_PROBE_DISTANCE_TOLERANCE_M = 0.002
ARTICULATED_ATTACHMENT_PROBE_MAX_WAYPOINTS = 5
ARTICULATED_ATTACHMENT_PROBE_MAX_SEGMENT_M = 0.015
ARTICULATED_ATTACHMENT_PROBE_MAX_REASON_CHARS = 1024
ARTICULATED_ATTACHMENT_ASSESSMENT_SCHEMA = (
    "openeta.articulated_attachment_assessment.v1"
)

ARTICULATED_ATTACHMENT_ASSESSMENT_PROMPT = """You are an independent attachment reviewer.
The robot closed on an articulated handle and executed one host-frozen 5 cm probe.
Compare the ordered before/after agentview and wrist images. Return PASS only when
the same target handle or articulated body visibly co-moved along the probe path and
remains engaged by the gripper. Return FAIL only when direct evidence shows the handle
stayed behind, moved inconsistently, or separated from the gripper. Return UNKNOWN for
occlusion, conflicting views, identity ambiguity, or insufficient motion evidence.
Do not infer PASS from controller success, gripper closure, or reward. Return exactly:
{"verdict":"PASS|FAIL|UNKNOWN","reason":"concise visual evidence"}
"""


class AttachmentProbeError(ValueError):
    """Raised when an articulated attachment-probe proposal is invalid."""


def build_prepare_attachment_probe_handler() -> ToolHandler:
    """Build the read-only articulated probe compiler."""

    def handler(context: ToolExecutionContext):
        try:
            outputs = prepare_attachment_probe(
                context.parameters,
                observation=context.observation,
                supervision_context=context.metadata.get("supervision_context"),
            )
        except AttachmentProbeError as exc:
            return make_tool_result(
                context,
                success=False,
                content=f"articulated attachment probe rejected: {exc}",
                outputs={
                    "reason": "articulated_attachment_probe_rejected",
                    "checked_by": "host_probe_geometry",
                },
                diagnostics=[
                    {
                        "code": "articulated_attachment_probe_rejected",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            )
        return make_tool_result(
            context,
            success=True,
            content="articulated attachment probe prepared and frozen",
            outputs=outputs,
        )

    return handler


def build_assess_attachment_probe_handler(backend: PlannerBackend) -> ToolHandler:
    """Build the independent before/after articulated attachment reviewer."""

    def handler(context: ToolExecutionContext):
        try:
            outputs = assess_attachment_probe(
                context,
                backend=backend,
            )
        except (AttachmentProbeError, ValueError) as exc:
            return make_tool_result(
                context,
                success=False,
                content=f"articulated attachment assessment failed: {exc}",
                outputs={
                    "reason": "articulated_attachment_assessment_failed",
                    "checked_by": "independent_attachment_reviewer",
                },
                diagnostics=[
                    {
                        "code": "articulated_attachment_assessment_failed",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            )
        return make_tool_result(
            context,
            success=True,
            content=f"articulated attachment assessment: {outputs['verdict']}",
            outputs=outputs,
        )

    return handler


def assess_attachment_probe(
    context: ToolExecutionContext,
    *,
    backend: PlannerBackend,
) -> JsonDict:
    """Assess articulated co-motion from the frozen probe's before/after views."""

    memory = _memory_context(context.metadata.get("supervision_context"))
    execution = _mapping(memory.get("grasp_execution"), "grasp_execution")
    gate = _mapping(memory.get("attachment_gate"), "attachment_gate")
    probe = _mapping(
        memory.get("articulated_attachment_probe"),
        "articulated_attachment_probe",
    )
    if (
        execution.get("status") != "required"
        or execution.get("stage") != "attachment"
        or execution.get("attachment_mode") != "articulated_handle"
        or probe.get("status") != "completed"
    ):
        raise AttachmentProbeError("no completed articulated probe is awaiting assessment")
    if str(gate.get("verdict") or "UNKNOWN").upper() != "UNKNOWN":
        raise AttachmentProbeError("the articulated attachment gate is already resolved")
    assessment_count = int(gate.get("assessment_count") or 0)
    refresh_completed = gate.get("unknown_refresh_completed") is True
    if assessment_count >= 2:
        raise AttachmentProbeError(
            "the articulated attachment assessment budget is exhausted"
        )
    if assessment_count >= 1 and not refresh_completed:
        raise AttachmentProbeError("one fresh observation is required before reassessment")
    before = [
        path
        for path in probe.get("pre_probe_image_paths", [])
        if isinstance(path, str) and path
    ]
    after = _current_rgb_paths(context.observation)
    if len(before) != 2 or len(after) != 2:
        raise AttachmentProbeError(
            "exactly one agentview and one wrist RGB image are required before and after"
        )
    paths = [*before, *after]
    result = backend.decide(
        PlannerBackendRequest(
            system_prompt=ARTICULATED_ATTACHMENT_ASSESSMENT_PROMPT,
            tool_context={
                "schema_version": ARTICULATED_ATTACHMENT_ASSESSMENT_SCHEMA,
                "role": "independent_articulated_attachment_reviewer",
                "task": str(context.metadata.get("task") or ""),
                "candidate_id": probe.get("candidate_id"),
                "motion_type": probe.get("motion_type"),
                "distance_m": probe.get("distance_m"),
                "direction_world_xyz": probe.get("direction_world_xyz"),
                "image_order": [
                    {"image_number": 1, "role": "before_agentview"},
                    {"image_number": 2, "role": "before_wrist"},
                    {"image_number": 3, "role": "after_agentview"},
                    {"image_number": 4, "role": "after_wrist"},
                ],
                "vision_image_paths": paths,
            },
            metadata={"isolated_context": True},
        )
    )
    payload = result.payload
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("reviewer returned invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("reviewer must return one JSON object")
    verdict = str(payload.get("verdict") or "").strip().upper()
    if verdict not in {"PASS", "FAIL", "UNKNOWN"}:
        raise ValueError("reviewer returned an invalid verdict")
    reason = str(payload.get("reason") or "").strip()
    return {
        "schema_version": ARTICULATED_ATTACHMENT_ASSESSMENT_SCHEMA,
        "candidate_id": probe.get("candidate_id"),
        "scene_epoch": memory.get("scene_epoch"),
        "verdict": verdict,
        "reason": reason,
        "assessment_index": assessment_count + 1,
        "checked_by": "independent_attachment_reviewer",
        "provider": result.provider,
        "model": result.model,
    }


def prepare_attachment_probe(
    parameters: Mapping[str, Any],
    *,
    observation: Any,
    supervision_context: object,
) -> JsonDict:
    """Validate an agent proposal and freeze one bounded 5 cm probe action."""

    memory = _memory_context(supervision_context)
    execution = _mapping(memory.get("grasp_execution"), "grasp_execution")
    policy = _mapping(memory.get("grasp_candidate_policy"), "grasp_candidate_policy")
    if execution.get("status") != "required" or execution.get("stage") != "prepare_probe":
        raise AttachmentProbeError(
            "prepare_attachment_probe is allowed only after an articulated handle close"
        )
    if policy.get("interaction_family") != "articulated_handle":
        raise AttachmentProbeError("the active candidate is not an articulated handle")
    candidate_id = str(execution.get("candidate_id") or "")
    compiled_grasp_id = str(execution.get("compiled_grasp_id") or "")
    if not candidate_id or not compiled_grasp_id:
        raise AttachmentProbeError("active candidate provenance is incomplete")
    active = _mapping(policy.get("active_candidate"), "active_candidate")
    if str(active.get("id") or "") != candidate_id:
        raise AttachmentProbeError("active candidate does not match grasp execution")
    scene_epoch = _nonnegative_int(memory.get("scene_epoch"), "scene_epoch")
    if _nonnegative_int(execution.get("scene_epoch"), "grasp_execution.scene_epoch") != scene_epoch:
        raise AttachmentProbeError("grasp execution is stale for the current scene epoch")
    if observation is None:
        raise AttachmentProbeError("a current observation is required")
    pose = getattr(getattr(observation, "robot", None), "end_effector_pose", None)
    pose = _mapping(pose, "observation.robot.end_effector_pose")
    start_xyz = _vector3(pose.get("xyz"), "observation.robot.end_effector_pose.xyz")
    rotation = _optional_rotation(pose)
    motion_type = str(parameters.get("motion_type") or "").strip().lower()
    reason = str(parameters.get("reason") or "").strip()
    if len(reason) > ARTICULATED_ATTACHMENT_PROBE_MAX_REASON_CHARS:
        raise AttachmentProbeError("reason is too long")
    if motion_type == "linear":
        direction = _normalise(
            _vector3(parameters.get("direction_world_xyz"), "direction_world_xyz")
        )
        endpoint = [
            start_xyz[index] + ARTICULATED_ATTACHMENT_PROBE_DISTANCE_M * direction[index]
            for index in range(3)
        ]
        target_pose = _world_pose(
            endpoint,
            rotation=rotation,
            candidate_id=candidate_id,
            compiled_grasp_id=compiled_grasp_id,
            scene_epoch=scene_epoch,
        )
        frozen_path = [target_pose]
        tool_name = "move_to"
        tool_parameters: JsonDict = {
            "target_pose": target_pose,
            "enable_collision_check": True,
        }
    elif motion_type == "arc":
        offsets_value = parameters.get("waypoint_offsets_world_xyz")
        if not isinstance(offsets_value, Sequence) or isinstance(offsets_value, (str, bytes)):
            raise AttachmentProbeError("waypoint_offsets_world_xyz must be a list")
        if not 2 <= len(offsets_value) <= ARTICULATED_ATTACHMENT_PROBE_MAX_WAYPOINTS:
            raise AttachmentProbeError("arc probes require between 2 and 5 waypoints")
        offsets = [
            _vector3(value, f"waypoint_offsets_world_xyz[{index}]")
            for index, value in enumerate(offsets_value)
        ]
        absolute = [
            [start_xyz[index] + offset[index] for index in range(3)] for offset in offsets
        ]
        length = _path_length([start_xyz, *absolute])
        if abs(length - ARTICULATED_ATTACHMENT_PROBE_DISTANCE_M) > (
            ARTICULATED_ATTACHMENT_PROBE_DISTANCE_TOLERANCE_M
        ):
            raise AttachmentProbeError(
                "arc probe path length must be 0.05 m within 0.002 m"
            )
        for segment_index, (left, right) in enumerate(zip([start_xyz, *absolute], absolute)):
            if _distance(left, right) > ARTICULATED_ATTACHMENT_PROBE_MAX_SEGMENT_M + 1e-9:
                raise AttachmentProbeError(
                    f"arc probe segment {segment_index} exceeds 0.015 m"
                )
        frozen_path = [
            _world_pose(
                point,
                rotation=rotation,
                candidate_id=candidate_id,
                compiled_grasp_id=compiled_grasp_id,
                scene_epoch=scene_epoch,
                waypoint_index=index,
            )
            for index, point in enumerate(absolute)
        ]
        direction = _normalise(
            [absolute[-1][index] - start_xyz[index] for index in range(3)]
        )
        tool_name = "follow_eef_trajectory"
        tool_parameters = {
            "trajectory": frozen_path,
            "enable_collision_check": True,
        }
    else:
        raise AttachmentProbeError("motion_type must be 'linear' or 'arc'")
    frozen_payload = {
        "candidate_id": candidate_id,
        "compiled_grasp_id": compiled_grasp_id,
        "scene_epoch": scene_epoch,
        "motion_type": motion_type,
        "start_eef_xyz": _round_vector(start_xyz),
        "path": frozen_path,
    }
    path_sha256 = hashlib.sha256(
        json.dumps(frozen_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _stamp_probe_metadata(tool_parameters, path_sha256=path_sha256)
    pre_probe_images = _current_rgb_paths(observation)
    if len(pre_probe_images) != 2:
        raise AttachmentProbeError(
            "prepare_attachment_probe requires current agentview and wrist RGB images"
        )
    return {
        "schema_version": ARTICULATED_ATTACHMENT_PROBE_SCHEMA,
        "status": "prepared",
        "candidate_id": candidate_id,
        "compiled_grasp_id": compiled_grasp_id,
        "scene_epoch": scene_epoch,
        "interaction_family": "articulated_handle",
        "motion_type": motion_type,
        "distance_m": ARTICULATED_ATTACHMENT_PROBE_DISTANCE_M,
        "start_eef_xyz": _round_vector(start_xyz),
        "direction_world_xyz": _round_vector(direction),
        "frozen_path": frozen_path,
        "path_sha256": path_sha256,
        "required_action": {"name": tool_name, "parameters": tool_parameters},
        "pre_probe_image_paths": pre_probe_images,
        "proposal_reason": reason,
        "checked_by": "host_probe_geometry",
    }


def _memory_context(value: object) -> JsonDict:
    context = value if isinstance(value, Mapping) else {}
    memory = context.get("memory") if isinstance(context, Mapping) else None
    if not isinstance(memory, Mapping):
        raise AttachmentProbeError("host supervision memory is unavailable")
    return dict(memory)


def _mapping(value: object, field: str) -> JsonDict:
    if not isinstance(value, Mapping):
        raise AttachmentProbeError(f"{field} must be an object")
    return dict(value)


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise AttachmentProbeError(f"{field} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise AttachmentProbeError(f"{field} must be a non-negative integer") from exc
    if parsed < 0:
        raise AttachmentProbeError(f"{field} must be a non-negative integer")
    return parsed


def _vector3(value: object, field: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise AttachmentProbeError(f"{field} must contain three finite numbers")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise AttachmentProbeError(f"{field} must contain three finite numbers")
        try:
            parsed = float(item)
        except (TypeError, ValueError) as exc:
            raise AttachmentProbeError(f"{field} must contain three finite numbers") from exc
        if not math.isfinite(parsed):
            raise AttachmentProbeError(f"{field} must contain three finite numbers")
        result.append(parsed)
    return result


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-9:
        raise AttachmentProbeError("probe direction must be non-zero")
    return [value / norm for value in vector]


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((right[index] - left[index]) ** 2 for index in range(3)))


def _path_length(points: list[list[float]]) -> float:
    return sum(_distance(left, right) for left, right in zip(points, points[1:]))


def _optional_rotation(pose: Mapping[str, Any]) -> JsonDict:
    for key in ("rotation_matrix", "euler_xyz_deg", "quat_xyzw"):
        value = pose.get(key)
        if value is not None:
            if key == "rotation_matrix":
                if (
                    not isinstance(value, Sequence)
                    or isinstance(value, (str, bytes))
                    or len(value) != 3
                ):
                    raise AttachmentProbeError("rotation_matrix must be a finite 3x3 matrix")
                rows = [
                    _vector3(row, f"observation.robot.end_effector_pose.{key}[{index}]")
                    for index, row in enumerate(value)
                ]
                return {key: rows}
            expected = 3 if key == "euler_xyz_deg" else 4
            if (
                not isinstance(value, Sequence)
                or isinstance(value, (str, bytes))
                or len(value) != expected
            ):
                raise AttachmentProbeError(f"{key} must contain {expected} finite numbers")
            parsed: list[float] = []
            for item in value:
                if isinstance(item, bool):
                    raise AttachmentProbeError(
                        f"{key} must contain {expected} finite numbers"
                    )
                try:
                    number = float(item)
                except (TypeError, ValueError) as exc:
                    raise AttachmentProbeError(
                        f"{key} must contain {expected} finite numbers"
                    ) from exc
                if not math.isfinite(number):
                    raise AttachmentProbeError(
                        f"{key} must contain {expected} finite numbers"
                    )
                parsed.append(number)
            if key == "quat_xyzw":
                norm = math.sqrt(sum(number * number for number in parsed))
                if norm <= 1e-9:
                    raise AttachmentProbeError("quat_xyzw must be non-zero")
                parsed = [number / norm for number in parsed]
            return {key: parsed}
    return {}


def _world_pose(
    xyz: list[float],
    *,
    rotation: Mapping[str, Any],
    candidate_id: str,
    compiled_grasp_id: str,
    scene_epoch: int,
    waypoint_index: int | None = None,
) -> JsonDict:
    pose: JsonDict = {
        "frame": "world",
        "xyz": _round_vector(xyz),
        **dict(rotation),
        "probe_type": "articulated_attachment",
        "source_grasp_id": candidate_id,
        "compiled_grasp_id": compiled_grasp_id,
        "scene_epoch": scene_epoch,
    }
    if waypoint_index is not None:
        pose["waypoint_index"] = waypoint_index
    return pose


def _round_vector(value: Sequence[float]) -> list[float]:
    return [round(float(item), 9) for item in value]


def _stamp_probe_metadata(parameters: JsonDict, *, path_sha256: str) -> None:
    if isinstance(parameters.get("target_pose"), dict):
        parameters["target_pose"]["probe_path_sha256"] = path_sha256
    trajectory = parameters.get("trajectory")
    if isinstance(trajectory, list):
        for pose in trajectory:
            if isinstance(pose, dict):
                pose["probe_path_sha256"] = path_sha256


def _current_rgb_paths(observation: Any) -> list[str]:
    metadata = getattr(observation, "metadata", None)
    artifacts = metadata.get("image_artifacts") if isinstance(metadata, Mapping) else None
    if not isinstance(artifacts, list):
        return []
    preferred = {"agentview": 0, "wrist": 1}
    ranked: list[tuple[int, int, str]] = []
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping) or artifact.get("kind") != "rgb":
            continue
        frame_id = str(artifact.get("frame_id") or "")
        path = artifact.get("path")
        if frame_id not in preferred or not isinstance(path, str) or not path:
            continue
        ranked.append((preferred[frame_id], index, path))
    ranked.sort()
    selected: dict[int, str] = {}
    for rank, _, path in ranked:
        selected.setdefault(rank, path)
    return [selected[rank] for rank in sorted(selected)]
