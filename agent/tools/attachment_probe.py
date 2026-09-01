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

ROLE_AWARE_ARTICULATED_ATTACHMENT_ASSESSMENT_PROMPT = (
    ARTICULATED_ATTACHMENT_ASSESSMENT_PROMPT.replace(
        "agentview and wrist images",
        "scene-primary and wrist-primary images",
    )
)


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
            content=(
                "articulated attachment probe geometry frozen; run the returned "
                "ordered IK preview requests, then execute by receipt id(s)"
            ),
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
    probe = _mapping(
        memory.get("articulated_attachment_probe"),
        "articulated_attachment_probe",
    )
    if probe.get("status") != "completed":
        raise AttachmentProbeError("the referenced probe has not completed")
    requested_probe_id = str(context.parameters.get("probe_id") or "").strip()
    if not requested_probe_id or requested_probe_id != str(probe.get("probe_id") or ""):
        raise AttachmentProbeError("probe_id must reference the completed frozen probe")
    gripper_evidence = _require_probe_gripper_evidence(
        memory,
        context.observation,
        operation="attachment assessment",
    )
    before = [
        path
        for path in probe.get("pre_probe_image_paths", [])
        if isinstance(path, str) and path
    ]
    after = _current_rgb_paths(context.observation)
    role_aware = _has_backend_neutral_camera_roles(context.observation)
    if len(before) != 2 or len(after) != 2:
        required_views = (
            "scene-primary and one wrist-primary"
            if role_aware
            else "agentview and one wrist"
        )
        raise AttachmentProbeError(
            f"exactly one {required_views} RGB image are required before and after"
        )
    paths = [*before, *after]
    image_order = (
        [
            {"image_number": 1, "role": "before_scene_primary"},
            {"image_number": 2, "role": "before_wrist_primary"},
            {"image_number": 3, "role": "after_scene_primary"},
            {"image_number": 4, "role": "after_wrist_primary"},
        ]
        if role_aware
        else [
            {"image_number": 1, "role": "before_agentview"},
            {"image_number": 2, "role": "before_wrist"},
            {"image_number": 3, "role": "after_agentview"},
            {"image_number": 4, "role": "after_wrist"},
        ]
    )
    result = backend.decide(
        PlannerBackendRequest(
            system_prompt=(
                ROLE_AWARE_ARTICULATED_ATTACHMENT_ASSESSMENT_PROMPT
                if role_aware
                else ARTICULATED_ATTACHMENT_ASSESSMENT_PROMPT
            ),
            tool_context={
                "schema_version": ARTICULATED_ATTACHMENT_ASSESSMENT_SCHEMA,
                "role": "independent_articulated_attachment_reviewer",
                "task": str(context.metadata.get("task") or ""),
                "probe_id": probe.get("probe_id"),
                "candidate_id": probe.get("candidate_id"),
                "motion_type": probe.get("motion_type"),
                "distance_m": probe.get("distance_m"),
                "direction_world_xyz": probe.get("direction_world_xyz"),
                "image_order": image_order,
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
        "probe_id": probe.get("probe_id"),
        "candidate_id": probe.get("candidate_id"),
        "scene_epoch": memory.get("scene_epoch"),
        "verdict": verdict,
        "reason": reason,
        "gripper_evidence": gripper_evidence,
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
    compiled_grasp_id = str(parameters.get("compiled_grasp_id") or "").strip()
    graph = _mapping(memory.get("provenance_evidence_graph"), "provenance_evidence_graph")
    nodes = graph.get("nodes")
    grasp_node = next(
        (
            dict(node)
            for node in nodes
            if isinstance(node, Mapping)
            and node.get("kind") == "compiled_targeted_grasp"
            and str(node.get("compiled_grasp_id") or "") == compiled_grasp_id
        ),
        None,
    ) if isinstance(nodes, Sequence) else None
    if not compiled_grasp_id or not isinstance(grasp_node, dict):
        raise AttachmentProbeError(
            "compiled_grasp_id must name a grasp in provenance_evidence_graph"
        )
    if grasp_node.get("freshness") == "superseded_target_evidence":
        raise AttachmentProbeError(
            "compiled grasp evidence was superseded by a different selected target"
        )
    if grasp_node.get("freshness") == "invalidated_contact_geometry":
        raise AttachmentProbeError(
            "compiled grasp contact geometry was invalidated after gripper reopen; "
            "compile and execute a current contact branch before preparing a probe"
        )
    candidate_id = str(grasp_node.get("candidate_id") or "")
    if not candidate_id:
        raise AttachmentProbeError("compiled grasp candidate provenance is incomplete")
    scene_epoch = _nonnegative_int(memory.get("scene_epoch"), "scene_epoch")
    robot_motion_epoch = _nonnegative_int(
        memory.get("robot_motion_epoch", 0),
        "robot_motion_epoch",
    )
    if observation is None:
        raise AttachmentProbeError("a current observation is required")
    gripper_evidence = _require_probe_gripper_evidence(
        memory,
        observation,
        operation="attachment probe preparation",
    )
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
    probe_id = f"probe:{path_sha256}"
    _stamp_probe_metadata(tool_parameters, path_sha256=path_sha256)
    preview_requests = [
        {
            "tool": "ik_preview_check",
            "parameters": {
                "probe_id": probe_id,
                "waypoint_index": index,
                "position_tolerance_m": 0.01,
                "orientation_tolerance_rad": 0.10,
                "check_endpoint_collision": True,
            },
        }
        for index, _pose in enumerate(frozen_path)
    ]
    execution_parameters = (
        {
            "ik_receipt_id": "<receipt id from the single preview above>",
            "enable_collision_check": True,
        }
        if tool_name == "move_to"
        else {
            "ik_receipt_ids": [
                f"<receipt id from preview {index + 1}>"
                for index in range(len(frozen_path))
            ],
            "enable_collision_check": True,
        }
    )
    pre_probe_images = _current_rgb_paths(observation)
    if len(pre_probe_images) != 2:
        raise AttachmentProbeError(
            "prepare_attachment_probe requires current agentview and wrist RGB images"
        )
    return {
        "schema_version": ARTICULATED_ATTACHMENT_PROBE_SCHEMA,
        "status": "prepared",
        "probe_id": probe_id,
        "candidate_id": candidate_id,
        "compiled_grasp_id": compiled_grasp_id,
        "scene_epoch": scene_epoch,
        "robot_motion_epoch": robot_motion_epoch,
        "interaction_family": "articulated_handle",
        "motion_type": motion_type,
        "distance_m": ARTICULATED_ATTACHMENT_PROBE_DISTANCE_M,
        "start_eef_xyz": _round_vector(start_xyz),
        "direction_world_xyz": _round_vector(direction),
        "frozen_path": frozen_path,
        "path_sha256": path_sha256,
        "frozen_motion": {"name": tool_name, "parameters": tool_parameters},
        "ik_preview_requests": preview_requests,
        "execution_handoff": {
            "tool": tool_name,
            "parameters": execution_parameters,
            "instruction": (
                "Run every IK preview in order, then pass only the returned receipt "
                "id or ids to the named motion tool. Do not copy the frozen poses."
            ),
        },
        "pre_probe_image_paths": pre_probe_images,
        "proposal_reason": reason,
        "gripper_evidence": gripper_evidence,
        "checked_by": "host_probe_geometry",
    }


def _require_probe_gripper_evidence(
    memory: Mapping[str, Any],
    observation: Any,
    *,
    operation: str,
) -> JsonDict:
    """Reject probes that contradict host command or measured aperture evidence."""

    commanded_value = memory.get("gripper_command_state")
    commanded = dict(commanded_value) if isinstance(commanded_value, Mapping) else {}
    if commanded.get("position") != 0 or commanded.get("state") != "closed":
        raise AttachmentProbeError(
            f"{operation} requires the latest acknowledged gripper command to be "
            "closed; execute a valid contact close before preparing or assessing a probe"
        )

    proxy_value = commanded.get("attachment_proxy_receipt")
    proxy = dict(proxy_value) if isinstance(proxy_value, Mapping) else {}
    if proxy.get("status") != "tentative":
        status = str(proxy.get("status") or "missing")
        reason = str(proxy.get("reason") or "no tentative close receipt")
        raise AttachmentProbeError(
            f"{operation} requires a tentative non-empty close receipt; latest "
            f"attachment_proxy_status={status!r}, reason={reason!r}. Inspect current "
            "dual-view evidence, repair contact, close again, then prepare a new probe"
        )

    robot = getattr(observation, "robot", None)
    measured_value = getattr(robot, "gripper_state", None)
    measured = dict(measured_value) if isinstance(measured_value, Mapping) else {}
    is_open = measured.get("open")
    openness = measured.get("openness")
    has_numeric_openness = (
        isinstance(openness, (int, float))
        and not isinstance(openness, bool)
        and math.isfinite(float(openness))
    )
    # Continuous aperture is the more informative signal. Some simulator
    # adapters label a partially obstructed grasp as ``open=True`` even when
    # the measured aperture is far below fully open. Fall back to the coarse
    # boolean only when no finite aperture measurement is available.
    definitely_open = (
        float(openness) >= 0.8 if has_numeric_openness else is_open is True
    )
    if definitely_open:
        raise AttachmentProbeError(
            f"{operation} contradicts the current measured gripper state: "
            f"open={is_open!r}, openness={openness!r}. Re-establish contact and close "
            "the gripper before using attachment evidence"
        )
    return {
        "commanded_position": 0,
        "attachment_proxy_status": "tentative",
        "attachment_proxy_reason": proxy.get("reason"),
        "measured_open": is_open,
        "measured_openness": openness,
        "checked_by": "host_gripper_evidence",
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
    preferred_roles = {"scene_primary": 0, "wrist_primary": 1}
    preferred_frames = {"agentview": 0, "wrist": 1}
    ranked: list[tuple[int, int, str]] = []
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping) or artifact.get("kind") != "rgb":
            continue
        frame_id = str(artifact.get("frame_id") or "")
        role = str(artifact.get("role") or "")
        path = artifact.get("path")
        rank = preferred_roles.get(role)
        if rank is None:
            rank = preferred_frames.get(frame_id)
        if rank is None or not isinstance(path, str) or not path:
            continue
        ranked.append((rank, index, path))
    ranked.sort()
    selected: dict[int, str] = {}
    for rank, _, path in ranked:
        selected.setdefault(rank, path)
    return [selected[rank] for rank in sorted(selected)]


def _has_backend_neutral_camera_roles(observation: Any) -> bool:
    metadata = getattr(observation, "metadata", None)
    artifacts = metadata.get("image_artifacts") if isinstance(metadata, Mapping) else None
    if not isinstance(artifacts, list):
        return False
    return any(
        isinstance(artifact, Mapping)
        and str(artifact.get("role") or "")
        in {"scene_primary", "wrist_primary"}
        for artifact in artifacts
    )
