"""Host-owned preparation and visual review for bounded attachment probes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from adapter.protocol import JsonDict
from agent.backends.planner import PlannerBackend, PlannerBackendRequest
from agent.tools.registry import ToolExecutionContext, ToolHandler, make_tool_result
from agent.tools.gripper_evidence import (
    gripper_actuation_receipt_error,
    measured_gripper_open,
)


ARTICULATED_ATTACHMENT_PROBE_SCHEMA = "openeta.articulated_attachment_probe.v1"
ARTICULATED_ATTACHMENT_PROBE_DISTANCE_M = 0.05
ARTICULATED_ATTACHMENT_PROBE_DISTANCE_TOLERANCE_M = 0.002
ARTICULATED_ATTACHMENT_PROBE_MAX_WAYPOINTS = 5
ARTICULATED_ATTACHMENT_PROBE_MAX_SEGMENT_M = 0.015
ARTICULATED_ATTACHMENT_PROBE_MAX_REASON_CHARS = 1024
ARTICULATED_ATTACHMENT_ASSESSMENT_SCHEMA = (
    "openeta.articulated_attachment_assessment.v1"
)

# A carried-object proxy is intentionally conservative and is not meaningful
# for an articulated mechanism such as a drawer.  Some simulator backends bind
# the compiled contact to the correct mechanism surface, but cannot arm a
# carried-object proxy because the EEF is far from the mechanism body's centre.
# In that one case, a matching reached compiled-contact receipt is the stronger
# evidence for allowing a short articulated probe.
_ARTICULATED_PROXY_NOT_APPLICABLE_REASONS = frozenset(
    {
        "authorized_target_outside_contact_envelope",
        "close_not_supported_by_host_contact_envelope",
    }
)

ARTICULATED_ATTACHMENT_ASSESSMENT_PROMPT = """You are an independent attachment reviewer.
The robot closed on a target, either a portable object or an articulated handle,
and executed one host-frozen 5 cm probe.
Compare the ordered before/after agentview and wrist images. Return PASS only when
the same target object, handle, or articulated body visibly co-moved along the probe
path and remains engaged by the gripper. Return FAIL only when direct evidence shows
the target stayed behind, moved inconsistently, or separated from the gripper. Return
UNKNOWN for occlusion, conflicting views, identity ambiguity, or insufficient motion evidence.
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
    """Build the read-only bounded attachment-probe compiler."""

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
                content=f"attachment probe rejected: {exc}",
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
                "attachment probe geometry frozen; run the returned "
                "ordered IK preview requests, then execute by receipt id(s)"
            ),
            outputs=outputs,
        )

    return handler


def build_assess_attachment_probe_handler(backend: PlannerBackend) -> ToolHandler:
    """Build the independent before/after attachment reviewer."""

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
                content=f"attachment assessment failed: {exc}",
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
            content=f"attachment assessment: {outputs['verdict']}",
            outputs=outputs,
        )

    return handler


def assess_attachment_probe(
    context: ToolExecutionContext,
    *,
    backend: PlannerBackend,
) -> JsonDict:
    """Assess target co-motion from the frozen probe's before/after views."""

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
        compiled_grasp_id=str(probe.get("compiled_grasp_id") or ""),
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
                "role": "independent_attachment_reviewer",
                "task": str(context.metadata.get("task") or ""),
                "probe_id": probe.get("probe_id"),
                "candidate_id": probe.get("candidate_id"),
                "motion_type": probe.get("motion_type"),
                "distance_m": probe.get("distance_m"),
                "direction_world_xyz": probe.get("direction_world_xyz"),
                "interaction_family": probe.get("interaction_family"),
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
    if verdict == "PASS" and gripper_evidence.get("measured_open") is not False:
        verdict = "UNKNOWN"
        reason = "Visual PASS is not corroborated by a valid non-open aperture. " + reason
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
    # The probe uses the current measured EEF pose, not old contact coordinates.
    # Released contact geometry is not attachment proof, but does not prevent
    # an identity-bound experiment with newly captured before/after evidence.
    candidate_id = str(grasp_node.get("candidate_id") or "")
    if not candidate_id:
        raise AttachmentProbeError("compiled grasp candidate provenance is incomplete")
    target_geometry_family = str(
        grasp_node.get("target_geometry_family") or ""
    ).strip().lower()
    interaction_family = (
        "articulated_handle"
        if target_geometry_family == "articulated_handle"
        else "portable_object"
        if target_geometry_family
        else "unspecified_target"
    )
    probe_type = (
        "articulated_attachment"
        if interaction_family == "articulated_handle"
        else "portable_object_attachment"
        if interaction_family == "portable_object"
        else "generic_attachment"
    )
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
        compiled_grasp_id=compiled_grasp_id,
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
            probe_type=probe_type,
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
                probe_type=probe_type,
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
        "interaction_family": interaction_family,
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
        "interaction_family": interaction_family,
        "target_geometry_family": target_geometry_family or None,
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
    memory: Mapping[str, Any], observation: Any, *, operation: str,
    compiled_grasp_id: str,
) -> JsonDict:
    """Observe probe conditions; do not require proof of attachment to test it.

    Missing command/proxy history is uncertainty, not a veto on a bounded
    Agent-chosen probe. Assessment still needs the frozen before/after evidence.
    """
    commanded_value = memory.get("gripper_command_state")
    commanded = dict(commanded_value) if isinstance(commanded_value, Mapping) else {}
    proxy_value = commanded.get("attachment_proxy_receipt")
    proxy = dict(proxy_value) if isinstance(proxy_value, Mapping) else {}
    robot = getattr(observation, "robot", None)
    measured_value = getattr(robot, "gripper_state", None)
    measured = dict(measured_value) if isinstance(measured_value, Mapping) else {}
    measured_open = measured_gripper_open(measured)
    warnings = []
    if commanded.get("position") != 0 or ("latched" in commanded and commanded["latched"] is not True):
        warnings.append("no_acknowledged_close_latch")
    if "gripper_actuation_receipt" in commanded:
        error = gripper_actuation_receipt_error(commanded["gripper_actuation_receipt"], position=0)
        if error:
            warnings.append("inconsistent_gripper_actuation_receipt")
    if proxy.get("status") != "tentative":
        warnings.append("no_tentative_attachment_proxy")
    if measured_open is True:
        warnings.append("measured_gripper_open")
    elif measured_open is None:
        warnings.append("measured_aperture_unknown")
    contact = _matching_compiled_contact_receipt(memory, compiled_grasp_id=compiled_grasp_id)
    return {
        "commanded_position": commanded.get("position"),
        "attachment_proxy_status": proxy.get("status", "missing"),
        "attachment_proxy_reason": proxy.get("reason"),
        "probe_evidence_basis": "bounded_probe_not_attachment_proof",
        "compiled_grasp_id": compiled_grasp_id,
        "compiled_contact_reached": bool(contact and contact.get("reached_target")),
        "measured_open": measured_open,
        "measured_openness": measured.get("openness") if measured_open is not None else None,
        "checked_by": "host_gripper_evidence",
        "advisories": warnings, "blocking": False,
    }


def _matching_compiled_contact_receipt(
    memory: Mapping[str, Any],
    *,
    compiled_grasp_id: str,
) -> JsonDict | None:
    """Return reached host contact evidence for this exact compiled grasp."""

    value = memory.get("latest_compiled_contact_execution")
    receipt = dict(value) if isinstance(value, Mapping) else {}
    if not compiled_grasp_id:
        return None
    if str(receipt.get("compiled_grasp_id") or "") != compiled_grasp_id:
        return None
    if receipt.get("reached_target") is not True:
        return None
    schema = str(receipt.get("schema_version") or "")
    if schema and schema != "openeta.compiled_contact_execution.v1":
        return None
    return receipt


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
    probe_type: str,
    waypoint_index: int | None = None,
) -> JsonDict:
    pose: JsonDict = {
        "frame": "world",
        "xyz": _round_vector(xyz),
        **dict(rotation),
        "probe_type": probe_type,
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
