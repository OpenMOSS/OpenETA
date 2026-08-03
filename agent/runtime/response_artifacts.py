"""Materialize full JSON responses into local artifact files."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from adapter.protocol import JsonDict


DEFAULT_RESPONSE_ARTIFACT_OUTPUT_ROOT = Path("tmp") / "tool_result"


@dataclass(frozen=True, slots=True)
class JsonResponseArtifact:
    """Local reference for one materialized JSON response."""

    index: str
    path: str
    chars: int
    grep_hint: str

    def to_dict(self) -> JsonDict:
        return {
            "type": "json",
            "index": self.index,
            "path": self.path,
            "chars": self.chars,
            "grep_hint": self.grep_hint,
        }


def materialize_json_response(
    payload: JsonDict,
    *,
    output_root: str | Path | None = None,
    bundle_id: str | None = None,
    name: str = "response",
) -> JsonResponseArtifact:
    """Write a JSON response to disk and return a lightweight reference."""

    if not isinstance(payload, dict):
        raise TypeError("materialize_json_response expects a dict payload")
    bundle = _safe_token(bundle_id or str(uuid4()))
    root = Path(output_root or DEFAULT_RESPONSE_ARTIFACT_OUTPUT_ROOT) / bundle
    root.mkdir(parents=True, exist_ok=True)
    index = _safe_token(name)
    path = root / f"{index}.json"
    sanitized = _strip_preview_values(payload)
    text = json.dumps(sanitized, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return JsonResponseArtifact(
        index=index,
        path=str(path.resolve()),
        chars=len(text),
        grep_hint=f"grep -n '<pattern>' {path.resolve()}",
    )


def build_response_reference(
    payload: JsonDict,
    artifact: JsonResponseArtifact,
    *,
    image_artifacts: list[JsonDict] | None = None,
    max_cameras: int = 8,
) -> JsonDict:
    """Return the small response object exposed to the agent context."""

    reference: JsonDict = {
        "response_path": artifact.path,
        "response_chars": artifact.chars,
        "response_omitted": True,
        "grep_hint": artifact.grep_hint,
    }
    for key in (
        "ok",
        "success",
        "error",
        "error_type",
        "message",
        "handle",
        "session_id",
        "env_id",
        "reward",
        "terminated",
        "truncated",
    ):
        if key not in payload:
            continue
        value = payload.get(key)
        if _is_small_scalar(value):
            reference[key] = value

    cameras = _collect_camera_refs(payload, max_cameras=max_cameras)
    if cameras:
        reference["cameras"] = cameras
    observation_summary = build_observation_summary(payload, max_cameras=max_cameras)
    if observation_summary:
        reference["observation_summary"] = observation_summary
    motion_summary = build_motion_summary(payload)
    if motion_summary:
        reference["motion_summary"] = motion_summary
    for key in ("envs", "tasks", "items", "results"):
        if key in payload and isinstance(payload[key], list):
            reference[f"{key}_count"] = len(payload[key])
    if image_artifacts:
        reference["image_artifacts"] = list(image_artifacts)
    return reference


def build_observation_summary(
    payload: JsonDict,
    *,
    max_cameras: int = 8,
    max_objects: int = 32,
) -> JsonDict:
    """Return planner-useful simulator state without inline image payloads."""

    observation = payload.get("observation")
    if not isinstance(observation, dict):
        observation = payload if _looks_like_observation(payload) else {}
    if not observation:
        return {}

    summary: JsonDict = {}
    task = observation.get("task")
    if task is not None and _is_small_scalar(task):
        summary["task"] = task

    cameras = _collect_camera_refs(observation, max_cameras=max_cameras)
    if cameras:
        summary["cameras"] = cameras

    robot = observation.get("robot")
    if isinstance(robot, dict):
        robot_summary = _compact_robot_state(robot)
        if robot_summary:
            summary["robot"] = robot_summary

    objects = observation.get("objects")
    if isinstance(objects, list):
        compact_objects = [
            compact
            for item in objects[:max_objects]
            if isinstance(item, dict)
            and (compact := _compact_object_state(item))
        ]
        summary["object_count"] = len(objects)
        summary["objects"] = compact_objects
    return summary


def build_motion_summary(payload: JsonDict) -> JsonDict:
    """Return compact controller outcome fields used for closed-loop recovery."""

    summary: JsonDict = {}
    collision = payload.get("collision")
    if isinstance(collision, dict):
        summary["collision"] = _compact_scalar_mapping(collision)
    for key in ("start", "end", "target"):
        value = payload.get(key)
        if isinstance(value, dict):
            summary[key] = _compact_pose(value)
    steps = payload.get("steps_executed")
    if isinstance(steps, int):
        summary["steps_executed"] = steps

    reached = payload.get("reached_target")
    if isinstance(reached, bool):
        summary["reached_target"] = reached
    elif all(key in summary for key in ("end", "target")):
        inferred = _position_target_reached(summary["end"], summary["target"])
        if inferred is not None:
            summary["reached_target"] = inferred
    compact_collision = summary.get("collision", {})
    if (
        compact_collision.get("detected") is True
        and compact_collision.get("new_or_worsened") is not False
    ):
        summary["reached_target"] = False
    return summary


def _strip_preview_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _strip_preview_values(item)
            for key, item in value.items()
            if str(key).lower() != "preview"
        }
    if isinstance(value, list):
        return [_strip_preview_values(item) for item in value]
    return value


def _collect_camera_refs(payload: Any, *, max_cameras: int) -> list[JsonDict]:
    cameras: list[JsonDict] = []

    def visit(value: Any) -> None:
        if len(cameras) >= max_cameras:
            return
        if isinstance(value, dict):
            if any(key in value for key in ("rgb_path", "depth_path", "image_path")):
                cameras.append(_compact_camera_ref(value))
                return
            for item in value.values():
                visit(item)
                if len(cameras) >= max_cameras:
                    return
        elif isinstance(value, list):
            for item in value:
                visit(item)
                if len(cameras) >= max_cameras:
                    return

    visit(payload)
    return cameras


def _looks_like_observation(payload: JsonDict) -> bool:
    return any(key in payload for key in ("task", "cameras", "robot", "objects", "proprio"))


def _compact_robot_state(robot: JsonDict) -> JsonDict:
    compact: JsonDict = {}
    end_effector = robot.get("end_effector_pose")
    if isinstance(end_effector, dict):
        pose = _compact_pose(end_effector)
        if pose:
            compact["end_effector_pose"] = pose
    gripper = robot.get("gripper_state")
    if isinstance(gripper, dict):
        compact["gripper_state"] = _compact_scalar_mapping(gripper)
    joints = robot.get("joint_positions")
    if isinstance(joints, list):
        compact["joint_position_count"] = len(joints)
    return compact


def _compact_object_state(item: JsonDict) -> JsonDict:
    compact: JsonDict = {}
    for key in ("name", "category", "instance_id", "id"):
        value = item.get(key)
        if value is not None and _is_small_scalar(value):
            compact[key] = value
    for key in ("position", "orientation", "quat_xyzw"):
        value = item.get(key)
        if _is_numeric_sequence(value, lengths={3, 4}):
            compact[key] = list(value)
    return compact


def _compact_pose(pose: JsonDict) -> JsonDict:
    compact: JsonDict = {}
    for key in ("x", "y", "z", "roll", "pitch", "yaw", "frame"):
        value = pose.get(key)
        if value is not None and _is_small_scalar(value):
            compact[key] = value
    for key, lengths in (
        ("xyz", {3}),
        ("position", {3}),
        ("quat_xyzw", {4}),
        ("rotation_matrix", {3, 9}),
    ):
        value = pose.get(key)
        if key == "rotation_matrix" and _is_matrix3(value):
            compact[key] = [list(row) for row in value]
        elif _is_numeric_sequence(value, lengths=lengths):
            compact[key] = list(value)
    return compact


def _compact_scalar_mapping(value: JsonDict) -> JsonDict:
    return {
        str(key): item
        for key, item in value.items()
        if _is_small_scalar(item)
    }


def _position_target_reached(end: JsonDict, target: JsonDict) -> bool | None:
    end_xyz = _xyz_from_pose(end)
    target_xyz = _xyz_from_pose(target)
    if end_xyz is None or target_xyz is None:
        return None
    return all(abs(end_xyz[idx] - target_xyz[idx]) <= 0.01 for idx in range(3))


def _xyz_from_pose(pose: JsonDict) -> list[float] | None:
    xyz = pose.get("xyz") or pose.get("position")
    if _is_numeric_sequence(xyz, lengths={3}):
        return [float(value) for value in xyz]
    if all(isinstance(pose.get(axis), int | float) for axis in ("x", "y", "z")):
        return [float(pose[axis]) for axis in ("x", "y", "z")]
    return None


def _is_numeric_sequence(value: Any, *, lengths: set[int]) -> bool:
    return (
        isinstance(value, list | tuple)
        and len(value) in lengths
        and all(isinstance(item, int | float) for item in value)
    )


def _is_matrix3(value: Any) -> bool:
    return (
        isinstance(value, list | tuple)
        and len(value) == 3
        and all(_is_numeric_sequence(row, lengths={3}) for row in value)
    )


def _compact_camera_ref(camera: JsonDict) -> JsonDict:
    compact: JsonDict = {}
    for key in (
        "frame_id",
        "camera",
        "width",
        "height",
        "rgb_ref",
        "rgb_path",
        "depth_ref",
        "depth_path",
        "image_ref",
        "image_path",
        "intrinsics",
        "anygrasp_intrinsics",
    ):
        value = camera.get(key)
        if value is not None and (_is_small_scalar(value) or _is_intrinsics_dict(value)):
            compact[key] = value
    return compact


def _is_intrinsics_dict(value: Any) -> bool:
    return isinstance(value, dict) and all(
        key in value and _is_small_scalar(value.get(key))
        for key in ("fx", "fy", "cx", "cy")
    )


def _is_small_scalar(value: Any) -> bool:
    return value is None or isinstance(value, bool | int | float) or (
        isinstance(value, str) and len(value) <= 300
    )


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    return token[:80] or "item"
