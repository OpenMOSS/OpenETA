"""Session-scoped observation packet indexing and artifact resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from adapter.protocol import EnvObservation, JsonDict


OBSERVATION_PACKET_INDEX_SCHEMA_VERSION = "openeta.observation_packet_index.v1"


class ObservationPacketResolutionError(ValueError):
    """A fail-closed source packet lookup error suitable for tool feedback."""

    def __init__(self, code: str, message: str, *, details: JsonDict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})

    def to_dict(self) -> JsonDict:
        return {
            "code": self.code,
            "message": str(self),
            **self.details,
        }


def build_observation_packet_entries(
    observation: EnvObservation,
    *,
    observation_index: int,
    scene_epoch: int,
    object_scene_epoch: int,
    robot_motion_epoch: int,
    compact_ids: bool = False,
) -> list[JsonDict]:
    """Build immutable packet index entries without copying image pixels."""

    raw_artifacts = observation.metadata.get("image_artifacts")
    if not isinstance(raw_artifacts, list):
        return []
    cameras = {camera.frame_id: camera for camera in observation.cameras}
    grouped: dict[str, list[JsonDict]] = {}
    for raw in raw_artifacts:
        if (
            not isinstance(raw, dict)
            or raw.get("kind") not in {"rgb", "depth", "image"}
            or not isinstance(raw.get("path"), str)
            or not str(raw.get("path") or "")
        ):
            continue
        packet_id = str(raw.get("packet_id") or "").strip()
        if not packet_id:
            continue
        artifact: JsonDict = {
            "kind": str(raw["kind"]),
            "frame_id": str(raw.get("frame_id") or ""),
            "path": str(raw["path"]),
        }
        camera = cameras.get(str(artifact["frame_id"]))
        role = str(raw.get("role") or getattr(camera, "role", "") or "")
        if role:
            artifact["role"] = role
        for name in ("index", "format", "width", "height", "byte_size"):
            if raw.get(name) is not None:
                artifact[name] = raw[name]
        grouped.setdefault(packet_id, []).append(artifact)

    entries: list[JsonDict] = []
    for packet_ordinal, (origin_packet_id, artifacts) in enumerate(grouped.items(), start=1):
        packet_id = (
            _compact_packet_id(
                origin_packet_id,
                observation_index=observation_index,
                packet_ordinal=packet_ordinal,
                packet_count=len(grouped),
            )
            if compact_ids
            else origin_packet_id
        )
        _validate_unique_artifact_keys(origin_packet_id, artifacts)
        frame_ids = {str(item.get("frame_id") or "") for item in artifacts}
        camera_rows: list[JsonDict] = []
        for frame_id in sorted(frame_ids):
            camera = cameras.get(frame_id)
            row: JsonDict = {"frame_id": frame_id}
            if camera is not None:
                if camera.role:
                    row["role"] = camera.role
                if camera.intrinsics:
                    row["intrinsics"] = _plain_json(camera.intrinsics)
                if camera.extrinsics:
                    row["extrinsics"] = _plain_json(camera.extrinsics)
                if camera.timestamp_s is not None:
                    row["timestamp_s"] = float(camera.timestamp_s)
            camera_rows.append(row)
        entry: JsonDict = {
                "schema_version": OBSERVATION_PACKET_INDEX_SCHEMA_VERSION,
                "packet_id": packet_id,
                "observation_index": observation_index,
                "task": observation.task,
                "scene_epoch": scene_epoch,
                "object_scene_epoch": object_scene_epoch,
                "robot_motion_epoch": robot_motion_epoch,
                "robot": {
                    "end_effector_pose": _plain_json(
                        observation.robot.end_effector_pose
                    ),
                    "gripper_state": _plain_json(observation.robot.gripper_state),
                },
                "artifacts": artifacts,
                "cameras": camera_rows,
            }
        if packet_id != origin_packet_id:
            # Kept only for host-side audit/debugging. Agent-facing references
            # use the short session-scoped packet_id.
            entry["origin_packet_id"] = origin_packet_id
        entries.append(entry)
    return entries


def _compact_packet_id(
    origin_packet_id: str,
    *,
    observation_index: int,
    packet_ordinal: int,
    packet_count: int,
) -> str:
    """Return a short session-scoped opaque ID for model-visible references."""

    if len(origin_packet_id) <= 32:
        return origin_packet_id
    base = f"obs-{max(0, int(observation_index)):04d}"
    return base if packet_count == 1 else f"{base}-p{packet_ordinal}"


def resolve_packet_source(
    entry: JsonDict,
    *,
    camera_frame_id: str = "",
    require_kind: str = "rgb",
    require_files: bool = True,
) -> JsonDict:
    """Resolve one packet to an exact camera artifact and aligned source metadata."""

    packet_id = str(entry.get("packet_id") or "").strip()
    artifacts = [
        dict(item)
        for item in entry.get("artifacts", [])
        if isinstance(item, dict)
    ]
    available_frames = sorted(
        {
            str(item.get("frame_id") or "")
            for item in artifacts
            if item.get("kind") == require_kind
        }
    )
    requested_frame = camera_frame_id.strip()
    if requested_frame:
        resolved_frame = requested_frame
    elif "agentview" in available_frames:
        resolved_frame = "agentview"
    else:
        primary_frames = sorted(
            {
                str(item.get("frame_id") or "")
                for item in artifacts
                if item.get("kind") == require_kind
                and item.get("role") == "scene_primary"
            }
        )
        if len(primary_frames) == 1:
            resolved_frame = primary_frames[0]
        elif len(available_frames) == 1:
            resolved_frame = available_frames[0]
        else:
            raise ObservationPacketResolutionError(
                "source_camera_ambiguous",
                "The source packet contains multiple RGB camera frames; provide camera_frame_id.",
                details={
                    "source_packet_id": packet_id,
                    "available_camera_frame_ids": available_frames,
                },
            )

    matches = [
        item
        for item in artifacts
        if item.get("kind") == require_kind
        and str(item.get("frame_id") or "") == resolved_frame
    ]
    if not matches:
        raise ObservationPacketResolutionError(
            "source_camera_not_found",
            f"Camera frame {resolved_frame!r} has no {require_kind} artifact in the source packet.",
            details={
                "source_packet_id": packet_id,
                "requested_camera_frame_id": resolved_frame,
                "available_camera_frame_ids": available_frames,
            },
        )
    if len(matches) != 1:
        raise ObservationPacketResolutionError(
            "source_artifact_ambiguous",
            "The source packet contains duplicate artifacts for the requested camera and kind.",
            details={
                "source_packet_id": packet_id,
                "camera_frame_id": resolved_frame,
                "kind": require_kind,
                "candidate_count": len(matches),
            },
        )
    source = matches[0]
    path = str(source.get("path") or "")
    if require_files and (not path or not Path(path).is_file()):
        raise ObservationPacketResolutionError(
            "source_artifact_missing",
            "The packet index exists, but its resolved local artifact is missing.",
            details={
                "source_packet_id": packet_id,
                "camera_frame_id": resolved_frame,
                "kind": require_kind,
                "resolved_path": path,
            },
        )

    result: JsonDict = {
        "packet_id": packet_id,
        "frame_id": resolved_frame,
        require_kind: path,
    }
    role = str(source.get("role") or "")
    if role:
        result["role"] = role
    aligned_depth = next(
        (
            item
            for item in artifacts
            if item.get("kind") == "depth"
            and str(item.get("frame_id") or "") == resolved_frame
        ),
        None,
    )
    if aligned_depth is not None and isinstance(aligned_depth.get("path"), str):
        result["depth"] = str(aligned_depth["path"])
    camera = next(
        (
            item
            for item in entry.get("cameras", [])
            if isinstance(item, dict)
            and str(item.get("frame_id") or "") == resolved_frame
        ),
        None,
    )
    if isinstance(camera, dict):
        for name in ("intrinsics", "extrinsics"):
            value = camera.get(name)
            if isinstance(value, dict) and value:
                result[name] = _plain_json(value)
        if camera.get("timestamp_s") is not None:
            result["timestamp_s"] = camera["timestamp_s"]
        if not role and isinstance(camera.get("role"), str) and camera["role"]:
            result["role"] = camera["role"]
    for name in (
        "observation_index",
        "scene_epoch",
        "object_scene_epoch",
        "robot_motion_epoch",
    ):
        if entry.get(name) is not None:
            result[name] = entry[name]
    robot = entry.get("robot")
    if isinstance(robot, dict):
        end_effector_pose = robot.get("end_effector_pose")
        if isinstance(end_effector_pose, dict) and end_effector_pose:
            result["current_eef_pose"] = _plain_json(end_effector_pose)
        gripper_state = robot.get("gripper_state")
        if isinstance(gripper_state, dict) and gripper_state:
            result["gripper_state"] = _plain_json(gripper_state)
    return result


def packet_integrity_fingerprint(entry: JsonDict) -> str:
    """Return canonical packet content used to detect ID reuse."""

    immutable = {
        "packet_id": entry.get("packet_id"),
        "task": entry.get("task"),
        "artifacts": entry.get("artifacts", []),
        "cameras": entry.get("cameras", []),
        "robot": entry.get("robot", {}),
    }
    return json.dumps(immutable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def find_packet_id_for_path(entries: Iterable[JsonDict], path: object) -> str:
    """Return the packet id owning an exact local artifact path."""

    return str(find_packet_reference_for_path(entries, path).get("source_packet_id") or "")


def find_packet_reference_for_path(
    entries: Iterable[JsonDict],
    path: object,
) -> JsonDict:
    """Return the exact packet and frame owning one local artifact path."""

    if not isinstance(path, str) or not path:
        return {}
    try:
        requested = Path(path).resolve(strict=False)
    except (OSError, ValueError):
        return {}
    matches: dict[tuple[str, str], JsonDict] = {}
    for entry in entries:
        for artifact in entry.get("artifacts", []):
            if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
                continue
            try:
                candidate = Path(str(artifact["path"])).resolve(strict=False)
            except (OSError, ValueError):
                continue
            if candidate == requested:
                packet_id = str(entry.get("packet_id") or "")
                frame_id = str(artifact.get("frame_id") or "")
                if packet_id:
                    matches[(packet_id, frame_id)] = {
                        "source_packet_id": packet_id,
                        "camera_frame_id": frame_id,
                    }
    return next(iter(matches.values())) if len(matches) == 1 else {}


def _validate_unique_artifact_keys(packet_id: str, artifacts: list[JsonDict]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for artifact in artifacts:
        key = (str(artifact.get("frame_id") or ""), str(artifact.get("kind") or ""))
        path = str(artifact.get("path") or "")
        previous = seen.get(key)
        if previous is not None and previous != path:
            raise ObservationPacketResolutionError(
                "duplicate_source_artifact",
                "One observation packet contains multiple paths for the same camera and kind.",
                details={
                    "source_packet_id": packet_id,
                    "camera_frame_id": key[0],
                    "kind": key[1],
                    "paths": [previous, path],
                },
            )
        seen[key] = path


def _plain_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))
