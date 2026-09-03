"""Observation projection and artifact writing for direct UniVTAC smoke tests."""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from sim.envs.univtac.contract import (
    ArtifactRef,
    IncompleteTactilePacketError,
    UniVTACContractError,
    UniVTACTaskSnapshot,
)

REQUIRED_TACTILE_FIELDS = frozenset({"rgb", "rgb_marker", "marker", "depth", "pose"})


@dataclass(frozen=True)
class SnapshotCapture:
    snapshot: UniVTACTaskSnapshot
    rgb_markers: dict[str, np.ndarray]
    observation_key_tree: dict[str, Any]


def to_numpy(value: Any) -> np.ndarray:
    """Convert NumPy-compatible or torch-like values without importing torch."""

    if isinstance(value, np.ndarray):
        return value
    candidate = value
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    if hasattr(candidate, "numpy"):
        candidate = candidate.numpy()
    return np.asarray(candidate)


def array_summary(value: Any) -> dict[str, Any]:
    array = to_numpy(value)
    if array.size == 0:
        raise UniVTACContractError("observation array must not be empty")
    if not np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_:
        raise UniVTACContractError(f"observation array must be numeric, got {array.dtype}")
    if not np.isfinite(array).all():
        raise UniVTACContractError("observation array contains non-finite values")
    return {
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
        "range": [float(array.min()), float(array.max())],
    }


def observation_key_tree(value: Any) -> Any:
    """Describe the runtime observation tree without embedding tensors."""

    if isinstance(value, Mapping):
        return {str(key): observation_key_tree(child) for key, child in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [observation_key_tree(child) for child in value]
    try:
        return array_summary(value)
    except (TypeError, ValueError, UniVTACContractError):
        return {"type": type(value).__name__}


def validate_tactile_packets(
    observation: Mapping[str, Any],
    *,
    strict_two_tactile_sensors: bool,
    fail_on_missing_rgb_marker: bool,
) -> tuple[str, ...]:
    tactile = observation.get("tactile")
    if not isinstance(tactile, Mapping) or not tactile:
        raise IncompleteTactilePacketError("observation.tactile is missing or empty")
    sensor_names = tuple(sorted(str(name) for name in tactile))
    if strict_two_tactile_sensors and len(sensor_names) != 2:
        raise IncompleteTactilePacketError(
            f"expected exactly two tactile sensors, discovered {len(sensor_names)}: {sensor_names}"
        )
    for sensor_name in sensor_names:
        packet = tactile.get(sensor_name)
        if not isinstance(packet, Mapping):
            raise IncompleteTactilePacketError(
                f"tactile packet for {sensor_name!r} is not a mapping"
            )
        missing_fields = sorted(REQUIRED_TACTILE_FIELDS - packet.keys())
        if missing_fields:
            raise IncompleteTactilePacketError(
                f"tactile packet for {sensor_name!r} is missing fields: {missing_fields}"
            )
        if fail_on_missing_rgb_marker and "rgb_marker" not in packet:
            raise IncompleteTactilePacketError(
                f"tactile packet for {sensor_name!r} is missing rgb_marker"
            )
        if "rgb_marker" in packet:
            marker = to_numpy(packet["rgb_marker"])
            if marker.ndim != 3 or marker.shape[-1] != 3:
                raise IncompleteTactilePacketError(
                    f"{sensor_name}.rgb_marker must have HxWx3 shape, got {marker.shape}"
                )
            if marker.size == 0 or not np.issubdtype(marker.dtype, np.number):
                raise IncompleteTactilePacketError(
                    f"{sensor_name}.rgb_marker has invalid dtype {marker.dtype}"
                )
            if not np.isfinite(marker).all():
                raise IncompleteTactilePacketError(
                    f"{sensor_name}.rgb_marker contains non-finite values"
                )
    return sensor_names


def _safe_component(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return safe or "unnamed"


def _relative_path(path: Path, output_root: Path) -> str:
    try:
        return path.resolve().relative_to(output_root.resolve()).as_posix()
    except ValueError as exc:
        raise UniVTACContractError(f"artifact is outside output root: {path}") from exc


def _to_uint8_rgb(value: Any) -> tuple[np.ndarray, np.ndarray]:
    source = to_numpy(value)
    if source.ndim != 3 or source.shape[-1] != 3:
        raise UniVTACContractError(f"RGB artifact must have HxWx3 shape, got {source.shape}")
    if source.size == 0 or not np.issubdtype(source.dtype, np.number):
        raise UniVTACContractError(f"RGB artifact has invalid dtype {source.dtype}")
    if not np.isfinite(source).all():
        raise UniVTACContractError("RGB artifact contains non-finite values")
    minimum = float(source.min())
    maximum = float(source.max())
    if minimum < 0:
        raise UniVTACContractError(f"RGB artifact has negative values: min={minimum}")
    if np.issubdtype(source.dtype, np.floating) and maximum <= 1.0:
        stored = np.rint(source * 255.0).astype(np.uint8)
    elif maximum <= 255.0:
        stored = np.rint(source).astype(np.uint8)
    else:
        raise UniVTACContractError(f"RGB artifact exceeds PNG range: max={maximum}")
    return source, stored


def save_png_artifact(value: Any, path: Path, output_root: Path) -> ArtifactRef:
    source, stored = _to_uint8_rgb(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(stored, mode="RGB").save(path)
    summary = array_summary(source)
    return ArtifactRef(
        path=_relative_path(path, output_root),
        shape=tuple(summary["shape"]),
        dtype=summary["dtype"],
        value_range=tuple(summary["range"]),
        encoding="png",
        stored_dtype=str(stored.dtype),
    )


def save_npy_artifact(value: Any, path: Path, output_root: Path) -> ArtifactRef:
    array = to_numpy(value)
    summary = array_summary(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    return ArtifactRef(
        path=_relative_path(path, output_root),
        shape=tuple(summary["shape"]),
        dtype=summary["dtype"],
        value_range=tuple(summary["range"]),
        encoding="npy",
        stored_dtype=str(array.dtype),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise UniVTACContractError("metadata contains a non-finite float")
        return value
    array = to_numpy(value)
    if array.ndim == 0:
        return _json_safe(array.item())
    if array.size > 256:
        raise UniVTACContractError(
            f"metadata array is too large to embed in JSON: shape={array.shape}"
        )
    return array.tolist()


def capture_snapshot(
    observation: Mapping[str, Any],
    *,
    output_root: Path,
    seed_dir: Path,
    task_name: str,
    seed: int,
    phase: str,
    action_id: str,
    simulator_step: int,
    take_action_count: int,
    task_instruction: str,
    task_metadata: Mapping[str, Any],
    native_check_success: bool | None,
    save_host_only: bool,
    strict_two_tactile_sensors: bool,
    fail_on_missing_rgb_marker: bool,
) -> SnapshotCapture:
    """Project a native UniVTAC observation and persist referenced artifacts."""

    sensor_names = validate_tactile_packets(
        observation,
        strict_two_tactile_sensors=strict_two_tactile_sensors,
        fail_on_missing_rgb_marker=fail_on_missing_rgb_marker,
    )
    cameras = observation.get("observation")
    if not isinstance(cameras, Mapping) or "head" not in cameras:
        raise UniVTACContractError("observation must contain external head camera")
    if not isinstance(cameras["head"], Mapping) or "rgb" not in cameras["head"]:
        raise UniVTACContractError("observation.head.rgb is missing")
    embodiment = observation.get("embodiment")
    if not isinstance(embodiment, Mapping):
        raise UniVTACContractError("observation.embodiment is missing")
    for required in ("joint", "ee"):
        if required not in embodiment:
            raise UniVTACContractError(f"observation.embodiment.{required} is missing")

    phase_dir = seed_dir / ("pre" if phase == "pre_action" else "post")
    operator_camera: dict[str, Any] = {}
    camera_metadata: dict[str, Any] = {}
    operator_artifacts: list[str] = []
    for camera_name, packet in sorted(cameras.items()):
        if not isinstance(packet, Mapping) or "rgb" not in packet:
            continue
        safe_name = _safe_component(str(camera_name))
        artifact = save_png_artifact(
            packet["rgb"], phase_dir / "camera" / f"{safe_name}_rgb.png", output_root
        )
        operator_camera[str(camera_name)] = {"rgb": artifact.to_dict()}
        camera_metadata[str(camera_name)] = {
            "available_fields": sorted(str(key) for key in packet),
            "rgb": array_summary(packet["rgb"]),
        }
        operator_artifacts.append(artifact.path)

    tactile_packets = observation["tactile"]
    operator_tactile: dict[str, Any] = {}
    tactile_metadata: dict[str, Any] = {}
    rgb_markers: dict[str, np.ndarray] = {}
    host_tactile: dict[str, Any] = {}
    host_artifacts: list[str] = []
    for sensor_name in sensor_names:
        packet = tactile_packets[sensor_name]
        safe_name = _safe_component(sensor_name)
        marker_array = to_numpy(packet["rgb_marker"]).copy()
        marker_artifact = save_png_artifact(
            marker_array,
            phase_dir / "tactile" / f"{safe_name}_rgb_marker.png",
            output_root,
        )
        operator_tactile[sensor_name] = {"rgb_marker": marker_artifact.to_dict()}
        tactile_metadata[sensor_name] = {
            "available_fields": sorted(str(key) for key in packet),
            "rgb_marker": array_summary(marker_array),
        }
        if save_host_only and "press_depth" in packet:
            tactile_metadata[sensor_name]["press_depth"] = array_summary(
                packet["press_depth"]
            )
        rgb_markers[sensor_name] = marker_array
        operator_artifacts.append(marker_artifact.path)

        if save_host_only:
            sensor_host: dict[str, Any] = {}
            reference = save_png_artifact(
                packet["rgb"],
                seed_dir / "host_only" / phase / "tactile" / f"{safe_name}_rgb.png",
                output_root,
            )
            sensor_host["rgb"] = reference.to_dict()
            host_artifacts.append(reference.path)
            for source_key, contract_key in (
                ("depth", "depth"),
                ("marker", "native_marker_motion"),
                ("pose", "tactile_pose"),
            ):
                if source_key not in packet:
                    raise IncompleteTactilePacketError(
                        f"{sensor_name} is missing required host-only field {source_key}"
                    )
                reference = save_npy_artifact(
                    packet[source_key],
                    seed_dir
                    / "host_only"
                    / phase
                    / "tactile"
                    / f"{safe_name}_{contract_key}.npy",
                    output_root,
                )
                sensor_host[contract_key] = reference.to_dict()
                host_artifacts.append(reference.path)
            if "press_depth" in packet:
                reference = save_npy_artifact(
                    packet["press_depth"],
                    seed_dir
                    / "host_only"
                    / phase
                    / "tactile"
                    / f"{safe_name}_press_depth.npy",
                    output_root,
                )
                sensor_host["press_depth"] = reference.to_dict()
                host_artifacts.append(reference.path)
            host_tactile[sensor_name] = sensor_host

    operator_proprio: dict[str, Any] = {}
    proprio_metadata: dict[str, Any] = {}
    for name in ("joint", "ee"):
        array = to_numpy(embodiment[name])
        summary = array_summary(array)
        operator_proprio[name] = array.tolist()
        proprio_metadata[name] = summary

    operator_visible = {
        "step_identifiers": {
            "snapshot_id": f"{action_id}:{phase}",
            "action_id": action_id,
            "phase": phase,
            "simulator_step": int(simulator_step),
            "take_action_count": int(take_action_count),
        },
        "task_instruction": str(task_instruction),
        "cameras": operator_camera,
        "tactile": operator_tactile,
        "proprio": operator_proprio,
    }

    host_only: dict[str, Any] = {}
    if save_host_only:
        host_only = {
            "tactile": host_tactile,
            "actor_state": {},
            "task_metadata": _json_safe(copy.deepcopy(dict(task_metadata))),
        }
        actor = observation.get("actor")
        if isinstance(actor, Mapping):
            for actor_name, actor_pose in sorted(actor.items()):
                reference = save_npy_artifact(
                    actor_pose,
                    seed_dir
                    / "host_only"
                    / phase
                    / "actor"
                    / f"{_safe_component(str(actor_name))}_actor_pose.npy",
                    output_root,
                )
                host_only["actor_state"][str(actor_name)] = {
                    "actor_pose": reference.to_dict()
                }
                host_artifacts.append(reference.path)
        if native_check_success is not None:
            host_only["native_check_success"] = bool(native_check_success)

    snapshot = UniVTACTaskSnapshot(
        snapshot_id=f"{action_id}:{phase}",
        task_name=task_name,
        seed=int(seed),
        phase=phase,
        action_id=action_id,
        simulator_step=int(simulator_step),
        take_action_count=int(take_action_count),
        task_instruction=str(task_instruction),
        external_camera=camera_metadata,
        tactile_sensors=tactile_metadata,
        proprio=proprio_metadata,
        operator_visible=operator_visible,
        host_only=host_only,
        artifacts={
            "operator_visible": sorted(operator_artifacts),
            "host_only": sorted(host_artifacts),
        },
    )
    return SnapshotCapture(
        snapshot=snapshot,
        rgb_markers=rgb_markers,
        observation_key_tree=observation_key_tree(observation),
    )
