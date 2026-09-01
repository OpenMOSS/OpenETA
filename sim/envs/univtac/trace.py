"""Trace persistence and tactile pre/post binding for UniVTAC smoke tests."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sim.envs.univtac.contract import (
    TactileTransition,
    UniVTACContractError,
    UniVTACTaskSnapshot,
    resolve_artifact_path,
    validate_transition_binding,
)
from sim.envs.univtac.observation import save_npy_artifact, save_png_artifact


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic strict JSON while keeping partial trace directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_snapshot(path: Path, snapshot: UniVTACTaskSnapshot) -> None:
    write_json(path, snapshot.to_dict())


def write_exception(path: Path, exc: BaseException) -> dict[str, Any]:
    payload = {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    write_json(path, payload)
    return payload


def _iter_artifact_paths(value: Any):
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) == "path" and isinstance(child, str):
                yield child
            yield from _iter_artifact_paths(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_artifact_paths(child)


def verify_artifacts(output_root: Path, *payloads: Mapping[str, Any]) -> None:
    missing: list[str] = []
    for payload in payloads:
        for relative_path in _iter_artifact_paths(payload):
            if not resolve_artifact_path(output_root, relative_path).is_file():
                missing.append(relative_path)
    if missing:
        raise UniVTACContractError(f"artifact saving is incomplete: {sorted(set(missing))}")


def build_transition(
    *,
    output_root: Path,
    seed_dir: Path,
    pre: UniVTACTaskSnapshot,
    post: UniVTACTaskSnapshot,
    pre_rgb_markers: Mapping[str, np.ndarray],
    post_rgb_markers: Mapping[str, np.ndarray],
    requested_displacement_mm: tuple[float, float, float],
    native_check_success: bool,
) -> TactileTransition:
    validate_transition_binding(pre, post)
    if set(pre_rgb_markers) != set(post_rgb_markers):
        raise UniVTACContractError(
            "pre/post tactile sensor sets differ: "
            f"{sorted(pre_rgb_markers)} != {sorted(post_rgb_markers)}"
        )

    tactile_differences: dict[str, Any] = {}
    artifact_paths: list[str] = []
    for sensor_name in sorted(pre_rgb_markers):
        before = np.asarray(pre_rgb_markers[sensor_name])
        after = np.asarray(post_rgb_markers[sensor_name])
        if before.shape != after.shape:
            raise UniVTACContractError(
                f"pre/post rgb_marker shape mismatch for {sensor_name}: "
                f"{before.shape} != {after.shape}"
            )
        difference = np.abs(after.astype(np.float64) - before.astype(np.float64))
        changed_pixels = np.any(difference > 0, axis=-1)
        safe_name = "".join(
            character if character.isalnum() or character in "_.-" else "_"
            for character in sensor_name
        )
        npy_ref = save_npy_artifact(
            difference,
            seed_dir / "difference" / f"{safe_name}_abs_rgb_marker_difference.npy",
            output_root,
        )
        png_ref = save_png_artifact(
            difference,
            seed_dir / "difference" / f"{safe_name}_abs_rgb_marker_difference.png",
            output_root,
        )
        artifact_paths.extend([npy_ref.path, png_ref.path])
        tactile_differences[sensor_name] = {
            "mean_absolute_rgb_marker_difference": float(difference.mean()),
            "max_absolute_rgb_marker_difference": float(difference.max()),
            "changed_pixel_ratio": float(changed_pixels.mean()),
            "absolute_difference_npy": npy_ref.to_dict(),
            "absolute_difference_png": png_ref.to_dict(),
        }

    transition = TactileTransition(
        action_id=pre.action_id,
        action_type="task.move/task.atom.move_by_displacement",
        requested_displacement_mm=requested_displacement_mm,
        pre_snapshot_id=pre.snapshot_id,
        post_snapshot_id=post.snapshot_id,
        pre_simulator_step=pre.simulator_step,
        post_simulator_step=post.simulator_step,
        pre_take_action_count=pre.take_action_count,
        post_take_action_count=post.take_action_count,
        tactile_differences=tactile_differences,
        host_only={"native_check_success": bool(native_check_success)},
        artifacts={"difference": sorted(artifact_paths)},
    )
    validate_transition_binding(pre, post, transition)
    return transition
