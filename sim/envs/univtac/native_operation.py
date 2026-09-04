"""Pure contracts and trace helpers for native UniVTAC operation runs."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sim.envs.univtac.contract import UniVTACContractError, validate_transition_binding
from sim.envs.univtac.observation import SnapshotCapture, save_npy_artifact, save_png_artifact
from sim.envs.univtac.planner_diagnostics import safe_diagnostic_value

EXPERT_SEEDS = (1_000_000, 1_000_001, 1_000_002)
EXPERT_SEGMENTS = ("align_key", "settle_alignment", "pull_key_out")
TASK_NAME = "pull_out_key"
TASK_INSTRUCTION = "Pull the key out of the slot."


def validate_native_expert_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the fixed R1.0 native expert development protocol."""

    required = {
        "task",
        "seeds",
        "task_config",
        "mode",
        "device",
        "task_instruction",
        "save_host_only",
        "strict_two_tactile_sensors",
        "timeout_seconds",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise UniVTACContractError(f"native expert config is missing: {missing}")
    expected = {
        "task": TASK_NAME,
        "seeds": list(EXPERT_SEEDS),
        "task_config": "demo",
        "mode": "collect",
        "device": "cuda:0",
        "task_instruction": TASK_INSTRUCTION,
        "save_host_only": True,
        "strict_two_tactile_sensors": True,
    }
    for key, value in expected.items():
        if config[key] != value:
            raise UniVTACContractError(
                f"native expert protocol requires {key}={value!r}, got {config[key]!r}"
            )
    timeout = float(config["timeout_seconds"])
    if timeout <= 0:
        raise UniVTACContractError("timeout_seconds must be positive")
    validated = dict(config)
    validated["timeout_seconds"] = timeout
    return validated


def segment_name(move_index: int) -> str:
    if move_index < 1:
        raise UniVTACContractError("move index is one-based")
    if move_index <= len(EXPERT_SEGMENTS):
        return EXPERT_SEGMENTS[move_index - 1]
    return f"unexpected_move_{move_index}"


def serialize_native_action(action: Any) -> dict[str, Any]:
    """Serialize one native Action without importing its implementation."""

    target_pose = getattr(action, "target_pose", None)
    pose: dict[str, Any] | None = None
    if target_pose is not None:
        pose = {
            "position": safe_diagnostic_value(getattr(target_pose, "p", None)),
            "quaternion": safe_diagnostic_value(getattr(target_pose, "q", None)),
        }
    return {
        "repr": str(action),
        "action_type": str(getattr(action, "action", type(action).__name__)),
        "target_pose": pose,
        "target_gripper_position": safe_diagnostic_value(
            getattr(action, "target_gripper_pos", None)
        ),
        "args": safe_diagnostic_value(copy.deepcopy(getattr(action, "args", {}))),
    }


def _tactile_differences(
    *,
    output_root: Path,
    transition_dir: Path,
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], list[str]]:
    if set(before) != set(after):
        raise UniVTACContractError("native move pre/post tactile sensor sets differ")
    records: dict[str, Any] = {}
    paths: list[str] = []
    for sensor_name in sorted(before):
        pre = np.asarray(before[sensor_name])
        post = np.asarray(after[sensor_name])
        if pre.shape != post.shape:
            raise UniVTACContractError(
                f"native move tactile shape mismatch for {sensor_name}: {pre.shape} != {post.shape}"
            )
        difference = np.abs(post.astype(np.float64) - pre.astype(np.float64))
        changed = np.any(difference > 0, axis=-1)
        safe_name = "".join(
            character if character.isalnum() or character in "_.-" else "_"
            for character in sensor_name
        )
        npy_ref = save_npy_artifact(
            difference,
            transition_dir / "difference" / f"{safe_name}_abs_rgb_marker_difference.npy",
            output_root,
        )
        png_ref = save_png_artifact(
            difference,
            transition_dir / "difference" / f"{safe_name}_abs_rgb_marker_difference.png",
            output_root,
        )
        paths.extend((npy_ref.path, png_ref.path))
        records[sensor_name] = {
            "mean_absolute_rgb_marker_difference": float(difference.mean()),
            "max_absolute_rgb_marker_difference": float(difference.max()),
            "changed_pixel_ratio": float(changed.mean()),
            "absolute_difference_npy": npy_ref.to_dict(),
            "absolute_difference_png": png_ref.to_dict(),
        }
    return records, sorted(paths)


def build_native_move_transition(
    *,
    output_root: Path,
    transition_dir: Path,
    seed: int,
    move_index: int,
    semantic_segment: str,
    before: SnapshotCapture,
    after: SnapshotCapture,
    native_actions_before: Sequence[Mapping[str, Any]],
    native_actions_after: Sequence[Mapping[str, Any]],
    move_args: Any,
    move_kwargs: Any,
    move_returned: Any,
    plan_success_before: bool,
    plan_success_after: bool,
    planner_call_indices: Sequence[int],
    post_settle_delay_steps: int,
    enforce_native_order: bool = True,
) -> dict[str, Any]:
    """Build one move-level transition from the unmodified native call."""

    validate_transition_binding(before.snapshot, after.snapshot)
    if enforce_native_order and semantic_segment != segment_name(move_index):
        raise UniVTACContractError("semantic segment does not match native move order")
    tactile, artifacts = _tactile_differences(
        output_root=output_root,
        transition_dir=transition_dir,
        before=before.rgb_markers,
        after=after.rgb_markers,
    )
    return {
        "schema_version": "openeta.univtac.native_move_transition.v1",
        "task": TASK_NAME,
        "seed": int(seed),
        "action_id": before.snapshot.action_id,
        "semantic_segment": semantic_segment,
        "native_move_call_index": int(move_index),
        "before_snapshot_id": before.snapshot.snapshot_id,
        "after_snapshot_id": after.snapshot.snapshot_id,
        "before_snapshot_path": "snapshot_before.json",
        "after_snapshot_path": "snapshot_after.json",
        "simulator_step_range": [
            before.snapshot.simulator_step,
            after.snapshot.simulator_step,
        ],
        "take_action_count_range": [
            before.snapshot.take_action_count,
            after.snapshot.take_action_count,
        ],
        "native_actions_before": [copy.deepcopy(dict(item)) for item in native_actions_before],
        "native_actions_after": [copy.deepcopy(dict(item)) for item in native_actions_after],
        "move_args": safe_diagnostic_value(move_args),
        "move_kwargs": safe_diagnostic_value(move_kwargs),
        "move_returned": safe_diagnostic_value(move_returned),
        "plan_success_before": bool(plan_success_before),
        "plan_success_after": bool(plan_success_after),
        "planner_call_indices": [int(index) for index in planner_call_indices],
        "post_settle_delay_steps": int(post_settle_delay_steps),
        "tactile_differences": tactile,
        "artifacts": {"difference": artifacts},
        "host_only": {"native_outcome": None},
    }


def attach_native_outcome(
    transition: Mapping[str, Any], outcome: Mapping[str, Any]
) -> dict[str, Any]:
    updated = copy.deepcopy(dict(transition))
    updated.setdefault("host_only", {})["native_outcome"] = copy.deepcopy(dict(outcome))
    return updated


def expert_episode_success(
    *, plan_success: bool, native_check_success: bool, native_check_early_stop: bool | None
) -> bool:
    return bool(
        plan_success
        and native_check_success
        and native_check_early_stop is False
    )


def summarize_native_expert_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if [int(row["seed"]) for row in rows] != list(EXPERT_SEEDS):
        raise UniVTACContractError("expert results must contain the three fixed seeds in order")
    successes = sum(bool(row["expert_episode_success"]) for row in rows)
    return {
        "schema_version": "openeta.univtac.native_expert_summary.v1",
        "task": TASK_NAME,
        "seeds": list(EXPERT_SEEDS),
        "native_expert_success_count": successes,
        "native_expert_success_rate": successes / len(EXPERT_SEEDS),
        "development_baseline_label": "3-seed native expert development baseline",
        "agent_stage_allowed": successes >= 1,
        "results": [copy.deepcopy(dict(row)) for row in rows],
    }
