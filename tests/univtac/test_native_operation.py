from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import yaml

from sim.envs.univtac.contract import UniVTACTaskSnapshot
from sim.envs.univtac.native_operation import (
    EXPERT_SEEDS,
    EXPERT_SEGMENTS,
    build_native_move_transition,
    expert_episode_success,
    segment_name,
    serialize_native_action,
    summarize_native_expert_results,
    validate_native_expert_config,
)
from sim.envs.univtac.observation import SnapshotCapture

REPO_ROOT = Path(__file__).resolve().parents[2]


def _snapshot(*, phase: str, step: int, action_count: int) -> UniVTACTaskSnapshot:
    action_id = "native-expert-1000000-align_key"
    return UniVTACTaskSnapshot(
        snapshot_id=f"{action_id}:{phase}",
        task_name="pull_out_key",
        seed=1_000_000,
        phase=phase,
        action_id=action_id,
        simulator_step=step,
        take_action_count=action_count,
        task_instruction="Pull the key out of the slot.",
        external_camera={},
        tactile_sensors={},
        proprio={},
        operator_visible={
            "cameras": {},
            "tactile": {},
            "proprio": {"joint": [], "ee": []},
            "task_instruction": "Pull the key out of the slot.",
            "step_identifiers": {
                "snapshot_id": f"{action_id}:{phase}",
                "action_id": action_id,
                "phase": phase,
                "simulator_step": step,
                "take_action_count": action_count,
            },
        },
        host_only={},
        artifacts={},
    )


def _capture(*, phase: str, step: int, value: int) -> SnapshotCapture:
    markers = {
        "left_tactile": np.full((2, 3, 3), value, dtype=np.uint8),
        "right_tactile": np.full((2, 3, 3), value + 1, dtype=np.uint8),
    }
    return SnapshotCapture(
        snapshot=_snapshot(phase=phase, step=step, action_count=step),
        rgb_markers=markers,
        observation_key_tree={},
    )


def test_native_expert_config_fixes_seeds_and_protocol() -> None:
    path = REPO_ROOT / "configs/univtac/pull_out_key_native_expert.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert validate_native_expert_config(payload)["seeds"] == list(EXPERT_SEEDS)
    changed = dict(payload, seeds=[1_000_000, 1_000_001, 1_000_003])
    try:
        validate_native_expert_config(changed)
    except ValueError as exc:
        assert "three fixed seeds" not in str(exc) or "requires seeds" in str(exc)
    else:
        raise AssertionError("replacement expert seed must be rejected")


def test_native_segment_order_and_success_semantics() -> None:
    assert tuple(segment_name(index) for index in range(1, 4)) == EXPERT_SEGMENTS
    assert segment_name(4) == "unexpected_move_4"
    assert expert_episode_success(
        plan_success=True,
        native_check_success=True,
        native_check_early_stop=False,
    )
    assert not expert_episode_success(
        plan_success=True,
        native_check_success=True,
        native_check_early_stop=None,
    )
    assert not expert_episode_success(
        plan_success=False,
        native_check_success=True,
        native_check_early_stop=False,
    )


def test_native_action_serialization_and_transition_artifacts(tmp_path: Path) -> None:
    class Pose:
        p = np.array([1.0, 2.0, 3.0])
        q = np.array([0.0, 0.0, 0.0, 1.0])

    class Action:
        action = "move"
        target_pose = Pose()
        target_gripper_pos = None

        def __init__(self) -> None:
            self.args = {"constraint_pose": [0, 0, 0, 1, 1, 1]}

        def __str__(self) -> str:
            return "Action(move)"

    serialized = serialize_native_action(Action())
    assert serialized["action_type"] == "move"
    assert serialized["args"]["constraint_pose"] == [0, 0, 0, 1, 1, 1]

    before = _capture(phase="pre_action", step=10, value=1)
    after = _capture(phase="post_action", step=20, value=4)
    transition = build_native_move_transition(
        output_root=tmp_path,
        transition_dir=tmp_path / "transitions" / "align_key",
        seed=1_000_000,
        move_index=1,
        semantic_segment="align_key",
        before=before,
        after=after,
        native_actions_before=[serialized],
        native_actions_after=[serialized],
        move_args=[],
        move_kwargs={"delay": False},
        move_returned=True,
        plan_success_before=True,
        plan_success_after=True,
        planner_call_indices=[2],
        post_settle_delay_steps=0,
    )
    assert transition["simulator_step_range"] == [10, 20]
    assert transition["tactile_differences"]["left_tactile"][
        "mean_absolute_rgb_marker_difference"
    ] == 3.0
    for path in transition["artifacts"]["difference"]:
        assert not Path(path).is_absolute()
        assert (tmp_path / path).is_file()


def test_summary_keeps_all_fixed_seeds_in_denominator() -> None:
    rows = [
        {"seed": seed, "expert_episode_success": seed == 1_000_001}
        for seed in EXPERT_SEEDS
    ]
    summary = summarize_native_expert_results(rows)
    assert summary["native_expert_success_count"] == 1
    assert summary["native_expert_success_rate"] == 1 / 3
    assert [row["seed"] for row in summary["results"]] == list(EXPERT_SEEDS)


def test_child_calls_native_play_once_and_checkers_once_in_source() -> None:
    path = REPO_ROOT / "scripts/univtac/probe_pull_out_key_native_expert.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert calls.count("play_once") == 1
    assert calls.count("check_success") == 1
    assert calls.count("check_early_stop") == 1
    assert "_play_once" not in calls
