from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from scripts.univtac.run_branching_task_qualification import _episode_row
from sim.envs.univtac.branching_qualification import (
    MAX_SIMULATOR_EPISODES,
    SEEDS,
    TASKS,
    classify_insert_correction,
    expert_segment_name,
    invoke_passthrough,
    qualify_task,
    select_counterfactual,
    select_tactile_icl_task,
    validate_branching_config,
    validate_counterfactual_pair,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _row(seed: int, decision_class: str, *, success: bool = True, **extra):
    return {
        "seed": seed,
        "decision_class": decision_class,
        "expert_episode_success": success,
        "decision_snapshot_complete": True,
        "action_trace_complete": True,
        **extra,
    }


def test_config_fixes_tasks_seeds_and_episode_budget() -> None:
    payload = yaml.safe_load(
        (REPO_ROOT / "configs/univtac/branching_task_qualification.yaml").read_text()
    )
    validated = validate_branching_config(payload)
    assert validated["tasks"] == list(TASKS)
    assert validated["seeds"] == list(SEEDS)
    assert validated["max_simulator_episodes"] == MAX_SIMULATOR_EPISODES == 8
    with pytest.raises(ValueError, match="requires seeds"):
        validate_branching_config({**payload, "seeds": [*SEEDS, 1_000_003]})


def test_native_decision_classes_and_segment_order() -> None:
    assert classify_insert_correction(-0.0011) == "negative_x_correction"
    assert classify_insert_correction(-0.001) == "near_zero_x_correction"
    assert classify_insert_correction(0.0011) == "positive_x_correction"
    assert expert_segment_name(task="insert_hole", move_index=2) == "corrective_xz"
    assert expert_segment_name(task="lift_bottle", move_index=5) == "gripper_rotate_4"
    assert (
        expert_segment_name(task="lift_bottle", move_index=6, lift_mid_success=True)
        == "open_gripper"
    )
    assert (
        expert_segment_name(task="lift_bottle", move_index=6, lift_mid_success=False)
        == "corrective_tilt"
    )


def test_candidate_requires_success_diversity_and_complete_evidence() -> None:
    rows = [
        _row(SEEDS[0], "negative_x_correction", native_x_move=-0.002),
        _row(SEEDS[1], "positive_x_correction", native_x_move=0.002),
        _row(SEEDS[2], "near_zero_x_correction", success=False, native_x_move=0.0),
    ]
    summary = qualify_task("insert_hole", rows)
    assert summary["native_expert_success_count"] == 2
    assert summary["branching_candidate"] is True
    assert [row["seed"] for row in summary["results"]] == list(SEEDS)
    assert not qualify_task("insert_hole", [dict(row, decision_class="near_zero_x_correction") for row in rows])["branching_candidate"]
    incomplete = [*rows[:2], dict(rows[2], decision_snapshot_complete=False, success=True)]
    assert not qualify_task("insert_hole", incomplete)["branching_candidate"]


def test_counterfactual_priority_and_invariants() -> None:
    insert = qualify_task(
        "insert_hole",
        [
            _row(SEEDS[0], "negative_x_correction", native_x_move=-0.002),
            _row(SEEDS[1], "positive_x_correction", native_x_move=0.002),
            _row(SEEDS[2], "positive_x_correction", native_x_move=0.003),
        ],
    )
    lift = qualify_task(
        "lift_bottle",
        [
            _row(SEEDS[0], "correct_pose_then_release", check_mid_success=False),
            _row(SEEDS[1], "direct_release", check_mid_success=True),
            _row(SEEDS[2], "direct_release", check_mid_success=True),
        ],
    )
    assert select_counterfactual({"lift_bottle": lift, "insert_hole": insert}) == {
        "task": "insert_hole",
        "seed": SEEDS[0],
    }
    common = {
        "task": "insert_hole",
        "seed": SEEDS[0],
        "final_insert_z": -0.04,
        "corrective_time_dilation": 0.5,
        "final_time_dilation": 0.5,
    }
    correct = {
        **common,
        "condition": "correct",
        "native_x_move": -0.002,
        "applied_x_move": -0.002,
        "native_z_move": -0.0002,
        "applied_z_move": -0.0002,
        "expert_episode_success": True,
    }
    wrong = {
        **common,
        "condition": "wrong",
        "native_x_move": -0.0021,
        "applied_x_move": 0.0021,
        "native_z_move": -0.00021,
        "applied_z_move": -0.00021,
        "expert_episode_success": False,
    }
    validate_counterfactual_pair(correct, wrong)
    assert select_tactile_icl_task({"task": "insert_hole"}, correct, wrong) == (
        "insert_hole",
        "correct_succeeds_wrong_fails",
    )
    with pytest.raises(ValueError, match="only flip its native x sign"):
        validate_counterfactual_pair(correct, {**wrong, "applied_x_move": 0.003})


def test_passthrough_preserves_arguments_return_and_exception() -> None:
    calls = []
    sentinel = object()

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    observed = invoke_passthrough(
        original,
        (1, 2),
        {"tag": "native"},
        before=lambda args, kwargs: calls.append(("before", args, kwargs)),
        after=lambda result: calls.append(("after", result)),
    )
    assert observed is sentinel
    assert calls[1] == ((1, 2), {"tag": "native"})

    def broken(*_args, **_kwargs):
        raise RuntimeError("native failure")

    with pytest.raises(RuntimeError, match="native failure"):
        invoke_passthrough(broken, (), {}, before=lambda *_: None, after=lambda *_: None)


def test_child_uses_native_expert_entrypoint_and_official_checkers_once() -> None:
    path = REPO_ROOT / "scripts/univtac/probe_branching_task.py"
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
    source = path.read_text(encoding="utf-8")
    assert "codex" not in source.lower()
    assert "artifact_hash" not in source


def test_launcher_cleanup_is_authoritative_when_sim_close_does_not_return(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "episode"
    episode.mkdir()
    (episode / "child_result.json").write_text(
        """{
          "status": "completed",
          "classification": "native_episode_success",
          "cleanup": {
            "task_close": true,
            "simulation_app_close_invoked": true,
            "simulation_app_close_returned": false
          },
          "counters": {}
        }""",
        encoding="utf-8",
    )
    (episode / "final_result.json").write_text(
        """{
          "plan_success": true,
          "native_check_success": true,
          "native_check_early_stop": false,
          "expert_episode_success": true,
          "decision_snapshot_complete": true,
          "action_trace_complete": true
        }""",
        encoding="utf-8",
    )
    row = _episode_row(
        task="insert_hole",
        seed=SEEDS[0],
        condition="expert",
        episode_root=episode,
        lifecycle={"returncode": 0, "timed_out": False, "cleanup_complete": True},
        error=None,
    )
    assert row["infrastructure_valid"] is True
