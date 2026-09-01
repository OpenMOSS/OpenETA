from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.univtac.probe_ftp1_eval_seeds import (
    FIXED_PROBE_SEEDS,
    PRIMARY_TASKS,
    PROBE_IMPORT_FLAGS,
    classify_task,
    reset_is_valid,
    should_run_conditional_control,
    validate_author_bundle,
    validate_probe_config,
)
from sim.envs.univtac.planner_diagnostics import PlannerDiagnosticRecorder


REPO_ROOT = Path(__file__).resolve().parents[2]


def _config() -> dict[str, object]:
    return {
        "canonical_commit": "89fa681d6c014cce28300946b7526db808e0b1c1",
        "tasks": list(PRIMARY_TASKS),
        "seeds": list(FIXED_PROBE_SEEDS),
        "conditional_control": {
            "task": "lift_can",
            "seeds": list(FIXED_PROBE_SEEDS),
        },
        "required_valid_seeds": 3,
        "gpu": 0,
    }


def _record(seed: int, *, valid: bool = False, stage: str | None = None) -> dict:
    return {
        "task": "insert_hole",
        "seed": seed,
        "reset_returned": valid,
        "plan_success": valid,
        "observation_available": valid,
        "tactile_sensor_names": ["left", "right"] if valid else [],
        "rgb_marker_valid": valid,
        "reached_reset": stage != "startup_before_reset",
        "failure_stage": stage,
        "first_failed_call": 4 if stage == "reset_pre_move_planner" else None,
        "status": "INVALID_PARTIAL_POSE_COST_METRIC" if stage else None,
        "constraint_pose": [None, None, None, None, None, 0.2] if stage else None,
    }


def test_probe_seeds_are_fixed_and_cannot_be_replaced_or_extended() -> None:
    validate_probe_config(_config())
    changed = _config()
    changed["seeds"] = [1000000, 1000001, 1000003]
    with pytest.raises(ValueError, match="exactly"):
        validate_probe_config(changed)


def test_reset_validity_ignores_task_and_policy_success() -> None:
    record = _record(1000000, valid=True)
    record["check_success"] = False
    record["policy_success"] = False
    assert reset_is_valid(record) is True


def test_exact_startup_failure_is_not_a_planner_failure() -> None:
    startup_records = [
        _record(seed, stage="startup_before_reset") for seed in FIXED_PROBE_SEEDS
    ]
    planner_records = [
        _record(seed, stage="reset_pre_move_planner") for seed in FIXED_PROBE_SEEDS
    ]
    assert classify_task(startup_records)["classification"] == "startup_blocked"
    assert (
        classify_task(planner_records)["classification"]
        == "all_three_same_planner_failure"
    )
    runtime_records = [
        _record(seed, stage="reset_runtime_error") for seed in FIXED_PROBE_SEEDS
    ]
    assert classify_task(runtime_records)["classification"] == "runtime_error"


def test_probe_contract_never_loads_ftp1_openpi_or_openeta() -> None:
    assert PROBE_IMPORT_FLAGS == {
        "ftp1_model_loaded": False,
        "openpi_imported": False,
        "openeta_imported": False,
        "planner_behavior_modified": False,
    }


def test_planner_recorder_preserves_arguments_and_return_identity() -> None:
    sentinel = object()
    calls = []

    class Planner:
        def plan_path(self, *args, **kwargs):
            calls.append((args, kwargs))
            return sentinel

    class Task:
        plan_success = True
        step_count = 0
        atom_id = 0
        atom_tag = ""
        take_action_cnt = 0

        def __init__(self) -> None:
            self._robot_manager = SimpleNamespace(
                planner=Planner(), get_ee_pose=lambda: None
            )

        def move(self, actions, *args, **kwargs):
            return True

    task = Task()
    recorder = PlannerDiagnosticRecorder(task)
    recorder.install()
    result = task._robot_manager.planner.plan_path(
        "q", "qd", "target", "robot", pre_dis=0.1, constraint_pose=[None] * 6,
        time_dilation_factor=0.2,
    )

    assert result is sentinel
    assert calls == [
        (
            ("q", "qd", "target", "robot"),
            {
                "pre_dis": 0.1,
                "constraint_pose": [None] * 6,
                "time_dilation_factor": 0.2,
            },
        )
    ]


def test_comparison_keeps_every_seed_instead_of_an_average() -> None:
    summary = classify_task([_record(seed, valid=True) for seed in FIXED_PROBE_SEEDS])
    assert set(summary["seeds"]) == {str(seed) for seed in FIXED_PROBE_SEEDS}
    assert "average" not in summary


def test_lift_can_triggers_only_when_all_nine_primary_runs_are_invalid() -> None:
    invalid = [
        {**_record(seed), "task": task}
        for task in PRIMARY_TASKS
        for seed in FIXED_PROBE_SEEDS
    ]
    assert should_run_conditional_control(invalid) is True
    invalid[0] = {**invalid[0], **_record(1000000, valid=True)}
    assert should_run_conditional_control(invalid) is False
    assert should_run_conditional_control(invalid[:-1]) is False


def test_author_bundle_rejects_binary_large_or_secret_content(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("safe reproduction", encoding="utf-8")
    validate_author_bundle(tmp_path)

    (tmp_path / "weights.pt").write_bytes(b"binary")
    with pytest.raises(ValueError, match="disallowed"):
        validate_author_bundle(tmp_path)
    (tmp_path / "weights.pt").unlink()
    (tmp_path / "runtime.json").write_text("API_KEY=not-allowed", encoding="utf-8")
    with pytest.raises(ValueError, match="credential"):
        validate_author_bundle(tmp_path)
    (tmp_path / "runtime.json").unlink()
    (tmp_path / "token.txt").write_text("API_KEY=still-not-allowed", encoding="utf-8")
    with pytest.raises(ValueError, match="credential"):
        validate_author_bundle(tmp_path)


def test_vendored_univtac_tracked_diff_is_empty() -> None:
    completed = subprocess.run(
        ["git", "diff", "--quiet", "--", "third_party/ftp1-policy/UniVTAC"],
        cwd=REPO_ROOT,
        check=False,
    )
    assert completed.returncode == 0
