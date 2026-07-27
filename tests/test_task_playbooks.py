from __future__ import annotations

import json
from pathlib import Path

import pytest

from adapter.protocol import EnvObservation, RobotState
from agent.runtime.memory import AgentMemory
from agent.runtime.planner import _matched_task_playbook
from agent.runtime.task_playbooks import (
    DEFAULT_TASK_PLAYBOOK_ROOT,
    TaskPlaybookError,
    extract_task_playbook_candidate,
    load_task_playbooks,
    select_task_playbook,
    task_text_sha256,
    validate_task_playbook,
)


def test_default_playbook_matches_exact_task_and_calibration() -> None:
    selected = select_task_playbook(
        load_task_playbooks(),
        environment_id="openeta/libero_libero_object_task0-v0",
        suite="libero_object",
        task_index=0,
        task="pick up the alphabet soup and place it in the basket",
        calibration_id="graspnet-eef-panda-p8",
    )

    assert selected is not None
    assert selected["playbook_id"] == "libero-object-task0-alphabet-soup"
    assert selected["status"] == "candidate"
    assert selected["guidance"]["object_priors"][0]["canonical_asset_key"] == (
        "libero/alphabet_soup"
    )


def test_playbook_does_not_match_changed_task_or_calibration() -> None:
    playbooks = load_task_playbooks()

    assert (
        select_task_playbook(
            playbooks,
            environment_id="openeta/libero_libero_object_task0-v0",
            suite="libero_object",
            task_index=0,
            task="pick up the tomato sauce and place it in the basket",
            calibration_id="graspnet-eef-panda-p8",
        )
        is None
    )
    assert (
        select_task_playbook(
            playbooks,
            environment_id="openeta/libero_libero_object_task0-v0",
            suite="libero_object",
            task_index=0,
            task="pick up the alphabet soup and place it in the basket",
            calibration_id="other-calibration",
        )
        is None
    )


def test_planner_context_match_uses_workspace_playbook_registry() -> None:
    task = "pick up the alphabet soup and place it in the basket"
    memory = AgentMemory()
    memory.start_session(
        task=task,
        metadata={
            "workspace": {
                "task_playbook_root": str(DEFAULT_TASK_PLAYBOOK_ROOT),
                "grasp_profile_id": "graspnet-eef-panda-p8",
            }
        },
    )
    observation = EnvObservation(
        task=task,
        cameras=[],
        robot=RobotState(end_effector_pose={"xyz": [0.0, 0.0, 0.5]}),
        objects=[],
        metadata={
            "env_id": "openeta/libero_libero_object_task0-v0",
            "suite": "libero_object",
            "task_index": 0,
        },
    )

    selected = _matched_task_playbook(
        observation=observation,
        memory=memory,
        task=task,
    )

    assert selected is not None
    assert selected["playbook_id"] == "libero-object-task0-alphabet-soup"


def test_playbook_rejects_executable_world_pose_guidance() -> None:
    payload = json.loads(
        (DEFAULT_TASK_PLAYBOOK_ROOT / "candidate/libero-object-task0-alphabet-soup.json").read_text(
            encoding="utf-8"
        )
    )
    payload["guidance"]["target_pose"] = {"frame": "world", "xyz": [1, 2, 3]}

    with pytest.raises(TaskPlaybookError, match="target_pose"):
        validate_task_playbook(payload)


def test_success_rollout_extracts_non_executable_candidate(tmp_path: Path) -> None:
    rollout = tmp_path / "tool_calls.jsonl"
    records = [
        {
            "event": {
                "phase": "start",
                "name": "retrieve_asset_reference",
                "parameters": {"target_object": "test can"},
            }
        },
        {
            "event": {
                "phase": "start",
                "name": "compile_grasp_seed",
                "parameters": {
                    "target_geometry_family": "upright_can",
                    "strategy_id": "top-down",
                    "camera_pose": {"width": 0.06, "source_backend": "anygrasp"},
                },
            }
        },
        {
            "event": {
                "phase": "start",
                "name": "move_to",
                "parameters": {"target_pose": {"grasp_stage": "full_lift", "xyz": [1, 2, 3]}},
            }
        },
    ]
    rollout.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    outcome = {
        "episode_id": "episode-1",
        "env_id": "openeta/test-v0",
        "session_id": "session-1",
        "status": "success",
        "assistance": {"assisted": False},
        "episode": {
            "task": "pick up test can",
            "session_id": "session-1",
            "metadata": {},
            "steps": [
                {
                    "observation": {
                        "metadata": {
                            "env_id": "openeta/test-v0",
                            "suite": "test_suite",
                            "task_index": 2,
                            "calibration_profile_id": "calibration",
                        }
                    },
                    "step_result": {"reward": 1.0},
                }
            ],
        },
    }

    candidate = extract_task_playbook_candidate(
        outcome=outcome,
        rollout_tool_calls=rollout,
    )

    assert candidate["scope"]["task_text_sha256"] == task_text_sha256("pick up test can")
    assert candidate["guidance"]["observed_object_queries"] == ["test can"]
    assert candidate["guidance"]["successful_stage_sequence"] == ["full_lift"]
    assert "xyz" not in json.dumps(candidate["guidance"])
