from __future__ import annotations

import json

import pytest

from sim.envs.univtac.contract import (
    ArtifactRef,
    TactileTransition,
    UniVTACContractError,
    UniVTACTaskSnapshot,
    validate_transition_binding,
)


def _snapshot(
    *, phase: str = "pre_action", action_id: str = "action-1"
) -> UniVTACTaskSnapshot:
    def artifact(path: str):
        return {
            "path": path,
            "shape": [4, 5, 3],
            "dtype": "uint8",
            "value_range": [0.0, 255.0],
            "encoding": "png",
            "stored_dtype": "uint8",
        }

    return UniVTACTaskSnapshot(
        snapshot_id=f"{action_id}:{phase}",
        task_name="insert_hole",
        seed=0,
        phase=phase,
        action_id=action_id,
        simulator_step=10 if phase == "pre_action" else 11,
        take_action_count=0,
        task_instruction="insert hole",
        external_camera={"head": {"shape": [360, 640, 3]}},
        tactile_sensors={
            "sensor_a": {"rgb_marker": {"shape": [300, 400, 3]}},
            "sensor_b": {"rgb_marker": {"shape": [300, 400, 3]}},
        },
        proprio={"joint": {"shape": [9]}, "ee": {"shape": [7]}},
        operator_visible={
            "step_identifiers": {
                "snapshot_id": f"{action_id}:{phase}",
                "action_id": action_id,
                "phase": phase,
                "simulator_step": 10 if phase == "pre_action" else 11,
                "take_action_count": 0,
            },
            "task_instruction": "insert hole",
            "cameras": {"head": {"rgb": artifact("seed_0/pre/head.png")}},
            "tactile": {
                "sensor_a": {
                    "rgb_marker": artifact("seed_0/pre/sensor_a_marker.png")
                }
            },
            "proprio": {"joint": [0.0] * 9, "ee": [0.0] * 7},
        },
        host_only={"tactile": {"sensor_a": {"depth": {"path": "host/a.npy"}}}},
        artifacts={
            "operator_visible": ["seed_0/pre/head.png"],
            "host_only": ["host/a.npy"],
        },
    )


def test_snapshot_json_serialization_is_deterministic() -> None:
    first = _snapshot()
    second = UniVTACTaskSnapshot.from_dict(
        json.loads(json.dumps(first.to_dict(), sort_keys=False))
    )

    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == first.to_dict()


def test_artifact_paths_must_be_relative() -> None:
    with pytest.raises(UniVTACContractError, match="must be relative"):
        ArtifactRef(
            path="/tmp/tactile.png",
            shape=(4, 4, 3),
            dtype="uint8",
            value_range=(0.0, 255.0),
            encoding="png",
            stored_dtype="uint8",
        )

    with pytest.raises(UniVTACContractError, match="must be relative"):
        ArtifactRef(
            path="seed_0/../escape.npy",
            shape=(1,),
            dtype="float32",
            value_range=(0.0, 0.0),
            encoding="npy",
            stored_dtype="float32",
        )


def test_operator_and_host_payloads_do_not_share_mutable_objects() -> None:
    shared = []
    snapshot = UniVTACTaskSnapshot(
        snapshot_id="action-1:pre_action",
        task_name="insert_hole",
        seed=0,
        phase="pre_action",
        action_id="action-1",
        simulator_step=1,
        take_action_count=0,
        task_instruction="insert hole",
        external_camera={},
        tactile_sensors={},
        proprio={},
        operator_visible={
            "step_identifiers": {
                "snapshot_id": "action-1:pre_action",
                "action_id": "action-1",
                "phase": "pre_action",
                "simulator_step": 1,
                "take_action_count": 0,
            },
            "task_instruction": "insert hole",
            "cameras": {},
            "tactile": {},
            "proprio": {"joint": shared, "ee": []},
        },
        host_only={"payload": shared},
        artifacts={},
    )

    snapshot.operator_visible["proprio"]["joint"].append("operator")

    assert snapshot.host_only["payload"] == []
    assert shared == []


def test_pre_post_transition_requires_one_action_id() -> None:
    pre = _snapshot(phase="pre_action", action_id="action-1")
    post = _snapshot(phase="post_action", action_id="action-1")
    transition = TactileTransition(
        action_id="action-1",
        action_type="task.move/task.atom.move_by_displacement",
        requested_displacement_mm=(0.0, 0.0, -2.0),
        pre_snapshot_id=pre.snapshot_id,
        post_snapshot_id=post.snapshot_id,
        pre_simulator_step=pre.simulator_step,
        post_simulator_step=post.simulator_step,
        pre_take_action_count=0,
        post_take_action_count=0,
        tactile_differences={},
        host_only={"native_check_success": False},
    )
    validate_transition_binding(pre, post, transition)

    mismatched_post = _snapshot(phase="post_action", action_id="action-2")
    with pytest.raises(UniVTACContractError, match="action_id mismatch"):
        validate_transition_binding(pre, mismatched_post)
