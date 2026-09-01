from __future__ import annotations

import numpy as np
import pytest

from sim.envs.univtac.contract import PrivilegedVisibilityError, UniVTACTaskSnapshot
from sim.envs.univtac.observation import (
    IncompleteTactilePacketError,
    validate_tactile_packets,
)


def _base_snapshot(**overrides):
    payload = {
        "snapshot_id": "action-1:pre_action",
        "task_name": "insert_hole",
        "seed": 0,
        "phase": "pre_action",
        "action_id": "action-1",
        "simulator_step": 1,
        "take_action_count": 0,
        "task_instruction": "insert hole",
        "external_camera": {},
        "tactile_sensors": {},
        "proprio": {},
        "operator_visible": {
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
            "proprio": {"joint": [], "ee": []},
        },
        "host_only": {},
        "artifacts": {},
    }
    payload.update(overrides)
    return UniVTACTaskSnapshot(**payload)


@pytest.mark.parametrize(
    "privileged_key",
    [
        "tactile_pose",
        "actor_pose",
        "target_pose",
        "hole_pose",
        "relative_object_pose",
        "check_success",
        "eval_success",
        "plan_success",
    ],
)
def test_operator_visible_rejects_privileged_keys(privileged_key: str) -> None:
    with pytest.raises(PrivilegedVisibilityError, match=privileged_key):
        visible = _base_snapshot().operator_visible
        visible[privileged_key] = [0.0]
        _base_snapshot(operator_visible=visible)


def test_operator_visible_rejects_non_allowlisted_structure() -> None:
    visible = _base_snapshot().operator_visible
    visible["actor"] = {"pose": [0.0] * 7}
    with pytest.raises(PrivilegedVisibilityError, match="top-level keys"):
        _base_snapshot(operator_visible=visible)


def test_complete_tactile_packet_is_required() -> None:
    observation = {
        "tactile": {
            "sensor_a": {"rgb_marker": np.zeros((4, 5, 3), dtype=np.uint8)},
            "sensor_b": {},
        }
    }
    with pytest.raises(IncompleteTactilePacketError, match="missing fields"):
        validate_tactile_packets(
            observation,
            strict_two_tactile_sensors=True,
            fail_on_missing_rgb_marker=True,
        )


def test_rgb_marker_shape_and_sensor_count_are_strict() -> None:
    def packet(rgb_marker):
        return {
            "rgb": np.zeros((4, 5, 3), dtype=np.uint8),
            "rgb_marker": rgb_marker,
            "marker": np.zeros((2, 2, 2), dtype=np.float32),
            "depth": np.zeros((4, 5), dtype=np.float32),
            "pose": np.zeros((7,), dtype=np.float32),
        }

    one_sensor = {
        "tactile": {
            "sensor_a": packet(np.zeros((4, 5, 3), dtype=np.uint8))
        }
    }
    with pytest.raises(IncompleteTactilePacketError, match="exactly two"):
        validate_tactile_packets(
            one_sensor,
            strict_two_tactile_sensors=True,
            fail_on_missing_rgb_marker=True,
        )

    bad_shape = {
        "tactile": {
            "sensor_a": packet(np.zeros((4, 5), dtype=np.uint8)),
            "sensor_b": packet(np.zeros((4, 5, 3), dtype=np.uint8)),
        }
    }
    with pytest.raises(IncompleteTactilePacketError, match="HxWx3"):
        validate_tactile_packets(
            bad_shape,
            strict_two_tactile_sensors=True,
            fail_on_missing_rgb_marker=True,
        )
