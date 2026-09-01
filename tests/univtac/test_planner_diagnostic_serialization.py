from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sim.envs.univtac.planner_diagnostics import (
    safe_diagnostic_value,
    serialize_motion_gen_result,
)
from sim.envs.univtac.runtime import (
    HIGH_RES_GELPAD_RELATIVE_PATH,
    HIGH_RES_ROBOT_RELATIVE_PATH,
    UniVTACRuntimeError,
    assert_high_res_assets,
    safe_runtime_path_snapshot,
)


class CompleteResult:
    def __init__(self) -> None:
        self.success = np.asarray(True)
        self.status = "SUCCESS"
        self.valid_query = np.asarray(True)
        self.attempts = 2
        self.solve_time = 0.125
        self.large_payload = np.arange(64)


class MinimalResult:
    def __init__(self) -> None:
        self.success = np.asarray(False)


def test_planner_diagnostic_serialization_is_deterministic() -> None:
    result = CompleteResult()
    first = serialize_motion_gen_result(result)
    second = serialize_motion_gen_result(result)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["status"] == "SUCCESS"
    assert first["attempts"] == 2


def test_missing_optional_motion_gen_fields_do_not_raise() -> None:
    payload = serialize_motion_gen_result(MinimalResult())

    assert "success" in payload
    assert "status" not in payload
    assert "solve_time" not in payload
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_safe_tensor_like_payload_uses_shape_dtype_and_small_values() -> None:
    payload = safe_diagnostic_value(np.asarray([1.0, 2.0], dtype=np.float32))

    assert payload == {
        "shape": [2],
        "dtype": "float32",
        "values": [1.0, 2.0],
    }


def test_asset_validation_rejects_high_res_to_low_res_symlink(tmp_path: Path) -> None:
    robot = tmp_path / HIGH_RES_ROBOT_RELATIVE_PATH
    robot.parent.mkdir(parents=True)
    low_res_robot = robot.parent / "uipc_gelpads.usd"
    low_res_robot.write_bytes(b"low-res")
    robot.symlink_to(low_res_robot.name)

    gelpad = tmp_path / HIGH_RES_GELPAD_RELATIVE_PATH
    gelpad.parent.mkdir(parents=True)
    gelpad.write_bytes(b"high-res")

    with pytest.raises(UniVTACRuntimeError, match="low-resolution alias"):
        assert_high_res_assets(tmp_path)


def test_runtime_path_snapshot_does_not_copy_secrets() -> None:
    marker = "do-not-log-this-secret"
    environment = {
        "WANDB_API_KEY": marker,
        "AUTHORIZATION": marker,
        "LD_LIBRARY_PATH": "/opt/cuda/lib:/ordinary/lib",
    }
    snapshot = safe_runtime_path_snapshot(
        sys_path=["/checkout/UniVTAC", "/ordinary/python"],
        environment=environment,
    )
    encoded = json.dumps(snapshot, sort_keys=True)

    assert marker not in encoded
    assert "WANDB_API_KEY" not in encoded
    assert snapshot["related_ld_library_path_entries"] == ["/opt/cuda/lib"]
