"""Tests for the Robotiq 2F-85 grasp calibration profile and embodiment selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.runtime.calibration_registry import (
    DEFAULT_GRASP_CALIBRATION_PROFILE,
    ROBOTIQ_2F85_CALIBRATION_PROFILE,
    resolve_grasp_calibration_profile,
)
from agent.tools.grasp_geometry import (
    GraspGeometryError,
    _load_profile,
    _validate_profile,
    compile_grasp_seed,
    load_grasp_strategies,
    DEFAULT_GRASP_STRATEGY_ROOT,
)
from agent.tools.grasp_strategies import (
    select_grasp_strategy,
    strategy_grasp_width_bounds,
)

_FIXED_CAM_EXTRINSICS = {
    "pos": [0.4512457429, -1.0011748029, 0.7842067477],
    "mat": [
        0.5887602350, 0.6916454180, -0.4183156720,
        0.7933781865, -0.3954737914, 0.4627650953,
        0.1546364730, -0.6043402156, -0.7815756298,
    ],
    "matrix_layout": "row_major",
}


def _candidate(width: float) -> dict:
    return {
        "id": "grasp_000",
        "frame": "camera",
        "camera_frame": "opencv",
        "width": width,
        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "translation_xyz": [0.02, -0.01, 0.42],
    }


def test_robotiq_profile_loads_and_validates() -> None:
    profile, _ = _load_profile(Path(ROBOTIQ_2F85_CALIBRATION_PROFILE))
    _validate_profile(profile, target_class="")
    assert profile["calibration_id"] == "graspnet-eef-robotiq-2f85"
    assert profile["robot_model"] == "UR5e"
    assert profile["gripper_model"] == "Robotiq2F85"
    assert profile["max_gripper_width_m"] == 0.085
    # The estimated calibration must stay flagged until a physical canary runs.
    assert profile["status"] == "candidate"
    assert profile["provenance"]["unverified"] is True


def test_unsupported_embodiment_is_rejected() -> None:
    profile, _ = _load_profile(Path(ROBOTIQ_2F85_CALIBRATION_PROFILE))
    profile = dict(profile)
    profile["gripper_model"] = "SomeUnknownGripper"
    with pytest.raises(GraspGeometryError, match="not a supported embodiment"):
        _validate_profile(profile, target_class="")


def test_registry_selects_by_embodiment_fingerprint() -> None:
    assert (
        resolve_grasp_calibration_profile(environment_id="openeta-ur5e")
        == ROBOTIQ_2F85_CALIBRATION_PROFILE
    )
    assert (
        resolve_grasp_calibration_profile(environment_id="openeta-sim")
        == DEFAULT_GRASP_CALIBRATION_PROFILE
    )
    assert (
        resolve_grasp_calibration_profile(
            fingerprint={"robot_model": "UR5e", "gripper_model": "Robotiq2F85"}
        )
        == ROBOTIQ_2F85_CALIBRATION_PROFILE
    )


def test_width_gate_uses_robotiq_stroke() -> None:
    strategies = load_grasp_strategies(Path(DEFAULT_GRASP_STRATEGY_ROOT))
    robotiq, sha = _load_profile(Path(ROBOTIQ_2F85_CALIBRATION_PROFILE))
    params = {
        "camera_pose": _candidate(0.083),
        "camera_extrinsics": _FIXED_CAM_EXTRINSICS,
        "scene_epoch": 0,
    }
    # 0.083 m exceeds the Panda 0.08 gate but is within the Robotiq 0.085 stroke.
    out = compile_grasp_seed(
        params, profile=robotiq, profile_sha256=sha, strategies=strategies
    )
    assert out["calibration_id"] == "graspnet-eef-robotiq-2f85"

    panda, panda_sha = _load_profile(Path(DEFAULT_GRASP_CALIBRATION_PROFILE))
    with pytest.raises(GraspGeometryError, match=r"width must be in \[0.0, 0.08\]"):
        compile_grasp_seed(
            params, profile=panda, profile_sha256=panda_sha, strategies=strategies
        )


@pytest.mark.parametrize(
    ("geometry_family", "strategy_id"),
    [
        ("upright_can", "top-down-vertical-robotiq-2f85"),
        ("upright_bottle", "top-down-vertical-robotiq-2f85"),
        ("boxed_item", "top-down-vertical-robotiq-2f85"),
        ("bowl", "top-down-bowl-robotiq-2f85"),
        ("drawer_handle", "top-down-drawer-handle-robotiq-2f85"),
    ],
)
def test_robotiq_auto_strategy_uses_full_2f85_width(
    geometry_family: str,
    strategy_id: str,
) -> None:
    strategies = load_grasp_strategies(Path(DEFAULT_GRASP_STRATEGY_ROOT))

    strategy, reason = select_grasp_strategy(
        strategies,
        calibration_id="graspnet-eef-robotiq-2f85",
        target_geometry_family=geometry_family,
    )

    assert reason == "automatic_geometry_family"
    assert strategy is not None
    assert strategy["strategy_id"] == strategy_id
    assert strategy_grasp_width_bounds(strategy) == pytest.approx((0.005, 0.085))
