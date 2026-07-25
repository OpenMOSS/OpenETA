from __future__ import annotations

from pathlib import Path

import pytest

from agent.runtime.calibration_registry import resolve_grasp_calibration_profile
from agent.tools.grasp_strategies import (
    GraspStrategyError,
    load_grasp_strategies,
    select_grasp_strategy,
    validate_grasp_strategy,
)


def test_default_strategy_matches_only_truthful_geometry_family() -> None:
    strategies = load_grasp_strategies()

    strategy, selection = select_grasp_strategy(
        strategies,
        calibration_id="graspnet-eef-panda-p8",
        target_geometry_family="upright_can",
    )
    assert strategy is not None
    assert strategy["strategy_id"] == "top-down-vertical-panda-p8"
    assert selection == "automatic_geometry_family"

    bowl, bowl_selection = select_grasp_strategy(
        strategies,
        calibration_id="graspnet-eef-panda-p8",
        target_geometry_family="bowl",
    )
    assert bowl is not None
    assert bowl["strategy_id"] == "top-down-bowl-panda-p8"
    assert bowl_selection == "automatic_geometry_family"

    generic, generic_selection = select_grasp_strategy(
        strategies,
        calibration_id="graspnet-eef-panda-p8",
        target_geometry_family="apple",
    )
    assert generic is None
    assert generic_selection == "generic_fallback"

    handle, handle_selection = select_grasp_strategy(
        strategies,
        calibration_id="graspnet-eef-panda-p8",
        target_geometry_family="articulated_handle",
    )
    assert handle is not None
    assert handle["strategy_id"] == "top-down-drawer-handle-panda-p8"
    assert handle_selection == "automatic_geometry_family"


def test_explicit_incompatible_strategy_fails_closed() -> None:
    strategies = load_grasp_strategies()

    with pytest.raises(GraspStrategyError, match="unknown or incompatible"):
        select_grasp_strategy(
            strategies,
            calibration_id="other-calibration",
            strategy_id="top-down-vertical-panda-p8",
        )


def test_strategy_validator_rejects_physically_invalid_width_bounds() -> None:
    with pytest.raises(GraspStrategyError, match="width bounds"):
        validate_grasp_strategy(
            {
                "schema_version": "openeta.grasp_strategy.v1",
                "status": "candidate",
                "strategy_id": "bad",
                "compatibility": {"calibration_ids": ["calibration"]},
                "automatic_activation": {"target_geometry_families": []},
                "constraints": {"grasp_width_bounds_m": [0.09, 0.21]},
                "pose_policy": {
                    "orientation": "preserve_candidate",
                    "approach_axis": "preserve_candidate",
                },
            }
        )


def test_strategy_validator_accepts_bowl_geometry_policies() -> None:
    strategy = validate_grasp_strategy(
        {
            "schema_version": "openeta.grasp_strategy.v1",
            "status": "candidate",
            "strategy_id": "bowl",
            "compatibility": {"calibration_ids": ["calibration"]},
            "automatic_activation": {"target_geometry_families": ["bowl"]},
            "constraints": {"grasp_width_bounds_m": [0.01, 0.08]},
            "candidate_filter": {"min_downward_alignment": 0.5},
            "alignment_policy": {"target_region": "nearest_shallow_surface"},
            "motion_policy": {"precontact_distance_m": 0.05},
            "pose_policy": {
                "orientation": "top_down_preserve_yaw",
                "approach_axis": "world_-Z",
            },
        }
    )

    assert strategy["alignment_policy"]["target_region"] == "nearest_shallow_surface"


def test_calibration_registry_matches_libero_panda_and_rejects_unknown_robot() -> None:
    selected = resolve_grasp_calibration_profile(
        environment_id="libero_10",
        fingerprint={
            "robot_model": "Panda",
            "gripper_model": "PandaGripper",
            "grasp_frame": "graspnet",
        },
    )
    assert isinstance(selected, Path)
    assert selected.name == "graspnet-eef-panda-p8.json"
    assert (
        resolve_grasp_calibration_profile(environment_id="openeta/test-v0")
        == selected
    )

    assert (
        resolve_grasp_calibration_profile(
            environment_id="libero_10",
            fingerprint={"robot_model": "UR5"},
        )
        is None
    )
