"""Unit tests for the deterministic grasp geometry scorer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tools.embodied_gateway import grasp_geometry_score


def _rx(approach_z: float) -> np.ndarray:
    """Rotation about X whose approach axis (column 2) has the given world z."""
    theta = math.acos(max(-1.0, min(1.0, approach_z)))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(theta), -math.sin(theta)],
            [0.0, math.sin(theta), math.cos(theta)],
        ]
    )


def _target_for_offsets(
    rotation: np.ndarray, translation: np.ndarray, offsets_m: np.ndarray
) -> np.ndarray:
    """World target that yields the given grip-frame offsets."""
    return translation + rotation @ offsets_m


def test_top_down_scores_above_inverted() -> None:
    translation = np.zeros(3)
    target = np.array([0.0, 0.0, 0.025])
    down = grasp_geometry_score(
        rotation=_rx(-1.0), translation=translation, width=0.07, target_xyz=target
    )
    up = grasp_geometry_score(
        rotation=np.eye(3), translation=translation, width=0.07, target_xyz=target
    )
    assert down["approach_z"] == pytest.approx(-1.0)
    assert up["approach_z"] == pytest.approx(1.0)
    assert down["geometry_score"] > up["geometry_score"]


def test_centered_target_beats_lateral_offset() -> None:
    rotation = _rx(-0.93)
    translation = np.zeros(3)
    centered = grasp_geometry_score(
        rotation=rotation,
        translation=translation,
        width=0.08,
        target_xyz=_target_for_offsets(
            rotation, translation, np.array([0.0, 0.0, 0.025])
        ),
    )
    offset = grasp_geometry_score(
        rotation=rotation,
        translation=translation,
        width=0.08,
        target_xyz=_target_for_offsets(
            rotation, translation, np.array([0.0, 0.03, 0.025])
        ),
    )
    assert centered["lateral_offset_m"] == pytest.approx(0.0, abs=1e-9)
    assert offset["lateral_offset_m"] == pytest.approx(0.03, abs=1e-9)
    assert centered["geometry_score"] > offset["geometry_score"]


def test_regression_ranking_matches_live_grasp010_over_grasp000() -> None:
    """Lock in the live seed-0 case: grasp_010 must outrank grasp_000."""
    translation = np.zeros(3)

    r010 = _rx(-0.932)
    s010 = grasp_geometry_score(
        rotation=r010,
        translation=translation,
        width=0.09,
        target_xyz=_target_for_offsets(
            r010, translation, np.array([-0.0048, 0.0206, 0.0254])
        ),
    )

    r000 = _rx(-0.813)
    s000 = grasp_geometry_score(
        rotation=r000,
        translation=translation,
        width=0.061,
        target_xyz=_target_for_offsets(
            r000, translation, np.array([0.0012, 0.0247, 0.0325])
        ),
    )

    assert s010["geometry_score"] == pytest.approx(2.292, abs=1e-3)
    assert s010["geometry_score"] > s000["geometry_score"]


def test_without_target_still_prefers_top_down() -> None:
    down = grasp_geometry_score(
        rotation=_rx(-0.9), translation=np.zeros(3), width=0.07
    )
    side = grasp_geometry_score(
        rotation=_rx(-0.3), translation=np.zeros(3), width=0.07
    )
    assert down["geometry_score"] > side["geometry_score"]


def test_custom_approach_axis_swaps_preference() -> None:
    """A drawer-style horizontal approach should outrank top-down when asked."""
    # Approach axis along world +X (e.g. pulling a drawer handle towards the robot).
    drawer_rotation = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    top_down_rotation = _rx(-1.0)
    translation = np.zeros(3)

    drawer_first = grasp_geometry_score(
        rotation=drawer_rotation,
        translation=translation,
        width=0.05,
        approach_axis_world=(1.0, 0.0, 0.0),
    )
    top_down_first = grasp_geometry_score(
        rotation=top_down_rotation,
        translation=translation,
        width=0.05,
        approach_axis_world=(1.0, 0.0, 0.0),
    )
    assert drawer_first["approach_alignment"] == pytest.approx(1.0)
    assert drawer_first["geometry_score"] > top_down_first["geometry_score"]

    # With the default tabletop axis the ranking flips back.
    default_drawer = grasp_geometry_score(
        rotation=drawer_rotation, translation=translation, width=0.05
    )
    default_top_down = grasp_geometry_score(
        rotation=top_down_rotation, translation=translation, width=0.05
    )
    assert default_top_down["geometry_score"] > default_drawer["geometry_score"]
