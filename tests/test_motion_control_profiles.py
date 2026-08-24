import numpy as np
import pytest

from sim.controllers.mink_goal import summarize_cartesian_progress_window
from adapter.motion_profiles import (
    MOTION_EXPERIMENT_CONDITION_ENV,
    motion_control_profile,
)


def test_default_motion_profile_preserves_a_baseline(monkeypatch):
    monkeypatch.delenv(MOTION_EXPERIMENT_CONDITION_ENV, raising=False)

    profile = motion_control_profile()

    assert profile.condition == "A"
    assert profile.stable_arrival_enabled is False
    assert profile.progress_stall_enabled is False
    assert profile.sequential_route_preview_enabled is False
    assert profile.nominal_joint_velocity_limit_rad_s == pytest.approx(0.5)
    assert profile.carrying_joint_velocity_limit_rad_s == pytest.approx(0.2)


@pytest.mark.parametrize("condition", ["B", "C"])
def test_bounded_profiles_enable_settling_and_progress_guard(condition):
    profile = motion_control_profile(condition)

    assert profile.stable_arrival_enabled is True
    assert profile.stable_steps_required == 3
    assert profile.progress_stall_enabled is True
    assert profile.progress_window_steps == 30
    assert profile.nominal_joint_velocity_limit_rad_s < 0.5
    assert profile.carrying_joint_velocity_limit_rad_s < 0.2
    assert profile.sequential_route_preview_enabled is (condition == "C")


def test_unknown_motion_profile_fails_closed():
    with pytest.raises(ValueError, match="must be A, B, or C"):
        motion_control_profile("experimental-unknown")


def test_progress_window_separates_aligned_progress_and_cross_track_drift():
    metrics = summarize_cartesian_progress_window(
        [
            (np.asarray([0.0, 0.0, 0.0]), 0.10),
            (np.asarray([0.02, 0.01, 0.0]), 0.081),
        ],
        target_direction_xyz=np.asarray([1.0, 0.0, 0.0]),
    )

    assert metrics["command_aligned_progress_m"] == pytest.approx(0.02)
    assert metrics["cross_track_drift_m"] == pytest.approx(0.01)
    assert metrics["position_error_improvement_m"] == pytest.approx(0.019)
