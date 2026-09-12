import math

import pytest

from adapter.motion_profiles import motion_control_profile
from sim.controllers.fixture_grip import stabilize_grip
from sim.controllers.orientation_tracking import orientation_hold_multiplier


@pytest.mark.parametrize('angle', [0., .0434, .047, .05])
def test_nearly_unchanged_validated_pose_holds_orientation_without_claiming_grip(angle):
    base = motion_control_profile('A')
    profile, grip_weight, kind = stabilize_grip(
        base, enabled=True, authorization=None, gripper_command=-1,
        contact={}, orientation_change_rad=angle, seeded=True)
    assert orientation_hold_multiplier(seeded=True, orientation_change_rad=angle) == 25.
    assert profile is base and grip_weight == 1. and kind == 'none'


@pytest.mark.parametrize('angle', [.050001, math.pi / 2, math.pi, None, -1., math.nan, math.inf])
def test_deliberate_rotation_or_missing_constraint_keeps_baseline_weight(angle):
    assert orientation_hold_multiplier(seeded=True, orientation_change_rad=angle) == 1.


def test_unseeded_controller_keeps_baseline():
    assert orientation_hold_multiplier(seeded=False, orientation_change_rad=0.) == 1.
