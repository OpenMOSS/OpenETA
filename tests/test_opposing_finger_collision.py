"""Only designed opposing-finger closure contacts may leave arm self coverage."""
from itertools import product

import pytest

from sim.controllers.collision_recovery import robot_self_collision_groups


def expand(groups):
    return [pair for left, right in groups for pair in product(left, right)]


def test_preserves_all_other_self_pairs_and_does_not_add_world_exemptions():
    # Arm 1/2, palm 3, left finger/pad 4/5, right finger/pad 6/7.
    robot = set(range(1, 8))
    left, right = {4, 5}, {6, 7}
    actual = expand(robot_self_collision_groups(robot, left | {99}, right))
    expected = [(a, b) for a, b in product(sorted(robot), repeat=2)
                if not ((a in left and b in right) or (a in right and b in left))]
    assert actual == expected
    assert (1, 7) in actual and (3, 5) in actual and (5, 4) in actual
    assert all(99 not in pair for pair in actual)


@pytest.mark.parametrize("left,right", [(set(), set()), ({2}, set()), ({2, 3}, {3, 4})])
def test_missing_or_ambiguous_sides_preserve_existing_self_coverage(left, right):
    robot = {1, 2, 3, 4}
    assert expand(robot_self_collision_groups(robot, left, right)) == list(product(sorted(robot), repeat=2))
