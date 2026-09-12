"""Preserve nearly unchanged Cartesian orientation without assuming a grasp."""

import math


def orientation_hold_multiplier(*, seeded, orientation_change_rad):
    """Weight a validated pose's hold, keeping deliberate rotations at baseline.

    This only changes a QP task cost. It does not imply contact, alter velocity
    limits or arrival criteria, relax collisions, or change the target pose.
    The hold band and multiplier match the existing held-translation policy.
    """
    if (seeded and orientation_change_rad is not None
            and math.isfinite(orientation_change_rad)
            and 0 <= orientation_change_rad <= 0.05):
        return 25.0
    return 1.0
