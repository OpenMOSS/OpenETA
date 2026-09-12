"""Bounded, endpoint-only Panda parallel-jaw orientation selection.

This module creates no IK authority and performs no motion. Scores rank only
independently authorized candidates; they do not certify a controller path.
"""
import math

import numpy as np


def equivalent_rotations(rotation):
    r = np.asarray(rotation, dtype=float)
    if (r.shape != (3, 3) or not np.isfinite(r).all()
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(r), 1., atol=1e-7)):
        raise ValueError("Expected a proper rotation matrix")
    return (r.copy(), r @ np.diag([-1., -1., 1.]))


def rotation_distance(current, target):
    return float(np.arccos(np.clip((np.trace(np.asarray(current).T @ target)-1)/2, -1., 1.)))


def endpoint_score(candidate, angle_rad):
    """Use existing IK metrics; limited-joint travel is NOT angle-wrapped.

    Endpoint v1 uses RMS joint travel in radians and the WORST joint margin,
    since the receipt does not carry every joint's limits. No path claim.
    """
    values = [candidate.get('joint_travel_l2_rad'), candidate.get('joint_margin_min_rad'), angle_rad]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("Finite IK joint travel and limit margin are required")
    travel, margin, angle = map(float, values)
    if travel < 0 or margin < 0 or not 0 <= angle <= math.pi + 1e-9:
        raise ValueError("Invalid or joint-limit-violating candidate metrics")
    movement = travel / math.sqrt(7.)
    penalty = max(0., 1. - margin/.10)**2
    return {'score': movement + 2.*penalty + .2*angle/math.pi,
            'joint_travel_rms_rad': movement, 'joint_margin_min_rad': margin,
            'limit_penalty': penalty, 'rotation_deg': math.degrees(angle)}
