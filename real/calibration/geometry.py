"""Rigid-transform and rotation helpers for eye-to-hand calibration.

Coordinate convention used throughout the calibration package:
    ``T_A_B`` transforms points expressed in frame B into frame A.

Frames:
    base   : robot base frame (UR base_link)
    ee     : robot end-effector frame from proprio
    cam    : camera optical frame for the image/intrinsics
    board  : checkerboard frame whose origin is the first inner corner

These helpers are dependency-light (numpy only) so they can be unit-tested
without OpenCV or any hardware present.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def transform_from_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 homogeneous transform from rotation R and translation t."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform without a general matrix inverse."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def quaternion_xyzw_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert an (x, y, z, w) quaternion to a 3x3 rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Zero quaternion is not valid.")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to an (x, y, z, w) quaternion."""
    trace = np.trace(R)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def mean_rotation_matrix(rotations: Sequence[np.ndarray]) -> np.ndarray:
    """Average rotations via quaternion mean (hemisphere-aligned)."""
    quaternions = []
    for R in rotations:
        q = rotation_matrix_to_quaternion_xyzw(R)
        if quaternions and np.dot(q, quaternions[0]) < 0:
            q = -q
        quaternions.append(q)
    q_mean = np.mean(np.asarray(quaternions), axis=0)
    q_mean = q_mean / np.linalg.norm(q_mean)
    return quaternion_xyzw_to_rotation_matrix(*q_mean)


def mean_transform(transforms: Sequence[np.ndarray]) -> np.ndarray:
    """Average a list of rigid transforms (quaternion mean + centroid)."""
    if not transforms:
        raise ValueError("Cannot average an empty transform list.")
    R_mean = mean_rotation_matrix([T[:3, :3] for T in transforms])
    t_mean = np.mean([T[:3, 3] for T in transforms], axis=0)
    return transform_from_rt(R_mean, t_mean)


def rotation_error_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle between two rotations, in degrees."""
    R_delta = R_a.T @ R_b
    trace = np.clip((np.trace(R_delta) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(trace))
