"""Small, dependency-free pose maths for UR-style axis-angle (rotvec) poses.

UR reports/consumes orientation as a rotation vector (axis * angle). The agent
speaks roll/pitch/yaw degrees. These helpers convert between rotvec, quaternion
(``[x, y, z, w]``) and rpy, and compose relative rotations — all with the stdlib
``math`` module so the real-robot path needs no scipy.
"""

from __future__ import annotations

import math

Vec3 = list[float]
Quat = list[float]  # [x, y, z, w]


def rotvec_to_quat(rotvec: Vec3) -> Quat:
    """Axis-angle rotation vector -> unit quaternion ``[x, y, z, w]``."""
    rx, ry, rz = rotvec
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    s = math.sin(angle / 2.0)
    return [rx / angle * s, ry / angle * s, rz / angle * s, math.cos(angle / 2.0)]


def quat_to_rotvec(q: Quat) -> Vec3:
    """Unit quaternion ``[x, y, z, w]`` -> axis-angle rotation vector."""
    x, y, z, w = _normalize(q)
    w = max(-1.0, min(1.0, w))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-9:  # no rotation
        return [0.0, 0.0, 0.0]
    return [x / s * angle, y / s * angle, z / s * angle]


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> Quat:
    """Roll/pitch/yaw (radians, extrinsic X-Y-Z) -> quaternion ``[x, y, z, w]``."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    # q = qz(yaw) * qy(pitch) * qx(roll)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def quat_mul(a: Quat, b: Quat) -> Quat:
    """Hamilton product ``a * b`` for ``[x, y, z, w]`` quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def quat_angle_between(a: Quat, b: Quat) -> float:
    """Smallest rotation angle (radians) between two orientations."""
    a, b = _normalize(a), _normalize(b)
    dot = abs(sum(ai * bi for ai, bi in zip(a, b)))
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


def _normalize(q: Quat) -> Quat:
    n = math.sqrt(sum(c * c for c in q))
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [c / n for c in q]
