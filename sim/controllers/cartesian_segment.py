"""Bounded Cartesian reference tracking, using only requested/measured robot poses.

This is a local segment executor, not a path planner. An obstructed segment
returns to the caller instead of inventing a different route around an object.
"""
from collections import deque
import math
import numpy as np
from scipy.spatial.transform import Rotation

ENABLE_ENV = 'OPENETA_LIBERO_CARTESIAN_SEGMENT'


class CartesianSegment:
    def __init__(self, start_xyz, target_xyz, start_quat, target_quat, *, recovery=False):
        self.start = np.asarray(start_xyz, dtype=float)
        self.delta = np.asarray(target_xyz, dtype=float) - self.start
        self.length = float(np.linalg.norm(self.delta))
        self.start_rotation = Rotation.from_quat(start_quat)
        self.constrained_rotation = target_quat is not None
        end = Rotation.from_quat(target_quat) if self.constrained_rotation else self.start_rotation
        self.rotvec = (end * self.start_rotation.inv()).as_rotvec()
        self.angle = float(np.linalg.norm(self.rotvec))
        self.recovery = recovery
        self.max_cross_track_m = 0.020 if recovery else 0.010
        self.max_rotation_deviation_rad = math.radians(12 if recovery else 6)
        self.candidate_cross_track_m = 0.018 if recovery else 0.008
        self.candidate_rotation_rad = math.radians(10 if recovery else 4)
        self.samples = deque(maxlen=26)
        self.peak_cross_track_m = 0.0
        self.peak_rotation_deviation_rad = 0.0
        self.backtracked_steps = 0
        self.minimum_step_scale = 1.0

    def pose(self, fraction):
        t = float(np.clip(fraction, 0, 1))
        return self.start + t * self.delta, (
            Rotation.from_rotvec(t * self.rotvec) * self.start_rotation).as_quat()

    def fractions(self, xyz, quat):
        position = float(np.dot(np.asarray(xyz)-self.start, self.delta) / self.length**2) if self.length > 1e-9 else 1.
        actual = (Rotation.from_quat(quat) * self.start_rotation.inv()).as_rotvec()
        rotation = float(np.dot(actual, self.rotvec) / self.angle**2) if self.angle > 1e-9 else 1.
        return position, rotation

    def reference(self, xyz, quat):
        p, r = self.fractions(xyz, quat)
        active, increments = [], []
        if self.length > 0.002:
            active.append(p); increments.append(0.015 / self.length)
        if self.constrained_rotation and self.angle > 0.05:
            active.append(r); increments.append(math.radians(2) / self.angle)
        fraction = min(active, default=1.)
        lookahead = min(increments, default=1.)
        return self.pose(max(0., fraction) + lookahead)

    def errors(self, xyz, quat):
        p, r = self.fractions(xyz, quat)
        fraction = p if self.length > 0.002 else r
        expected_xyz, expected_quat = self.pose(fraction)
        distance = float(np.linalg.norm(np.asarray(xyz)-expected_xyz))
        rotation = float((Rotation.from_quat(quat) * Rotation.from_quat(expected_quat).inv()).magnitude()) if self.constrained_rotation else 0.
        return distance, rotation

    def candidate_rejections(self, current_xyz, current_quat, xyz, quat):
        """Robot-relative tracking bounds violated by a hypothetical next step."""
        reasons = []
        if np.linalg.norm(np.asarray(xyz)-current_xyz) > 0.008 + 1e-9:
            reasons.append('position_step_limit')
        if (Rotation.from_quat(quat)*Rotation.from_quat(current_quat).inv()).magnitude() > math.radians(4) + 1e-9:
            reasons.append('rotation_step_limit')
        lateral, rotation = self.errors(xyz, quat)
        current_lateral, current_rotation = self.errors(current_xyz, current_quat)
        def recovering(value, current, bound):
            return value <= bound or (current > bound and value < current - 1e-8)
        if not recovering(lateral, current_lateral, self.candidate_cross_track_m):
            reasons.append('position_corridor')
        if not recovering(rotation, current_rotation, self.candidate_rotation_rad):
            reasons.append('rotation_corridor')
        return reasons

    def candidate_allowed(self, current_xyz, current_quat, xyz, quat):
        return not self.candidate_rejections(current_xyz, current_quat, xyz, quat)

    def observe(self, xyz, quat, position_error, orientation_error):
        lateral, rotation = self.errors(xyz, quat)
        self.peak_cross_track_m = max(self.peak_cross_track_m, lateral)
        self.peak_rotation_deviation_rad = max(self.peak_rotation_deviation_rad, rotation)
        if lateral > self.max_cross_track_m or rotation > self.max_rotation_deviation_rad:
            return 'cartesian_path_deviation'
        self.samples.append((float(position_error), float(orientation_error or 0.)))
        if len(self.samples) == self.samples.maxlen:
            first_p, first_r = self.samples[0]
            # Either meaningful translation or angular progress keeps the
            # motion alive; pure rotation and slow settling must not be
            # mistaken for a stationary Cartesian translation.
            p_progress = first_p - min(x[0] for x in list(self.samples)[1:])
            r_progress = first_r - min(x[1] for x in list(self.samples)[1:])
            if p_progress < 0.0005 and r_progress < math.radians(0.5):
                return 'cartesian_progress_stalled'
        return None

    def receipt(self):
        return {'enabled': True, 'reference': 'straight_translation_shortest_rotation',
                'cross_track_limit_m': self.max_cross_track_m,
                'rotation_deviation_limit_rad': self.max_rotation_deviation_rad,
                'peak_cross_track_m': self.peak_cross_track_m,
                'peak_rotation_deviation_rad': self.peak_rotation_deviation_rad,
                'backtracked_steps': self.backtracked_steps,
                'minimum_step_scale': self.minimum_step_scale}


def checked_segment_velocity(configuration, velocity, arm_indices, dt, segment,
                             pose_fn, geometric_check, rejected_tracking=None, *,
                             candidate_trace=None, geometry_feedback=None, variant='primary'):
    """Return the largest bounded candidate; each candidate gets exact checks.

    geometric_check is supplied by the worker and retains robot, attachment
    and joint guards. This helper has no object identity or task information.
    """
    current_xyz, current_quat = pose_fn(configuration.q)
    for scale in (1., .5, .25, .125, .0625, .03125):
        candidate_velocity = np.zeros_like(velocity)
        candidate_velocity[arm_indices] = velocity[arm_indices] * scale
        q = configuration.integrate(candidate_velocity, dt)
        xyz, quat = pose_fn(q)
        tracking = segment.candidate_rejections(current_xyz, current_quat, xyz, quat)
        row = {'variant': variant, 'scale': scale,
               'position_step_m': float(np.linalg.norm(np.asarray(xyz)-current_xyz)),
               'rotation_step_rad': float((Rotation.from_quat(quat)*Rotation.from_quat(current_quat).inv()).magnitude()),
               'tracking_rejections': tracking, 'tracking_passed': not tracking,
               'geometry_status': 'not_checked', 'checks_passed': False}
        if candidate_trace is not None:
            candidate_trace.append(row)
        if tracking:
            if rejected_tracking is not None:
                rejected_tracking.update(segment.candidate_rejections(current_xyz, current_quat, xyz, quat))
            continue
        accepted = geometric_check(q, xyz, quat)
        row['geometry_status'] = 'passed' if accepted else 'rejected'
        if geometry_feedback is not None:
            row.update(geometry_feedback())
        if not accepted:
            continue
        if scale < 1.:
            segment.backtracked_steps += 1
            segment.minimum_step_scale = min(segment.minimum_step_scale, scale)
        row['checks_passed'] = True
        return candidate_velocity
    return None
