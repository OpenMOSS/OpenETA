import math
import numpy as np
import pytest
Rotation = pytest.importorskip('scipy.spatial.transform').Rotation

from sim.controllers.cartesian_segment import CartesianSegment, checked_segment_velocity

I = [0, 0, 0, 1]


def test_reference_stays_on_segment_instead_of_chasing_drift():
    path = CartesianSegment([0,0,0], [0,0,.1], I, I)
    xyz, quat = path.reference([.006,0,.03], I)
    np.testing.assert_allclose(xyz, [0,0,.045])
    np.testing.assert_allclose(quat, I)
    assert not path.candidate_allowed(np.array([.006,0,.03]), I, [.009,0,.035], I)


def test_rotation_uses_short_arc_and_sign_invariant_quaternions():
    start = Rotation.from_euler('z', 170, degrees=True).as_quat()
    end = Rotation.from_euler('z', -170, degrees=True).as_quat()
    path = CartesianSegment([0,0,0], [0,0,0], start, -end)
    assert abs(path.angle - math.radians(20)) < 1e-10
    _, middle = path.pose(.5)
    assert abs(Rotation.from_quat(middle).magnitude()-math.pi) < 1e-10


def test_reference_waits_for_rotation_during_combined_move():
    end = Rotation.from_euler('z', 90, degrees=True).as_quat()
    path = CartesianSegment([0,0,0], [.1,0,0], I, end)
    xyz, quat = path.reference([.05,0,0], I)
    assert xyz[0] < .01
    assert Rotation.from_quat(quat).magnitude() <= math.radians(6) + 1e-10


def test_stall_requires_absence_of_both_position_and_angular_progress():
    path = CartesianSegment([0,0,0], [0,0,.1], I, I)
    for _ in range(25):
        assert path.observe([0,0,.02], I, .08, 0) is None
    assert path.observe([0,0,.02], I, .08, 0) == 'cartesian_progress_stalled'
    rotation = CartesianSegment([0,0,0], [0,0,0], I,
        Rotation.from_euler('z', 90, degrees=True).as_quat())
    for angle in range(40):
        quat = Rotation.from_euler('z', angle, degrees=True).as_quat()
        assert rotation.observe([0,0,0], quat, 0, math.radians(90-angle)) is None


def test_actual_deviation_is_recorded_and_stopped():
    path = CartesianSegment([0,0,0], [0,0,.1], I, I)
    assert path.observe([.011,0,.05], I, .05, 0) == 'cartesian_path_deviation'
    assert path.receipt()['peak_cross_track_m'] == .011


def test_prediction_headroom_allows_recovery_but_not_further_departure():
    path = CartesianSegment([0,0,0], [0,0,.1], I, I)
    now = np.array([.009,0,.04])
    assert path.candidate_allowed(now, I, [.0085,0,.041], I)
    assert not path.candidate_allowed(now, I, [.0095,0,.041], I)


def test_backtracking_checks_geometry_and_never_commands_nonarm_dofs():
    class Configuration:
        q = np.zeros(4)
        def integrate(self, velocity, dt):
            assert velocity[3] == 0
            return self.q + velocity*dt
    path = CartesianSegment([0,0,0], [0,0,.1], I, I)
    checked=[]
    def geometry(q, xyz, quat):
        checked.append(float(q[2]))
        return q[2] <= .003
    v = checked_segment_velocity(Configuration(), np.array([0,0,.16,10.]), np.array([0,1,2]),
        .05, path, lambda q: (q[:3], I), geometry)
    np.testing.assert_allclose(v, [0,0,.04,0])
    assert checked == [.008,.004,.002]
    assert path.backtracked_steps == 1
    assert checked_segment_velocity(Configuration(), v, np.array([0,1,2]), .05,
        path, lambda q:(q[:3],I), lambda *args:False) is None
