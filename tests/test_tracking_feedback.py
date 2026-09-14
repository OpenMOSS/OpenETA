import numpy as np
from scipy.spatial.transform import Rotation
from sim.controllers.cartesian_segment import CartesianSegment, checked_segment_velocity

I = [0, 0, 0, 1]


def test_position_and_rotation_corridors_are_distinguished():
    line = CartesianSegment([0, 0, 0], [0, 0, .1], I, I)
    reasons = line.candidate_rejections(np.array([0, 0, .02]), I, [.009, 0, .02], I)
    assert set(reasons) == {'position_step_limit', 'position_corridor'}
    turn = CartesianSegment([0, 0, 0], [0, 0, 0], I,
                            Rotation.from_euler('z', 90, degrees=True).as_quat())
    wrong_axis = Rotation.from_euler('x', 5, degrees=True).as_quat()
    assert set(turn.candidate_rejections(np.zeros(3), I, np.zeros(3), wrong_axis)) == {
        'rotation_step_limit', 'rotation_corridor'}


def test_checked_segment_retains_bounded_rejection_evidence():
    class Configuration:
        q = np.array([.009, 0, .04])
        def integrate(self, velocity, dt):
            return self.q + velocity * dt
    path = CartesianSegment([0, 0, 0], [0, 0, .1], I, I)
    reasons = set()
    result = checked_segment_velocity(Configuration(), np.array([.01, 0, .01]),
        np.array([0, 1, 2]), .05, path, lambda q: (q, I), lambda *args: True, reasons)
    assert result is None
    assert reasons == {'position_corridor'}
