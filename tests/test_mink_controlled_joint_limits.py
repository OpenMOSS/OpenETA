import numpy as np
import pytest

from sim.controllers.mink_goal import (
    _arm_configuration_limit,
    _fixed_nonrobot_velocity_limit,
)


def model_and_config():
    mujoco = pytest.importorskip("mujoco")
    mink = pytest.importorskip("mink")
    model = mujoco.MjModel.from_xml_string('''
      <mujoco><compiler angle="radian"/><worldbody>
        <body><joint name="arm" type="hinge" range="-2 2"/>
          <geom type="sphere" size="0.01"/></body>
        <body><joint name="finger" type="slide" range="0 0.04"/>
          <geom type="sphere" size="0.01"/></body>
      </worldbody></mujoco>''')
    return mink, model, mink.Configuration(model)


@pytest.mark.parametrize("finger_q", [-0.0016, 0.04075])
def test_uncontrolled_finger_violation_does_not_invalidate_arm_qp(finger_q):
    mink, model, configuration = model_and_config()
    configuration.update(q=np.array([0.0, finger_q]))
    zero = np.zeros(model.nv)
    full = mink.ConfigurationLimit(model).compute_qp_inequalities(configuration, 0.05)
    assert not np.all(full.G @ zero <= full.h)
    arm = _arm_configuration_limit(model, np.array([0])).compute_qp_inequalities(configuration, 0.05)
    fixed = _fixed_nonrobot_velocity_limit(model, np.array([0])).compute_qp_inequalities(configuration, 0.05)
    assert np.all(arm.G @ zero <= arm.h)
    assert np.all(fixed.G @ zero <= fixed.h)
    # Non-arm motion remains forbidden; no hypothetical finger repair is used.
    assert not np.all(fixed.G @ np.array([0.0, 0.001]) <= fixed.h)
    posture = mink.PostureTask(model, cost=[1.0, 0.0])
    posture.set_target(np.array([0.5, finger_q]))
    velocity = mink.solve_ik(configuration, [posture], 0.05, solver="quadprog",
        safety_break=False, limits=[_arm_configuration_limit(model, np.array([0])),
                                  _fixed_nonrobot_velocity_limit(model, np.array([0]))])
    assert velocity[0] > 0
    assert abs(velocity[1]) < 1e-10


@pytest.mark.parametrize("arm_q,unsafe_delta", [(1.99, 0.1), (-1.99, -0.1), (2.01, 0.0)])
def test_controlled_arm_hard_limit_is_retained(arm_q, unsafe_delta):
    mink, model, configuration = model_and_config()
    configuration.update(q=np.array([arm_q, 0.02]))
    full = mink.ConfigurationLimit(model, gain=0.95).compute_qp_inequalities(configuration, 0.05)
    arm = _arm_configuration_limit(model, np.array([0])).compute_qp_inequalities(configuration, 0.05)
    np.testing.assert_array_equal(arm.G, full.G[[0, 2]])
    np.testing.assert_array_equal(arm.h, full.h[[0, 2]])
    assert not np.all(arm.G @ np.array([unsafe_delta, 0.0]) <= arm.h)


@pytest.mark.parametrize("indices", [[0, 0], [2]])
def test_missing_or_duplicate_controlled_limit_fails_closed(indices):
    _, model, _ = model_and_config()
    with pytest.raises(RuntimeError, match="every controlled arm"):
        _arm_configuration_limit(model, np.array(indices))
