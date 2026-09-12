import json

import numpy as np
import pytest

from sim.controllers.gripper_contact import contact_summary


def test_contact_telemetry_distinguishes_pads_body_and_robot_self_contact():
    mj = pytest.importorskip('mujoco')
    model = mj.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="left"><joint type="slide" axis="1 0 0"/>
        <geom name="lp" type="sphere" size=".01" pos="-.019 0 0"/>
        <geom name="lb" type="sphere" size=".01" pos="-.019 0 .05"/>
      </body>
      <body name="right"><joint type="slide" axis="1 0 0"/>
        <geom name="rp" type="sphere" size=".01" pos=".019 0 0"/>
        <geom name="rb" type="sphere" size=".01" pos=".019 0 .05"/>
      </body>
      <body name="object"><joint type="slide" axis="0 0 1"/>
        <geom name="target" type="sphere" size=".01"/>
      </body>
      </worldbody></mujoco>''')
    ids = {n: model.geom(n).id for n in ['lp', 'lb', 'rp', 'rb', 'target']}
    groups = {'left_finger': [ids['lp'], ids['lb']], 'right_finger': [ids['rp'], ids['rb']],
              'left_fingerpad': [ids['lp']], 'right_fingerpad': [ids['rp']]}
    robot = [ids[n] for n in ['lp', 'lb', 'rp', 'rb']]
    for q, expected in [([0, 0, 0], 'bilateral_pads'), ([0, .1, 0], 'single_pad'),
                        ([0, 0, .05], 'finger_body_only'), ([0, 0, .2], 'no_contact'),
                        ([.019, 0, .2], 'no_contact')]:
        state = np.array(q)
        result = contact_summary(model, state, groups, robot)
        assert result['contact_pattern'] == expected
        assert result['retention_proven'] is False
        np.testing.assert_array_equal(state, q)
        assert 'target' not in json.dumps(result)


def test_contact_feedback_survives_adapter_and_rejects_private_extra_fields():
    from agent.runtime.response_artifacts import build_motion_summary
    from agent.tools.sim_mcp import _agent_visible_simulator_response
    from tools.codex_feedback import motion_feedback
    raw = {'stop_reason': 'gripper_horizon_completed', 'steps_executed': 60,
           'gripper_contact': {'available': True, 'contact_pattern': 'finger_body_only',
               'left_fingerpad_contact': False, 'right_fingerpad_contact': False,
               'private_extra': '/secret/fixture42', 'retention_proven': True}}
    summary = build_motion_summary(_agent_visible_simulator_response(raw))
    assert 'secret' not in json.dumps(summary)
    command = {'tool_calls': [{'name': 'gripper_control', 'result': {'details': {
        'outputs': {'motion_summary': summary}}}}]}
    result = motion_feedback(command, 'gripper_control')
    assert result['motion_summary']['gripper_contact']['contact_pattern'] == 'finger_body_only'
    assert result['motion_summary']['gripper_contact']['retention_proven'] is False
    assert 'Both pads are not in contact' in result['recovery']['message']
    assert 'secret' not in json.dumps(result)


def test_failed_approach_feedback_calls_for_correction_before_closing():
    from tools.codex_feedback import motion_feedback
    command = {'tool_calls': [{'name': 'move_to', 'result': {'details': {'outputs': {
        'motion_summary': {'stop_reason': 'iteration_limit', 'steps_executed': 150,
                           'reached_target': False, 'position_error_m': .019}}}}}]}
    result = motion_feedback(command)
    assert 'do not close at the assumed target' in result['recovery']['message']
