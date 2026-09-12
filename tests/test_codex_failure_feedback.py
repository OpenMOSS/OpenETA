import json

import pytest

from test_codex_motion_hook import rig, POSE, response
from test_codex_atomic import atomic, body, mark
from agent.tools.registry import ToolResult
from tools.codex_feedback import motion_feedback


@pytest.mark.parametrize('code,action', [
    ('cartesian_segment_blocked', 'replan_cartesian_segment'),
    ('cartesian_path_deviation', 'inspect_path_deviation'),
    ('cartesian_progress_stalled', 'replan_stalled_segment'),
])
def test_path_feedback_survives_real_response_compaction_without_private_geometry(code, action):
    from agent.tools.sim_mcp import _agent_visible_simulator_response
    from agent.runtime.response_artifacts import build_motion_summary
    raw = {'stop_reason': 'control_step_failed', 'steps_executed': 7,
           'controller_failure': {'code': code, 'private_obstacle': 'secret_caddy_42'},
           'controller_receipt': {'cartesian_tracking_enabled': True,
               'path_peak_cross_track_m': .004, 'path_peak_rotation_deviation_rad': .02,
               'path_backtracked_steps': 3}}
    public = _agent_visible_simulator_response(raw)
    command = {'tool_calls': [{'name': 'move_to', 'result': {'details': {'outputs': {
        'motion_summary': build_motion_summary(public)}}}}]}
    feedback = motion_feedback(command)
    assert feedback['motion_summary']['controller_failure']['code'] == code
    assert feedback['motion_summary']['path_tracking']['path_peak_cross_track_m'] == .004
    assert feedback['physics_executed'] is True
    assert feedback['recovery']['action'] == action
    assert 'secret_caddy' not in json.dumps(feedback)


def failed_handler(reason, steps=0):
    def handler(ctx):
        return ToolResult(False, 'private fixture_42 /secret/response.json', {'outputs': {
            'response': {'error': 'private fixture_42', 'response_path': '/secret/response.json'},
            'motion_summary': {'steps_executed': steps, 'stop_reason': reason, 'reached_target': False,
                'position_error_m': .03,
                'collision': {'detected': True, 'collision_class': 'attached_object_world',
                              'obstacle': 'fixture_42', 'predicted_attached_center_xyz': [4,5,6]},
                'controller_failure': {'code': 'constraint_escape_preview_rejected', 'recovery': '/secret/response.json'}}}})
    return handler


@pytest.mark.parametrize('reason,steps', [('contact_authorization_unresolved',0),('collision_detected',0),('control_step_failed',4)])
def test_failed_motion_through_real_host_keeps_verdict_and_execution_count(rig, reason, steps):
    host, _ = rig
    host.runtime.tools.bind_handler('move_to', failed_handler(reason,steps), replace=True)
    r=host.call('move_to', {'target_pose': POSE})
    d=response(r);hook=d['motion_hook']
    assert r.isError and d['error']['code']==reason
    assert hook['reason_code']==reason
    assert hook['motion_summary']['steps_executed']==steps
    assert hook['physics_executed']==(steps>0)
    assert hook['motion_dispatched'] is True
    assert hook['recovery']['message']
    assert 'fixture_42' not in json.dumps(hook)
    assert '/secret' not in json.dumps(hook)


def test_atomic_zero_step_rejection_preserves_fresh_point_and_does_not_leak(atomic):
    host,_=atomic
    point=body(mark(host))['feedback']['point']['point_id']
    host.runtime.tools.bind_handler('move_to',failed_handler('contact_authorization_unresolved'),replace=True)
    generation=host.atomic.generation
    result=host.call('move_to',{'point_id':point,'contact_point_id':point})
    d=body(result)
    assert result.isError
    assert d['feedback']['motion']['physics_executed'] is False
    assert d['error']['code']=='contact_authorization_unresolved'
    assert host.atomic.generation==generation
    assert host.atomic.point(point,contact=True)
    assert 'fixture_42' not in json.dumps(d) and '/secret' not in json.dumps(d)


def test_unknown_private_reason_is_not_projected():
    command={'tool_calls':[{'name':'move_to','result':{'details':{'outputs':{
        'motion_summary':{'stop_reason':'private_fixture_42','steps_executed':0,'position_error_m':float('nan')}
    }}}}]}
    r=motion_feedback(command)
    assert r['reason_code']=='tool_execution_failed'
    assert 'private_fixture_42' not in json.dumps(r)
    assert 'position_error_m' not in r['motion_summary']


def test_collision_class_and_phase_survive_both_feedback_boundaries():
    from agent.tools.sim_mcp import _agent_visible_simulator_response
    from agent.runtime.response_artifacts import build_motion_summary
    raw={'stop_reason':'collision_detected','steps_executed':3,
         'collision':{'detected':True,'collision_type':'robot_world','robot_part':'gripper',
          'check_mode':'post_step_configuration','checked_during':'gripper_actuation',
          'prediction_checked':False,'contact_binding_active':True,
          'geom1_name':'private_finger','geom2_name':'private_cabinet','minimum_distance_m':-.009}}
    public=_agent_visible_simulator_response(raw)
    cmd={'tool_calls':[{'name':'gripper_control','result':{'details':{'outputs':{
        'motion_summary':build_motion_summary(public)}}}}]}
    feedback=motion_feedback(cmd,'gripper_control')
    col=feedback['motion_summary']['collision']
    assert col['collision_class']=='robot_world' and col['robot_part']=='gripper'
    assert col['check_mode']=='post_step_configuration' and col['prediction_checked'] is False
    assert 'private_' not in json.dumps(feedback) and 'minimum_distance' not in json.dumps(feedback)
    assert 'after a physics step' in feedback['recovery']['message']


def test_validated_ik_failure_remains_execution_failure():
    cmd={'tool_calls':[{'name':'move_to','result':{'details':{'outputs':{'motion_summary':{
        'stop_reason':'iteration_limit','steps_executed':150,'orientation_error_deg':24.,
        'controller_receipt':{'ik_seed_validated':True,'orientation_within_tolerance':False,
            'arm_joint_margin_min_rad':-.001,'nearest_limit_joint_index':5,
            'full_pose_outcome':'execution_failed_after_validated_ik'}}}}}}]}
    feedback=motion_feedback(cmd)
    assert feedback['motion_summary']['pose_diagnostics']['ik_seed_validated']
    assert feedback['motion_summary']['pose_diagnostics']['arm_joint_margin_min_rad']==-.001
    assert 'endpoint IK seed passed' in feedback['recovery']['message']


def test_failed_constraint_escape_keeps_actual_collision_cause_and_redaction():
    from agent.tools.sim_mcp import _agent_visible_simulator_response
    from agent.runtime.response_artifacts import build_motion_summary
    raw = {'stop_reason': 'control_step_failed', 'steps_executed': 3,
           'controller_failure': {'code': 'constraint_escape_preview_rejected'},
           'collision': {'detected': True, 'collision_type': 'robot_world',
                         'robot_part': 'gripper', 'gripper_part': 'base_or_palm',
                         'obstacle_relation': 'outside_contact_target',
                         'contact_binding_active': True,
                         'check_mode': 'pre_actuation_configuration',
                         'geom2_name': 'private_bowl', 'minimum_distance_m': -.0011}}
    summary = build_motion_summary(_agent_visible_simulator_response(raw))
    command = {'tool_calls': [{'name': 'move_to', 'result': {'details': {
        'outputs': {'motion_summary': summary}}}}]}
    feedback = motion_feedback(command)
    assert feedback['reason_code'] == 'control_step_failed'
    assert feedback['physics_executed'] is True
    assert feedback['recovery']['action'] == 'replan_collision'
    assert 'base/palm' in feedback['recovery']['message']
    assert 'outside that target' in feedback['recovery']['message']
    assert 'before execution' in feedback['recovery']['message']
    assert 'private_bowl' not in json.dumps(feedback)
    assert 'minimum_distance' not in json.dumps(feedback)


def test_atomic_gripper_zero_step_stop_keeps_fresh_measurement(atomic):
    host,_=atomic
    point=body(mark(host))['feedback']['point']['point_id']
    generation=host.atomic.generation
    host.runtime.tools.bind_handler('gripper_control',failed_handler('collision_detected',0),replace=True)
    result=host.call('gripper_control',{'action':'close','contact_point_id':point})
    assert result.isError and body(result)['feedback']['motion']['physics_executed'] is False
    assert host.atomic.generation==generation
    assert host.atomic.point(point,contact=True)
