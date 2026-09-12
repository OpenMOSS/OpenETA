"""Endpoint rejection must not be mistaken for an existing collision."""
import json

import pytest

from agent.runtime.response_artifacts import build_motion_summary
from agent.tools import sim_mcp
from sim.mcp_server.collision import check_attached_object_collision
from tools.codex_feedback import motion_feedback


def scene(width=.06):
    return ({'object_name':'private_held', 'relative_xyz':[0,0,-.07],
             'dims':[width,.06,.14], 'status':'tentative'},
            {'name':'private_container','category':'basket',
             'aabb_min':[-.12,.15,-.01], 'aabb_max':[.12,.36,.16]})


def feedback(raw):
    public=sim_mcp._agent_visible_simulator_response(raw)
    summary=build_motion_summary(public)
    command={'tool_calls':[{'name':'move_to','result':{'details':{'outputs':{
        'motion_summary':summary}}}}]}
    return public,motion_feedback(command)


@pytest.mark.parametrize('width,constraint,action', [
    (.06,'outside_receptacle_corridor','realign_carried_object_before_descent'),
    (.25,'receptacle_corridor_too_narrow','reconsider_carried_object_orientation'),
])
def test_real_checker_constraint_survives_projection_without_geometry(width,constraint,action):
    attachment,basket=scene(width)
    hit,info=check_attached_object_collision(attachment,[basket],[0,.15,.24])
    assert hit and info['placement_constraint']==constraint
    raw={'stop_reason':'collision_detected','steps_executed':0,
         'collision':{'detected':True,'check_stage':'target_endpoint',**info}}
    public,atomic=feedback(raw)
    assert atomic['motion_summary']['collision']['check_stage']=='target_endpoint'
    assert atomic['motion_summary']['collision']['placement_constraint']==constraint
    assert atomic['physics_executed'] is False
    assert atomic['recovery']['action']==action
    assert 'does not establish a collision at the current pose' in atomic['recovery']['message']
    recovery=sim_mcp._motion_target_miss_recovery_options(public)
    assert [x['action'] for x in recovery]==['replan_rejected_endpoint']
    assert recovery[0]['parameters']['enable_collision_check'] is True
    combined=json.dumps([public,atomic,recovery])
    for secret in ['private_held','private_container','required_center_delta','predicted_attached_center',
                   'overlap_volume','"receptacle_corridor":','world-frame XY delta']:
        assert secret not in combined
    if width==.06:
        assert not check_attached_object_collision(attachment,[basket],[0,.255,.24])[0]
        assert 'raising alone' in atomic['recovery']['message']


def test_unknown_collision_extensions_are_not_forwarded():
    public,atomic=feedback({'stop_reason':'collision_detected','steps_executed':0,
        'collision':{'detected':True,'collision_type':'attached_object_world',
                     'check_stage':'private_endpoint_name','placement_constraint':'private_shape_name'}})
    assert 'private_' not in json.dumps([public,atomic])
    assert 'placement_constraint' not in atomic['motion_summary']['collision']
    assert 'check_stage' not in atomic['motion_summary']['collision']


def test_zero_steps_without_stage_do_not_assert_existing_collision():
    recovery=sim_mcp._motion_target_miss_recovery_options({'stop_reason':'collision_detected',
        'steps_executed':0,'collision':{'detected':True,'collision_class':'robot_world'}})
    assert 'current configuration is already' not in json.dumps(recovery)
    assert 'Step count alone does not establish' in recovery[0]['reason']


def test_server_rejects_only_endpoint_and_does_not_dispatch_worker(monkeypatch):
    from sim.mcp_server import server as s
    attachment,basket=scene()
    meta={'backend':'libero','_attachment_proxy':attachment,'_collision_objects':[basket]}
    monkeypatch.setattr(s,'_session_envs',{'test':{'h':meta}})
    monkeypatch.setattr(s,'_touch_session',lambda *a:None)
    monkeypatch.setattr(s,'require_controller_capability',lambda *a,**k:{'goal_executor':'openeta.worker_mink_goal.v1'})
    monkeypatch.setattr(s,'cartesian_scales',lambda *a,**k:(.05,.5))
    monkeypatch.setattr(s,'cartesian_command_frame',lambda *a,**k:'world')
    def forbidden(*a,**k):raise AssertionError('unsafe endpoint dispatched to worker')
    monkeypatch.setattr(s,'_proxy_controller_goal',forbidden)
    raw=s.move_to.__wrapped__(handle='h',session_id='test',x=0,y=.15,z=.24)
    assert raw['code']=='attached_object_endpoint_collision'
    assert raw['steps_executed']==0
    public,atomic=feedback(raw)
    assert atomic['motion_summary']['collision']['check_stage']=='target_endpoint'
    assert atomic['motion_summary']['collision']['placement_constraint']=='outside_receptacle_corridor'
    assert atomic['recovery']['action']=='realign_carried_object_before_descent'
