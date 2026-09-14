import json
import pytest
from test_codex_motion_hook import rig
from test_codex_atomic import atomic, body


@pytest.mark.parametrize('extra', [
    {'xyz_m':[0,0,1.151]}, {'delta_m':[.01,0,0], 'contact_point_id':'p'},
    {'delta_m':[.01,0,0], 'orientation_mode':'parallel_jaw_symmetric'},
    {'xyz_m':[0,0,1.05], 'waypoints':[{'xyz_m':[0,0,1.02]}]},
])
def test_recovery_rejects_out_of_scope_before_dispatch(atomic, extra):
    host,state=atomic
    result=host.call('move_to', {'motion_mode':'recovery', **extra})
    assert result.isError and state['calls']==[]
    assert host.atomic.pending_recovery_pose is None


def test_recovery_bound_to_current_pose_then_cleared(atomic):
    from agent.tools.registry import ToolResult
    host,state=atomic;seen=[]
    def handler(ctx):
        seen.append(host.atomic.motion_mode(ctx.parameters['target_pose']))
        wrong={**ctx.parameters['target_pose'], 'xyz':[1,2,3]}
        if seen[-1]=='recovery':
            with pytest.raises(ValueError, match='different requested pose'):host.atomic.motion_mode(wrong)
        return ToolResult(True, 'result', {'outputs':{'motion_summary':{
            'reached_target':True, 'stop_reason':'target_reached','steps_executed':2}}})
    host.runtime.tools.bind_handler('move_to',handler,replace=True)
    result=host.call('move_to',{'delta_m':[.01,0,0],'motion_mode':'recovery'})
    assert not result.isError
    assert seen==['recovery'] and host.atomic.pending_recovery_pose is None
    wire=body(result)
    assert any(x['reason_code']=='bounded_recovery_selected' for x in wire['interventions'])
    assert wire['recent_motion_progress'][-1]['reason_code']=='target_reached'
    assert wire['episode']['execution_accounting']=={'preflight_calls':2,'physical_dispatches':1,
        'physical_calls_with_steps':1,'physical_unknown_outcomes':0}
    assert wire['episode']['remaining_budget']['estimated_strict_moves']==9


def test_history_unavailable_measurement_does_not_mask_known_motion_result(atomic):
    host,_=atomic
    wire=body(host.call('move_to',{'delta_m':[.01,0,0]}))
    assert wire['feedback']['motion']['reason_code']=='target_reached'
    row=wire['recent_motion_progress'][-1]
    assert row['actual_state_available'] is False
    assert 'measured_translation_m' not in row


def test_candidate_trace_survives_filters_without_private_fields():
    from agent.tools.sim_mcp import _agent_visible_simulator_response
    from agent.runtime.response_artifacts import build_motion_summary
    from tools.codex_feedback import motion_feedback
    trace=[{'variant':'primary','scale':1,'position_step_m':.012,
            'tracking_passed':False,'tracking_rejections':['position_step_limit'],
            'geometry_status':'not_checked','geom_name':'SECRET'},
           {'variant':'track_constrained','scale':.5,'tracking_passed':True,
            'geometry_status':'rejected','geometry_rejections':['robot_collision'],
            'obstacles':[{'robot_part':'arm','obstacle_relation':'unbound_world','xyz':[1,2,3]}]}]
    raw={'stop_reason':'control_step_failed','steps_executed':0,
         'controller_failure':{'code':'cartesian_segment_blocked','candidate_trace':trace}}
    summary=build_motion_summary(_agent_visible_simulator_response(raw))
    out=motion_feedback({'tool_calls':[{'name':'move_to','result':{'details':{'outputs':{'motion_summary':summary}}}}]})
    actual=out['motion_summary']['controller_failure']['candidate_trace']
    assert actual[0]['geometry_status']=='not_checked'
    assert actual[1]['geometry_rejections']==['robot_collision']
    assert 'SECRET' not in json.dumps(out) and 'xyz' not in json.dumps(actual)


def test_private_proxy_only_forwards_host_resolved_recovery_mode():
    from agent.tools.sim_mcp import SimulatorMcpToolProxy, SimulatorMcpToolProxyConfig
    proxy=SimulatorMcpToolProxy(transport=object(), config=SimulatorMcpToolProxyConfig(handle='test-env'))
    pose={'frame':'world','xyz':[0,0,1],'quat_xyzw':[0,0,0,1]}
    parameters={'target_pose':pose,'motion_mode':'recovery'}
    assert 'motion_mode' not in proxy._move_to_arguments(parameters)
    args=proxy._move_to_arguments(parameters,metadata={'_motion_mode_resolver':lambda p:'recovery'})
    assert args['motion_mode']=='recovery' and [args[k] for k in ('x','y','z')]==pose['xyz']


def test_verified_fallback_is_reported_even_without_backtracking():
    from tools.codex_interventions import motion_interventions
    rows=motion_interventions({'reason_code':'target_reached','execution_state':'reached',
        'motion_summary':{'control_adjustments':{'verified_qp_fallback_steps':2}}})
    assert rows[0]['effect']=='adjusted' and 'fallback candidates' in rows[0]['message']
    from sim.controllers.candidate_feedback import public_candidate_trace
    assert public_candidate_trace([{'variant':{'bad':'field'}}])==[]
