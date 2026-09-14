import json
import pytest
from test_codex_motion_hook import rig
from test_codex_atomic import atomic, body
from test_codex_failure_feedback import failed_handler
from test_codex_host import host


@pytest.mark.parametrize('boundary', ['schema', 'closed', 'budget'])
def test_early_rejection_does_not_reuse_previous_motion(atomic, boundary):
    host,state=atomic
    first=body(host.call('move_to', {'delta_m':[.01,0,0]}))
    assert first['feedback']['motion']['reason_code']=='target_reached'
    before=len(state['calls'])
    if boundary=='closed':host.close()
    if boundary=='budget':host.max_requests=host.requests
    result=host.call('move_to', {'delta_m':'bad'} if boundary=='schema' else {'delta_m':[.02,0,0]})
    wire=body(result)
    assert result.isError and len(state['calls'])==before
    assert wire['feedback'] is None
    assert wire['interventions'][-1]['execution_state']=='not_started'
    assert wire['current_request']['sequence'] > first['current_request']['sequence']


def test_status_explicitly_recovers_previous_receipt_without_stale_current_feedback(atomic):
    host,state=atomic
    previous=body(host.call('move_to',{'delta_m':[.01,0,0]}))
    status=body(host.call('episode_status',{}))
    assert status['feedback'] is None and status['interventions']==[]
    assert status['previous_request']['current_request']==previous['current_request']
    assert status['previous_request']['feedback']==previous['feedback']
    assert body(host.call('episode_status',{}))['previous_request']==status['previous_request']
    assert (host.output/'native-feedback.jsonl').exists()


def test_adjusted_orientation_and_horizon_are_reported(atomic):
    host,state=atomic;host.atomic.symmetry_load_uncertain=True
    r=body(host.call('move_to',{'delta_m':[.01,0,0], 'approach_world':[0,0,-2], 'jaw_world':[2,0,1]}))
    events={x['reason_code']:x for x in r['interventions']}
    assert events['orientation_frame_normalized']['details']['resolved_directions']=={
        'approach_world':[0.,0.,-1.],'jaw_world':[1.,0.,0.]}
    assert events['loaded_reorientation_horizon']['details']['selected_steps']==300
    assert state['calls'][-1][1]['num_steps']==300


@pytest.mark.parametrize('steps,expected',[(0,'not_started'),(3,'partial')])
def test_partial_route_and_unattempted_segments_are_reported(atomic,steps,expected):
    host,state=atomic
    host.runtime.tools.bind_handler('move_to',failed_handler('collision_detected',steps),replace=True)
    r=body(host.call('move_to',{'waypoints':[{'xyz_m':[0,0,1.1]}],'xyz_m':[.1,0,1.1]}))
    route=next(x for x in r['interventions'] if x['reason_code']=='route_stopped')
    assert route['execution_state']==expected
    assert route['details']['remaining_indices']==[1]
    motion=next(x for x in r['interventions'] if x['source']=='motion_pipeline')
    assert motion['execution_state']==expected
    assert motion['details']['route_segment_index']==0


def test_stall_context_survives_three_boundaries_without_private_geometry():
    from agent.tools.sim_mcp import _agent_visible_simulator_response
    from agent.runtime.response_artifacts import build_motion_summary
    from tools.codex_feedback import motion_feedback
    row={'robot_part':'gripper','gripper_part':'base_or_palm','obstacle_relation':'outside_contact_target','geom_name':'SECRET','xyz':[1,2,3]}
    raw={'stop_reason':'local_convergence_stalled','steps_executed':26,'collision':{'detected':False},
         'controller_failure':{'code':'cartesian_progress_stalled','stall_context':{
             'qp_evidence':'available','contact_evidence':'available','active_clearance_constraints':[row],
             'measured_robot_contacts':[row],'requested_joint_speed_max_rad_s':.003,
             'commanded_joint_speed_max_rad_s':.002,'private_points':[1,2,3], 'causal_attribution':'proven'}}}
    summary=build_motion_summary(_agent_visible_simulator_response(raw))
    wire=motion_feedback({'tool_calls':[{'name':'move_to','result':{'details':{'outputs':{'motion_summary':summary}}}}]})
    ctx=wire['motion_summary']['controller_failure']['stall_context']
    assert ctx['causal_attribution']=='not_established'
    assert ctx['measured_robot_contacts'][0]['gripper_part']=='base_or_palm'
    assert wire['motion_summary']['collision']['detected'] is False
    assert wire['physics_executed'] is True
    assert 'base/palm' in wire['recovery']['message']
    assert 'not absence of contact' in wire['recovery']['message']
    assert 'SECRET' not in json.dumps(wire) and 'private_points' not in json.dumps(wire)


@pytest.mark.parametrize('gate,code,phrase', [
    ('motion_reconciliation', 'transport_outcome_unknown', 'retirement'),
    ('fresh_observation_obligation', 'fresh_observation_required', 'no fresh observation'),
])
def test_substitution_explains_known_host_obligation(host, monkeypatch, gate, code, phrase):
    from agent.runtime.planner import _invariant_obligation_decision
    original_context = host.context
    monkeypatch.setattr(host, 'context', lambda: {**original_context(), gate:{'required':True}})
    monkeypatch.setattr('agent.runtime.planner._invariant_obligation_decision',
        lambda context, **kwargs: _invariant_obligation_decision({**context, gate:{'required':True}}, **kwargs))
    wire = body(host.call('gripper_control', {'position':1}))
    event = next(row for row in wire['interventions'] if row['effect']=='substituted')
    assert event['reason_code']==code and phrase in event['message']
    assert event['execution_state']=='not_started'
    assert event['details']=={'requested_stage':'gripper_control',
                             'executed_stage':'observe' if gate=='fresh_observation_obligation' else 'internal_operation'}
    assert host.backend.pending is None


def test_atomic_repair_retains_only_callable_public_hints(atomic):
    host, _ = atomic
    host.last_command = {'metadata':{'repair_bundle':{
        'code':'invalid_source_packet', 'private_geometry':'SECRET',
        'allowed_next_calls':[{'tool':'episode_status','parameters':{}},
                              {'tool':'move_to','parameters':{'bundle_id':'SECRET'}}]}}}
    wire = body(host.result(error={'code':'invalid_source_packet'}))
    assert wire['repair']['allowed_next_calls']==[{'tool':'episode_status','parameters':{}}]
    assert 'SECRET' not in json.dumps(wire)


def test_nominal_settings_are_not_claimed_as_an_adjustment():
    from tools.codex_interventions import motion_interventions
    motion = {'reason_code':'target_reached', 'execution_state':'completed',
              'motion_summary':{'control_adjustments':{'transport_profile':'nominal', 'path_backtracked_steps':0}}}
    assert motion_interventions(motion)[0]['effect']=='configured'
    motion['motion_summary']['control_adjustments']['path_backtracked_steps']=3
    assert motion_interventions(motion)[0]['effect']=='adjusted'
