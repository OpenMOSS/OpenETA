import json
from sim.controllers.collision_feedback import rejected_candidate_obstacles, public_candidate_obstacles


def test_rejected_candidate_classes_survive_all_public_boundaries_without_actual_collision():
    from agent.tools.sim_mcp import _agent_visible_simulator_response
    from agent.runtime.response_artifacts import build_motion_summary
    from tools.codex_feedback import motion_feedback
    raw = {'stop_reason': 'control_step_failed', 'steps_executed': 4,
           'collision': {'detected': False},
           'controller_receipt': {'ik_seed_validated': True},
           'controller_failure': {'code': 'cartesian_segment_blocked',
              'constraints': ['robot_collision'],
              'tracking_constraints': ['rotation_corridor','secret_tracking',{'x':1}],
              'candidate_obstacles': [
                  {'robot_part':'gripper','gripper_part':'base_or_palm',
                   'obstacle_relation':'outside_contact_target',
                   'constraint_boundary':'clearance_recovery',
                   'geom1_name':'secret_palm', 'geom2_id':400,'position':[1,2,3]},
                  {'robot_part':'secret_part', 'obstacle_relation': {'secret':'nested'}}]}}
    summary = build_motion_summary(_agent_visible_simulator_response(raw))
    feedback = motion_feedback({'tool_calls':[{'name':'move_to', 'result':{'details':{'outputs':{'motion_summary':summary}}}}]})
    rows=feedback['motion_summary']['controller_failure']['candidate_obstacles']
    assert rows == [{'robot_part':'gripper','gripper_part':'base_or_palm',
                     'obstacle_relation':'outside_contact_target','constraint_boundary':'clearance_recovery'}]
    assert feedback['motion_summary']['collision']['detected'] is False
    assert feedback['physics_executed'] is True
    text=feedback['recovery']['message']
    assert 'base/palm' in text and 'outside the authorized' in text
    assert feedback['motion_summary']['controller_failure']['tracking_constraints'] == ['rotation_corridor']
    assert 'rotation corridor' in text
    assert 'IK seed passed validation' in text and 'do not close' in text
    assert 'secret' not in json.dumps(feedback) and 'position' not in json.dumps(rows)


def test_recovery_explains_only_rejecting_pairs_not_unrelated_nearby_geometry():
    policy={'hard_stop_distance_m':-.001, 'minimum_distance_from_collisions_m':.005,
            'robot_geom_ids':{1,2,3},'gripper_geom_ids':{2,3}, 'finger_geom_ids':{3},
            'authorized_target_geom_ids':{10}, 'authorized_target_geom_count':1}
    current={(1,11):.001,(3,10):.002}
    candidate={(1,11):0.,(3,10):.003}
    assert rejected_candidate_obstacles(current,candidate,policy,recovering=True) == [
        {'robot_part':'arm','obstacle_relation':'outside_contact_target','constraint_boundary':'clearance_recovery'}]
    assert rejected_candidate_obstacles(current,candidate,policy,recovering=False) == []
    candidate[(3,10)]=-.002
    rows=rejected_candidate_obstacles(current,candidate,policy,recovering=False)
    assert rows[0]['gripper_part']=='finger' and rows[0]['obstacle_relation']=='authorized_target'


def test_candidate_projection_is_bounded_and_rejects_nested_or_unrecognized_data():
    assert public_candidate_obstacles({'robot_part':'arm'}) == []
    assert public_candidate_obstacles([{'robot_part':{'x':'arm'}},'secret',{'robot_part':'private_geom'}]) == []
    assert public_candidate_obstacles([{'robot_part':'arm','private_xyz':[1,2,3]}]*100) == [{'robot_part':'arm'}]
