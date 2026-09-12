from dataclasses import replace

import pytest

from adapter.motion_profiles import motion_control_profile
from sim.controllers.fixture_grip import stabilize_grip


def inputs():
    return dict(enabled=True, seeded=True, gripper_command=1,
        authorization={'ok': True, 'contact_kind': 'articulated_fixture'},
        contact={'available': True, 'left_fingerpad_contact': True, 'right_fingerpad_contact': True},
        orientation_change_rad=0.001)


@pytest.mark.parametrize('changes', [
    {'enabled': False}, {'seeded': False}, {'gripper_command': -1},
    {'authorization': None}, {'authorization': {'ok': False, 'contact_kind': 'articulated_fixture'}},
    {'authorization': {'ok': True, 'contact_kind': 'free_object'}},
    {'contact': {'available': False}},
    {'contact': {'available': True, 'left_fingerpad_contact': True, 'right_fingerpad_contact': False}},
    {'orientation_change_rad': None}, {'orientation_change_rad': .2},
    {'orientation_change_rad': float('nan')},
])
def test_unverified_grip_and_deliberate_rotation_keep_original_controller(changes):
    base = motion_control_profile('A')
    profile, weight, kind = stabilize_grip(base, **(inputs() | changes))
    assert profile is base and weight == 1


@pytest.mark.parametrize('condition', ['A', 'B', 'C'])
def test_stabilization_keeps_baseline_identity_and_never_raises_existing_limits(condition):
    base = motion_control_profile(condition)
    profile, weight, kind = stabilize_grip(base, **inputs())
    assert profile.condition == base.condition
    assert profile.stable_arrival_enabled and profile.stable_steps_required >= 3
    assert profile.nominal_joint_velocity_limit_rad_s <= min(.2, base.nominal_joint_velocity_limit_rad_s)
    assert profile.carrying_joint_velocity_limit_rad_s <= base.carrying_joint_velocity_limit_rad_s
    assert profile.progress_stall_enabled == base.progress_stall_enabled
    assert profile.sequential_route_preview_enabled == base.sequential_route_preview_enabled
    assert weight == 25
    assert kind == 'fixture'
    slower = replace(base, nominal_joint_velocity_limit_rad_s=.08, stable_steps_required=5)
    profile, _, _ = stabilize_grip(slower, **inputs())
    assert profile.nominal_joint_velocity_limit_rad_s == .08 and profile.stable_steps_required == 5


@pytest.mark.parametrize('status,expected', [('tentative', True), ('confirmed', True), ('released', False), ('', False)])
def test_held_object_requires_resolved_attachment_and_measured_bilateral_contact(status, expected):
    values = inputs() | {'authorization': None, 'orientation_change_rad': .001,
                        'attachment_proxy': {'object_name': 'held-object', 'status': status}}
    base = motion_control_profile('A')
    profile, weight, kind = stabilize_grip(base, **values)
    assert (kind == 'attached_object') is expected
    assert (weight == 25) is expected
    for bad in ({'contact': {'available': False}}, {'gripper_command': -1},
                {'attachment_proxy': {'object_name': '', 'status': 'confirmed'}}):
        profile, weight, kind = stabilize_grip(base, **(values | bad))
        assert profile is base and weight == 1 and kind == 'none'


@pytest.mark.parametrize('angle,expected_weight', [(0., 25.), (.03, 25.), (.05, 25.),
    (.051, 1.), (.15, 1.), (.785, 1.), (3.14159, 1.)])
def test_deliberate_held_rotation_preserves_speed_and_arrival_without_hold_weight(angle, expected_weight):
    values = inputs() | {'authorization': None, 'orientation_change_rad': angle,
                        'attachment_proxy': {'object_name': 'held-object', 'status': 'tentative'}}
    base = motion_control_profile('A')
    profile, weight, kind = stabilize_grip(base, **values)
    assert kind == 'attached_object'
    assert weight == expected_weight
    assert profile.stable_arrival_enabled and profile.stable_steps_required >= 3
    assert profile.carrying_joint_velocity_limit_rad_s <= min(.2, base.carrying_joint_velocity_limit_rad_s)
    assert profile.nominal_joint_velocity_limit_rad_s <= min(.2, base.nominal_joint_velocity_limit_rad_s)
    assert profile.progress_stall_enabled == base.progress_stall_enabled


def test_collision_feedback_distinguishes_non_target_obstacle_without_scene_identity():
    import json
    from sim.controllers.collision_feedback import classify_collision
    from agent.tools.sim_mcp import _agent_visible_simulator_response
    from agent.runtime.response_artifacts import build_motion_summary
    from tools.codex_feedback import motion_feedback
    policy = {'robot_geom_ids': [1, 2, 3], 'gripper_geom_ids': [2, 3],
              'finger_geom_ids': [3], 'authorized_target_geom_ids': [10],
              'authorized_target_geom_count': 1}
    raw = {'detected': True, 'geom1_id': 2, 'geom2_id': 11,
           'geom2_name': 'private_cabinet_id', 'minimum_distance_m': -.004,
           'check_mode': 'pre_actuation_configuration'}
    classified = classify_collision(raw, policy)
    assert classified['gripper_part'] == 'base_or_palm'
    assert classified['obstacle_relation'] == 'outside_contact_target'
    summary = build_motion_summary(_agent_visible_simulator_response({
        'collision': classified, 'stop_reason': 'collision_detected', 'steps_executed': 1}))
    out = motion_feedback({'tool_calls': [{'name': 'move_to', 'result': {
        'details': {'outputs': {'motion_summary': summary}}}}]})
    assert out['motion_summary']['collision']['obstacle_relation'] == 'outside_contact_target'
    assert 'Re-marking the same target will not clear' in out['recovery']['message']
    assert 'base/palm' in out['recovery']['message']
    assert all(s not in json.dumps(out) for s in ['private_cabinet_id', 'geom1_id', '-0.004'])
    assert classify_collision(raw | {'geom1_id': 3, 'geom2_id': 10}, policy)['obstacle_relation'] == 'authorized_target'
    assert classify_collision(raw | {'geom1_id': 3, 'geom2_id': 1}, policy)['obstacle_relation'] == 'robot_self'
    assert classify_collision(raw, policy | {'authorized_target_geom_count': 0})['obstacle_relation'] == 'unbound_world'
