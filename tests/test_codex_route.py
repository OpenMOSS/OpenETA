import json
from copy import deepcopy
import pytest
from test_codex_motion_hook import rig
from test_codex_atomic import atomic, body, mark
from tools.codex_atomic_geometry import quat_matrix, body_to_grip_site


@pytest.fixture(autouse=True)
def measured_dummy_environment(atomic, monkeypatch):
    # The generic dummy emits no pose. Supply measured state between runner steps,
    # as the simulator adapter does, while retaining real hooks and budgets.
    host, state = atomic
    execute = host._execute
    measured = deepcopy(host.runner.current_observation.robot.end_effector_pose)
    def step(payload, **kwargs):
        nonlocal measured
        result = execute(payload, **kwargs)
        if payload['name'] == 'move_to' and state['calls'][-1][0] == 'move_to':
            measured = deepcopy(state['calls'][-1][1]['target_pose'])
            measured['xyz'][0] += .0005  # next IK must start from this actual state
        host.runner.current_observation.robot.end_effector_pose = deepcopy(measured)
        return result
    monkeypatch.setattr(host, '_execute', step)


def test_route_uses_fresh_budgeted_ik_for_every_pose_and_inherits_orientation(atomic):
    host, state = atomic
    request = {'waypoints': [{'xyz_m': [0, 0, 1.1], 'approach_world': [0, 0, -1], 'jaw_world': [1, 0, 0]},
                              {'xyz_m': [.1, 0, 1.1]}], 'xyz_m': [.1, 0, 1.]}
    result = host.call('move_to', request)
    assert not result.isError, json.dumps(body(result), indent=2)
    route = body(result)['feedback']['route']
    assert route['completed_count'] == 3 and route['remaining_indices'] == []
    assert host.requests == 1 and host.runner.tool_call_count == 9
    assert route['internal_tool_calls'] == 9
    assert route['segments'][0]['actual_robot']['grip_xyz_m'][0] == .0005
    assert route['segments'][1]['actual_robot']['grip_xyz_m'][0] == .1005
    assert [n for n, _ in state['calls']] == ['ik_preview_check', 'move_to'] * 3
    moves = [p for n, p in state['calls'] if n == 'move_to']
    assert [p['target_pose']['xyz'] for p in moves] == [[0, 0, 1.1], [.1, 0, 1.1], [.1, 0, 1.]]
    assert all(p['target_pose']['quat_xyzw'] == moves[0]['target_pose']['quat_xyzw'] for p in moves)
    rows = [json.loads(s) for s in host.atomic.audit.read_text().splitlines()]
    assert len(rows) == 1 and len(rows[0]['feedback']['route']['segments']) == 3


@pytest.mark.parametrize('bad', [
    {'xyz_m': [1, 0, 1], 'point_id': 'unknown'},
    {'point_id': 'unknown'}, {'approach_world': [0, 0, 0], 'jaw_world': [1, 0, 0]},
    {'offset_m': [0, 0, .1]}, {},
])
def test_invalid_later_waypoint_never_executes_first_segment(atomic, bad):
    host, state = atomic
    assert host.call('move_to', {'waypoints': [{'xyz_m': [0, 0, 1.1]}, bad], 'xyz_m': [0, 0, 1]}).isError
    assert state['calls'] == [] and host.runner.tool_call_count == 0


@pytest.mark.parametrize('extra', [
    {'delta_m': [0, 0, .1]}, {'orientation_mode': 'parallel_jaw_symmetric'},
    {'contact_point_id': 'unknown'},
])
def test_unsupported_route_modes_do_not_actuate(atomic, extra):
    host, state = atomic
    assert host.call('move_to', {'waypoints': [{'xyz_m': [0, 0, 1.1]}], 'xyz_m': [0, 0, 1], **extra}).isError
    assert not state['calls']


def test_route_budget_expires_between_segments_without_dispatching_rest(atomic):
    host, state = atomic
    host.runner.max_turns = 5
    r = host.call('move_to', {'waypoints': [{'xyz_m': [0, 0, 1.1]}], 'xyz_m': [.1, 0, 1.1]})
    route = body(r)['feedback']['route']
    assert r.isError and route['completed_count'] == 1 and route['stopped_index'] == 1
    assert [n for n, _ in state['calls']] == ['ik_preview_check', 'move_to', 'ik_preview_check']
    assert route['segments'][-1]['feedback']['motion']['execution_state'] == 'not_started'


def test_route_stops_on_missing_arrival_even_if_native_handler_returns_success(atomic):
    host, state = atomic
    state['stop_reason'] = 'iteration_limit'
    r = host.call('move_to', {'waypoints': [{'xyz_m': [0, 0, 1.1]}], 'xyz_m': [.1, 0, 1.1]})
    route = body(r)['feedback']['route']
    assert r.isError and route['completed_count'] == 0 and route['remaining_indices'] == [1]
    assert len(state['calls']) == 2
    assert route['segments'][0]['feedback']['motion']['execution_state'] == 'unknown'


def test_route_preview_resolves_points_and_draws_without_ik(atomic):
    host, state = atomic
    point = body(mark(host))['feedback']['point']['point_id']
    r = host.call('move_to', {'waypoints': [{'point_id': point, 'offset_m': [0, 0, .1]}],
                              'xyz_m': [.1, 0, 1.1], 'preview': True})
    assert not r.isError, body(r)
    assert not state['calls'] and host.runner.tool_call_count == 0
    assert body(r)['feedback']['route']['path_check'] == 'not_run'
    assert any(c.type == 'image' for c in r.content)


def test_route_limit_is_validated_before_execution(atomic):
    host, state = atomic
    r = host.call('move_to', {'waypoints': [{'xyz_m': [0, 0, 1]}] * 5, 'xyz_m': [0, 0, 1.1]})
    assert r.isError and not state['calls']


@pytest.mark.parametrize('steps,execution', [(0, 'not_started'), (3, 'partial')])
def test_route_stops_on_known_zero_or_partial_collision(atomic, steps, execution):
    from test_codex_failure_feedback import failed_handler
    host, state = atomic
    host.runtime.tools.bind_handler('move_to', failed_handler('collision_detected', steps), replace=True)
    r = host.call('move_to', {'waypoints': [{'xyz_m': [0, 0, 1.1]}], 'xyz_m': [.1, 0, 1.1]})
    route = body(r)['feedback']['route']
    assert r.isError and route['completed_count'] == 0 and route['stopped_index'] == 0
    assert route['remaining_indices'] == [1] and route['automatic_resume'] is False
    assert route['segments'][0]['feedback']['motion']['execution_state'] == execution
    assert host.runner.tool_call_count == 3
    assert [n for n, _ in state['calls']] == ['ik_preview_check']


def test_transport_stop_between_segments_prevents_further_ik_or_motion(atomic, monkeypatch):
    host, state = atomic
    original = host.atomic._move
    def first_then_disconnect(pose):
        error = original(pose)
        host.request_stop()
        return error
    monkeypatch.setattr(host.atomic, '_move', first_then_disconnect)
    r = host.call('move_to', {'waypoints': [{'xyz_m': [0, 0, 1.1]}], 'xyz_m': [.1, 0, 1.1]})
    route = body(r)['feedback']['route']
    assert r.isError and route['completed_count'] == 1
    assert [n for n, _ in state['calls']] == ['ik_preview_check', 'move_to']
    assert route['segments'][1]['feedback']['motion']['execution_state'] == 'not_started'
    progress = [json.loads(s) for s in (host.output / 'atomic-route-progress.jsonl').read_text().splitlines()]
    assert progress[0]['phase'] == 'segment_start'
    assert progress[-1]['phase'] == 'route_result'
    assert progress[-1]['route']['completed_count'] == 1
