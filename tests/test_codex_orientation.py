import math
from copy import deepcopy

import numpy as np
import pytest

from test_codex_atomic import atomic, body
from test_codex_motion_hook import rig
from agent.tools.registry import ToolResult
from agent.tools.sim_mcp import _ik_preview_receipt, _ik_execution_authorization
from tools.codex_orientation import equivalent_rotations, endpoint_score
from tools.codex_atomic_geometry import orientation, quat_matrix


ARGS = {'delta_m': [.01, 0, 0], 'approach_world': [0, 0, 1],
        'jaw_world': [-1, 0, 0], 'orientation_mode': 'parallel_jaw_symmetric'}


def bind_previews(host, state, *, feasible=(True, True), metrics=None):
    state['receipts'] = []
    state['preview_count'] = 0
    host.runner.current_observation.robot.gripper_state['openness'] = 1.
    if metrics is None:
        metrics = [(2., .2), (.3, 1.)]
    def preview(ctx):
        i = state['preview_count']
        state['preview_count'] += 1
        state['calls'].append(('ik_preview_check', deepcopy(ctx.parameters)))
        if state.get('transport_failure'):
            raise TimeoutError('Synthetic IK transport timeout')
        travel, margin = metrics[i]
        reach = {'feasible': feasible[i], 'status': 'reachable' if feasible[i] else 'unknown',
                 'kinematic_status': 'reachable' if feasible[i] else 'unknown',
                 'reason_code': 'ik_solution_found' if feasible[i] else 'ik_search_no_solution',
                 'collision': {'checked': True, 'collision': False},
                 'best_candidate': {'joint_positions': [float(i)]*7,
                     'joint_margin_min_rad': margin, 'joint_travel_l2_rad': travel}}
        receipt = _ik_preview_receipt(ctx.parameters, reach)
        state['receipts'].append(receipt)
        authorization = _ik_execution_authorization(receipt)
        return ToolResult(True, 'Synthetic candidate IK', {'outputs': {
            'ik_preview_receipt': receipt, 'ik_receipt_id': receipt['receipt_id'],
            'execution_authorization': authorization, 'reachability': reach,
            'motion_execution_ref': {'ik_receipt_id': receipt['receipt_id']} if authorization['authorized_for_move_to'] else {}}})
    host.runtime.tools.bind_handler('ik_preview_check', preview, replace=True)


def test_equivalence_is_local_roll_and_not_quaternion_sign():
    r = orientation([0, 0, 0, 1], [1, 2, -3], [2, -1, 0])
    a, b = equivalent_rotations(r)
    assert np.allclose(a[:, 2], b[:, 2])
    assert np.allclose(a[:, 0], -b[:, 0])
    assert np.allclose(b.T @ b, np.eye(3)) and np.isclose(np.linalg.det(b), 1)
    assert not np.allclose(a, b)
    with pytest.raises(ValueError):
        equivalent_rotations(np.diag([-1, 1, 1]))


def test_joint_limit_penalty_can_outweigh_shorter_cartesian_rotation():
    near = endpoint_score({'joint_travel_l2_rad': .1, 'joint_margin_min_rad': .001}, 0.)
    far = endpoint_score({'joint_travel_l2_rad': .8, 'joint_margin_min_rad': .3}, math.pi)
    assert far['score'] < near['score']


@pytest.mark.parametrize('metrics', [{'joint_travel_l2_rad': float('nan'), 'joint_margin_min_rad': .2},
                                   {'joint_travel_l2_rad': 1., 'joint_margin_min_rad': -.001},
                                   {'joint_travel_l2_rad': True, 'joint_margin_min_rad': .2}, {}])
def test_invalid_metrics_cannot_be_ranked(metrics):
    with pytest.raises(ValueError):
        endpoint_score(metrics, .2)


@pytest.mark.parametrize('metrics,selected', [([(2., .2), (.3, 1.)], 1), ([(.1, .2), (3., .2)], 0)])
def test_symmetric_move_executes_exact_selected_receipt_and_pose(atomic, metrics, selected):
    host, state = atomic
    bind_previews(host, state, metrics=metrics)
    result = host.call('move_to', ARGS)
    assert not result.isError, body(result)
    assert [n for n, _ in state['calls']] == ['ik_preview_check', 'ik_preview_check', 'move_to']
    assert host.runner.tool_call_count == host.runner.turn_index == 5
    move = state['calls'][-1][1]
    assert move['ik_receipt_id'] == state['receipts'][selected]['receipt_id']
    assert move['target_pose'] == state['receipts'][selected]['target_pose']
    info = body(result)['feedback']['orientation_selection']
    assert info['selected_index'] == selected and info['path_check'] == 'not_run'
    assert info['symmetry_applied'] is bool(selected)
    expected_jaw = np.asarray(ARGS['jaw_world']) * (-1 if selected else 1)
    assert np.allclose(body(result)['feedback']['target']['jaw_world'], expected_jaw)
    assert np.allclose(quat_matrix(info['selected_target']['quat_xyzw'])[:, 0], expected_jaw)


@pytest.mark.parametrize('feasible', [(False, True), (True, False), (False, False)])
def test_negative_ik_only_allows_other_authorized_candidate(atomic, feasible):
    host, state = atomic
    bind_previews(host, state, feasible=feasible)
    result = host.call('move_to', ARGS)
    moves = [args for name, args in state['calls'] if name == 'move_to']
    if any(feasible):
        assert not result.isError, body(result)
        assert len(moves) == 1
        assert moves[0]['ik_receipt_id'] == state['receipts'][feasible.index(True)]['receipt_id']
    else:
        assert result.isError and not moves
        assert body(result)['feedback']['motion']['physics_executed'] is False


@pytest.mark.parametrize('block', ['load', 'contact', 'closed', 'unknown'])
def test_symmetry_not_allowed_when_manipulation_state_is_constrained(atomic, block):
    host, state = atomic
    bind_previews(host, state)
    if block == 'load':
        host.atomic.symmetry_load_uncertain = True
    elif block == 'contact':
        host.atomic.contact = {'active': True}
    elif block == 'closed':
        host.runner.current_observation.robot.gripper_state['openness'] = .4
    else:
        host.runner.current_observation.robot.gripper_state.pop('openness')
    assert host.call('move_to', ARGS).isError
    assert not state['calls']


def test_symmetric_preview_remains_render_only(atomic):
    host, state = atomic
    bind_previews(host, state)
    result = host.call('move_to', {**ARGS, 'preview': True})
    assert not result.isError, body(result)
    assert not state['calls'] and host.runner.tool_call_count == 0
    assert body(result)['feedback']['orientation_selection']['selection_status'] == 'geometry_only'


def test_no_orientation_request_preserves_pose_and_uses_single_preview(atomic):
    host, state = atomic
    result = host.call('move_to', {'delta_m': [.01, 0, 0], 'orientation_mode': 'parallel_jaw_symmetric'})
    assert not result.isError, body(result)
    assert [name for name, _ in state['calls']] == ['ik_preview_check', 'move_to']
    assert host.runner.tool_call_count == 3


def test_symmetry_budget_and_transport_errors_do_not_dispatch(atomic):
    host, state = atomic
    bind_previews(host, state)
    host.runner.max_turns = 4
    result = host.call('move_to', ARGS)
    assert result.isError, body(result)
    assert not any(name == 'move_to' for name, _ in state['calls'])
    assert body(result)['feedback']['motion']['physics_executed'] is False


def test_symmetry_transport_error_does_not_retry_second_candidate(atomic):
    host, state = atomic
    bind_previews(host, state)
    state['transport_failure'] = True
    assert host.call('move_to', ARGS).isError
    assert [name for name, _ in state['calls']] == ['ik_preview_check']


def test_epoch_change_after_previews_stops_selected_motion(atomic, monkeypatch):
    from tools import codex_motion
    host, state = atomic
    bind_previews(host, state)
    original = codex_motion.run_motion_hook
    epoch = host.runtime.memory.robot_motion_epoch()
    def preflight(*args, **kwargs):
        result = original(*args, **kwargs)
        if state['preview_count'] == 2:
            monkeypatch.setattr(host.runtime.memory, 'robot_motion_epoch', lambda: epoch + 1)
        return result
    monkeypatch.setattr(codex_motion, 'run_motion_hook', preflight)
    result = host.call('move_to', ARGS)
    assert result.isError, body(result)
    assert body(result)['feedback']['motion']['reason_code'] == 'stale_orientation_preflight'
    assert not any(name == 'move_to' for name, _ in state['calls'])


def test_chosen_contact_pose_matches_authorization_binding(atomic):
    from agent.tools.registry import ENVIRONMENT_AUTHORITY
    from test_codex_atomic import mark
    host, state = atomic
    bind_previews(host, state)
    point = body(mark(host))['feedback']['point']['point_id']
    original = host.runtime.tools._handlers['move_to']
    def move(ctx):
        grant = host.atomic.motion_authorization(ctx.parameters['target_pose'])
        assert grant['point_id'] == point
        # Selected physical site is identity; shared controller still takes
        # the hand-body quaternion, whose site has a -90 degree Z offset.
        expected_body = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        assert np.allclose(quat_matrix(ctx.parameters['target_pose']['quat_xyzw']), expected_body)
        return original(ctx)
    host.runtime.tools.bind_handler('move_to', move, replace=True, authority=ENVIRONMENT_AUTHORITY)
    result = host.call('move_to', {**ARGS, 'contact_point_id': point})
    assert not result.isError, body(result)
    assert host.atomic.symmetry_load_uncertain
