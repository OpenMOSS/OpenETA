"""Bounded atomic routes; each segment uses the ordinary Host motion hook."""
from copy import deepcopy
import json

from tools.codex_atomic_geometry import vector, orientation, matrix_quat


def compile_route(atomic, args):
    if args.get('motion_mode', 'strict') != 'strict':
        raise ValueError('Recovery uses one short segment, not a waypoint route')
    """Validate the whole request before actuation, without preauthorizing IK."""
    if args.get('contact_point_id') or atomic.contact is not None:
        raise ValueError('Routes require no active contact binding; perform contact moves separately')
    if args.get('orientation_mode', 'strict') != 'strict':
        raise ValueError('Routes use strict orientation; select symmetric orientation in a separate move')
    final = {k: v for k, v in args.items() if k not in ('waypoints', 'preview', 'orientation_mode', 'motion_mode')}
    if not final:
        raise ValueError('A route requires a top-level final target')
    state = atomic.state()
    xyz = vector(state['grip_xyz_m'])
    rotation = orientation([0, 0, 0, 1], state['approach_world'], state['jaw_world'])
    poses = []
    for step in [*args['waypoints'], final]:
        if set(step) - {'xyz_m', 'point_id', 'offset_m', 'approach_world', 'jaw_world'}:
            raise ValueError('Route poses accept absolute xyz_m, point_id+offset_m, or orientation only; no deltas/contact')
        if not step or ('xyz_m' in step and 'point_id' in step) or ('offset_m' in step and 'point_id' not in step):
            raise ValueError('Each route pose needs one position form or an orientation')
        if 'xyz_m' in step:
            xyz = vector(step['xyz_m'])
        elif 'point_id' in step:
            xyz = vector(atomic.point(step['point_id'])['xyz_m']) + vector(step.get('offset_m', [0, 0, 0]))
        rotation = orientation(matrix_quat(rotation), step.get('approach_world'), step.get('jaw_world'))
        poses.append({'xyz_m': xyz.tolist(), 'approach_world': rotation[:, 2].tolist(),
                      'jaw_world': rotation[:, 0].tolist()})
    return poses


def run_route(atomic, args):
    poses = compile_route(atomic, args)
    host = atomic.host
    start_calls = host.runner.tool_call_count
    receipt = {'schema_version': 'openeta.atomic_route.v1', 'segment_count': len(poses),
               'completed_count': 0, 'segments': [], 'targets': poses,
               'index_base': 0, 'remaining_indices': list(range(len(poses))),
               'automatic_resume': False, 'path_check': 'not_run'}
    if args.get('preview'):
        atomic.feedback = {'route': receipt, 'motion': 'preview_only; no IK or physics executed'}
        atomic.overlay = {'route': [atomic.state()['grip_xyz_m'], *[p['xyz_m'] for p in poses]]}
        return None
    error = None
    def journal(phase, index):
        try:
            with (host.output / 'atomic-route-progress.jsonl').open('a') as stream:
                stream.write(json.dumps({'request_index': host.requests, 'phase': phase,
                    'segment_index': index, 'route': receipt}) + '\n')
        except OSError:
            # Preserve known physical outcomes when disk recording fails.
            receipt['audit_status'] = 'write_failed'

    for index, pose in enumerate(poses):
        journal('segment_start', index)
        atomic.feedback = None
        intervention_start = len(host.interventions)
        # Never reserve or refresh a whole path's authorizations at its start.
        try:
            error = atomic._move(pose)
        except (ValueError, KeyError, OSError) as exc:
            error = {'code': 'route_segment_error', 'message': str(exc)}
        finally:
            atomic.pending = None
            for intervention in host.interventions[intervention_start:]:
                intervention.setdefault('details', {})['route_segment_index'] = index
        # Same observation/epoch publication as consecutive native move calls,
        # without encoding intermediate RGB images.
        host.publish_observation()
        feedback = deepcopy(atomic.feedback or {})
        motion = feedback.get('motion') or {}
        reached = (not error and motion.get('reason_code') == 'target_reached'
                   and (motion.get('motion_summary') or {}).get('reached_target') is True)
        receipt['segments'].append({'index': index, 'reached': reached,
                                    'actual_robot': atomic.state(), 'feedback': feedback,
                                    'error': error})
        journal('segment_result', index)
        if not reached:
            error = error or {'code': 'route_segment_not_reached',
                             'message': 'Segment arrival is unconfirmed; inspect actual state and replan'}
            receipt['stopped_index'] = index
            receipt['remaining_indices'] = list(range(index + 1, len(poses)))
            break
        receipt['completed_count'] += 1
        receipt['remaining_indices'] = list(range(index + 1, len(poses)))
    receipt.update(status='stopped' if error else 'completed',
                   internal_tool_calls=host.runner.tool_call_count - start_calls,
                   path_check='per_segment', actual_robot=atomic.state())
    if error:
        host.record_intervention('host', 'stopped', 'route_stopped',
            'The route stopped at the reported segment; later segments were not attempted and will not resume automatically.',
            execution_state=host.request_execution_state(),
            details={'completed_count':receipt['completed_count'], 'stopped_index':receipt['stopped_index'],
                     'remaining_indices':receipt['remaining_indices']})
    # _move's final feedback is retained alongside the complete route receipt.
    atomic.feedback = {**(atomic.feedback or {}), 'route': receipt}
    journal('route_result', len(receipt['segments'])-1)
    return error
