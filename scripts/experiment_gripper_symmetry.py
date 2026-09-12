"""Paired controller experiment, not an autonomous or exact historical replay.

Run with the LIBERO interpreter and Mink profile. Every condition starts from
one complete snapshot in this dedicated test environment. No production/live
episode is stepped or rolled back by this diagnostic.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from sim import bench_worker as worker
from sim.controllers.gripper_guard import suspend_camera_observables
from sim.controllers.mink_goal import execute_libero_mink_goal, _libero_runtime
from sim.private_state import capture, restore
from sim.reachability import check_endpoint_reachability
from tools.codex_orientation import equivalent_rotations, rotation_distance, endpoint_score


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {'model_calls': 0, 'autonomous_task_sample': False, 'historical_replay': False,
              'condition': 'recorded arm start; zero velocity; new seed=0 scene; paired complete snapshot',
              'candidates': [], 'runs': []}
    files = list(source.glob('workspace/sessions/*/artifacts/responses/*/*-move_to/move_to-response.json'))
    def response(index):
        return json.loads(next(p for p in files if '-'+index+'-' in p.parent.name).read_text())
    prior, original = response('0007'), response('0009')
    original_receipt_id = original['controller_receipt']['ik_execution_seed_receipt_id']
    for row in map(json.loads, (source/'host-commands.jsonl').read_text().splitlines()):
        for call in row['command']['tool_calls']:
            receipt = call.get('result', {}).get('details', {}).get('outputs', {}).get('ik_preview_receipt')
            if receipt and receipt['receipt_id'] == original_receipt_id:
                recorded_seed = receipt['best_candidate']['joint_positions']
    env = None
    try:
        env = worker._make_env_locked('openeta/libero_libero_goal_task3-v0', seed=0,
                                     image_width=128, image_height=128, include_objects=True)
        worker._reset_with_image(env, seed=0)
        ue = env.unwrapped
        raw, robot = _libero_runtime(ue)
        raw.sim.data.qpos[robot._ref_joint_pos_indexes] = prior['observation']['robot']['joint_positions']
        raw.sim.data.qvel[:] = 0.
        raw.sim.forward()
        robot.controller.update(force=True)
        robot.controller.reset_goal()
        snapshot = capture(env, 'paired-start', root=output/'private')
        report['snapshot'] = str(snapshot)
        start_q = raw.sim.data.qpos.copy()
        start_v = raw.sim.data.qvel.copy()
        start = Rotation.from_quat(original['start']['quat_xyzw']).as_matrix()
        target = original['target']
        xyz = [target[k] for k in ('x', 'y', 'z')]
        rotations = equivalent_rotations(Rotation.from_quat(target['quat_xyzw']).as_matrix())
        for index, rotation in enumerate(rotations):
            began = time.monotonic()
            reach = check_endpoint_reachability(env, target_xyz=xyz,
                target_quat_xyzw=Rotation.from_matrix(rotation).as_quat().tolist(),
                preserve_current_orientation=False, position_tolerance_m=.002, orientation_tolerance_rad=.05)
            assert np.array_equal(start_q, raw.sim.data.qpos) and np.array_equal(start_v, raw.sim.data.qvel)
            item = {'index': index, 'quat_xyzw': Rotation.from_matrix(rotation).as_quat().tolist(),
                    'ik': reach, 'ik_wall_s': time.monotonic()-began}
            if reach.get('feasible') is True:
                item.update(endpoint_score(reach['best_candidate'], rotation_distance(start, rotation)))
            report['candidates'].append(item)
            print(json.dumps({'candidate': index, 'ik': reach.get('reason_code'),
                              'score': item.get('score'), 'margin': item.get('joint_margin_min_rad')}), flush=True)
        feasible = [c for c in report['candidates'] if 'score' in c]
        report['endpoint_selected_index'] = min(feasible, key=lambda c:c['score'])['index'] if feasible else None
        conditions = [('recorded_strict_seed', rotations[0], recorded_seed)]
        conditions.extend((f'fresh_candidate_{c["index"]}', rotations[c['index']], c['ik']['best_candidate']['joint_positions']) for c in feasible)
        for name, rotation, joints in conditions:
            restore(env, snapshot)
            assert np.array_equal(start_q, raw.sim.data.qpos) and np.array_equal(start_v, raw.sim.data.qvel)
            samples = []
            def step(action, render):
                result = worker._step_with_image(env, action, render=False)
                samples.append(raw.sim.data.qpos[robot._ref_joint_pos_indexes].tolist())
                return result
            began = time.monotonic()
            with suspend_camera_observables(raw):
                result = execute_libero_mink_goal(ue, target_xyz=xyz,
                    target_quat_xyzw=Rotation.from_matrix(rotation).as_quat().tolist(),
                    preserve_current_orientation=False, max_steps=150,
                    position_tolerance_m=.002, orientation_tolerance_rad=.05,
                    gripper_command=-1., enable_collision_check=True,
                    contact_authorization=None, attachment_proxy=None,
                    ik_execution_seed={'schema_version':'openeta.ik_execution_seed.v1',
                                       'receipt_id':name, 'joint_positions':joints}, step_callback=step)
            entry = {'condition': name, 'wall_s': time.monotonic()-began,
                     'result': {k:v for k,v in result.items() if k!='observation'}, 'joint_path':samples}
            report['runs'].append(entry)
            (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({'condition':name, 'stop':result['stop_reason'],
                'steps':result['steps_executed'], 'orientation_error_deg':result.get('orientation_error_deg'),
                'position_error_m':result.get('position_error_m'), 'wall_s':entry['wall_s']}), flush=True)
        report['completed'] = True
    finally:
        if env is not None:
            env.close()
            report['cleanup'] = True
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
