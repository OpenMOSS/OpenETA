"""Offline curation of native expert successes; no simulation or policy calls."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np

from sim.envs.univtac.trace import write_json


def curate(source: Path, destination: Path, seed: int) -> dict:
    final = json.loads((source/'final_result.json').read_text())
    child = json.loads((source/'child_result.json').read_text())
    if not (final['expert_episode_success'] and final['native_check_success']
            and final['plan_success'] and not final['native_check_early_stop']
            and child['counters']['play_once_call_count'] == 1):
        raise ValueError('Source is not a confirmed native expert success')
    destination.mkdir(parents=True, exist_ok=False)
    transitions = sorted((source/'transitions').glob('*/transition.json'))
    first = json.loads(transitions[0].read_text())['simulator_step_range'][0]
    example = {'task_goal':'Insert the held object into the hole.',
               'source_kind':'native_Isaac51_expert_success', 'trajectory_id':f'insert_hole_support_{seed}',
               'sampling':'sparse action boundaries; no within-action frames',
               'robot_state_convention':{'ee':'native base-frame EE position in metres followed by quaternion wxyz; not OpenETA TCP',
                                         'joint':'seven arm positions in radians followed by two finger positions in metres'},
               'native_outcome':{'available':True, 'success':True}, 'steps':[]}
    review = []
    for i, path in enumerate(transitions, 1):
        transition = json.loads(path.read_text())
        step = {'index':i}
        for phase in ('before','after'):
            snapshot = json.loads((path.parent/f'snapshot_{phase}.json').read_text())['operator_visible']
            simulator_step = snapshot['step_identifiers']['simulator_step']
            observation = {'native_step':simulator_step, 'simulation_seconds_since_handoff':(simulator_step-first)/120,
                           'robot':copy.deepcopy(snapshot['proprio']), 'vision':[], 'touch':[]}
            for name, payload in [(n,snapshot['cameras'][n]['rgb']) for n in ('head','wrist')] + [
                (n,snapshot['tactile'][n]['rgb_marker']) for n in ('left_tactile','right_tactile')]:
                relative = Path('images')/f'{i:02d}_{phase}_{name}.png'
                (destination/relative).parent.mkdir(exist_ok=True)
                shutil.copyfile(source/payload['path'], destination/relative)
                observation['vision' if name in ('head','wrist') else 'touch'].append({'label':name,'path':str(relative)})
            step[phase] = observation
            review.append({'step':i, 'phase':phase, **observation})
        native = transition['native_actions_after'][0]
        target = native['target_pose']
        position = target['position']['values']
        step['action'] = {
            'actual_api':'native task.move; NOT OpenETA move_to',
            'action_type':native['action_type'],
            'target_native_ee_position_m':position,
            'target_native_ee_quaternion_wxyz':target['quaternion']['values'],
            'coordinate_frame':'native robot-base EE frame; not OpenETA gripper-center TCP',
            'requested_translation_from_before_ee_m':(np.array(position)-np.array(step['before']['robot']['ee'][:3])).tolist(),
            'gripper_request':'not explicitly changed by this native expert move',
            'time_dilation_factor':transition['move_kwargs']['time_dilation_factor'],
            'observed_duration_seconds':(transition['simulator_step_range'][1]-transition['simulator_step_range'][0])/120,
            'executed_translation_m':(np.array(step['after']['robot']['ee'][:3])-np.array(step['before']['robot']['ee'][:3])).tolist(),
        }
        step['feedback'] = {'move_returned':transition['move_returned'],
                            'planner_success':transition['plan_success_after']}
        example['steps'].append(step)
    visual = copy.deepcopy(example)
    for step in visual['steps']:
        for phase in ('before','after'):
            del step[phase]['touch']
    write_json(destination/'tactile-action.json', example)
    write_json(destination/'visual-action.json', visual)
    write_json(destination/'sparse_review.json', {'frames':review})
    manifest = {'manifest_visibility':'host_only_provenance',
                'agent_visible_files':['visual-action.json','tactile-action.json'], 'seed':seed, 'source_root':str(source.resolve()), 'source_kind':'R1.2 native play_once, collect mode',
                'source_commit':'371fac67917307026be8f00869fcc1b61c623a9f',
                'native_success':True, 'new_simulator_episodes':0, 'transition_count':len(transitions),
                'source_final_result':final, 'source_counters':child['counters'],
                'source_task_config':child['task_config'],
                'native_execution':'original planner/dense move, default force=True; not dynamic OpenETA IK',
                'time_basis':'collect control=physics=120 Hz, decimation=1; boundary step differences / 120',
                'robot_pose_convention':'native EE position metres + quaternion wxyz; arm joints radians, fingers metres',
                'limitations':['Only before/after snapshots; no dense motion/tactile history or instantaneous speed',
                    'Sensor frame IDs, gripper target-buffer history and per-frame physics counters not retained',
                    'Final checker follows native delay; no new image invented at checker time',
                    'Not official dense HDF5 collection: R1.2 disabled native save/video frequency'],
                'paired_exports':['visual-action.json','tactile-action.json']}
    write_json(destination/'manifest.json',manifest)
    return manifest
