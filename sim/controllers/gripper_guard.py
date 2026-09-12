"""Check LIBERO finger actuation at every physics control tick, without rendering.

This checks current and actual post-step configurations, not a dynamics forecast.
An opening command may escape an existing penetration only with verified progress.
"""
import numpy as np
from contextlib import contextmanager

from sim.controllers.mink_goal import (
    _libero_runtime, _libero_collision_policy, _collision_pair_distances,
    _collision_distance_report, _collision_receipt,
)
from sim.controllers.collision_recovery import verified_collision_boundary_escape
from sim.controllers.collision_feedback import classify_collision


@contextmanager
def suspend_camera_observables(raw):
    observables = getattr(getattr(raw,'env',None),'_observables',{})
    disabled=[]
    for name,observable in observables.items():
        if (name.endswith('_image') or name.endswith('_depth') or 'segmentation' in name) and observable.is_enabled():
            observable.set_enabled(False);disabled.append(observable)
    try:
        yield
    finally:
        for observable in disabled:observable.set_enabled(True)


def execute_checked_gripper(env, **kwargs):
    raw,_ = _libero_runtime(env)
    # render=False suppresses transport encoding, but robosuite's camera
    # observables otherwise still render INSIDE env.step(). Suspend them too.
    with suspend_camera_observables(raw):
        return _execute_checked_gripper(env, **kwargs)


def _execute_checked_gripper(env, *, action, max_steps, contact_authorization, step_callback):
    import mink
    if type(max_steps) is not int or not 1 <= max_steps <= 60:
        raise ValueError('gripper horizon must be 1..60 control steps')
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (8,) or not np.isfinite(action).all() or np.any(action[:7] != 0) or action[7] not in (-1, 1):
        raise ValueError('checked gripper requires zero arm velocity and a binary finger command')
    raw, robot = _libero_runtime(env)
    policy = _libero_collision_policy(raw, robot, contact_authorization=contact_authorization, attachment_proxy=None)
    config = mink.Configuration(raw.sim.model._model, q=raw.sim.data.qpos.copy())
    result = {}; steps = 0; stop = 'gripper_horizon_completed'
    before = _collision_pair_distances(config, policy['protected_pairs'])
    collision = _collision_distance_report(config, policy['protected_pairs'],
        distance_limit_m=policy['hard_stop_distance_m'], pair_distances=before)
    mode = 'pre_actuation_configuration'
    # Do not squeeze further into an already protected contact. Opening is
    # checked as recovery against the actual next configuration below.
    if collision['detected'] and action[7] > 0:
        stop = 'collision_detected'
    else:
        for _ in range(max_steps):
            result = step_callback(action, False); steps += 1
            config.update(q=raw.sim.data.qpos.copy())
            after = _collision_pair_distances(config, policy['protected_pairs'])
            collision = _collision_distance_report(config, policy['protected_pairs'],
                distance_limit_m=policy['hard_stop_distance_m'], pair_distances=after)
            mode = 'post_step_configuration'
            escape = action[7] < 0 and verified_collision_boundary_escape(before, after,
                hard_stop_distance_m=policy['hard_stop_distance_m'])
            if collision['detected'] and not escape:
                stop = 'collision_detected'; break
            if result.get('error'):
                stop = 'control_step_failed'; break
            if result.get('terminated') or result.get('truncated'):
                stop = 'episode_terminated'; break
            before = after
        else:
            if collision['detected']:
                stop = 'collision_detected'
    collision = classify_collision({**_collision_receipt(policy), **collision,
        'check_mode': mode, 'prediction_checked': False,
        'checked_during': 'gripper_actuation'}, policy)
    return {**result, 'steps_executed': steps, 'stop_reason': stop,
            'collision': collision, 'gripper_horizon_completed': stop == 'gripper_horizon_completed'}
