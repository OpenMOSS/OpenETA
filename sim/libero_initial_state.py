"""Operator-selected official LIBERO initial states, never an Agent tool."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def configure_initial_state(env, benchmark, task_index):
    value = os.environ.get('OPENETA_LIBERO_INIT_STATE_INDEX')
    if value is None:
        return
    index = int(value)
    if index < 0:
        raise ValueError('Official LIBERO initial-state index must be nonnegative')
    from libero.libero import get_libero_path
    import torch
    task = benchmark.get_task(task_index)
    path = Path(get_libero_path('init_states')) / task.problem_folder / task.init_states_file
    # Trusted, pinned official benchmark files contain NumPy arrays, not just tensors.
    states = torch.load(path, weights_only=False)
    if index >= len(states):
        raise ValueError('Official LIBERO initial-state index is out of range')
    state = np.asarray(states[index], dtype=np.float64).copy()
    if state.ndim != 1 or not state.size or not np.isfinite(state).all():
        raise ValueError('Invalid official LIBERO state vector')
    env._official_initial_state = state
    env._official_initial_state_metadata = {
        'suite': task.problem_folder, 'task': task.name, 'task_index': task_index,
        'initial_state_index': index, 'initial_states_file': str(path),
        'initial_states_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'selected_state_sha256': hashlib.sha256(state.tobytes()).hexdigest(),
        'settle_steps': 5,
    }


def apply_initial_state(env, observation):
    state = getattr(env, '_official_initial_state', None)
    if state is None:
        return observation
    raw = env._env
    observation = raw.set_init_state(state.copy())
    # Match the official evaluator's five zero-action settling steps, using
    # this harness's actual controller dimension (8 for joint velocity).
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(5):
        observation, _, _, _ = raw.step(action)
    if raw.check_success():
        raise RuntimeError('Official initial state already satisfies task success')
    directory = os.environ.get('OPENETA_PRIVATE_STATE_DIR')
    if directory:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        receipt = dict(env._official_initial_state_metadata,
                       reset_seed=env._reset_rng.seed,
                       action_dimension=int(action.size),
                       actual_initial_state_sha256=hashlib.sha256(
                           np.asarray(raw.get_sim_state(), dtype=np.float64).tobytes()).hexdigest())
        with (root / 'initialization.jsonl').open('a') as handle:
            handle.write(json.dumps(receipt) + '\n')
    return observation
