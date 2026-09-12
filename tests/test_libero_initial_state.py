import json
import numpy as np
import pytest

from sim.env_registry import _LibEnvWrapper


class Raw:
    def __init__(self):
        self.actions = []
        self.state = None
        self.success = False
    def seed(self, seed):
        pass
    def reset(self):
        return {'phase': 'random_reset'}
    def set_init_state(self, state):
        self.state = state.copy()
        return {'phase': 'official_state'}
    def step(self, action):
        self.actions.append(action.copy())
        return {'phase': 'settled'}, 0, False, {}
    def check_success(self):
        return self.success
    def get_sim_state(self):
        return self.state.copy()


def test_official_reset_reuses_state_and_controller_dimension_without_public_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENETA_PRIVATE_STATE_DIR', str(tmp_path))
    raw = Raw()
    env = _LibEnvWrapper(raw, controller='JOINT_VELOCITY', seed=0)
    env._official_initial_state = np.arange(9, dtype=float)
    env._official_initial_state_metadata = {'initial_state_index': 0}
    first = env.reset(seed=0)
    second = env.reset(seed=0)
    assert first == second == ({'phase': 'settled'}, {})
    assert len(raw.actions) == 10
    assert all(a.shape == (8,) and not a.any() for a in raw.actions)
    rows = [json.loads(x) for x in (tmp_path/'initialization.jsonl').read_text().splitlines()]
    assert rows[0] == rows[1] and rows[0]['reset_seed'] == 0
    raw.state[0] = -99
    assert env._official_initial_state[0] == 0


def test_unselected_initial_state_preserves_procedural_reset():
    raw = Raw()
    assert _LibEnvWrapper(raw).reset(seed=0) == ({'phase': 'random_reset'}, {})
    assert raw.actions == [] and raw.state is None


def test_initially_solved_scene_is_rejected():
    raw = Raw()
    raw.success = True
    env = _LibEnvWrapper(raw)
    env._official_initial_state = np.arange(9, dtype=float)
    with pytest.raises(RuntimeError, match='already satisfies'):
        env.reset(seed=0)
