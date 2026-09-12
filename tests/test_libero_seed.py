import random
import numpy as np
import pytest
from sim.env_registry import _LibEnvWrapper


class Raw:
    def seed(self, seed):
        np.random.seed(seed)
    def reset(self):
        return {'placement': [float(np.random.uniform()), random.random()]}


def test_explicit_seed_repeats_without_changing_other_environments_rng():
    env = _LibEnvWrapper(Raw(), seed=19)
    np.random.seed(4); random.seed(4)
    expected_np = np.random.RandomState(4).uniform()
    expected_py = random.Random(4).random()
    a,_=env.reset(seed=0); b,_=env.reset(seed=0)
    assert a==b
    assert np.random.uniform()==expected_np and random.random()==expected_py
    c,_=env.reset(seed=1)
    assert c!=a


def test_construction_seed_and_unseeded_reset_sequence_are_local():
    a,b=_LibEnvWrapper(Raw(),seed=42),_LibEnvWrapper(Raw(),seed=42)
    assert a.reset()==b.reset()
    second=a.reset()
    _LibEnvWrapper(Raw(),seed=71).reset(seed=99)
    assert second==b.reset()
    assert a.reset()!=second


@pytest.mark.parametrize('seed', [True,-1,2**32,0.5,'0'])
def test_invalid_seed_rejected_before_backend_reset(seed):
    with pytest.raises(ValueError):_LibEnvWrapper(Raw()).reset(seed=seed)
