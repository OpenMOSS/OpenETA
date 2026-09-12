"""Isolate LIBERO's legacy global random generators inside its worker."""
from contextlib import contextmanager
import numbers
import random
import numpy as np


def validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or not 0 <= seed < 2**32:
        raise ValueError('LIBERO seed must be an integer in [0, 2**32)')
    return int(seed)


class LiberoRandomState:
    def __init__(self, seed):
        self.reseed(seed)

    def reseed(self, seed):
        self.seed = validate_seed(seed)
        self.numpy = np.random.RandomState(self.seed).get_state()
        self.python = random.Random(self.seed).getstate()

    @contextmanager
    def scope(self):
        # Worker creation/reset are serialized by its simulator/GL locks.
        previous_numpy, previous_python = np.random.get_state(), random.getstate()
        np.random.set_state(self.numpy)
        random.setstate(self.python)
        try:
            yield
        finally:
            self.numpy, self.python = np.random.get_state(), random.getstate()
            np.random.set_state(previous_numpy)
            random.setstate(previous_python)
