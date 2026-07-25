"""OpenETA simulation layer.

Two entry points:

1. **Quick eval** — ``gym.make("openeta/<env_type>_<task>-v0")``
   returns a :class:`~sim.unified_env.UnifiedEnv` with unified obs dict.

2. **OpenETA protocol** — :func:`~sim.adapter.make_sim_adapter`
   returns a :class:`~sim.adapter.UnifiedSimulatorAdapter` that
   implements leader-defined ``SimulatorAdapter``, producing
   ``EnvObservation`` / consuming ``EnvAction`` per the adapter protocol.
"""

import sim.env_registry  # noqa: F401 — side-effect: populates gym registry

from sim.unified_env import UnifiedEnv, make_unified
from sim.adapter import UnifiedSimulatorAdapter, make_sim_adapter

__all__ = [
    "UnifiedEnv",
    "UnifiedSimulatorAdapter",
    "make_unified",
    "make_sim_adapter",
]
