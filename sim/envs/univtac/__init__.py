"""Pure-Python contracts and trace helpers for direct UniVTAC smoke tests.

This package must remain importable without Isaac Sim. The simulator task is
loaded dynamically by ``scripts/univtac/smoke_insert_hole.py`` only after
``AppLauncher`` has started.
"""

from sim.envs.univtac.contract import (
    ArtifactRef,
    IncompleteTactilePacketError,
    PrivilegedVisibilityError,
    TactileTransition,
    UniVTACContractError,
    UniVTACTaskSnapshot,
    validate_transition_binding,
)

__all__ = [
    "ArtifactRef",
    "IncompleteTactilePacketError",
    "PrivilegedVisibilityError",
    "TactileTransition",
    "UniVTACContractError",
    "UniVTACTaskSnapshot",
    "validate_transition_binding",
]
