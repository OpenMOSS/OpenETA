"""Shared deterministic reset seeding for BEHAVIOR adapters."""

from __future__ import annotations

import hashlib
import random

import numpy as np


def derive_behavior_seed(
    base_seed: int,
    *,
    worker_seed_offset: int,
    row_index: int,
    reset_count: int,
    stream: int = 0,
) -> int:
    """Derive a stable distributed seed from the complete row identity."""

    components = (
        int(base_seed),
        int(worker_seed_offset),
        int(row_index),
        int(reset_count),
        int(stream),
    )
    payload = ":".join(str(value) for value in components).encode("ascii")
    digest = hashlib.blake2b(
        payload,
        digest_size=8,
        person=b"OpenETA-BEHAVIOR",
    ).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def seed_behavior_reset_rngs(seed: int) -> None:
    """Seed every RNG used by OmniGibson task and object sampling."""

    # Keep torch lazy so importing the lightweight direct-env catalog does not
    # eagerly load the simulator's heaviest Python dependency.
    import torch

    resolved = int(seed)
    random.seed(resolved)
    np.random.seed(resolved % (2**32))
    torch.manual_seed(resolved)
