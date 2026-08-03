"""Runtime controls shared by LIBERO registry and config paths."""

from __future__ import annotations

import os


LIBERO_MAX_EPISODE_STEPS_ENV = "OPENETA_LIBERO_MAX_EPISODE_STEPS"
# Interactive embodied runs routinely spend hundreds of simulator steps on
# closed-loop Cartesian convergence before a task is complete. Keep the
# benchmark override available, but make the local/default horizon large
# enough that a manual or Operator session does not terminate mid-action.
DEFAULT_LIBERO_MAX_EPISODE_STEPS = 5000


def get_libero_max_episode_steps() -> int:
    """Return the configured positive LIBERO episode horizon."""

    raw_value = os.environ.get(
        LIBERO_MAX_EPISODE_STEPS_ENV,
        str(DEFAULT_LIBERO_MAX_EPISODE_STEPS),
    )
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{LIBERO_MAX_EPISODE_STEPS_ENV} must be a positive integer; "
            f"got {raw_value!r}"
        ) from exc
    if value <= 0:
        raise ValueError(
            f"{LIBERO_MAX_EPISODE_STEPS_ENV} must be a positive integer; "
            f"got {raw_value!r}"
        )
    return value
