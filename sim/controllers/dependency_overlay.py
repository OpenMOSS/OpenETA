"""Validation for host-owned controller dependency overlays."""

from __future__ import annotations

from pathlib import Path


MINK_FORBIDDEN_RUNTIME_SHADOWS = (
    "mujoco",
    "numpy",
    "scipy",
)


def validate_mink_dependency_overlay(path: str | Path) -> Path:
    """Reject overlays that replace LIBERO's binary/runtime compatibility set."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise RuntimeError(
            "OPENETA_LIBERO_MINK_DEPENDENCY_PATH is not a directory: "
            f"{resolved}"
        )
    shadows = [
        name
        for name in MINK_FORBIDDEN_RUNTIME_SHADOWS
        if (resolved / name).exists()
    ]
    if shadows:
        raise RuntimeError(
            "OPENETA_LIBERO_MINK_DEPENDENCY_PATH must be a minimal overlay and "
            "must not shadow the LIBERO runtime. Remove these packages from the "
            f"overlay: {', '.join(shadows)}. Keep only Mink, qpsolvers, quadprog, "
            "and their metadata; use MuJoCo/NumPy/SciPy from sim/venvs/libero."
        )
    return resolved
