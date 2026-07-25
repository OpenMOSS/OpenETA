"""Deterministic embodiment-calibration selection from environment fingerprints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


DEFAULT_GRASP_CALIBRATION_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "calibrations"
    / "candidate"
    / "graspnet-eef-panda-p8.json"
)

_CALIBRATION_RULES = (
    {
        "calibration_id": "graspnet-eef-panda-p8",
        "profile_path": DEFAULT_GRASP_CALIBRATION_PROFILE,
        "environment_contains": ("libero",),
        "robot_models": ("panda", "franka panda"),
        "gripper_models": ("pandagripper", "panda"),
        "grasp_frames": ("graspnet",),
    },
)


def resolve_grasp_calibration_profile(
    *,
    environment_id: str = "",
    fingerprint: Mapping[str, Any] | None = None,
) -> Path | None:
    """Return the unique calibration matching host-owned embodiment metadata.

    An empty fingerprint preserves the current single-profile local default.
    Once an environment or robot identity is available, unknown combinations
    fail closed instead of asking the Planner to guess a calibration.
    """

    identity = {str(key): value for key, value in dict(fingerprint or {}).items()}
    environment = _normalized(
        identity.get("environment")
        or identity.get("environment_backend")
        or identity.get("env_id")
        or environment_id
    )
    robot_model = _normalized(identity.get("robot_model"))
    gripper_model = _normalized(identity.get("gripper_model"))
    grasp_frame = _normalized(identity.get("grasp_frame"))
    if not any((environment, robot_model, gripper_model, grasp_frame)):
        return DEFAULT_GRASP_CALIBRATION_PROFILE
    if environment.startswith("openeta/test") and not any(
        (robot_model, gripper_model, grasp_frame)
    ):
        return DEFAULT_GRASP_CALIBRATION_PROFILE

    matches: list[Path] = []
    for rule in _CALIBRATION_RULES:
        if environment and not any(
            token in environment for token in rule["environment_contains"]
        ):
            continue
        if robot_model and robot_model not in rule["robot_models"]:
            continue
        if gripper_model and gripper_model not in rule["gripper_models"]:
            continue
        if grasp_frame and grasp_frame not in rule["grasp_frames"]:
            continue
        matches.append(Path(rule["profile_path"]))
    if len(matches) == 1:
        return matches[0]
    return None


def require_grasp_calibration_profile(
    *,
    environment_id: str = "",
    fingerprint: Mapping[str, Any] | None = None,
) -> Path:
    """Resolve one profile or report an explicit host compatibility error."""

    profile = resolve_grasp_calibration_profile(
        environment_id=environment_id,
        fingerprint=fingerprint,
    )
    if profile is None:
        raise LookupError(
            "no grasp calibration matches the environment embodiment fingerprint"
        )
    return profile


def load_grasp_calibration_capabilities(
    profile_path: str | Path,
) -> dict[str, Any]:
    """Load physical gripper capabilities from one host-selected profile."""

    resolved = Path(profile_path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("grasp calibration profile must contain one JSON object")
    try:
        max_width = float(payload.get("max_gripper_width_m"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "grasp calibration max_gripper_width_m must be a finite number"
        ) from exc
    if not math.isfinite(max_width) or not 0.0 < max_width <= 0.2:
        raise ValueError("grasp calibration max_gripper_width_m must be in (0, 0.2]")
    return {
        "calibration_id": str(payload.get("calibration_id") or ""),
        "profile_path": str(resolved),
        "max_gripper_width_m": max_width,
    }


def _normalized(value: object) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())
