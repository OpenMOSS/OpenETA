"""Explicit simulator action codecs for MCP control tools.

Raw action layouts are simulator implementation details.  The MCP server
accepts stable world-frame motion and gripper commands, then delegates their
encoding here.  Unknown backends and undeclared layouts fail closed instead
of guessing that XYZ belongs in action slots 0..2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ControlCodecError(ValueError):
    code: str
    backend: str
    detail: str

    def __str__(self) -> str:
        return self.detail


_DEFAULT_ACTION_DIMS = {
    "metaworld": 4,
    "libero": 7,
    "maniskill": 7,
    "robocasa": 12,
    "dummy": 7,
}

# Canonical LIBERO Panda gripper references.  The controller-facing closed
# reference is 2.5 mm, while the simulated no-load mechanism can settle closer
# to 1 mm.  Operator-facing code therefore treats apertures at or below
# 3.5 mm as near-closed rather than as evidence that an object obstructed the
# fingers.
LIBERO_PANDA_OPEN_ENDPOINT_M = 0.08
LIBERO_PANDA_CLOSED_ENDPOINT_M = 0.0025
LIBERO_PANDA_ENDPOINT_TOLERANCE_M = 0.001
LIBERO_PANDA_NEAR_CLOSED_THRESHOLD_M = (
    LIBERO_PANDA_CLOSED_ENDPOINT_M + LIBERO_PANDA_ENDPOINT_TOLERANCE_M
)


def _action_dim(meta: dict[str, Any], backend: str) -> int:
    dim = int(meta.get("action_dim") or 0) or _DEFAULT_ACTION_DIMS.get(backend, 0)
    if dim <= 0:
        raise ControlCodecError(
            "unknown_action_layout",
            backend,
            f"No declared action dimension for backend {backend!r}",
        )
    return dim


def _declared_behavior_layout(meta: dict[str, Any]) -> dict[str, Any]:
    spec = meta.get("control_spec")
    cartesian = spec.get("cartesian_delta") if isinstance(spec, dict) else None
    if not isinstance(cartesian, dict) or not cartesian.get("supported"):
        raise ControlCodecError(
            "unsupported_cartesian_control",
            "behavior",
            "BEHAVIOR move_to requires an explicitly declared IK cartesian_delta layout",
        )
    return cartesian


def cartesian_scales(meta: dict[str, Any], backend: str) -> tuple[float, float]:
    """Return metres/radians represented by a normalized action of 1.0."""
    if backend == "behavior":
        layout = _declared_behavior_layout(meta)
        return (
            float(layout.get("position_scale_m", 0.05)),
            float(layout.get("rotation_scale_rad", 0.25)),
        )
    # LIBERO's direct wrapper uses robosuite's default OSC_POSE controller.
    # Its public controller config is output_max=[0.05, 0.05, 0.05,
    # 0.5, 0.5, 0.5].  These values are the physical delta represented by a
    # normalized action of 1.0.  The old 0.009/0.05 pair was an empirical
    # movement hint, not the controller contract; using it mis-scaled both
    # translational and rotational corrections, causing move_to to saturate
    # and exhaust its step budget while still moving.
    # ManiSkill's PDEEPoseController clips EE deltas to pos_upper/rot_upper
    # (0.1 m / 0.1 rad for the stock Panda config) and normalizes actions to
    # [-1, 1], so a normalized action of 1.0 is a 0.1 m / 0.1 rad delta.
    # MetaWorld SawyerXYZEnv moves the mocap target by action * action_scale
    # (0.01 m per normalized unit); the hand then tracks the mocap.
    return ({
        "metaworld": 0.01,
        "libero": 0.05,
        "maniskill": 0.1,
        "robocasa": 0.05,
        "dummy": 0.005,
    }.get(backend, 0.0), {
        "metaworld": 0.05,
        "libero": 0.5,
        "maniskill": 0.1,
        "robocasa": 0.05,
        "dummy": 0.05,
    }.get(backend, 0.05))


def cartesian_command_frame(meta: dict[str, Any], backend: str) -> str:
    """Return the frame consumed by a backend's Cartesian delta controller."""
    if backend == "behavior":
        return str(_declared_behavior_layout(meta).get("command_frame", "robot_base"))
    if backend == "robocasa":
        return "robot_base"
    if backend == "maniskill":
        # PDEEPoseController uses frame="root_translation:root_aligned_body_rotation":
        # position deltas are expressed in the robot root (base) frame.
        return "robot_base"
    if backend in _DEFAULT_ACTION_DIMS:
        return "world"
    raise ControlCodecError(
        "unsupported_cartesian_control", backend, f"move_to is unsupported for {backend!r}"
    )


def make_cartesian_action(
    meta: dict[str, Any],
    delta_xyz: tuple[float, float, float] | list[float],
    backend: str,
    delta_rot: list[float] | None = None,
    *,
    gripper_command: float = 0.0,
) -> list[float]:
    """Encode one normalized Cartesian delta.

    ``gripper_command=0`` is an explicit hold command for the continuous
    Panda gripper action slot.  A Cartesian move must not silently open or
    close the gripper; callers that want a gripper transition use the
    dedicated gripper primitive instead.
    """
    if not isinstance(gripper_command, (int, float)) or not -1.0 <= float(
        gripper_command
    ) <= 1.0:
        raise ControlCodecError(
            "invalid_gripper_command",
            backend,
            "gripper_command must be a finite normalized value in [-1, 1]",
        )
    if backend == "behavior":
        layout = _declared_behavior_layout(meta)
    elif backend not in _DEFAULT_ACTION_DIMS:
        raise ControlCodecError(
            "unsupported_cartesian_control", backend, f"move_to is unsupported for {backend!r}"
        )

    dim = _action_dim(meta, backend)
    action = [0.0] * dim

    if backend == "behavior":
        position_indices = list(layout.get("position_indices", []))
        rotation_indices = list(layout.get("rotation_indices", []))
        if len(position_indices) != 3 or any(int(index) >= dim for index in position_indices):
            raise ControlCodecError(
                "invalid_control_layout", backend, "BEHAVIOR position_indices must contain 3 valid slots"
            )
        for index, value in zip(position_indices, delta_xyz):
            action[int(index)] = float(value)
        if delta_rot is not None:
            if len(rotation_indices) != 3 or any(int(index) >= dim for index in rotation_indices):
                raise ControlCodecError(
                    "invalid_control_layout", backend, "BEHAVIOR rotation_indices must contain 3 valid slots"
                )
            for index, value in zip(rotation_indices, delta_rot):
                action[int(index)] = float(value)
        return action

    if dim < 3:
        raise ControlCodecError("invalid_control_layout", backend, "Cartesian action needs at least 3 slots")
    action[:3] = [float(value) for value in delta_xyz[:3]]
    if delta_rot is not None:
        if backend == "metaworld":
            raise ControlCodecError(
                "unsupported_orientation_control", backend, "MetaWorld has no orientation action slots"
            )
        if dim < 6:
            raise ControlCodecError("invalid_control_layout", backend, "Orientation action needs 6 slots")
        action[3:6] = [float(value) for value in delta_rot[:3]]
    if backend == "robocasa" and dim == 12:
        action[11] = -1.0
    elif dim >= 7:
        action[-1] = float(gripper_command)
    elif backend in {"libero", "maniskill"}:
        raise ControlCodecError(
            "invalid_control_layout",
            backend,
            "Cartesian gripper hold requires an action layout with a gripper slot",
        )
    return action


def make_gripper_action(meta: dict[str, Any], *, open_gripper: bool, backend: str) -> list[float]:
    """Encode one gripper command using a declared or known backend layout."""
    if backend == "behavior":
        spec = meta.get("control_spec")
        gripper = spec.get("gripper") if isinstance(spec, dict) else None
        if not isinstance(gripper, dict) or not gripper.get("supported"):
            raise ControlCodecError(
                "unsupported_gripper_control", backend, "BEHAVIOR gripper layout was not declared"
            )
    elif backend not in _DEFAULT_ACTION_DIMS:
        raise ControlCodecError(
            "unsupported_gripper_control", backend, f"gripper control is unsupported for {backend!r}"
        )

    dim = _action_dim(meta, backend)
    action = [0.0] * dim
    if backend == "behavior":
        indices = [int(index) for index in gripper.get("indices", [])]
        if not indices or any(index >= dim for index in indices):
            raise ControlCodecError(
                "invalid_control_layout", backend, "BEHAVIOR gripper indices are invalid"
            )
        value = float(gripper.get("open_value" if open_gripper else "close_value"))
        for index in indices:
            action[index] = value
        return action
    if backend == "robocasa":
        if dim < 7:
            raise ControlCodecError("invalid_control_layout", backend, "RoboCasa gripper requires slot 6")
        action[6] = -1.0 if open_gripper else 1.0
        if dim == 12:
            action[11] = -1.0
    else:
        action[-1] = -1.0 if open_gripper else 1.0
    return action


def codec_error_result(error: ControlCodecError) -> dict[str, Any]:
    """Convert a codec failure to the MCP server's explicit error envelope."""
    return {
        "ok": False,
        "error": error.detail,
        "code": error.code,
        "backend": error.backend,
    }
