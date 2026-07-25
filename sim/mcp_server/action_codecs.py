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
    return ({
        "metaworld": 0.005,
        "libero": 0.009,
        "maniskill": 0.003,
        "robocasa": 0.05,
        "dummy": 0.005,
    }.get(backend, 0.0), 0.05)


def cartesian_command_frame(meta: dict[str, Any], backend: str) -> str:
    """Return the frame consumed by a backend's Cartesian delta controller."""
    if backend == "behavior":
        return str(_declared_behavior_layout(meta).get("command_frame", "robot_base"))
    if backend == "robocasa":
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
) -> list[float]:
    """Encode one normalized Cartesian delta without guessing action slots."""
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
