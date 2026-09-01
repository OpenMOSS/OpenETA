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


def require_controller_capability(
    meta: dict[str, Any],
    backend: str,
    *,
    orientation_requested: bool,
) -> dict[str, Any]:
    """Return the declared controller block or fail closed for LIBERO.

    Other backends retain their existing codec contracts.  LIBERO is made
    strict first because its outer closed-loop executor assumes OSC_POSE; a
    future JOINT_VELOCITY environment must never be driven as if it were OSC.
    """

    spec = meta.get("control_spec")
    controller = spec.get("controller") if isinstance(spec, dict) else None
    cartesian = spec.get("cartesian_delta") if isinstance(spec, dict) else None
    if backend != "libero":
        return dict(controller) if isinstance(controller, dict) else {}
    if not isinstance(controller, dict) or not isinstance(cartesian, dict):
        raise ControlCodecError(
            "controller_capability_missing",
            backend,
            "LIBERO worker did not declare openeta.sim_control.v1 controller and "
            "cartesian_delta capabilities. Restart/redeploy the matching worker; "
            "move_to will not guess an OSC_POSE action layout.",
        )
    controller_id = str(controller.get("controller_id") or "")
    command_interface = str(controller.get("command_interface") or "")
    executor = str(controller.get("goal_executor") or "")
    osc_contract = (
        controller_id == "robosuite.osc_pose"
        and command_interface == "normalized_cartesian_delta_pose"
        and executor == "openeta.outer_closed_loop_cartesian.v1"
        and cartesian.get("supported") is True
    )
    mink_contract = (
        controller_id == "mink.robosuite_joint_velocity"
        and command_interface == "joint_velocity"
        and executor == "openeta.worker_mink_goal.v1"
        and cartesian.get("supported") is False
    )
    if not osc_contract and not mink_contract:
        raise ControlCodecError(
            "controller_capability_mismatch",
            backend,
            "LIBERO move_to supports only a declared robosuite.osc_pose outer "
            "executor or mink.robosuite_joint_velocity worker-local executor. "
            "The environment declared "
            f"controller_id={controller_id or '<missing>'}, "
            f"command_interface={command_interface or '<missing>'}, "
            f"goal_executor={executor or '<missing>'}. Use a matching worker "
            "deployment; no OSC fallback was attempted.",
        )
    if controller.get("supports_position") is not True or (
        orientation_requested and controller.get("supports_orientation") is not True
    ):
        raise ControlCodecError(
            "controller_goal_unsupported",
            backend,
            "The declared LIBERO controller does not support the requested "
            f"{'full-pose' if orientation_requested else 'position'} goal.",
        )
    return dict(controller)


def trunk_layout(meta: dict[str, Any]) -> dict[str, Any] | None:
    """Return the declared trunk layout, or None when the robot has no trunk.

    Absent is a legitimate answer -- fixed-base and trunk-less robots exist --
    so this returns None rather than raising, unlike the Cartesian layout whose
    absence means a caller asked for motion that cannot be encoded.
    """
    spec = meta.get("control_spec")
    trunk = spec.get("trunk") if isinstance(spec, dict) else None
    if not isinstance(trunk, dict) or not trunk.get("supported"):
        return None
    return trunk


def trunk_hold_values(meta: dict[str, Any], joint_positions: list[float],
                      joint_names: list[str]) -> tuple[list[int], list[float]]:
    """Normalized trunk commands that keep the trunk where it currently is.

    Returns ``(slots, values)`` to write into an action, or ``([], [])`` when the
    trunk cannot be resolved -- caller then leaves the slots untouched.

    A trunk slot left at 0.0 is not neutral.  ``JointController`` runs in
    position mode with ``use_delta_commands=False``, and
    ``Controller._preprocess_command`` scales the [-1,1] input onto the joint
    limits, so 0.0 resolves to ``(lower+upper)/2``.  For R1Pro's torso_joint1
    (limits -1.1345..1.8326) that is 0.349 rad, not zero: every action built as
    ``[0.0] * dim`` silently commands the trunk to a mid-range pose.  Holding
    position means re-normalising the *current* angle through the inverse of
    that scaling.

    Joints are located **by name**, never by slicing.  R1Pro reports 28 joints
    with the two arms interleaved, so a positional guess at where the torso sits
    picks up base DOF instead -- the same failure mode the collision mapping
    exists to prevent, and just as silent here.
    """
    trunk = trunk_layout(meta)
    if not trunk:
        return [], []
    slots = [int(i) for i in (trunk.get("indices") or [])]
    names = [str(n) for n in (trunk.get("joint_names") or [])]
    lower = [float(v) for v in (trunk.get("limits_lower") or [])]
    upper = [float(v) for v in (trunk.get("limits_upper") or [])]
    # Without names+limits the current angle cannot be re-normalised, so there
    # is no honest hold value.  Report nothing rather than a plausible guess.
    if not (slots and names) or not (len(slots) == len(names) == len(lower) == len(upper)):
        return [], []
    if not joint_names or len(joint_names) != len(joint_positions):
        return [], []

    index_of = {str(n): i for i, n in enumerate(joint_names)}
    values: list[float] = []
    for k, name in enumerate(names):
        i = index_of.get(name)
        if i is None:
            return [], []  # a trunk joint the observation does not report
        span = (upper[k] - lower[k]) / 2.0
        mid = (upper[k] + lower[k]) / 2.0
        if not (span > 1e-9):
            return [], []  # zero-width or inverted limit: not invertible
        # Inverse of Controller._preprocess_command's input->output scaling.
        v = (float(joint_positions[i]) - mid) / span
        values.append(max(-1.0, min(1.0, v)))
    return slots, values


def cartesian_scales(meta: dict[str, Any], backend: str) -> tuple[float, float]:
    """Return metres/radians represented by a normalized action of 1.0."""
    if backend == "behavior":
        layout = _declared_behavior_layout(meta)
        return (
            float(layout.get("position_scale_m", 0.05)),
            float(layout.get("rotation_scale_rad", 0.25)),
        )
    # These are controller command scales, not the empirically observed EEF
    # displacement after one physics step. LIBERO uses robosuite OSC_POSE;
    # its shipped controller config maps normalized XYZ to +/-0.05 m and
    # axis-angle rotation to +/-0.5 rad. The old 0.009/0.05 values were motion
    # hints accidentally reused as codec scales, overdriving the closed-loop
    # controller (especially orientation by 10x).
    return {
        "metaworld": (0.005, 0.05),
        "libero": (0.05, 0.5),
        "maniskill": (0.003, 0.05),
        "robocasa": (0.05, 0.05),
        "dummy": (0.005, 0.05),
    }.get(backend, (0.0, 0.0))


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
