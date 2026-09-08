"""Interpret gripper measurements separately from command acknowledgement."""

from collections.abc import Mapping
import math


def measured_gripper_open(state: Mapping) -> bool | None:
    """Classify aperture, never attachment or a controller's command latch.

    A finite normalized aperture takes precedence over the coarse boolean.
    An explicitly malformed aperture is unknown, not permission to substitute
    a possibly contradictory boolean. Legacy boolean-only telemetry is supported.
    """
    openness = state.get("openness")
    if openness is not None:
        if (
            isinstance(openness, (int, float))
            and not isinstance(openness, bool)
            and 0.0 <= openness <= 1.0
            and math.isfinite(openness)
        ):
            return openness >= 0.8
        return None
    value = state.get("open")
    return value if isinstance(value, bool) else None


def gripper_actuation_receipt_error(receipt: object, *, position: int) -> str | None:
    """Validate a supplied modern receipt; absence is handled by the caller."""
    if not isinstance(receipt, Mapping):
        return "gripper actuation receipt must be an object"
    if receipt.get("schema_version") != "openeta.gripper_actuation_receipt.v1":
        return "unsupported gripper actuation receipt schema"
    if receipt.get("command") != ("open" if position == 1 else "close"):
        return "gripper actuation receipt command does not match the requested position"
    if receipt.get("command_latched") is not True:
        return "gripper actuation receipt does not confirm the command latch"
    steps = receipt.get("steps_executed")
    if type(steps) is not int or steps <= 0:
        return "gripper actuation receipt requires positive integer settling steps"
    return None
