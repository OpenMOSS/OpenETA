"""Host entry-point simulator timeout, independent of planner/provider waits."""
from __future__ import annotations

import math


DEFAULT_SIM_MCP_TIMEOUT_S = 300.0


def validate_simulator_timeout_s(value: object) -> float:
    message = "simulator timeout must be a positive finite number of seconds"
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(message)
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(message) from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(message)
    return seconds
