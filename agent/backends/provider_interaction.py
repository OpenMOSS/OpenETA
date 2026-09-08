"""Validate provider-reported console timing, never planner-authored content."""
from __future__ import annotations

import math


def manual_provider_interaction(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    if (value.get("schema_version") != "manual_vlm.provider_interaction.v1"
            or value.get("mode") != "manual_console"):
        return {}
    request_id = value.get("request_id")
    wait_s = value.get("wait_s")
    if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
        return {}
    if isinstance(wait_s, bool) or not isinstance(wait_s, (float, int)):
        return {}
    try:
        wait_s = float(wait_s)
    except OverflowError:
        return {}
    if not math.isfinite(wait_s) or wait_s < 0:
        return {}
    return {
        "schema_version": "manual_vlm.provider_interaction.v1",
        "mode": "manual_console", "request_id": request_id, "wait_s": wait_s,
    }
