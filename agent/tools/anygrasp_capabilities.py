"""Fail-closed AnyGrasp deployment compatibility discovery."""

from __future__ import annotations

import math
import threading
import time
from typing import Callable

from adapter.protocol import JsonDict
from agent.tools.sim_mcp import SseSimulatorMcpTransport


ANYGRASP_CAPABILITY_SCHEMA = "openeta.anygrasp_capabilities.v1"
ANYGRASP_COMPATIBILITY_SCHEMA = "openeta.backend_compatibility.v1"
ANYGRASP_WIDTH_TOLERANCE_M = 1e-6
_CAPABILITY_CACHE_TTL_S = 30.0
_CAPABILITY_CACHE: dict[str, tuple[float, JsonDict]] = {}
_CAPABILITY_CACHE_LOCK = threading.Lock()

AnyGraspCapabilityQuery = Callable[..., JsonDict]


def query_anygrasp_capabilities(
    *,
    url: str,
    timeout_s: float = 10.0,
) -> JsonDict:
    """Read deployment geometry from the dedicated MCP capability tool."""

    endpoint = str(url or "").strip()
    if not endpoint:
        raise ValueError("AnyGrasp MCP URL is required")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("AnyGrasp capability timeout must be positive")

    now = time.monotonic()
    with _CAPABILITY_CACHE_LOCK:
        cached = _CAPABILITY_CACHE.get(endpoint)
        if cached is not None and now - cached[0] <= _CAPABILITY_CACHE_TTL_S:
            return dict(cached[1])

    transport = SseSimulatorMcpTransport(endpoint)
    catalog = transport.list_tools(timeout_s=timeout_s)
    tool_names = {
        str(item.get("name") or "")
        for item in catalog.get("tools", [])
        if isinstance(item, dict)
    }
    if "get_capabilities" not in tool_names:
        raise RuntimeError(
            "AnyGrasp MCP does not expose get_capabilities; redeploy the service "
            "with explicit deployment geometry before using it"
        )
    payload = transport.call_tool("get_capabilities", {}, timeout_s=timeout_s)
    if payload.get("success") is not True:
        details = payload.get("details")
        reason = details.get("reason") if isinstance(details, dict) else ""
        raise RuntimeError(
            "AnyGrasp capability query failed"
            + (f": {reason}" if reason else "")
        )
    details = payload.get("details")
    capabilities = details.get("capabilities") if isinstance(details, dict) else None
    if not isinstance(capabilities, dict):
        raise RuntimeError("AnyGrasp capability response has no capabilities object")
    _validate_capabilities(capabilities)
    result = dict(capabilities)
    with _CAPABILITY_CACHE_LOCK:
        _CAPABILITY_CACHE[endpoint] = (time.monotonic(), result)
    return dict(result)


def check_anygrasp_compatibility(
    *,
    url: str,
    physical_max_gripper_width_m: float,
    timeout_s: float = 10.0,
    query: AnyGraspCapabilityQuery = query_anygrasp_capabilities,
) -> JsonDict:
    """Return a planner-safe availability verdict for one deployment."""

    endpoint = str(url or "").strip()
    expected = float(physical_max_gripper_width_m)
    base: JsonDict = {
        "schema_version": ANYGRASP_COMPATIBILITY_SCHEMA,
        "backend": "anygrasp",
        "configured": bool(endpoint),
        "url": endpoint,
        "available": False,
        "compatible": False,
        "physical_max_gripper_width_m": expected,
    }
    if not endpoint:
        return {**base, "reason": "backend_not_configured"}
    if not math.isfinite(expected) or expected <= 0:
        return {
            **base,
            "reason": "invalid_physical_gripper_width",
            "message": "AnyGrasp is unavailable: physical gripper width is invalid.",
        }
    try:
        capabilities = query(url=endpoint, timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001 - preflight returns structured failure.
        return {
            **base,
            "reason": "capability_discovery_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "message": (
                "AnyGrasp is unavailable because its deployment geometry cannot "
                "be verified. Redeploy the AnyGrasp MCP service with the physical "
                f"max gripper width {expected:.6f} m and get_capabilities support."
            ),
        }

    actual = float(capabilities["max_gripper_width_m"])
    report = {
        **base,
        "capabilities": capabilities,
        "backend_max_gripper_width_m": actual,
    }
    if abs(actual - expected) > ANYGRASP_WIDTH_TOLERANCE_M:
        return {
            **report,
            "reason": "gripper_width_mismatch",
            "message": (
                "AnyGrasp is unavailable: deployment max_gripper_width_m "
                f"({actual:.6f} m) does not match the execution gate/calibration "
                f"width ({expected:.6f} m). The detector geometry is deployment-"
                "bound; redeploy AnyGrasp with the matching width."
            ),
        }
    return {
        **report,
        "available": True,
        "compatible": True,
        "reason": "compatible",
        "message": (
            "AnyGrasp deployment geometry matches the execution gate/calibration "
            f"width ({expected:.6f} m)."
        ),
    }


def clear_anygrasp_capability_cache() -> None:
    """Clear the short-lived cache used by tests and explicit redeploy flows."""

    with _CAPABILITY_CACHE_LOCK:
        _CAPABILITY_CACHE.clear()


def _validate_capabilities(capabilities: JsonDict) -> None:
    if capabilities.get("schema_version") != ANYGRASP_CAPABILITY_SCHEMA:
        raise RuntimeError("AnyGrasp capability schema is missing or unsupported")
    try:
        width = float(capabilities.get("max_gripper_width_m"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("AnyGrasp capability max_gripper_width_m is invalid") from exc
    if not math.isfinite(width) or width <= 0:
        raise RuntimeError("AnyGrasp capability max_gripper_width_m is invalid")
