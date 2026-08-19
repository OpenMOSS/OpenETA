from __future__ import annotations

import pytest

import agent.tools.anygrasp_capabilities as capabilities_module
from agent.tools.anygrasp_capabilities import (
    check_anygrasp_compatibility,
    clear_anygrasp_capability_cache,
    query_anygrasp_capabilities,
)


def _capabilities(width: float) -> dict:
    return {
        "schema_version": "openeta.anygrasp_capabilities.v1",
        "backend": "anygrasp_mcp",
        "model": "anygrasp_sdk",
        "max_gripper_width_m": width,
        "gripper_height_m": 0.03,
        "depth_truncation_m": 1.0,
        "max_candidates": 20,
        "geometry_change_requires_redeployment": True,
    }


def test_compatibility_accepts_matching_deployment_width() -> None:
    report = check_anygrasp_compatibility(
        url="http://anygrasp.example/sse",
        physical_max_gripper_width_m=0.08,
        query=lambda **_kwargs: _capabilities(0.08),
    )

    assert report["available"] is True
    assert report["compatible"] is True
    assert report["reason"] == "compatible"


def test_compatibility_rejects_mismatched_deployment_width() -> None:
    report = check_anygrasp_compatibility(
        url="http://anygrasp.example/sse",
        physical_max_gripper_width_m=0.08,
        query=lambda **_kwargs: _capabilities(0.1),
    )

    assert report["available"] is False
    assert report["compatible"] is False
    assert report["reason"] == "gripper_width_mismatch"
    assert report["backend_max_gripper_width_m"] == pytest.approx(0.1)
    assert report["physical_max_gripper_width_m"] == pytest.approx(0.08)
    assert "redeploy AnyGrasp" in report["message"]


def test_compatibility_rejects_legacy_server_without_capability_metadata() -> None:
    def fail(**_kwargs):
        raise RuntimeError("get_capabilities is missing")

    report = check_anygrasp_compatibility(
        url="http://anygrasp.example/sse",
        physical_max_gripper_width_m=0.08,
        query=fail,
    )

    assert report["available"] is False
    assert report["reason"] == "capability_discovery_failed"
    assert "cannot be verified" in report["message"]


def test_query_reads_dedicated_mcp_capability_tool(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class Transport:
        def __init__(self, url: str) -> None:
            calls.append(("init", url))

        def list_tools(self, *, timeout_s=None):
            calls.append(("list_tools", timeout_s))
            return {"tools": [{"name": "detect_grasps"}, {"name": "get_capabilities"}]}

        def call_tool(self, name, arguments, *, timeout_s=None):
            calls.append((name, (arguments, timeout_s)))
            return {
                "success": True,
                "details": {"capabilities": _capabilities(0.08)},
            }

    clear_anygrasp_capability_cache()
    monkeypatch.setattr(capabilities_module, "SseSimulatorMcpTransport", Transport)

    result = query_anygrasp_capabilities(
        url="http://anygrasp.example/sse",
        timeout_s=3.0,
    )

    assert result["max_gripper_width_m"] == 0.08
    assert calls == [
        ("init", "http://anygrasp.example/sse"),
        ("list_tools", 3.0),
        ("get_capabilities", ({}, 3.0)),
    ]
