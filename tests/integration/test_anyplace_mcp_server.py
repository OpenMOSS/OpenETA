from __future__ import annotations

import asyncio
import base64
import json
import math
import os
from datetime import timedelta
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("OPENETA_RUN_ANYPLACE_INTEGRATION") != "1",
    reason="Set OPENETA_RUN_ANYPLACE_INTEGRATION=1 to run real AnyPlace MCP integration.",
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_anyplace_mcp_stdio_predicts_aligned_rgbd_sample() -> None:
    pytest.importorskip("mcp")

    python = _required_env("OPENETA_ANYPLACE_PYTHON")
    anyplace_root = _required_env("OPENETA_ANYPLACE_ROOT")
    config_path = _required_env("OPENETA_ANYPLACE_CONFIG_PATH")
    request = {
        "rgb": _image_payload(Path(_required_env("OPENETA_ANYPLACE_SAMPLE_RGB"))),
        "depth": _image_payload(Path(_required_env("OPENETA_ANYPLACE_SAMPLE_DEPTH"))),
        "object_mask": _image_payload(Path(_required_env("OPENETA_ANYPLACE_SAMPLE_OBJECT_MASK"))),
        "placement_region_mask": _image_payload(
            Path(_required_env("OPENETA_ANYPLACE_SAMPLE_PLACEMENT_REGION_MASK"))
        ),
        "intrinsics": _json_file("OPENETA_ANYPLACE_SAMPLE_INTRINSICS_JSON"),
        "selected_grasp": _json_file("OPENETA_ANYPLACE_SAMPLE_SELECTED_GRASP_JSON"),
    }

    async def run_call() -> dict:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=python,
            args=[
                str(REPO_ROOT / "tools" / "anyplace_mcp_server.py"),
                "--transport",
                "stdio",
                "--anyplace-root",
                anyplace_root,
                "--config-path",
                config_path,
            ],
            cwd=str(REPO_ROOT),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert "predict_placement" in [tool.name for tool in tools.tools]
                result = await session.call_tool(
                    "predict_placement",
                    request,
                    read_timeout_seconds=timedelta(minutes=10),
                )
        assert result.isError is False
        for item in result.content:
            if getattr(item, "type", None) == "text":
                return json.loads(item.text)
        raise AssertionError("AnyPlace MCP predict_placement returned no text content")

    payload = asyncio.run(run_call())

    assert payload["success"] is True
    details = payload["details"]
    assert details["frame"] == "camera"
    assert details["camera_frame"] == "opencv"
    assert details["candidate_count"] == 5
    assert len(details["placement_candidates"]) == 5
    for index, candidate in enumerate(details["placement_candidates"]):
        assert candidate["id"] == f"placement_{index:03d}"
        assert candidate["source_grasp_id"] == request["selected_grasp"]["id"]
        transform = candidate["object_placement_transform"]
        assert transform["convention"] == "p_placed = R @ p_current + t"
        _assert_transform_matrix(transform["transform_matrix"])
        place_grasp = candidate["place_grasp_pose"]
        assert place_grasp["frame"] == "camera"
        assert place_grasp["camera_frame"] == "opencv"


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} is required when OPENETA_RUN_ANYPLACE_INTEGRATION=1")
    return value


def _json_file(env_name: str) -> dict:
    value = json.loads(Path(_required_env(env_name)).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        pytest.fail(f"{env_name} must point to a JSON object")
    return value


def _image_payload(path: Path) -> dict[str, str]:
    return {
        "format": path.suffix.lstrip(".").lower() or "png",
        "base64": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


def _assert_transform_matrix(value: list[list[float]]) -> None:
    assert len(value) == 4
    assert all(len(row) == 4 for row in value)
    assert all(isinstance(item, float) for row in value for item in row)
    assert all(math.isfinite(item) for row in value for item in row)
