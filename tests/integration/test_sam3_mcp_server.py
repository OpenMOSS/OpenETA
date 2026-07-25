from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("OPENETA_RUN_SAM3_INTEGRATION") != "1",
    reason="Set OPENETA_RUN_SAM3_INTEGRATION=1 to run real SAM3 MCP integration.",
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_IMAGE = REPO_ROOT / "tests" / "fixtures" / "sam3" / "sam_test.png"


def test_sam3_mcp_stdio_segments_fixture(tmp_path: Path) -> None:
    pytest.importorskip("mcp")

    def text_payload(result) -> dict:
        assert result.isError is False
        for item in result.content:
            if getattr(item, "type", None) == "text":
                return json.loads(item.text)
        raise AssertionError("SAM3 MCP call returned no text content")

    async def run_calls() -> tuple[dict, dict]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=os.environ.get("OPENETA_SAM3_PYTHON", sys.executable),
            args=[
                str(REPO_ROOT / "tools" / "sam3_mcp_server.py"),
                "--transport",
                "stdio",
            ],
            cwd=str(REPO_ROOT),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                tool_names = [tool.name for tool in tools.tools]
                assert "segment" in tool_names
                assert "segment_points" in tool_names
                text_result = await session.call_tool(
                    "segment",
                    {
                        "image_base64": base64.b64encode(FIXTURE_IMAGE.read_bytes()).decode(
                            "ascii"
                        ),
                        "image_format": "png",
                        "prompt": "black shoe",
                    },
                    read_timeout_seconds=timedelta(minutes=10),
                )
                point_result = await session.call_tool(
                    "segment_points",
                    {
                        "image_base64": base64.b64encode(
                            FIXTURE_IMAGE.read_bytes()
                        ).decode("ascii"),
                        "image_format": "png",
                        "points": [{"x": 125.0, "y": 112.0, "label": 1}],
                    },
                    read_timeout_seconds=timedelta(minutes=10),
                )
        return text_payload(text_result), text_payload(point_result)

    payload, point_payload = asyncio.run(run_calls())

    assert payload["success"] is True
    details = payload["details"]
    assert details["detection_count"] >= 1
    assert details["detections"]
    assert details["detections"][0]["mask"]["format"] == "png"
    assert base64.b64decode(details["detections"][0]["mask"]["base64"])

    assert point_payload["success"] is True
    point_details = point_payload["details"]
    assert point_details["prompt_type"] == "points"
    assert point_details["points"] == [{"x": 125.0, "y": 112.0, "label": 1}]
    assert point_details["detection_count"] == 3
    assert len(point_details["detections"]) == 3
    assert len(point_details["artifacts"]) == 3
    assert [candidate["rank"] for candidate in point_details["detections"]] == [
        0,
        1,
        2,
    ]
    assert all(
        base64.b64decode(candidate["mask"]["base64"])
        for candidate in point_details["detections"]
    )
