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

    async def run_call() -> dict:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=sys.executable,
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
                assert "segment" in [tool.name for tool in tools.tools]
                result = await session.call_tool(
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
        assert result.isError is False
        for item in result.content:
            if getattr(item, "type", None) == "text":
                return json.loads(item.text)
        raise AssertionError("SAM3 MCP segment returned no text content")

    payload = asyncio.run(run_call())

    assert payload["success"] is True
    details = payload["details"]
    assert details["detection_count"] >= 1
    assert details["detections"]
    assert details["detections"][0]["mask"]["format"] == "png"
    assert base64.b64decode(details["detections"][0]["mask"]["base64"])
