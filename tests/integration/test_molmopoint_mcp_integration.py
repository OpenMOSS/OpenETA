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
    os.environ.get("OPENETA_RUN_MOLMOPOINT_INTEGRATION") != "1",
    reason="Set OPENETA_RUN_MOLMOPOINT_INTEGRATION=1 to run real MolmoPoint MCP integration.",
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = "Look at the object in Image 1. In Image 2, point to the same object."


def test_molmopoint_mcp_stdio_matches_reference_object_in_scene() -> None:
    pytest.importorskip("mcp")
    python = _required_env("OPENETA_MOLMOPOINT_PYTHON")
    hf_home = _required_env("OPENETA_MOLMOPOINT_HF_HOME")
    model_id = os.environ.get("OPENETA_MOLMOPOINT_MODEL_ID", "allenai/MolmoPoint-8B")
    revision = _required_env("OPENETA_MOLMOPOINT_MODEL_REVISION")
    reference = Path(_required_env("OPENETA_MOLMOPOINT_SAMPLE_REFERENCE"))
    scene = Path(_required_env("OPENETA_MOLMOPOINT_SAMPLE_SCENE"))
    target_box = _target_box()
    request = {
        "images": [_image_payload(reference), _image_payload(scene)],
        "prompt": os.environ.get("OPENETA_MOLMOPOINT_SAMPLE_PROMPT", DEFAULT_PROMPT),
    }

    async def run_call() -> dict:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=python,
            args=[
                str(REPO_ROOT / "tools" / "molmopoint_mcp_server.py"),
                "--transport",
                "stdio",
                "--model-id",
                model_id,
                "--model-revision",
                revision,
                "--hf-home",
                hf_home,
            ],
            cwd=str(REPO_ROOT),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert "point_image" in [tool.name for tool in tools.tools]
                result = await session.call_tool(
                    "point_image",
                    request,
                    read_timeout_seconds=timedelta(minutes=10),
                )
        assert result.isError is False
        for item in result.content:
            if getattr(item, "type", None) == "text":
                return json.loads(item.text)
        raise AssertionError("MolmoPoint MCP point_image returned no JSON text content")

    payload = asyncio.run(run_call())
    assert payload["success"] is True
    details = payload["details"]
    assert details["tool"] == "molmopoint"
    assert details["model"] == model_id
    assert details["point_count"] == len(details["points"])
    assert details["coordinate_convention"] == {
        "origin": "top_left",
        "x_direction": "right",
        "y_direction": "down",
        "units": "pixels",
    }
    image_metadata = details["metadata"]["images"]
    for point_index, point in enumerate(details["points"]):
        assert point["id"] == f"point_{point_index:03d}"
        image_index = point["image_index"]
        assert 0 <= image_index < 2
        assert math.isfinite(point["pixel_x"])
        assert math.isfinite(point["pixel_y"])
        assert 0 <= point["pixel_x"] < image_metadata[image_index]["width"]
        assert 0 <= point["pixel_y"] < image_metadata[image_index]["height"]

    x0, y0, x1, y1 = target_box
    assert any(
        point["image_index"] == 1
        and x0 <= point["pixel_x"] <= x1
        and y0 <= point["pixel_y"] <= y1
        for point in details["points"]
    ), f"No scene-image point fell inside expected target box {target_box}"

    serialized = json.dumps(payload)
    assert "base64" not in serialized
    assert "raw_generation" not in serialized
    assert str(reference) not in serialized
    assert str(scene) not in serialized
    assert hf_home not in serialized


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} is required when OPENETA_RUN_MOLMOPOINT_INTEGRATION=1")
    return value


def _image_payload(path: Path) -> dict[str, str]:
    image_format = path.suffix.lstrip(".").lower()
    if image_format == "jpg":
        image_format = "jpeg"
    if image_format not in {"png", "jpeg"}:
        pytest.fail(f"MolmoPoint fixture must be PNG or JPEG: {path}")
    return {
        "format": image_format,
        "base64": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


def _target_box() -> list[float]:
    raw = _required_env("OPENETA_MOLMOPOINT_SAMPLE_TARGET_BOX_XYXY")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        pytest.fail(f"OPENETA_MOLMOPOINT_SAMPLE_TARGET_BOX_XYXY must be JSON: {exc}")
    if not isinstance(value, list) or len(value) != 4:
        pytest.fail("OPENETA_MOLMOPOINT_SAMPLE_TARGET_BOX_XYXY must contain four numbers")
    return [float(item) for item in value]
