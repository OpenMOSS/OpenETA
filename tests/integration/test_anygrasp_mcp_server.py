from __future__ import annotations

import asyncio
import base64
import json
import os
from datetime import timedelta
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("OPENETA_RUN_ANYGRASP_INTEGRATION") != "1",
    reason="Set OPENETA_RUN_ANYGRASP_INTEGRATION=1 to run real AnyGrasp MCP integration.",
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_anygrasp_mcp_stdio_detects_official_sample(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    pytest.importorskip("PIL")
    pytest.importorskip("numpy")

    python = _required_env("OPENETA_ANYGRASP_PYTHON")
    sdk_root = _required_env("OPENETA_ANYGRASP_SDK_ROOT")
    sample_dir = Path(_required_env("OPENETA_ANYGRASP_SAMPLE_DIR"))
    checkpoint_path = _required_env("OPENETA_ANYGRASP_CHECKPOINT_PATH")
    target_mask_path = _write_binary_target_mask(sample_dir / "seg_mask.png", tmp_path)

    request = {
        "mode": "targeted",
        "rgb": _image_payload(sample_dir / "color.png"),
        "depth": _image_payload(sample_dir / "depth.png"),
        "target_mask": _image_payload(target_mask_path),
        "intrinsics": {
            "fx": 927.17,
            "fy": 927.37,
            "cx": 651.32,
            "cy": 349.62,
            "scale": 1000.0,
        },
        "collision_detection": True,
        "dense_grasp": False,
    }

    async def run_call() -> dict:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=python,
            args=[
                str(REPO_ROOT / "tools" / "anygrasp_mcp_server.py"),
                "--transport",
                "stdio",
                "--sdk-root",
                sdk_root,
                "--checkpoint-path",
                checkpoint_path,
            ],
            cwd=str(REPO_ROOT),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert "detect_grasps" in [tool.name for tool in tools.tools]
                result = await session.call_tool(
                    "detect_grasps",
                    request,
                    read_timeout_seconds=timedelta(minutes=10),
                )
        assert result.isError is False
        for item in result.content:
            if getattr(item, "type", None) == "text":
                return json.loads(item.text)
        raise AssertionError("AnyGrasp MCP detect_grasps returned no text content")

    payload = asyncio.run(run_call())

    assert payload["success"] is True
    details = payload["details"]
    assert details["candidate_count"] >= 1
    assert len(details["grasp_candidates"]) == details["candidate_count"]
    _assert_candidate_shape(details["grasp_candidates"][0])


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} is required when OPENETA_RUN_ANYGRASP_INTEGRATION=1")
    return value


def _image_payload(path: Path) -> dict[str, str]:
    return {
        "format": path.suffix.lower().lstrip(".") or "png",
        "base64": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


def _write_binary_target_mask(seg_mask_path: Path, tmp_path: Path) -> Path:
    import numpy as np
    from PIL import Image

    seg_mask = np.asarray(Image.open(seg_mask_path))
    target = ((seg_mask == 1).astype(np.uint8) * 255)
    path = tmp_path / "target_mask.png"
    Image.fromarray(target, mode="L").save(path)
    return path


def _assert_candidate_shape(candidate: dict) -> None:
    assert candidate["frame"] == "camera"
    assert candidate["camera_frame"] == "opencv"
    assert isinstance(candidate["score"], float)
    assert len(candidate["translation_xyz"]) == 3
    assert len(candidate["gripper_tip_position_xyz"]) == 3
    assert len(candidate["rotation_matrix"]) == 3
    assert all(len(row) == 3 for row in candidate["rotation_matrix"])
    assert isinstance(candidate["depth"], float)
    assert isinstance(candidate["width"], float)
    assert isinstance(candidate["height"], float)
