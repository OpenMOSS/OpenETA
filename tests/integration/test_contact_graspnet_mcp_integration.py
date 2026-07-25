from __future__ import annotations

import asyncio
import base64
import json
import math
import os
from datetime import timedelta
from pathlib import Path

import pytest

from agent.tools.handlers import (
    build_contact_graspnet_handler,
    build_stdio_contact_graspnet_mcp_predictor,
)
from agent.tools.registry import ToolExecutionContext, build_default_tool_registry


pytestmark = pytest.mark.skipif(
    os.environ.get("OPENETA_RUN_CONTACT_GRASPNET_INTEGRATION") != "1",
    reason=(
        "Set OPENETA_RUN_CONTACT_GRASPNET_INTEGRATION=1 to run the real "
        "Contact-GraspNet MCP integration."
    ),
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_contact_graspnet_mcp_stdio_predicts_targeted_grasps() -> None:
    pytest.importorskip("mcp")
    python = _required_env("OPENETA_CONTACT_GRASPNET_PYTHON")
    backend_root = _required_env("OPENETA_CONTACT_GRASPNET_ROOT")
    checkpoint_dir = _required_env("OPENETA_CONTACT_GRASPNET_CHECKPOINT_DIR")
    request = {
        "depth": _png_payload(Path(_required_env("OPENETA_CONTACT_GRASPNET_SAMPLE_DEPTH"))),
        "object_mask": _png_payload(
            Path(_required_env("OPENETA_CONTACT_GRASPNET_SAMPLE_OBJECT_MASK"))
        ),
        "intrinsics": _json_object("OPENETA_CONTACT_GRASPNET_SAMPLE_INTRINSICS_JSON"),
    }

    async def run_call() -> dict:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=python,
            args=[
                str(REPO_ROOT / "tools" / "contact_graspnet_mcp_server.py"),
                "--transport",
                "stdio",
                "--contact-graspnet-root",
                backend_root,
                "--checkpoint-dir",
                checkpoint_dir,
            ],
            cwd=str(REPO_ROOT),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert "predict_grasps" in [tool.name for tool in tools.tools]
                result = await session.call_tool(
                    "predict_grasps",
                    request,
                    read_timeout_seconds=timedelta(minutes=10),
                )
        assert result.isError is False
        for item in result.content:
            if getattr(item, "type", None) == "text":
                return json.loads(item.text)
        raise AssertionError("Contact-GraspNet MCP returned no JSON text content")

    payload = asyncio.run(run_call())

    assert payload["success"] is True
    details = payload["details"]
    assert details["tool"] == "contact_graspnet"
    assert details["mode"] == "targeted"
    assert details["frame"] == "camera"
    assert details["camera_frame"] == "opencv"
    assert details["grasp_frame"] == "graspnet"
    assert 1 <= details["candidate_count"] <= 20
    assert len(details["grasp_candidates"]) == details["candidate_count"]
    max_gripper_width = details["metadata"]["max_gripper_width"]
    for index, candidate in enumerate(details["grasp_candidates"]):
        _assert_candidate(candidate, index, max_gripper_width)
    serialized = json.dumps(payload)
    assert "base64" not in serialized
    assert "point_cloud" not in serialized
    assert backend_root not in serialized
    assert checkpoint_dir not in serialized


def test_contact_graspnet_agent_handler_calls_real_stdio_mcp(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    python = _required_env("OPENETA_CONTACT_GRASPNET_PYTHON")
    backend_root = _required_env("OPENETA_CONTACT_GRASPNET_ROOT")
    checkpoint_dir = _required_env("OPENETA_CONTACT_GRASPNET_CHECKPOINT_DIR")
    rgb = _required_env("OPENETA_CONTACT_GRASPNET_SAMPLE_RGB")
    depth = _required_env("OPENETA_CONTACT_GRASPNET_SAMPLE_DEPTH")
    object_mask = _required_env("OPENETA_CONTACT_GRASPNET_SAMPLE_OBJECT_MASK")
    intrinsics = _json_object("OPENETA_CONTACT_GRASPNET_SAMPLE_INTRINSICS_JSON")
    predictor = build_stdio_contact_graspnet_mcp_predictor(
        command=python,
        args=[
            str(REPO_ROOT / "tools" / "contact_graspnet_mcp_server.py"),
            "--transport",
            "stdio",
            "--contact-graspnet-root",
            backend_root,
            "--checkpoint-dir",
            checkpoint_dir,
        ],
        cwd=REPO_ROOT,
    )
    handler = build_contact_graspnet_handler(
        predictor,
        output_root=tmp_path / "tool-result",
    )
    spec = build_default_tool_registry().get("contact_graspnet")
    context = ToolExecutionContext(
        name="contact_graspnet",
        spec=spec,
        parameters={
            "rgb": rgb,
            "depth": depth,
            "object_mask": {
                "mask_ref": object_mask,
                "source_image": rgb,
                "label": "upstream example object",
            },
            "intrinsics": intrinsics,
        },
    )

    result = handler(context)

    assert result.success is True
    assert 1 <= result.details["candidate_count"] <= 20
    assert result.details["source"]["rgb"] == rgb
    assert result.details["source"]["object_mask"] == object_mask
    assert Path(result.details["request_ref"]).is_file()
    assert Path(result.details["raw_output_ref"]).is_file()
    assert Path(result.details["tool_result_ref"]).is_file()
    for ref_name in ("request_ref", "raw_output_ref", "tool_result_ref"):
        saved = Path(result.details[ref_name]).read_text(encoding="utf-8")
        assert "base64" not in saved
        assert "point_cloud" not in saved
        assert backend_root not in saved
        assert checkpoint_dir not in saved


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} is required for Contact-GraspNet integration")
    return value


def _png_payload(path: Path) -> dict[str, str]:
    if path.suffix.lower() != ".png":
        pytest.fail(f"Contact-GraspNet fixture must be PNG: {path}")
    return {
        "format": "png",
        "base64": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


def _json_object(env_name: str) -> dict:
    value = json.loads(Path(_required_env(env_name)).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        pytest.fail(f"{env_name} must point to a JSON object")
    return value


def _assert_candidate(candidate: dict, index: int, max_gripper_width: float) -> None:
    assert candidate["id"] == f"grasp_{index:03d}"
    assert candidate["frame"] == "camera"
    assert candidate["camera_frame"] == "opencv"
    assert candidate["grasp_frame"] == "graspnet"
    assert candidate["source_model"] == "contact_graspnet"
    assert candidate["gripper_model"] == "panda"
    assert math.isfinite(candidate["score"])
    assert candidate["gripper_depth"] == pytest.approx(0.1034)
    assert 0 <= candidate["width"] <= max_gripper_width
    assert len(candidate["translation_xyz"]) == 3
    assert len(candidate["rotation_matrix"]) == 3
    assert all(len(row) == 3 for row in candidate["rotation_matrix"])
    assert len(candidate["gripper_tip_position_xyz"]) == 3
    assert len(candidate["contact_point_xyz"]) == 3
    assert "depth" not in candidate
    assert "height" not in candidate
