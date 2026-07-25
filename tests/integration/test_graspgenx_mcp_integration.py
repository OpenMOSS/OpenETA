from __future__ import annotations

import asyncio
import base64
import json
import math
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("OPENETA_RUN_GRASPGENX_INTEGRATION") != "1",
    reason=(
        "Set OPENETA_RUN_GRASPGENX_INTEGRATION=1 to run the real GraspGenX "
        "MCP integration."
    ),
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_graspgenx_mcp_stdio_predicts_targeted_grasps() -> None:
    pytest.importorskip("mcp")
    python = _required_env("OPENETA_GRASPGENX_PYTHON")
    backend_root = _required_env("OPENETA_GRASPGENX_ROOT")
    checkpoint_root = _required_env("OPENETA_GRASPGENX_CHECKPOINT_ROOT")
    gripper_root = _required_env("OPENETA_GRASPGENX_GRIPPER_DESCRIPTIONS_ROOT")
    gripper_name = _required_env("OPENETA_GRASPGENX_SAMPLE_GRIPPER_NAME")
    request = {
        "depth": _image_payload(
            Path(_required_env("OPENETA_GRASPGENX_SAMPLE_DEPTH"))
        ),
        "object_mask": _image_payload(
            Path(_required_env("OPENETA_GRASPGENX_SAMPLE_OBJECT_MASK"))
        ),
        "intrinsics": _json_object("OPENETA_GRASPGENX_SAMPLE_INTRINSICS_JSON"),
        "gripper_name": gripper_name,
        "up_direction_camera": _json_list(
            "OPENETA_GRASPGENX_SAMPLE_UP_DIRECTION_JSON"
        ),
    }

    async def run_call() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=python,
            args=[
                str(REPO_ROOT / "tools" / "graspgenx_mcp_server.py"),
                "--transport",
                "stdio",
                "--graspgenx-root",
                backend_root,
                "--checkpoint-root",
                checkpoint_root,
                "--gripper-descriptions-root",
                gripper_root,
                "--device",
                "cuda:0",
            ],
            cwd=str(REPO_ROOT),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                by_name = {tool.name: tool for tool in tools.tools}
                assert set(by_name) == {"list_grippers", "predict_grasps"}
                enum = by_name["predict_grasps"].inputSchema["properties"][
                    "gripper_name"
                ]["enum"]
                assert gripper_name in enum
                before = _json_text(
                    await session.call_tool(
                        "list_grippers",
                        {},
                        read_timeout_seconds=timedelta(minutes=2),
                    )
                )
                prediction = _json_text(
                    await session.call_tool(
                        "predict_grasps",
                        request,
                        read_timeout_seconds=timedelta(minutes=10),
                    )
                )
                after = _json_text(
                    await session.call_tool(
                        "list_grippers",
                        {},
                        read_timeout_seconds=timedelta(minutes=2),
                    )
                )
        return before, prediction, after

    before, payload, after = asyncio.run(run_call())

    assert before["success"] is True
    assert before["details"]["model_loaded"] is False
    assert before["details"]["gripper_count"] >= 1
    assert after["success"] is True
    assert after["details"]["model_loaded"] is True

    assert payload["success"] is True, payload
    details = payload["details"]
    assert details["tool"] == "predict_grasps"
    assert details["planner"] == "graspmoe"
    assert details["deterministic"] is False
    assert details["frame"] == "camera"
    assert details["camera_frame"] == "opencv"
    assert details["grasp_frame"] == "graspnet"
    assert details["gripper_name"] == gripper_name
    assert 1 <= details["candidate_count"] <= 20
    assert len(details["grasp_candidates"]) == details["candidate_count"]
    scores = [candidate["score"] for candidate in details["grasp_candidates"]]
    assert scores == sorted(scores, reverse=True)
    for rank, candidate in enumerate(details["grasp_candidates"]):
        _assert_candidate(candidate, rank, gripper_name)

    metadata = details["metadata"]
    assert metadata["model_loaded"] is True
    assert metadata["model_input_point_count"] >= 100
    assert metadata["returned_candidate_count"] == details["candidate_count"]
    assert metadata["generator_checkpoint_sha256"]
    assert metadata["discriminator_checkpoint_sha256"]
    serialized = json.dumps(payload)
    assert "base64" not in serialized
    assert "point_cloud" not in serialized
    assert backend_root not in serialized
    assert checkpoint_root not in serialized
    assert gripper_root not in serialized


def _assert_candidate(
    candidate: dict[str, Any], rank: int, gripper_name: str
) -> None:
    import numpy as np

    assert candidate["id"] == f"graspgenx_{rank:03d}"
    assert candidate["rank"] == rank
    assert candidate["gripper_name"] == gripper_name
    assert candidate["candidate_source"] in {"diffusion", "obb"}
    assert candidate["frame"] == "camera"
    assert candidate["camera_frame"] == "opencv"
    assert candidate["grasp_frame"] == "graspnet"
    assert math.isfinite(candidate["score"])
    assert candidate["depth"] > 0
    assert candidate["width"] > 0
    assert candidate["height"] > 0
    _assert_transform(np.asarray(candidate["transform_matrix"], dtype=np.float64))
    native = candidate["model_native_grasp_pose"]
    assert native["grasp_frame"] == "graspgenx"
    _assert_transform(np.asarray(native["transform_matrix"], dtype=np.float64))


def _assert_transform(transform: Any) -> None:
    import numpy as np

    assert transform.shape == (4, 4)
    assert np.isfinite(transform).all()
    np.testing.assert_allclose(transform[3], [0, 0, 0, 1], atol=1e-6)
    rotation = transform[:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-5)


def _json_text(result: Any) -> dict[str, Any]:
    assert result.isError is False
    for item in result.content:
        if getattr(item, "type", None) == "text":
            return json.loads(item.text)
    raise AssertionError("GraspGenX MCP returned no JSON text content")


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(
            f"{name} is required when OPENETA_RUN_GRASPGENX_INTEGRATION=1"
        )
    return value


def _image_payload(path: Path) -> dict[str, str]:
    return {
        "format": path.suffix.lower().lstrip(".") or "png",
        "base64": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


def _json_object(name: str) -> dict[str, Any]:
    value = json.loads(_required_env(name))
    if not isinstance(value, dict):
        pytest.fail(f"{name} must contain a JSON object")
    return value


def _json_list(name: str) -> list[Any]:
    value = json.loads(_required_env(name))
    if not isinstance(value, list):
        pytest.fail(f"{name} must contain a JSON list")
    return value
