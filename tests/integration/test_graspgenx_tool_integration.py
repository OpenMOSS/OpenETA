from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent.tools.handlers import (
    build_graspgenx_handler,
    build_stdio_graspgenx_mcp_gripper_lister,
    build_stdio_graspgenx_mcp_predictor,
)
from agent.tools.registry import ToolExecutionContext, build_default_tool_registry


pytestmark = pytest.mark.skipif(
    os.environ.get("OPENETA_RUN_GRASPGENX_TOOL_INTEGRATION") != "1",
    reason=(
        "Set OPENETA_RUN_GRASPGENX_TOOL_INTEGRATION=1 for the real GraspGenX "
        "agent-tool integration."
    ),
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_real_graspgenx_agent_tool_generates_candidates_and_visuals(
    tmp_path: Path,
) -> None:
    python = _required_env("OPENETA_GRASPGENX_PYTHON")
    backend_root = _required_env("OPENETA_GRASPGENX_ROOT")
    checkpoint_root = _required_env("OPENETA_GRASPGENX_CHECKPOINT_ROOT")
    gripper_root = _required_env("OPENETA_GRASPGENX_GRIPPER_DESCRIPTIONS_ROOT")
    rgb = Path(_required_env("OPENETA_GRASPGENX_SAMPLE_RGB"))
    depth = Path(_required_env("OPENETA_GRASPGENX_SAMPLE_DEPTH"))
    mask = Path(_required_env("OPENETA_GRASPGENX_SAMPLE_OBJECT_MASK"))
    gripper_name = _required_env("OPENETA_GRASPGENX_SAMPLE_GRIPPER_NAME")
    intrinsics = _json_object("OPENETA_GRASPGENX_SAMPLE_INTRINSICS_JSON")
    up_direction = _json_list("OPENETA_GRASPGENX_SAMPLE_UP_DIRECTION_JSON")
    args = [
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
    ]
    predictor = build_stdio_graspgenx_mcp_predictor(
        command=python,
        args=args,
        cwd=REPO_ROOT,
    )
    lister = build_stdio_graspgenx_mcp_gripper_lister(
        command=python,
        args=args,
        cwd=REPO_ROOT,
    )
    handler = build_graspgenx_handler(
        predictor,
        lister,
        output_root=tmp_path / "tool_result" / "graspgenx",
    )
    result = handler(
        ToolExecutionContext(
            name="graspgenx",
            spec=build_default_tool_registry().get("graspgenx"),
            parameters={
                "rgb": str(rgb),
                "depth": str(depth),
                "object_mask": {
                    "type": "segmentation_mask",
                    "mask_ref": str(mask),
                    "source_image": str(rgb),
                },
                "intrinsics": intrinsics,
                "gripper_name": gripper_name,
                "up_direction_camera": up_direction,
            },
        )
    )

    assert result.success is True, result.details
    assert 1 <= result.details["candidate_count"] <= 20
    assert result.details["source"]["source_tool"] == "graspgenx"
    assert result.details["source"]["gripper_name"] == gripper_name
    scores = [item["score"] for item in result.details["grasp_candidates"]]
    assert scores == sorted(scores, reverse=True)
    assert all(
        "model_native_grasp_pose" not in item
        for item in result.details["grasp_candidates"]
    )
    images = [
        item for item in result.details["artifacts"] if item.get("kind") == "image"
    ]
    assert {item["selection"] for item in images} == {"top_1", "top_10"}
    assert all(Path(item["path"]).is_file() for item in images)

    request_ref = Path(result.details["request_ref"])
    raw_ref = Path(result.details["raw_output_ref"])
    result_ref = Path(result.details["tool_result_ref"])
    for path in (request_ref, raw_ref, result_ref):
        assert path.is_file()
        text = path.read_text()
        assert '"base64"' not in text
        assert "point_cloud" not in text
        assert backend_root not in text
        assert checkpoint_root not in text
        assert gripper_root not in text
    assert "model_native_grasp_pose" in raw_ref.read_text()
    assert "model_native_grasp_pose" not in result_ref.read_text()


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} is required for real GraspGenX tool integration")
    return value


def _json_object(name: str) -> dict[str, object]:
    value = json.loads(_required_env(name))
    if not isinstance(value, dict):
        pytest.fail(f"{name} must contain a JSON object")
    return value


def _json_list(name: str) -> list[object]:
    value = json.loads(_required_env(name))
    if not isinstance(value, list):
        pytest.fail(f"{name} must contain a JSON list")
    return value
