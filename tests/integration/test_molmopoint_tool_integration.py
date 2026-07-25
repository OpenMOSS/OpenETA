from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent.tools.handlers import (
    build_molmopoint_handler,
    build_stdio_molmopoint_mcp_pointer,
)
from agent.tools.registry import ToolExecutionContext, build_default_tool_registry


pytestmark = pytest.mark.skipif(
    os.environ.get("OPENETA_RUN_MOLMOPOINT_TOOL_INTEGRATION") != "1",
    reason="Set OPENETA_RUN_MOLMOPOINT_TOOL_INTEGRATION=1 for real tool integration.",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = "Look at the object in Image 1. In Image 2, point to the same object."


def test_real_molmopoint_agent_tool_generates_trusted_points_and_visuals(tmp_path: Path) -> None:
    python = _required_env("OPENETA_MOLMOPOINT_PYTHON")
    hf_home = _required_env("OPENETA_MOLMOPOINT_HF_HOME")
    revision = _required_env("OPENETA_MOLMOPOINT_MODEL_REVISION")
    reference = Path(_required_env("OPENETA_MOLMOPOINT_SAMPLE_REFERENCE"))
    scene = Path(_required_env("OPENETA_MOLMOPOINT_SAMPLE_SCENE"))
    target_box = _target_box()
    pointer = build_stdio_molmopoint_mcp_pointer(
        command=python,
        args=[
            str(REPO_ROOT / "tools/molmopoint_mcp_server.py"),
            "--transport",
            "stdio",
            "--model-revision",
            revision,
            "--hf-home",
            hf_home,
        ],
        cwd=REPO_ROOT,
    )
    spec = build_default_tool_registry().get("molmopoint")
    context = ToolExecutionContext(
        name="molmopoint",
        spec=spec,
        parameters={
            "images": [str(reference), str(scene)],
            "prompt": os.environ.get("OPENETA_MOLMOPOINT_SAMPLE_PROMPT", DEFAULT_PROMPT),
        },
    )
    result = build_molmopoint_handler(pointer, output_root=tmp_path / "runs")(context)
    assert result.success is True
    outputs = result.details["outputs"]
    x0, y0, x1, y1 = target_box
    assert any(
        point["image_index"] == 1
        and x0 <= point["pixel_x"] <= x1
        and y0 <= point["pixel_y"] <= y1
        for point in outputs["points"]
    )
    image_artifacts = [
        artifact for artifact in result.details["artifacts"] if artifact["kind"] == "image"
    ]
    assert len(image_artifacts) == 3
    assert all(Path(artifact["path"]).is_file() for artifact in image_artifacts)
    for json_path in (tmp_path / "runs").glob("*/*.json"):
        text = json_path.read_text()
        assert "base64" not in text
        assert "raw_generation" not in text
        assert hf_home not in text


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} is required for real MolmoPoint tool integration")
    return value


def _target_box() -> list[float]:
    value = json.loads(_required_env("OPENETA_MOLMOPOINT_SAMPLE_TARGET_BOX_XYXY"))
    if not isinstance(value, list) or len(value) != 4:
        pytest.fail("OPENETA_MOLMOPOINT_SAMPLE_TARGET_BOX_XYXY must contain four numbers")
    return [float(item) for item in value]
