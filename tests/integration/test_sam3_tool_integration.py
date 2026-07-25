from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent.tools.handlers import (
    build_sam3_handler,
    build_sse_sam3_mcp_segmenter,
)
from agent.tools.registry import ToolExecutionContext, build_default_tool_registry


pytestmark = pytest.mark.skipif(
    os.environ.get("OPENETA_RUN_SAM3_TOOL_INTEGRATION") != "1",
    reason="Set OPENETA_RUN_SAM3_TOOL_INTEGRATION=1 for real SAM3 tool integration.",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_IMAGE = REPO_ROOT / "tests" / "fixtures" / "sam3" / "sam_test.png"


def test_real_sam3_point_tool_materializes_three_selectable_candidates(
    tmp_path: Path,
) -> None:
    url = os.environ.get("OPENETA_SAM3_URL", "http://127.0.0.1:8773/sse")
    text_segmenter = build_sse_sam3_mcp_segmenter(url=url, tool_name="segment")
    point_segmenter = build_sse_sam3_mcp_segmenter(
        url=url,
        tool_name="segment_points",
    )
    handler = build_sam3_handler(
        text_segmenter,
        segment_points=point_segmenter,
        output_root=tmp_path / "image" / "sam3",
        result_output_root=tmp_path / "tool_result" / "sam3",
    )
    spec = build_default_tool_registry().get("sam3")
    result = handler(
        ToolExecutionContext(
            name="sam3",
            spec=spec,
            parameters={
                "mode": "points",
                "image": str(FIXTURE_IMAGE),
                "points": [{"x": 125.0, "y": 112.0, "label": 1}],
            },
        )
    )

    assert result.success is True
    assert result.details["mode"] == "points"
    assert result.details["prompt_type"] == "points"
    assert result.details["points"] == [{"x": 125.0, "y": 112.0, "label": 1}]
    assert result.details["detection_count"] == 3
    assert result.details["selection_required"] is True
    assert result.details["selected_detection"] is None
    assert [item["rank"] for item in result.details["detections"]] == [0, 1, 2]
    assert {item["backend_index"] for item in result.details["detections"]} == {
        0,
        1,
        2,
    }
    assert all(Path(item["mask_ref"]).is_file() for item in result.details["detections"])

    bundle = result.details["selection_bundle"]
    assert bundle["candidate_count"] == 3
    assert Path(bundle["contact_sheet_ref"]).is_file()
    assert all(Path(item["overlay_ref"]).is_file() for item in result.details["detections"])

    result_dir = Path(result.details["raw_output_ref"]).parent
    for name in ("request.json", "response.raw.json", "tool_result.json"):
        path = result_dir / name
        assert path.is_file()
        payload = json.loads(path.read_text())
        assert '"base64":' not in json.dumps(payload)
