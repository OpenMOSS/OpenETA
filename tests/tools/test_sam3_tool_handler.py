from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

pytest.importorskip("gymnasium")

from agent.tools.handlers import (
    DEFAULT_SAM3_IMAGE_OUTPUT_ROOT,
    DEFAULT_SAM3_RESULT_OUTPUT_ROOT,
    build_sam3_handler,
)
from agent.tools.registry import ToolExecutionContext, build_default_tool_registry


FIXTURE_IMAGE = Path(__file__).resolve().parents[1] / "fixtures" / "sam3" / "sam_test.png"
PNG_1X1 = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c6360000002000100ffff03000006000557bfab0d000000"
        "0049454e44ae426082"
    )
).decode("ascii")


def _context(parameters: dict) -> ToolExecutionContext:
    spec = build_default_tool_registry().get("sam3")
    return ToolExecutionContext(name="sam3", spec=spec, parameters=parameters)


def test_sam3_default_roots_use_repo_tmp_layout() -> None:
    assert DEFAULT_SAM3_IMAGE_OUTPUT_ROOT == Path("tmp") / "image" / "sam3"
    assert DEFAULT_SAM3_RESULT_OUTPUT_ROOT == Path("tmp") / "tool_result" / "sam3"


def test_sam3_handler_fails_closed_without_image() -> None:
    handler = build_sam3_handler(lambda request: {})

    result = handler(_context({"prompt": "black shoe"}))

    assert result.success is False
    assert result.details["reason"] == "missing_image"
    assert result.details["raw_output_ref"] is None
    assert result.details["detection_count"] == 0
    assert result.details["detections"] == []


def test_sam3_handler_fails_closed_without_prompt() -> None:
    handler = build_sam3_handler(lambda request: {})

    result = handler(_context({"image": "front"}))

    assert result.success is False
    assert result.details["reason"] == "missing_prompt"
    assert result.details["source_image"] == "front"
    assert result.details["detection_count"] == 0


def test_sam3_handler_fails_closed_on_success_without_details() -> None:
    handler = build_sam3_handler(lambda request: {"success": True})

    result = handler(_context({"image": str(FIXTURE_IMAGE), "prompt": "black shoe"}))

    assert result.success is False
    assert result.details["reason"] == "inconsistent_detection_outputs"


def test_sam3_handler_fails_closed_on_success_without_detection_count() -> None:
    handler = build_sam3_handler(
        lambda request: {"success": True, "details": {"detections": []}}
    )

    result = handler(_context({"image": str(FIXTURE_IMAGE), "prompt": "black shoe"}))

    assert result.success is False
    assert result.details["reason"] == "inconsistent_detection_outputs"


def test_sam3_handler_fails_closed_on_success_without_detections() -> None:
    handler = build_sam3_handler(
        lambda request: {"success": True, "details": {"detection_count": 0}}
    )

    result = handler(_context({"image": str(FIXTURE_IMAGE), "prompt": "black shoe"}))

    assert result.success is False
    assert result.details["reason"] == "inconsistent_detection_outputs"


def test_sam3_handler_encodes_image_path_and_materializes_success(tmp_path: Path) -> None:
    calls: list[dict] = []

    def segment(request: dict) -> dict:
        calls.append(request)
        return {
            "success": True,
            "content": "SAM3 segmentation completed.",
            "details": {
                "tool": "sam3",
                "backend": "sam3_mcp",
                "model": "sam3",
                "prompt": "black shoe",
                "source_image": "server-side-value",
                "raw_output_ref": "raw.json",
                "detection_count": 1,
                "detections": [
                    {
                        "label": "black shoe",
                        "score": 0.7,
                        "bbox_xyxy": [1, 2, 3, 4],
                        "mask": {"format": "png", "base64": PNG_1X1},
                        "area_px": 42,
                    }
                ],
                "artifacts": [
                    {"artifact_type": "overlay", "format": "png", "base64": PNG_1X1}
                ],
                "metadata": {"backend_version": "sam3@test"},
            },
        }

    handler = build_sam3_handler(segment, output_root=tmp_path)
    result = handler(_context({"image": str(FIXTURE_IMAGE), "prompt": "black shoe"}))

    assert calls[0]["prompt"] == "black shoe"
    assert calls[0]["image_format"] == "png"
    assert base64.b64decode(calls[0]["image_base64"])
    assert result.success is True
    assert result.details["source_image"] == str(FIXTURE_IMAGE)
    assert Path(result.details["raw_output_ref"]).exists()
    assert result.details["detection_count"] == 1
    assert result.details["selection_required"] is False
    assert result.details["selected_detection"]["id"] == "detection_000"
    assert result.details["detections"][0]["score"] == 0.7
    assert Path(result.details["detections"][0]["mask_ref"]).exists()
    assert Path(result.details["artifacts"][0]["artifact_ref"]).exists()
    mask_artifact = next(
        artifact
        for artifact in result.details["artifacts"]
        if artifact.get("type") == "segmentation_mask"
    )
    assert mask_artifact["label"] == "black shoe"
    assert Path(mask_artifact["path"]).exists()
    assert mask_artifact["path"] == result.details["detections"][0]["mask_ref"]
    raw_payload = json.loads(Path(result.details["raw_output_ref"]).read_text())
    raw_text = json.dumps(raw_payload)
    assert PNG_1X1 not in raw_text
    assert "base64_omitted" in raw_text
    assert "base64" not in json.dumps(result.details)


def test_sam3_multiple_detections_require_explicit_selection(tmp_path: Path) -> None:
    valid_mask = base64.b64encode(FIXTURE_IMAGE.read_bytes()).decode("ascii")
    handler = build_sam3_handler(
        lambda request: {
            "success": True,
            "details": {
                "detection_count": 2,
                "detections": [
                    {
                        "label": "lower-ranked",
                        "score": 0.7,
                        "bbox_xyxy": [10, 10, 20, 20],
                        "mask": {"format": "png", "base64": valid_mask},
                        "area_px": 100,
                    },
                    {
                        "label": "higher-ranked",
                        "score": 0.8,
                        "bbox_xyxy": [30, 30, 40, 40],
                        "mask": {"format": "png", "base64": valid_mask},
                        "area_px": 90,
                    },
                ],
                "artifacts": [],
            },
        },
        output_root=tmp_path,
    )

    result = handler(_context({"image": str(FIXTURE_IMAGE), "prompt": "soup can"}))

    assert result.success is True
    assert result.details["selection_required"] is True
    assert result.details["selected_detection"] is None
    assert [item["id"] for item in result.details["detections"]] == [
        "detection_000",
        "detection_001",
    ]
    assert result.details["ranking"] == "score_descending"
    assert [item["label"] for item in result.details["detections"]] == [
        "higher-ranked",
        "lower-ranked",
    ]
    assert [item["backend_index"] for item in result.details["detections"]] == [1, 0]
    assert [item["rank"] for item in result.details["detections"]] == [0, 1]
    bundle = result.details["selection_bundle"]
    assert Path(bundle["original_image_ref"]) == FIXTURE_IMAGE
    assert Path(bundle["contact_sheet_ref"]).exists()
    assert bundle["candidate_count"] == 2
    for candidate in result.details["detections"]:
        assert Path(candidate["overlay_ref"]).exists()
        assert Path(candidate["crop_ref"]).exists()


def test_sam3_handler_can_split_json_results_and_images(tmp_path: Path) -> None:
    def segment(_request: dict) -> dict:
        return {
            "success": True,
            "content": "SAM3 segmentation completed.",
            "details": {
                "detection_count": 1,
                "detections": [
                    {
                        "label": "black shoe",
                        "score": 0.7,
                        "bbox_xyxy": [1, 2, 3, 4],
                        "mask": {"format": "png", "base64": PNG_1X1},
                        "area_px": 42,
                    }
                ],
                "artifacts": [
                    {"artifact_type": "overlay", "format": "png", "base64": PNG_1X1}
                ],
            },
        }

    image_root = tmp_path / "image" / "sam3"
    result_root = tmp_path / "tool_result" / "sam3"
    handler = build_sam3_handler(
        segment,
        output_root=image_root,
        result_output_root=result_root,
    )

    result = handler(_context({"image": str(FIXTURE_IMAGE), "prompt": "black shoe"}))

    raw_output_ref = Path(result.details["raw_output_ref"])
    mask_ref = Path(result.details["detections"][0]["mask_ref"])
    overlay_ref = Path(result.details["artifacts"][0]["artifact_ref"])
    mask_artifact = next(
        artifact
        for artifact in result.details["artifacts"]
        if artifact.get("type") == "segmentation_mask"
    )
    assert result.success is True
    assert raw_output_ref.is_relative_to(result_root)
    assert mask_ref.is_relative_to(image_root)
    assert overlay_ref.is_relative_to(image_root)
    assert Path(mask_artifact["path"]).is_relative_to(image_root)


def test_sam3_handler_fails_when_image_path_is_missing() -> None:
    handler = build_sam3_handler(lambda request: {})

    result = handler(_context({"image": "missing-image.png", "prompt": "black shoe"}))

    assert result.success is False
    assert result.details["reason"] == "image_not_found"


def test_sam3_handler_accepts_empty_detection_success() -> None:
    handler = build_sam3_handler(
        lambda request: {
            "success": True,
            "details": {
                "raw_output_ref": "raw.json",
                "detection_count": 0,
                "detections": [],
                "artifacts": [],
                "metadata": {},
            },
        }
    )

    result = handler(_context({"image": str(FIXTURE_IMAGE), "prompt": "missing thing"}))

    assert result.success is True
    assert result.details["detection_count"] == 0
    assert result.details["detections"] == []
    assert "no detections" in result.content


def test_sam3_handler_preserves_segment_failure_shape() -> None:
    handler = build_sam3_handler(
        lambda request: {
            "success": False,
            "content": "SAM3 segmentation failed: image not found.",
            "details": {
                "raw_output_ref": None,
                "reason": "image_not_found",
                "metadata": {"backend_version": "sam3@test"},
            },
        }
    )

    result = handler(_context({"image": str(FIXTURE_IMAGE), "prompt": "black shoe"}))

    assert result.success is False
    assert result.details["reason"] == "image_not_found"
    assert result.details["detection_count"] == 0
    assert result.details["detections"] == []
    assert result.details["artifacts"] == []
    assert result.details["metadata"] == {"backend_version": "sam3@test"}


def test_sam3_handler_structures_segment_exceptions() -> None:
    def segment(request: dict) -> dict:
        raise RuntimeError("server unavailable")

    handler = build_sam3_handler(segment)
    result = handler(_context({"image": str(FIXTURE_IMAGE), "prompt": "black shoe"}))

    assert result.success is False
    assert result.details["reason"] == "mcp_call_failed"
    assert result.details["metadata"]["error_type"] == "RuntimeError"


def test_sam3_handler_rejects_box_without_mask_ref() -> None:
    handler = build_sam3_handler(
        lambda request: {
            "success": True,
            "details": {
                "raw_output_ref": "raw.json",
                "detection_count": 1,
                "detections": [
                    {
                        "label": "black shoe",
                        "score": 0.7,
                        "bbox_xyxy": [1, 2, 3, 4],
                        "area_px": 42,
                    }
                ],
                "metadata": {},
            },
        }
    )

    result = handler(_context({"image": str(FIXTURE_IMAGE), "prompt": "black shoe"}))

    assert result.success is False
    assert result.details["reason"] == "inconsistent_detection_outputs"


def test_sam3_handler_rejects_detection_without_mask_base64() -> None:
    handler = build_sam3_handler(
        lambda request: {
            "success": True,
            "details": {
                "raw_output_ref": None,
                "detection_count": 1,
                "detections": [
                    {
                        "label": "black shoe",
                        "score": 0.7,
                        "bbox_xyxy": [1, 2, 3, 4],
                        "mask": {"format": "png"},
                        "area_px": 42,
                    }
                ],
                "metadata": {},
            },
        }
    )

    result = handler(_context({"image": str(FIXTURE_IMAGE), "prompt": "black shoe"}))

    assert result.success is False
    assert result.details["reason"] == "inconsistent_detection_outputs"
