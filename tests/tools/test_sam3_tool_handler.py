from __future__ import annotations

import base64
from copy import deepcopy
import io
import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("gymnasium")

from adapter.protocol import CameraFrame, EnvObservation, RobotState
from agent.tools import handlers as handlers_module
from agent.tools.handlers import (
    DEFAULT_SAM3_IMAGE_OUTPUT_ROOT,
    DEFAULT_SAM3_RESULT_OUTPUT_ROOT,
    build_sam3_handler,
    build_sse_sam3_mcp_segmenter,
    build_stdio_sam3_mcp_segmenter,
)
from agent.tools.registry import ToolExecutionContext, build_default_tool_registry


FIXTURE_IMAGE = Path(__file__).resolve().parents[1] / "fixtures" / "sam3" / "sam_test.png"
PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42Y"
    "AAAAASUVORK5CYII="
)


def _context(
    parameters: dict,
    *,
    session_id: str = "",
    observation: EnvObservation | None = None,
) -> ToolExecutionContext:
    spec = build_default_tool_registry().get("sam3")
    return ToolExecutionContext(
        name="sam3",
        spec=spec,
        parameters=parameters,
        observation=observation,
        metadata={"session_id": session_id} if session_id else {},
    )


def _observation_with_rgb_artifact(frame_id: str, path: Path) -> EnvObservation:
    return EnvObservation(
        task="pick object",
        cameras=[CameraFrame(frame_id=frame_id, rgb=[])],
        robot=RobotState(),
        metadata={
            "image_artifacts": [
                {"kind": "rgb", "frame_id": frame_id, "path": str(path)},
                {"kind": "depth", "frame_id": frame_id, "path": "other-depth.png"},
            ]
        },
    )


def _point_png(size: tuple[int, int], box: tuple[int, int, int, int]) -> tuple[str, int]:
    image = Image.new("L", size, 0)
    left, top, right, bottom = box
    image.paste(255, box)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii"), (right - left) * (
        bottom - top
    )


def _overlay_png(size: tuple[int, int]) -> str:
    image = Image.new("RGBA", size, (20, 40, 60, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _point_response(request: dict, *, image_size: tuple[int, int]) -> dict:
    boxes = [(20, 30, 80, 100), (25, 35, 70, 90), (35, 45, 60, 75)]
    detections = []
    artifacts = []
    for rank, (score, box) in enumerate(zip((0.9, 0.7, 0.4), boxes)):
        encoded, area = _point_png(image_size, box)
        detections.append(
            {
                "label": "point_prompt",
                "score": score,
                "rank": rank,
                "backend_index": rank,
                "bbox_xyxy": list(box),
                "area_px": area,
                "mask": {"format": "png", "base64": encoded},
            }
        )
        artifacts.append(
            {
                "artifact_type": "candidate_overlay",
                "rank": rank,
                "backend_index": rank,
                "format": "png",
                "base64": _overlay_png(image_size),
            }
        )
    points = request["points"]
    foreground_count = sum(point["label"] == 1 for point in points)
    return {
        "success": True,
        "content": "SAM3 point segmentation completed.",
        "details": {
            "tool": "sam3",
            "backend": "sam3_mcp",
            "model": "sam3",
            "prompt_type": "points",
            "points": points,
            "detection_count": 3,
            "detections": detections,
            "ranking": "score_descending",
            "artifacts": artifacts,
            "metadata": {
                "prompt_type": "points",
                "coordinate_units": "pixels",
                "coordinate_origin": "top_left",
                "multimask_output": True,
                "point_count": len(points),
                "foreground_point_count": foreground_count,
                "background_point_count": len(points) - foreground_count,
                "candidate_count": 3,
                "image_size": list(image_size),
            },
        },
    }


def test_sam3_default_roots_use_repo_tmp_layout() -> None:
    assert DEFAULT_SAM3_IMAGE_OUTPUT_ROOT == Path("tmp") / "image" / "sam3"
    assert DEFAULT_SAM3_RESULT_OUTPUT_ROOT == Path("tmp") / "tool_result" / "sam3"


def test_sam3_spec_exposes_text_and_point_modes() -> None:
    spec = build_default_tool_registry().get("sam3")

    assert {"mode", "image", "prompt", "points"}.issubset(spec.parameters)
    assert "label=1" in spec.parameters["points"]


def test_stdio_sam3_builder_can_target_segment_points(monkeypatch) -> None:
    calls = []

    async def fake_call(**kwargs):
        calls.append(kwargs)
        return {"success": True}

    monkeypatch.setattr(handlers_module, "_call_stdio_mcp_tool", fake_call)
    call = build_stdio_sam3_mcp_segmenter(
        command="python",
        args=["server.py"],
        tool_name="segment_points",
    )

    assert call({"points": []}) == {"success": True}
    assert calls[0]["tool_name"] == "segment_points"
    assert calls[0]["timeout_seconds"] == 600.0


def test_sse_sam3_builder_can_target_segment_points(monkeypatch) -> None:
    calls = []

    class FakeTransport:
        def __init__(self, url: str) -> None:
            assert url == "http://sam3.example/sse"

        def call_tool(self, name, arguments, *, timeout_s=None):
            calls.append((name, arguments, timeout_s))
            return {"success": True}

    monkeypatch.setattr(handlers_module, "SseSimulatorMcpTransport", FakeTransport)
    call = build_sse_sam3_mcp_segmenter(
        url="http://sam3.example/sse",
        tool_name="segment_points",
    )
    request = {"points": []}

    assert call(request) == {"success": True}
    assert calls == [("segment_points", request, 600.0)]


def test_sam3_point_mode_routes_and_materializes_three_candidates(tmp_path: Path) -> None:
    calls: list[dict] = []
    image_size = Image.open(FIXTURE_IMAGE).size

    def point_segment(request: dict) -> dict:
        calls.append(request)
        return _point_response(request, image_size=image_size)

    handler = build_sam3_handler(
        lambda _request: pytest.fail("text MCP must not be called"),
        segment_points=point_segment,
        output_root=tmp_path / "images",
        result_output_root=tmp_path / "results",
    )
    result = handler(
        _context(
            {
                "mode": "points",
                "image": str(FIXTURE_IMAGE),
                "points": [
                    {"x": 50, "y": 60, "label": 1},
                    {"x": 100.0, "y": 80.0, "label": 0},
                ],
            }
        )
    )

    assert result.success is True
    assert calls[0]["points"] == [
        {"x": 50.0, "y": 60.0, "label": 1},
        {"x": 100.0, "y": 80.0, "label": 0},
    ]
    assert "prompt" not in calls[0]
    assert result.details["mode"] == "points"
    assert result.details["prompt_type"] == "points"
    assert result.details["prompt"] == ""
    assert result.details["points"] == calls[0]["points"]
    assert result.details["detection_count"] == 3
    assert result.details["selection_required"] is True
    assert result.details["selected_detection"] is None
    assert [item["id"] for item in result.details["detections"]] == [
        "detection_000",
        "detection_001",
        "detection_002",
    ]
    point_overlays = [
        item
        for item in result.details["artifacts"]
        if item.get("artifact_type") == "candidate_overlay"
    ]
    assert len(point_overlays) == 3
    assert len({item["artifact_ref"] for item in point_overlays}) == 3
    for item in point_overlays:
        assert Path(item["artifact_ref"]).is_file()
    mask_artifacts = [
        item
        for item in result.details["artifacts"]
        if item.get("type") == "segmentation_mask"
    ]
    assert all(item["mode"] == "points" for item in mask_artifacts)
    assert all(item["points"] == calls[0]["points"] for item in mask_artifacts)
    result_dir = Path(result.details["raw_output_ref"]).parent
    assert (result_dir / "request.json").is_file()
    assert (result_dir / "response.raw.json").is_file()
    assert (result_dir / "tool_result.json").is_file()
    for path in result_dir.glob("*.json"):
        assert '"base64":' not in path.read_text()


def test_sam3_point_audit_recursively_scrubs_base64_fields(tmp_path: Path) -> None:
    image_size = Image.open(FIXTURE_IMAGE).size

    def point_segment(request: dict) -> dict:
        response = _point_response(request, image_size=image_size)
        response["image_base64"] = "top-level-secret-payload"
        response["details"]["debug"] = {
            "base64": "nested-secret-payload",
            "rgb_base64": "nested-rgb-secret-payload",
            "maskBase64": "camel-case-secret-payload",
            "base64_payload": "prefixed-secret-payload",
            "image_base64_png": "suffixed-secret-payload",
            "base64_omitted": "untrusted-marker-payload",
        }
        return response

    result = build_sam3_handler(
        lambda _request: pytest.fail("text MCP must not be called"),
        segment_points=point_segment,
        output_root=tmp_path / "images",
        result_output_root=tmp_path / "results",
    )(
        _context(
            {
                "mode": "points",
                "image": str(FIXTURE_IMAGE),
                "points": [{"x": 50, "y": 60, "label": 1}],
            }
        )
    )

    assert result.success is True
    raw_path = Path(result.details["raw_output_ref"])
    raw_text = raw_path.read_text()
    raw = json.loads(raw_text)["response"]
    assert '"base64":' not in raw_text
    assert "top-level-secret-payload" not in raw_text
    assert "nested-secret-payload" not in raw_text
    assert "nested-rgb-secret-payload" not in raw_text
    assert "camel-case-secret-payload" not in raw_text
    assert "prefixed-secret-payload" not in raw_text
    assert "suffixed-secret-payload" not in raw_text
    assert "untrusted-marker-payload" not in raw_text
    assert raw["image_base64_omitted"] is True
    assert raw["details"]["debug"] == {
        "base64_omitted": True,
        "rgb_base64_omitted": True,
        "maskBase64_omitted": True,
        "base64_payload_omitted": True,
        "image_base64_png_omitted": True,
    }


@pytest.mark.parametrize("success", [1, "true", "false", {"value": True}])
def test_sam3_text_rejects_non_boolean_success(
    tmp_path: Path,
    success,
) -> None:
    result = build_sam3_handler(
        lambda _request: {
            "success": success,
            "details": {
                "detection_count": 1,
                "detections": [
                    {
                        "label": "black shoe",
                        "score": 0.9,
                        "bbox_xyxy": [0, 0, 1, 1],
                        "area_px": 1,
                        "mask": {"format": "png", "base64": PNG_1X1},
                    }
                ],
                "artifacts": [],
            },
        },
        output_root=tmp_path / "images",
        result_output_root=tmp_path / "results",
    )(_context({"image": str(FIXTURE_IMAGE), "prompt": "black shoe"}))

    assert result.success is False
    assert result.details["reason"] == "inconsistent_detection_outputs"
    assert result.details["detection_count"] == 0
    assert not list((tmp_path / "images").rglob("*.png"))


def test_sam3_selection_visualization_failure_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_visualization(**_kwargs):
        raise OSError("selection output unavailable")

    monkeypatch.setattr(
        handlers_module,
        "_build_sam3_selection_artifacts",
        fail_visualization,
    )
    result = build_sam3_handler(
        lambda _request: {
            "success": True,
            "details": {
                "detection_count": 1,
                "detections": [
                    {
                        "label": "black shoe",
                        "score": 0.9,
                        "bbox_xyxy": [0, 0, 1, 1],
                        "area_px": 1,
                        "mask": {"format": "png", "base64": PNG_1X1},
                    }
                ],
                "artifacts": [],
            },
        },
        output_root=tmp_path / "images",
        result_output_root=tmp_path / "results",
    )(_context({"image": str(FIXTURE_IMAGE), "prompt": "black shoe"}))

    assert result.success is False
    assert result.details["reason"] == "artifact_write_failed"
    assert result.details["detection_count"] == 0
    assert result.details["detections"] == []
    assert result.details["metadata"] == {"error_type": "OSError"}


def test_sam3_handler_fails_closed_without_image(tmp_path: Path) -> None:
    handler = build_sam3_handler(lambda request: {}, output_root=tmp_path)

    result = handler(_context({"prompt": "black shoe"}))

    assert result.success is False
    assert result.details["reason"] == "missing_image"
    assert Path(result.details["raw_output_ref"]).is_file()
    assert result.details["detection_count"] == 0
    assert result.details["detections"] == []


def test_sam3_handler_fails_closed_without_prompt() -> None:
    handler = build_sam3_handler(lambda request: {})

    result = handler(_context({"image": "front"}))

    assert result.success is False
    assert result.details["reason"] == "missing_prompt"
    assert result.details["source_image"] == "front"
    assert result.details["detection_count"] == 0


@pytest.mark.parametrize(
    ("parameters", "reason"),
    [
        ({"mode": "other", "image": str(FIXTURE_IMAGE)}, "invalid_mode"),
        (
            {
                "mode": "text",
                "image": str(FIXTURE_IMAGE),
                "prompt": "shoe",
                "points": [{"x": 1, "y": 2, "label": 1}],
            },
            "conflicting_prompt_inputs",
        ),
        (
            {
                "mode": "points",
                "image": str(FIXTURE_IMAGE),
                "prompt": "shoe",
                "points": [{"x": 1, "y": 2, "label": 1}],
            },
            "conflicting_prompt_inputs",
        ),
        ({"mode": "points", "image": str(FIXTURE_IMAGE)}, "missing_points"),
        (
            {
                "mode": "points",
                "image": str(FIXTURE_IMAGE),
                "points": [{"x": 1, "y": 2, "label": 0}],
            },
            "invalid_points",
        ),
        (
            {
                "mode": "points",
                "image": str(FIXTURE_IMAGE),
                "points": [{"x": 9999, "y": 2, "label": 1}],
            },
            "point_out_of_bounds",
        ),
        (
            {
                "mode": "points",
                "image": str(FIXTURE_IMAGE),
                "points": [{"pixel_x": 1, "pixel_y": 2, "label": 1}],
            },
            "invalid_points",
        ),
    ],
)
def test_sam3_point_preflight_failures_are_persisted(
    tmp_path: Path,
    parameters: dict,
    reason: str,
) -> None:
    result = build_sam3_handler(
        lambda _request: pytest.fail("text MCP must not be called"),
        segment_points=lambda _request: pytest.fail("point MCP must not be called"),
        output_root=tmp_path / "images",
        result_output_root=tmp_path / "results",
    )(_context(parameters))

    assert result.success is False
    assert result.details["reason"] == reason
    result_dir = Path(result.details["raw_output_ref"]).parent
    assert json.loads((result_dir / "response.raw.json").read_text()) == {
        "mcp_called": False,
        "reason": reason,
    }
    assert (result_dir / "request.json").is_file()
    assert (result_dir / "tool_result.json").is_file()


def test_sam3_point_mode_fails_when_backend_is_not_bound(tmp_path: Path) -> None:
    result = build_sam3_handler(
        lambda _request: pytest.fail("text MCP must not be called"),
        output_root=tmp_path / "images",
        result_output_root=tmp_path / "results",
    )(
        _context(
            {
                "mode": "points",
                "image": str(FIXTURE_IMAGE),
                "points": [{"x": 1, "y": 2, "label": 1}],
            }
        )
    )

    assert result.success is False
    assert result.details["reason"] == "point_backend_unavailable"


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
    result = handler(
        _context(
            {"image": str(FIXTURE_IMAGE), "prompt": "black shoe"},
            session_id="sam-session",
        )
    )

    assert calls[0]["prompt"] == "black shoe"
    assert calls[0]["image_format"] == "png"
    assert base64.b64decode(calls[0]["image_base64"])
    assert result.success is True
    assert result.details["source_image"] == str(FIXTURE_IMAGE)
    assert Path(result.details["raw_output_ref"]).exists()
    assert Path(result.details["raw_output_ref"]).relative_to(tmp_path).parts[0] == (
        "sam-session"
    )
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


def test_sam3_handler_resolves_frame_id_from_current_observation(tmp_path: Path) -> None:
    calls: list[dict] = []
    handler = build_sam3_handler(
        lambda request: calls.append(request)
        or {"success": True, "details": {"detection_count": 0, "detections": []}},
        output_root=tmp_path,
    )

    result = handler(
        _context(
            {"image": "agentview", "prompt": "cube"},
            observation=_observation_with_rgb_artifact("agentview", FIXTURE_IMAGE),
        )
    )

    assert result.success is True
    assert result.details["source_image"] == str(FIXTURE_IMAGE)
    assert base64.b64decode(calls[0]["image_base64"]) == FIXTURE_IMAGE.read_bytes()


def test_sam3_handler_resolves_stale_path_to_current_observation(tmp_path: Path) -> None:
    current_image = tmp_path / "call-with-nonce" / "cameras.0.agentview.rgb.png"
    current_image.parent.mkdir()
    current_image.write_bytes(FIXTURE_IMAGE.read_bytes())
    stale_image = tmp_path / "call-without-nonce" / current_image.name
    calls: list[dict] = []
    handler = build_sam3_handler(
        lambda request: calls.append(request)
        or {"success": True, "details": {"detection_count": 0, "detections": []}},
        output_root=tmp_path,
    )

    result = handler(
        _context(
            {"image": str(stale_image), "prompt": "cube"},
            observation=_observation_with_rgb_artifact("agentview", current_image),
        )
    )

    assert result.success is True
    assert result.details["source_image"] == str(current_image)
    assert base64.b64decode(calls[0]["image_base64"]) == FIXTURE_IMAGE.read_bytes()


def test_sam3_handler_does_not_resolve_frame_outside_current_observation() -> None:
    handler = build_sam3_handler(lambda request: {})

    result = handler(
        _context(
            {"image": "wrist", "prompt": "cube"},
            observation=_observation_with_rgb_artifact("agentview", FIXTURE_IMAGE),
        )
    )

    assert result.success is False
    assert result.details["reason"] == "image_not_found"
    assert result.details["source_image"] == "wrist"


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
    raw_detections = json.loads(Path(result.details["raw_output_ref"]).read_text())[
        "response"
    ]["details"]["detections"]
    assert raw_detections[0]["mask"]["artifact_ref"] == result.details["detections"][
        1
    ]["mask_ref"]
    assert raw_detections[1]["mask"]["artifact_ref"] == result.details["detections"][
        0
    ]["mask_ref"]


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


def test_sam3_roi_preserves_full_frame_and_clamps_mask(tmp_path: Path) -> None:
    from PIL import Image

    calls: list[dict] = []
    with Image.open(FIXTURE_IMAGE) as source:
        width, height = source.size
    full_mask = Image.new("L", (width, height), color=255)
    buffer = BytesIO()
    full_mask.save(buffer, format="PNG")
    encoded_mask = base64.b64encode(buffer.getvalue()).decode("ascii")

    def segment(request: dict) -> dict:
        calls.append(request)
        return {
            "success": True,
            "details": {
                "detection_count": 1,
                "detections": [
                    {
                        "label": "object",
                        "score": 0.9,
                        "bbox_xyxy": [0, 0, width, height],
                        "mask": {"format": "png", "base64": encoded_mask},
                        "area_px": width * height,
                    }
                ],
                "artifacts": [],
            },
        }

    handler = build_sam3_handler(segment, output_root=tmp_path)
    result = handler(
        _context(
            {
                "image": str(FIXTURE_IMAGE),
                "prompt": "alphabet soup can",
                "roi_bbox_xyxy": [80, 50, 150, 120],
            },
            session_id="roi-session",
        )
    )

    assert result.success is True
    assert len(calls) == 1
    attention_bytes = base64.b64decode(calls[0]["image_base64"])
    with Image.open(BytesIO(attention_bytes)) as attention:
        assert attention.size == (width, height)
        assert attention.convert("RGB").getpixel((0, 0)) == (0, 0, 0)
        assert attention.getbbox() is not None
    effective = result.details["effective_roi_bbox_xyxy"]
    left, top, right, bottom = effective
    with Image.open(result.details["detections"][0]["mask_ref"]) as mask:
        assert mask.size == (width, height)
        assert mask.getpixel((0, 0)) == 0
        assert mask.getpixel((left, top)) == 255
        assert mask.getpixel((right - 1, bottom - 1)) == 255
        assert result.details["detections"][0]["area_px"] == (
            (right - left) * (bottom - top)
        )
    roi_artifact = next(
        artifact
        for artifact in result.details["artifacts"]
        if artifact.get("type") == "sam3_roi_attention_image"
    )
    assert Path(roi_artifact["path"]).is_file()
    assert "roi-session" in Path(roi_artifact["path"]).parts
    assert result.details["segmentation_mode"] == "roi_attention"
    assert result.details["fallback_attempted"] is False


def test_sam3_roi_retries_once_with_generic_prompt(tmp_path: Path) -> None:
    from PIL import Image

    calls: list[dict] = []
    with Image.open(FIXTURE_IMAGE) as source:
        width, height = source.size
    full_mask = Image.new("L", (width, height), color=255)
    buffer = BytesIO()
    full_mask.save(buffer, format="PNG")
    encoded_mask = base64.b64encode(buffer.getvalue()).decode("ascii")

    def segment(request: dict) -> dict:
        calls.append(request)
        detections = []
        if len(calls) == 2:
            detections = [
                {
                    "label": "foreground object",
                    "score": 0.6,
                    "bbox_xyxy": [10, 10, 100, 100],
                    "mask": {"format": "png", "base64": encoded_mask},
                    "area_px": width * height,
                }
            ]
        return {
            "success": True,
            "details": {
                "detection_count": len(detections),
                "detections": detections,
                "artifacts": [],
            },
        }

    result = build_sam3_handler(segment, output_root=tmp_path)(
        _context(
            {
                "image": str(FIXTURE_IMAGE),
                "prompt": "alphabet soup can",
                "roi_bbox_xyxy": [10, 10, 100, 100],
            }
        )
    )

    assert [call["prompt"] for call in calls] == [
        "alphabet soup can",
        "foreground object",
    ]
    assert result.success is True
    assert result.details["prompt"] == "alphabet soup can"
    assert result.details["sam_prompt_used"] == "foreground object"
    assert result.details["fallback_attempted"] is True
    assert result.details["fallback_prompt"] == "foreground object"


def test_sam3_point_prompt_routes_to_segment_points_and_preserves_candidates(
    tmp_path: Path,
) -> None:
    from PIL import Image

    text_calls: list[dict] = []
    point_calls: list[dict] = []
    with Image.open(FIXTURE_IMAGE) as source:
        width, height = source.size
    mask = Image.new("L", (width, height), color=0)
    for x in range(80, min(width, 150)):
        for y in range(50, min(height, 120)):
            mask.putpixel((x, y), 255)
    buffer = BytesIO()
    mask.save(buffer, format="PNG")
    encoded_mask = base64.b64encode(buffer.getvalue()).decode("ascii")

    def segment_points(request: dict) -> dict:
        point_calls.append(request)
        detections = [
            {
                "label": "point_prompt",
                "score": score,
                "bbox_xyxy": [80, 50, 150, 120],
                "mask": {"format": "png", "base64": encoded_mask},
                "area_px": 4900,
                "backend_index": backend_index,
            }
            for score, backend_index in ((0.9, 2), (0.6, 1), (0.3, 0))
        ]
        return {
            "success": True,
            "content": "SAM3 point segmentation completed.",
            "details": {
                "detection_count": 3,
                "detections": detections,
                "artifacts": [
                    {
                        "artifact_type": "candidate_overlay",
                        "rank": rank,
                        "format": "png",
                        "base64": PNG_1X1,
                    }
                    for rank in range(3)
                ],
                "metadata": {"prompt_type": "points"},
            },
        }

    result = build_sam3_handler(
        lambda request: text_calls.append(request) or {},
        segment_points=segment_points,
        output_root=tmp_path,
    )(
        _context(
            {
                "image": str(FIXTURE_IMAGE),
                "positive_points": [{"x": 100.0, "y": 80.0, "label": 1}],
            },
            session_id="point-session",
        )
    )

    assert result.success is True
    assert text_calls == []
    assert point_calls[0]["points"] == [{"x": 100.0, "y": 80.0, "label": 1}]
    assert "prompt" not in point_calls[0]
    assert result.details["segmentation_mode"] == "point_prompt"
    assert result.details["positive_points"] == point_calls[0]["points"]
    assert [candidate["backend_index"] for candidate in result.details["detections"]] == [
        2,
        1,
        0,
    ]
    overlays = [
        artifact["artifact_ref"]
        for artifact in result.details["artifacts"]
        if artifact.get("artifact_type") == "candidate_overlay"
    ]
    assert len(overlays) == len(set(overlays)) == 3
    assert all(Path(path).is_file() for path in overlays)


def test_sam3_point_prompt_rejects_out_of_bounds_point() -> None:
    calls: list[dict] = []
    handler = build_sam3_handler(
        lambda _request: {},
        segment_points=lambda request: calls.append(request) or {},
    )

    result = handler(
        _context(
            {
                "image": str(FIXTURE_IMAGE),
                "positive_points": [{"x": -1, "y": 5, "label": 1}],
            }
        )
    )

    assert result.success is False
    assert result.details["reason"] == "invalid_positive_points"
    assert calls == []


def test_sam3_point_prompt_fails_when_backend_tool_is_unavailable() -> None:
    result = build_sam3_handler(lambda _request: {})(
        _context(
            {
                "image": str(FIXTURE_IMAGE),
                "positive_points": [{"x": 5, "y": 5, "label": 1}],
            }
        )
    )

    assert result.success is False
    assert result.details["reason"] == "point_prompt_not_configured"


def test_sam3_roi_rejects_out_of_bounds_bbox() -> None:
    calls: list[dict] = []
    handler = build_sam3_handler(lambda request: calls.append(request) or {})

    result = handler(
        _context(
            {
                "image": str(FIXTURE_IMAGE),
                "prompt": "object",
                "roi_bbox_xyxy": [-1, 0, 10, 10],
            }
        )
    )

    assert result.success is False
    assert result.details["reason"] == "invalid_roi_bbox"
    assert calls == []


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


@pytest.mark.parametrize(
    "mutate",
    [
        lambda response: response.update(success=1),
        lambda response: response.update(success="true"),
        lambda response: response["details"].update(points=[]),
        lambda response: response["details"].update(detection_count=2),
        lambda response: response["details"]["detections"][0].update(score=float("nan")),
        lambda response: response["details"]["detections"][1].update(backend_index=0),
        lambda response: response["details"]["detections"][1].update(rank=2),
        lambda response: response["details"]["detections"][0]["mask"].update(
            base64=PNG_1X1
        ),
        lambda response: response["details"]["detections"][0].update(area_px=1),
        lambda response: response["details"]["detections"][0].update(
            bbox_xyxy=[0, 0, 1, 1]
        ),
        lambda response: response["details"].update(artifacts=[]),
        lambda response: response["details"]["metadata"].update(
            coordinate_units="normalized"
        ),
    ],
)
def test_sam3_point_invalid_success_is_rejected_atomically(
    tmp_path: Path,
    mutate,
) -> None:
    image_size = Image.open(FIXTURE_IMAGE).size

    def point_segment(request: dict) -> dict:
        response = deepcopy(_point_response(request, image_size=image_size))
        mutate(response)
        return response

    result = build_sam3_handler(
        lambda _request: pytest.fail("text MCP must not be called"),
        segment_points=point_segment,
        output_root=tmp_path / "images",
        result_output_root=tmp_path / "results",
    )(
        _context(
            {
                "mode": "points",
                "image": str(FIXTURE_IMAGE),
                "points": [{"x": 50, "y": 60, "label": 1}],
            }
        )
    )

    assert result.success is False
    assert result.details["reason"] == "inconsistent_detection_outputs"
    assert result.details["detection_count"] == 0
    assert result.details["detections"] == []
    assert not list((tmp_path / "images").rglob("*.png"))
    raw = json.loads(Path(result.details["raw_output_ref"]).read_text())
    assert raw["mcp_called"] is True
    assert raw["reason"] == "inconsistent_detection_outputs"
    assert '"base64":' not in json.dumps(raw)
