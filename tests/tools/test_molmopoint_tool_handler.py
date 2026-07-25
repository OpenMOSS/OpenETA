from __future__ import annotations

import base64
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from agent.tools import handlers as handlers_module
from agent.tools.handlers import (
    DEFAULT_MOLMOPOINT_OUTPUT_ROOT,
    build_molmopoint_handler,
    build_sse_molmopoint_mcp_pointer,
    build_stdio_molmopoint_mcp_pointer,
)
from agent.tools.registry import ToolEffect, ToolExecutionContext, build_default_tool_registry


REVISION = "188130f961c8e0888a34e11121a1423c461a01ba"


def _write_image(path: Path, *, size: tuple[int, int], mode: str = "RGB") -> Path:
    color = 20 if mode == "L" else (20, 40, 60)
    Image.new(mode, size, color).save(path)
    return path


def _parameters(tmp_path: Path) -> dict[str, Any]:
    first = _write_image(tmp_path / "first.png", size=(120, 190))
    second = _write_image(tmp_path / "second.jpg", size=(320, 240))
    return {
        "images": [str(first), str(second)],
        "prompt": "  Look at Image 1. In Image 2, point to the same object.  ",
    }


def _response(request: dict[str, Any], *, points: list[dict[str, Any]] | None = None) -> dict:
    image_metadata = []
    for image_index, payload in enumerate(request["images"]):
        raw = base64.b64decode(payload["base64"])
        from io import BytesIO

        with Image.open(BytesIO(raw)) as image:
            image_metadata.append(
                {
                    "image_index": image_index,
                    "format": "jpeg" if image.format == "JPEG" else image.format.lower(),
                    "width": image.width,
                    "height": image.height,
                    "source_image_mode": image.mode,
                    "model_image_mode": "RGB",
                }
            )
    values = points if points is not None else [
        {"id": "point_000", "image_index": 0, "pixel_x": 64.4, "pixel_y": 85.6},
        {"id": "point_001", "image_index": 1, "pixel_x": 150.0, "pixel_y": 120.0},
    ]
    return {
        "success": True,
        "content": "MolmoPoint image pointing completed.",
        "details": {
            "tool": "molmopoint",
            "backend": "molmopoint_mcp",
            "model": "allenai/MolmoPoint-8B",
            "point_count": len(values),
            "points": values,
            "coordinate_convention": {
                "origin": "top_left",
                "x_direction": "right",
                "y_direction": "down",
                "units": "pixels",
            },
            "artifacts": [],
            "metadata": {
                "model_revision": REVISION,
                "images": image_metadata,
                "model_load_seconds": 5.9,
                "inference_seconds": 2.2,
                "total_seconds": 8.4,
                "unknown_backend_field": "must-not-pass",
            },
        },
    }


def _context(parameters: dict[str, Any]) -> ToolExecutionContext:
    spec = build_default_tool_registry().get("molmopoint")
    return ToolExecutionContext(name="molmopoint", spec=spec, parameters=parameters)


def test_molmopoint_spec_is_visible_without_dummy_handler() -> None:
    tools = build_default_tool_registry()
    spec = tools.get("molmopoint")
    assert spec.category == "perception"
    assert spec.effect == ToolEffect.READ_ONLY
    assert spec.batchable is False
    assert set(spec.parameters) == {"images", "prompt"}
    assert tools.can_execute("molmopoint") is False


def test_molmopoint_default_root_uses_repo_tmp_layout() -> None:
    assert DEFAULT_MOLMOPOINT_OUTPUT_ROOT == Path("tmp") / "tool_result" / "molmopoint"


def test_stdio_builder_uses_point_image_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    async def fake_call(**kwargs):
        calls.append(kwargs)
        return {"success": True}

    monkeypatch.setattr(handlers_module, "_call_stdio_mcp_tool", fake_call)
    pointer = build_stdio_molmopoint_mcp_pointer(command="python", args=["server.py"])
    assert pointer({"images": [], "prompt": "Find it."}) == {"success": True}
    assert calls[0]["tool_name"] == "point_image"
    assert calls[0]["timeout_seconds"] == 600.0


def test_sse_builder_uses_point_image_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class FakeTransport:
        def __init__(self, url: str) -> None:
            assert url == "http://molmo.example/sse"

        def call_tool(self, name, arguments, *, timeout_s=None):
            calls.append((name, arguments, timeout_s))
            return {"success": True}

    monkeypatch.setattr(handlers_module, "SseSimulatorMcpTransport", FakeTransport)
    pointer = build_sse_molmopoint_mcp_pointer(url="http://molmo.example/sse")
    request = {"images": [], "prompt": "Find it."}
    assert pointer(request) == {"success": True}
    assert calls == [("point_image", request, 600.0)]


def test_handler_normalizes_inputs_and_materializes_visual_success(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def point(request: dict[str, Any]) -> dict:
        calls.append(request)
        return _response(request)

    context = _context(_parameters(tmp_path))
    result = build_molmopoint_handler(point, output_root=tmp_path / "runs")(context)
    assert result.success is True
    assert context.parameters == {
        "images": [str((tmp_path / "first.png").resolve()), str((tmp_path / "second.jpg").resolve())],
        "prompt": "Look at Image 1. In Image 2, point to the same object.",
    }
    assert [payload["format"] for payload in calls[0]["images"]] == ["png", "jpeg"]
    outputs = result.details["outputs"]
    assert outputs["point_count"] == 2
    assert outputs["image_sources"] == context.parameters["images"]
    assert "unknown_backend_field" not in outputs["metadata"]
    artifact_types = [item["type"] for item in result.details["artifacts"]]
    assert artifact_types.count("molmopoint_point_overlay") == 2
    assert "molmopoint_contact_sheet" in artifact_types
    for artifact in result.details["artifacts"]:
        assert Path(artifact["path"]).is_file()
    run_dir = next((tmp_path / "runs").iterdir())
    for filename in ("request.json", "response.raw.json", "tool_result.json"):
        text = (run_dir / filename).read_text()
        assert "base64" not in text
        assert "raw_generation" not in text


def test_zero_points_is_success_and_still_generates_visuals(tmp_path: Path) -> None:
    result = build_molmopoint_handler(
        lambda request: _response(request, points=[]),
        output_root=tmp_path / "runs",
    )(_context(_parameters(tmp_path)))
    assert result.success is True
    assert result.details["outputs"]["points"] == []
    assert len([a for a in result.details["artifacts"] if a["kind"] == "image"]) == 3


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda value, root: value.update(images=[]), "invalid_image_count"),
        (lambda value, root: value.update(prompt="   "), "invalid_prompt"),
        (lambda value, root: value.update(images=[str(root / "missing.png")]), "image_not_found"),
        (lambda value, root: value.update(images=[str(_write_image(root / "bad.bmp", size=(8, 8)))]), "unsupported_image_format"),
    ],
)
def test_local_failures_are_persisted_without_calling_mcp(tmp_path: Path, mutate, reason: str) -> None:
    parameters = _parameters(tmp_path)
    mutate(parameters, tmp_path)
    result = build_molmopoint_handler(
        lambda _request: pytest.fail("MCP must not be called"),
        output_root=tmp_path / "runs",
    )(_context(parameters))
    assert result.success is False
    assert result.details["diagnostics"] == [{"code": reason}]
    assert result.details["outputs"]["points"] == []
    raw_artifact = next(a for a in result.details["artifacts"] if a["type"] == "molmopoint_raw_response")
    assert json.loads(Path(raw_artifact["path"]).read_text()) == {
        "mcp_called": False,
        "reason": reason,
    }


def test_transport_failure_is_persisted_without_exception_text(tmp_path: Path) -> None:
    def fail(_request):
        raise RuntimeError("private transport detail")

    result = build_molmopoint_handler(fail, output_root=tmp_path / "runs")(
        _context(_parameters(tmp_path))
    )
    assert result.success is False
    raw_artifact = next(a for a in result.details["artifacts"] if a["type"] == "molmopoint_raw_response")
    text = Path(raw_artifact["path"]).read_text()
    assert json.loads(text) == {"mcp_called": True, "reason": "mcp_call_failed"}
    assert "private transport detail" not in text


@pytest.mark.parametrize(
    "mutate",
    [
        lambda response: response["details"].update(point_count=99),
        lambda response: response["details"]["points"][0].update(object_id=1),
        lambda response: response["details"]["points"][0].update(pixel_x=9999.0),
        lambda response: response["details"]["metadata"].update(model_revision="short"),
        lambda response: response["details"].update(backend="other"),
    ],
)
def test_invalid_success_is_rejected_atomically(tmp_path: Path, mutate) -> None:
    def point(request):
        response = deepcopy(_response(request))
        mutate(response)
        return response

    result = build_molmopoint_handler(point, output_root=tmp_path / "runs")(
        _context(_parameters(tmp_path))
    )
    assert result.success is False
    assert result.details["outputs"]["points"] == []
    assert result.details["diagnostics"] == [{"code": "inconsistent_point_outputs"}]
    assert not any(a.get("kind") == "image" for a in result.details["artifacts"])
