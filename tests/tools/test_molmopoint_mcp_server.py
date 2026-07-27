from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("mcp")

from tools import molmopoint_mcp_server as server


class _Backend:
    def point_images(self, **_kwargs: Any) -> dict[str, Any]:
        return {"success": True}


def test_point_image_calls_configured_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "_BACKEND", _Backend())
    assert server.point_image(images=[], prompt="Find a can.") == {"success": True}


def test_point_image_without_backend_returns_structured_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_BACKEND", None)
    result = server.point_image(images=[], prompt="Find a can.")
    assert result["success"] is False
    assert result["details"]["reason"] == "model_load_failed"
    assert result["details"]["points"] == []


def test_mcp_callable_has_minimal_schema_and_examples() -> None:
    signature = inspect.signature(server.point_image)
    assert list(signature.parameters) == ["images", "prompt"]
    docstring = inspect.getdoc(server.point_image) or ""
    assert "Point to any green can" in docstring
    assert "Point to every red block" in docstring
    assert "Locate the cup to the left" in docstring
    assert "Look at the object in Image 1" in docstring
    assert "image_index=1" in docstring
    assert "Zero points is a successful result" in docstring
    assert "does not return confidence" in docstring


def test_main_builds_stdio_backend_from_local_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    monkeypatch.setattr(server, "_validate_cuda_runtime", lambda _parser: object())
    monkeypatch.setattr(server, "_resolve_local_snapshot", lambda **_kwargs: snapshot)
    monkeypatch.setattr(server.mcp, "run", lambda *, transport: calls.append(transport))
    monkeypatch.setattr(server, "_BACKEND", None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "molmopoint_mcp_server.py",
            "--transport",
            "stdio",
            "--model-revision",
            "a" * 40,
        ],
    )
    assert server.main() == 0
    assert calls == ["stdio"]
    assert server._BACKEND is not None
    assert server._BACKEND.model_path == snapshot
    assert server._BACKEND.model_revision == "a" * 40


def test_cuda_validation_rejects_unavailable_device() -> None:
    parser = SimpleNamespace(error=lambda message: (_ for _ in ()).throw(ValueError(message)))
    original = sys.modules.get("torch")
    sys.modules["torch"] = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    try:
        with pytest.raises(ValueError, match="CPU fallback is not supported"):
            server._validate_cuda_runtime(parser)
    finally:
        if original is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original
