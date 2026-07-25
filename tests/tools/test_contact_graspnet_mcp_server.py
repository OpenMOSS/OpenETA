from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("mcp")

from tools import contact_graspnet_mcp_server


class _Backend:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error

    def predict_grasps(self, **_kwargs: Any) -> dict[str, Any]:
        if self.error is not None:
            raise self.error
        return {"success": True}


def _request() -> dict[str, Any]:
    return {"depth": {}, "object_mask": {}, "intrinsics": {}}


def test_predict_grasps_releases_cuda_cache_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(contact_graspnet_mcp_server, "_BACKEND", _Backend())
    monkeypatch.setattr(
        contact_graspnet_mcp_server,
        "_release_cuda_cache",
        lambda: calls.append("release"),
    )

    assert contact_graspnet_mcp_server.predict_grasps(**_request()) == {"success": True}
    assert calls == ["release"]


def test_predict_grasps_releases_cuda_cache_after_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        contact_graspnet_mcp_server,
        "_BACKEND",
        _Backend(error=RuntimeError("failed")),
    )
    monkeypatch.setattr(
        contact_graspnet_mcp_server,
        "_release_cuda_cache",
        lambda: calls.append("release"),
    )

    with pytest.raises(RuntimeError, match="failed"):
        contact_graspnet_mcp_server.predict_grasps(**_request())
    assert calls == ["release"]


@pytest.mark.parametrize("cuda_available", [False, True])
def test_release_cuda_cache_is_safe_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
) -> None:
    calls: list[str] = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available,
            empty_cache=lambda: calls.append("empty_cache"),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    contact_graspnet_mcp_server._release_cuda_cache()

    assert calls == (["empty_cache"] if cuda_available else [])


def test_unconfigured_backend_returns_structured_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contact_graspnet_mcp_server, "_BACKEND", None)

    result = contact_graspnet_mcp_server.predict_grasps(**_request())

    assert result["success"] is False
    assert result["details"]["reason"] == "model_load_failed"
    assert result["details"]["candidate_count"] == 0
    assert result["details"]["grasp_candidates"] == []


def test_mcp_callable_has_minimal_schema_and_documentation() -> None:
    signature = inspect.signature(contact_graspnet_mcp_server.predict_grasps)
    assert list(signature.parameters) == ["depth", "object_mask", "intrinsics"]
    docstring = inspect.getdoc(contact_graspnet_mcp_server.predict_grasps) or ""
    assert "scale=1000" in docstring
    assert "0.2-1.8 meters" in docstring
    assert "RGB is deliberately" in docstring
    assert "Panda-compatible" in docstring
    assert "GraspNet grasp-frame" in docstring


def test_main_builds_stdio_backend_from_external_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "backend"
    checkpoint_dir = tmp_path / "checkpoint"
    root.mkdir()
    (checkpoint_dir / "checkpoints").mkdir(parents=True)
    (checkpoint_dir / "config.yaml").write_text("DATA: {}\n", encoding="utf-8")
    (checkpoint_dir / "checkpoints" / "model.pt").write_bytes(b"placeholder")
    calls: list[str] = []

    def run(*, transport: str) -> None:
        calls.append(transport)

    monkeypatch.setattr(contact_graspnet_mcp_server.mcp, "run", run)
    monkeypatch.setattr(contact_graspnet_mcp_server, "_BACKEND", None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "contact_graspnet_mcp_server.py",
            "--transport",
            "stdio",
            "--contact-graspnet-root",
            str(root),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ],
    )

    assert contact_graspnet_mcp_server.main() == 0
    assert calls == ["stdio"]
    backend = contact_graspnet_mcp_server._BACKEND
    assert backend is not None
    assert backend.contact_graspnet_root == root
    assert backend.checkpoint_dir == checkpoint_dir
    assert backend.depth_min == 0.2
    assert backend.depth_max == 1.8
    assert backend.max_candidates == 20
