from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("mcp")

from tools import anygrasp_mcp_server


class _Backend:
    def __init__(self, result: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.result = result or {"success": True}
        self.error = error

    def detect_grasps(self, **_kwargs: Any) -> dict[str, Any]:
        if self.error is not None:
            raise self.error
        return self.result


def _request() -> dict[str, Any]:
    return {
        "rgb": {},
        "depth": {},
        "intrinsics": {},
        "mode": "targeted",
        "target_mask": {},
    }


def test_detect_grasps_releases_cuda_cache_after_backend_result(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(anygrasp_mcp_server, "_BACKEND", _Backend())
    monkeypatch.setattr(anygrasp_mcp_server, "_release_cuda_cache", lambda: calls.append("release"))

    result = anygrasp_mcp_server.detect_grasps(**_request())

    assert result == {"success": True}
    assert calls == ["release"]


def test_detect_grasps_releases_cuda_cache_after_backend_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        anygrasp_mcp_server,
        "_BACKEND",
        _Backend(error=RuntimeError("inference failed")),
    )
    monkeypatch.setattr(anygrasp_mcp_server, "_release_cuda_cache", lambda: calls.append("release"))

    with pytest.raises(RuntimeError, match="inference failed"):
        anygrasp_mcp_server.detect_grasps(**_request())

    assert calls == ["release"]


@pytest.mark.parametrize("cuda_available", [False, True])
def test_release_cuda_cache_only_calls_torch_when_cuda_is_available(
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

    anygrasp_mcp_server._release_cuda_cache()

    assert calls == (["empty_cache"] if cuda_available else [])
