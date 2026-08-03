from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("mcp")

from tools import anyplace_mcp_server


class _Backend:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error

    def predict_placement(self, **_kwargs: Any) -> dict[str, Any]:
        if self.error is not None:
            raise self.error
        return {"success": True}


def _request() -> dict[str, Any]:
    return {
        "rgb": {},
        "depth": {},
        "object_mask": {},
        "placement_region_mask": {},
        "intrinsics": {},
        "selected_grasp": {},
    }


def test_predict_placement_releases_cache_after_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(anyplace_mcp_server, "_BACKEND", _Backend())
    monkeypatch.setattr(anyplace_mcp_server, "_release_cuda_cache", lambda: calls.append("release"))

    result = anyplace_mcp_server.predict_placement(**_request())

    assert result == {"success": True}
    assert calls == ["release"]


def test_predict_placement_releases_cache_after_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(anyplace_mcp_server, "_BACKEND", _Backend(error=RuntimeError("failed")))
    monkeypatch.setattr(anyplace_mcp_server, "_release_cuda_cache", lambda: calls.append("release"))

    with pytest.raises(RuntimeError, match="failed"):
        anyplace_mcp_server.predict_placement(**_request())

    assert calls == ["release"]


@pytest.mark.parametrize("cuda_available", [False, True])
def test_release_cuda_cache_only_empties_when_cuda_is_available(
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

    anyplace_mcp_server._release_cuda_cache()

    assert calls == (["empty_cache"] if cuda_available else [])
