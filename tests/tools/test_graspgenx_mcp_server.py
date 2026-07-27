from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_args

import pytest

pytest.importorskip("mcp")

from tools import graspgenx_mcp_server


class _Backend:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.model_loaded = False
        self.grippers = {"franka_panda": object(), "robotiq_2f_85": object()}
        self.invalid_grippers: dict[str, str] = {}

    def predict_grasps(self, **_kwargs: Any) -> dict[str, Any]:
        if self.error is not None:
            raise self.error
        return {"success": True}

    def list_grippers(self) -> dict[str, Any]:
        return {
            "success": True,
            "details": {
                "tool": "list_grippers",
                "model_loaded": self.model_loaded,
            },
        }


def _request() -> dict[str, Any]:
    return {
        "depth": {},
        "object_mask": {},
        "intrinsics": {},
        "gripper_name": "franka_panda",
        "up_direction_camera": [0.0, 0.0, -1.0],
    }


def test_predict_grasps_releases_cuda_cache_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(graspgenx_mcp_server, "_BACKEND", _Backend())
    monkeypatch.setattr(
        graspgenx_mcp_server,
        "_release_cuda_cache",
        lambda: calls.append("release"),
    )

    assert graspgenx_mcp_server.predict_grasps(**_request()) == {"success": True}
    assert calls == ["release"]


def test_predict_grasps_releases_cuda_cache_after_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        graspgenx_mcp_server,
        "_BACKEND",
        _Backend(error=RuntimeError("failed")),
    )
    monkeypatch.setattr(
        graspgenx_mcp_server,
        "_release_cuda_cache",
        lambda: calls.append("release"),
    )

    with pytest.raises(RuntimeError, match="failed"):
        graspgenx_mcp_server.predict_grasps(**_request())
    assert calls == ["release"]


@pytest.mark.parametrize("cuda_available", [False, True])
def test_release_cuda_cache_is_safe_without_cuda(
    monkeypatch: pytest.MonkeyPatch, cuda_available: bool
) -> None:
    calls: list[str] = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available,
            empty_cache=lambda: calls.append("empty_cache"),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    graspgenx_mcp_server._release_cuda_cache()

    assert calls == (["empty_cache"] if cuda_available else [])


def test_unconfigured_backend_returns_atomic_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(graspgenx_mcp_server, "_BACKEND", None)
    monkeypatch.setattr(graspgenx_mcp_server, "_release_cuda_cache", lambda: None)

    result = graspgenx_mcp_server.predict_grasps(**_request())

    assert result["success"] is False
    assert result["details"]["reason"] == "model_load_failed"
    assert result["details"]["candidate_count"] == 0
    assert result["details"]["grasp_candidates"] == []


def test_list_grippers_does_not_release_cuda_or_load_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _Backend()
    monkeypatch.setattr(graspgenx_mcp_server, "_BACKEND", backend)
    monkeypatch.setattr(
        graspgenx_mcp_server,
        "_release_cuda_cache",
        lambda: pytest.fail("list_grippers must not touch CUDA"),
    )

    result = graspgenx_mcp_server.list_grippers()

    assert result["success"] is True
    assert result["details"]["model_loaded"] is False


def test_predict_callable_has_complete_schema_and_documentation() -> None:
    signature = inspect.signature(graspgenx_mcp_server.predict_grasps)
    assert list(signature.parameters) == [
        "depth",
        "object_mask",
        "intrinsics",
        "gripper_name",
        "up_direction_camera",
    ]
    docstring = inspect.getdoc(graspgenx_mcp_server.predict_grasps) or ""
    assert "RGB is not" in docstring
    assert "scale=1000" in docstring
    assert "camera_z < 1 meter" in docstring
    assert "above 3500" in docstring
    assert "without a 0.7 cutoff" in docstring
    assert "camera/opencv" in docstring
    assert "GraspGenX" in docstring and "GraspNet/AnyGrasp" in docstring


def test_build_mcp_uses_dynamic_gripper_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered: dict[str, Any] = {}

    class FakeFastMCP:
        def __init__(self, name: str, **_kwargs: Any) -> None:
            assert name == "openeta-graspgenx"

        def tool(self):
            def register(function):
                registered[function.__name__] = function
                return function

            return register

    monkeypatch.setattr(graspgenx_mcp_server, "FastMCP", FakeFastMCP)

    graspgenx_mcp_server.build_mcp(["robotiq_2f_85", "franka_panda"])

    predict = registered["predict_grasps"]
    assert get_args(predict.__annotations__["gripper_name"]) == (
        "franka_panda",
        "robotiq_2f_85",
    )
    assert set(registered) == {"predict_grasps", "list_grippers"}


def test_real_fastmcp_schema_exposes_dynamic_gripper_enum() -> None:
    mcp = graspgenx_mcp_server.build_mcp(
        ["robotiq_2f_85", "franka_panda"]
    )

    tools = asyncio.run(mcp.list_tools())
    by_name = {tool.name: tool for tool in tools}

    assert set(by_name) == {"predict_grasps", "list_grippers"}
    assert by_name["predict_grasps"].inputSchema["properties"]["gripper_name"][
        "enum"
    ] == ["franka_panda", "robotiq_2f_85"]


def test_health_reports_transport_readiness_without_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _Backend()
    monkeypatch.setattr(graspgenx_mcp_server, "_BACKEND", backend)

    health = graspgenx_mcp_server.health_payload()

    assert health == {
        "ok": True,
        "server": "openeta-graspgenx",
        "tools": ["list_grippers", "predict_grasps"],
        "model_loaded": False,
        "gripper_count": 2,
    }


def test_main_builds_stdio_backend_from_explicit_offline_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "GraspGenX"
    (source / "graspgenx").mkdir(parents=True)
    (source / "graspgenx" / "__init__.py").write_text("", encoding="utf-8")
    calls: list[str] = []
    backend = _Backend()

    class Runner:
        def run(self, *, transport: str) -> None:
            calls.append(transport)

    monkeypatch.setattr(
        graspgenx_mcp_server,
        "GraspGenXBackend",
        lambda **_kwargs: backend,
    )
    monkeypatch.setattr(
        graspgenx_mcp_server,
        "build_mcp",
        lambda _names: Runner(),
    )
    monkeypatch.setattr(graspgenx_mcp_server, "_BACKEND", None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "graspgenx_mcp_server.py",
            "--transport",
            "stdio",
            "--graspgenx-root",
            str(source),
            "--checkpoint-root",
            str(tmp_path / "checkpoints" / "release"),
            "--gripper-descriptions-root",
            str(tmp_path / "gripper_descriptions"),
            "--device",
            "cuda:2",
        ],
    )

    assert graspgenx_mcp_server.main() == 0
    assert calls == ["stdio"]
    assert graspgenx_mcp_server._BACKEND is backend
