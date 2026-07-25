from __future__ import annotations

import base64
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from agent.tools import handlers as handlers_module
from agent.tools.handlers import (
    DEFAULT_GRASPGENX_OUTPUT_ROOT,
    build_graspgenx_gripper_list_handler,
    build_graspgenx_handler,
    build_sse_graspgenx_mcp_gripper_lister,
    build_sse_graspgenx_mcp_predictor,
    build_stdio_graspgenx_mcp_gripper_lister,
    build_stdio_graspgenx_mcp_predictor,
    bind_dummy_tool_handlers,
)
from agent.tools.registry import ToolEffect, ToolExecutionContext, build_default_tool_registry


INTRINSICS = {"fx": 100.0, "fy": 100.0, "cx": 8.0, "cy": 8.0, "scale": 1000.0}


def _write_fixture(tmp_path: Path, *, size: tuple[int, int] = (16, 16)) -> dict[str, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rgb = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", size, (80, 120, 160)).save(rgb)
    Image.fromarray(np.full((size[1], size[0]), 500, dtype=np.uint16)).save(depth)
    Image.fromarray(np.full((size[1], size[0]), 255, dtype=np.uint8)).save(mask)
    return {"rgb": str(rgb), "depth": str(depth), "mask": str(mask)}


def _parameters(tmp_path: Path) -> dict[str, Any]:
    paths = _write_fixture(tmp_path)
    return {
        "rgb": paths["rgb"],
        "depth": paths["depth"],
        "object_mask": {
            "type": "segmentation_mask",
            "mask_ref": paths["mask"],
            "source_image": paths["rgb"],
            "label": "box",
            "score": 0.91,
        },
        "intrinsics": dict(INTRINSICS),
        "gripper_name": "franka_panda",
        "up_direction_camera": [0.0, 0.0, -2.0],
    }


def _candidate(index: int, *, score: float) -> dict[str, Any]:
    translation = [0.01 * index, 0.0, 0.5]
    transform = [
        [1.0, 0.0, 0.0, translation[0]],
        [0.0, 1.0, 0.0, translation[1]],
        [0.0, 0.0, 1.0, translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return {
        "id": f"graspgenx_{index:03d}",
        "rank": index,
        "backend_index": index + 3,
        "source_model": "graspgenx",
        "gripper_name": "franka_panda",
        "candidate_source": "diffusion" if index % 2 == 0 else "obb",
        "frame": "camera",
        "camera_frame": "opencv",
        "grasp_frame": "graspnet",
        "convention": "p_camera = R @ p_grasp + t",
        "score": score,
        "translation_xyz": translation,
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "transform_matrix": transform,
        "gripper_tip_position_xyz": [translation[0], 0.0, 0.6],
        "depth": 0.1,
        "width": 0.08,
        "height": 0.04,
        "model_native_grasp_pose": {
            "frame": "camera",
            "camera_frame": "opencv",
            "grasp_frame": "graspgenx",
            "convention": "p_camera = R @ p_grasp + t",
            "transform_matrix": transform,
        },
    }


def _success_response() -> dict[str, Any]:
    candidates = [_candidate(0, score=0.9), _candidate(1, score=0.8)]
    return {
        "success": True,
        "content": "GraspGenX grasp prediction completed.",
        "details": {
            "tool": "predict_grasps",
            "backend": "graspgenx_mcp",
            "model": "graspgenx",
            "planner": "graspmoe",
            "deterministic": False,
            "frame": "camera",
            "camera_frame": "opencv",
            "grasp_frame": "graspnet",
            "gripper_name": "franka_panda",
            "candidate_count": len(candidates),
            "grasp_candidates": candidates,
            "ranking": "score_descending",
            "artifacts": [],
            "metadata": {
                "returned_candidate_count": len(candidates),
                "object_points": [[1, 2, 3]],
                "checkpoint_root": "/private/checkpoints",
            },
        },
    }


def _listing_response() -> dict[str, Any]:
    return {
        "success": True,
        "content": "GraspGenX gripper listing completed.",
        "details": {
            "tool": "list_grippers",
            "gripper_count": 1,
            "grippers": [
                {
                    "name": "franka_panda",
                    "gripper_type": "parallel_jaw",
                    "fingertip_depth": 0.1,
                    "sweep_volume_open": {
                        "extents_xyz": [0.08, 0.04, 0.12],
                        "offset_xyz": [0.0, 0.0, 0.06],
                    },
                    "sweep_volume_mid": {
                        "extents_xyz": [0.04, 0.04, 0.12],
                        "offset_xyz": [0.0, 0.0, 0.06],
                    },
                    "asset_family": "x_grippers",
                }
            ],
            "model_loaded": False,
        },
    }


def _context(parameters: dict[str, Any]) -> ToolExecutionContext:
    spec = build_default_tool_registry().get("graspgenx")
    return ToolExecutionContext(name="graspgenx", spec=spec, parameters=parameters)


def test_specs_are_visible_without_dummy_handlers() -> None:
    tools = bind_dummy_tool_handlers(build_default_tool_registry())
    prediction = tools.get("graspgenx")
    listing = tools.get("list_graspgenx_grippers")

    assert prediction.category == "manipulation"
    assert prediction.effect == ToolEffect.PLANNING
    assert set(prediction.parameters) == {
        "rgb",
        "depth",
        "object_mask",
        "intrinsics",
        "gripper_name",
        "up_direction_camera",
    }
    assert listing.effect == ToolEffect.READ_ONLY
    assert listing.safe_by_default is True
    assert tools.can_execute("graspgenx") is False
    assert tools.can_execute("list_graspgenx_grippers") is False


def test_default_root_uses_repo_tmp_layout() -> None:
    assert DEFAULT_GRASPGENX_OUTPUT_ROOT == Path("tmp") / "tool_result" / "graspgenx"


def test_handler_sends_geometry_only_and_generates_audited_visuals(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def predict(request: dict[str, Any]) -> dict[str, Any]:
        calls.append(request)
        return _success_response()

    parameters = _parameters(tmp_path)
    result = build_graspgenx_handler(
        predict,
        _listing_response,
        output_root=tmp_path / "results",
    )(_context(parameters))

    assert result.success is True
    assert set(calls[0]) == {
        "depth",
        "object_mask",
        "intrinsics",
        "gripper_name",
        "up_direction_camera",
    }
    assert base64.b64decode(calls[0]["depth"]["base64"])
    assert base64.b64decode(calls[0]["object_mask"]["base64"])
    assert calls[0]["up_direction_camera"] == [0.0, 0.0, -1.0]
    assert result.details["candidate_count"] == 2
    assert [item["id"] for item in result.details["grasp_candidates"]] == [
        "graspgenx_000",
        "graspgenx_001",
    ]
    assert "model_native_grasp_pose" not in result.details["grasp_candidates"][0]
    assert result.details["source"] == {
        "source_tool": "graspgenx",
        "mode": "targeted",
        "rgb": parameters["rgb"],
        "depth": parameters["depth"],
        "object_mask": parameters["object_mask"]["mask_ref"],
        "intrinsics": INTRINSICS,
        "gripper_name": "franka_panda",
        "up_direction_camera": [0.0, 0.0, -1.0],
    }
    assert result.details["best_grasp_candidate"]["id"] == "graspgenx_000"
    assert result.details["active_grasp_candidate"]["id"] == "graspgenx_000"

    artifacts = result.details["artifacts"]
    overlays = [item for item in artifacts if item["kind"] == "image"]
    assert {item["selection"] for item in overlays} == {"top_1", "top_10"}
    assert all(Path(item["path"]).is_file() for item in overlays)
    for key in ("request_ref", "raw_output_ref", "tool_result_ref"):
        assert Path(result.details[key]).is_file()
    request = json.loads(Path(result.details["request_ref"]).read_text())
    assert request["object_mask"]["label"] == "box"
    raw = Path(result.details["raw_output_ref"]).read_text()
    assert "model_native_grasp_pose" in raw
    for key in ("request_ref", "raw_output_ref", "tool_result_ref"):
        text = Path(result.details[key]).read_text()
        assert "base64" not in text
        assert "object_points" not in text
        assert "checkpoint_root" not in text


def test_handler_accepts_resolved_mask_source_symlink(tmp_path: Path) -> None:
    parameters = _parameters(tmp_path)
    alias = tmp_path / "rgb-alias.png"
    alias.symlink_to(Path(parameters["rgb"]))
    parameters["object_mask"]["source_image"] = str(alias)

    result = build_graspgenx_handler(
        lambda _request: _success_response(),
        _listing_response,
        output_root=tmp_path / "results",
    )(_context(parameters))

    assert result.success is True


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda value, root: value.update(rgb=""), "missing_rgb"),
        (lambda value, root: value.update(depth=""), "missing_depth"),
        (lambda value, root: value.update(object_mask="mask.png"), "invalid_object_mask"),
        (lambda value, root: value.pop("intrinsics"), "missing_intrinsics"),
        (
            lambda value, root: value.update(intrinsics={**INTRINSICS, "scale": 0}),
            "invalid_intrinsics",
        ),
        (lambda value, root: value.update(gripper_name=""), "missing_gripper_name"),
        (
            lambda value, root: value.update(up_direction_camera=[0, 0, 0]),
            "invalid_up_direction_camera",
        ),
        (lambda value, root: value.update(rgb=str(root / "missing.png")), "rgb_not_found"),
        (
            lambda value, root: value.update(depth=str(root / "missing.png")),
            "depth_not_found",
        ),
        (
            lambda value, root: value["object_mask"].update(
                mask_ref=str(root / "missing.png")
            ),
            "object_mask_not_found",
        ),
        (
            lambda value, root: value["object_mask"].update(
                source_image=str(_write_fixture(root / "other")["rgb"])
            ),
            "object_mask_source_mismatch",
        ),
    ],
)
def test_handler_persists_preflight_failures(
    tmp_path: Path,
    mutate,
    reason: str,
) -> None:
    parameters = _parameters(tmp_path)
    mutate(parameters, tmp_path)
    result = build_graspgenx_handler(
        lambda _request: pytest.fail("prediction MCP must not be called"),
        lambda: pytest.fail("listing MCP must not be called"),
        output_root=tmp_path / "results",
    )(_context(parameters))

    assert result.success is False
    assert result.details["reason"] == reason
    assert result.details["candidate_count"] == 0
    assert Path(result.details["request_ref"]).is_file()
    assert Path(result.details["raw_output_ref"]).is_file()
    assert Path(result.details["tool_result_ref"]).is_file()
    raw = json.loads(Path(result.details["raw_output_ref"]).read_text())
    assert raw == {"mcp_called": False, "reason": reason}


def test_handler_rejects_image_shape_mismatch_before_mcp(tmp_path: Path) -> None:
    parameters = _parameters(tmp_path)
    Image.new("RGB", (8, 8), (0, 0, 0)).save(parameters["rgb"])

    result = build_graspgenx_handler(
        lambda _request: pytest.fail("prediction MCP must not be called"),
        lambda: pytest.fail("listing MCP must not be called"),
        output_root=tmp_path / "results",
    )(_context(parameters))

    assert result.success is False
    assert result.details["reason"] == "image_shape_mismatch"
    assert result.details["metadata"]["rgb_size"] == [8, 8]


def test_handler_preserves_backend_failure_and_scrubs_raw_response(tmp_path: Path) -> None:
    response = {
        "success": False,
        "content": "GraspGenX grasp prediction failed: unsupported_gripper.",
        "details": {
            "reason": "unsupported_gripper",
            "metadata": {
                "checkpoint_root": "/private/checkpoints",
                "base64": "secret",
                "point_cloud": [[1, 2, 3]],
                "gripper_name": "missing",
            },
        },
    }
    result = build_graspgenx_handler(
        lambda _request: response,
        _listing_response,
        output_root=tmp_path / "results",
    )(_context(_parameters(tmp_path)))

    assert result.success is False
    assert result.details["reason"] == "unsupported_gripper"
    raw = Path(result.details["raw_output_ref"]).read_text()
    assert "checkpoint_root" not in raw
    assert "base64" not in raw
    assert "point_cloud" not in raw


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["details"].update(candidate_count=3),
        lambda value: value["details"].update(camera_frame="opengl"),
        lambda value: value["details"]["grasp_candidates"][1].update(
            id="graspgenx_000"
        ),
        lambda value: value["details"]["grasp_candidates"][1].update(score=0.95),
        lambda value: value["details"]["grasp_candidates"][0][
            "transform_matrix"
        ][0].__setitem__(3, 1.0),
        lambda value: value["details"]["grasp_candidates"][0].update(
            gripper_name="other"
        ),
    ],
)
def test_handler_rejects_inconsistent_success_atomically(
    tmp_path: Path,
    mutate,
) -> None:
    response = _success_response()
    mutate(response)
    result = build_graspgenx_handler(
        lambda _request: response,
        _listing_response,
        output_root=tmp_path / "results",
    )(_context(_parameters(tmp_path)))

    assert result.success is False
    assert result.details["reason"] == "inconsistent_grasp_outputs"
    assert result.details["candidate_count"] == 0
    assert result.details["grasp_candidates"] == []


def test_visualization_failure_is_nonfatal(tmp_path: Path) -> None:
    result = build_graspgenx_handler(
        lambda _request: _success_response(),
        lambda: {"success": False},
        output_root=tmp_path / "results",
    )(_context(_parameters(tmp_path)))

    assert result.success is True
    assert result.details["candidate_count"] == 2
    assert result.details["diagnostics"][0]["code"] == "visualization_failed"
    assert not [item for item in result.details["artifacts"] if item["kind"] == "image"]


def test_gripper_list_handler_returns_validated_sorted_capabilities() -> None:
    result = build_graspgenx_gripper_list_handler(_listing_response)(
        ToolExecutionContext(
            name="list_graspgenx_grippers",
            spec=build_default_tool_registry().get("list_graspgenx_grippers"),
            parameters={},
        )
    )

    assert result.success is True
    assert result.details["gripper_count"] == 1
    assert result.details["grippers"][0]["name"] == "franka_panda"
    assert result.details["model_loaded"] is False
    assert "request_ref" not in result.details


def test_gripper_list_handler_rejects_duplicate_or_unsorted_results() -> None:
    response = _listing_response()
    duplicate = deepcopy(response["details"]["grippers"][0])
    response["details"]["grippers"] = [duplicate, duplicate]
    response["details"]["gripper_count"] = 2

    result = build_graspgenx_gripper_list_handler(lambda: response)(
        ToolExecutionContext(
            name="list_graspgenx_grippers",
            spec=build_default_tool_registry().get("list_graspgenx_grippers"),
            parameters={},
        )
    )

    assert result.success is False
    assert result.details["reason"] == "inconsistent_gripper_outputs"


def test_stdio_builders_use_expected_tools_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_call(**kwargs):
        calls.append(kwargs)
        return {"success": True}

    monkeypatch.setattr(handlers_module, "_call_stdio_mcp_tool", fake_call)
    predictor = build_stdio_graspgenx_mcp_predictor(command="python")
    lister = build_stdio_graspgenx_mcp_gripper_lister(command="python")

    assert predictor({"depth": {}}) == {"success": True}
    assert lister() == {"success": True}
    assert [item["tool_name"] for item in calls] == ["predict_grasps", "list_grippers"]
    assert all(item["timeout_seconds"] == 600.0 for item in calls)


def test_sse_builders_use_expected_tools_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any], float | None]] = []

    class FakeTransport:
        def __init__(self, url: str) -> None:
            assert url == "http://graspgenx.example/sse"

        def call_tool(self, name, arguments, *, timeout_s=None):
            calls.append((name, arguments, timeout_s))
            return {"success": True}

    monkeypatch.setattr(handlers_module, "SseSimulatorMcpTransport", FakeTransport)
    predictor = build_sse_graspgenx_mcp_predictor(
        url="http://graspgenx.example/sse"
    )
    lister = build_sse_graspgenx_mcp_gripper_lister(
        url="http://graspgenx.example/sse"
    )

    assert predictor({"depth": {}}) == {"success": True}
    assert lister() == {"success": True}
    assert calls == [
        ("predict_grasps", {"depth": {}}, 600.0),
        ("list_grippers", {}, 600.0),
    ]
