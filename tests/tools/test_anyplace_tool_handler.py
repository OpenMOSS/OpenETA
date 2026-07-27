from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest

from agent.tools import handlers as handlers_module
from agent.tools.handlers import (
    DEFAULT_ANYPLACE_OUTPUT_ROOT,
    build_anyplace_handler,
    build_sse_anyplace_mcp_placer,
    build_stdio_anyplace_mcp_placer,
)
from agent.tools.registry import ToolEffect, ToolExecutionContext, build_default_tool_registry


IMAGE_BYTES = b"openeta-test-image"
INTRINSICS = {"fx": 618.0, "fy": 618.0, "cx": 256.0, "cy": 256.0, "scale": 1000.0}


def _write_image(path: Path) -> Path:
    path.write_bytes(IMAGE_BYTES)
    return path


def _candidate(**overrides: Any) -> dict[str, Any]:
    value = {
        "id": "grasp_003",
        "frame": "camera",
        "camera_frame": "opencv",
        "score": 0.7,
        "translation_xyz": [0.1, 0.2, 0.3],
        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "gripper_tip_position_xyz": [0.13, 0.2, 0.3],
        "depth": 0.03,
        "width": 0.06,
        "height": 0.03,
        "future_anygrasp_metadata": {"allowed": True},
    }
    value.update(overrides)
    return value


def _parameters(tmp_path: Path) -> dict[str, Any]:
    rgb = _write_image(tmp_path / "rgb.png")
    depth = _write_image(tmp_path / "depth.png")
    object_mask = _write_image(tmp_path / "object-mask.png")
    placement_mask = _write_image(tmp_path / "placement-mask.png")
    return {
        "rgb": str(rgb),
        "depth": str(depth),
        "object_mask": str(object_mask),
        "placement_region_mask": {
            "type": "segmentation_mask",
            "mask_ref": str(placement_mask),
            "source_image": str(rgb),
            "label": "rack slot",
            "score": 0.91,
        },
        "intrinsics": dict(INTRINSICS),
        "selected_grasp": {
            "candidate": _candidate(),
            "source": {
                "mode": "targeted",
                "rgb": str(rgb),
                "depth": str(depth),
                "object_mask": str(object_mask),
                "intrinsics": dict(INTRINSICS),
            },
        },
    }


def _place_pose(index: int) -> dict[str, Any]:
    return {
        "id": f"place_grasp_{index:03d}",
        "frame": "camera",
        "camera_frame": "opencv",
        "score": 0.7,
        "translation_xyz": [0.1 + index, 0.2, 0.3],
        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "gripper_tip_position_xyz": [0.13 + index, 0.2, 0.3],
        "depth": 0.03,
        "width": 0.06,
        "height": 0.03,
    }


def _success_response(*, source_grasp_id: str = "grasp_003") -> dict[str, Any]:
    candidates = []
    for index in range(5):
        candidates.append(
            {
                "id": f"placement_{index:03d}",
                "source_grasp_id": source_grasp_id,
                "object_placement_transform": {
                    "frame": "camera",
                    "camera_frame": "opencv",
                    "convention": "p_placed = R @ p_current + t",
                    "transform_matrix": [
                        [1.0, 0.0, 0.0, float(index)],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
                "place_grasp_pose": _place_pose(index),
                "ignored_backend_field": "allowed",
            }
        )
    return {
        "success": True,
        "content": "AnyPlace placement prediction completed.",
        "details": {
            "tool": "anyplace",
            "backend": "anyplace_mcp",
            "model": "anyplace_multitask",
            "frame": "camera",
            "camera_frame": "opencv",
            "candidate_count": 5,
            "placement_candidates": candidates,
            "metadata": {
                "raw_object_point_count": 2048,
                "point_cloud": [[0.0, 0.0, 0.0]],
                "base64": "must-not-leak",
            },
        },
    }


def _context(
    parameters: dict[str, Any],
    *,
    session_id: str = "",
) -> ToolExecutionContext:
    spec = build_default_tool_registry().get("anyplace")
    return ToolExecutionContext(
        name="anyplace",
        spec=spec,
        parameters=parameters,
        metadata={"session_id": session_id} if session_id else {},
    )


def test_anyplace_spec_is_visible_without_dummy_handler() -> None:
    tools = build_default_tool_registry()
    spec = tools.get("anyplace")

    assert spec.category == "manipulation"
    assert spec.effect == ToolEffect.PLANNING
    assert spec.safe_by_default is False
    assert tools.can_execute("anyplace") is False


def test_anyplace_default_root_uses_repo_tmp_layout() -> None:
    assert DEFAULT_ANYPLACE_OUTPUT_ROOT == Path("tmp") / "tool_result" / "anyplace"


def test_anyplace_stdio_builder_uses_predict_tool_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    async def fake_call(**kwargs):
        calls.append(kwargs)
        return {"success": True}

    monkeypatch.setattr(handlers_module, "_call_stdio_mcp_tool", fake_call)
    placer = build_stdio_anyplace_mcp_placer(command="python", args=["server.py"])

    assert placer({"rgb": {}}) == {"success": True}
    assert calls[0]["tool_name"] == "predict_placement"
    assert calls[0]["timeout_seconds"] == 600.0


def test_anyplace_sse_builder_uses_predict_tool_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakeTransport:
        def __init__(self, url: str) -> None:
            assert url == "http://anyplace.example/sse"

        def call_tool(self, name, arguments, *, timeout_s=None):
            calls.append((name, arguments, timeout_s))
            return {"success": True}

    monkeypatch.setattr(handlers_module, "SseSimulatorMcpTransport", FakeTransport)
    placer = build_sse_anyplace_mcp_placer(url="http://anyplace.example/sse")

    assert placer({"rgb": {}}) == {"success": True}
    assert calls == [("predict_placement", {"rgb": {}}, 600.0)]


def test_anyplace_handler_encodes_inputs_and_materializes_success(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def predict(request: dict[str, Any]) -> dict[str, Any]:
        calls.append(request)
        return _success_response()

    parameters = _parameters(tmp_path)
    handler = build_anyplace_handler(predict, output_root=tmp_path / "runs")

    result = handler(_context(parameters, session_id="place-session"))

    assert result.success is True
    assert result.details["candidate_count"] == 5
    assert [item["id"] for item in result.details["placement_candidates"]] == [
        f"placement_{index:03d}" for index in range(5)
    ]
    assert all(
        item["place_grasp_pose"]["source_grasp_id"] == "grasp_003"
        for item in result.details["placement_candidates"]
    )
    assert result.details["selected_grasp_id"] == "grasp_003"
    assert result.details["selected_grasp_source"]["source_tool"] == "anygrasp"
    assert all(
        item["place_grasp_pose"]["source_tool"] == "anygrasp"
        for item in result.details["placement_candidates"]
    )
    assert calls[0]["selected_grasp"]["id"] == "grasp_003"
    assert "future_anygrasp_metadata" not in calls[0]["selected_grasp"]
    for key in ("rgb", "depth", "object_mask", "placement_region_mask"):
        assert calls[0][key]["format"] == "png"
        assert base64.b64decode(calls[0][key]["base64"]) == IMAGE_BYTES

    raw_output_ref = Path(result.details["raw_output_ref"])
    run_dir = raw_output_ref.parent
    assert run_dir.relative_to(tmp_path / "runs").parts[0] == "place-session"
    request_text = (run_dir / "request.json").read_text()
    raw_text = raw_output_ref.read_text()
    tool_result_text = (run_dir / "tool_result.json").read_text()
    assert '"label": "rack slot"' in request_text
    assert "future_anygrasp_metadata" in request_text
    for text in (request_text, raw_text, tool_result_text):
        assert "base64" not in text
        assert "point_cloud" not in text
    assert "ignored_backend_field" in raw_text
    assert "ignored_backend_field" not in tool_result_text


def test_anyplace_handler_forwards_depth_truncation(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    parameters = _parameters(tmp_path)
    parameters["depth_truncation"] = 2.0

    result = build_anyplace_handler(
        lambda request: calls.append(request) or _success_response(),
        output_root=tmp_path / "runs",
    )(_context(parameters))

    assert result.success is True
    assert calls[0]["depth_truncation"] == 2.0
    raw_output_ref = Path(result.details["raw_output_ref"])
    request_text = (raw_output_ref.parent / "request.json").read_text()
    assert '"depth_truncation": 2.0' in request_text


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), True, "2.0"])
def test_anyplace_handler_rejects_invalid_depth_truncation(
    tmp_path: Path,
    value: Any,
) -> None:
    parameters = _parameters(tmp_path)
    parameters["depth_truncation"] = value
    calls: list[dict[str, Any]] = []

    result = build_anyplace_handler(lambda request: calls.append(request) or _success_response())(
        _context(parameters)
    )

    assert result.success is False
    assert result.details["reason"] == "invalid_depth_truncation"
    assert calls == []


def test_anyplace_reuses_anygrasp_depth_cutoff_scale_and_restores_metres(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []
    parameters = _parameters(tmp_path)
    parameters["selected_grasp"]["source"]["depth_cutoff_factor"] = 1.5

    def predict(request: dict[str, Any]) -> dict[str, Any]:
        calls.append(request)
        response = _success_response()
        for index, candidate in enumerate(response["details"]["placement_candidates"]):
            candidate["object_placement_transform"]["transform_matrix"][0][3] = (
                float(index) / 1.5
            )
            pose = candidate["place_grasp_pose"]
            pose["translation_xyz"] = [
                (0.1 + index) / 1.5,
                0.2 / 1.5,
                0.3 / 1.5,
            ]
            pose["gripper_tip_position_xyz"] = [
                (0.13 + index) / 1.5,
                0.2 / 1.5,
                0.3 / 1.5,
            ]
            for key in ("depth", "width", "height"):
                pose[key] = float(pose[key]) / 1.5
        return response

    result = build_anyplace_handler(
        predict,
        output_root=tmp_path / "runs",
    )(_context(parameters))

    assert result.success is True
    assert calls[0]["intrinsics"]["scale"] == 1500.0
    assert calls[0]["selected_grasp"]["translation_xyz"] == pytest.approx(
        [0.1 / 1.5, 0.2 / 1.5, 0.3 / 1.5]
    )
    candidate = result.details["placement_candidates"][2]
    assert candidate["object_placement_transform"]["transform_matrix"][0][3] == (
        pytest.approx(2.0)
    )
    assert candidate["place_grasp_pose"]["translation_xyz"] == pytest.approx(
        [2.1, 0.2, 0.3]
    )
    assert candidate["place_grasp_pose"]["width"] == pytest.approx(0.06)
    assert result.details["metadata"]["depth_cutoff_factor"] == 1.5
    assert result.details["source"]["intrinsics"]["scale"] == 1000.0


def test_anyplace_handler_accepts_resolved_symlink_provenance(tmp_path: Path) -> None:
    parameters = _parameters(tmp_path)
    rgb_link = tmp_path / "rgb-link.png"
    rgb_link.symlink_to(Path(parameters["rgb"]).name)
    parameters["selected_grasp"]["source"]["rgb"] = str(rgb_link)
    parameters["placement_region_mask"]["source_image"] = str(rgb_link)
    handler = build_anyplace_handler(
        lambda request: _success_response(), output_root=tmp_path / "runs"
    )

    result = handler(_context(parameters))

    assert result.success is True


def test_anyplace_accepts_normalized_graspgenx_source_and_annotates_place_pose(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []
    parameters = _parameters(tmp_path)
    graspgenx_candidate = _candidate(
        id="graspgenx_000",
        rank=0,
        backend_index=3,
        source_model="graspgenx",
        gripper_name="franka_panda",
        candidate_source="diffusion",
        grasp_frame="graspnet",
        convention="p_camera = R @ p_grasp + t",
        transform_matrix=[
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ],
        model_native_grasp_pose={"must_not_reach_anyplace": True},
    )
    parameters["selected_grasp"] = {
        "candidate": graspgenx_candidate,
        "source": {
            "source_tool": "graspgenx",
            "mode": "targeted",
            "rgb": parameters["rgb"],
            "depth": parameters["depth"],
            "object_mask": parameters["object_mask"],
            "intrinsics": dict(INTRINSICS),
            "gripper_name": "franka_panda",
            "up_direction_camera": [0.0, 0.0, -1.0],
        },
    }

    result = build_anyplace_handler(
        lambda request: calls.append(request)
        or _success_response(source_grasp_id="graspgenx_000"),
        output_root=tmp_path / "runs",
    )(_context(parameters))

    assert result.success is True
    selected = calls[0]["selected_grasp"]
    assert selected["id"] == "graspgenx_000"
    assert selected["grasp_frame"] == "graspnet"
    assert selected["gripper_name"] == "franka_panda"
    assert selected["transform_matrix"][0][3] == pytest.approx(0.1)
    assert "model_native_grasp_pose" not in selected
    assert result.details["selected_grasp_source"]["source_tool"] == "graspgenx"
    assert result.details["selected_grasp_source"]["gripper_name"] == "franka_panda"
    for candidate in result.details["placement_candidates"]:
        assert candidate["source_grasp_id"] == "graspgenx_000"
        assert candidate["place_grasp_pose"]["source_tool"] == "graspgenx"
        assert candidate["place_grasp_pose"]["gripper_name"] == "franka_panda"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda source, candidate: source.pop("gripper_name"),
        lambda source, candidate: source.pop("up_direction_camera"),
        lambda source, candidate: candidate.update(gripper_name="other"),
    ],
)
def test_anyplace_rejects_incomplete_graspgenx_source(
    tmp_path: Path,
    mutation,
) -> None:
    parameters = _parameters(tmp_path)
    candidate = _candidate(gripper_name="franka_panda")
    source = parameters["selected_grasp"]["source"]
    source.update(
        source_tool="graspgenx",
        gripper_name="franka_panda",
        up_direction_camera=[0.0, 0.0, -1.0],
    )
    parameters["selected_grasp"]["candidate"] = candidate
    mutation(source, candidate)

    result = build_anyplace_handler(
        lambda request: pytest.fail("AnyPlace MCP must not be called")
    )(_context(parameters))

    assert result.success is False
    assert result.details["reason"] == "selected_grasp_requires_targeted_source"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda value, path: value["selected_grasp"]["source"].update(rgb=str(path)),
            "source_rgb_mismatch",
        ),
        (
            lambda value, path: value["selected_grasp"]["source"].update(depth=str(path)),
            "source_depth_mismatch",
        ),
        (
            lambda value, path: value["selected_grasp"]["source"].update(object_mask=str(path)),
            "source_object_mask_mismatch",
        ),
        (
            lambda value, path: value["selected_grasp"]["source"]["intrinsics"].update(fx=619.0),
            "source_intrinsics_mismatch",
        ),
        (
            lambda value, path: value["placement_region_mask"].update(source_image=str(path)),
            "placement_mask_source_mismatch",
        ),
    ],
)
def test_anyplace_handler_rejects_provenance_mismatch(
    tmp_path: Path,
    mutation,
    reason: str,
) -> None:
    parameters = _parameters(tmp_path)
    other = _write_image(tmp_path / "other.png")
    mutation(parameters, other)
    calls = []
    handler = build_anyplace_handler(lambda request: calls.append(request) or _success_response())

    result = handler(_context(parameters))

    assert result.success is False
    assert result.details["reason"] == reason
    assert calls == []


def test_anyplace_handler_requires_targeted_grasp_source(tmp_path: Path) -> None:
    parameters = _parameters(tmp_path)
    parameters["selected_grasp"]["source"]["mode"] = "scene"

    result = build_anyplace_handler(lambda request: _success_response())(_context(parameters))

    assert result.success is False
    assert result.details["reason"] == "selected_grasp_requires_targeted_source"


def test_anyplace_handler_preserves_structured_backend_failure(tmp_path: Path) -> None:
    handler = build_anyplace_handler(
        lambda request: {
            "success": False,
            "content": "AnyPlace placement prediction failed: object pointcloud too small.",
            "details": {
                "reason": "object_pointcloud_too_small",
                "metadata": {"raw_object_point_count": 100},
            },
        },
        output_root=tmp_path / "runs",
    )

    result = handler(_context(_parameters(tmp_path)))

    assert result.success is False
    assert result.details["reason"] == "object_pointcloud_too_small"
    assert result.details["metadata"] == {"raw_object_point_count": 100}
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    ("path_key", "reason"),
    [
        ("rgb", "rgb_not_found"),
        ("depth", "depth_not_found"),
        ("object_mask", "object_mask_not_found"),
        ("placement_region_mask", "placement_region_mask_not_found"),
    ],
)
def test_anyplace_handler_reports_missing_files(
    tmp_path: Path,
    path_key: str,
    reason: str,
) -> None:
    parameters = _parameters(tmp_path)
    missing = str(tmp_path / f"missing-{path_key}.png")
    if path_key == "placement_region_mask":
        parameters[path_key]["mask_ref"] = missing
    else:
        parameters[path_key] = missing
        source_key = path_key if path_key != "object_mask" else "object_mask"
        parameters["selected_grasp"]["source"][source_key] = missing
        if path_key == "rgb":
            parameters["placement_region_mask"]["source_image"] = missing

    result = build_anyplace_handler(lambda request: _success_response())(_context(parameters))

    assert result.success is False
    assert result.details["reason"] == reason


@pytest.mark.parametrize(
    "mutate_response",
    [
        lambda response: response["details"].update(candidate_count=4),
        lambda response: response["details"]["placement_candidates"][1].update(id="placement_000"),
        lambda response: response["details"]["placement_candidates"][0].update(
            source_grasp_id="other"
        ),
        lambda response: response["details"]["placement_candidates"][0][
            "object_placement_transform"
        ]["transform_matrix"].__setitem__(3, [0.0, 0.0, 0.0, 2.0]),
        lambda response: response["details"]["placement_candidates"][0][
            "place_grasp_pose"
        ].update(width=0.07),
        lambda response: response["details"]["placement_candidates"][0][
            "place_grasp_pose"
        ].update(
            rotation_matrix=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        ),
    ],
)
def test_anyplace_handler_rejects_entire_malformed_success(tmp_path: Path, mutate_response) -> None:
    response = _success_response()
    mutate_response(response)
    handler = build_anyplace_handler(lambda request: response, output_root=tmp_path / "runs")

    result = handler(_context(_parameters(tmp_path)))

    assert result.success is False
    assert result.details["reason"] == "inconsistent_placement_outputs"
    assert result.details["candidate_count"] == 0
    assert result.details["placement_candidates"] == []
    assert not (tmp_path / "runs").exists()
