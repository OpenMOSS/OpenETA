from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from agent.tools import handlers as handlers_module
from agent.tools.handlers import (
    DEFAULT_CONTACT_GRASPNET_OUTPUT_ROOT,
    build_contact_graspnet_handler,
    build_sse_contact_graspnet_mcp_predictor,
    build_stdio_contact_graspnet_mcp_predictor,
    bind_dummy_tool_handlers,
)
from agent.tools.registry import (
    ToolEffect,
    ToolExecutionContext,
    build_default_tool_registry,
)


IMAGE_BYTES = b"openeta-contact-graspnet-test-image"
INTRINSICS = {"fx": 618.0, "fy": 618.0, "cx": 256.0, "cy": 256.0, "scale": 1000.0}


def _write_image(path: Path) -> Path:
    path.write_bytes(IMAGE_BYTES)
    return path


def _parameters(tmp_path: Path) -> dict[str, Any]:
    rgb = _write_image(tmp_path / "rgb.png")
    depth = _write_image(tmp_path / "depth.png")
    mask = _write_image(tmp_path / "mask.png")
    return {
        "rgb": str(rgb),
        "depth": str(depth),
        "object_mask": {
            "type": "segmentation_mask",
            "mask_ref": str(mask),
            "source_image": str(rgb),
            "label": "bottle",
            "score": 0.9,
        },
        "intrinsics": dict(INTRINSICS),
    }


def _candidate(index: int = 0, *, score: float = 0.8, **overrides: Any) -> dict[str, Any]:
    translation = [0.1 + index * 0.01, 0.2, 0.3]
    value = {
        "id": f"grasp_{index:03d}",
        "frame": "camera",
        "camera_frame": "opencv",
        "grasp_frame": "graspnet",
        "source_model": "contact_graspnet",
        "gripper_model": "panda",
        "score": score,
        "translation_xyz": translation,
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "gripper_depth": 0.1034,
        "width": 0.04,
        "gripper_tip_position_xyz": [translation[0] + 0.1034, 0.2, 0.3],
        "contact_point_xyz": [translation[0] + 0.1034, 0.18, 0.3],
    }
    value.update(overrides)
    return value


def _success_response(candidates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    values = candidates if candidates is not None else [_candidate()]
    return {
        "success": True,
        "content": "Contact-GraspNet grasp prediction completed.",
        "details": {
            "tool": "contact_graspnet",
            "backend": "contact_graspnet_mcp",
            "model": "contact_graspnet_pytorch_unofficial",
            "mode": "targeted",
            "frame": "camera",
            "camera_frame": "opencv",
            "grasp_frame": "graspnet",
            "candidate_count": len(values),
            "grasp_candidates": values,
            "artifacts": [],
            "metadata": {
                "max_gripper_width": 0.08,
                "model_candidate_count": len(values),
            },
        },
    }


def _context(
    parameters: dict[str, Any],
    *,
    session_id: str = "",
) -> ToolExecutionContext:
    spec = build_default_tool_registry().get("contact_graspnet")
    return ToolExecutionContext(
        name="contact_graspnet",
        spec=spec,
        parameters=parameters,
        metadata={"session_id": session_id} if session_id else {},
    )


def test_contact_graspnet_spec_is_visible_without_dummy_handler() -> None:
    tools = bind_dummy_tool_handlers(build_default_tool_registry())
    spec = tools.get("contact_graspnet")

    assert spec.category == "manipulation"
    assert spec.effect == ToolEffect.PLANNING
    assert set(spec.parameters) == {"rgb", "depth", "object_mask", "intrinsics"}
    assert tools.can_execute("contact_graspnet") is False


def test_contact_graspnet_default_root_uses_repo_tmp_layout() -> None:
    assert DEFAULT_CONTACT_GRASPNET_OUTPUT_ROOT == (
        Path("tmp") / "tool_result" / "contact_graspnet"
    )


def test_handler_sends_no_rgb_and_materializes_success(tmp_path: Path) -> None:
    parameters = _parameters(tmp_path)
    calls: list[dict[str, Any]] = []

    def predict(request: dict[str, Any]) -> dict[str, Any]:
        calls.append(request)
        return _success_response([_candidate(0, score=0.9), _candidate(1, score=0.8)])

    output_root = tmp_path / "results"
    result = build_contact_graspnet_handler(predict, output_root=output_root)(
        _context(parameters, session_id="contact-session")
    )

    assert result.success is True
    assert set(calls[0]) == {"depth", "object_mask", "intrinsics"}
    assert calls[0]["intrinsics"] == INTRINSICS
    assert base64.b64decode(calls[0]["depth"]["base64"]) == IMAGE_BYTES
    assert base64.b64decode(calls[0]["object_mask"]["base64"]) == IMAGE_BYTES
    assert result.details["candidate_count"] == 2
    assert result.details["source"] == {
        "mode": "targeted",
        "rgb": parameters["rgb"],
        "depth": parameters["depth"],
        "object_mask": parameters["object_mask"]["mask_ref"],
        "intrinsics": INTRINSICS,
    }
    assert result.details["artifacts"] == []

    request_ref = Path(result.details["request_ref"])
    raw_ref = Path(result.details["raw_output_ref"])
    tool_result_ref = Path(result.details["tool_result_ref"])
    assert request_ref.is_file() and raw_ref.is_file() and tool_result_ref.is_file()
    assert request_ref.relative_to(output_root).parts[0] == "contact-session"
    saved_request = json.loads(request_ref.read_text())
    assert saved_request["object_mask"]["label"] == "bottle"
    for path in (request_ref, raw_ref, tool_result_ref):
        text = path.read_text()
        assert "base64" not in text
        assert "point_cloud" not in text


def test_handler_accepts_resolved_symlink_provenance(tmp_path: Path) -> None:
    parameters = _parameters(tmp_path)
    alias = tmp_path / "rgb-alias.png"
    alias.symlink_to(Path(parameters["rgb"]))
    parameters["object_mask"]["source_image"] = str(alias)

    result = build_contact_graspnet_handler(
        lambda _request: _success_response(),
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
                source_image=str(_write_image(root / "other.png"))
            ),
            "object_mask_source_mismatch",
        ),
    ],
)
def test_handler_persists_local_validation_failures(
    tmp_path: Path,
    mutate,
    reason: str,
) -> None:
    parameters = _parameters(tmp_path)
    mutate(parameters, tmp_path)
    result = build_contact_graspnet_handler(
        lambda _request: pytest.fail("MCP must not be called"),
        output_root=tmp_path / "results",
    )(_context(parameters))

    assert result.success is False
    assert result.details["reason"] == reason
    assert result.details["candidate_count"] == 0
    assert Path(result.details["request_ref"]).is_file()
    assert result.details["raw_output_ref"] is None
    saved_result = json.loads(Path(result.details["tool_result_ref"]).read_text())
    assert saved_result["details"]["reason"] == reason


def test_handler_persists_mcp_failure_and_scrubs_transport_payloads(tmp_path: Path) -> None:
    response = {
        "success": False,
        "content": "Contact-GraspNet grasp prediction failed: model_load_failed.",
        "details": {
            "reason": "model_load_failed",
            "metadata": {
                "checkpoint_dir": "/private/checkpoint",
                "base64": "secret",
                "point_cloud": [[1, 2, 3]],
                "error_type": "RuntimeError",
            },
        },
    }
    result = build_contact_graspnet_handler(
        lambda _request: response,
        output_root=tmp_path / "results",
    )(_context(_parameters(tmp_path)))

    assert result.success is False
    assert result.details["reason"] == "model_load_failed"
    raw_text = Path(result.details["raw_output_ref"]).read_text()
    assert "checkpoint_dir" not in raw_text
    assert "base64" not in raw_text
    assert "point_cloud" not in raw_text
    assert "error_type" in raw_text


def test_handler_persists_mcp_transport_exception_without_raw_response(
    tmp_path: Path,
) -> None:
    def fail(_request):
        raise TimeoutError("private transport detail")

    result = build_contact_graspnet_handler(
        fail,
        output_root=tmp_path / "results",
    )(_context(_parameters(tmp_path)))

    assert result.success is False
    assert result.details["reason"] == "mcp_call_failed"
    assert result.details["metadata"] == {"error_type": "TimeoutError"}
    assert result.details["raw_output_ref"] is None
    assert Path(result.details["request_ref"]).is_file()
    assert Path(result.details["tool_result_ref"]).is_file()
    assert "private transport detail" not in Path(
        result.details["tool_result_ref"]
    ).read_text()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda response: response["details"].update(candidate_count=3),
        lambda response: response["details"].update(frame="world"),
        lambda response: response["details"]["metadata"].pop("max_gripper_width"),
        lambda response: response["details"]["grasp_candidates"][1].update(id="grasp_000"),
        lambda response: response["details"]["grasp_candidates"][1].update(score=0.95),
        lambda response: response["details"]["grasp_candidates"][1].update(
            rotation_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, -1]]
        ),
        lambda response: response["details"]["grasp_candidates"][1].update(
            gripper_depth=0.03
        ),
        lambda response: response["details"]["grasp_candidates"][1].update(width=0.2),
        lambda response: response["details"]["grasp_candidates"][1].update(
            gripper_tip_position_xyz=[0, 0, 0]
        ),
    ],
)
def test_handler_rejects_any_inconsistent_candidate_atomically(
    tmp_path: Path,
    mutate,
) -> None:
    response = _success_response([_candidate(0, score=0.9), _candidate(1, score=0.8)])
    mutate(response)
    result = build_contact_graspnet_handler(
        lambda _request: response,
        output_root=tmp_path / "results",
    )(_context(_parameters(tmp_path)))

    assert result.success is False
    assert result.details["reason"] == "inconsistent_grasp_outputs"
    assert result.details["candidate_count"] == 0
    assert result.details["grasp_candidates"] == []
    assert Path(result.details["raw_output_ref"]).is_file()


def test_runtime_wraps_contact_source_under_outputs(tmp_path: Path) -> None:
    tools = build_default_tool_registry()
    tools.bind_handler(
        "contact_graspnet",
        build_contact_graspnet_handler(
            lambda _request: _success_response(),
            output_root=tmp_path / "results",
        ),
    )

    result = tools.call("contact_graspnet", _parameters(tmp_path))

    assert result.success is True
    assert result.details["outputs"]["source"]["mode"] == "targeted"
    assert result.details["outputs"]["grasp_candidates"][0]["source_model"] == (
        "contact_graspnet"
    )


def test_contact_graspnet_stdio_builder_uses_predict_tool_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    async def fake_call(**kwargs):
        calls.append(kwargs)
        return {"success": True}

    monkeypatch.setattr(handlers_module, "_call_stdio_mcp_tool", fake_call)
    predictor = build_stdio_contact_graspnet_mcp_predictor(
        command="python",
        args=["server.py"],
    )

    assert predictor({"depth": {}}) == {"success": True}
    assert calls[0]["tool_name"] == "predict_grasps"
    assert calls[0]["timeout_seconds"] == 600.0


def test_contact_graspnet_sse_builder_uses_predict_tool_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakeTransport:
        def __init__(self, url: str) -> None:
            assert url == "http://contact.example/sse"

        def call_tool(self, name, arguments, *, timeout_s=None):
            calls.append((name, arguments, timeout_s))
            return {"success": True}

    monkeypatch.setattr(handlers_module, "SseSimulatorMcpTransport", FakeTransport)
    predictor = build_sse_contact_graspnet_mcp_predictor(
        url="http://contact.example/sse"
    )

    assert predictor({"depth": {}}) == {"success": True}
    assert calls == [("predict_grasps", {"depth": {}}, 600.0)]
