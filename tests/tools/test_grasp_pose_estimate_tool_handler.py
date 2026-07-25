from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.tools.handlers import build_grasp_pose_estimate_handler
from agent.tools.registry import (
    ToolEffect,
    ToolExecutionContext,
    ToolResult,
    build_default_tool_registry,
)


INTRINSICS = {
    "fx": 100.0,
    "fy": 100.0,
    "cx": 8.0,
    "cy": 8.0,
    "scale": 1000.0,
}


def _parameters(tmp_path: Path, *, mode: str = "targeted") -> dict[str, Any]:
    rgb = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    mask = tmp_path / "mask.png"
    for path in (rgb, depth, mask):
        path.write_bytes(b"fixture")
    parameters: dict[str, Any] = {
        "mode": mode,
        "rgb": str(rgb),
        "depth": str(depth),
        "intrinsics": dict(INTRINSICS),
        "camera_frame_id": "agentview",
        "scene_epoch": 4,
        "hints": {"dense_sampling": True, "depth_cutoff_factor": 1.25},
    }
    if mode == "targeted":
        parameters["object_mask"] = {
            "mask_ref": str(mask),
            "source_image": str(rgb),
            "result_id": "sam3-result",
            "detection_id": "detection_000",
        }
    return parameters


def _context(parameters: dict[str, Any]) -> ToolExecutionContext:
    spec = build_default_tool_registry().get("grasp_pose_estimate")
    return ToolExecutionContext(
        name="grasp_pose_estimate",
        spec=spec,
        parameters=parameters,
        metadata={"session_id": "session-a"},
    )


def _candidate(candidate_id: str, *, score: float) -> dict[str, Any]:
    return {
        "id": candidate_id,
        "frame": "camera",
        "camera_frame": "opencv",
        "grasp_frame": "graspnet",
        "score": score,
        "translation_xyz": [0.1, 0.2, 0.3],
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "gripper_tip_position_xyz": [0.13, 0.2, 0.3],
        "depth": 0.03,
        "width": 0.06,
        "height": 0.03,
    }


def _success(*candidates: dict[str, Any], source: dict[str, Any] | None = None) -> ToolResult:
    return ToolResult(
        True,
        details={
            "candidate_count": len(candidates),
            "grasp_candidates": list(candidates),
            "source": dict(source or {}),
            "artifacts": [],
        },
    )


def _failure(reason: str) -> ToolResult:
    return ToolResult(
        False,
        content=f"backend failed: {reason}",
        details={"reason": reason, "candidate_count": 0, "grasp_candidates": []},
    )


def test_tool_spec_exposes_only_backend_neutral_inputs() -> None:
    spec = build_default_tool_registry().get("grasp_pose_estimate")

    assert spec.effect == ToolEffect.PLANNING
    assert set(spec.parameters) == {
        "mode",
        "rgb",
        "depth",
        "object_mask",
        "intrinsics",
        "camera_frame_id",
        "scene_epoch",
        "hints",
    }
    assert "AnyGrasp" in spec.description
    assert "Contact-GraspNet" in spec.description
    assert "GraspGenX" in spec.description


def test_falls_back_and_normalizes_backend_provenance(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def anygrasp(context: ToolExecutionContext) -> ToolResult:
        calls.append((context.name, dict(context.parameters)))
        return _failure("no_grasp_candidates")

    def contact(context: ToolExecutionContext) -> ToolResult:
        calls.append((context.name, dict(context.parameters)))
        return _success(
            _candidate("contact-native-1", score=0.7),
            _candidate("contact-native-0", score=0.9),
        )

    handler = build_grasp_pose_estimate_handler(
        {"anygrasp": anygrasp, "contact_graspnet": contact}
    )

    result = handler(_context(_parameters(tmp_path)))

    assert result.success is True
    assert [name for name, _ in calls] == ["anygrasp", "contact_graspnet"]
    assert calls[0][1]["target_mask"].endswith("mask.png")
    assert calls[0][1]["dense_grasp"] is True
    assert calls[1][1]["object_mask"]["source_image"].endswith("rgb.png")
    assert result.details["selected_backend"] == "contact_graspnet"
    assert [attempt["status"] for attempt in result.details["backend_attempts"]] == [
        "failed",
        "success",
    ]
    candidates = result.details["grasp_candidates"]
    assert [candidate["score"] for candidate in candidates] == [0.9, 0.7]
    assert [candidate["backend_candidate_id"] for candidate in candidates] == [
        "contact-native-0",
        "contact-native-1",
    ]
    assert candidates[0]["source_tool"] == "grasp_pose_estimate"
    assert candidates[0]["source_backend"] == "contact_graspnet"
    assert result.details["source"]["scene_epoch"] == 4
    assert result.details["source"]["camera_frame_id"] == "agentview"


def test_host_excluded_backend_is_skipped(tmp_path: Path) -> None:
    calls: list[str] = []

    def backend(context: ToolExecutionContext) -> ToolResult:
        calls.append(context.name)
        return _success(_candidate(f"{context.name}-0", score=0.8))

    parameters = _parameters(tmp_path)
    parameters["hints"]["excluded_backends"] = ["anygrasp"]
    result = build_grasp_pose_estimate_handler(
        {"anygrasp": backend, "contact_graspnet": backend}
    )(_context(parameters))

    assert result.success is True
    assert calls == ["contact_graspnet"]
    assert result.details["selected_backend"] == "contact_graspnet"
    assert result.details["backend_attempts"][0] == {
        "backend": "anygrasp",
        "status": "skipped",
        "reason": "excluded_by_host_fallback",
        "candidate_count": 0,
    }


def test_scene_mode_uses_only_scene_compatible_backend(tmp_path: Path) -> None:
    called: list[str] = []

    def backend(context: ToolExecutionContext) -> ToolResult:
        called.append(context.name)
        return _success(_candidate("scene-0", score=0.5))

    handler = build_grasp_pose_estimate_handler(
        {
            "anygrasp": backend,
            "contact_graspnet": backend,
            "graspgenx": backend,
        },
        backend_order=("contact_graspnet", "graspgenx", "anygrasp"),
    )

    result = handler(_context(_parameters(tmp_path, mode="scene")))

    assert result.success is True
    assert called == ["anygrasp"]
    assert [attempt["status"] for attempt in result.details["backend_attempts"]] == [
        "ineligible",
        "ineligible",
        "success",
    ]


def test_enhanced_candidate_depth_uses_anygrasp_without_collision_filter(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def backend(context: ToolExecutionContext) -> ToolResult:
        calls.append((context.name, dict(context.parameters)))
        return _success(_candidate("candidate-0", score=0.8))

    parameters = _parameters(tmp_path)
    parameters["hints"]["depth_enhancement"] = {
        "candidate_generation_only": True,
        "requires_sensor_safety_check": True,
        "safety_depth_png": str(tmp_path / "safety.png"),
    }
    result = build_grasp_pose_estimate_handler(
        {
            "contact_graspnet": backend,
            "graspgenx": backend,
            "anygrasp": backend,
        },
        backend_order=("contact_graspnet", "graspgenx", "anygrasp"),
    )(_context(parameters))

    assert result.success is True
    assert [attempt["status"] for attempt in result.details["backend_attempts"]] == [
        "ineligible",
        "ineligible",
        "success",
    ]
    assert len(calls) == 1
    assert calls[0][0] == "anygrasp"
    assert calls[0][1]["collision_detection"] is False
    assert result.details["source"]["requires_sensor_safety_check"] is True


def test_invalid_common_input_fails_before_backend_dispatch(tmp_path: Path) -> None:
    parameters = _parameters(tmp_path)
    parameters["object_mask"]["source_image"] = str(tmp_path / "other.png")
    called = False

    def backend(_context: ToolExecutionContext) -> ToolResult:
        nonlocal called
        called = True
        return _success(_candidate("unused", score=1.0))

    result = build_grasp_pose_estimate_handler({"anygrasp": backend})(
        _context(parameters)
    )

    assert result.success is False
    assert result.details["reason"] == "object_mask_source_mismatch"
    assert result.details["retryable"] is False
    assert result.details["backend_attempts"] == []
    assert called is False


def test_non_fallback_backend_error_stops_dispatch(tmp_path: Path) -> None:
    called: list[str] = []

    def invalid(context: ToolExecutionContext) -> ToolResult:
        called.append(context.name)
        return _failure("invalid_backend_request")

    def unused(context: ToolExecutionContext) -> ToolResult:
        called.append(context.name)
        return _success(_candidate("unused", score=1.0))

    handler = build_grasp_pose_estimate_handler(
        {"anygrasp": invalid, "contact_graspnet": unused}
    )

    result = handler(_context(_parameters(tmp_path)))

    assert result.success is False
    assert result.details["reason"] == "invalid_backend_request"
    assert result.details["retryable"] is False
    assert called == ["anygrasp"]


def test_model_load_failure_falls_back_to_next_backend(tmp_path: Path) -> None:
    calls: list[str] = []

    def unavailable(context: ToolExecutionContext) -> ToolResult:
        calls.append(context.name)
        return _failure("model_load_failed")

    def valid(context: ToolExecutionContext) -> ToolResult:
        calls.append(context.name)
        return _success(_candidate("contact-0", score=0.8))

    result = build_grasp_pose_estimate_handler(
        {"anygrasp": unavailable, "contact_graspnet": valid}
    )(_context(_parameters(tmp_path)))

    assert result.success is True
    assert result.details["selected_backend"] == "contact_graspnet"
    assert calls == ["anygrasp", "contact_graspnet"]


def test_malformed_success_falls_back_to_next_backend(tmp_path: Path) -> None:
    calls: list[str] = []

    def malformed(context: ToolExecutionContext) -> ToolResult:
        calls.append(context.name)
        return _success({"score": 0.9})

    def valid(context: ToolExecutionContext) -> ToolResult:
        calls.append(context.name)
        return _success(_candidate("valid-0", score=0.8))

    result = build_grasp_pose_estimate_handler(
        {"anygrasp": malformed, "contact_graspnet": valid}
    )(_context(_parameters(tmp_path)))

    assert result.success is True
    assert result.details["selected_backend"] == "contact_graspnet"
    assert calls == ["anygrasp", "contact_graspnet"]
    assert result.details["backend_attempts"][0]["reason"] == (
        "inconsistent_grasp_outputs"
    )
