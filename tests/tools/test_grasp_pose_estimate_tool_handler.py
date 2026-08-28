from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

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
            "quality": {
                "status": "usable",
                "area_fraction": 0.008,
            },
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
        "bundle_id",
        "backend_preference",
        "mode",
        "rgb",
        "depth",
        "object_mask",
        "intrinsics",
        "camera_frame_id",
        "scene_epoch",
        "hints",
    }
    backend_description = str(spec.parameters["backend_preference"])
    assert "anygrasp" in backend_description
    assert "graspgenx" in backend_description
    assert "contact" not in backend_description.lower()


def test_falls_back_and_normalizes_backend_provenance(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def anygrasp(context: ToolExecutionContext) -> ToolResult:
        calls.append((context.name, dict(context.parameters)))
        return _failure("no_grasp_candidates")

    def graspgenx(context: ToolExecutionContext) -> ToolResult:
        calls.append((context.name, dict(context.parameters)))
        return _success(
            _candidate("contact-native-1", score=0.7),
            _candidate("contact-native-0", score=0.9),
        )

    handler = build_grasp_pose_estimate_handler(
        {"anygrasp": anygrasp, "graspgenx": graspgenx}
    )

    result = handler(_context(_parameters(tmp_path)))

    assert result.success is True
    assert [name for name, _ in calls] == ["anygrasp", "graspgenx"]
    assert calls[0][1]["target_mask"].endswith("mask.png")
    assert calls[0][1]["dense_grasp"] is True
    assert calls[0][1]["depth_cutoff_factor"] == 1.25
    assert calls[1][1]["object_mask"]["source_image"].endswith("rgb.png")
    assert result.details["selected_backend"] == "graspgenx"
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
    assert candidates[0]["source_backend"] == "graspgenx"
    assert candidates[0]["execution_reference_point"] == "translation_xyz"
    assert candidates[0]["execution_contact_center_xyz"] == [0.1, 0.2, 0.3]
    assert result.details["source"]["scene_epoch"] == 4
    assert result.details["source"]["camera_frame_id"] == "agentview"
    assert result.details["target_mask_quality"]["area_fraction"] == 0.008
    assert result.details["source"]["target_mask_quality"]["status"] == "usable"


def test_calibration_can_explicitly_select_gripper_tip_reference(tmp_path: Path) -> None:
    parameters = _parameters(tmp_path)
    parameters["hints"]["execution_reference_point"] = "gripper_tip_position_xyz"
    result = build_grasp_pose_estimate_handler(
        {"anygrasp": lambda _context: _success(_candidate("grasp-0", score=0.9))},
        backend_order=("anygrasp",),
    )(_context(parameters))

    assert result.success is True
    candidate = result.details["grasp_candidates"][0]
    assert candidate["execution_reference_point"] == "gripper_tip_position_xyz"
    assert candidate["execution_contact_center_xyz"] == [0.13, 0.2, 0.3]


def test_graspgenx_receives_depth_cutoff_factor_from_unified_hints(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    def graspgenx(context: ToolExecutionContext) -> ToolResult:
        calls.append(dict(context.parameters))
        return _success(_candidate("graspgenx-native-0", score=0.8))

    result = build_grasp_pose_estimate_handler(
        {"graspgenx": graspgenx},
        backend_order=("graspgenx",),
        graspgenx_gripper_name="franka_panda",
        graspgenx_up_direction_camera=(0.0, 0.0, -1.0),
    )(_context(_parameters(tmp_path)))

    assert result.success is True
    assert calls[0]["depth_cutoff_factor"] == 1.25
    assert calls[0]["intrinsics"] == INTRINSICS


def test_graspgenx_prefers_packet_owned_up_direction_hint(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def graspgenx(context: ToolExecutionContext) -> ToolResult:
        calls.append(dict(context.parameters))
        return _success(_candidate("graspgenx-native-0", score=0.8))

    parameters = _parameters(tmp_path)
    parameters["hints"]["up_direction_camera"] = [0.0, -1.0, 0.0]
    result = build_grasp_pose_estimate_handler(
        {"graspgenx": graspgenx},
        backend_order=("graspgenx",),
        graspgenx_gripper_name="franka_panda",
        graspgenx_up_direction_camera=(0.0, 0.0, -1.0),
    )(_context(parameters))

    assert result.success is True
    assert calls[0]["up_direction_camera"] == [0.0, -1.0, 0.0]


def test_advisor_runs_after_final_filtering_and_reranking(tmp_path: Path) -> None:
    parameters = _parameters(tmp_path)
    Image.new("RGB", (16, 16), (50, 60, 70)).save(parameters["rgb"])
    Image.new("I;16", (16, 16), 500).save(parameters["depth"])
    Image.new("L", (16, 16), 255).save(parameters["object_mask"]["mask_ref"])
    parameters["hints"]["max_gripper_width_m"] = 0.08
    seen = []

    class Advisor:
        def advise(self, selection_bundle, *, task):
            seen.append((selection_bundle, task))
            ids = [item["candidate_id"] for item in selection_bundle["candidates"]]
            return {
                "schema_version": "openeta.grasp_selection_advice.v1",
                "status": "completed",
                "decision": "recommend",
                "recommended_candidate_id": ids[1],
                "alternatives": [ids[0]],
                "confidence": 0.75,
                "reasons": ["candidate two is visually centered"],
                "rejected": {},
                "uncertainties": [],
                "bundle_id": selection_bundle["bundle_id"],
                "advisor_role": "read_only_grasp_pose_advisor",
            }

    def backend(_context: ToolExecutionContext) -> ToolResult:
        too_wide = _candidate("wide", score=1.0)
        too_wide["width"] = 0.1
        return _success(
            too_wide,
            _candidate("lower", score=0.5),
            _candidate("higher", score=0.8),
        )

    result = build_grasp_pose_estimate_handler(
        {"anygrasp": backend},
        backend_order=("anygrasp",),
        advisor=Advisor(),
        selection_output_root=tmp_path / "selection",
    )(_context(parameters))

    assert result.success is True
    assert len(seen) == 1
    bundle = seen[0][0]
    assert seen[0][1] == ""
    assert [item["backend_candidate_id"] for item in bundle["candidates"]] == [
        "higher",
        "lower",
    ]
    assert result.details["grasp_selection_advice"]["recommended_candidate_id"].endswith(
        "-001"
    )
    assert Path(result.details["grasp_selection_bundle"]["overview_ref"]).is_file()
    assert "must still choose" in result.content


def test_advisor_abstention_is_prominent_in_tool_feedback(tmp_path: Path) -> None:
    parameters = _parameters(tmp_path)
    Image.new("RGB", (16, 16), (50, 60, 70)).save(parameters["rgb"])
    Image.new("I;16", (16, 16), 500).save(parameters["depth"])
    Image.new("L", (16, 16), 255).save(parameters["object_mask"]["mask_ref"])

    class Advisor:
        def advise(self, selection_bundle, *, task):
            return {
                "schema_version": "openeta.grasp_selection_advice.v1",
                "status": "completed",
                "decision": "abstain",
                "recommended_candidate_id": "",
                "alternatives": [],
                "confidence": 0.4,
                "reasons": ["all contacts lie on the tapered top shoulder"],
                "rejected": {},
                "uncertainties": [],
                "bundle_id": selection_bundle["bundle_id"],
                "advisor_role": "read_only_grasp_pose_advisor",
            }

    result = build_grasp_pose_estimate_handler(
        {
            "anygrasp": lambda _context: _success(
                _candidate("one", score=0.8),
                _candidate("two", score=0.7),
            )
        },
        backend_order=("anygrasp",),
        advisor=Advisor(),
        selection_output_root=tmp_path / "selection",
    )(_context(parameters))

    assert result.success is True
    assert "advisor abstained" in result.content
    assert "all contacts lie on the tapered top shoulder" in result.content
    assert "Do not silently choose rank 0" in result.content


def test_estimator_surfaces_compatible_strategy_without_applying_it(tmp_path: Path) -> None:
    parameters = _parameters(tmp_path)
    parameters["target_geometry_family"] = "boxed_item"
    result = build_grasp_pose_estimate_handler(
        {"anygrasp": lambda _context: _success(_candidate("one", score=0.8))},
        backend_order=("anygrasp",),
        grasp_strategy_root=Path(__file__).parents[2] / "agent" / "strategies" / "grasp",
        grasp_calibration_id="graspnet-eef-panda-p8",
    )(_context(parameters))

    assert result.success is True
    options = result.details["explicit_grasp_strategy_options"]
    assert options[0]["strategy_id"] == "top-down-vertical-panda-p8"
    assert options[0]["activation"] == "explicit_agent_choice_required"
    assert "evidence_summary" not in options[0]
    assert "milk" not in result.content.lower()
    assert "None was applied" in result.content
    assert "compare the raw recommendation" in result.content
    assert "Task-specific rollout provenance" in result.content
    assert result.details["grasp_candidates"][0].get("strategy_id") is None


def test_host_excluded_backend_is_skipped(tmp_path: Path) -> None:
    calls: list[str] = []

    def backend(context: ToolExecutionContext) -> ToolResult:
        calls.append(context.name)
        return _success(_candidate(f"{context.name}-0", score=0.8))

    parameters = _parameters(tmp_path)
    parameters["hints"]["excluded_backends"] = ["anygrasp"]
    result = build_grasp_pose_estimate_handler(
        {"anygrasp": backend, "graspgenx": backend}
    )(_context(parameters))

    assert result.success is True
    assert calls == ["graspgenx"]
    assert result.details["selected_backend"] == "graspgenx"
    assert result.details["backend_attempts"][0] == {
        "backend": "anygrasp",
        "status": "skipped",
        "reason": "excluded_by_host_fallback",
        "candidate_count": 0,
    }


def test_agent_backend_preference_reorders_facade_and_keeps_fallback(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def backend(context: ToolExecutionContext) -> ToolResult:
        calls.append(context.name)
        if context.name == "graspgenx":
            return _failure("no_grasp_candidates")
        return _success(_candidate(f"{context.name}-0", score=0.8))

    parameters = _parameters(tmp_path)
    parameters["backend_preference"] = ["graspgenx"]
    result = build_grasp_pose_estimate_handler(
        {
            "anygrasp": backend,
            "graspgenx": backend,
        }
    )(_context(parameters))

    assert result.success is True
    assert calls == ["graspgenx", "anygrasp"]
    assert result.details["selected_backend"] == "anygrasp"
    candidate = result.details["grasp_candidates"][0]
    assert candidate["execution_reference_point"] == "translation_xyz"
    assert candidate["execution_contact_center_xyz"] == [0.1, 0.2, 0.3]
    assert result.details["backend_policy"] == {
        "schema_version": "openeta.grasp_backend_policy.v1",
        "agent_control_scope": "attempt_order_only",
        "requested_preference": ["graspgenx"],
        "configured_order": ["anygrasp", "graspgenx"],
        "effective_order": ["graspgenx", "anygrasp"],
        "unconfigured_requested_backends": [],
        "remaining_configured_backends_retained": True,
        "fallback_policy": "existing_structured_failure_policy",
        "selected_backend": "anygrasp",
    }
    assert "remaining configured backends stayed available" in result.content


@pytest.mark.parametrize(
    "preference",
    [[], ["unknown"], ["anygrasp", "anygrasp"], "graspgenx"],
)
def test_invalid_agent_backend_preference_fails_before_backend_call(
    tmp_path: Path,
    preference: object,
) -> None:
    called = False

    def backend(_context: ToolExecutionContext) -> ToolResult:
        nonlocal called
        called = True
        return _success(_candidate("candidate-0", score=0.8))

    parameters = _parameters(tmp_path)
    parameters["backend_preference"] = preference
    result = build_grasp_pose_estimate_handler({"anygrasp": backend})(
        _context(parameters)
    )

    assert result.success is False
    assert result.details["reason"] == "invalid_backend_preference"
    assert result.details["backend_attempts"] == []
    assert called is False
    assert "Invalid backend_preference" in result.content


def test_scene_mode_uses_only_scene_compatible_backend(tmp_path: Path) -> None:
    called: list[str] = []

    def backend(context: ToolExecutionContext) -> ToolResult:
        called.append(context.name)
        return _success(_candidate("scene-0", score=0.5))

    handler = build_grasp_pose_estimate_handler(
        {
            "anygrasp": backend,
            "graspgenx": backend,
        },
        backend_order=("graspgenx", "anygrasp"),
    )

    result = handler(_context(_parameters(tmp_path, mode="scene")))

    assert result.success is True
    assert called == ["anygrasp"]
    assert [attempt["status"] for attempt in result.details["backend_attempts"]] == [
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
            "graspgenx": backend,
            "anygrasp": backend,
        },
        backend_order=("graspgenx", "anygrasp"),
    )(_context(parameters))

    assert result.success is True
    assert [attempt["status"] for attempt in result.details["backend_attempts"]] == [
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
        {"anygrasp": invalid, "graspgenx": unused}
    )

    result = handler(_context(_parameters(tmp_path)))

    assert result.success is False
    assert result.details["reason"] == "invalid_backend_request"
    assert result.details["retryable"] is False
    assert called == ["anygrasp"]


def test_backend_depth_failure_keeps_actionable_diagnostics(tmp_path: Path) -> None:
    def outside_depth_range(_context: ToolExecutionContext) -> ToolResult:
        return ToolResult(
            False,
            content="target mask is beyond the service cutoff",
            details={
                "reason": "target_mask_outside_depth_range",
                "metadata": {
                    "depth_truncation": 1.0,
                    "target_mask_pixel_count": 2265,
                    "target_depth_min_m": 1.126,
                    "target_depth_max_m": 1.22,
                    "suggested_depth_cutoff_factor": 1.356,
                },
            },
        )

    result = build_grasp_pose_estimate_handler({"anygrasp": outside_depth_range})(
        _context(_parameters(tmp_path))
    )

    assert result.success is False
    attempt = result.details["backend_attempts"][0]
    assert attempt["reason"] == "target_mask_outside_depth_range"
    assert attempt["diagnostics"] == {
        "depth_truncation": 1.0,
        "target_mask_pixel_count": 2265,
        "target_depth_min_m": 1.126,
        "target_depth_max_m": 1.22,
        "suggested_depth_cutoff_factor": 1.356,
    }


def test_host_width_limit_removes_infeasible_candidates_from_main_queue(
    tmp_path: Path,
) -> None:
    def backend(_context: ToolExecutionContext) -> ToolResult:
        too_wide = _candidate("wide", score=0.99)
        too_wide["width"] = 0.094
        feasible = _candidate("feasible", score=0.7)
        feasible["width"] = 0.06
        return _success(too_wide, feasible)

    parameters = _parameters(tmp_path)
    parameters["hints"]["max_gripper_width_m"] = 0.08
    result = build_grasp_pose_estimate_handler({"anygrasp": backend})(
        _context(parameters)
    )

    assert result.success is True
    assert result.details["raw_candidate_count"] == 2
    assert result.details["candidate_count"] == 1
    assert result.details["grasp_candidates"][0]["backend_candidate_id"] == "feasible"
    assert result.details["rejected_candidates"] == [
        {
            "backend_candidate_id": "wide",
            "backend_index": 0,
            "score": 0.99,
            "width_m": 0.094,
            "reason": "exceeds_physical_gripper_width",
            "max_gripper_width_m": 0.08,
        }
    ]


def test_host_reports_backend_gripper_geometry_mismatch(tmp_path: Path) -> None:
    def backend(_context: ToolExecutionContext) -> ToolResult:
        too_wide = _candidate("wide", score=0.99)
        too_wide["width"] = 0.094
        feasible = _candidate("feasible", score=0.7)
        return ToolResult(
            True,
            details={
                "candidate_count": 2,
                "grasp_candidates": [too_wide, feasible],
                "metadata": {"max_gripper_width": 0.1},
                "artifacts": [],
            },
        )

    parameters = _parameters(tmp_path)
    parameters["hints"]["max_gripper_width_m"] = 0.08
    result = build_grasp_pose_estimate_handler({"anygrasp": backend})(
        _context(parameters)
    )

    assert result.success is False
    assert result.details["reason"] == "no_compatible_backend"
    attempt = result.details["backend_attempts"][0]
    assert attempt["reason"] == "backend_gripper_width_mismatch"
    assert attempt["diagnostics"]["backend_diagnostics"][0] == {
        "code": "grasp_pose_estimate_failed",
        "reason": "backend_gripper_width_mismatch",
        "retryable": False,
        "backend": "anygrasp",
        "backend_max_gripper_width_m": 0.1,
        "physical_max_gripper_width_m": 0.08,
        "requires_redeployment": True,
    }
    assert "Redeploy AnyGrasp" in result.content


def test_anygrasp_width_mismatch_falls_back_to_compatible_backend(
    tmp_path: Path,
) -> None:
    def anygrasp(_context: ToolExecutionContext) -> ToolResult:
        return ToolResult(
            True,
            details={
                "candidate_count": 1,
                "grasp_candidates": [_candidate("anygrasp-0", score=0.9)],
                "metadata": {"max_gripper_width": 0.1},
                "artifacts": [],
            },
        )

    def graspgenx(_context: ToolExecutionContext) -> ToolResult:
        return _success(_candidate("graspgenx-0", score=0.7))

    parameters = _parameters(tmp_path)
    parameters["hints"]["max_gripper_width_m"] = 0.08
    result = build_grasp_pose_estimate_handler(
        {"anygrasp": anygrasp, "graspgenx": graspgenx}
    )(_context(parameters))

    assert result.success is True
    assert result.details["selected_backend"] == "graspgenx"
    assert result.details["backend_attempts"][0]["reason"] == (
        "backend_gripper_width_mismatch"
    )


def test_target_mask_projection_exposes_edge_geometry_without_selecting(
    tmp_path: Path,
) -> None:
    parameters = _parameters(tmp_path)
    mask_path = Path(parameters["object_mask"]["mask_ref"])
    mask = Image.new("L", (16, 16), 0)
    for y in range(4, 13):
        for x in range(4, 13):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)

    edge = _candidate("edge", score=0.8)
    edge["translation_xyz"] = [0.0, -0.02, 0.5]
    edge["gripper_tip_position_xyz"] = [0.0, 0.0, 0.5]
    result = build_grasp_pose_estimate_handler(
        {"anygrasp": lambda _context: _success(edge)}
    )(_context(parameters))

    assert result.success is True
    diagnostic = result.details["diagnostics"][0]
    assert diagnostic["code"] == "target_mask_candidate_projection"
    assert diagnostic["mask_bbox_xyxy"] == [4, 4, 12, 12]
    assert diagnostic["mask_centroid_xy"] == [8.0, 8.0]
    assert diagnostic["candidates"][0]["candidate_id"].endswith("-000")
    assert diagnostic["candidates"][0]["translation"] == {
        "pixel_xy": [8.0, 4.0],
        "inside_target_mask": True,
        "bbox_fraction_xy": [0.5, 0.0],
    }
    assert diagnostic["candidates"][0]["gripper_tip"] == {
        "pixel_xy": [8.0, 8.0],
        "inside_target_mask": True,
        "bbox_fraction_xy": [0.5, 0.5],
    }
    assert result.details["active_grasp_candidate"]["backend_candidate_id"] == "edge"


def test_model_load_failure_falls_back_to_next_backend(tmp_path: Path) -> None:
    calls: list[str] = []

    def unavailable(context: ToolExecutionContext) -> ToolResult:
        calls.append(context.name)
        return _failure("model_load_failed")

    def valid(context: ToolExecutionContext) -> ToolResult:
        calls.append(context.name)
        return _success(_candidate("graspgenx-0", score=0.8))

    result = build_grasp_pose_estimate_handler(
        {"anygrasp": unavailable, "graspgenx": valid}
    )(_context(_parameters(tmp_path)))

    assert result.success is True
    assert result.details["selected_backend"] == "graspgenx"
    assert calls == ["anygrasp", "graspgenx"]


def test_malformed_success_falls_back_to_next_backend(tmp_path: Path) -> None:
    calls: list[str] = []

    def malformed(context: ToolExecutionContext) -> ToolResult:
        calls.append(context.name)
        return _success({"score": 0.9})

    def valid(context: ToolExecutionContext) -> ToolResult:
        calls.append(context.name)
        return _success(_candidate("valid-0", score=0.8))

    result = build_grasp_pose_estimate_handler(
        {"anygrasp": malformed, "graspgenx": valid}
    )(_context(_parameters(tmp_path)))

    assert result.success is True
    assert result.details["selected_backend"] == "graspgenx"
    assert calls == ["anygrasp", "graspgenx"]
    assert result.details["backend_attempts"][0]["reason"] == (
        "inconsistent_grasp_outputs"
    )
