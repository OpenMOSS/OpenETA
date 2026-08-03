from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agent.tools.handlers import DEFAULT_ANYGRASP_OUTPUT_ROOT, build_anygrasp_handler
from agent.tools.registry import ToolExecutionContext, build_default_tool_registry
from scripts.embodied.grasp_pose_viewer import load_anygrasp_poses


PNG_1X1 = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c6360000002000100ffff03000006000557bfab0d000000"
        "0049454e44ae426082"
    )
).decode("ascii")


def _write_png(path: Path) -> Path:
    path.write_bytes(base64.b64decode(PNG_1X1))
    return path


def _context(parameters: dict[str, Any]) -> ToolExecutionContext:
    spec = build_default_tool_registry().get("anygrasp")
    return ToolExecutionContext(name="anygrasp", spec=spec, parameters=parameters)


def test_anygrasp_default_root_uses_repo_tmp_tool_result_layout() -> None:
    assert DEFAULT_ANYGRASP_OUTPUT_ROOT == Path("tmp") / "tool_result" / "anygrasp"


def _valid_parameters(tmp_path: Path) -> dict[str, Any]:
    rgb = _write_png(tmp_path / "rgb.png")
    depth = _write_png(tmp_path / "depth.png")
    target_mask = _write_png(tmp_path / "mask.png")
    return {
        "mode": "targeted",
        "rgb": str(rgb),
        "depth": str(depth),
        "target_mask": str(target_mask),
        "intrinsics": {
            "fx": 927.17,
            "fy": 927.37,
            "cx": 651.32,
            "cy": 349.62,
            "scale": 1000.0,
        },
        "collision_detection": True,
        "dense_grasp": False,
    }


def _success_response(candidates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    grasp_candidates = candidates if candidates is not None else [_candidate()]
    return {
        "success": True,
        "content": "AnyGrasp grasp detection completed.",
        "details": {
            "tool": "anygrasp",
            "backend": "anygrasp_mcp",
            "model": "anygrasp_sdk",
            "mode": "targeted",
            "candidate_count": len(grasp_candidates),
            "grasp_candidates": grasp_candidates,
            "artifacts": [],
            "metadata": {"duration_s": 0.1},
        },
    }


def _candidate(**overrides: Any) -> dict[str, Any]:
    candidate = {
        "id": "grasp_000",
        "frame": "camera",
        "camera_frame": "opencv",
        "score": 0.5,
        "translation_xyz": [0.1, 0.2, 0.3],
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "depth": 0.03,
        "width": 0.06,
        "height": 0.03,
        "gripper_tip_position_xyz": [0.13, 0.2, 0.3],
    }
    candidate.update(overrides)
    return candidate


def test_anygrasp_tool_spec_exposes_public_parameters() -> None:
    spec = build_default_tool_registry().get("anygrasp")

    assert set(spec.parameters) == {
        "mode",
        "rgb",
        "depth",
        "intrinsics",
        "target_mask",
        "approach_steering",
        "approach_thresh",
        "collision_detection",
        "dense_grasp",
    }
    assert "fx, fy, cx, cy, scale" in spec.parameters["intrinsics"]
    assert "same observe/render camera_packet.anygrasp_intrinsics" in spec.parameters["intrinsics"]
    assert "required for targeted mode" in spec.parameters["target_mask"]
    assert "details.outputs.selected_detection.mask_ref" in spec.parameters["target_mask"]
    assert "details.outputs.detections[i].mask_ref" in spec.parameters["target_mask"]
    assert "detections[0]" not in spec.parameters["target_mask"]


@pytest.mark.parametrize(
    ("parameters", "reason"),
    [
        ({"depth": "depth.png", "intrinsics": {}}, "missing_rgb"),
        ({"rgb": "rgb.png", "intrinsics": {}}, "missing_depth"),
        ({"rgb": "rgb.png", "depth": "depth.png"}, "missing_intrinsics"),
        (
            {"rgb": "rgb.png", "depth": "depth.png", "intrinsics": {}, "mode": "targeted"},
            "missing_target_mask",
        ),
        (
            {
                "rgb": "rgb.png",
                "depth": "depth.png",
                "target_mask": "mask.png",
                "intrinsics": {},
                "mode": "scene",
            },
            "target_mask_not_allowed_in_scene_mode",
        ),
    ],
)
def test_anygrasp_handler_fails_closed_for_missing_or_invalid_public_inputs(
    parameters: dict[str, Any],
    reason: str,
) -> None:
    handler = build_anygrasp_handler(lambda request: {})

    result = handler(_context(parameters))

    assert result.success is False
    assert result.details["reason"] == reason
    assert result.details["candidate_count"] == 0
    assert result.details["grasp_candidates"] == []


@pytest.mark.parametrize(
    ("missing_field", "reason"),
    [
        ("rgb", "rgb_not_found"),
        ("depth", "depth_not_found"),
        ("target_mask", "target_mask_not_found"),
    ],
)
def test_anygrasp_handler_reports_split_missing_path_reasons(
    tmp_path: Path,
    missing_field: str,
    reason: str,
) -> None:
    parameters = _valid_parameters(tmp_path)
    parameters[missing_field] = str(tmp_path / f"missing-{missing_field}.png")
    handler = build_anygrasp_handler(lambda request: {})

    result = handler(_context(parameters))

    assert result.success is False
    assert result.details["reason"] == reason


def test_anygrasp_handler_encodes_inputs_and_materializes_success(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def detect_grasps(request: dict[str, Any]) -> dict[str, Any]:
        calls.append(request)
        return _success_response()

    parameters = _valid_parameters(tmp_path)
    handler = build_anygrasp_handler(detect_grasps, output_root=tmp_path / "runs")

    result = handler(_context(parameters))

    assert result.success is True
    assert calls[0]["mode"] == "targeted"
    assert calls[0]["rgb"]["format"] == "png"
    assert calls[0]["depth"]["format"] == "png"
    assert calls[0]["target_mask"]["format"] == "png"
    assert base64.b64decode(calls[0]["rgb"]["base64"])
    assert result.details["source_rgb"] == parameters["rgb"]
    assert result.details["source_depth"] == parameters["depth"]
    assert result.details["target_mask"] == parameters["target_mask"]
    assert result.details["source"] == {
        "mode": "targeted",
        "rgb": parameters["rgb"],
        "depth": parameters["depth"],
        "object_mask": parameters["target_mask"],
        "intrinsics": parameters["intrinsics"],
    }
    assert result.details["candidate_count"] == 1
    assert result.details["grasp_candidates"][0]["id"] == "grasp_000"
    assert result.details["best_grasp_candidate"]["id"] == "grasp_000"
    assert result.details["active_grasp_candidate"]["id"] == "grasp_000"
    assert result.details["ranking"] == "score_descending"

    raw_output_ref = Path(result.details["raw_output_ref"])
    assert result.details["result_id"] == raw_output_ref.parent.name
    run_dir = raw_output_ref.parent
    assert raw_output_ref.name == "response.raw.json"
    assert raw_output_ref.exists()
    canonical_ref = Path(result.details["canonical_grasp_candidates_ref"])
    assert canonical_ref == run_dir / "grasp_candidates.canonical.json"
    assert canonical_ref.exists()
    canonical = json.loads(canonical_ref.read_text())
    assert canonical["schema_version"] == "openeta.canonical_grasp_candidates.v1"
    assert canonical["result_id"] == result.details["result_id"]
    assert canonical["grasp_candidates"] == result.details["grasp_candidates"]
    assert (run_dir / "request.json").exists()
    assert (run_dir / "tool_result.json").exists()

    request_text = (run_dir / "request.json").read_text()
    raw_text = raw_output_ref.read_text()
    tool_result_text = (run_dir / "tool_result.json").read_text()
    assert PNG_1X1 not in request_text
    assert "base64" not in request_text
    assert "base64" not in json.dumps(result.details)
    assert "base64" not in tool_result_text
    assert "grasp_candidates" in raw_text


def test_anygrasp_handler_defensively_ranks_candidates(tmp_path: Path) -> None:
    response = _success_response(
        [
            _candidate(
                id="backend-low",
                score=0.2,
                translation_xyz=[0.1, 0.2, 0.3],
            ),
            _candidate(
                id="backend-high",
                score=0.9,
                translation_xyz=[0.9, 0.8, 0.7],
            ),
        ]
    )
    handler = build_anygrasp_handler(
        lambda request: response,
        output_root=tmp_path / "runs",
    )

    result = handler(_context(_valid_parameters(tmp_path)))

    candidates = result.details["grasp_candidates"]
    assert [candidate["score"] for candidate in candidates] == [0.9, 0.2]
    assert [candidate["id"] for candidate in candidates] == ["grasp_000", "grasp_001"]
    assert [candidate["rank"] for candidate in candidates] == [0, 1]
    assert [candidate["backend_index"] for candidate in candidates] == [1, 0]

    canonical_ref = Path(result.details["canonical_grasp_candidates_ref"])
    canonical = json.loads(canonical_ref.read_text())
    assert canonical["grasp_candidates"] == candidates
    poses = load_anygrasp_poses(
        canonical_ref,
        camera_to_world_opencv=np.eye(4),
        proposal_set_id="proposal-test",
        expected_result_id=result.details["result_id"],
    )
    assert [pose.source_id for pose in poses] == ["grasp_000", "grasp_001"]
    np.testing.assert_allclose(
        poses[0].world_from_grip_site[:3, 3],
        candidates[0]["translation_xyz"],
    )
    np.testing.assert_allclose(
        poses[1].world_from_grip_site[:3, 3],
        candidates[1]["translation_xyz"],
    )


def test_anygrasp_source_is_absent_for_scene_success(tmp_path: Path) -> None:
    parameters = _valid_parameters(tmp_path)
    parameters["mode"] = "scene"
    parameters.pop("target_mask")
    handler = build_anygrasp_handler(lambda request: _success_response())

    result = handler(_context(parameters))

    assert result.success is True
    assert "source" not in result.details


def test_anygrasp_source_is_absent_for_failure(tmp_path: Path) -> None:
    handler = build_anygrasp_handler(
        lambda request: {
            "success": False,
            "details": {"reason": "no_grasp_candidates", "metadata": {}},
        }
    )

    result = handler(_context(_valid_parameters(tmp_path)))

    assert result.success is False
    assert "source" not in result.details


def test_anygrasp_handler_scrubs_artifact_base64_from_outputs(tmp_path: Path) -> None:
    def detect_grasps(request: dict[str, Any]) -> dict[str, Any]:
        response = _success_response()
        response["details"]["artifacts"] = [
            {"artifact_type": "debug_overlay", "format": "png", "base64": PNG_1X1}
        ]
        return response

    handler = build_anygrasp_handler(detect_grasps, output_root=tmp_path / "runs")

    result = handler(_context(_valid_parameters(tmp_path)))

    assert result.success is True
    assert "base64" not in json.dumps(result.details)
    raw_output_ref = Path(result.details["raw_output_ref"])
    run_dir = raw_output_ref.parent
    raw_text = raw_output_ref.read_text()
    tool_result_text = (run_dir / "tool_result.json").read_text()
    assert PNG_1X1 not in raw_text
    assert PNG_1X1 not in tool_result_text
    assert "base64_omitted" in raw_text
    assert "base64_omitted" not in tool_result_text
    assert '"base64"' not in tool_result_text


@pytest.mark.parametrize(
    "response",
    [
        {"success": True},
        {"success": True, "details": {}},
        {"success": True, "details": {"candidate_count": 1}},
        {"success": True, "details": {"candidate_count": 0, "grasp_candidates": []}},
        {
            "success": True,
            "details": {"candidate_count": 1, "grasp_candidates": [_candidate(score="bad")]},
        },
        {
            "success": True,
            "details": {
                "candidate_count": 1,
                "grasp_candidates": [_candidate(rotation_matrix=[[1.0]])],
            },
        },
    ],
)
def test_anygrasp_handler_rejects_malformed_success(
    tmp_path: Path,
    response: dict[str, Any],
) -> None:
    handler = build_anygrasp_handler(lambda request: response, output_root=tmp_path / "runs")

    result = handler(_context(_valid_parameters(tmp_path)))

    assert result.success is False
    assert result.details["reason"] == "inconsistent_grasp_outputs"
    assert result.details["candidate_count"] == 0
    assert result.details["grasp_candidates"] == []


def test_anygrasp_handler_preserves_backend_empty_candidate_failure(tmp_path: Path) -> None:
    handler = build_anygrasp_handler(
        lambda request: {
            "success": False,
            "content": "AnyGrasp returned no grasp candidates.",
            "details": {
                "reason": "no_grasp_candidates",
                "candidate_count": 0,
                "grasp_candidates": [],
                "metadata": {"duration_s": 0.2},
            },
        }
    )

    result = handler(_context(_valid_parameters(tmp_path)))

    assert result.success is False
    assert result.details["reason"] == "no_grasp_candidates"
    assert result.details["candidate_count"] == 0
    assert result.details["grasp_candidates"] == []
    assert result.details["metadata"] == {"duration_s": 0.2}


def test_anygrasp_handler_preserves_depth_diagnostic_failure(tmp_path: Path) -> None:
    metadata = {
        "depth_dtype": "uint16",
        "depth_raw_min": 0,
        "depth_raw_max": 850,
        "depth_metric_min": 0.0,
        "depth_metric_max": 850.0,
        "intrinsics": {
            "fx": 600.0,
            "fy": 600.0,
            "cx": 320.0,
            "cy": 240.0,
            "scale": 1.0,
        },
        "depth_truncation": 1.0,
        "valid_point_count": 0,
    }
    handler = build_anygrasp_handler(
        lambda request: {
            "success": False,
            "content": "AnyGrasp grasp detection failed: depth_scale_mismatch.",
            "details": {
                "reason": "depth_scale_mismatch",
                "candidate_count": 0,
                "grasp_candidates": [],
                "metadata": metadata,
            },
        }
    )

    result = handler(_context(_valid_parameters(tmp_path)))

    assert result.success is False
    assert result.details["reason"] == "depth_scale_mismatch"
    assert result.details["metadata"] == metadata


def test_anygrasp_handler_structures_callable_exceptions(tmp_path: Path) -> None:
    def detect_grasps(request: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("server unavailable")

    handler = build_anygrasp_handler(detect_grasps)

    result = handler(_context(_valid_parameters(tmp_path)))

    assert result.success is False
    assert result.details["reason"] == "mcp_call_failed"
    assert result.details["metadata"]["error_type"] == "RuntimeError"
