from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from adapter.protocol import EnvAction
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.depth_enhancement import (
    PROVENANCE_MONO_FILLED,
    PROVENANCE_SENSOR,
    DepthEnhancementConfig,
    DepthPriorPrediction,
    enhance_rgbd_depth,
    materialize_depth_enhancement,
)
from agent.tools.handlers import build_depth_prior_handler
from agent.tools.registry import ToolExecutionContext
from agent.tools.registry import build_default_tool_registry


def _rgb(height: int = 3, width: int = 3) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = 64
    image[..., 1] = 128
    image[..., 2] = 192
    return image


def _intrinsics() -> dict[str, float]:
    return {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0, "scale": 1000.0}


def test_noop_depth_enhancement_preserves_sensor_depth() -> None:
    sensor = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, np.nan],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    result = enhance_rgbd_depth(
        rgb=_rgb(),
        sensor_depth_m=sensor,
        intrinsics=_intrinsics(),
        camera_id="wrist",
    )

    assert result.enabled is False
    assert result.reason == "no_depth_prior"
    assert np.isnan(result.fused_depth_m[0, 2])
    assert np.isnan(result.fused_depth_m[1, 2])
    assert np.allclose(result.fused_depth_m[result.provenance_mask == PROVENANCE_SENSOR], 1.0)
    assert result.quality["use_for_collision_clearance"] is False


def test_depth_prior_fills_holes_after_scale_alignment_without_replacing_sensor() -> None:
    sensor = np.array(
        [
            [2.0, 2.0, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    prior = DepthPriorPrediction(
        depth_m=np.full((3, 3), 4.0, dtype=np.float32),
        confidence=np.ones((3, 3), dtype=np.float32),
        metadata={"backend": "fixture", "model": "mock-unidepth"},
    )

    result = enhance_rgbd_depth(
        rgb=_rgb(),
        sensor_depth_m=sensor,
        intrinsics=_intrinsics(),
        camera_id="wrist",
        calibration_profile_id="profile-a",
        prior_prediction=prior,
        config=DepthEnhancementConfig(min_alignment_pixels=4, edge_guard_pixels=0),
    )

    assert result.enabled is True
    assert result.reason == "enhanced"
    assert result.alignment["mode"] == "scale_only"
    assert result.alignment["scale"] == 0.5
    assert result.fused_depth_m[1, 1] == 2.0
    assert result.provenance_mask[1, 1] == PROVENANCE_MONO_FILLED
    assert np.all(result.provenance_mask[sensor > 0] == PROVENANCE_SENSOR)
    assert result.quality["use_for_grasp_candidate_generation"] is True
    assert result.quality["use_for_collision_clearance"] is False


def test_depth_prior_keeps_sensor_values_when_model_disagrees() -> None:
    sensor = np.full((3, 3), 2.0, dtype=np.float32)
    sensor[1, 1] = 0.0
    mono = np.full((3, 3), 4.0, dtype=np.float32)
    mono[0, 0] = 4.8
    prior = DepthPriorPrediction(
        depth_m=mono,
        confidence=np.ones((3, 3), dtype=np.float32),
        metadata={"model": "mock-unidepth"},
    )

    result = enhance_rgbd_depth(
        rgb=_rgb(),
        sensor_depth_m=sensor,
        intrinsics=_intrinsics(),
        prior_prediction=prior,
        config=DepthEnhancementConfig(
            min_alignment_pixels=4,
            edge_guard_pixels=0,
            disagreement_threshold_m=0.05,
        ),
    )

    assert result.fused_depth_m[0, 0] == 2.0
    assert result.provenance_mask[0, 0] == PROVENANCE_SENSOR
    assert result.fused_depth_m[1, 1] > 0
    assert result.provenance_mask[1, 1] == PROVENANCE_MONO_FILLED
    assert result.quality["large_disagreement_ratio"] > 0


def test_aligned_prior_is_range_checked_after_scale() -> None:
    sensor = np.full((3, 3), 1.0, dtype=np.float32)
    sensor[1, 1] = 0.0
    mono = np.full((3, 3), 0.5, dtype=np.float32)
    mono[1, 1] = 4.0

    result = enhance_rgbd_depth(
        rgb=_rgb(),
        sensor_depth_m=sensor,
        intrinsics=_intrinsics(),
        prior_prediction=DepthPriorPrediction(
            depth_m=mono,
            confidence=np.ones((3, 3), dtype=np.float32),
        ),
        config=DepthEnhancementConfig(
            min_alignment_pixels=4,
            edge_guard_pixels=0,
        ),
    )

    assert result.alignment["scale"] == 2.0
    assert np.isnan(result.fused_depth_m[1, 1])
    assert result.provenance_mask[1, 1] == 0


def test_uncertainty_uses_lower_is_better_semantics() -> None:
    sensor = np.full((3, 3), 1.0, dtype=np.float32)
    sensor[1, 1] = 0.0
    uncertainty = np.zeros((3, 3), dtype=np.float32)
    uncertainty[1, 1] = 100.0

    result = enhance_rgbd_depth(
        rgb=_rgb(),
        sensor_depth_m=sensor,
        intrinsics=_intrinsics(),
        prior_prediction=DepthPriorPrediction(
            depth_m=np.ones((3, 3), dtype=np.float32),
            confidence=uncertainty,
            confidence_semantics="lower_is_better",
        ),
        config=DepthEnhancementConfig(
            min_alignment_pixels=4,
            edge_guard_pixels=0,
        ),
    )

    assert np.isnan(result.fused_depth_m[1, 1])
    assert result.prior["confidence_semantics"] == "lower_is_better"


def test_low_sensor_confidence_is_not_preserved_as_safety_geometry() -> None:
    sensor = np.ones((3, 3), dtype=np.float32)
    confidence = np.zeros((3, 3), dtype=np.float32)

    result = enhance_rgbd_depth(
        rgb=_rgb(),
        sensor_depth_m=sensor,
        intrinsics=_intrinsics(),
        sensor_confidence=confidence,
    )

    assert not np.isfinite(result.safety_depth_m).any()
    assert not np.isfinite(result.fused_depth_m).any()
    assert not (result.provenance_mask == PROVENANCE_SENSOR).any()


def test_registration_and_timestamp_gates_fail_closed() -> None:
    prior = DepthPriorPrediction(depth_m=np.ones((3, 3), dtype=np.float32))

    unregistered = enhance_rgbd_depth(
        rgb=_rgb(),
        sensor_depth_m=np.ones((3, 3), dtype=np.float32),
        intrinsics=_intrinsics(),
        prior_prediction=prior,
        registration_status="unregistered",
    )
    stale = enhance_rgbd_depth(
        rgb=_rgb(),
        sensor_depth_m=np.ones((3, 3), dtype=np.float32),
        intrinsics=_intrinsics(),
        prior_prediction=prior,
        registration_status="registered",
        rgb_timestamp_s=1.0,
        depth_timestamp_s=1.2,
    )

    assert unregistered.enabled is False
    assert unregistered.reason == "rgb_depth_not_registered"
    assert stale.enabled is False
    assert stale.reason == "rgb_depth_timestamp_skew"


def test_materialize_depth_enhancement_writes_lightweight_report(tmp_path: Path) -> None:
    sensor = np.full((3, 3), 2.0, dtype=np.float32)
    sensor[1, 1] = 0.0
    prior = DepthPriorPrediction(
        depth_m=np.full((3, 3), 4.0, dtype=np.float32),
        confidence=np.ones((3, 3), dtype=np.float32),
        metadata={"backend": "fixture", "model": "mock-unidepth"},
    )
    result = enhance_rgbd_depth(
        rgb=_rgb(),
        sensor_depth_m=sensor,
        intrinsics=_intrinsics(),
        camera_id="wrist",
        prior_prediction=prior,
        config=DepthEnhancementConfig(min_alignment_pixels=4, edge_guard_pixels=0),
    )

    artifact = materialize_depth_enhancement(
        result,
        sensor_depth_m=sensor,
        output_root=tmp_path / "tool_result",
        image_root=tmp_path / "image",
        bundle_id="bundle-a",
        session_id="session-a",
    )
    payload = json.loads(Path(artifact.report_path).read_text(encoding="utf-8"))

    assert payload["schema_version"] == "openeta.depth_enhancement.v1"
    assert payload["enabled"] is True
    assert payload["outputs"]["fused_depth_npy"] == artifact.fused_depth_npy
    assert "fused_depth_m" not in payload
    assert Path(artifact.fused_depth_npy).is_file()
    assert Path(artifact.point_cloud_npz).is_file()
    assert Path(artifact.safety_point_cloud_npz).is_file()
    assert Path(artifact.sensor_depth_png).is_file()
    assert Path(artifact.fused_depth_png).is_file()
    assert Path(artifact.safety_depth_png).is_file()
    assert Path(artifact.provenance_mask_png).is_file()
    assert Path(artifact.report_path).relative_to(tmp_path / "tool_result").parts[0] == "session-a"
    assert Path(artifact.fused_depth_png).relative_to(tmp_path / "image").parts[0] == "session-a"
    assert artifact.to_dict()["outputs"]["point_cloud_npz"] == artifact.point_cloud_npz
    assert payload["outputs"]["depth_scale"] == 1000.0


def test_default_registry_exposes_enhance_depth_tool() -> None:
    spec = build_default_tool_registry().get("enhance_depth")

    assert spec.category == "perception"
    assert spec.effect.value == "read_only"
    assert "prior_depth" in spec.parameters
    assert "sensor_confidence" in spec.parameters


def test_runtime_enhance_depth_tool_materializes_report(tmp_path: Path) -> None:
    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.png"
    prior_path = tmp_path / "prior.npy"
    Image.fromarray(_rgb()).save(rgb_path)
    sensor_depth_mm = np.full((3, 3), 2000, dtype=np.uint16)
    sensor_depth_mm[1, 1] = 0
    Image.fromarray(sensor_depth_mm).save(depth_path)
    np.save(prior_path, np.full((3, 3), 4.0, dtype=np.float32))

    runtime = OpenEtaAgentRuntime()
    input_intrinsics = {**_intrinsics(), "scale": 500.0}
    result = runtime.tools.call(
        "enhance_depth",
        {
            "rgb": str(rgb_path),
            "depth": str(depth_path),
            "prior_depth": str(prior_path),
            "intrinsics": input_intrinsics,
            "camera_id": "wrist",
            "bundle_id": "depth-test",
            "config": {"min_alignment_pixels": 4, "edge_guard_pixels": 0},
        },
        metadata={"session_id": "session-depth"},
    )

    assert result.success is True
    assert result.details["result_type"] == "perception"
    outputs = result.details["outputs"]
    assert outputs["enabled"] is True
    assert outputs["reason"] == "enhanced"
    assert Path(outputs["report_path"]).is_file()
    assert Path(outputs["fused_depth_npy"]).is_file()
    assert Path(outputs["point_cloud_npz"]).is_file()
    assert Path(outputs["provenance_mask_png"]).is_file()
    assert outputs["candidate_intrinsics"]["scale"] == 1000.0
    assert outputs["safety_intrinsics"]["scale"] == 1000.0
    assert outputs["depth_units"] == "millimeters"
    assert result.details["artifacts"][0]["schema_version"] == "openeta.depth_enhancement.v1"
    assert "fused_depth_m" not in json.dumps(result.details)


def test_depth_prior_handler_materializes_metric_prior(tmp_path: Path) -> None:
    rgb_path = tmp_path / "rgb.png"
    Image.fromarray(_rgb()).save(rgb_path)
    seen_requests = []

    def estimate(request):
        seen_requests.append(request)
        return {
            "success": True,
            "details": {
                "backend": "unidepth_mcp",
                "model": "UniDepthV2",
                "depth_m": [[1.0, 1.1], [1.2, 1.3]],
                "confidence": [[0.9, 0.8], [0.7, 0.6]],
            },
        }

    spec = build_default_tool_registry().get("estimate_depth_prior")
    result = build_depth_prior_handler(estimate, output_root=tmp_path / "depth_prior")(
        ToolExecutionContext(
            name="estimate_depth_prior",
            spec=spec,
            parameters={
                "rgb": str(rgb_path),
                "intrinsics": _intrinsics(),
                "camera_id": "wrist",
            },
            metadata={"session_id": "session-depth-prior"},
        )
    )

    outputs = result.details["outputs"]
    assert result.success is True
    assert seen_requests[0]["intrinsics"]["fx"] == 100.0
    assert seen_requests[0]["rgb"]["base64"]
    assert outputs["backend"] == "unidepth_mcp"
    assert outputs["model"] == "UniDepthV2"
    assert Path(outputs["prior_depth"]).is_file()
    assert Path(outputs["prior_confidence"]).is_file()
    assert np.load(outputs["prior_depth"]).shape == (2, 2)
    assert "enhance_depth" in outputs["next_tool_hint"]
    assert "base64" not in json.dumps(result.details)


def test_depth_prior_handler_keeps_uncertainty_direction(tmp_path: Path) -> None:
    rgb_path = tmp_path / "rgb.png"
    Image.fromarray(_rgb()).save(rgb_path)

    def estimate(_request):
        return {
            "success": True,
            "details": {
                "depth_m": [[1.0, 1.0], [1.0, 1.0]],
                "uncertainty": [[0.1, 0.2], [0.3, 0.4]],
            },
        }

    spec = build_default_tool_registry().get("estimate_depth_prior")
    result = build_depth_prior_handler(estimate, output_root=tmp_path / "prior")(
        ToolExecutionContext(
            name="estimate_depth_prior",
            spec=spec,
            parameters={"rgb": str(rgb_path), "intrinsics": _intrinsics()},
        )
    )

    assert result.success is True
    assert result.details["outputs"]["prior_confidence_semantics"] == (
        "lower_is_better"
    )


def test_memory_derives_depth_prior_packet_from_tool_result(tmp_path: Path) -> None:
    runtime = OpenEtaAgentRuntime()
    runtime.memory.start_session(task="real robot depth")
    rgb_path = tmp_path / "rgb.png"
    Image.fromarray(_rgb()).save(rgb_path)

    def estimate(_request):
        return {
            "success": True,
            "details": {
                "backend": "unidepth_mcp",
                "model": "UniDepthV2",
                "depth_m": [[1.0, 1.1], [1.2, 1.3]],
                "confidence": [[0.9, 0.8], [0.7, 0.6]],
            },
        }

    spec = build_default_tool_registry().get("estimate_depth_prior")
    result = build_depth_prior_handler(estimate, output_root=tmp_path / "depth_prior")(
        ToolExecutionContext(
            name="estimate_depth_prior",
            spec=spec,
            parameters={
                "rgb": str(rgb_path),
                "intrinsics": _intrinsics(),
                "camera_id": "wrist",
            },
            metadata={"session_id": runtime.memory.session_id or ""},
        )
    )
    runtime.memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {"kind": "tool_call", "name": "estimate_depth_prior"},
                "tool_calls": [
                    {
                        "name": "estimate_depth_prior",
                        "status": "executed",
                        "result": {
                            "success": result.success,
                            "content": result.content,
                            "details": result.details,
                        },
                    }
                ],
            },
        )
    )

    artifacts = runtime.memory.get_memory(namespace="artifacts")["artifacts"]
    packet = artifacts["estimate_depth_prior_depth_prior_wrist"]["value"]
    summary = runtime.memory.planning_context()["working_memory"]["artifacts"][
        "estimate_depth_prior_depth_prior_wrist"
    ]
    assert packet["backend"] == "unidepth_mcp"
    assert packet["model"] == "UniDepthV2"
    assert packet["prior_depth"] == result.details["outputs"]["prior_depth"]
    assert packet["prior_confidence"] == result.details["outputs"]["prior_confidence"]
    assert "enhance_depth" in summary["next_tool_hint"]
    assert summary["prior_depth"] == packet["prior_depth"]


def test_memory_records_only_clear_sensor_safety_evidence(tmp_path: Path) -> None:
    safety_depth = tmp_path / "safety.png"
    safety_cloud = tmp_path / "safety.npz"
    report = tmp_path / "report.json"
    for path in (safety_depth, safety_cloud, report):
        path.write_bytes(b"fixture")
    request = {
        "kind": "enhanced_grasp_sensor_safety_check",
        "candidate_id": "gpe-1",
        "scene_epoch": 2,
        "safety_depth_png": str(safety_depth),
        "safety_point_cloud_npz": str(safety_cloud),
        "report_path": str(report),
    }
    memory = OpenEtaAgentRuntime().memory
    memory.start_session(task="safe enhanced grasp")
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {
                        "name": "obstacle_avoidance",
                        "result": {
                            "success": True,
                            "details": {
                                "parameters": {"path": request},
                                "outputs": {"clear": True},
                                "artifacts": [],
                            },
                        },
                    }
                ]
            },
        )
    )

    values = [
        entry.get("value")
        for entry in memory.artifacts.values()
        if isinstance(entry, dict)
    ]
    evidence = next(
        value
        for value in values
        if isinstance(value, dict)
        and value.get("type") == "enhanced_grasp_sensor_safety_check"
    )
    assert evidence["candidate_id"] == "gpe-1"
    assert evidence["clear"] is True


def test_memory_derives_depth_enhancement_packet_from_tool_result(tmp_path: Path) -> None:
    runtime = OpenEtaAgentRuntime()
    runtime.memory.start_session(task="real robot depth")
    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.png"
    prior_path = tmp_path / "prior.npy"
    Image.fromarray(_rgb()).save(rgb_path)
    sensor_depth_mm = np.full((3, 3), 2000, dtype=np.uint16)
    sensor_depth_mm[1, 1] = 0
    Image.fromarray(sensor_depth_mm).save(depth_path)
    np.save(prior_path, np.full((3, 3), 4.0, dtype=np.float32))
    result = runtime.tools.call(
        "enhance_depth",
        {
            "rgb": str(rgb_path),
            "depth": str(depth_path),
            "prior_depth": str(prior_path),
            "intrinsics": _intrinsics(),
            "camera_id": "wrist",
            "config": {"min_alignment_pixels": 4, "edge_guard_pixels": 0},
        },
        metadata={"session_id": runtime.memory.session_id or ""},
    )

    runtime.memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {"kind": "tool_call", "name": "enhance_depth"},
                "tool_calls": [
                    {
                        "name": "enhance_depth",
                        "status": "executed",
                        "result": {
                            "success": result.success,
                            "content": result.content,
                            "details": result.details,
                        },
                    }
                ],
            },
        )
    )

    artifacts = runtime.memory.get_memory(namespace="artifacts")["artifacts"]
    packet = artifacts["enhance_depth_depth_enhancement_wrist"]["value"]
    summary = runtime.memory.planning_context()["working_memory"]["artifacts"][
        "enhance_depth_depth_enhancement_wrist"
    ]
    assert packet["enabled"] is True
    assert packet["fused_depth_npy"] == result.details["outputs"]["fused_depth_npy"]
    assert packet["quality"]["use_for_grasp_candidate_generation"] is True
    assert summary["fused_depth_npy"] == packet["fused_depth_npy"]
    assert summary["quality"]["use_for_collision_clearance"] is False
    assert "mono-filled geometry" in summary["next_tool_hint"]
