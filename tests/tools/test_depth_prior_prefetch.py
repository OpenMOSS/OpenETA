from __future__ import annotations

import base64
from pathlib import Path
import threading

import pytest

pytest.importorskip("gymnasium")

from adapter.protocol import CameraFrame, EnvObservation, RobotState
from agent.tools.depth_prefetch import DepthPriorPrefetchCoordinator
from agent.tools.handlers import build_sam3_handler
from agent.tools.registry import (
    ToolExecutionContext,
    build_default_tool_registry,
    make_tool_result,
)


PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _observation(rgb_path: Path, *, with_intrinsics: bool = True) -> EnvObservation:
    intrinsics = (
        {"fx": 100.0, "fy": 100.0, "cx": 0.5, "cy": 0.5, "scale": 1000.0} if with_intrinsics else {}
    )
    return EnvObservation(
        task="pick object",
        cameras=[
            CameraFrame(
                frame_id="wrist",
                rgb=[],
                intrinsics=intrinsics,
            )
        ],
        robot=RobotState(),
        metadata={
            "camera_model": "pinhole",
            "calibration_profile_id": "wrist-v1",
            "image_artifacts": [
                {
                    "kind": "rgb",
                    "frame_id": "wrist",
                    "path": str(rgb_path),
                }
            ],
        },
    )


def _context(
    name: str,
    parameters: dict,
    *,
    observation: EnvObservation,
) -> ToolExecutionContext:
    spec = build_default_tool_registry().get(name)
    return ToolExecutionContext(
        name=name,
        spec=spec,
        parameters=parameters,
        observation=observation,
        metadata={"session_id": "session-prefetch"},
    )


def _sam_response() -> dict:
    return {
        "success": True,
        "content": "SAM3 segmentation completed with no detections.",
        "details": {
            "detection_count": 0,
            "detections": [],
        },
    }


def test_sam3_prefetch_overlaps_and_explicit_depth_call_reuses_result(
    tmp_path: Path,
) -> None:
    rgb_path = tmp_path / "wrist.png"
    rgb_path.write_bytes(base64.b64decode(PNG_1X1))
    observation = _observation(rgb_path)
    depth_started = threading.Event()
    release_depth = threading.Event()
    depth_calls = 0

    def depth_handler(context: ToolExecutionContext):
        nonlocal depth_calls
        depth_calls += 1
        depth_started.set()
        assert release_depth.wait(timeout=2.0)
        return make_tool_result(
            context,
            success=True,
            content="Depth prior estimated.",
            outputs={"prior_depth": str(tmp_path / "prior.npy")},
        )

    depth_spec = build_default_tool_registry().get("estimate_depth_prior")
    coordinator = DepthPriorPrefetchCoordinator(depth_handler, spec=depth_spec)

    def segment(_request: dict) -> dict:
        assert depth_started.wait(timeout=2.0)
        release_depth.set()
        return _sam_response()

    sam_handler = build_sam3_handler(
        segment,
        depth_prior_prefetch=coordinator.prefetch_for_sam3,
        output_root=tmp_path / "sam-images",
        result_output_root=tmp_path / "sam-results",
    )
    sam_result = sam_handler(
        _context(
            "sam3",
            {"mode": "text", "image": str(rgb_path), "prompt": "cup"},
            observation=observation,
        )
    )

    assert sam_result.success is True
    assert sam_result.details["depth_prior_prefetch"]["status"] in {
        "running",
        "ready",
    }
    assert depth_calls == 1

    depth_result = coordinator.handler(
        _context(
            "estimate_depth_prior",
            {
                "rgb": str(rgb_path),
                "intrinsics": dict(observation.cameras[0].intrinsics),
            },
            observation=observation,
        )
    )

    assert depth_result.success is True
    assert depth_result.details["outputs"]["prefetch"]["cache_hit"] is True
    assert depth_result.details["outputs"]["prefetch"]["retried_after_prefetch_failure"] is False
    assert depth_calls == 1


def test_explicit_depth_call_retries_failed_background_prefetch(
    tmp_path: Path,
) -> None:
    rgb_path = tmp_path / "wrist.png"
    rgb_path.write_bytes(base64.b64decode(PNG_1X1))
    observation = _observation(rgb_path)
    depth_finished = threading.Event()
    depth_calls = 0

    def depth_handler(context: ToolExecutionContext):
        nonlocal depth_calls
        depth_calls += 1
        if depth_calls == 1:
            result = make_tool_result(
                context,
                success=False,
                content="Transient remote failure.",
            )
            depth_finished.set()
            return result
        return make_tool_result(
            context,
            success=True,
            content="Depth prior estimated.",
        )

    depth_spec = build_default_tool_registry().get("estimate_depth_prior")
    coordinator = DepthPriorPrefetchCoordinator(depth_handler, spec=depth_spec)
    prefetch = coordinator.prefetch_for_sam3(
        _context("sam3", {}, observation=observation),
        str(rgb_path),
    )
    assert prefetch["status"] in {"running", "failed"}
    assert depth_finished.wait(timeout=2.0)

    result = coordinator.handler(
        _context(
            "estimate_depth_prior",
            {
                "rgb": str(rgb_path),
                "intrinsics": dict(observation.cameras[0].intrinsics),
            },
            observation=observation,
        )
    )

    assert result.success is True
    assert result.details["outputs"]["prefetch"] == {
        "cache_hit": True,
        "retried_after_prefetch_failure": True,
        "wait_seconds": pytest.approx(0.0, abs=0.1),
    }
    assert depth_calls == 2


def test_sam3_continues_when_depth_prefetch_has_no_intrinsics(
    tmp_path: Path,
) -> None:
    rgb_path = tmp_path / "wrist.png"
    rgb_path.write_bytes(base64.b64decode(PNG_1X1))
    observation = _observation(rgb_path, with_intrinsics=False)
    depth_calls = 0

    def depth_handler(context: ToolExecutionContext):
        nonlocal depth_calls
        depth_calls += 1
        return make_tool_result(context, success=True)

    depth_spec = build_default_tool_registry().get("estimate_depth_prior")
    coordinator = DepthPriorPrefetchCoordinator(depth_handler, spec=depth_spec)
    sam_handler = build_sam3_handler(
        lambda _request: _sam_response(),
        depth_prior_prefetch=coordinator.prefetch_for_sam3,
        output_root=tmp_path / "sam-images",
        result_output_root=tmp_path / "sam-results",
    )

    result = sam_handler(
        _context(
            "sam3",
            {"mode": "text", "image": str(rgb_path), "prompt": "cup"},
            observation=observation,
        )
    )

    assert result.success is True
    assert result.details["depth_prior_prefetch"] == {
        "status": "skipped",
        "reason": "matching_camera_intrinsics_unavailable",
    }
    assert depth_calls == 0
