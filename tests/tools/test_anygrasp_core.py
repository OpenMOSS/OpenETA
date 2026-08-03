from __future__ import annotations

import base64
import hashlib
import io
from typing import Any

import numpy as np
import pytest

from tools.anygrasp_core import (
    AnyGraspBackend,
    AnyGraspInputError,
    build_point_cloud_from_rgbd,
    normalise_grasp_candidates,
    validate_detect_grasps_options,
    validate_intrinsics,
)


def _intrinsics() -> dict[str, float]:
    return {"fx": 2.0, "fy": 2.0, "cx": 0.0, "cy": 0.0, "scale": 1000.0}


def test_build_point_cloud_rejects_shape_mismatch() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.ones((3, 2), dtype=np.uint16)

    with pytest.raises(AnyGraspInputError, match="image_shape_mismatch"):
        build_point_cloud_from_rgbd(
            rgb=rgb,
            depth=depth,
            target_mask=None,
            intrinsics=_intrinsics(),
            mode="scene",
            depth_truncation=1.0,
            workspace_limits=None,
        )


def test_build_point_cloud_rejects_empty_target_mask() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.uint16) * 100
    target_mask = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(AnyGraspInputError, match="empty_target_mask"):
        build_point_cloud_from_rgbd(
            rgb=rgb,
            depth=depth,
            target_mask=target_mask,
            intrinsics=_intrinsics(),
            mode="targeted",
            depth_truncation=1.0,
            workspace_limits=None,
        )


def test_validate_intrinsics_rejects_missing_or_invalid_values() -> None:
    with pytest.raises(AnyGraspInputError, match="invalid_intrinsics"):
        validate_intrinsics({"fx": 1.0, "fy": 1.0, "cx": 0.0, "scale": 1.0})
    with pytest.raises(AnyGraspInputError, match="invalid_depth_scale"):
        validate_intrinsics({"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0})
    with pytest.raises(AnyGraspInputError, match="invalid_intrinsics"):
        validate_intrinsics({"fx": 0.0, "fy": 1.0, "cx": 0.0, "cy": 0.0, "scale": 1.0})
    with pytest.raises(AnyGraspInputError, match="invalid_intrinsics"):
        validate_intrinsics({"fx": float("nan"), "fy": 1.0, "cx": 0.0, "cy": 0.0, "scale": 1.0})
    with pytest.raises(AnyGraspInputError, match="invalid_intrinsics"):
        validate_intrinsics({"fx": 1.0, "fy": 1.0, "cx": float("inf"), "cy": 0.0, "scale": 1.0})
    with pytest.raises(AnyGraspInputError, match="invalid_depth_scale"):
        validate_intrinsics({"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0, "scale": 0.0})
    with pytest.raises(AnyGraspInputError, match="invalid_depth_scale"):
        validate_intrinsics({"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0, "scale": float("inf")})


def test_build_point_cloud_rejects_uint16_depth_scale_mismatch() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.full((2, 2), 850, dtype=np.uint16)

    with pytest.raises(AnyGraspInputError, match="depth_scale_mismatch") as raised:
        build_point_cloud_from_rgbd(
            rgb=rgb,
            depth=depth,
            target_mask=None,
            intrinsics={**_intrinsics(), "scale": 1.0},
            mode="scene",
            depth_truncation=1.0,
            workspace_limits=None,
        )

    assert raised.value.metadata == {
        "depth_dtype": "uint16",
        "depth_raw_min": 850,
        "depth_raw_max": 850,
        "depth_metric_min": 850.0,
        "depth_metric_max": 850.0,
        "depth_truncation": 1.0,
        "valid_point_count": 0,
    }


def test_build_point_cloud_reports_empty_after_depth_filter() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.zeros((2, 2), dtype=np.uint16)

    with pytest.raises(
        AnyGraspInputError,
        match="empty_point_cloud_after_depth_filter",
    ) as raised:
        build_point_cloud_from_rgbd(
            rgb=rgb,
            depth=depth,
            target_mask=None,
            intrinsics=_intrinsics(),
            mode="scene",
            depth_truncation=1.0,
            workspace_limits=None,
        )

    assert raised.value.metadata["valid_point_count"] == 0
    assert raised.value.metadata["depth_metric_min"] == 0.0
    assert raised.value.metadata["depth_metric_max"] == 0.0


def test_backend_returns_structured_depth_scale_diagnostics() -> None:
    backend = AnyGraspBackend(sdk_root=".", checkpoint_path=".")
    result = backend.detect_grasps(
        rgb=_png_payload(np.zeros((2, 2, 3), dtype=np.uint8)),
        depth=_png_payload(np.full((2, 2), 850, dtype=np.uint16)),
        intrinsics={**_intrinsics(), "scale": 1.0},
        mode="scene",
    )

    assert result["success"] is False
    assert result["details"]["reason"] == "depth_scale_mismatch"
    assert result["details"]["metadata"]["depth_dtype"] == "uint16"
    assert result["details"]["metadata"]["depth_raw_max"] == 850
    assert result["details"]["metadata"]["valid_point_count"] == 0
    assert "for uint16 millimeter depth, use scale=1000" in result["content"]


def test_backend_prioritises_empty_point_cloud_input_diagnostics() -> None:
    backend = AnyGraspBackend(sdk_root=".", checkpoint_path=".")
    result = backend.detect_grasps(
        rgb=_png_payload(np.zeros((2, 2, 3), dtype=np.uint8)),
        depth=_png_payload(np.full((2, 2), 1500, dtype=np.uint16)),
        intrinsics=_intrinsics(),
        mode="scene",
    )

    assert result["success"] is False
    assert result["details"]["reason"] == "empty_point_cloud_after_depth_filter"
    assert result["details"]["metadata"]["valid_point_count"] == 0
    assert result["details"]["metadata"]["depth_raw_min"] == 1500
    assert result["details"]["metadata"]["depth_raw_max"] == 1500
    assert result["details"]["metadata"]["depth_metric_min"] == 1.5
    assert result["details"]["metadata"]["depth_metric_max"] == 1.5
    assert result["details"]["metadata"]["intrinsics"] == _intrinsics()
    assert "for uint16 millimeter depth, use scale=1000" in result["content"]


def test_backend_returns_invalid_depth_scale_before_decoding() -> None:
    backend = AnyGraspBackend(sdk_root=".", checkpoint_path=".")
    result = backend.detect_grasps(
        rgb={},
        depth={},
        intrinsics={**_intrinsics(), "scale": 0.0},
        mode="scene",
    )

    assert result["success"] is False
    assert result["details"]["reason"] == "invalid_depth_scale"
    assert "for uint16 millimeter depth, use scale=1000" in result["content"]


def test_backend_success_metadata_describes_depth_conversion() -> None:
    backend = AnyGraspBackend(sdk_root=".", checkpoint_path=".")
    backend._get_detector = lambda: _FakeDetector(_FakeGraspGroup())  # type: ignore[method-assign]
    result = backend.detect_grasps(
        rgb=_png_payload(np.zeros((2, 2, 3), dtype=np.uint8)),
        depth=_png_payload(np.full((2, 2), 100, dtype=np.uint16)),
        target_mask=_png_payload(np.full((2, 2), 255, dtype=np.uint8)),
        intrinsics=_intrinsics(),
        mode="targeted",
    )

    assert result["success"] is True
    metadata = result["details"]["metadata"]
    assert metadata["depth_dtype"] == "uint16"
    assert metadata["depth_raw_min"] == 100
    assert metadata["depth_raw_max"] == 100
    assert metadata["depth_metric_min"] == pytest.approx(0.1)
    assert metadata["depth_metric_max"] == pytest.approx(0.1)
    assert metadata["intrinsics"] == _intrinsics()
    assert metadata["depth_truncation"] == 1.0
    assert metadata["valid_point_count"] == 4
    assert metadata["point_count"] == 4
    assert metadata["service"]["configuration"]["depth_truncation"] == 1.0


def test_backend_service_info_fingerprints_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.tar"
    checkpoint.write_bytes(b"known checkpoint")

    backend = AnyGraspBackend(sdk_root=tmp_path, checkpoint_path=checkpoint)
    info = backend.service_info()

    assert info["checkpoint"] == {
        "name": "checkpoint.tar",
        "sha256": hashlib.sha256(b"known checkpoint").hexdigest(),
        "size_bytes": len(b"known checkpoint"),
    }
    assert info["configuration"] == {
        "max_gripper_width": 0.1,
        "gripper_height": 0.03,
        "depth_truncation": 1.0,
        "max_candidates": 20,
    }
    assert info["openeta"]["revision"]
    assert info["anygrasp_sdk"] == {"revision": None, "dirty": None}


def test_backend_reserves_no_grasp_candidates_for_model_output() -> None:
    backend = AnyGraspBackend(sdk_root=".", checkpoint_path=".")
    backend._get_detector = lambda: _FakeDetector(None)  # type: ignore[method-assign]
    result = backend.detect_grasps(
        rgb=_png_payload(np.zeros((2, 2, 3), dtype=np.uint8)),
        depth=_png_payload(np.full((2, 2), 100, dtype=np.uint16)),
        intrinsics=_intrinsics(),
        mode="scene",
    )

    assert result["success"] is False
    assert result["details"]["reason"] == "no_grasp_candidates"
    assert result["details"]["metadata"]["valid_point_count"] == 4


def test_validate_options_rejects_invalid_values() -> None:
    with pytest.raises(AnyGraspInputError, match="invalid_option"):
        validate_detect_grasps_options(collision_detection="yes", dense_grasp=False)
    with pytest.raises(AnyGraspInputError, match="invalid_approach_steering"):
        validate_detect_grasps_options(
            collision_detection=True,
            dense_grasp=False,
            approach_steering=[0.0, 1.0],
        )


def test_normalise_grasp_candidates_computes_gripper_tip_position() -> None:
    grasps = [_FakeGrasp()]

    candidates = normalise_grasp_candidates(grasps)

    assert candidates == [
        {
            "id": "grasp_000",
            "frame": "camera",
            "camera_frame": "opencv",
            "score": 0.7,
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
            "backend_index": 0,
            "rank": 0,
        }
    ]


class _FakeGrasp:
    score = 0.7
    width = 0.06
    height = 0.03
    depth = 0.03
    rotation_matrix = np.eye(3)
    translation = np.array([0.1, 0.2, 0.3])


def test_normalise_grasp_candidates_defensively_sorts_by_score() -> None:
    lower = _FakeGrasp()
    lower.score = 0.2
    higher = _FakeGrasp()
    higher.score = 0.9

    candidates = normalise_grasp_candidates([lower, higher])

    assert [candidate["score"] for candidate in candidates] == [0.9, 0.2]
    assert [candidate["rank"] for candidate in candidates] == [0, 1]
    assert [candidate["backend_index"] for candidate in candidates] == [1, 0]
    assert [candidate["id"] for candidate in candidates] == ["grasp_000", "grasp_001"]


class _FakeGraspGroup:
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return self
        return _FakeGrasp()

    def nms(self) -> _FakeGraspGroup:
        return self

    def sort_by_score(self) -> _FakeGraspGroup:
        return self


class _FakeDetector:
    def __init__(self, grasps: Any) -> None:
        self.grasps = grasps

    def get_grasp(self, _points: Any, _params: dict[str, Any]) -> Any:
        return self.grasps


def _png_payload(array: np.ndarray) -> dict[str, str]:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return {
        "format": "png",
        "base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }
