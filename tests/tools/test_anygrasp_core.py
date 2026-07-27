from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import pytest

from tools.anygrasp_core import (
    AnyGraspBackend,
    AnyGraspInputError,
    build_point_cloud_from_rgbd,
    normalise_grasp_candidates,
    validate_depth_bounds,
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
        "depth_min": 0.0,
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


def test_build_point_cloud_near_cut_drops_sub_working_distance_points() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    # Two rows: 100 mm (0.1 m, inside D435 dead zone) and 400 mm (0.4 m, valid).
    depth = np.array([[100, 100], [400, 400]], dtype=np.uint16)

    points, _colors, _steering, metadata = build_point_cloud_from_rgbd(
        rgb=rgb,
        depth=depth,
        target_mask=None,
        intrinsics=_intrinsics(),
        mode="scene",
        depth_truncation=1.0,
        workspace_limits=None,
        depth_min=0.2,
    )

    assert metadata["depth_min"] == 0.2
    assert metadata["valid_point_count"] == 2
    assert points.shape[0] == 2
    assert np.allclose(points[:, 2], 0.4)


def test_build_point_cloud_far_cut_drops_long_range_outliers() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    # 400 mm target vs 8000 mm stray return; a 1.5 m far cut keeps only the target.
    depth = np.array([[400, 400], [8000, 8000]], dtype=np.uint16)

    points, _colors, _steering, metadata = build_point_cloud_from_rgbd(
        rgb=rgb,
        depth=depth,
        target_mask=None,
        intrinsics=_intrinsics(),
        mode="scene",
        depth_truncation=1.5,
        workspace_limits=None,
    )

    assert metadata["depth_truncation"] == 1.5
    assert metadata["valid_point_count"] == 2
    assert np.allclose(points[:, 2], 0.4)


def test_validate_depth_bounds_defaults_to_backend_far_cut() -> None:
    far, near = validate_depth_bounds(
        depth_truncation=None, depth_min=None, default_truncation=1.0
    )
    assert far == 1.0
    assert near == 0.0


def test_validate_depth_bounds_accepts_per_call_overrides() -> None:
    far, near = validate_depth_bounds(
        depth_truncation=2.0, depth_min=0.2, default_truncation=1.0
    )
    assert far == 2.0
    assert near == 0.2


def test_validate_depth_bounds_rejects_non_positive_far() -> None:
    with pytest.raises(AnyGraspInputError, match="invalid_depth_truncation"):
        validate_depth_bounds(
            depth_truncation=0.0, depth_min=None, default_truncation=1.0
        )


def test_validate_depth_bounds_rejects_negative_near() -> None:
    with pytest.raises(AnyGraspInputError, match="invalid_depth_min"):
        validate_depth_bounds(
            depth_truncation=1.0, depth_min=-0.1, default_truncation=1.0
        )


def test_validate_depth_bounds_rejects_near_above_far() -> None:
    with pytest.raises(AnyGraspInputError, match="invalid_depth_bounds"):
        validate_depth_bounds(
            depth_truncation=0.5, depth_min=0.6, default_truncation=1.0
        )


def test_validate_depth_bounds_rejects_bool_inputs() -> None:
    with pytest.raises(AnyGraspInputError, match="invalid_depth_truncation"):
        validate_depth_bounds(
            depth_truncation=True, depth_min=None, default_truncation=1.0
        )
