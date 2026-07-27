from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import pytest
from PIL import Image

from tools.anyplace_core import AnyPlaceBackend
from tools.anyplace_core import AnyPlaceInputError
from tools.anyplace_core import DEFAULT_DEPTH_TRUNCATION
from tools.anyplace_core import build_masked_pointclouds_from_rgbd
from tools.anyplace_core import normalise_placement_candidates
from tools.anyplace_core import normalise_selected_grasp
from tools.anyplace_core import validate_depth_truncation
from tools.anyplace_core import validate_intrinsics


def _image_payload(array: np.ndarray) -> dict[str, str]:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return {
        "format": "png",
        "base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def _intrinsics() -> dict[str, float]:
    return {"fx": 100.0, "fy": 100.0, "cx": 0.0, "cy": 0.0, "scale": 1000.0}


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    depth = np.full((64, 64), 500, dtype=np.uint16)
    object_mask = np.zeros((64, 64), dtype=np.uint8)
    object_mask[:, :32] = 255
    placement_mask = np.zeros((64, 64), dtype=np.uint8)
    placement_mask[:, 32:] = 255
    return rgb, depth, object_mask, placement_mask


def _selected_grasp(**overrides: Any) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "id": "grasp_003",
        "frame": "camera",
        "camera_frame": "opencv",
        "score": 0.7,
        "translation_xyz": [0.1, 0.2, 0.3],
        "rotation_matrix": np.eye(3).tolist(),
        "depth": 0.03,
        "width": 0.06,
        "height": 0.03,
        "gripper_tip_position_xyz": [0.13, 0.2, 0.3],
    }
    candidate.update(overrides)
    return candidate


def _request() -> dict[str, Any]:
    rgb, depth, object_mask, placement_mask = _arrays()
    return {
        "rgb": _image_payload(rgb),
        "depth": _image_payload(depth),
        "object_mask": _image_payload(object_mask),
        "placement_region_mask": _image_payload(placement_mask),
        "intrinsics": _intrinsics(),
        "selected_grasp": _selected_grasp(),
    }


def _assert_reason(reason: str, func, *args, **kwargs) -> None:
    with pytest.raises(AnyPlaceInputError) as exc_info:
        func(*args, **kwargs)
    assert exc_info.value.reason == reason


def test_validate_intrinsics_requires_anygrasp_camera_fields() -> None:
    assert validate_intrinsics(_intrinsics()) == _intrinsics()
    _assert_reason("missing_intrinsics", validate_intrinsics, None)
    _assert_reason(
        "invalid_intrinsics",
        validate_intrinsics,
        {"fx": 0.0, "fy": 1.0, "cx": 0.0, "cy": 0.0, "scale": 1000.0},
    )


def test_build_masked_pointclouds_projects_metric_opencv_points() -> None:
    rgb, depth, object_mask, placement_mask = _arrays()

    object_points, placement_points = build_masked_pointclouds_from_rgbd(
        rgb=rgb,
        depth=depth,
        object_mask=object_mask,
        placement_region_mask=placement_mask,
        intrinsics=_intrinsics(),
        depth_truncation=1.0,
    )

    assert object_points.shape == (2048, 3)
    assert placement_points.shape == (2048, 3)
    assert object_points.dtype == np.float32
    assert placement_points.dtype == np.float32
    np.testing.assert_allclose(object_points[0], [0.0, 0.0, 0.5])
    np.testing.assert_allclose(placement_points[0], [0.16, 0.0, 0.5])


def test_build_masked_pointclouds_rejects_shape_mismatch() -> None:
    rgb, depth, object_mask, placement_mask = _arrays()

    _assert_reason(
        "image_shape_mismatch",
        build_masked_pointclouds_from_rgbd,
        rgb=rgb,
        depth=depth,
        object_mask=object_mask[:-1],
        placement_region_mask=placement_mask,
        intrinsics=_intrinsics(),
        depth_truncation=1.0,
    )


@pytest.mark.parametrize(
    ("array_name", "reason"),
    [
        ("object_mask", "empty_object_mask"),
        ("placement_region_mask", "empty_placement_region_mask"),
    ],
)
def test_build_masked_pointclouds_rejects_empty_masks(array_name: str, reason: str) -> None:
    rgb, depth, object_mask, placement_mask = _arrays()
    values = {
        "rgb": rgb,
        "depth": depth,
        "object_mask": object_mask,
        "placement_region_mask": placement_mask,
        "intrinsics": _intrinsics(),
        "depth_truncation": 1.0,
    }
    values[array_name] = np.zeros_like(values[array_name])

    _assert_reason(reason, build_masked_pointclouds_from_rgbd, **values)


def test_build_masked_pointclouds_rejects_no_valid_depth() -> None:
    rgb, depth, object_mask, placement_mask = _arrays()
    depth.fill(0)

    _assert_reason(
        "empty_object_pointcloud",
        build_masked_pointclouds_from_rgbd,
        rgb=rgb,
        depth=depth,
        object_mask=object_mask,
        placement_region_mask=placement_mask,
        intrinsics=_intrinsics(),
        depth_truncation=1.0,
    )


def test_build_masked_pointclouds_uses_configurable_far_depth_cut() -> None:
    rgb, depth, object_mask, placement_mask = _arrays()
    depth[:, :32] = 500
    depth[:, 32:] = 1200

    _assert_reason(
        "empty_placement_region_pointcloud",
        build_masked_pointclouds_from_rgbd,
        rgb=rgb,
        depth=depth,
        object_mask=object_mask,
        placement_region_mask=placement_mask,
        intrinsics=_intrinsics(),
        depth_truncation=1.0,
    )

    object_points, placement_points = build_masked_pointclouds_from_rgbd(
        rgb=rgb,
        depth=depth,
        object_mask=object_mask,
        placement_region_mask=placement_mask,
        intrinsics=_intrinsics(),
        depth_truncation=2.0,
    )

    assert object_points.shape == (2048, 3)
    assert placement_points.shape == (2048, 3)
    np.testing.assert_allclose(object_points[0], [0.0, 0.0, 0.5])
    np.testing.assert_allclose(placement_points[0], [0.384, 0.0, 1.2])


@pytest.mark.parametrize(
    ("mask_name", "reason"),
    [
        ("object_mask", "object_pointcloud_too_small"),
        ("placement_region_mask", "placement_region_pointcloud_too_small"),
    ],
)
def test_build_masked_pointclouds_requires_1024_points_per_mask(
    mask_name: str,
    reason: str,
) -> None:
    rgb, depth, object_mask, placement_mask = _arrays()
    masks = {
        "object_mask": object_mask,
        "placement_region_mask": placement_mask,
    }
    masks[mask_name].fill(0)
    masks[mask_name][:16, :16] = 255

    _assert_reason(
        reason,
        build_masked_pointclouds_from_rgbd,
        rgb=rgb,
        depth=depth,
        object_mask=masks["object_mask"],
        placement_region_mask=masks["placement_region_mask"],
        intrinsics=_intrinsics(),
        depth_truncation=1.0,
    )


def test_normalise_selected_grasp_accepts_anygrasp_candidate() -> None:
    assert normalise_selected_grasp(_selected_grasp()) == _selected_grasp()


@pytest.mark.parametrize(
    "candidate",
    [
        None,
        _selected_grasp(frame="world"),
        _selected_grasp(camera_frame="opengl"),
        _selected_grasp(rotation_matrix=[[1.0]]),
        _selected_grasp(rotation_matrix=np.diag([1.0, 1.0, -1.0]).tolist()),
        _selected_grasp(translation_xyz=[float("nan"), 0.0, 0.0]),
    ],
)
def test_normalise_selected_grasp_rejects_invalid_candidates(candidate: Any) -> None:
    expected = "missing_selected_grasp" if candidate is None else "invalid_selected_grasp"
    _assert_reason(expected, normalise_selected_grasp, candidate)


def test_normalise_placement_candidates_composes_transform_with_selected_grasp() -> None:
    poses = np.tile(np.eye(4, dtype=np.float64), (5, 1, 1))
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    poses[0, :3, :3] = rotation
    poses[0, :3, 3] = [1.0, 2.0, 3.0]
    grasp = normalise_selected_grasp(_selected_grasp())

    candidates = normalise_placement_candidates(poses, selected_grasp=grasp)

    assert [candidate["id"] for candidate in candidates] == [
        "placement_000",
        "placement_001",
        "placement_002",
        "placement_003",
        "placement_004",
    ]
    first = candidates[0]
    assert first["source_grasp_id"] == "grasp_003"
    transform = first["object_placement_transform"]
    assert transform["frame"] == "camera"
    assert transform["camera_frame"] == "opencv"
    assert transform["convention"] == "p_placed = R @ p_current + t"
    np.testing.assert_allclose(transform["transform_matrix"], poses[0])
    place = first["place_grasp_pose"]
    np.testing.assert_allclose(place["rotation_matrix"], rotation)
    np.testing.assert_allclose(place["translation_xyz"], [0.8, 2.1, 3.3])
    np.testing.assert_allclose(place["gripper_tip_position_xyz"], [0.8, 2.13, 3.3])
    assert place["score"] == 0.7
    assert place["depth"] == 0.03
    assert place["width"] == 0.06
    assert place["height"] == 0.03


@pytest.mark.parametrize(
    "raw_candidates",
    [
        [],
        np.eye(4),
        np.tile(np.eye(4), (4, 1, 1)),
        np.ones((5, 3, 4)),
    ],
)
def test_normalise_placement_candidates_requires_exactly_five_transforms(
    raw_candidates: Any,
) -> None:
    expected = "no_placement_candidates" if isinstance(raw_candidates, list) else "inconsistent_placement_outputs"
    _assert_reason(
        expected,
        normalise_placement_candidates,
        raw_candidates,
        selected_grasp=normalise_selected_grasp(_selected_grasp()),
    )


@pytest.mark.parametrize("failure", ["non_finite", "bad_bottom_row", "reflection"])
def test_normalise_placement_candidates_fails_atomically_for_invalid_transform(
    failure: str,
) -> None:
    poses = np.tile(np.eye(4, dtype=np.float64), (5, 1, 1))
    if failure == "non_finite":
        poses[3, 0, 0] = np.inf
    elif failure == "bad_bottom_row":
        poses[3, 3, 0] = 1.0
    else:
        poses[3, :3, :3] = np.diag([1.0, 1.0, -1.0])

    _assert_reason(
        "inconsistent_placement_outputs",
        normalise_placement_candidates,
        poses,
        selected_grasp=normalise_selected_grasp(_selected_grasp()),
    )


def test_backend_fails_before_model_load_for_missing_rgb(tmp_path) -> None:
    backend = AnyPlaceBackend(
        anyplace_root=tmp_path / "missing-root",
        config_path=tmp_path / "missing.yaml",
    )
    request = _request()
    request["rgb"] = None

    result = backend.predict_placement(**request)

    assert result["success"] is False
    assert result["details"]["reason"] == "missing_rgb"
    assert result["details"]["candidate_count"] == 0
    assert result["details"]["placement_candidates"] == []


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("rgb", "rgb_decode_failed"),
        ("depth", "depth_decode_failed"),
        ("object_mask", "object_mask_decode_failed"),
        ("placement_region_mask", "placement_region_mask_decode_failed"),
    ],
)
def test_backend_reports_image_decode_failures(tmp_path, field: str, reason: str) -> None:
    backend = AnyPlaceBackend(
        anyplace_root=tmp_path / "missing-root",
        config_path=tmp_path / "missing.yaml",
    )
    request = _request()
    request[field] = {"format": "png", "base64": "not-base64"}

    result = backend.predict_placement(**request)

    assert result["success"] is False
    assert result["details"]["reason"] == reason


def test_backend_returns_composed_candidates_without_pointcloud_payloads(tmp_path, monkeypatch) -> None:
    backend = AnyPlaceBackend(anyplace_root=tmp_path, config_path=tmp_path / "config.yaml")
    monkeypatch.setattr(backend, "_get_loaded_backend", lambda: {})
    monkeypatch.setattr(
        backend,
        "_predict_with_loaded_backend",
        lambda **_kwargs: np.tile(np.eye(4, dtype=np.float64), (5, 1, 1)),
    )

    result = backend.predict_placement(**_request())

    assert result["success"] is True
    details = result["details"]
    assert details["frame"] == "camera"
    assert details["camera_frame"] == "opencv"
    assert details["candidate_count"] == 5
    assert details["metadata"]["object_point_count"] == 2048
    assert details["metadata"]["placement_region_point_count"] == 2048
    assert details["metadata"]["depth_truncation"] == DEFAULT_DEPTH_TRUNCATION
    assert details["metadata"]["model_sample_count"] == 1024
    assert "object_pointcloud" not in str(result)
    assert "placement_region_pointcloud" not in str(result)


def test_backend_accepts_per_call_depth_truncation(tmp_path, monkeypatch) -> None:
    backend = AnyPlaceBackend(anyplace_root=tmp_path, config_path=tmp_path / "config.yaml")
    monkeypatch.setattr(backend, "_get_loaded_backend", lambda: {})
    monkeypatch.setattr(
        backend,
        "_predict_with_loaded_backend",
        lambda **_kwargs: np.tile(np.eye(4, dtype=np.float64), (5, 1, 1)),
    )
    request = _request()
    rgb, depth, object_mask, placement_mask = _arrays()
    depth.fill(1200)
    request.update(
        {
            "rgb": _image_payload(rgb),
            "depth": _image_payload(depth),
            "object_mask": _image_payload(object_mask),
            "placement_region_mask": _image_payload(placement_mask),
            "depth_truncation": 1.5,
        }
    )

    result = backend.predict_placement(**request)

    assert result["success"] is True
    assert result["details"]["metadata"]["depth_truncation"] == 1.5


def test_validate_depth_truncation_rejects_invalid_values() -> None:
    assert validate_depth_truncation(None, default_truncation=2.0) == 2.0
    assert validate_depth_truncation(1.5, default_truncation=2.0) == 1.5
    for value in (0.0, -1.0, float("nan"), True, "2.0"):
        _assert_reason(
            "invalid_depth_truncation",
            validate_depth_truncation,
            value,
            default_truncation=2.0,
        )


def test_mcp_returns_structured_failure_when_backend_is_not_configured(monkeypatch) -> None:
    pytest.importorskip("mcp")
    from tools import anyplace_mcp_server

    monkeypatch.setattr(anyplace_mcp_server, "_BACKEND", None)

    result = anyplace_mcp_server.predict_placement()

    assert result["success"] is False
    assert result["details"]["reason"] == "model_load_failed"
    assert result["details"]["frame"] == "camera"
    assert result["details"]["camera_frame"] == "opencv"
    assert result["details"]["candidate_count"] == 0
    assert result["details"]["placement_candidates"] == []
