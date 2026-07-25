from __future__ import annotations

import base64
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tools.contact_graspnet_core import (
    GRIPPER_DEPTH,
    ContactGraspNetBackend,
    ContactGraspNetInputError,
    build_targeted_point_clouds,
    decode_png_payload,
    load_checkpoint_model_state,
    normalise_grasp_candidates,
    seed_inference,
    validate_intrinsics,
)


def _intrinsics() -> dict[str, float]:
    return {"fx": 100.0, "fy": 100.0, "cx": 1.5, "cy": 1.5, "scale": 1000.0}


def _png_payload(array: np.ndarray) -> dict[str, str]:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return {
        "format": "png",
        "base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def test_validate_intrinsics_normalises_finite_values() -> None:
    assert validate_intrinsics(
        {"fx": 100, "fy": "101", "cx": 2, "cy": 3, "scale": "1000"}
    ) == {"fx": 100.0, "fy": 101.0, "cx": 2.0, "cy": 3.0, "scale": 1000.0}


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (None, "missing_intrinsics"),
        ({"fx": 1, "fy": 1, "cx": 0, "scale": 1}, "invalid_intrinsics"),
        (
            {"fx": 0, "fy": 1, "cx": 0, "cy": 0, "scale": 1},
            "invalid_intrinsics",
        ),
        (
            {"fx": 1, "fy": 1, "cx": 0, "cy": 0},
            "invalid_depth_scale",
        ),
        (
            {"fx": 1, "fy": 1, "cx": 0, "cy": 0, "scale": 0},
            "invalid_depth_scale",
        ),
    ],
)
def test_validate_intrinsics_rejects_invalid_values(value: Any, reason: str) -> None:
    with pytest.raises(ContactGraspNetInputError, match=reason):
        validate_intrinsics(value)


def test_decode_png_payload_requires_declared_and_actual_png() -> None:
    from PIL import Image

    payload = _png_payload(np.ones((2, 2), dtype=np.uint16))
    decoded = decode_png_payload(
        payload,
        Image=Image,
        np=np,
        convert=None,
        missing_reason="missing_depth",
        decode_reason="depth_decode_failed",
    )
    assert decoded.dtype == np.uint16

    with pytest.raises(ContactGraspNetInputError, match="unsupported_image_format"):
        decode_png_payload(
            {**payload, "format": "npy"},
            Image=Image,
            np=np,
            convert=None,
            missing_reason="missing_depth",
            decode_reason="depth_decode_failed",
        )


def test_build_targeted_point_clouds_projects_opencv_geometry() -> None:
    depth = np.full((4, 4), 500, dtype=np.uint16)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255

    scene, object_points, metadata = build_targeted_point_clouds(
        depth_array=depth,
        object_mask_array=mask,
        intrinsics=_intrinsics(),
        depth_min=0.2,
        depth_max=1.8,
    )

    assert scene.shape == (16, 3)
    assert object_points.shape == (4, 3)
    np.testing.assert_allclose(object_points[:, 2], 0.5)
    assert metadata["scene_valid_point_count"] == 16
    assert metadata["object_valid_point_count"] == 4
    assert metadata["local_region_point_count"] == 16
    assert metadata["model_input_point_count"] == 20_000


def test_build_targeted_point_clouds_reports_depth_scale_mismatch() -> None:
    with pytest.raises(ContactGraspNetInputError, match="depth_scale_mismatch") as raised:
        build_targeted_point_clouds(
            depth_array=np.full((2, 2), 500, dtype=np.uint16),
            object_mask_array=np.ones((2, 2), dtype=np.uint8),
            intrinsics={**_intrinsics(), "scale": 1.0},
            depth_min=0.2,
            depth_max=1.8,
        )

    assert raised.value.metadata["depth_raw_max"] == 500
    assert raised.value.metadata["scene_valid_point_count"] == 0


@pytest.mark.parametrize(
    ("depth", "mask", "reason"),
    [
        (
            np.ones((2, 2), dtype=np.uint8),
            np.ones((2, 2), dtype=np.uint8),
            "unsupported_depth_format",
        ),
        (
            np.full((2, 2), 500, dtype=np.uint16),
            np.ones((3, 2), dtype=np.uint8),
            "image_shape_mismatch",
        ),
        (
            np.full((2, 2), 500, dtype=np.uint16),
            np.zeros((2, 2), dtype=np.uint8),
            "empty_object_mask",
        ),
        (
            np.zeros((2, 2), dtype=np.uint16),
            np.ones((2, 2), dtype=np.uint8),
            "empty_point_cloud_after_depth_filter",
        ),
        (
            np.full((2, 2), 500, dtype=np.uint16),
            np.array([[0, 0], [0, 255]], dtype=np.uint8),
            "empty_object_point_cloud",
        ),
    ],
)
def test_build_targeted_point_clouds_rejects_invalid_geometry(
    depth: np.ndarray,
    mask: np.ndarray,
    reason: str,
) -> None:
    if reason == "empty_object_point_cloud":
        depth = depth.copy()
        depth[1, 1] = 0
    with pytest.raises(ContactGraspNetInputError, match=reason):
        build_targeted_point_clouds(
            depth_array=depth,
            object_mask_array=mask,
            intrinsics=_intrinsics(),
            depth_min=0.2,
            depth_max=1.8,
        )


def test_normalise_candidates_converts_axes_and_stably_sorts() -> None:
    grasps = np.tile(np.eye(4, dtype=np.float64), (3, 1, 1))
    grasps[:, :3, 3] = [[0.0, 0.0, 0.5], [0.1, 0.0, 0.5], [0.2, 0.0, 0.5]]
    scores = np.array([0.2, 0.9, 0.9])
    contacts = np.array([[0.0, 0.0, 0.6], [1.0, 0.0, 0.6], [2.0, 0.0, 0.6]])
    openings = np.array([0.03, 0.04, 0.05])

    candidates, raw_count = normalise_grasp_candidates(
        grasps=grasps,
        scores=scores,
        contact_points=contacts,
        gripper_openings=openings,
        max_gripper_width=0.08,
        max_candidates=2,
    )

    assert raw_count == 3
    assert [candidate["id"] for candidate in candidates] == ["grasp_000", "grasp_001"]
    assert [candidate["score"] for candidate in candidates] == [0.9, 0.9]
    assert [candidate["contact_point_xyz"][0] for candidate in candidates] == [1.0, 2.0]
    np.testing.assert_allclose(
        candidates[0]["rotation_matrix"],
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(
        candidates[0]["gripper_tip_position_xyz"],
        [0.1, 0.0, 0.5 + GRIPPER_DEPTH],
    )
    assert candidates[0]["grasp_frame"] == "graspnet"
    assert candidates[0]["gripper_model"] == "panda"
    assert candidates[0]["gripper_depth"] == GRIPPER_DEPTH
    assert candidates[0]["width"] == 0.04
    assert "depth" not in candidates[0]
    assert "height" not in candidates[0]
    assert "transform_matrix" not in candidates[0]


@pytest.mark.parametrize("failure", ["empty", "reflection", "nan", "opening"])
def test_normalise_candidates_fails_atomically(failure: str) -> None:
    grasps = np.tile(np.eye(4, dtype=np.float64), (2, 1, 1))
    scores = np.array([0.5, 0.4])
    contacts = np.zeros((2, 3))
    openings = np.array([0.04, 0.05])
    expected = "inconsistent_grasp_outputs"
    if failure == "empty":
        grasps = np.empty((0, 4, 4))
        scores = np.empty((0,))
        contacts = np.empty((0, 3))
        openings = np.empty((0,))
        expected = "no_grasp_candidates"
    elif failure == "reflection":
        grasps[1, :3, :3] = np.diag([1.0, 1.0, -1.0])
    elif failure == "nan":
        scores[1] = np.nan
    else:
        openings[1] = 0.2

    with pytest.raises(ContactGraspNetInputError, match=expected):
        normalise_grasp_candidates(
            grasps=grasps,
            scores=scores,
            contact_points=contacts,
            gripper_openings=openings,
            max_gripper_width=0.08,
        )


class _FakeBackend(ContactGraspNetBackend):
    def _get_loaded_backend(self) -> dict[str, Any]:
        return {
            "backend_commit": "a" * 40,
            "checkpoint_sha256": "b" * 64,
            "max_gripper_width": 0.08,
        }

    def _predict_raw(self, **_kwargs: Any) -> tuple[Any, Any, Any, Any]:
        grasp = np.eye(4, dtype=np.float64)[None, ...]
        grasp[0, 2, 3] = 0.5
        return grasp, np.array([0.7]), np.array([[0.0, 0.0, 0.6]]), np.array([0.04])


def test_backend_returns_contract_without_transport_payloads() -> None:
    backend = _FakeBackend(contact_graspnet_root=".", checkpoint_dir=".")
    result = backend.predict_grasps(
        depth=_png_payload(np.full((4, 4), 500, dtype=np.uint16)),
        object_mask=_png_payload(np.full((4, 4), 255, dtype=np.uint8)),
        intrinsics=_intrinsics(),
    )

    assert result["success"] is True
    details = result["details"]
    assert details["candidate_count"] == 1
    assert details["grasp_candidates"][0]["source_model"] == "contact_graspnet"
    assert details["metadata"]["model_candidate_count"] == 1
    assert details["metadata"]["checkpoint_sha256"] == "b" * 64
    serialized = str(result)
    assert "base64" not in serialized
    assert "point_cloud" not in serialized


def test_backend_returns_structured_depth_diagnostic_failure() -> None:
    backend = _FakeBackend(contact_graspnet_root=".", checkpoint_dir=".")
    result = backend.predict_grasps(
        depth=_png_payload(np.full((2, 2), 500, dtype=np.uint16)),
        object_mask=_png_payload(np.ones((2, 2), dtype=np.uint8)),
        intrinsics={**_intrinsics(), "scale": 1.0},
    )

    assert result["success"] is False
    assert result["details"]["reason"] == "depth_scale_mismatch"
    assert result["details"]["candidate_count"] == 0
    assert "use scale=1000" in result["content"]


def test_load_checkpoint_model_state_uses_restricted_loader(tmp_path: Path) -> None:
    calls: dict[str, Any] = {}
    safe_globals: list[Any] = []

    def load(path: Path, **kwargs: Any) -> dict[str, Any]:
        calls["path"] = path
        calls.update(kwargs)
        return {"model": {"weight": "value"}}

    fake_torch = SimpleNamespace(
        serialization=SimpleNamespace(add_safe_globals=lambda values: safe_globals.extend(values)),
        load=load,
    )
    checkpoint = tmp_path / "model.pt"

    state = load_checkpoint_model_state(
        torch=fake_torch,
        np=np,
        checkpoint_path=checkpoint,
        device="cuda:0",
    )

    assert state == {"weight": "value"}
    assert calls == {
        "path": checkpoint,
        "map_location": "cuda:0",
        "weights_only": True,
    }
    assert safe_globals


def test_seed_inference_resets_numpy_and_torch_rngs() -> None:
    calls: list[tuple[str, int]] = []
    fake_torch = SimpleNamespace(
        manual_seed=lambda seed: calls.append(("cpu", seed)),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=lambda seed: calls.append(("cuda", seed)),
        ),
    )

    seed_inference(seed=7, np=np, torch=fake_torch)
    first = np.random.random(3)
    seed_inference(seed=7, np=np, torch=fake_torch)
    second = np.random.random(3)

    np.testing.assert_allclose(first, second)
    assert calls == [("cpu", 7), ("cuda", 7), ("cpu", 7), ("cuda", 7)]
