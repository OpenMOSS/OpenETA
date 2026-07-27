from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tools.graspgenx_core import (
    MAX_RETURNED_CANDIDATES,
    GraspGenXBackend,
    GraspGenXInputError,
    GripperDescription,
    build_targeted_point_clouds,
    decode_image_payload,
    normalise_grasp_candidates,
    rotation_aligning_up_to_z,
    scan_gripper_descriptions,
    validate_cuda_device_name,
    validate_intrinsics,
    validate_raw_grasp_outputs,
    validate_up_direction,
)


def _image_payload(array: np.ndarray, *, format_name: str = "PNG") -> dict[str, str]:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=format_name)
    return {
        "format": "advisory-value",
        "base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def _intrinsics() -> dict[str, float]:
    return {"fx": 100.0, "fy": 100.0, "cx": 5.0, "cy": 5.0, "scale": 1000.0}


def _gripper_config() -> dict[str, Any]:
    return {
        "type": "parallel_2f",
        "fingertip": [0.01, 0.02, 0.1],
        "sweep_volume": {
            "extents": [0.08, 0.02, 0.03],
            "offset": [0.0, 0.0, 0.09],
            "extents2": [0.04, 0.02, 0.03],
            "offset2": [0.0, 0.0, 0.09],
        },
    }


def _write_gripper(root: Path, name: str, *, valid: bool = True) -> Path:
    asset = (
        root
        / "gripper_descriptions"
        / "assets"
        / "x_grippers"
        / name
    )
    asset.mkdir(parents=True)
    config = _gripper_config()
    if not valid:
        config["fingertip"] = [0.0, 0.0, -1.0]
    (asset / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (asset / "coll_mesh.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
    )
    return asset


def _backend_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "GraspGenX"
    package = source / "graspgenx"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")

    checkpoints = tmp_path / "checkpoints" / "release"
    for component, epoch in (("gen", 736), ("dis", 1056)):
        directory = checkpoints / component
        directory.mkdir(parents=True)
        (directory / "config.yaml").write_text("model: fake\n", encoding="utf-8")
        (directory / f"epoch_{epoch}.pth").write_bytes(b"checkpoint")

    grippers = tmp_path / "gripper_descriptions_checkout"
    _write_gripper(grippers, "franka_panda")
    return source, checkpoints, grippers


def test_validate_intrinsics_and_up_direction_normalise_values() -> None:
    assert validate_intrinsics(
        {"fx": "100", "fy": 101, "cx": 2, "cy": 3, "scale": "1000"}
    ) == {"fx": 100.0, "fy": 101.0, "cx": 2.0, "cy": 3.0, "scale": 1000.0}
    np.testing.assert_allclose(validate_up_direction([0, 0, -2]), [0, 0, -1])


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
    ],
)
def test_validate_intrinsics_rejects_invalid_values(value: Any, reason: str) -> None:
    with pytest.raises(GraspGenXInputError, match=reason):
        validate_intrinsics(value)


@pytest.mark.parametrize("value", [None, [0, 0, 0], [1, 2], [1, float("nan"), 0]])
def test_validate_up_direction_rejects_invalid_values(value: Any) -> None:
    with pytest.raises(GraspGenXInputError, match="invalid_up_direction"):
        validate_up_direction(value)


def test_decode_image_uses_actual_bytes_not_advisory_format() -> None:
    from PIL import Image

    payload = _image_payload(np.full((2, 2), 500, dtype=np.uint16))
    decoded = decode_image_payload(
        payload,
        Image=Image,
        np=np,
        convert=None,
        missing_reason="missing_depth",
        decode_reason="depth_decode_failed",
    )
    assert decoded.dtype == np.uint16
    np.testing.assert_array_equal(decoded, np.full((2, 2), 500, dtype=np.uint16))


def test_build_point_clouds_projects_opencv_geometry_without_downsampling() -> None:
    depth = np.full((61, 61), 500, dtype=np.uint16)
    mask = np.full((61, 61), 255, dtype=np.uint8)

    object_points, scene_points, metadata = build_targeted_point_clouds(
        depth_array=depth,
        object_mask_array=mask,
        intrinsics=_intrinsics(),
    )

    assert object_points.shape == (3721, 3)
    assert scene_points.shape == (0, 3)
    assert metadata["model_input_point_count"] == 3721
    np.testing.assert_allclose(object_points[:, 2], 0.5)


def test_build_point_clouds_uses_camera_z_strict_one_meter_cutoff() -> None:
    depth = np.full((11, 11), 999, dtype=np.uint16)
    depth[0, 0] = 1000
    mask = np.full((11, 11), 255, dtype=np.uint8)

    object_points, _scene, metadata = build_targeted_point_clouds(
        depth_array=depth,
        object_mask_array=mask,
        intrinsics=_intrinsics(),
    )

    assert object_points.shape == (120, 3)
    assert metadata["depth_truncation"] == 1.0
    assert float(object_points[:, 2].max()) < 1.0


def test_build_point_clouds_reports_scale_and_point_count_diagnostics() -> None:
    with pytest.raises(GraspGenXInputError, match="depth_scale_mismatch") as raised:
        build_targeted_point_clouds(
            depth_array=np.full((11, 11), 500, dtype=np.uint16),
            object_mask_array=np.full((11, 11), 255, dtype=np.uint8),
            intrinsics={**_intrinsics(), "scale": 1.0},
        )
    assert raised.value.metadata["depth_raw_max"] == 500.0
    assert raised.value.metadata["valid_point_count"] == 0

    with pytest.raises(GraspGenXInputError, match="insufficient_object_points") as raised:
        build_targeted_point_clouds(
            depth_array=np.full((11, 11), 500, dtype=np.uint16),
            object_mask_array=np.pad(
                np.full((9, 11), 255, dtype=np.uint8), ((0, 2), (0, 0))
            ),
            intrinsics=_intrinsics(),
        )
    assert raised.value.metadata["object_point_count"] == 99


@pytest.mark.parametrize(
    ("depth", "mask", "reason"),
    [
        (
            np.ones((2, 2, 1), dtype=np.uint16),
            np.ones((2, 2), dtype=np.uint8),
            "image_shape_mismatch",
        ),
        (
            np.ones((2, 2), dtype=np.uint16),
            np.ones((3, 2), dtype=np.uint8),
            "image_shape_mismatch",
        ),
        (
            np.ones((11, 11), dtype=np.uint16),
            np.zeros((11, 11), dtype=np.uint8),
            "empty_object_mask",
        ),
        (
            np.zeros((11, 11), dtype=np.uint16),
            np.ones((11, 11), dtype=np.uint8),
            "empty_point_cloud_after_depth_filter",
        ),
    ],
)
def test_build_point_clouds_rejects_invalid_geometry(
    depth: np.ndarray, mask: np.ndarray, reason: str
) -> None:
    with pytest.raises(GraspGenXInputError, match=reason):
        build_targeted_point_clouds(
            depth_array=depth,
            object_mask_array=mask,
            intrinsics=_intrinsics(),
        )


def test_alignment_rotates_camera_up_to_world_z() -> None:
    rotation = rotation_aligning_up_to_z([0.0, 0.0, -1.0])
    np.testing.assert_allclose(rotation @ [0.0, 0.0, -1.0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_raw_output_validation_is_atomic() -> None:
    grasps = np.tile(np.eye(4), (2, 1, 1))
    scores = np.array([0.7, 0.6])
    validated = validate_raw_grasp_outputs(grasps, scores, ["diff", "obb"])
    assert validated[0].shape == (2, 4, 4)

    grasps[1, :3, :3] = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(GraspGenXInputError, match="inconsistent_grasp_outputs"):
        validate_raw_grasp_outputs(grasps, scores, ["diff", "obb"])

    with pytest.raises(GraspGenXInputError, match="no_grasp_candidates"):
        validate_raw_grasp_outputs(np.empty((0, 4, 4)), np.empty((0,)), [])


def test_normalise_candidates_builds_dual_pose_and_real_tip() -> None:
    native = np.eye(4)[None, ...]
    native[0, :3, 3] = [0.2, 0.3, 0.4]
    gripper = GripperDescription(
        name="test_gripper",
        gripper_type="parallel_2f",
        fingertip_xyz=(0.01, 0.02, 0.1),
        sweep_open_extents_xyz=(0.08, 0.02, 0.03),
        sweep_open_offset_xyz=(0.0, 0.0, 0.1),
        sweep_mid_extents_xyz=(0.04, 0.02, 0.03),
        sweep_mid_offset_xyz=(0.0, 0.0, 0.1),
    )

    candidates = normalise_grasp_candidates(
        camera_native_grasps=native,
        scores=np.array([0.42]),
        branch_tags=["diff"],
        selected_indices=[0],
        gripper=gripper,
    )

    candidate = candidates[0]
    np.testing.assert_allclose(
        candidate["rotation_matrix"],
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(candidate["gripper_tip_position_xyz"], [0.21, 0.32, 0.5])
    np.testing.assert_allclose(
        candidate["model_native_grasp_pose"]["transform_matrix"], native[0]
    )
    assert candidate["depth"] == 0.1
    assert candidate["width"] == 0.08
    assert candidate["height"] == 0.02


def test_scan_grippers_excludes_invalid_assets_and_sorts_public_list(
    tmp_path: Path,
) -> None:
    _write_gripper(tmp_path, "valid_b")
    _write_gripper(tmp_path, "valid_a")
    _write_gripper(tmp_path, "invalid", valid=False)

    grippers, invalid = scan_gripper_descriptions(tmp_path)

    assert list(grippers) == ["valid_a", "valid_b"]
    assert list(invalid) == ["invalid"]
    assert grippers["valid_a"].to_public_dict()["asset_family"] == "x_grippers"


def test_scan_grippers_rejects_unmaterialized_lfs_collision_mesh(
    tmp_path: Path,
) -> None:
    asset = _write_gripper(tmp_path, "lfs_pointer")
    (asset / "coll_mesh.obj").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:deadbeef\n"
        "size 1234\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no valid inference-ready"):
        scan_gripper_descriptions(tmp_path)


def test_validate_cuda_device_name_rejects_cpu_fallback() -> None:
    assert validate_cuda_device_name("cuda") == "cuda:0"
    assert validate_cuda_device_name("cuda:2") == "cuda:2"
    with pytest.raises(ValueError, match="cuda"):
        validate_cuda_device_name("cpu")


def test_backend_is_lazy_and_list_grippers_does_not_load_model(tmp_path: Path) -> None:
    source, checkpoints, grippers = _backend_layout(tmp_path)
    backend = GraspGenXBackend(
        graspgenx_root=source,
        checkpoint_root=checkpoints,
        gripper_descriptions_root=grippers,
    )

    result = backend.list_grippers()

    assert backend.model_loaded is False
    assert result["details"]["model_loaded"] is False
    assert result["details"]["gripper_count"] == 1


def test_offline_paths_are_set_before_official_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, checkpoints, grippers = _backend_layout(tmp_path)
    backend = GraspGenXBackend(
        graspgenx_root=source,
        checkpoint_root=checkpoints,
        gripper_descriptions_root=grippers,
    )
    monkeypatch.setenv("GRASPGENX_CHECKPOINT_DIR", "test-placeholder")
    monkeypatch.setenv("GRASPGENX_GRIPPER_CFG_DIR", "test-placeholder")
    observed_environment: dict[str, str | None] = {}

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def device(_index: int):
            return contextlib.nullcontext()

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = FakeCuda()  # type: ignore[attr-defined]

    fake_package = types.ModuleType("graspgenx")
    fake_package.__path__ = []  # type: ignore[attr-defined]
    fake_server = types.ModuleType("graspgenx.grasp_server")
    fake_server.GraspGenXSampler = object  # type: ignore[attr-defined]

    def load_model(_config: Any) -> object:
        observed_environment["checkpoints"] = os.environ.get(
            "GRASPGENX_CHECKPOINT_DIR"
        )
        observed_environment["grippers"] = os.environ.get(
            "GRASPGENX_GRIPPER_CFG_DIR"
        )
        return object()

    fake_server.load_grasp_gen_model = load_model  # type: ignore[attr-defined]
    fake_samplers = types.ModuleType("graspgenx.samplers")
    fake_samplers.run_planner_on_object = object()  # type: ignore[attr-defined]
    fake_utils = types.ModuleType("graspgenx.utils")
    fake_utils.__path__ = []  # type: ignore[attr-defined]
    fake_checkpoint_io = types.ModuleType("graspgenx.utils.checkpoint_io")
    fake_checkpoint_io.load_model_cfg = lambda *_args: object()  # type: ignore[attr-defined]
    fake_collision = types.ModuleType("graspgenx.utils.collision_filter")
    fake_collision.filter_colliding_grasps = object()  # type: ignore[attr-defined]

    for name, module in {
        "torch": fake_torch,
        "graspgenx": fake_package,
        "graspgenx.grasp_server": fake_server,
        "graspgenx.samplers": fake_samplers,
        "graspgenx.utils": fake_utils,
        "graspgenx.utils.checkpoint_io": fake_checkpoint_io,
        "graspgenx.utils.collision_filter": fake_collision,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    loaded = backend._get_loaded_backend()

    assert loaded["model"] is not None
    assert observed_environment == {
        "checkpoints": str(checkpoints.parent),
        "grippers": str(grippers),
    }


def test_run_planner_uses_fixed_graspmoe_configuration(tmp_path: Path) -> None:
    source, checkpoints, grippers = _backend_layout(tmp_path)
    backend = GraspGenXBackend(
        graspgenx_root=source,
        checkpoint_root=checkpoints,
        gripper_descriptions_root=grippers,
    )
    captured: dict[str, Any] = {}

    class FakeCuda:
        @staticmethod
        def device(_index: int):
            return contextlib.nullcontext()

    fake_torch = SimpleNamespace(
        cuda=FakeCuda(), inference_mode=lambda: contextlib.nullcontext()
    )

    def run_planner(points: np.ndarray, sampler: object, **kwargs: Any):
        captured["points"] = points
        captured["sampler"] = sampler
        captured.update(kwargs)
        return np.eye(4)[None, ...], np.array([0.5]), ["diff"], None

    backend._run_planner(
        loaded={
            "torch": fake_torch,
            "device_index": 0,
            "run_planner": run_planner,
        },
        sampler_entry={"sampler": "sampler"},
        object_points_aligned=np.ones((101, 3), dtype=np.float32),
    )

    assert captured["planner"] == "graspmoe"
    assert captured["grasp_threshold"] == -1.0
    assert captured["num_grasps"] == 200
    assert captured["topk_num_grasps"] == -1
    assert captured["moe_num_yaws"] == 36
    assert captured["moe_z_offsets_cm"] == (-2.0, 0.0)
    assert captured["moe_obb_density"] == "dense-topandside"


class _FakeBackend(GraspGenXBackend):
    def _get_loaded_backend(self) -> dict[str, Any]:
        return {
            "backend_commit": "a" * 40,
            "checkpoint_version": "release",
            "generator_checkpoint_sha256": "b" * 64,
            "discriminator_checkpoint_sha256": "c" * 64,
        }

    def _get_sampler_entry(
        self, _loaded: dict[str, Any], _gripper_name: str
    ) -> dict[str, Any]:
        return {"collision_surface_points": np.zeros((2000, 3), dtype=np.float32)}

    def _run_planner(self, **_kwargs: Any) -> tuple[Any, Any, Any]:
        grasps = np.tile(np.eye(4), (3, 1, 1))
        grasps[:, 2, 3] = 0.5
        return grasps, np.array([0.2, 0.9, 0.9]), ["diff", "obb", "diff"]


def test_backend_returns_ranked_contract_without_transport_payloads(
    tmp_path: Path,
) -> None:
    source, checkpoints, grippers = _backend_layout(tmp_path)
    backend = _FakeBackend(
        graspgenx_root=source,
        checkpoint_root=checkpoints,
        gripper_descriptions_root=grippers,
    )
    depth = np.full((11, 11), 500, dtype=np.uint16)
    mask = np.full((11, 11), 255, dtype=np.uint8)

    result = backend.predict_grasps(
        depth=_image_payload(depth),
        object_mask=_image_payload(mask),
        intrinsics=_intrinsics(),
        gripper_name="franka_panda",
        up_direction_camera=[0, 0, 1],
    )

    assert result["success"] is True
    details = result["details"]
    assert details["candidate_count"] == 3
    assert [item["score"] for item in details["grasp_candidates"]] == [0.9, 0.9, 0.2]
    assert [item["backend_index"] for item in details["grasp_candidates"]] == [1, 2, 0]
    assert details["metadata"]["collision_filter_applied"] is False
    assert details["metadata"]["generator_checkpoint_sha256"] == "b" * 64
    serialized = str(result)
    assert "base64" not in serialized
    assert "point_cloud" not in serialized


def test_collision_selection_checks_ranked_batches_until_twenty(
    tmp_path: Path,
) -> None:
    source, checkpoints, grippers = _backend_layout(tmp_path)
    backend = GraspGenXBackend(
        graspgenx_root=source,
        checkpoint_root=checkpoints,
        gripper_descriptions_root=grippers,
    )
    poses = np.tile(np.eye(4), (40, 1, 1))
    scores = np.linspace(0.0, 1.0, 40)
    calls: list[int] = []

    def filter_collisions(**kwargs: Any) -> np.ndarray:
        count = len(kwargs["grasp_poses"])
        calls.append(count)
        mask = np.ones(count, dtype=bool)
        mask[:2] = False
        return mask

    selected, metadata = backend._select_collision_free(
        loaded={"filter_collisions": filter_collisions},
        sampler_entry={
            "collision_surface_points": np.zeros((2000, 3), dtype=np.float32)
        },
        scene_points=np.ones((10, 3), dtype=np.float32),
        camera_native_grasps=poses,
        scores=scores,
    )

    assert len(selected) == MAX_RETURNED_CANDIDATES
    assert selected[0] == 37
    assert calls == [16, 16]
    assert metadata["collision_checked_count"] == 32
    assert metadata["collision_rejected_count"] == 4


def test_collision_selection_reports_all_grasps_colliding(tmp_path: Path) -> None:
    source, checkpoints, grippers = _backend_layout(tmp_path)
    backend = GraspGenXBackend(
        graspgenx_root=source,
        checkpoint_root=checkpoints,
        gripper_descriptions_root=grippers,
    )

    with pytest.raises(GraspGenXInputError, match="all_grasps_colliding") as raised:
        backend._select_collision_free(
            loaded={
                "filter_collisions": lambda **kwargs: np.zeros(
                    len(kwargs["grasp_poses"]), dtype=bool
                )
            },
            sampler_entry={
                "collision_surface_points": np.zeros((2000, 3), dtype=np.float32)
            },
            scene_points=np.ones((10, 3), dtype=np.float32),
            camera_native_grasps=np.tile(np.eye(4), (3, 1, 1)),
            scores=np.array([0.9, 0.8, 0.7]),
        )

    assert raised.value.metadata["collision_rejected_count"] == 3
    assert raised.value.metadata["returned_candidate_count"] == 0


def test_backend_failure_is_atomic_for_inconsistent_model_output(
    tmp_path: Path,
) -> None:
    class InvalidBackend(_FakeBackend):
        def _run_planner(self, **_kwargs: Any) -> tuple[Any, Any, Any]:
            grasps = np.eye(4)[None, ...]
            grasps[0, :3, :3] = np.diag([1.0, 1.0, -1.0])
            return grasps, np.array([0.5]), ["diff"]

    source, checkpoints, grippers = _backend_layout(tmp_path)
    backend = InvalidBackend(
        graspgenx_root=source,
        checkpoint_root=checkpoints,
        gripper_descriptions_root=grippers,
    )
    result = backend.predict_grasps(
        depth=_image_payload(np.full((11, 11), 500, dtype=np.uint16)),
        object_mask=_image_payload(np.full((11, 11), 255, dtype=np.uint8)),
        intrinsics=_intrinsics(),
        gripper_name="franka_panda",
        up_direction_camera=[0, 0, 1],
    )

    assert result["success"] is False
    assert result["details"]["reason"] == "inconsistent_grasp_outputs"
    assert result["details"]["candidate_count"] == 0
    assert result["details"]["grasp_candidates"] == []
