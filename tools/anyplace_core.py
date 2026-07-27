"""AnyPlace placement backend for the OpenETA AnyPlace MCP server."""

from __future__ import annotations

import base64
import contextlib
import io
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_NAME = "anyplace_multitask"
FRAME = "camera"
CAMERA_FRAME = "opencv"
POSE_CONVENTION = "p_placed = R @ p_current + t"
DEFAULT_CANDIDATE_LIMIT = 5
MODEL_SAMPLE_COUNT = 1024
DEFAULT_DEPTH_TRUNCATION = 2.0


class AnyPlaceInputError(Exception):
    """Input or normalized-output data cannot satisfy the AnyPlace contract."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class PointCloudLimits:
    """Point-count limits for one AnyPlace point cloud input."""

    min_points: int
    max_points: int


OBJECT_POINTCLOUD_LIMITS = PointCloudLimits(min_points=1024, max_points=200000)
PLACEMENT_REGION_POINTCLOUD_LIMITS = PointCloudLimits(min_points=1024, max_points=500000)


class _NoOpVisualizer:
    """Meshcat-compatible sink for official AnyPlace helpers that always draw."""

    def __getitem__(self, _name: str) -> "_NoOpVisualizer":
        return self

    def set_object(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def set_transform(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class AnyPlaceBackend:
    """Lazy in-process AnyPlace backend.

    This wrapper intentionally does not invoke official CLI scripts. It loads
    the AnyPlace model inside the MCP server process and calls the official
    internal policy function directly.
    """

    def __init__(
        self,
        *,
        anyplace_root: str | Path,
        config_path: str | Path,
        seed: int = 0,
        depth_truncation: float = DEFAULT_DEPTH_TRUNCATION,
    ) -> None:
        self.anyplace_root = Path(anyplace_root)
        self.config_path = Path(config_path)
        self.seed = seed
        self.depth_truncation = depth_truncation
        self._loaded: dict[str, Any] | None = None

    def predict_placement(
        self,
        *,
        rgb: dict[str, Any] | None,
        depth: dict[str, Any] | None,
        object_mask: dict[str, Any] | None,
        placement_region_mask: dict[str, Any] | None,
        intrinsics: dict[str, Any] | None,
        selected_grasp: dict[str, Any] | None,
        depth_truncation: float | None = None,
    ) -> dict[str, Any]:
        """Predict camera-frame placement poses from one aligned RGBD snapshot."""

        start = time.perf_counter()
        metadata = _metadata_base()
        try:
            resolved_depth_truncation = validate_depth_truncation(
                depth_truncation,
                default_truncation=self.depth_truncation,
            )
            metadata["depth_truncation"] = resolved_depth_truncation
            parsed_intrinsics = validate_intrinsics(intrinsics)
            np, Image = _load_numeric_deps()
            rgb_array = _decode_image_payload(
                rgb,
                Image=Image,
                np=np,
                convert="RGB",
                missing_reason="missing_rgb",
                decode_reason="rgb_decode_failed",
            )
            depth_array = _decode_image_payload(
                depth,
                Image=Image,
                np=np,
                convert=None,
                missing_reason="missing_depth",
                decode_reason="depth_decode_failed",
            )
            object_mask_array = _decode_image_payload(
                object_mask,
                Image=Image,
                np=np,
                convert="L",
                missing_reason="missing_object_mask",
                decode_reason="object_mask_decode_failed",
            )
            placement_mask_array = _decode_image_payload(
                placement_region_mask,
                Image=Image,
                np=np,
                convert="L",
                missing_reason="missing_placement_region_mask",
                decode_reason="placement_region_mask_decode_failed",
            )
            object_pcd, placement_pcd = build_masked_pointclouds_from_rgbd(
                rgb=rgb_array,
                depth=depth_array,
                object_mask=object_mask_array,
                placement_region_mask=placement_mask_array,
                intrinsics=parsed_intrinsics,
                depth_truncation=resolved_depth_truncation,
            )
            parsed_grasp = normalise_selected_grasp(selected_grasp)
            metadata.update(
                {
                    "object_point_count": int(object_pcd.shape[0]),
                    "placement_region_point_count": int(placement_pcd.shape[0]),
                }
            )
        except AnyPlaceInputError as exc:
            return _failure_result(
                reason=exc.reason,
                content=f"AnyPlace placement prediction failed: {exc.reason}.",
                metadata=_with_duration(metadata, start),
            )

        try:
            backend = self._get_loaded_backend()
        except AnyPlaceInputError as exc:
            return _failure_result(
                reason=exc.reason,
                content=f"AnyPlace placement prediction failed: {exc.reason}.",
                metadata=_with_duration(metadata, start),
            )
        except Exception as exc:  # noqa: BLE001
            return _failure_result(
                reason="model_load_failed",
                content=f"AnyPlace placement prediction failed: model load failed: {exc}",
                metadata=_with_duration(
                    {**metadata, "error_type": type(exc).__name__},
                    start,
                ),
            )

        try:
            raw_candidates = self._predict_with_loaded_backend(
                backend=backend,
                object_pcd=object_pcd,
                placement_region_pcd=placement_pcd,
            )
            candidates = normalise_placement_candidates(
                raw_candidates,
                selected_grasp=parsed_grasp,
                expected_count=DEFAULT_CANDIDATE_LIMIT,
            )
        except AnyPlaceInputError as exc:
            return _failure_result(
                reason=exc.reason,
                content=f"AnyPlace placement prediction failed: {exc.reason}.",
                metadata=_with_duration(metadata, start),
            )
        except Exception as exc:  # noqa: BLE001
            return _failure_result(
                reason="model_inference_failed",
                content=f"AnyPlace placement prediction failed: model inference failed: {exc}",
                metadata=_with_duration(
                    {**metadata, "error_type": type(exc).__name__},
                    start,
                ),
            )

        return {
            "success": True,
            "content": "AnyPlace placement prediction completed.",
            "details": {
                "tool": "anyplace",
                "backend": "anyplace_mcp",
                "model": MODEL_NAME,
                "frame": FRAME,
                "camera_frame": CAMERA_FRAME,
                "candidate_count": len(candidates),
                "placement_candidates": candidates,
                "metadata": _with_duration(metadata, start),
            },
        }

    def _get_loaded_backend(self) -> dict[str, Any]:
        if self._loaded is not None:
            return self._loaded
        if not self.anyplace_root.exists():
            raise RuntimeError(f"anyplace root does not exist: {self.anyplace_root}")
        if not self.config_path.exists():
            raise RuntimeError(f"config path does not exist: {self.config_path}")

        if str(self.anyplace_root) not in sys.path:
            sys.path.insert(0, str(self.anyplace_root))
        anyplace_source_dir = self.anyplace_root / "anyplace"
        os.environ["ANYPLACE_SOURCE_DIR"] = str(anyplace_source_dir)
        os.environ.setdefault("ANYPLACE_DATA_DIR", str(anyplace_source_dir))

        import torch

        if not torch.cuda.is_available():
            raise AnyPlaceInputError("device_unavailable")

        import numpy as np
        from anyplace.model.transformer.policy import NSMTransformerImplicit
        from anyplace.model.transformer.policy import NSMTransformerSingleTransformationRegression
        from anyplace.utils import config_util, util
        from anyplace.utils.anyplace.multistep_pose_regression_anyplace import (
            policy_inference_methods_dict,
        )
        from anyplace.utils.mesh_util import three_util

        args = config_util.load_config(str(self.config_path), demo_train_eval="eval")
        args = config_util.recursive_attr_dict(args)
        ckpt_path = Path(args.experiment.eval.ckpt_path)
        if not ckpt_path.is_absolute():
            ckpt_path = self.config_path.parent / ckpt_path
        if not ckpt_path.exists():
            raise RuntimeError(f"checkpoint path does not exist: {ckpt_path}")

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        with contextlib.redirect_stdout(sys.stderr):
            pose_refine_ckpt = torch.load(
                str(ckpt_path),
                map_location=torch.device("cpu"),
                weights_only=False,
            )
            config_util.update_recursive(
                args.model.refine_pose,
                config_util.recursive_attr_dict(pose_refine_ckpt["args"]["model"]["refine_pose"]),
            )
            pr_type = args.model.refine_pose.type
            pr_args = config_util.copy_attr_dict(args.model[pr_type])
            if args.model.refine_pose.get("model_kwargs") is not None:
                custom_pr_args = args.model.refine_pose.model_kwargs[pr_type]
                config_util.update_recursive(pr_args, custom_pr_args)

            if pr_type == "nsm_transformer":
                pr_model_cls = NSMTransformerSingleTransformationRegression
                pr_model = pr_model_cls(
                    mc_vis=_NoOpVisualizer(),
                    feat_dim=args.model.refine_pose.feat_dim,
                    **pr_args,
                ).cuda()
            elif pr_type == "nsm_implicit":
                pr_model_cls = NSMTransformerImplicit
                pr_model = pr_model_cls(
                    mc_vis=_NoOpVisualizer(),
                    feat_dim=args.model.refine_pose.feat_dim,
                    is_train=False,
                    **pr_args,
                ).cuda()
            else:
                raise RuntimeError(f"unsupported AnyPlace refine pose type: {pr_type}")
            pr_model.load_state_dict(pose_refine_ckpt["refine_pose_model_state_dict"])
            pr_model.eval()
            if hasattr(pr_model, "eval_sample"):
                pr_model.set_eval_sample(True)

        reso_grid = args.data.voxel_grid.reso_grid
        padding_grid = args.data.voxel_grid.padding
        raster_pts = three_util.get_raster_points(reso_grid, padding=padding_grid)
        raster_pts = raster_pts.reshape(reso_grid, reso_grid, reso_grid, 3)
        raster_pts = raster_pts.transpose(2, 1, 0, 3)
        raster_pts = raster_pts.reshape(-1, 3)
        rot_grid = util.generate_healpix_grid(size=args.data.rot_grid_samples)
        args.data.rot_grid_bins = rot_grid.shape[0]
        scene_extents = np.asarray(args.data.coarse_aff.scene_extents, dtype=np.float64)
        args.data.coarse_aff.scene_scale = 1.0 / float(np.max(scene_extents))

        infer_relation_policy = policy_inference_methods_dict[
            args.experiment.eval.inference_method
        ]

        self._loaded = {
            "args": args,
            "pr_model": pr_model,
            "raster_pts": raster_pts,
            "rot_grid": rot_grid,
            "infer_relation_policy": infer_relation_policy,
            "torch": torch,
        }
        return self._loaded

    def _predict_with_loaded_backend(
        self,
        *,
        backend: dict[str, Any],
        object_pcd: Any,
        placement_region_pcd: Any,
    ) -> Any:
        args = backend["args"]
        infer_relation_policy = backend["infer_relation_policy"]
        torch = backend["torch"]
        exp_args = args.experiment
        infer_kwargs: dict[str, Any] = {
            "gt_child_cent": None,
            "export_viz": False,
            "export_viz_dirname": None,
            "export_viz_relative_trans_guess": None,
            "compute_coverage_scores": False,
            "out_coverage_dirname1": None,
            "out_coverage_dirname2": None,
            "iteration": 0,
        }
        if getattr(exp_args.eval, "multi_aff_rot", False):
            infer_kwargs["multi_aff_rot"] = True

        multi_mesh_dict = {
            "parent_file": None,
            "parent_scale": None,
            "parent_pose": None,
            "child_file": None,
            "child_scale": None,
            "child_pose": None,
            "multi": True,
        }

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        with torch.no_grad(), contextlib.redirect_stdout(sys.stderr):
            relative_trans_preds = infer_relation_policy(
                _NoOpVisualizer(),
                placement_region_pcd,
                object_pcd,
                None,
                backend["pr_model"],
                None,
                scene_mean=args.data.coarse_aff.scene_mean,
                scene_scale=args.data.coarse_aff.scene_scale,
                grid_pts=backend["raster_pts"],
                rot_grid=backend["rot_grid"],
                viz=False,
                n_iters=exp_args.eval.n_refine_iters,
                no_parent_crop=(not exp_args.parent_crop),
                return_top=(not exp_args.eval.return_rand),
                with_coll=exp_args.eval.with_coll,
                run_affordance=exp_args.eval.run_affordance,
                init_k_val=exp_args.eval.init_k_val,
                no_sc_score=exp_args.eval.no_success_classifier,
                init_parent_mean=exp_args.eval.init_parent_mean_pos,
                init_orig_ori=exp_args.eval.init_orig_ori,
                refine_anneal=exp_args.eval.refine_anneal,
                mesh_dict=multi_mesh_dict,
                add_per_iter_noise=exp_args.eval.add_per_iter_noise,
                per_iter_noise_kwargs=exp_args.eval.per_iter_noise_kwargs,
                variable_size_crop=exp_args.eval.variable_size_crop,
                timestep_emb_decay_factor=exp_args.eval.timestep_emb_decay_factor,
                remove_redundant_pose=exp_args.eval.remove_redundant_pose,
                **infer_kwargs,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        return relative_trans_preds


def validate_intrinsics(intrinsics: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(intrinsics, dict):
        raise AnyPlaceInputError("missing_intrinsics")
    parsed: dict[str, float] = {}
    for key in ("fx", "fy", "cx", "cy", "scale"):
        if key not in intrinsics:
            raise AnyPlaceInputError("invalid_intrinsics")
        try:
            value = float(intrinsics[key])
        except (TypeError, ValueError) as exc:
            raise AnyPlaceInputError("invalid_intrinsics") from exc
        if key in {"fx", "fy", "scale"} and value <= 0:
            raise AnyPlaceInputError("invalid_intrinsics")
        parsed[key] = value
    return parsed


def validate_depth_truncation(
    depth_truncation: Any,
    *,
    default_truncation: float,
) -> float:
    """Resolve the AnyPlace far depth cut for one call."""

    if depth_truncation is None:
        far = float(default_truncation)
    else:
        if isinstance(depth_truncation, bool) or not isinstance(
            depth_truncation,
            (int, float),
        ):
            raise AnyPlaceInputError("invalid_depth_truncation")
        far = float(depth_truncation)
    if not math.isfinite(far) or far <= 0:
        raise AnyPlaceInputError("invalid_depth_truncation")
    return far


def build_masked_pointclouds_from_rgbd(
    *,
    rgb: Any,
    depth: Any,
    object_mask: Any,
    placement_region_mask: Any,
    intrinsics: dict[str, float],
    depth_truncation: float,
) -> tuple[Any, Any]:
    """Project aligned RGBD masks into independent camera-frame point clouds."""

    np, _Image = _load_numeric_deps()
    rgb_array = np.asarray(rgb)
    depth_array = np.asarray(depth)
    object_mask_2d = np.asarray(object_mask) > 0
    placement_mask_2d = np.asarray(placement_region_mask) > 0

    if rgb_array.ndim != 3 or rgb_array.shape[2] < 3:
        raise AnyPlaceInputError("image_shape_mismatch")
    if depth_array.ndim != 2:
        raise AnyPlaceInputError("unsupported_depth_format")
    if rgb_array.shape[:2] != depth_array.shape:
        raise AnyPlaceInputError("image_shape_mismatch")
    if object_mask_2d.shape != depth_array.shape or placement_mask_2d.shape != depth_array.shape:
        raise AnyPlaceInputError("image_shape_mismatch")
    if not object_mask_2d.any():
        raise AnyPlaceInputError("empty_object_mask")
    if not placement_mask_2d.any():
        raise AnyPlaceInputError("empty_placement_region_mask")

    height, width = depth_array.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    points_z = depth_array.astype(np.float32) / float(intrinsics["scale"])
    points_x = (u - intrinsics["cx"]) / intrinsics["fx"] * points_z
    points_y = (v - intrinsics["cy"]) / intrinsics["fy"] * points_z
    points = np.stack([points_x, points_y, points_z], axis=-1)
    valid_depth = (points_z > 0) & (points_z < depth_truncation)

    object_points = points[valid_depth & object_mask_2d].astype(np.float32)
    placement_points = points[valid_depth & placement_mask_2d].astype(np.float32)
    object_points = validate_pointcloud_array(
        object_points,
        limits=OBJECT_POINTCLOUD_LIMITS,
        empty_reason="empty_object_pointcloud",
        too_small_reason="object_pointcloud_too_small",
    )
    placement_points = validate_pointcloud_array(
        placement_points,
        limits=PLACEMENT_REGION_POINTCLOUD_LIMITS,
        empty_reason="empty_placement_region_pointcloud",
        too_small_reason="placement_region_pointcloud_too_small",
    )
    return object_points, placement_points


def validate_pointcloud_array(
    array: Any,
    *,
    limits: PointCloudLimits,
    empty_reason: str = "empty_pointcloud",
    too_small_reason: str = "pointcloud_too_small",
) -> Any:
    np, _Image = _load_numeric_deps()
    if not isinstance(array, np.ndarray):
        raise AnyPlaceInputError("invalid_pointcloud_shape")
    if array.ndim != 2 or array.shape[1] != 3:
        raise AnyPlaceInputError("invalid_pointcloud_shape")
    point_count = int(array.shape[0])
    if point_count == 0:
        raise AnyPlaceInputError(empty_reason)
    if point_count < limits.min_points:
        raise AnyPlaceInputError(too_small_reason)
    if point_count > limits.max_points:
        raise AnyPlaceInputError("request_too_large")
    if array.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise AnyPlaceInputError("invalid_pointcloud_shape")
    if not np.isfinite(array).all():
        raise AnyPlaceInputError("invalid_pointcloud_shape")
    return array.astype(np.float32, copy=False)


def normalise_selected_grasp(selected_grasp: Any) -> dict[str, Any]:
    np, _Image = _load_numeric_deps()
    if not isinstance(selected_grasp, dict):
        raise AnyPlaceInputError("missing_selected_grasp")
    if selected_grasp.get("frame") != FRAME or selected_grasp.get("camera_frame") != CAMERA_FRAME:
        raise AnyPlaceInputError("invalid_selected_grasp")
    grasp_id = selected_grasp.get("id")
    if not isinstance(grasp_id, str) or not grasp_id:
        raise AnyPlaceInputError("invalid_selected_grasp")
    try:
        rotation = np.asarray(selected_grasp["rotation_matrix"], dtype=np.float64)
        translation = np.asarray(selected_grasp["translation_xyz"], dtype=np.float64)
        tip = np.asarray(selected_grasp["gripper_tip_position_xyz"], dtype=np.float64)
        score = float(selected_grasp["score"])
        depth = float(selected_grasp["depth"])
        width = float(selected_grasp["width"])
        height = float(selected_grasp["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnyPlaceInputError("invalid_selected_grasp") from exc
    if rotation.shape != (3, 3) or translation.shape != (3,) or tip.shape != (3,):
        raise AnyPlaceInputError("invalid_selected_grasp")
    if not all(np.isfinite(value).all() for value in (rotation, translation, tip)):
        raise AnyPlaceInputError("invalid_selected_grasp")
    if not np.isfinite([score, depth, width, height]).all():
        raise AnyPlaceInputError("invalid_selected_grasp")
    if depth < 0 or width < 0 or height < 0 or not _is_rotation_matrix(rotation):
        raise AnyPlaceInputError("invalid_selected_grasp")
    return {
        "id": grasp_id,
        "frame": FRAME,
        "camera_frame": CAMERA_FRAME,
        "score": score,
        "translation_xyz": _float_list(translation),
        "rotation_matrix": _float_matrix(rotation),
        "depth": depth,
        "width": width,
        "height": height,
        "gripper_tip_position_xyz": _float_list(tip),
    }


def normalise_placement_candidates(
    raw_candidates: Any,
    *,
    selected_grasp: dict[str, Any],
    expected_count: int = DEFAULT_CANDIDATE_LIMIT,
) -> list[dict[str, Any]]:
    np, _Image = _load_numeric_deps()
    try:
        arr = np.asarray(raw_candidates, dtype=np.float64)
    except Exception as exc:  # noqa: BLE001
        raise AnyPlaceInputError("inconsistent_placement_outputs") from exc
    if arr.size == 0:
        raise AnyPlaceInputError("no_placement_candidates")
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise AnyPlaceInputError("inconsistent_placement_outputs")
    if arr.shape[0] != expected_count:
        raise AnyPlaceInputError("inconsistent_placement_outputs")
    if not np.isfinite(arr).all():
        raise AnyPlaceInputError("inconsistent_placement_outputs")
    grasp_rotation = np.asarray(selected_grasp["rotation_matrix"], dtype=np.float64)
    grasp_translation = np.asarray(selected_grasp["translation_xyz"], dtype=np.float64)
    grasp_tip = np.asarray(selected_grasp["gripper_tip_position_xyz"], dtype=np.float64)
    candidates = []
    for idx, pose in enumerate(arr):
        if not _is_rigid_transform(pose):
            raise AnyPlaceInputError("inconsistent_placement_outputs")
        placement_rotation = pose[:3, :3]
        placement_translation = pose[:3, 3]
        place_rotation = placement_rotation @ grasp_rotation
        place_translation = placement_rotation @ grasp_translation + placement_translation
        place_tip = placement_rotation @ grasp_tip + placement_translation
        candidates.append(
            {
                "id": f"placement_{idx:03d}",
                "source_grasp_id": selected_grasp["id"],
                "object_placement_transform": {
                    "frame": FRAME,
                    "camera_frame": CAMERA_FRAME,
                    "convention": POSE_CONVENTION,
                    "transform_matrix": _float_matrix(pose),
                },
                "place_grasp_pose": {
                    "id": f"place_grasp_{idx:03d}",
                    "frame": FRAME,
                    "camera_frame": CAMERA_FRAME,
                    "score": selected_grasp["score"],
                    "translation_xyz": _float_list(place_translation),
                    "rotation_matrix": _float_matrix(place_rotation),
                    "depth": selected_grasp["depth"],
                    "width": selected_grasp["width"],
                    "height": selected_grasp["height"],
                    "gripper_tip_position_xyz": _float_list(place_tip),
                },
            }
        )
    return candidates


def _is_rigid_transform(matrix: Any) -> bool:
    np, _Image = _load_numeric_deps()
    return bool(
        matrix.shape == (4, 4)
        and np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        and _is_rotation_matrix(matrix[:3, :3])
    )


def _is_rotation_matrix(rotation: Any) -> bool:
    np, _Image = _load_numeric_deps()
    return bool(
        rotation.shape == (3, 3)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    )


def _decode_image_payload(
    payload: dict[str, Any] | None,
    *,
    Image: Any,
    np: Any,
    convert: str | None,
    missing_reason: str,
    decode_reason: str,
) -> Any:
    if not isinstance(payload, dict) or not payload.get("base64"):
        raise AnyPlaceInputError(missing_reason)
    try:
        data = base64.b64decode(payload["base64"], validate=True)
        image = Image.open(io.BytesIO(data))
        if convert:
            image = image.convert(convert)
        return np.asarray(image)
    except Exception as exc:  # noqa: BLE001
        raise AnyPlaceInputError(decode_reason) from exc


def _load_numeric_deps() -> tuple[Any, Any]:
    import numpy as np
    from PIL import Image

    return np, Image


def _float_list(values: Any) -> list[float]:
    return [float(value) for value in values]


def _float_matrix(values: Any) -> list[list[float]]:
    return [_float_list(row) for row in values]


def _metadata_base() -> dict[str, Any]:
    return {
        "pose_convention": POSE_CONVENTION,
        "candidate_limit": DEFAULT_CANDIDATE_LIMIT,
        "model_sample_count": MODEL_SAMPLE_COUNT,
    }


def _with_duration(metadata: dict[str, Any], start: float) -> dict[str, Any]:
    return {**metadata, "duration_s": round(time.perf_counter() - start, 4)}


def _failure_result(
    *,
    reason: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "success": False,
        "content": content,
        "details": {
            "tool": "anyplace",
            "backend": "anyplace_mcp",
            "model": MODEL_NAME,
            "frame": FRAME,
            "camera_frame": CAMERA_FRAME,
            "candidate_count": 0,
            "placement_candidates": [],
            "reason": reason,
            "metadata": dict(metadata or {}),
        },
    }
