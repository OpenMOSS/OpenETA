"""AnyGrasp grasp detection backend for the OpenETA AnyGrasp MCP server."""

from __future__ import annotations

import base64
import io
import math
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any


class AnyGraspInputError(Exception):
    """Input or normalized-output data cannot satisfy the AnyGrasp contract."""

    def __init__(self, reason: str, *, metadata: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.metadata = dict(metadata or {})


_DEPTH_SCALE_GUIDANCE = (
    "Depth in meters is raw_depth / intrinsics.scale; for uint16 millimeter "
    "depth, use scale=1000."
)


class AnyGraspBackend:
    """Lazy AnyGrasp SDK wrapper.

    Robot/camera/backend configuration is injected here, not passed by the
    planner-facing OpenETA tool call.
    """

    def __init__(
        self,
        *,
        sdk_root: str | Path,
        checkpoint_path: str | Path,
        max_gripper_width: float = 0.1,
        gripper_height: float = 0.03,
        depth_truncation: float = 1.0,
        max_candidates: int = 20,
    ) -> None:
        self.sdk_root = Path(sdk_root)
        self.checkpoint_path = Path(checkpoint_path)
        self.max_gripper_width = max_gripper_width
        self.gripper_height = gripper_height
        self.depth_truncation = depth_truncation
        self.max_candidates = max_candidates
        self._detector: Any | None = None

    def detect_grasps(
        self,
        *,
        rgb: dict[str, Any],
        depth: dict[str, Any],
        intrinsics: dict[str, Any],
        mode: str = "targeted",
        target_mask: dict[str, Any] | None = None,
        approach_steering: list[float] | None = None,
        approach_thresh: float | None = None,
        collision_detection: bool = True,
        dense_grasp: bool = False,
        depth_truncation: float | None = None,
        depth_min: float | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        metadata = self._metadata(
            mode=mode,
            collision_detection=collision_detection,
            dense_grasp=dense_grasp,
        )
        try:
            validate_mode(mode=mode, target_mask=target_mask)
            validate_detect_grasps_options(
                collision_detection=collision_detection,
                dense_grasp=dense_grasp,
                approach_steering=approach_steering,
                approach_thresh=approach_thresh,
            )
            resolved_truncation, resolved_depth_min = validate_depth_bounds(
                depth_truncation=depth_truncation,
                depth_min=depth_min,
                default_truncation=self.depth_truncation,
            )
            parsed_intrinsics = validate_intrinsics(intrinsics)
            metadata["intrinsics"] = parsed_intrinsics
            np, Image = _load_numeric_deps()
            rgb_array = _decode_image_payload(rgb, Image=Image, np=np, convert="RGB", reason="rgb_decode_failed")
            depth_array = _decode_image_payload(depth, Image=Image, np=np, convert=None, reason="depth_decode_failed")
            mask_array = (
                _decode_image_payload(
                    target_mask,
                    Image=Image,
                    np=np,
                    convert="L",
                    reason="target_mask_decode_failed",
                )
                if target_mask is not None
                else None
            )
            points, _colors, region_steering, depth_metadata = build_point_cloud_from_rgbd(
                rgb=rgb_array,
                depth=depth_array,
                target_mask=mask_array,
                intrinsics=parsed_intrinsics,
                mode=mode,
                depth_truncation=resolved_truncation,
                workspace_limits=None,
                depth_min=resolved_depth_min,
            )
            metadata.update(depth_metadata)
            metadata["point_count"] = int(points.shape[0])
            if region_steering is not None:
                metadata["region_point_count"] = int(region_steering.sum())

            try:
                detector = self._get_detector()
            except Exception as exc:  # noqa: BLE001
                return _failure_result(
                    reason="model_load_failed",
                    content=f"AnyGrasp grasp detection failed: model load failed: {exc}",
                    metadata=_with_duration(
                        {**metadata, "error_type": type(exc).__name__},
                        start,
                    ),
                    mode=mode,
                )
            params: dict[str, Any] = {
                "dense_grasp": dense_grasp,
                "collision_detection": collision_detection,
                "region_steering": region_steering,
                "approach_steering": approach_steering,
                "approach_thresh": np.pi if approach_thresh is None else approach_thresh,
            }
            grasps = detector.get_grasp(points, params)
            if grasps is None:
                return _failure_result(
                    reason="no_grasp_candidates",
                    content="AnyGrasp returned no grasp candidates.",
                    metadata=_with_duration(metadata, start),
                    mode=mode,
                )
            if not dense_grasp:
                grasps = grasps.nms()
            candidates = normalise_grasp_candidates(grasps)[: self.max_candidates]
            if not candidates:
                return _failure_result(
                    reason="no_grasp_candidates",
                    content="AnyGrasp returned no grasp candidates.",
                    metadata=_with_duration(metadata, start),
                    mode=mode,
                )
        except AnyGraspInputError as exc:
            metadata.update(exc.metadata)
            return _failure_result(
                reason=exc.reason,
                content=_input_failure_content(exc.reason),
                metadata=_with_duration(metadata, start),
                mode=mode,
            )
        except Exception as exc:  # noqa: BLE001 - backend failures are returned structurally.
            return _failure_result(
                reason="model_inference_failed",
                content=f"AnyGrasp grasp detection failed: {exc}",
                metadata=_with_duration(
                    {**metadata, "error_type": type(exc).__name__},
                    start,
                ),
                mode=mode,
            )

        return {
            "success": True,
            "content": "AnyGrasp grasp detection completed.",
            "details": {
                "tool": "anygrasp",
                "backend": "anygrasp_mcp",
                "model": "anygrasp_sdk",
                "mode": mode,
                "candidate_count": len(candidates),
                "grasp_candidates": candidates,
                "ranking": "score_descending",
                "artifacts": [],
                "metadata": _with_duration(metadata, start),
            },
        }

    def _get_detector(self) -> Any:
        if self._detector is not None:
            return self._detector
        detection_root = self.sdk_root / "grasp_detection"
        if str(detection_root) not in sys.path:
            sys.path.insert(0, str(detection_root))
        try:
            from gsnet import create_detector
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"failed to import AnyGrasp SDK: {exc}") from exc
        detector = create_detector(
            Namespace(
                checkpoint_path=str(self.checkpoint_path),
                max_gripper_width=self.max_gripper_width,
                gripper_height=self.gripper_height,
            )
        )
        if detector is None:
            raise RuntimeError("failed to create AnyGrasp detector")
        self._detector = detector
        return detector

    def _metadata(
        self,
        *,
        mode: str,
        collision_detection: bool,
        dense_grasp: bool,
    ) -> dict[str, Any]:
        return {
            "frame": "camera",
            "camera_frame": "opencv",
            "mode": mode,
            "intrinsics": {},
            "collision_detection": collision_detection,
            "dense_grasp": dense_grasp,
            "max_gripper_width": self.max_gripper_width,
            "gripper_height": self.gripper_height,
            "workspace_limits": None,
            "depth_truncation": self.depth_truncation,
            "max_candidates": self.max_candidates,
        }


def validate_mode(*, mode: str, target_mask: dict[str, Any] | None) -> None:
    if mode not in {"targeted", "scene"}:
        raise AnyGraspInputError("invalid_mode")
    if mode == "targeted" and target_mask is None:
        raise AnyGraspInputError("missing_target_mask")
    if mode == "scene" and target_mask is not None:
        raise AnyGraspInputError("target_mask_not_allowed_in_scene_mode")


def validate_intrinsics(intrinsics: dict[str, Any]) -> dict[str, float]:
    if not isinstance(intrinsics, dict):
        raise AnyGraspInputError("missing_intrinsics")
    parsed: dict[str, float] = {}
    for key in ("fx", "fy", "cx", "cy"):
        if key not in intrinsics:
            raise AnyGraspInputError("invalid_intrinsics")
        try:
            value = float(intrinsics[key])
        except (TypeError, ValueError) as exc:
            raise AnyGraspInputError("invalid_intrinsics") from exc
        if not math.isfinite(value) or (key in {"fx", "fy"} and value <= 0):
            raise AnyGraspInputError("invalid_intrinsics")
        parsed[key] = value
    if "scale" not in intrinsics:
        raise AnyGraspInputError("invalid_depth_scale")
    try:
        scale = float(intrinsics["scale"])
    except (TypeError, ValueError) as exc:
        raise AnyGraspInputError("invalid_depth_scale") from exc
    if not math.isfinite(scale) or scale <= 0:
        raise AnyGraspInputError("invalid_depth_scale")
    parsed["scale"] = scale
    return parsed


def validate_detect_grasps_options(
    *,
    collision_detection: Any,
    dense_grasp: Any,
    approach_steering: Any = None,
    approach_thresh: Any = None,
) -> None:
    if not isinstance(collision_detection, bool) or not isinstance(dense_grasp, bool):
        raise AnyGraspInputError("invalid_option")
    if approach_steering is not None:
        if not isinstance(approach_steering, list) or len(approach_steering) != 3:
            raise AnyGraspInputError("invalid_approach_steering")
        try:
            [float(value) for value in approach_steering]
        except (TypeError, ValueError) as exc:
            raise AnyGraspInputError("invalid_approach_steering") from exc
    if approach_thresh is not None and not isinstance(approach_thresh, (int, float)):
        raise AnyGraspInputError("invalid_option")


def validate_depth_bounds(
    *,
    depth_truncation: Any,
    depth_min: Any,
    default_truncation: float,
) -> tuple[float, float]:
    """Resolve per-call near/far depth cuts, falling back to the backend default.

    ``depth_truncation`` overrides the server-configured far cut for this call
    only; ``depth_min`` adds a near cut (metres) to drop sub-working-distance
    noise. Both are optional. The near cut must be finite and non-negative, the
    far cut strictly positive, and near must be strictly below far so the valid
    band is non-empty.
    """

    if depth_truncation is None:
        far = float(default_truncation)
    else:
        if isinstance(depth_truncation, bool) or not isinstance(
            depth_truncation, (int, float)
        ):
            raise AnyGraspInputError("invalid_depth_truncation")
        far = float(depth_truncation)
        if not math.isfinite(far) or far <= 0:
            raise AnyGraspInputError("invalid_depth_truncation")

    if depth_min is None:
        near = 0.0
    else:
        if isinstance(depth_min, bool) or not isinstance(depth_min, (int, float)):
            raise AnyGraspInputError("invalid_depth_min")
        near = float(depth_min)
        if not math.isfinite(near) or near < 0:
            raise AnyGraspInputError("invalid_depth_min")

    if near >= far:
        raise AnyGraspInputError("invalid_depth_bounds")
    return far, near


def build_point_cloud_from_rgbd(
    *,
    rgb: Any,
    depth: Any,
    target_mask: Any | None,
    intrinsics: dict[str, float],
    mode: str,
    depth_truncation: float,
    workspace_limits: list[float] | None,
    depth_min: float = 0.0,
) -> tuple[Any, Any, Any | None, dict[str, Any]]:
    np, _Image = _load_numeric_deps()
    rgb_array = np.asarray(rgb)
    depth_array = np.asarray(depth)
    if rgb_array.ndim != 3 or rgb_array.shape[2] < 3:
        raise AnyGraspInputError("image_shape_mismatch")
    if depth_array.ndim != 2:
        raise AnyGraspInputError("unsupported_depth_format")
    if rgb_array.shape[:2] != depth_array.shape:
        raise AnyGraspInputError("image_shape_mismatch")

    points_z, valid, depth_metadata = _inspect_depth(
        depth_array=depth_array,
        scale=intrinsics["scale"],
        depth_truncation=depth_truncation,
        depth_min=depth_min,
    )

    target_mask_2d = None
    if target_mask is not None:
        target_mask_2d = np.asarray(target_mask) > 0
        if target_mask_2d.shape != depth_array.shape:
            raise AnyGraspInputError("image_shape_mismatch")
        if not target_mask_2d.any():
            raise AnyGraspInputError("empty_target_mask")

    height, width = depth_array.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    points_x = (u - intrinsics["cx"]) / intrinsics["fx"] * points_z
    points_y = (v - intrinsics["cy"]) / intrinsics["fy"] * points_z
    points = np.stack([points_x, points_y, points_z], axis=-1)[valid].astype(np.float32)
    colors = (rgb_array[..., :3].astype(np.float32) / 255.0)[valid]

    region_steering = None
    if mode == "targeted":
        if target_mask_2d is None:
            raise AnyGraspInputError("missing_target_mask")
        region_steering = target_mask_2d[valid].astype(bool)
        if not region_steering.any():
            raise AnyGraspInputError("empty_target_mask")

    if workspace_limits is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = workspace_limits
        workspace_mask = (
            (points[:, 0] >= xmin)
            & (points[:, 0] <= xmax)
            & (points[:, 1] >= ymin)
            & (points[:, 1] <= ymax)
            & (points[:, 2] >= zmin)
            & (points[:, 2] <= zmax)
        )
        region_steering = (
            workspace_mask if region_steering is None else region_steering & workspace_mask
        )
        if not region_steering.any():
            raise AnyGraspInputError("empty_target_mask")

    return points, colors, region_steering, depth_metadata


def _inspect_depth(
    *,
    depth_array: Any,
    scale: float,
    depth_truncation: float,
    depth_min: float = 0.0,
) -> tuple[Any, Any, dict[str, Any]]:
    raw_min = depth_array.min().item()
    raw_max = depth_array.max().item()
    points_z = depth_array.astype("float32") / float(scale)
    # Near cut removes sub-working-distance noise (e.g. the D435 dead zone),
    # far cut removes stray long-range returns before they pollute the cloud.
    valid = (points_z > float(depth_min)) & (points_z < depth_truncation)
    metadata = {
        "depth_dtype": str(depth_array.dtype),
        "depth_raw_min": raw_min,
        "depth_raw_max": raw_max,
        "depth_metric_min": float(raw_min) / float(scale),
        "depth_metric_max": float(raw_max) / float(scale),
        "depth_min": float(depth_min),
        "depth_truncation": float(depth_truncation),
        "valid_point_count": int(valid.sum()),
    }
    is_uint16 = depth_array.dtype.kind == "u" and depth_array.dtype.itemsize == 2
    if is_uint16 and raw_max > 10 and scale <= 1:
        raise AnyGraspInputError("depth_scale_mismatch", metadata=metadata)
    if metadata["valid_point_count"] == 0:
        raise AnyGraspInputError(
            "empty_point_cloud_after_depth_filter",
            metadata=metadata,
        )
    return points_z, valid, metadata


def _input_failure_content(reason: str) -> str:
    content = f"AnyGrasp grasp detection failed: {reason}."
    if reason in {
        "invalid_depth_scale",
        "depth_scale_mismatch",
        "empty_point_cloud_after_depth_filter",
    }:
        return f"{content} {_DEPTH_SCALE_GUIDANCE}"
    return content


def normalise_grasp_candidates(grasps: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for idx in range(len(grasps)):
        grasp = grasps[idx]
        candidate = _candidate_from_grasp(idx, grasp)
        candidate["backend_index"] = idx
        candidates.append(candidate)
    candidates.sort(key=_candidate_score_sort_key)
    for rank, candidate in enumerate(candidates):
        candidate["id"] = f"grasp_{rank:03d}"
        candidate["rank"] = rank
    return candidates


def _candidate_score_sort_key(candidate: dict[str, Any]) -> float:
    score = float(candidate["score"])
    return -score if math.isfinite(score) else float("inf")


def _candidate_from_grasp(idx: int, grasp: Any) -> dict[str, Any]:
    np, _Image = _load_numeric_deps()
    rotation = np.asarray(grasp.rotation_matrix, dtype=np.float64)
    translation = np.asarray(grasp.translation, dtype=np.float64)
    depth = float(grasp.depth)
    tip_position = translation + depth * rotation[:3, 0]
    return {
        "id": f"grasp_{idx:03d}",
        "frame": "camera",
        "camera_frame": "opencv",
        "score": float(grasp.score),
        "translation_xyz": _float_list(translation),
        "rotation_matrix": [
            _float_list(rotation[0]),
            _float_list(rotation[1]),
            _float_list(rotation[2]),
        ],
        "depth": depth,
        "width": float(grasp.width),
        "height": float(grasp.height),
        "gripper_tip_position_xyz": _float_list(tip_position),
    }


def _load_numeric_deps() -> tuple[Any, Any]:
    import numpy as np
    from PIL import Image

    return np, Image


def _decode_image_payload(
    payload: dict[str, Any] | None,
    *,
    Image: Any,
    np: Any,
    convert: str | None,
    reason: str,
) -> Any:
    if not isinstance(payload, dict) or not payload.get("base64"):
        raise AnyGraspInputError(reason)
    try:
        data = base64.b64decode(payload["base64"], validate=True)
        image = Image.open(io.BytesIO(data))
        source_format = (image.format or "").upper()
        source_mode = image.mode
        if convert:
            image = image.convert(convert)
        array = np.asarray(image)
        if (
            convert is None
            and source_format == "PNG"
            and source_mode.startswith("I")
            and array.dtype.kind == "i"
        ):
            array = array.astype(np.uint16)
        return array
    except Exception as exc:  # noqa: BLE001
        raise AnyGraspInputError(reason) from exc


def _float_list(values: Any) -> list[float]:
    return [float(value) for value in values]


def _with_duration(metadata: dict[str, Any], start: float) -> dict[str, Any]:
    return {**metadata, "duration_s": round(time.perf_counter() - start, 4)}


def _failure_result(
    *,
    reason: str,
    content: str,
    metadata: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    return {
        "success": False,
        "content": content,
        "details": {
            "tool": "anygrasp",
            "backend": "anygrasp_mcp",
            "model": "anygrasp_sdk",
            "mode": mode,
            "candidate_count": 0,
            "grasp_candidates": [],
            "artifacts": [],
            "reason": reason,
            "metadata": metadata,
        },
    }
