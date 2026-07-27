"""GraspGenX backend for the OpenETA GraspGenX MCP server."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SERVER_NAME = "openeta-graspgenx"
TOOL_NAME = "predict_grasps"
LIST_TOOL_NAME = "list_grippers"
BACKEND_NAME = "graspgenx_mcp"
MODEL_NAME = "graspgenx"
PLANNER = "graspmoe"
FRAME = "camera"
CAMERA_FRAME = "opencv"
GRASP_FRAME = "graspnet"
NATIVE_GRASP_FRAME = "graspgenx"
POSE_CONVENTION = "p_camera = R @ p_grasp + t"

DEFAULT_DEPTH_TRUNCATION = 1.0
MIN_OBJECT_POINTS = 100
MAX_RETURNED_CANDIDATES = 20
NUM_GRASPS = 200
MOE_NUM_YAWS = 36
MOE_Z_OFFSETS_CM = (-2.0, 0.0)
MOE_OUTLIER_THRESHOLD = 0.014
MOE_OUTLIER_K = 20
MOE_OBB_MODE = "advanced"
MOE_SKIP_OBB_RULE = "auto"
MOE_OBB_DENSITY = "dense-topandside"
MOE_OBB_POSITION_SPACING_CM = 1.0
COLLISION_THRESHOLD = 0.02
MAX_COLLISION_SCENE_POINTS = 8192
NUM_COLLISION_SAMPLES = 2000
COLLISION_BATCH_SIZE = 16

_DEPTH_SCALE_GUIDANCE = (
    "Depth in meters is raw_depth / intrinsics.scale; for uint16 millimeter "
    "depth, use scale=1000."
)

# Columns are the GraspNet basis vectors expressed in the GraspGenX basis:
# GraspNet +X = GraspGenX +Z (approach), GraspNet +Y = GraspGenX +X
# (closing), and GraspNet +Z = GraspGenX +Y.
_NATIVE_FROM_GRASPNET_ROTATION = (
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
)


class GraspGenXInputError(Exception):
    """Input or normalized output cannot satisfy the GraspGenX contract."""

    def __init__(self, reason: str, *, metadata: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.metadata = dict(metadata or {})


@dataclass(frozen=True)
class GripperDescription:
    """Validated public geometry for one inference-ready gripper asset."""

    name: str
    gripper_type: str
    fingertip_xyz: tuple[float, float, float]
    sweep_open_extents_xyz: tuple[float, float, float]
    sweep_open_offset_xyz: tuple[float, float, float]
    sweep_mid_extents_xyz: tuple[float, float, float]
    sweep_mid_offset_xyz: tuple[float, float, float]
    asset_family: str = "x_grippers"

    @property
    def fingertip_depth(self) -> float:
        return self.fingertip_xyz[2]

    @property
    def width(self) -> float:
        return self.sweep_open_extents_xyz[0]

    @property
    def height(self) -> float:
        return self.sweep_open_extents_xyz[1]

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "gripper_type": self.gripper_type,
            "fingertip_depth": self.fingertip_depth,
            "sweep_volume_open": {
                "extents_xyz": list(self.sweep_open_extents_xyz),
                "offset_xyz": list(self.sweep_open_offset_xyz),
            },
            "sweep_volume_mid": {
                "extents_xyz": list(self.sweep_mid_extents_xyz),
                "offset_xyz": list(self.sweep_mid_offset_xyz),
            },
            "asset_family": self.asset_family,
        }


def scan_gripper_descriptions(
    gripper_descriptions_root: str | Path,
) -> tuple[dict[str, GripperDescription], dict[str, str]]:
    """Scan assets without importing GraspGenX or loading model weights.

    Invalid individual assets are excluded and returned in the diagnostics map.
    A missing asset root or an empty valid set is a service-configuration error.
    """

    root = Path(gripper_descriptions_root).expanduser().resolve()
    assets_root = root / "gripper_descriptions" / "assets" / "x_grippers"
    if not assets_root.is_dir():
        raise ValueError(
            "gripper descriptions root must contain "
            "gripper_descriptions/assets/x_grippers"
        )

    valid: dict[str, GripperDescription] = {}
    invalid: dict[str, str] = {}
    for asset_dir in sorted(path for path in assets_root.iterdir() if path.is_dir()):
        if asset_dir.name == "utils":
            continue
        try:
            valid[asset_dir.name] = _parse_gripper_description(asset_dir)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            invalid[asset_dir.name] = str(exc)
    if not valid:
        raise ValueError("no valid inference-ready gripper assets were found")
    return valid, invalid


def _parse_gripper_description(asset_dir: Path) -> GripperDescription:
    config_path = asset_dir / "config.json"
    collision_mesh_path = asset_dir / "coll_mesh.obj"
    if not config_path.is_file():
        raise ValueError("missing config.json")
    if not collision_mesh_path.is_file() or collision_mesh_path.stat().st_size == 0:
        raise ValueError("missing or empty coll_mesh.obj")
    _validate_collision_mesh_obj(collision_mesh_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    gripper_type = str(config["type"])
    if gripper_type not in {"parallel_2f", "revolute_2f", "revolute_3f"}:
        raise ValueError(f"unsupported gripper type: {gripper_type}")
    fingertip = _finite_vector(config["fingertip"], "fingertip")
    if fingertip[2] <= 0:
        raise ValueError("fingertip Z depth must be positive")

    sweep = config["sweep_volume"]
    open_extents = _positive_vector(sweep["extents"], "sweep_volume.extents")
    open_offset = _finite_vector(sweep["offset"], "sweep_volume.offset")
    mid_extents = _positive_vector(sweep["extents2"], "sweep_volume.extents2")
    mid_offset = _finite_vector(sweep["offset2"], "sweep_volume.offset2")
    return GripperDescription(
        name=asset_dir.name,
        gripper_type=gripper_type,
        fingertip_xyz=fingertip,
        sweep_open_extents_xyz=open_extents,
        sweep_open_offset_xyz=open_offset,
        sweep_mid_extents_xyz=mid_extents,
        sweep_mid_offset_xyz=mid_offset,
    )


def _validate_collision_mesh_obj(path: Path) -> None:
    has_vertex = False
    has_face = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            stripped = line.lstrip()
            if line_number == 0 and stripped.startswith(
                "version https://git-lfs.github.com/spec/"
            ):
                raise ValueError("coll_mesh.obj is an unmaterialized Git LFS pointer")
            has_vertex = has_vertex or stripped.startswith("v ")
            has_face = has_face or stripped.startswith("f ")
            if has_vertex and has_face:
                return
    raise ValueError("coll_mesh.obj must contain vertices and faces")


def _finite_vector(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain three values")
    try:
        parsed = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite numbers") from exc
    if not all(math.isfinite(item) for item in parsed):
        raise ValueError(f"{label} must contain finite numbers")
    return parsed  # type: ignore[return-value]


def _positive_vector(value: Any, label: str) -> tuple[float, float, float]:
    parsed = _finite_vector(value, label)
    if not all(item > 0 for item in parsed):
        raise ValueError(f"{label} values must be positive")
    return parsed


def validate_checkpoint_layout(checkpoint_root: str | Path) -> tuple[Path, Path]:
    """Return the exact generator/discriminator checkpoints selected at startup."""

    root = Path(checkpoint_root).expanduser().resolve()
    checkpoints: list[Path] = []
    for component in ("gen", "dis"):
        component_dir = root / component
        if not component_dir.is_dir() or not (component_dir / "config.yaml").is_file():
            raise ValueError(
                f"checkpoint root must contain {component}/config.yaml"
            )
        candidates = list(component_dir.glob("*.pth"))
        if not candidates:
            raise ValueError(f"checkpoint root must contain {component}/*.pth")
        checkpoints.append(max(candidates, key=_checkpoint_sort_key))
    return checkpoints[0], checkpoints[1]


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"epoch_(\d+)", path.stem)
    return (int(match.group(1)) if match else -1, path.name)


def validate_intrinsics(intrinsics: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(intrinsics, dict):
        raise GraspGenXInputError("missing_intrinsics")
    parsed: dict[str, float] = {}
    for key in ("fx", "fy", "cx", "cy"):
        if key not in intrinsics:
            raise GraspGenXInputError("invalid_intrinsics")
        try:
            value = float(intrinsics[key])
        except (TypeError, ValueError) as exc:
            raise GraspGenXInputError("invalid_intrinsics") from exc
        if not math.isfinite(value) or (key in {"fx", "fy"} and value <= 0):
            raise GraspGenXInputError("invalid_intrinsics")
        parsed[key] = value
    if "scale" not in intrinsics:
        raise GraspGenXInputError("invalid_depth_scale")
    try:
        scale = float(intrinsics["scale"])
    except (TypeError, ValueError) as exc:
        raise GraspGenXInputError("invalid_depth_scale") from exc
    if not math.isfinite(scale) or scale <= 0:
        raise GraspGenXInputError("invalid_depth_scale")
    parsed["scale"] = scale
    return parsed


def validate_up_direction(up_direction_camera: Any) -> Any:
    np, _Image = _load_image_dependencies()
    try:
        vector = np.asarray(up_direction_camera, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GraspGenXInputError("invalid_up_direction") from exc
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise GraspGenXInputError("invalid_up_direction")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise GraspGenXInputError("invalid_up_direction")
    return vector / norm


def decode_image_payload(
    payload: dict[str, Any] | None,
    *,
    Image: Any,
    np: Any,
    convert: str | None,
    missing_reason: str,
    decode_reason: str,
) -> Any:
    """Decode Pillow-supported bytes; the advisory format field is not trusted."""

    if not isinstance(payload, dict) or not payload.get("base64"):
        raise GraspGenXInputError(missing_reason)
    try:
        data = base64.b64decode(payload["base64"], validate=True)
        image = Image.open(io.BytesIO(data))
        source_format = (image.format or "").upper()
        source_mode = image.mode
        if convert is not None:
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
    except GraspGenXInputError:
        raise
    except Exception as exc:  # noqa: BLE001 - Pillow raises heterogeneous errors.
        raise GraspGenXInputError(decode_reason) from exc


def build_targeted_point_clouds(
    *,
    depth_array: Any,
    object_mask_array: Any,
    intrinsics: dict[str, float],
    depth_truncation: float = DEFAULT_DEPTH_TRUNCATION,
    min_object_points: int = MIN_OBJECT_POINTS,
) -> tuple[Any, Any, dict[str, Any]]:
    """Backproject aligned depth/mask into OpenCV camera-frame point clouds."""

    np, _Image = _load_image_dependencies()
    depth = np.asarray(depth_array)
    mask = np.asarray(object_mask_array)
    if depth.ndim != 2 or mask.ndim != 2 or depth.shape != mask.shape:
        raise GraspGenXInputError("image_shape_mismatch")
    if depth.dtype.kind not in {"u", "i", "f"}:
        raise GraspGenXInputError("depth_decode_failed")
    object_mask = mask > 0
    if not bool(object_mask.any()):
        raise GraspGenXInputError("empty_object_mask")

    depth_float = depth.astype(np.float64, copy=False)
    finite_raw = np.isfinite(depth_float)
    finite_values = depth_float[finite_raw]
    raw_min = float(finite_values.min()) if finite_values.size else None
    raw_max = float(finite_values.max()) if finite_values.size else None
    metric_depth = depth_float / float(intrinsics["scale"])
    valid = (
        finite_raw
        & (metric_depth > 0.0)
        & (metric_depth < float(depth_truncation))
    )
    metadata = {
        "depth_dtype": str(depth.dtype),
        "depth_raw_min": raw_min,
        "depth_raw_max": raw_max,
        "depth_metric_min": (
            None if raw_min is None else raw_min / float(intrinsics["scale"])
        ),
        "depth_metric_max": (
            None if raw_max is None else raw_max / float(intrinsics["scale"])
        ),
        "depth_truncation": float(depth_truncation),
        "valid_point_count": int(valid.sum()),
    }
    is_uint16 = depth.dtype.kind == "u" and depth.dtype.itemsize == 2
    if is_uint16 and raw_max is not None and raw_max > 10 and intrinsics["scale"] <= 1:
        raise GraspGenXInputError("depth_scale_mismatch", metadata=metadata)
    if metadata["valid_point_count"] == 0:
        raise GraspGenXInputError(
            "empty_point_cloud_after_depth_filter", metadata=metadata
        )

    object_valid = valid & object_mask
    object_count = int(object_valid.sum())
    scene_valid = valid & ~object_mask
    metadata.update(
        {
            "object_point_count": object_count,
            "scene_point_count": int(scene_valid.sum()),
            "model_input_point_count": object_count,
        }
    )
    if object_count < int(min_object_points):
        raise GraspGenXInputError("insufficient_object_points", metadata=metadata)

    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    points_x = (u - intrinsics["cx"]) / intrinsics["fx"] * metric_depth
    points_y = (v - intrinsics["cy"]) / intrinsics["fy"] * metric_depth
    organized = np.stack([points_x, points_y, metric_depth], axis=-1)
    object_points = np.ascontiguousarray(organized[object_valid], dtype=np.float32)
    scene_points = np.ascontiguousarray(organized[scene_valid], dtype=np.float32)
    return object_points, scene_points, metadata


def rotation_aligning_up_to_z(up_direction_camera: Any) -> Any:
    """Return R such that p_aligned=R@p_camera and R@up_camera=[0,0,1]."""

    np, _Image = _load_image_dependencies()
    source = validate_up_direction(up_direction_camera)
    target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if cosine > 1.0 - 1e-12:
        return np.eye(3, dtype=np.float64)
    if cosine < -1.0 + 1e-12:
        # Any unit axis orthogonal to source is valid for the 180-degree case.
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(source[0])) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(source, helper)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)

    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + skew + (skew @ skew) * ((1.0 - cosine) / (sine * sine))


def validate_raw_grasp_outputs(
    grasps: Any, scores: Any, branch_tags: Any
) -> tuple[Any, Any, list[str]]:
    np, _Image = _load_image_dependencies()
    try:
        grasp_array = np.asarray(grasps, dtype=np.float64)
        score_array = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GraspGenXInputError("inconsistent_grasp_outputs") from exc
    if grasp_array.size == 0:
        raise GraspGenXInputError("no_grasp_candidates")
    if grasp_array.ndim != 3 or grasp_array.shape[1:] != (4, 4):
        raise GraspGenXInputError("inconsistent_grasp_outputs")
    count = int(grasp_array.shape[0])
    if score_array.shape != (count,) or not isinstance(branch_tags, (list, tuple)):
        raise GraspGenXInputError("inconsistent_grasp_outputs")
    tags = [str(tag) for tag in branch_tags]
    if len(tags) != count or any(tag not in {"diff", "obb"} for tag in tags):
        raise GraspGenXInputError("inconsistent_grasp_outputs")
    if not np.isfinite(grasp_array).all() or not np.isfinite(score_array).all():
        raise GraspGenXInputError("inconsistent_grasp_outputs")
    if not all(_is_rigid_transform(matrix) for matrix in grasp_array):
        raise GraspGenXInputError("inconsistent_grasp_outputs")
    return grasp_array, score_array, tags


def transform_grasps_to_camera(grasps_aligned: Any, alignment_rotation: Any) -> Any:
    np, _Image = _load_image_dependencies()
    camera_from_aligned = np.eye(4, dtype=np.float64)
    camera_from_aligned[:3, :3] = np.asarray(alignment_rotation).T
    camera_grasps = np.einsum(
        "ij,kjl->kil", camera_from_aligned, np.asarray(grasps_aligned, dtype=np.float64)
    )
    if not all(_is_rigid_transform(matrix) for matrix in camera_grasps):
        raise GraspGenXInputError("inconsistent_grasp_outputs")
    return camera_grasps


def normalise_grasp_candidates(
    *,
    camera_native_grasps: Any,
    scores: Any,
    branch_tags: list[str],
    selected_indices: list[int],
    gripper: GripperDescription,
) -> list[dict[str, Any]]:
    """Build dual-frame candidates in already-ranked selected order."""

    np, _Image = _load_image_dependencies()
    native_grasps = np.asarray(camera_native_grasps, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    native_from_graspnet = np.eye(4, dtype=np.float64)
    native_from_graspnet[:3, :3] = np.asarray(
        _NATIVE_FROM_GRASPNET_ROTATION, dtype=np.float64
    )
    tip_native = np.asarray(gripper.fingertip_xyz, dtype=np.float64)

    candidates: list[dict[str, Any]] = []
    for rank, backend_index in enumerate(selected_indices):
        native_pose = native_grasps[backend_index]
        graspnet_pose = native_pose @ native_from_graspnet
        tip_camera = native_pose[:3, :3] @ tip_native + native_pose[:3, 3]
        if not _is_rigid_transform(graspnet_pose) or not np.isfinite(tip_camera).all():
            raise GraspGenXInputError("inconsistent_grasp_outputs")
        candidate = {
            "id": f"graspgenx_{rank:03d}",
            "source_model": MODEL_NAME,
            "gripper_name": gripper.name,
            "candidate_source": (
                "diffusion" if branch_tags[backend_index] == "diff" else "obb"
            ),
            "frame": FRAME,
            "camera_frame": CAMERA_FRAME,
            "grasp_frame": GRASP_FRAME,
            "convention": POSE_CONVENTION,
            "score": float(score_array[backend_index]),
            "translation_xyz": _float_list(graspnet_pose[:3, 3]),
            "rotation_matrix": _float_matrix(graspnet_pose[:3, :3]),
            "transform_matrix": _float_matrix(graspnet_pose),
            "gripper_tip_position_xyz": _float_list(tip_camera),
            "depth": float(gripper.fingertip_depth),
            "width": float(gripper.width),
            "height": float(gripper.height),
            "rank": rank,
            "backend_index": int(backend_index),
            "model_native_grasp_pose": {
                "frame": FRAME,
                "camera_frame": CAMERA_FRAME,
                "grasp_frame": NATIVE_GRASP_FRAME,
                "convention": POSE_CONVENTION,
                "transform_matrix": _float_matrix(native_pose),
            },
        }
        candidates.append(candidate)
    return candidates


class GraspGenXBackend:
    """Lazy, in-process wrapper around the official GraspGenX Python API."""

    def __init__(
        self,
        *,
        graspgenx_root: str | Path,
        checkpoint_root: str | Path,
        gripper_descriptions_root: str | Path,
        device: str = "cuda:0",
        depth_truncation: float = DEFAULT_DEPTH_TRUNCATION,
    ) -> None:
        self.graspgenx_root = Path(graspgenx_root).expanduser().resolve()
        self.checkpoint_root = Path(checkpoint_root).expanduser().resolve()
        self.gripper_descriptions_root = (
            Path(gripper_descriptions_root).expanduser().resolve()
        )
        self.device = validate_cuda_device_name(device)
        self.depth_truncation = float(depth_truncation)
        self.generator_checkpoint, self.discriminator_checkpoint = (
            validate_checkpoint_layout(self.checkpoint_root)
        )
        self.grippers, self.invalid_grippers = scan_gripper_descriptions(
            self.gripper_descriptions_root
        )
        self.assets_root = (
            self.gripper_descriptions_root / "gripper_descriptions" / "assets"
        )
        self._loaded: dict[str, Any] | None = None
        self._samplers: dict[str, dict[str, Any]] = {}

    @property
    def model_loaded(self) -> bool:
        return self._loaded is not None

    def list_grippers(self) -> dict[str, Any]:
        descriptions = [
            self.grippers[name].to_public_dict() for name in sorted(self.grippers)
        ]
        return {
            "success": True,
            "content": "GraspGenX gripper listing completed.",
            "details": {
                "tool": LIST_TOOL_NAME,
                "gripper_count": len(descriptions),
                "grippers": descriptions,
                "model_loaded": self.model_loaded,
            },
        }

    def predict_grasps(
        self,
        *,
        depth: dict[str, Any] | None,
        object_mask: dict[str, Any] | None,
        intrinsics: dict[str, Any] | None,
        gripper_name: str,
        up_direction_camera: Any,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        metadata = self._metadata_base(gripper_name=gripper_name)

        try:
            gripper = self._validate_gripper_name(gripper_name)
            parsed_intrinsics = validate_intrinsics(intrinsics)
            normalized_up = validate_up_direction(up_direction_camera)
            metadata["intrinsics"] = parsed_intrinsics
            metadata["up_direction_camera_normalized"] = _float_list(normalized_up)
            np, Image = _load_image_dependencies()
            depth_array = decode_image_payload(
                depth,
                Image=Image,
                np=np,
                convert=None,
                missing_reason="missing_depth",
                decode_reason="depth_decode_failed",
            )
            mask_array = decode_image_payload(
                object_mask,
                Image=Image,
                np=np,
                convert="L",
                missing_reason="missing_object_mask",
                decode_reason="object_mask_decode_failed",
            )
            object_points, scene_points, point_metadata = build_targeted_point_clouds(
                depth_array=depth_array,
                object_mask_array=mask_array,
                intrinsics=parsed_intrinsics,
                depth_truncation=self.depth_truncation,
            )
            metadata.update(point_metadata)
            alignment = rotation_aligning_up_to_z(normalized_up)
            object_points_aligned = np.ascontiguousarray(
                object_points @ alignment.T, dtype=np.float32
            )
        except GraspGenXInputError as exc:
            metadata.update(exc.metadata)
            return failure_result(
                reason=exc.reason,
                metadata=_with_duration(metadata, start),
            )

        try:
            loaded = self._get_loaded_backend()
            sampler_entry = self._get_sampler_entry(loaded, gripper.name)
            metadata.update(self._model_metadata(loaded))
        except GraspGenXInputError as exc:
            metadata.update(exc.metadata)
            return failure_result(
                reason=exc.reason,
                metadata=_with_duration(metadata, start),
            )
        except Exception as exc:  # noqa: BLE001 - third-party load boundary.
            return failure_result(
                reason="model_load_failed",
                metadata=_with_duration(
                    {**metadata, "error_type": type(exc).__name__}, start
                ),
            )

        try:
            raw_grasps, raw_scores, raw_tags = self._run_planner(
                loaded=loaded,
                sampler_entry=sampler_entry,
                object_points_aligned=object_points_aligned,
            )
        except Exception as exc:  # noqa: BLE001 - third-party inference boundary.
            return failure_result(
                reason="model_inference_failed",
                metadata=_with_duration(
                    {**metadata, "error_type": type(exc).__name__}, start
                ),
            )

        try:
            grasps_aligned, scores, tags = validate_raw_grasp_outputs(
                raw_grasps, raw_scores, raw_tags
            )
            camera_native_grasps = transform_grasps_to_camera(
                grasps_aligned, alignment
            )
            metadata.update(
                {
                    "generated_candidate_count": int(len(scores)),
                    "diffusion_candidate_count": tags.count("diff"),
                    "obb_candidate_count": tags.count("obb"),
                }
            )
            selected_indices, collision_metadata = self._select_collision_free(
                loaded=loaded,
                sampler_entry=sampler_entry,
                scene_points=scene_points,
                camera_native_grasps=camera_native_grasps,
                scores=scores,
            )
            metadata.update(collision_metadata)
            candidates = normalise_grasp_candidates(
                camera_native_grasps=camera_native_grasps,
                scores=scores,
                branch_tags=tags,
                selected_indices=selected_indices,
                gripper=gripper,
            )
            if len(candidates) != len(selected_indices):
                raise GraspGenXInputError("inconsistent_grasp_outputs")
            metadata["returned_candidate_count"] = len(candidates)
        except GraspGenXInputError as exc:
            metadata.update(exc.metadata)
            return failure_result(
                reason=exc.reason,
                metadata=_with_duration(metadata, start),
            )
        except Exception as exc:  # noqa: BLE001 - collision backend boundary.
            return failure_result(
                reason="model_inference_failed",
                metadata=_with_duration(
                    {**metadata, "error_type": type(exc).__name__}, start
                ),
            )

        return {
            "success": True,
            "content": "GraspGenX grasp prediction completed.",
            "details": {
                "tool": TOOL_NAME,
                "backend": BACKEND_NAME,
                "model": MODEL_NAME,
                "planner": PLANNER,
                "deterministic": False,
                "frame": FRAME,
                "camera_frame": CAMERA_FRAME,
                "grasp_frame": GRASP_FRAME,
                "gripper_name": gripper.name,
                "candidate_count": len(candidates),
                "grasp_candidates": candidates,
                "ranking": "score_descending",
                "artifacts": [],
                "metadata": _with_duration(metadata, start),
            },
        }

    def _validate_gripper_name(self, gripper_name: Any) -> GripperDescription:
        if not isinstance(gripper_name, str) or gripper_name not in self.grippers:
            raise GraspGenXInputError("unsupported_gripper")
        return self.grippers[gripper_name]

    def _get_loaded_backend(self) -> dict[str, Any]:
        if self._loaded is not None:
            return self._loaded
        if not (self.graspgenx_root / "graspgenx" / "__init__.py").is_file():
            raise RuntimeError("GraspGenX source root is incomplete")

        # These existing paths make the official package setup hook a no-op and
        # prevent its fallback auto-clone behavior during import.
        os.environ["GRASPGENX_CHECKPOINT_DIR"] = str(self.checkpoint_root.parent)
        os.environ["GRASPGENX_GRIPPER_CFG_DIR"] = str(
            self.gripper_descriptions_root
        )
        if str(self.graspgenx_root) not in sys.path:
            sys.path.insert(0, str(self.graspgenx_root))

        # GraspGenX configures third-party logging during its first imports.
        # Import while stdout points at stderr so those handlers can never
        # contaminate the stdio MCP JSON-RPC stream.
        with contextlib.redirect_stdout(sys.stderr):
            import torch

            from graspgenx.grasp_server import GraspGenXSampler, load_grasp_gen_model
            from graspgenx.samplers import run_planner_on_object
            from graspgenx.utils.checkpoint_io import load_model_cfg
            from graspgenx.utils.collision_filter import filter_colliding_grasps

        device_index = _resolve_cuda_device_index(torch, self.device)
        with torch.cuda.device(device_index), contextlib.redirect_stdout(sys.stderr):
            config = load_model_cfg(
                str(self.checkpoint_root / "gen"),
                str(self.checkpoint_root / "dis"),
                self.generator_checkpoint.name,
                self.discriminator_checkpoint.name,
            )
            model = load_grasp_gen_model(config)

        self._loaded = {
            "torch": torch,
            "device_index": device_index,
            "config": config,
            "model": model,
            "sampler_class": GraspGenXSampler,
            "run_planner": run_planner_on_object,
            "filter_collisions": filter_colliding_grasps,
            "backend_commit": _git_commit(self.graspgenx_root),
            "checkpoint_version": self.checkpoint_root.name,
            "generator_checkpoint_sha256": _sha256_file(
                self.generator_checkpoint
            ),
            "discriminator_checkpoint_sha256": _sha256_file(
                self.discriminator_checkpoint
            ),
        }
        return self._loaded

    def _get_sampler_entry(
        self, loaded: dict[str, Any], gripper_name: str
    ) -> dict[str, Any]:
        cached = self._samplers.get(gripper_name)
        if cached is not None:
            return cached
        np, _Image = _load_image_dependencies()
        torch = loaded["torch"]
        with torch.cuda.device(loaded["device_index"]), contextlib.redirect_stdout(
            sys.stderr
        ):
            sampler = loaded["sampler_class"](
                loaded["config"],
                gripper_name,
                assets_dir=str(self.assets_root),
                model=loaded["model"],
            )
        gripper_info = sampler.get_gripper_info()
        import trimesh

        sampled, _faces = trimesh.sample.sample_surface(
            gripper_info.collision_mesh, NUM_COLLISION_SAMPLES
        )
        surface_points = np.ascontiguousarray(sampled, dtype=np.float32)
        if surface_points.shape != (NUM_COLLISION_SAMPLES, 3):
            raise RuntimeError("invalid gripper collision surface sample")
        entry = {
            "sampler": sampler,
            "gripper_info": gripper_info,
            "collision_surface_points": surface_points,
        }
        self._samplers[gripper_name] = entry
        return entry

    def _run_planner(
        self,
        *,
        loaded: dict[str, Any],
        sampler_entry: dict[str, Any],
        object_points_aligned: Any,
    ) -> tuple[Any, Any, Any]:
        torch = loaded["torch"]
        with (
            torch.cuda.device(loaded["device_index"]),
            torch.inference_mode(),
            contextlib.redirect_stdout(sys.stderr),
        ):
            grasps, scores, tags, _obb = loaded["run_planner"](
                object_points_aligned,
                sampler_entry["sampler"],
                planner=PLANNER,
                grasp_threshold=-1.0,
                num_grasps=NUM_GRASPS,
                topk_num_grasps=-1,
                moe_num_yaws=MOE_NUM_YAWS,
                moe_z_offsets_cm=MOE_Z_OFFSETS_CM,
                moe_outlier_threshold=MOE_OUTLIER_THRESHOLD,
                moe_outlier_k=MOE_OUTLIER_K,
                moe_obb_mode=MOE_OBB_MODE,
                moe_skip_obb_rule=MOE_SKIP_OBB_RULE,
                moe_obb_density=MOE_OBB_DENSITY,
                moe_obb_position_spacing_cm=MOE_OBB_POSITION_SPACING_CM,
            )
        return grasps, scores, tags

    def _select_collision_free(
        self,
        *,
        loaded: dict[str, Any],
        sampler_entry: dict[str, Any],
        scene_points: Any,
        camera_native_grasps: Any,
        scores: Any,
    ) -> tuple[list[int], dict[str, Any]]:
        np, _Image = _load_image_dependencies()
        score_array = np.asarray(scores, dtype=np.float64)
        ranked = sorted(range(len(score_array)), key=lambda idx: (-score_array[idx], idx))
        if len(scene_points) == 0:
            return ranked[:MAX_RETURNED_CANDIDATES], {
                "collision_filter_applied": False,
                "collision_filter_reason": "no_scene_points",
                "collision_scene_point_count": 0,
                "collision_checked_count": 0,
                "collision_rejected_count": 0,
            }

        collision_scene = np.asarray(scene_points, dtype=np.float32)
        if len(collision_scene) > MAX_COLLISION_SCENE_POINTS:
            indices = np.random.default_rng().choice(
                len(collision_scene), MAX_COLLISION_SCENE_POINTS, replace=False
            )
            collision_scene = np.ascontiguousarray(collision_scene[indices])

        selected: list[int] = []
        checked = 0
        rejected = 0
        filter_fn = loaded["filter_collisions"]
        for offset in range(0, len(ranked), COLLISION_BATCH_SIZE):
            batch_indices = ranked[offset : offset + COLLISION_BATCH_SIZE]
            batch_poses = camera_native_grasps[batch_indices]
            free_mask = np.asarray(
                filter_fn(
                    scene_pc=collision_scene,
                    grasp_poses=batch_poses,
                    collision_threshold=COLLISION_THRESHOLD,
                    gripper_surface_points=sampler_entry[
                        "collision_surface_points"
                    ],
                    batch_size=COLLISION_BATCH_SIZE,
                    device=self.device,
                )
            )
            if free_mask.shape != (len(batch_indices),) or free_mask.dtype.kind != "b":
                raise GraspGenXInputError("inconsistent_grasp_outputs")
            checked += len(batch_indices)
            rejected += int((~free_mask).sum())
            selected.extend(
                idx for idx, is_free in zip(batch_indices, free_mask) if bool(is_free)
            )
            if len(selected) >= MAX_RETURNED_CANDIDATES:
                selected = selected[:MAX_RETURNED_CANDIDATES]
                break
        if not selected:
            raise GraspGenXInputError(
                "all_grasps_colliding",
                metadata={
                    "collision_filter_applied": True,
                    "collision_scene_point_count": int(len(collision_scene)),
                    "collision_checked_count": checked,
                    "collision_rejected_count": rejected,
                    "returned_candidate_count": 0,
                },
            )
        return selected, {
            "collision_filter_applied": True,
            "collision_scene_point_count": int(len(collision_scene)),
            "collision_checked_count": checked,
            "collision_rejected_count": rejected,
        }

    def _metadata_base(self, *, gripper_name: Any) -> dict[str, Any]:
        return {
            "frame": FRAME,
            "camera_frame": CAMERA_FRAME,
            "grasp_frame": GRASP_FRAME,
            "native_grasp_frame": NATIVE_GRASP_FRAME,
            "planner": PLANNER,
            "deterministic": False,
            "gripper_name": gripper_name,
            "depth_truncation": self.depth_truncation,
            "min_object_points": MIN_OBJECT_POINTS,
            "max_returned_candidates": MAX_RETURNED_CANDIDATES,
            "model_loaded": self.model_loaded,
            "intrinsics": {},
            "inference_options": {
                "num_grasps": NUM_GRASPS,
                "moe_num_yaws": MOE_NUM_YAWS,
                "moe_z_offsets_cm": list(MOE_Z_OFFSETS_CM),
                "moe_outlier_threshold": MOE_OUTLIER_THRESHOLD,
                "moe_outlier_k": MOE_OUTLIER_K,
                "moe_obb_mode": MOE_OBB_MODE,
                "moe_skip_obb_rule": MOE_SKIP_OBB_RULE,
                "moe_obb_density": MOE_OBB_DENSITY,
                "moe_obb_position_spacing_cm": MOE_OBB_POSITION_SPACING_CM,
                "collision_threshold": COLLISION_THRESHOLD,
                "max_collision_scene_points": MAX_COLLISION_SCENE_POINTS,
                "num_collision_samples": NUM_COLLISION_SAMPLES,
            },
            "backend_commit": None,
            "checkpoint_version": self.checkpoint_root.name,
            "generator_checkpoint_sha256": None,
            "discriminator_checkpoint_sha256": None,
        }

    @staticmethod
    def _model_metadata(loaded: dict[str, Any]) -> dict[str, Any]:
        return {
            "backend": MODEL_NAME,
            "backend_commit": loaded["backend_commit"],
            "checkpoint_version": loaded["checkpoint_version"],
            "generator_checkpoint_sha256": loaded[
                "generator_checkpoint_sha256"
            ],
            "discriminator_checkpoint_sha256": loaded[
                "discriminator_checkpoint_sha256"
            ],
            "planner": PLANNER,
            "deterministic": False,
            "model_loaded": True,
        }


def validate_cuda_device_name(device: Any) -> str:
    if not isinstance(device, str) or not re.fullmatch(r"cuda(?::\d+)?", device):
        raise ValueError("device must be cuda or cuda:N")
    return "cuda:0" if device == "cuda" else device


def _resolve_cuda_device_index(torch: Any, device: str) -> int:
    if not torch.cuda.is_available():
        raise GraspGenXInputError("device_unavailable")
    index = int(device.split(":", maxsplit=1)[1])
    if index < 0 or index >= int(torch.cuda.device_count()):
        raise GraspGenXInputError("device_unavailable")
    return index


def failure_result(
    *, reason: str, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    content = f"GraspGenX grasp prediction failed: {reason}."
    if reason in {
        "invalid_depth_scale",
        "depth_scale_mismatch",
        "empty_point_cloud_after_depth_filter",
    }:
        content = f"{content} {_DEPTH_SCALE_GUIDANCE}"
    return {
        "success": False,
        "content": content,
        "details": {
            "tool": TOOL_NAME,
            "reason": reason,
            "candidate_count": 0,
            "grasp_candidates": [],
            "metadata": dict(metadata or {}),
        },
    }


def _is_rigid_transform(matrix: Any) -> bool:
    np, _Image = _load_image_dependencies()
    matrix = np.asarray(matrix)
    return bool(
        matrix.shape == (4, 4)
        and np.isfinite(matrix).all()
        and np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        and _is_rotation_matrix(matrix[:3, :3])
    )


def _is_rotation_matrix(rotation: Any) -> bool:
    np, _Image = _load_image_dependencies()
    rotation = np.asarray(rotation)
    return bool(
        rotation.shape == (3, 3)
        and np.isfinite(rotation).all()
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    )


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit if len(commit) == 40 else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_image_dependencies() -> tuple[Any, Any]:
    import numpy as np
    from PIL import Image

    return np, Image


def _float_list(values: Any) -> list[float]:
    return [float(value) for value in values]


def _float_matrix(values: Any) -> list[list[float]]:
    return [_float_list(row) for row in values]


def _with_duration(metadata: dict[str, Any], start: float) -> dict[str, Any]:
    return {
        **metadata,
        "duration_ms": round((time.perf_counter() - start) * 1000.0, 3),
    }
