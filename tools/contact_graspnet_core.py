"""Contact-GraspNet backend for the OpenETA Contact-GraspNet MCP server."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import inspect
import io
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


TOOL_NAME = "contact_graspnet"
BACKEND_NAME = "contact_graspnet_mcp"
MODEL_NAME = "contact_graspnet_pytorch_unofficial"
MODE = "targeted"
FRAME = "camera"
CAMERA_FRAME = "opencv"
GRASP_FRAME = "graspnet"
SOURCE_MODEL = "contact_graspnet"
GRIPPER_MODEL = "panda"
GRIPPER_DEPTH = 0.1034
MODEL_INPUT_POINT_COUNT = 20_000
DEFAULT_DEPTH_MIN = 0.2
DEFAULT_DEPTH_MAX = 1.8
DEFAULT_MAX_CANDIDATES = 20
TARGET_SEGMENT_ID = 1
LOCAL_REGION_MIN_SIZE = 0.2
LOCAL_REGION_MAX_SIZE = 0.6

_CONTACT_TO_GRASPNET_AXES = (
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
)
_DEPTH_SCALE_GUIDANCE = (
    "Depth in meters is raw_depth / intrinsics.scale; for uint16 millimeter "
    "depth, use scale=1000."
)


class ContactGraspNetInputError(Exception):
    """Input or normalized output cannot satisfy the public MCP contract."""

    def __init__(self, reason: str, *, metadata: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.metadata = dict(metadata or {})


class ContactGraspNetBackend:
    """Lazy wrapper around an external Contact-GraspNet PyTorch checkout."""

    def __init__(
        self,
        *,
        contact_graspnet_root: str | Path,
        checkpoint_dir: str | Path,
        seed: int = 0,
        depth_min: float = DEFAULT_DEPTH_MIN,
        depth_max: float = DEFAULT_DEPTH_MAX,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
    ) -> None:
        self.contact_graspnet_root = Path(contact_graspnet_root)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.seed = int(seed)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.max_candidates = int(max_candidates)
        if not math.isfinite(self.depth_min) or self.depth_min < 0:
            raise ValueError("depth_min must be finite and non-negative")
        if not math.isfinite(self.depth_max) or self.depth_max <= self.depth_min:
            raise ValueError("depth_max must be finite and greater than depth_min")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        self._loaded: dict[str, Any] | None = None

    @property
    def config_path(self) -> Path:
        return self.checkpoint_dir / "config.yaml"

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "checkpoints" / "model.pt"

    def predict_grasps(
        self,
        *,
        depth: dict[str, Any] | None,
        object_mask: dict[str, Any] | None,
        intrinsics: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Predict normalized targeted grasps from aligned depth and mask PNGs."""

        start = time.perf_counter()
        metadata = self._metadata_base()
        try:
            parsed_intrinsics = validate_intrinsics(intrinsics)
            metadata["intrinsics"] = parsed_intrinsics
            np, Image = _load_image_dependencies()
            depth_array = decode_png_payload(
                depth,
                Image=Image,
                np=np,
                convert=None,
                missing_reason="missing_depth",
                decode_reason="depth_decode_failed",
            )
            mask_array = decode_png_payload(
                object_mask,
                Image=Image,
                np=np,
                convert="L",
                missing_reason="missing_object_mask",
                decode_reason="object_mask_decode_failed",
            )
            scene_points, object_points, point_metadata = build_targeted_point_clouds(
                depth_array=depth_array,
                object_mask_array=mask_array,
                intrinsics=parsed_intrinsics,
                depth_min=self.depth_min,
                depth_max=self.depth_max,
            )
            metadata.update(point_metadata)
        except ContactGraspNetInputError as exc:
            metadata.update(exc.metadata)
            return failure_result(
                reason=exc.reason,
                content=_failure_content(exc.reason),
                metadata=_with_duration(metadata, start),
            )

        try:
            loaded = self._get_loaded_backend()
            metadata.update(
                {
                    "backend_commit": loaded["backend_commit"],
                    "checkpoint_sha256": loaded["checkpoint_sha256"],
                    "max_gripper_width": loaded["max_gripper_width"],
                }
            )
        except ContactGraspNetInputError as exc:
            metadata.update(exc.metadata)
            return failure_result(
                reason=exc.reason,
                content=_failure_content(exc.reason),
                metadata=_with_duration(metadata, start),
            )
        except Exception as exc:  # noqa: BLE001 - returned as a structured boundary failure.
            return failure_result(
                reason="model_load_failed",
                content=_failure_content("model_load_failed"),
                metadata=_with_duration(
                    {**metadata, "error_type": type(exc).__name__},
                    start,
                ),
            )

        try:
            raw_grasps, raw_scores, raw_contacts, raw_openings = self._predict_raw(
                loaded=loaded,
                scene_points=scene_points,
                object_points=object_points,
            )
        except Exception as exc:  # noqa: BLE001 - third-party inference is untrusted.
            return failure_result(
                reason="model_inference_failed",
                content=_failure_content("model_inference_failed"),
                metadata=_with_duration(
                    {**metadata, "error_type": type(exc).__name__},
                    start,
                ),
            )

        try:
            candidates, raw_candidate_count = normalise_grasp_candidates(
                grasps=raw_grasps,
                scores=raw_scores,
                contact_points=raw_contacts,
                gripper_openings=raw_openings,
                max_gripper_width=loaded["max_gripper_width"],
                max_candidates=self.max_candidates,
            )
            metadata["model_candidate_count"] = raw_candidate_count
            metadata["returned_candidate_count"] = len(candidates)
        except ContactGraspNetInputError as exc:
            return failure_result(
                reason=exc.reason,
                content=_failure_content(exc.reason),
                metadata=_with_duration(metadata, start),
            )

        return {
            "success": True,
            "content": "Contact-GraspNet grasp prediction completed.",
            "details": {
                "tool": TOOL_NAME,
                "backend": BACKEND_NAME,
                "model": MODEL_NAME,
                "mode": MODE,
                "frame": FRAME,
                "camera_frame": CAMERA_FRAME,
                "grasp_frame": GRASP_FRAME,
                "candidate_count": len(candidates),
                "grasp_candidates": candidates,
                "artifacts": [],
                "metadata": _with_duration(metadata, start),
            },
        }

    def _get_loaded_backend(self) -> dict[str, Any]:
        if self._loaded is not None:
            return self._loaded
        if not self.contact_graspnet_root.is_dir():
            raise RuntimeError("Contact-GraspNet root does not exist")
        if not self.config_path.is_file() or not self.checkpoint_path.is_file():
            raise RuntimeError("Contact-GraspNet checkpoint layout is incomplete")
        if str(self.contact_graspnet_root) not in sys.path:
            sys.path.insert(0, str(self.contact_graspnet_root))

        import numpy as np
        import torch

        if not torch.cuda.is_available():
            raise ContactGraspNetInputError("device_unavailable")

        from contact_graspnet_pytorch import config_utils
        from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator

        config = config_utils.load_config(str(self.checkpoint_dir), batch_size=1)
        model_input_count = int(config["DATA"]["raw_num_points"])
        dataset_point_count = int(config["DATA"].get("ndataset_points", model_input_count))
        if (
            model_input_count != MODEL_INPUT_POINT_COUNT
            or dataset_point_count != MODEL_INPUT_POINT_COUNT
        ):
            raise RuntimeError(
                f"checkpoint input point counts must be {MODEL_INPUT_POINT_COUNT}"
            )
        if float(config["TEST"].get("center_to_tip", 0.0)) != 0.0:
            raise RuntimeError("checkpoint center_to_tip must be zero")
        max_gripper_width = float(config["DATA"]["gripper_width"])
        if not math.isfinite(max_gripper_width) or max_gripper_width <= 0:
            raise RuntimeError("checkpoint gripper width is invalid")

        with contextlib.redirect_stdout(sys.stderr):
            estimator = GraspEstimator(config)
            depth_default = inspect.signature(
                estimator.model.build_6d_grasp
            ).parameters["gripper_depth"].default
            if not math.isclose(float(depth_default), GRIPPER_DEPTH, abs_tol=1e-9):
                raise RuntimeError("backend Panda gripper depth is incompatible")
            model_state = load_checkpoint_model_state(
                torch=torch,
                np=np,
                checkpoint_path=self.checkpoint_path,
                device=estimator.device,
            )
            estimator.model.load_state_dict(model_state, strict=True)
            estimator.model.eval()

        self._loaded = {
            "estimator": estimator,
            "torch": torch,
            "max_gripper_width": max_gripper_width,
            "backend_commit": _git_commit(self.contact_graspnet_root),
            "checkpoint_sha256": _sha256_file(self.checkpoint_path),
        }
        return self._loaded

    def _predict_raw(
        self,
        *,
        loaded: dict[str, Any],
        scene_points: Any,
        object_points: Any,
    ) -> tuple[Any, Any, Any, Any]:
        np, _Image = _load_image_dependencies()
        torch = loaded["torch"]
        estimator = loaded["estimator"]
        seed_inference(seed=self.seed, np=np, torch=torch)

        with torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
            grasps, scores, contacts, openings = estimator.predict_scene_grasps(
                scene_points,
                pc_segments={TARGET_SEGMENT_ID: object_points.copy()},
                local_regions=True,
                filter_grasps=True,
                forward_passes=1,
                use_cam_boxes=True,
            )
        return (
            grasps.get(TARGET_SEGMENT_ID, []),
            scores.get(TARGET_SEGMENT_ID, []),
            contacts.get(TARGET_SEGMENT_ID, []),
            openings.get(TARGET_SEGMENT_ID, []),
        )

    def _metadata_base(self) -> dict[str, Any]:
        return {
            "frame": FRAME,
            "camera_frame": CAMERA_FRAME,
            "grasp_frame": GRASP_FRAME,
            "mode": MODE,
            "gripper_model": GRIPPER_MODEL,
            "gripper_depth": GRIPPER_DEPTH,
            "depth_min": self.depth_min,
            "depth_max": self.depth_max,
            "seed": self.seed,
            "max_candidates": self.max_candidates,
            "model_input_point_count": MODEL_INPUT_POINT_COUNT,
            "inference_options": {
                "local_regions": True,
                "use_cam_boxes": True,
                "filter_grasps": True,
                "skip_border_objects": False,
                "forward_passes": 1,
            },
            "intrinsics": {},
            "backend_commit": None,
            "checkpoint_sha256": None,
        }


def validate_intrinsics(intrinsics: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(intrinsics, dict):
        raise ContactGraspNetInputError("missing_intrinsics")
    parsed: dict[str, float] = {}
    for key in ("fx", "fy", "cx", "cy"):
        if key not in intrinsics:
            raise ContactGraspNetInputError("invalid_intrinsics")
        try:
            value = float(intrinsics[key])
        except (TypeError, ValueError) as exc:
            raise ContactGraspNetInputError("invalid_intrinsics") from exc
        if not math.isfinite(value) or (key in {"fx", "fy"} and value <= 0):
            raise ContactGraspNetInputError("invalid_intrinsics")
        parsed[key] = value
    if "scale" not in intrinsics:
        raise ContactGraspNetInputError("invalid_depth_scale")
    try:
        scale = float(intrinsics["scale"])
    except (TypeError, ValueError) as exc:
        raise ContactGraspNetInputError("invalid_depth_scale") from exc
    if not math.isfinite(scale) or scale <= 0:
        raise ContactGraspNetInputError("invalid_depth_scale")
    parsed["scale"] = scale
    return parsed


def decode_png_payload(
    payload: dict[str, Any] | None,
    *,
    Image: Any,
    np: Any,
    convert: str | None,
    missing_reason: str,
    decode_reason: str,
) -> Any:
    if not isinstance(payload, dict) or not payload.get("base64"):
        raise ContactGraspNetInputError(missing_reason)
    if str(payload.get("format", "")).lower() != "png":
        raise ContactGraspNetInputError("unsupported_image_format")
    try:
        data = base64.b64decode(payload["base64"], validate=True)
        image = Image.open(io.BytesIO(data))
        if (image.format or "").upper() != "PNG":
            raise ContactGraspNetInputError("unsupported_image_format")
        source_mode = image.mode
        if convert is not None:
            image = image.convert(convert)
        array = np.asarray(image)
        if convert is None and source_mode.startswith("I") and array.dtype.kind == "i":
            array = array.astype(np.uint16)
        return array
    except ContactGraspNetInputError:
        raise
    except Exception as exc:  # noqa: BLE001 - image parsers raise heterogeneous errors.
        raise ContactGraspNetInputError(decode_reason) from exc


def build_targeted_point_clouds(
    *,
    depth_array: Any,
    object_mask_array: Any,
    intrinsics: dict[str, float],
    depth_min: float,
    depth_max: float,
) -> tuple[Any, Any, dict[str, Any]]:
    np, _Image = _load_image_dependencies()
    depth = np.asarray(depth_array)
    mask = np.asarray(object_mask_array)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ContactGraspNetInputError("unsupported_depth_format")
    if mask.ndim != 2 or mask.shape != depth.shape:
        raise ContactGraspNetInputError("image_shape_mismatch")
    object_mask = mask > 0
    if not object_mask.any():
        raise ContactGraspNetInputError("empty_object_mask")

    raw_min = depth.min().item()
    raw_max = depth.max().item()
    points_z = depth.astype(np.float32) / float(intrinsics["scale"])
    valid = (points_z > depth_min) & (points_z < depth_max)
    metadata = {
        "depth_dtype": str(depth.dtype),
        "depth_raw_min": raw_min,
        "depth_raw_max": raw_max,
        "depth_metric_min": float(raw_min) / float(intrinsics["scale"]),
        "depth_metric_max": float(raw_max) / float(intrinsics["scale"]),
        "scene_valid_point_count": int(valid.sum()),
    }
    if raw_max > 10 and intrinsics["scale"] <= 1:
        raise ContactGraspNetInputError("depth_scale_mismatch", metadata=metadata)
    if not valid.any():
        raise ContactGraspNetInputError(
            "empty_point_cloud_after_depth_filter",
            metadata=metadata,
        )

    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    points_x = (u - intrinsics["cx"]) / intrinsics["fx"] * points_z
    points_y = (v - intrinsics["cy"]) / intrinsics["fy"] * points_z
    organized = np.stack([points_x, points_y, points_z], axis=-1)
    scene_points = organized[valid].astype(np.float32)
    object_points = organized[valid & object_mask].astype(np.float32)
    metadata["object_valid_point_count"] = int(object_points.shape[0])
    if object_points.shape[0] == 0:
        raise ContactGraspNetInputError("empty_object_point_cloud", metadata=metadata)

    filtered_object = _reject_median_outliers(object_points, m=0.4)
    metadata["object_filtered_point_count"] = int(filtered_object.shape[0])
    if filtered_object.shape[0] == 0:
        raise ContactGraspNetInputError(
            "empty_local_region_point_cloud",
            metadata=metadata,
        )
    max_bounds = np.max(filtered_object, axis=0)
    min_bounds = np.min(filtered_object, axis=0)
    extent = max_bounds - min_bounds
    center = min_bounds + extent / 2
    cube_size = float(
        np.minimum(
            np.maximum(np.max(extent) * 2, LOCAL_REGION_MIN_SIZE),
            LOCAL_REGION_MAX_SIZE,
        )
    )
    in_region = np.all(scene_points > (center - cube_size / 2), axis=1) & np.all(
        scene_points < (center + cube_size / 2),
        axis=1,
    )
    local_region_count = int(in_region.sum())
    metadata["local_region_point_count"] = local_region_count
    metadata["local_region_cube_size"] = cube_size
    metadata["model_input_point_count"] = MODEL_INPUT_POINT_COUNT
    if local_region_count == 0:
        raise ContactGraspNetInputError(
            "empty_local_region_point_cloud",
            metadata=metadata,
        )
    return scene_points, object_points, metadata


def normalise_grasp_candidates(
    *,
    grasps: Any,
    scores: Any,
    contact_points: Any,
    gripper_openings: Any,
    max_gripper_width: float,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> tuple[list[dict[str, Any]], int]:
    np, _Image = _load_image_dependencies()
    try:
        grasp_array = np.asarray(grasps, dtype=np.float64)
        score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
        contact_array = np.asarray(contact_points, dtype=np.float64)
        opening_array = np.asarray(gripper_openings, dtype=np.float64).reshape(-1)
    except Exception as exc:  # noqa: BLE001 - third-party arrays are untrusted.
        raise ContactGraspNetInputError("inconsistent_grasp_outputs") from exc

    if grasp_array.size == 0:
        raise ContactGraspNetInputError("no_grasp_candidates")
    if grasp_array.ndim != 3 or grasp_array.shape[1:] != (4, 4):
        raise ContactGraspNetInputError("inconsistent_grasp_outputs")
    count = int(grasp_array.shape[0])
    if contact_array.size != count * 3:
        raise ContactGraspNetInputError("inconsistent_grasp_outputs")
    contact_array = contact_array.reshape(count, 3)
    if score_array.shape != (count,) or opening_array.shape != (count,):
        raise ContactGraspNetInputError("inconsistent_grasp_outputs")
    if not (
        np.isfinite(grasp_array).all()
        and np.isfinite(score_array).all()
        and np.isfinite(contact_array).all()
        and np.isfinite(opening_array).all()
    ):
        raise ContactGraspNetInputError("inconsistent_grasp_outputs")
    if np.any(opening_array < 0) or np.any(opening_array > max_gripper_width + 1e-6):
        raise ContactGraspNetInputError("inconsistent_grasp_outputs")

    axes = np.asarray(_CONTACT_TO_GRASPNET_AXES, dtype=np.float64)
    normalized: list[tuple[int, float, list[float], list[list[float]], list[float], float]] = []
    for index in range(count):
        native_pose = grasp_array[index]
        if not _is_rigid_transform(native_pose):
            raise ContactGraspNetInputError("inconsistent_grasp_outputs")
        translation = native_pose[:3, 3]
        rotation = native_pose[:3, :3] @ axes
        if not _is_rotation_matrix(rotation):
            raise ContactGraspNetInputError("inconsistent_grasp_outputs")
        tip = translation + GRIPPER_DEPTH * rotation[:, 0]
        if not np.isfinite(tip).all():
            raise ContactGraspNetInputError("inconsistent_grasp_outputs")
        normalized.append(
            (
                index,
                float(score_array[index]),
                _float_list(translation),
                _float_matrix(rotation),
                _float_list(tip),
                float(opening_array[index]),
            )
        )

    ranked = sorted(normalized, key=lambda value: (-value[1], value[0]))[:max_candidates]
    candidates: list[dict[str, Any]] = []
    for output_index, (_, score, translation, rotation, tip, opening) in enumerate(ranked):
        candidates.append(
            {
                "id": f"grasp_{output_index:03d}",
                "frame": FRAME,
                "camera_frame": CAMERA_FRAME,
                "grasp_frame": GRASP_FRAME,
                "source_model": SOURCE_MODEL,
                "gripper_model": GRIPPER_MODEL,
                "score": score,
                "translation_xyz": translation,
                "rotation_matrix": rotation,
                "gripper_depth": GRIPPER_DEPTH,
                "width": opening,
                "gripper_tip_position_xyz": tip,
                "contact_point_xyz": _float_list(contact_array[ranked[output_index][0]]),
            }
        )
    return candidates, count


def failure_result(
    *,
    reason: str,
    content: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "success": False,
        "content": content,
        "details": {
            "tool": TOOL_NAME,
            "backend": BACKEND_NAME,
            "model": MODEL_NAME,
            "mode": MODE,
            "frame": FRAME,
            "camera_frame": CAMERA_FRAME,
            "grasp_frame": GRASP_FRAME,
            "candidate_count": 0,
            "grasp_candidates": [],
            "artifacts": [],
            "reason": reason,
            "metadata": metadata,
        },
    }


def _failure_content(reason: str) -> str:
    content = f"Contact-GraspNet grasp prediction failed: {reason}."
    if reason in {
        "invalid_depth_scale",
        "depth_scale_mismatch",
        "empty_point_cloud_after_depth_filter",
    }:
        return f"{content} {_DEPTH_SCALE_GUIDANCE}"
    return content


def _register_checkpoint_safe_globals(*, torch: Any, np: Any) -> None:
    numpy_core = getattr(np, "_core", None)
    if numpy_core is None:
        numpy_core = np.core
    torch.serialization.add_safe_globals(
        [
            (numpy_core.multiarray.scalar, "numpy.core.multiarray.scalar"),
            np.dtype,
            type(np.dtype(np.float64)),
        ]
    )


def load_checkpoint_model_state(
    *,
    torch: Any,
    np: Any,
    checkpoint_path: Path,
    device: Any,
) -> dict[str, Any]:
    """Load only the model state dict using PyTorch's restricted loader."""

    _register_checkpoint_safe_globals(torch=torch, np=np)
    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if not isinstance(state, dict) or not isinstance(state.get("model"), dict):
        raise RuntimeError("checkpoint does not contain a model state dict")
    return state["model"]


def seed_inference(*, seed: int, np: Any, torch: Any) -> None:
    """Reset all RNGs used by the dedicated inference process."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reject_median_outliers(data: Any, *, m: float) -> Any:
    np, _Image = _load_image_dependencies()
    distances = np.abs(data - np.median(data, axis=0, keepdims=True))
    return data[np.sum(distances, axis=1) < m]


def _is_rigid_transform(matrix: Any) -> bool:
    np, _Image = _load_image_dependencies()
    return bool(
        matrix.shape == (4, 4)
        and np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        and _is_rotation_matrix(matrix[:3, :3])
    )


def _is_rotation_matrix(rotation: Any) -> bool:
    np, _Image = _load_image_dependencies()
    return bool(
        rotation.shape == (3, 3)
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
    return {**metadata, "duration_s": round(time.perf_counter() - start, 4)}
