"""Real-robot RGB-D depth and point-cloud enhancement helpers."""

from __future__ import annotations

import json
from hashlib import sha256
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol
from uuid import uuid4

import numpy as np
from PIL import Image

from adapter.protocol import JsonDict
from agent.runtime.artifact_paths import artifact_session_root, safe_artifact_component


DEPTH_ENHANCEMENT_SCHEMA_VERSION = "openeta.depth_enhancement.v1"
DEFAULT_DEPTH_ENHANCEMENT_OUTPUT_ROOT = Path("tmp") / "tool_result" / "depth_enhancement"
DEFAULT_DEPTH_ENHANCEMENT_IMAGE_ROOT = Path("tmp") / "image"

PROVENANCE_UNKNOWN = 0
PROVENANCE_SENSOR = 1
PROVENANCE_MONO_FILLED = 2


@dataclass(frozen=True, slots=True)
class DepthPriorPrediction:
    """Metric monocular depth prior output for one RGB frame."""

    depth_m: Any
    confidence: Any | None = None
    confidence_semantics: str = "higher_is_better"
    points_camera: Any | None = None
    intrinsics: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DepthPriorBackend(Protocol):
    """Backend interface for local or remote monocular depth priors."""

    def infer(
        self,
        *,
        rgb: Any,
        intrinsics: Mapping[str, float],
        camera_model: str,
        profile_id: str,
    ) -> DepthPriorPrediction | None:
        ...


class NoopDepthPriorBackend:
    """Default backend that preserves raw RGB-D behavior."""

    def infer(
        self,
        *,
        rgb: Any,
        intrinsics: Mapping[str, float],
        camera_model: str,
        profile_id: str,
    ) -> DepthPriorPrediction | None:
        del rgb, intrinsics, camera_model, profile_id
        return None


@dataclass(frozen=True, slots=True)
class DepthEnhancementConfig:
    """Conservative fusion policy for real-robot depth enhancement."""

    min_depth_m: float = 0.02
    max_depth_m: float = 5.0
    min_alignment_pixels: int = 128
    alignment_trim_fraction: float = 0.1
    mono_confidence_drop_quantile: float = 0.2
    sensor_confidence_threshold: float = 0.5
    allow_mono_fill_low_confidence_sensor: bool = False
    min_alignment_scale: float = 0.5
    max_alignment_scale: float = 2.0
    max_fill_ratio: float = 0.5
    disagreement_threshold_m: float = 0.05
    max_large_disagreement_ratio: float = 0.2
    edge_guard_pixels: int = 1
    depth_edge_threshold_m: float = 0.05
    rgb_edge_threshold: float = 0.25
    require_registration: bool = False
    max_timestamp_skew_s: float = 0.05


@dataclass(slots=True)
class DepthEnhancementResult:
    """Enhanced frame outputs plus lightweight report metadata."""

    camera_id: str
    calibration_profile_id: str
    enabled: bool
    reason: str
    fused_depth_m: np.ndarray
    safety_depth_m: np.ndarray
    provenance_mask: np.ndarray
    points_camera: np.ndarray
    point_provenance: np.ndarray
    safety_points_camera: np.ndarray
    alignment: JsonDict
    quality: JsonDict
    prior: JsonDict
    source: JsonDict = field(default_factory=dict)
    diagnostics: list[JsonDict] = field(default_factory=list)

    def report(self) -> JsonDict:
        return {
            "schema_version": DEPTH_ENHANCEMENT_SCHEMA_VERSION,
            "enabled": self.enabled,
            "reason": self.reason,
            "camera_id": self.camera_id,
            "calibration_profile_id": self.calibration_profile_id,
            "source": self.source,
            "prior": self.prior,
            "alignment": self.alignment,
            "quality": self.quality,
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True, slots=True)
class DepthEnhancementArtifacts:
    """Local references for materialized depth enhancement artifacts."""

    report_path: str
    fused_depth_npy: str
    safety_depth_npy: str
    point_cloud_npz: str
    safety_point_cloud_npz: str
    sensor_depth_png: str
    fused_depth_png: str
    safety_depth_png: str
    provenance_mask_png: str
    chars: int

    def to_dict(self) -> JsonDict:
        grep_hint = f"grep -n '<pattern>' {self.report_path}"
        return {
            "type": "json",
            "schema_version": DEPTH_ENHANCEMENT_SCHEMA_VERSION,
            "index": "depth_enhancement",
            "path": self.report_path,
            "chars": self.chars,
            "grep_hint": grep_hint,
            "outputs": {
                "fused_depth_npy": self.fused_depth_npy,
                "candidate_depth_npy": self.fused_depth_npy,
                "safety_depth_npy": self.safety_depth_npy,
                "point_cloud_npz": self.point_cloud_npz,
                "candidate_point_cloud_npz": self.point_cloud_npz,
                "safety_point_cloud_npz": self.safety_point_cloud_npz,
                "sensor_depth_png": self.sensor_depth_png,
                "fused_depth_png": self.fused_depth_png,
                "candidate_depth_png": self.fused_depth_png,
                "safety_depth_png": self.safety_depth_png,
                "provenance_mask_png": self.provenance_mask_png,
            },
        }


def enhance_rgbd_depth(
    *,
    rgb: Any,
    sensor_depth_m: Any,
    intrinsics: Mapping[str, Any],
    camera_id: str = "camera",
    calibration_profile_id: str = "",
    camera_model: str = "pinhole",
    prior_prediction: DepthPriorPrediction | None = None,
    prior_backend: DepthPriorBackend | None = None,
    sensor_confidence: Any | None = None,
    registration_status: str = "",
    rgb_timestamp_s: float | None = None,
    depth_timestamp_s: float | None = None,
    scene_epoch: int | None = None,
    calibration_hash: str = "",
    config: DepthEnhancementConfig | None = None,
) -> DepthEnhancementResult:
    """Fuse RGB-D sensor depth with an optional metric monocular prior.

    The policy is sensor-first: valid sensor depth is preserved, and monocular
    depth is used only to fill missing sensor pixels after robust scale
    alignment against reliable sensor measurements.
    """

    cfg = config or DepthEnhancementConfig()
    rgb_array = np.asarray(rgb)
    sensor_depth = np.asarray(sensor_depth_m, dtype=np.float32)
    _validate_frame_inputs(rgb_array=rgb_array, depth=sensor_depth, intrinsics=intrinsics)
    parsed_intrinsics = _parse_intrinsics(intrinsics)

    sensor_valid = _valid_depth(
        sensor_depth,
        min_depth_m=cfg.min_depth_m,
        max_depth_m=cfg.max_depth_m,
    )
    edge_guard = _edge_guard(
        rgb_array=rgb_array,
        depth=sensor_depth,
        valid=sensor_valid,
        config=cfg,
    )
    confidence_reliable = _sensor_confidence_reliable(
        sensor_confidence,
        sensor_valid.shape,
        threshold=cfg.sensor_confidence_threshold,
    )
    sensor_reliable = sensor_valid & confidence_reliable & ~edge_guard
    safety_depth = np.where(sensor_reliable, sensor_depth, np.nan).astype(np.float32)
    source = {
        "scene_epoch": scene_epoch,
        "rgb_timestamp_s": rgb_timestamp_s,
        "depth_timestamp_s": depth_timestamp_s,
        "registration_status": registration_status or "unspecified",
        "calibration_hash": calibration_hash,
        "intrinsics": dict(parsed_intrinsics),
    }

    if prior_prediction is None and prior_backend is not None:
        prior_prediction = prior_backend.infer(
            rgb=rgb_array,
            intrinsics=parsed_intrinsics,
            camera_model=camera_model,
            profile_id=calibration_profile_id,
        )

    prior_info: JsonDict = {
        "backend": "none",
        "model": "none",
        "used_calibrated_camera": True,
        "confidence_semantics": "none",
    }
    diagnostics: list[JsonDict] = []

    precondition_reason = _enhancement_precondition_failure(
        registration_status=registration_status,
        rgb_timestamp_s=rgb_timestamp_s,
        depth_timestamp_s=depth_timestamp_s,
        config=cfg,
    )
    if prior_prediction is None or precondition_reason:
        fused = safety_depth.copy()
        provenance = np.where(
            sensor_reliable,
            PROVENANCE_SENSOR,
            PROVENANCE_UNKNOWN,
        ).astype(np.uint8)
        points, point_provenance = _backproject_depth(
            fused,
            provenance,
            intrinsics=parsed_intrinsics,
        )
        safety_points = points.copy()
        if precondition_reason:
            diagnostics.append({"code": precondition_reason})
        quality = _quality_report(
            enabled=False,
            sensor_valid=sensor_valid,
            provenance=provenance,
            large_disagreement_ratio=0.0,
            config=cfg,
        )
        return DepthEnhancementResult(
            camera_id=camera_id,
            calibration_profile_id=calibration_profile_id,
            enabled=False,
            reason=precondition_reason or "no_depth_prior",
            fused_depth_m=fused,
            safety_depth_m=safety_depth,
            provenance_mask=provenance,
            points_camera=points,
            point_provenance=point_provenance,
            safety_points_camera=safety_points,
            alignment={
                "mode": "none",
                "scale": 1.0,
                "offset_m": 0.0,
                "reliable_pixel_count": int(sensor_reliable.sum()),
            },
            quality=quality,
            prior=prior_info,
            source=source,
            diagnostics=diagnostics,
        )

    mono_depth = np.asarray(prior_prediction.depth_m, dtype=np.float32)
    if mono_depth.shape != sensor_depth.shape:
        raise ValueError("prior depth shape must match sensor depth shape")
    mono_valid = _valid_depth(mono_depth, min_depth_m=cfg.min_depth_m, max_depth_m=cfg.max_depth_m)
    mono_confident = _mono_confident(
        prior_prediction.confidence,
        mono_depth.shape,
        semantics=prior_prediction.confidence_semantics,
        config=cfg,
    )
    alignment_mask = sensor_reliable & mono_valid & mono_confident
    scale, alignment_count = _fit_scale_only(
        sensor_depth=sensor_depth,
        mono_depth=mono_depth,
        mask=alignment_mask,
        config=cfg,
    )
    alignment_reason = ""
    if scale is not None and not cfg.min_alignment_scale <= scale <= cfg.max_alignment_scale:
        alignment_reason = "alignment_scale_out_of_bounds"
        diagnostics.append(
            {
                "code": alignment_reason,
                "scale": float(scale),
                "min_alignment_scale": float(cfg.min_alignment_scale),
                "max_alignment_scale": float(cfg.max_alignment_scale),
            }
        )
        scale = None
    if scale is None:
        diagnostics.append(
            {
                "code": "insufficient_alignment_pixels",
                "alignment_pixel_count": int(alignment_count),
                "min_alignment_pixels": int(cfg.min_alignment_pixels),
            }
        )
        fused = safety_depth.copy()
        provenance = np.where(
            sensor_reliable,
            PROVENANCE_SENSOR,
            PROVENANCE_UNKNOWN,
        ).astype(np.uint8)
        enabled = False
        reason = alignment_reason or "insufficient_alignment_pixels"
        mono_aligned = mono_depth
    else:
        mono_aligned = (mono_depth * float(scale)).astype(np.float32)
        mono_aligned_valid = _valid_depth(
            mono_aligned,
            min_depth_m=cfg.min_depth_m,
            max_depth_m=cfg.max_depth_m,
        )
        fillable_sensor = ~sensor_valid
        if cfg.allow_mono_fill_low_confidence_sensor:
            fillable_sensor |= sensor_valid & ~confidence_reliable
        fill_mask = fillable_sensor & mono_aligned_valid & mono_confident & ~edge_guard
        fill_ratio = float(fill_mask.sum() / max(fill_mask.size, 1))
        if fill_ratio > cfg.max_fill_ratio:
            diagnostics.append(
                {
                    "code": "fill_ratio_exceeded",
                    "fill_ratio": fill_ratio,
                    "max_fill_ratio": float(cfg.max_fill_ratio),
                }
            )
            fill_mask[:] = False
        fused = safety_depth.copy()
        fused[fill_mask] = mono_aligned[fill_mask]
        provenance = np.full(sensor_depth.shape, PROVENANCE_UNKNOWN, dtype=np.uint8)
        provenance[sensor_reliable] = PROVENANCE_SENSOR
        provenance[fill_mask] = PROVENANCE_MONO_FILLED
        enabled = bool(fill_mask.any())
        reason = (
            "enhanced"
            if enabled
            else "fill_ratio_exceeded"
            if fill_ratio > cfg.max_fill_ratio
            else "no_fillable_pixels"
        )

    large_disagreement_ratio = _large_disagreement_ratio(
        sensor_depth=sensor_depth,
        mono_aligned=mono_aligned,
        mask=sensor_reliable & mono_valid,
        threshold_m=cfg.disagreement_threshold_m,
    )
    points, point_provenance = _backproject_depth(
        fused,
        provenance,
        intrinsics=parsed_intrinsics,
    )
    safety_points, _ = _backproject_depth(
        safety_depth,
        np.where(sensor_reliable, PROVENANCE_SENSOR, PROVENANCE_UNKNOWN).astype(np.uint8),
        intrinsics=parsed_intrinsics,
    )
    prior_info.update(_prior_report(prior_prediction))
    alignment = {
        "mode": "scale_only" if scale is not None else "none",
        "scale": 1.0 if scale is None else float(scale),
        "offset_m": 0.0,
        "reliable_pixel_count": int(alignment_count),
        "robust_loss": "trimmed_median_ratio",
        **_alignment_residuals(
            sensor_depth=sensor_depth,
            mono_aligned=mono_aligned,
            mask=sensor_reliable & _valid_depth(
                mono_aligned,
                min_depth_m=cfg.min_depth_m,
                max_depth_m=cfg.max_depth_m,
            ),
        ),
    }
    quality = _quality_report(
        enabled=enabled,
        sensor_valid=sensor_valid,
        provenance=provenance,
        large_disagreement_ratio=large_disagreement_ratio,
        config=cfg,
    )
    return DepthEnhancementResult(
        camera_id=camera_id,
        calibration_profile_id=calibration_profile_id,
        enabled=enabled,
        reason=reason,
        fused_depth_m=fused,
        safety_depth_m=safety_depth,
        provenance_mask=provenance,
        points_camera=points,
        point_provenance=point_provenance,
        safety_points_camera=safety_points,
        alignment=alignment,
        quality=quality,
        prior=prior_info,
        source=source,
        diagnostics=diagnostics,
    )


def materialize_depth_enhancement(
    result: DepthEnhancementResult,
    *,
    sensor_depth_m: Any,
    output_root: str | Path = DEFAULT_DEPTH_ENHANCEMENT_OUTPUT_ROOT,
    image_root: str | Path = DEFAULT_DEPTH_ENHANCEMENT_IMAGE_ROOT,
    bundle_id: str | None = None,
    session_id: str | None = None,
    source_rgb_path: str = "",
    source_depth_path: str = "",
    source_sensor_confidence_path: str = "",
) -> DepthEnhancementArtifacts:
    """Write enhanced depth, point cloud, masks, and report to local artifacts."""

    bundle = safe_artifact_component(bundle_id or f"depth-{uuid4().hex[:8]}", fallback="bundle")
    camera = safe_artifact_component(result.camera_id, fallback="camera")
    report_root = artifact_session_root(output_root, session_id) / bundle
    image_bundle_root = artifact_session_root(image_root, session_id)
    depth_image_root = image_bundle_root / "depth" / bundle
    mask_image_root = image_bundle_root / "mask" / bundle
    report_root.mkdir(parents=True, exist_ok=True)
    depth_image_root.mkdir(parents=True, exist_ok=True)
    mask_image_root.mkdir(parents=True, exist_ok=True)

    fused_depth_npy = report_root / f"{camera}-fused-depth.npy"
    safety_depth_npy = report_root / f"{camera}-safety-depth.npy"
    point_cloud_npz = report_root / f"{camera}-point-cloud.npz"
    safety_point_cloud_npz = report_root / f"{camera}-safety-point-cloud.npz"
    sensor_depth_png = depth_image_root / f"{camera}-sensor.png"
    fused_depth_png = depth_image_root / f"{camera}-fused.png"
    safety_depth_png = depth_image_root / f"{camera}-safety.png"
    provenance_mask_png = mask_image_root / f"{camera}-provenance.png"
    report_path = report_root / f"{camera}-depth-enhancement.json"

    np.save(fused_depth_npy, result.fused_depth_m.astype(np.float32))
    np.save(safety_depth_npy, result.safety_depth_m.astype(np.float32))
    np.savez_compressed(
        point_cloud_npz,
        points_camera=result.points_camera.astype(np.float32),
        provenance=result.point_provenance.astype(np.uint8),
    )
    np.savez_compressed(
        safety_point_cloud_npz,
        points_camera=result.safety_points_camera.astype(np.float32),
        provenance=np.full(
            result.safety_points_camera.shape[0],
            PROVENANCE_SENSOR,
            dtype=np.uint8,
        ),
    )
    Image.fromarray(_depth_to_uint16_mm(sensor_depth_m)).save(sensor_depth_png)
    Image.fromarray(_depth_to_uint16_mm(result.fused_depth_m)).save(fused_depth_png)
    Image.fromarray(_depth_to_uint16_mm(result.safety_depth_m)).save(safety_depth_png)
    Image.fromarray(result.provenance_mask.astype(np.uint8)).save(provenance_mask_png)

    report = result.report()
    report["source"].update(
        {
            "rgb_path": _resolved_path_or_empty(source_rgb_path),
            "rgb_sha256": _sha256_file_or_empty(source_rgb_path),
            "sensor_depth_path": _resolved_path_or_empty(source_depth_path),
            "sensor_depth_sha256": _sha256_file_or_empty(source_depth_path),
            "sensor_confidence_path": _resolved_path_or_empty(
                source_sensor_confidence_path
            ),
            "sensor_confidence_sha256": _sha256_file_or_empty(
                source_sensor_confidence_path
            ),
        }
    )
    report["outputs"] = {
        "fused_depth_npy": str(fused_depth_npy.resolve()),
        "candidate_depth_npy": str(fused_depth_npy.resolve()),
        "safety_depth_npy": str(safety_depth_npy.resolve()),
        "point_cloud_npz": str(point_cloud_npz.resolve()),
        "candidate_point_cloud_npz": str(point_cloud_npz.resolve()),
        "safety_point_cloud_npz": str(safety_point_cloud_npz.resolve()),
        "sensor_depth_png": str(sensor_depth_png.resolve()),
        "fused_depth_png": str(fused_depth_png.resolve()),
        "candidate_depth_png": str(fused_depth_png.resolve()),
        "safety_depth_png": str(safety_depth_png.resolve()),
        "depth_units": "millimeters",
        "depth_scale": 1000.0,
        "provenance_mask_png": str(provenance_mask_png.resolve()),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    report_path.write_text(text, encoding="utf-8")
    return DepthEnhancementArtifacts(
        report_path=str(report_path.resolve()),
        fused_depth_npy=str(fused_depth_npy.resolve()),
        safety_depth_npy=str(safety_depth_npy.resolve()),
        point_cloud_npz=str(point_cloud_npz.resolve()),
        safety_point_cloud_npz=str(safety_point_cloud_npz.resolve()),
        sensor_depth_png=str(sensor_depth_png.resolve()),
        fused_depth_png=str(fused_depth_png.resolve()),
        safety_depth_png=str(safety_depth_png.resolve()),
        provenance_mask_png=str(provenance_mask_png.resolve()),
        chars=len(text),
    )


def _validate_frame_inputs(
    *,
    rgb_array: np.ndarray,
    depth: np.ndarray,
    intrinsics: Mapping[str, Any],
) -> None:
    if rgb_array.ndim != 3 or rgb_array.shape[2] < 3:
        raise ValueError("rgb must have shape HxWxC with at least 3 channels")
    if depth.ndim != 2:
        raise ValueError("sensor_depth_m must have shape HxW")
    if rgb_array.shape[:2] != depth.shape:
        raise ValueError("rgb and sensor_depth_m shapes must match")
    _parse_intrinsics(intrinsics)


def _parse_intrinsics(intrinsics: Mapping[str, Any]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for key in ("fx", "fy", "cx", "cy"):
        if key not in intrinsics:
            raise ValueError(f"missing camera intrinsic {key}")
        value = float(intrinsics[key])
        if not np.isfinite(value) or (key in {"fx", "fy"} and value <= 0):
            raise ValueError(f"invalid camera intrinsic {key}")
        parsed[key] = value
    parsed["scale"] = float(intrinsics.get("scale", 1000.0))
    return parsed


def _valid_depth(depth: np.ndarray, *, min_depth_m: float, max_depth_m: float) -> np.ndarray:
    return np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)


def _sensor_confidence_reliable(
    confidence: Any | None,
    shape: tuple[int, int],
    *,
    threshold: float,
) -> np.ndarray:
    if confidence is None:
        return np.ones(shape, dtype=bool)
    values = np.asarray(confidence)
    if values.shape != shape:
        raise ValueError("sensor_confidence shape must match depth shape")
    if values.dtype == bool:
        return values.astype(bool)
    return np.isfinite(values) & (values >= float(threshold))


def _mono_confident(
    confidence: Any | None,
    shape: tuple[int, int],
    *,
    semantics: str,
    config: DepthEnhancementConfig,
) -> np.ndarray:
    if confidence is None:
        return np.ones(shape, dtype=bool)
    values = np.asarray(confidence, dtype=np.float32)
    if values.shape != shape:
        raise ValueError("prior confidence shape must match depth shape")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(shape, dtype=bool)
    quantile = min(max(float(config.mono_confidence_drop_quantile), 0.0), 0.95)
    if semantics == "higher_is_better":
        threshold = float(np.quantile(finite, quantile))
        return np.isfinite(values) & (values >= threshold)
    if semantics == "lower_is_better":
        threshold = float(np.quantile(finite, 1.0 - quantile))
        return np.isfinite(values) & (values <= threshold)
    raise ValueError(
        "prior confidence_semantics must be higher_is_better or lower_is_better"
    )


def _edge_guard(
    *,
    rgb_array: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    config: DepthEnhancementConfig,
) -> np.ndarray:
    if config.edge_guard_pixels <= 0:
        return np.zeros(depth.shape, dtype=bool)
    guard = np.zeros(depth.shape, dtype=bool)
    safe_depth = np.where(valid, depth, np.nan)
    dx = np.abs(np.diff(safe_depth, axis=1))
    dy = np.abs(np.diff(safe_depth, axis=0))
    guard[:, 1:] |= np.isfinite(dx) & (dx > config.depth_edge_threshold_m)
    guard[:, :-1] |= np.isfinite(dx) & (dx > config.depth_edge_threshold_m)
    guard[1:, :] |= np.isfinite(dy) & (dy > config.depth_edge_threshold_m)
    guard[:-1, :] |= np.isfinite(dy) & (dy > config.depth_edge_threshold_m)

    gray = rgb_array[..., :3].astype(np.float32).mean(axis=2) / 255.0
    rgb_dx = np.abs(np.diff(gray, axis=1))
    rgb_dy = np.abs(np.diff(gray, axis=0))
    guard[:, 1:] |= rgb_dx > config.rgb_edge_threshold
    guard[:, :-1] |= rgb_dx > config.rgb_edge_threshold
    guard[1:, :] |= rgb_dy > config.rgb_edge_threshold
    guard[:-1, :] |= rgb_dy > config.rgb_edge_threshold
    return _dilate(guard, radius=config.edge_guard_pixels)


def _dilate(mask: np.ndarray, *, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    result = mask.astype(bool)
    for _ in range(radius):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        grown = np.zeros_like(result)
        for dy in range(3):
            for dx in range(3):
                grown |= padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
        result = grown
    return result


def _fit_scale_only(
    *,
    sensor_depth: np.ndarray,
    mono_depth: np.ndarray,
    mask: np.ndarray,
    config: DepthEnhancementConfig,
) -> tuple[float | None, int]:
    ratios = sensor_depth[mask] / mono_depth[mask]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    count = int(ratios.size)
    if count < config.min_alignment_pixels:
        return None, count
    trim = min(max(float(config.alignment_trim_fraction), 0.0), 0.45)
    if trim > 0.0 and count >= 4:
        lo, hi = np.quantile(ratios, [trim, 1.0 - trim])
        ratios = ratios[(ratios >= lo) & (ratios <= hi)]
    if ratios.size == 0:
        return None, count
    scale = float(np.median(ratios))
    if not np.isfinite(scale) or scale <= 0:
        return None, count
    return scale, count


def _large_disagreement_ratio(
    *,
    sensor_depth: np.ndarray,
    mono_aligned: np.ndarray,
    mask: np.ndarray,
    threshold_m: float,
) -> float:
    if not mask.any():
        return 0.0
    delta = np.abs(sensor_depth[mask] - mono_aligned[mask])
    return float((delta > threshold_m).sum() / max(int(delta.size), 1))


def _backproject_depth(
    depth_m: np.ndarray,
    provenance: np.ndarray,
    *,
    intrinsics: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(depth_m) & (depth_m > 0) & (provenance != PROVENANCE_UNKNOWN)
    if not valid.any():
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.uint8)
    height, width = depth_m.shape
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    z = depth_m.astype(np.float32)
    x = (u - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = (v - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    points = np.stack([x, y, z], axis=-1)[valid].astype(np.float32)
    point_provenance = provenance[valid].astype(np.uint8)
    return points, point_provenance


def _quality_report(
    *,
    enabled: bool,
    sensor_valid: np.ndarray,
    provenance: np.ndarray,
    large_disagreement_ratio: float,
    config: DepthEnhancementConfig,
) -> JsonDict:
    total = int(sensor_valid.size)
    sensor_count = int((provenance == PROVENANCE_SENSOR).sum())
    mono_count = int((provenance == PROVENANCE_MONO_FILLED).sum())
    filled_ratio = mono_count / max(total, 1)
    return {
        "sensor_valid_ratio": sensor_count / max(total, 1),
        "filled_ratio": filled_ratio,
        "mono_only_ratio": filled_ratio,
        "large_disagreement_ratio": float(large_disagreement_ratio),
        "use_for_grasp_candidate_generation": bool(
            enabled
            and mono_count > 0
            and large_disagreement_ratio <= config.max_large_disagreement_ratio
        ),
        "use_for_collision_clearance": False,
    }


def _prior_report(prediction: DepthPriorPrediction) -> JsonDict:
    metadata = dict(prediction.metadata or {})
    return {
        "backend": str(metadata.get("backend") or metadata.get("source") or "depth_prior"),
        "model": str(metadata.get("model") or "unknown"),
        "used_calibrated_camera": True,
        "confidence_semantics": prediction.confidence_semantics,
    }


def _depth_to_uint16_mm(depth_m: Any) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(np.round(depth * 1000.0), 0, 65535).astype(np.uint16)


def _enhancement_precondition_failure(
    *,
    registration_status: str,
    rgb_timestamp_s: float | None,
    depth_timestamp_s: float | None,
    config: DepthEnhancementConfig,
) -> str:
    status = registration_status.strip().lower()
    if status and status not in {"registered", "aligned", "verified"}:
        return "rgb_depth_not_registered"
    if config.require_registration and not status:
        return "registration_status_missing"
    if rgb_timestamp_s is not None and depth_timestamp_s is not None:
        if abs(float(rgb_timestamp_s) - float(depth_timestamp_s)) > config.max_timestamp_skew_s:
            return "rgb_depth_timestamp_skew"
    return ""


def _alignment_residuals(
    *,
    sensor_depth: np.ndarray,
    mono_aligned: np.ndarray,
    mask: np.ndarray,
) -> JsonDict:
    residuals = np.abs(sensor_depth[mask] - mono_aligned[mask])
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        return {"median_residual_m": None, "p95_residual_m": None}
    return {
        "median_residual_m": float(np.median(residuals)),
        "p95_residual_m": float(np.quantile(residuals, 0.95)),
    }


def _resolved_path_or_empty(path_value: str) -> str:
    if not path_value:
        return ""
    return str(Path(path_value).expanduser().resolve())


def _sha256_file_or_empty(path_value: str) -> str:
    if not path_value:
        return ""
    path = Path(path_value).expanduser()
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""
