"""UniDepth V2 metric-depth backend for the OpenETA MCP service."""

from __future__ import annotations

import base64
import io
import math
import threading
import time
from pathlib import Path
from typing import Any, Mapping


DEFAULT_MODEL_ID = "lpiccinelli/unidepth-v2-vitl14"
CONFIDENCE_SEMANTICS = "higher_is_better"


class UniDepthInputError(Exception):
    """The request cannot satisfy the UniDepth service contract."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class UniDepthV2Backend:
    """Lazy wrapper around the official UniDepth V2 inference API."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        resolution_level: int = 4,
    ) -> None:
        if not 0 <= int(resolution_level) < 10:
            raise ValueError("resolution_level must be in [0, 10)")
        self.model_id = str(model_id)
        self.device_request = str(device)
        self.resolution_level = int(resolution_level)
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device = ""
        self._lock = threading.Lock()
        self._model_load_lock = threading.Lock()

    def estimate_depth(
        self,
        *,
        rgb: Mapping[str, Any],
        intrinsics: Mapping[str, Any],
        camera_id: str = "camera",
        camera_model: str = "pinhole",
        calibration_profile_id: str = "",
        bundle_id: str = "",
        resolution_level: int | None = None,
    ) -> dict[str, Any]:
        """Estimate metric depth and relative confidence for one RGB frame."""

        started = time.perf_counter()
        metadata = {
            "camera_id": str(camera_id),
            "camera_model": str(camera_model),
            "calibration_profile_id": str(calibration_profile_id),
            "bundle_id": str(bundle_id),
            "model_id": self.model_id,
        }
        try:
            if camera_model.lower() != "pinhole":
                raise UniDepthInputError("unsupported_camera_model")
            parsed_intrinsics = _parse_intrinsics(intrinsics)
            np, Image = _load_image_dependencies()
            rgb_array = _decode_rgb(rgb, np=np, Image=Image)
            level = self.resolution_level if resolution_level is None else int(resolution_level)
            if not 0 <= level < 10:
                raise UniDepthInputError("invalid_resolution_level")
            model, torch, device = self._get_model()
            rgb_tensor = (
                torch.from_numpy(rgb_array.copy())
                .permute(2, 0, 1)
                .contiguous()
                .to(device)
            )
            camera_matrix = torch.tensor(
                [
                    [parsed_intrinsics["fx"], 0.0, parsed_intrinsics["cx"]],
                    [0.0, parsed_intrinsics["fy"], parsed_intrinsics["cy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
                device=device,
            )
            with self._lock:
                model.resolution_level = level
                with torch.inference_mode():
                    predictions = model.infer(rgb_tensor, camera_matrix)
            depth = _prediction_hw(
                predictions,
                "depth",
                shape=rgb_array.shape[:2],
                np=np,
            )
            confidence = _prediction_hw(
                predictions,
                "confidence",
                shape=rgb_array.shape[:2],
                np=np,
            )
            valid_depth = np.isfinite(depth) & (depth > 0)
            if not bool(valid_depth.any()):
                raise RuntimeError("UniDepth returned no finite positive depth")
            metadata.update(
                {
                    "device": device,
                    "resolution_level": level,
                    "image_size": [int(rgb_array.shape[1]), int(rgb_array.shape[0])],
                    "depth_units": "metres",
                    "used_calibrated_camera": True,
                    "valid_depth_ratio": float(valid_depth.mean()),
                    "inference_seconds": round(time.perf_counter() - started, 6),
                }
            )
            return {
                "success": True,
                "content": "UniDepth V2 metric depth estimation completed.",
                "details": {
                    "tool": "estimate_depth",
                    "backend": "unidepth_v2_mcp",
                    "model": self.model_id,
                    "depth_npy_base64": _encode_npy(depth, np=np),
                    "confidence_npy_base64": _encode_npy(confidence, np=np),
                    "confidence_semantics": CONFIDENCE_SEMANTICS,
                    "intrinsics": parsed_intrinsics,
                    "artifacts": [],
                    "metadata": metadata,
                },
            }
        except UniDepthInputError as exc:
            return _failure(
                reason=exc.reason,
                content=f"UniDepth V2 input rejected: {exc.reason}.",
                metadata=_with_duration(metadata, started),
            )
        except Exception as exc:  # noqa: BLE001 - model failures stay structured.
            reason = (
                "model_load_failed"
                if self._model is None
                else "model_inference_failed"
            )
            return _failure(
                reason=reason,
                content=f"UniDepth V2 depth estimation failed: {exc}",
                metadata=_with_duration(
                    {**metadata, "error_type": type(exc).__name__},
                    started,
                ),
            )

    def _get_model(self) -> tuple[Any, Any, str]:
        if self._model is not None and self._torch is not None:
            return self._model, self._torch, self._device
        with self._model_load_lock:
            if self._model is not None and self._torch is not None:
                return self._model, self._torch, self._device
            try:
                import torch
                from unidepth.models import UniDepthV2
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"failed to import UniDepth V2 dependencies: {exc}"
                ) from exc
            device = _resolve_device(self.device_request, torch=torch)
            model_source = (
                str(Path(self.model_id).expanduser().resolve())
                if Path(self.model_id).expanduser().exists()
                else self.model_id
            )
            model = UniDepthV2.from_pretrained(model_source)
            model = model.to(device).eval()
            self._model = model
            self._torch = torch
            self._device = device
            return model, torch, device


def _load_image_dependencies() -> tuple[Any, Any]:
    import numpy as np
    from PIL import Image

    return np, Image


def _decode_rgb(payload: Mapping[str, Any], *, np: Any, Image: Any) -> Any:
    encoded = payload.get("base64")
    if not isinstance(encoded, str) or not encoded:
        raise UniDepthInputError("missing_rgb")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise UniDepthInputError("rgb_decode_failed") from exc
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise UniDepthInputError("invalid_rgb_shape")
    return array


def _parse_intrinsics(value: Mapping[str, Any]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for key in ("fx", "fy", "cx", "cy"):
        try:
            number = float(value[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise UniDepthInputError("invalid_intrinsics") from exc
        if not math.isfinite(number) or (key in {"fx", "fy"} and number <= 0):
            raise UniDepthInputError("invalid_intrinsics")
        parsed[key] = number
    return parsed


def _prediction_hw(
    predictions: Mapping[str, Any],
    key: str,
    *,
    shape: tuple[int, int],
    np: Any,
) -> Any:
    if key not in predictions:
        raise RuntimeError(f"UniDepth response missing {key}")
    value = predictions[key]
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype=np.float32).squeeze()
    if array.shape != shape:
        raise RuntimeError(
            f"UniDepth {key} shape {array.shape} does not match RGB shape {shape}"
        )
    return array


def _encode_npy(array: Any, *, np: Any) -> str:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _resolve_device(requested: str, *, torch: Any) -> str:
    value = requested.strip().lower()
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _failure(
    *,
    reason: str,
    content: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "success": False,
        "content": content,
        "details": {
            "tool": "estimate_depth",
            "backend": "unidepth_v2_mcp",
            "model": str(metadata.get("model_id") or DEFAULT_MODEL_ID),
            "reason": reason,
            "confidence_semantics": CONFIDENCE_SEMANTICS,
            "artifacts": [],
            "metadata": dict(metadata),
        },
    }


def _with_duration(metadata: Mapping[str, Any], started: float) -> dict[str, Any]:
    return {
        **dict(metadata),
        "inference_seconds": round(time.perf_counter() - started, 6),
    }
