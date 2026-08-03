"""SAM3 image segmentation backend for the OpenETA SAM3 MCP server."""

from __future__ import annotations

import base64
import io
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

_PROCESSOR: Any | None = None
_PROCESSOR_THRESHOLD: float | None = None


def segment_image_prompt(
    *,
    image_base64: str,
    prompt: str = "",
    box_xyxy: list[float] | tuple[float, float, float, float] | None = None,
    image_format: str = "png",
    confidence_threshold: float = 0.5,
    backend_version: str | None = None,
) -> dict[str, Any]:
    """Run SAM3 on one base64-encoded image plus one prompt."""

    start_time = time.perf_counter()
    metadata_base = _metadata_base(
        backend_version=backend_version,
        confidence_threshold=confidence_threshold,
    )
    if not image_base64:
        return _failure_result(
            prompt=prompt,
            reason="missing_image",
            content="SAM3 segmentation failed: missing image.",
            metadata=metadata_base,
        )
    if not prompt and box_xyxy is None:
        return _failure_result(
            prompt=prompt,
            reason="missing_prompt",
            content="SAM3 segmentation failed: missing text or box prompt.",
            metadata=metadata_base,
        )

    try:
        np, torch, Image = _load_numeric_deps()
        image_bytes = base64.b64decode(image_base64, validate=True)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - report image failures structurally.
        return _failure_result(
            prompt=prompt,
            reason="image_decode_failed",
            content=f"SAM3 segmentation failed: image decode failed: {exc}",
            metadata=_with_duration(metadata_base, start_time),
        )

    metadata_base["image_size"] = list(image.size)
    metadata_base["image_format"] = _normalise_format(image_format)
    prompt_type = "box" if box_xyxy is not None else "text"
    metadata_base["prompt_type"] = prompt_type
    normalized_box: list[float] | None = None
    if box_xyxy is not None:
        try:
            normalized_box = _box_xyxy_to_normalized_cxcywh(box_xyxy, image_size=image.size)
        except ValueError as exc:
            return _failure_result(
                prompt=prompt,
                reason="invalid_box_prompt",
                content=f"SAM3 segmentation failed: invalid box prompt: {exc}",
                metadata=_with_duration(metadata_base, start_time),
            )

    try:
        processor = _get_processor(confidence_threshold=confidence_threshold)
    except Exception as exc:  # noqa: BLE001
        return _failure_result(
            prompt=prompt,
            reason="model_load_failed",
            content=f"SAM3 segmentation failed: model load failed: {exc}",
            metadata=_with_duration(metadata_base, start_time),
        )

    try:
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else nullcontext()
        )
        with autocast_context:
            state = processor.set_image(image)
            output = (
                processor.add_geometric_prompt(box=normalized_box, label=True, state=state)
                if normalized_box is not None
                else processor.set_text_prompt(state=state, prompt=prompt)
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        return _failure_result(
            prompt=prompt,
            reason="model_inference_failed",
            content=f"SAM3 segmentation failed: model inference failed: {exc}",
            metadata=_with_duration(metadata_base, start_time),
        )

    try:
        detections, artifacts = _build_detections_and_artifacts(
            output=output,
            prompt=prompt,
            image=image,
        )
    except _InconsistentSam3Output:
        return _failure_result(
            prompt=prompt,
            reason="inconsistent_detection_outputs",
            content="SAM3 returned inconsistent detection outputs.",
            metadata=_with_duration(metadata_base, start_time),
        )
    except Exception as exc:  # noqa: BLE001
        return _failure_result(
            prompt=prompt,
            reason="artifact_encode_failed",
            content=f"SAM3 segmentation failed: artifact encode failed: {exc}",
            metadata=_with_duration(metadata_base, start_time),
        )

    content = (
        "SAM3 segmentation completed."
        if detections
        else "SAM3 segmentation completed with no detections."
    )
    return {
        "success": True,
        "content": content,
        "details": {
            "tool": "sam3",
            "backend": "sam3_mcp",
            "model": "sam3",
            "prompt": prompt,
            "prompt_type": prompt_type,
            **({"box_xyxy": [float(value) for value in box_xyxy]} if box_xyxy is not None else {}),
            "detection_count": len(detections),
            "detections": detections,
            "ranking": "score_descending",
            "artifacts": artifacts,
            "metadata": _with_duration(metadata_base, start_time),
        },
    }


def _box_xyxy_to_normalized_cxcywh(
    box_xyxy: list[float] | tuple[float, float, float, float],
    *,
    image_size: tuple[int, int],
) -> list[float]:
    if not isinstance(box_xyxy, (list, tuple)) or len(box_xyxy) != 4:
        raise ValueError("box_xyxy must contain x0, y0, x1, y1")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    try:
        x0, y0, x1, y1 = (float(value) for value in box_xyxy)
    except (TypeError, ValueError) as exc:
        raise ValueError("box coordinates must be finite numbers") from exc
    values = (x0, y0, x1, y1)
    if not all(value == value and value not in {float("inf"), float("-inf")} for value in values):
        raise ValueError("box coordinates must be finite numbers")
    x0 = max(0.0, min(float(width), x0))
    x1 = max(0.0, min(float(width), x1))
    y0 = max(0.0, min(float(height), y0))
    y1 = max(0.0, min(float(height), y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("box must have positive area after clipping")
    return [
        ((x0 + x1) * 0.5) / width,
        ((y0 + y1) * 0.5) / height,
        (x1 - x0) / width,
        (y1 - y0) / height,
    ]


def _get_processor(*, confidence_threshold: float) -> Any:
    global _PROCESSOR, _PROCESSOR_THRESHOLD
    if _PROCESSOR is None or _PROCESSOR_THRESHOLD != confidence_threshold:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model()
        _PROCESSOR = Sam3Processor(model, confidence_threshold=confidence_threshold)
        _PROCESSOR_THRESHOLD = confidence_threshold
    return _PROCESSOR


def _load_numeric_deps() -> tuple[Any, Any, Any]:
    import numpy as np
    import torch
    from PIL import Image

    return np, torch, Image


def _build_detections_and_artifacts(
    *,
    output: dict[str, Any],
    prompt: str,
    image: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    np, _torch, Image = _load_numeric_deps()
    masks = _normalise_masks(output.get("masks"))
    boxes = _boxes_to_list(output.get("boxes"))
    scores = _scores_to_list(output.get("scores"))

    mask_count = int(masks.shape[0]) if masks.size else 0
    if mask_count != len(boxes):
        raise _InconsistentSam3Output("box and mask counts differ")
    if len(scores) not in (0, mask_count):
        raise _InconsistentSam3Output("score count differs from detections")
    if mask_count == 0:
        return [], []

    detections: list[dict[str, Any]] = []
    overlay = np.asarray(image.convert("RGBA")).copy()
    for backend_index, mask in enumerate(masks):
        if not mask.any():
            raise _InconsistentSam3Output("empty mask in detection output")
        mask_png = _image_to_base64(
            Image.fromarray((mask.astype(np.uint8) * 255), mode="L"),
            fmt="png",
        )
        color = np.array([255, 0, 0, 110], dtype=np.uint8)
        overlay[mask] = (
            (0.55 * overlay[mask].astype(np.float32))
            + (0.45 * color.astype(np.float32))
        ).astype(np.uint8)
        detections.append(
            {
                "label": prompt,
                "score": scores[backend_index] if backend_index < len(scores) else None,
                "bbox_xyxy": [int(v) for v in boxes[backend_index]],
                "mask": {"format": "png", "base64": mask_png},
                "area_px": int(mask.sum()),
                "backend_index": backend_index,
            }
        )

    detections = _rank_sam3_detections(detections)

    overlay_png = _image_to_base64(Image.fromarray(overlay, mode="RGBA"), fmt="png")
    return detections, [
        {
            "artifact_type": "overlay",
            "format": "png",
            "base64": overlay_png,
        }
    ]


def _sam3_detection_sort_key(detection: dict[str, Any]) -> tuple[bool, float]:
    score = detection.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        parsed = float(score)
        if parsed == parsed and parsed not in {float("inf"), float("-inf")}:
            return True, parsed
    return False, float("-inf")


def _rank_sam3_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(detections, key=_sam3_detection_sort_key, reverse=True)
    for rank, detection in enumerate(ranked):
        detection["rank"] = rank
    return ranked


def _normalise_masks(masks: Any) -> Any:
    np, torch, _Image = _load_numeric_deps()
    if masks is None:
        return np.zeros((0, 0, 0), dtype=bool)
    if isinstance(masks, torch.Tensor):
        arr = masks.detach().cpu().numpy()
    else:
        arr = np.asarray(masks)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise _InconsistentSam3Output("mask tensor has unexpected shape")
    return arr > 0


def _boxes_to_list(boxes: Any) -> list[list[float]]:
    np, torch, _Image = _load_numeric_deps()
    if boxes is None:
        return []
    if isinstance(boxes, torch.Tensor):
        arr = boxes.detach().cpu().numpy()
    else:
        arr = np.asarray(boxes)
    arr = np.squeeze(arr)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise _InconsistentSam3Output("boxes have unexpected shape")
    return [[float(v) for v in row] for row in arr.tolist()]


def _scores_to_list(scores: Any) -> list[float]:
    np, torch, _Image = _load_numeric_deps()
    if scores is None:
        return []
    if isinstance(scores, torch.Tensor):
        return [float(x) for x in scores.detach().cpu().flatten().tolist()]
    if isinstance(scores, np.ndarray):
        return [float(x) for x in scores.flatten().tolist()]
    if isinstance(scores, (list, tuple)):
        return [float(x) for x in scores]
    return []


def _image_to_base64(image: Any, *, fmt: str) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=_pil_format(fmt))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _metadata_base(
    *,
    backend_version: str | None,
    confidence_threshold: float,
) -> dict[str, Any]:
    version = backend_version or _git_commit(REPO_ROOT)
    if version and not version.startswith("sam3@"):
        version = f"sam3@{version}"
    return {
        "confidence_threshold": confidence_threshold,
        "backend_version": version,
    }


def _with_duration(metadata: dict[str, Any], start_time: float) -> dict[str, Any]:
    result = dict(metadata)
    result["duration_s"] = time.perf_counter() - start_time
    return result


def _failure_result(
    *,
    prompt: str,
    reason: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "success": False,
        "content": content,
        "details": {
            "tool": "sam3",
            "backend": "sam3_mcp",
            "model": "sam3",
            "prompt": prompt,
            "detection_count": 0,
            "detections": [],
            "artifacts": [],
            "reason": reason,
            "metadata": dict(metadata or {}),
        },
    }


def _normalise_format(value: str) -> str:
    fmt = value.lower().lstrip(".") if value else "png"
    if fmt in {"jpg", "jpeg"}:
        return "jpeg"
    return "png" if fmt == "png" else fmt


def _pil_format(value: str) -> str:
    return "JPEG" if _normalise_format(value) == "jpeg" else "PNG"


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


class _InconsistentSam3Output(Exception):
    """SAM3 returned boxes/masks/scores that cannot form valid detections."""
