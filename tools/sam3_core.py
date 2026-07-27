"""SAM3 image segmentation backend for the OpenETA SAM3 MCP server."""

from __future__ import annotations

import base64
import io
import math
import subprocess
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

_MODEL: Any | None = None
_PROCESSOR: Any | None = None
_INFERENCE_LOCK = threading.Lock()

_POINT_CANDIDATE_COUNT = 3
_MAX_POINT_COUNT = 64


def segment_image_prompt(
    *,
    image_base64: str,
    prompt: str,
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
    metadata_base["prompt_type"] = "text"
    if not image_base64:
        return _failure_result(
            prompt=prompt,
            reason="missing_image",
            content="SAM3 segmentation failed: missing image.",
            metadata=metadata_base,
        )
    if not prompt:
        return _failure_result(
            prompt=prompt,
            reason="missing_prompt",
            content="SAM3 segmentation failed: missing prompt.",
            metadata=metadata_base,
        )

    try:
        _np, torch, Image = _load_numeric_deps()
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

    try:
        output = _run_text_inference(
            image=image,
            prompt=prompt,
            confidence_threshold=confidence_threshold,
            torch=torch,
        )
    except _Sam3ModelLoadFailed as exc:
        return _failure_result(
            prompt=prompt,
            reason="model_load_failed",
            content=f"SAM3 segmentation failed: model load failed: {exc}",
            metadata=_with_duration(metadata_base, start_time),
        )
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
            "prompt_type": "text",
            "detection_count": len(detections),
            "detections": detections,
            "ranking": "score_descending",
            "artifacts": artifacts,
            "metadata": _with_duration(metadata_base, start_time),
        },
    }


def segment_image_points(
    *,
    image_base64: str,
    points: list[dict[str, Any]] | None,
    image_format: str = "png",
    backend_version: str | None = None,
) -> dict[str, Any]:
    """Run SAM3 instance-interactive segmentation from pixel point prompts."""

    start_time = time.perf_counter()
    metadata_base = _metadata_base(
        backend_version=backend_version,
        confidence_threshold=None,
    )
    metadata_base.update(
        {
            "prompt_type": "points",
            "coordinate_units": "pixels",
            "coordinate_origin": "top_left",
            "multimask_output": True,
        }
    )
    if not image_base64:
        return _point_failure_result(
            points=[],
            reason="missing_image",
            content="SAM3 point segmentation failed: missing image.",
            metadata=metadata_base,
        )

    try:
        normalised_points = _normalise_point_prompt(points)
    except _PointPromptError as exc:
        return _point_failure_result(
            points=[],
            reason=exc.reason,
            content=f"SAM3 point segmentation failed: {exc}",
            metadata=_with_duration(metadata_base, start_time),
        )

    foreground_count = sum(point["label"] == 1 for point in normalised_points)
    background_count = len(normalised_points) - foreground_count
    metadata_base.update(
        {
            "point_count": len(normalised_points),
            "foreground_point_count": foreground_count,
            "background_point_count": background_count,
        }
    )

    try:
        from PIL import Image

        image_bytes = base64.b64decode(image_base64, validate=True)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - report image failures structurally.
        return _point_failure_result(
            points=normalised_points,
            reason="image_decode_failed",
            content=f"SAM3 point segmentation failed: image decode failed: {exc}",
            metadata=_with_duration(metadata_base, start_time),
        )

    import numpy as np

    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    metadata_base["image_size"] = list(image.size)
    metadata_base["image_format"] = _normalise_format(image_format)
    if not _points_within_image(normalised_points, image.size):
        return _point_failure_result(
            points=normalised_points,
            reason="point_out_of_bounds",
            content="SAM3 point segmentation failed: point is outside the source image.",
            metadata=_with_duration(metadata_base, start_time),
        )

    try:
        masks, scores = _run_point_inference(
            image=image,
            points=normalised_points,
            np=np,
            torch=torch,
        )
    except _Sam3ModelLoadFailed as exc:
        return _point_failure_result(
            points=normalised_points,
            reason="model_load_failed",
            content=f"SAM3 point segmentation failed: model load failed: {exc}",
            metadata=_with_duration(metadata_base, start_time),
        )
    except Exception as exc:  # noqa: BLE001
        return _point_failure_result(
            points=normalised_points,
            reason="model_inference_failed",
            content=f"SAM3 point segmentation failed: model inference failed: {exc}",
            metadata=_with_duration(metadata_base, start_time),
        )

    try:
        detections, artifacts = _build_point_detections_and_artifacts(
            masks=masks,
            scores=scores,
            points=normalised_points,
            image=image,
        )
    except _InconsistentSam3Output:
        return _point_failure_result(
            points=normalised_points,
            reason="inconsistent_detection_outputs",
            content="SAM3 returned inconsistent point-prompt outputs.",
            metadata=_with_duration(metadata_base, start_time),
        )
    except Exception as exc:  # noqa: BLE001
        return _point_failure_result(
            points=normalised_points,
            reason="artifact_encode_failed",
            content=f"SAM3 point segmentation failed: artifact encode failed: {exc}",
            metadata=_with_duration(metadata_base, start_time),
        )

    metadata_base["candidate_count"] = len(detections)
    return {
        "success": True,
        "content": "SAM3 point segmentation completed.",
        "details": {
            "tool": "sam3",
            "backend": "sam3_mcp",
            "model": "sam3",
            "prompt_type": "points",
            "points": normalised_points,
            "detection_count": len(detections),
            "detections": detections,
            "ranking": "score_descending",
            "artifacts": artifacts,
            "metadata": _with_duration(metadata_base, start_time),
        },
    }


def _get_processor(*, confidence_threshold: float) -> Any:
    global _MODEL, _PROCESSOR
    if _PROCESSOR is None:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(enable_inst_interactivity=True)
        processor = Sam3Processor(
            model,
            confidence_threshold=confidence_threshold,
        )
        _MODEL = model
        _PROCESSOR = processor
    else:
        _PROCESSOR.set_confidence_threshold(confidence_threshold)
    return _PROCESSOR


def _run_text_inference(
    *,
    image: Any,
    prompt: str,
    confidence_threshold: float,
    torch: Any,
) -> dict[str, Any]:
    state: Any = None
    output: Any = None
    with _INFERENCE_LOCK:
        try:
            try:
                processor = _get_processor(confidence_threshold=confidence_threshold)
            except Exception as exc:  # noqa: BLE001
                raise _Sam3ModelLoadFailed(str(exc)) from exc
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if torch.cuda.is_available()
                else nullcontext()
            )
            with autocast_context:
                state = processor.set_image(image)
                output = processor.set_text_prompt(state=state, prompt=prompt)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return _copy_detection_output_to_cpu(output, torch=torch)
        finally:
            state = None
            output = None
            _empty_cuda_cache(torch)


def _run_point_inference(
    *,
    image: Any,
    points: list[dict[str, Any]],
    np: Any,
    torch: Any,
) -> tuple[Any, Any]:
    state: Any = None
    raw_output: Any = None
    with _INFERENCE_LOCK:
        try:
            try:
                processor = _get_processor(confidence_threshold=0.5)
            except Exception as exc:  # noqa: BLE001
                raise _Sam3ModelLoadFailed(str(exc)) from exc
            point_coords = np.asarray(
                [[point["x"], point["y"]] for point in points],
                dtype=np.float32,
            )
            point_labels = np.asarray(
                [point["label"] for point in points],
                dtype=np.int32,
            )
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if torch.cuda.is_available()
                else nullcontext()
            )
            with autocast_context:
                state = processor.set_image(image)
                raw_output = processor.model.predict_inst(
                    state,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            masks, scores, _logits = raw_output
            return np.asarray(masks).copy(), np.asarray(scores).copy()
        finally:
            state = None
            raw_output = None
            _clear_interactive_predictor_state()
            _empty_cuda_cache(torch)


def _copy_detection_output_to_cpu(output: Any, *, torch: Any) -> dict[str, Any]:
    if not isinstance(output, dict):
        raise _InconsistentSam3Output("text output is not an object")
    copied: dict[str, Any] = {}
    for key in ("masks", "boxes", "scores"):
        value = output.get(key)
        copied[key] = (
            value.detach().cpu()
            if isinstance(value, torch.Tensor)
            else value
        )
    return copied


def _empty_cuda_cache(torch: Any) -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - cleanup must not mask inference errors.
        return


def _clear_interactive_predictor_state() -> None:
    if _PROCESSOR is None:
        return
    try:
        predictor = getattr(_PROCESSOR.model, "inst_interactive_predictor", None)
        if predictor is None:
            return
        predictor._features = None
        predictor._is_image_set = False
    except Exception:  # noqa: BLE001 - cleanup must not mask inference errors.
        return


def _normalise_point_prompt(
    points: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if points is None or points == []:
        raise _PointPromptError("missing_points", "at least one point is required")
    if not isinstance(points, list) or len(points) > _MAX_POINT_COUNT:
        raise _PointPromptError(
            "invalid_points",
            f"points must be a list containing at most {_MAX_POINT_COUNT} items",
        )

    normalised: list[dict[str, Any]] = []
    for point in points:
        if not isinstance(point, dict):
            raise _PointPromptError(
                "invalid_points",
                "each point must be an object with x, y, and label",
            )
        x = point.get("x")
        y = point.get("y")
        label = point.get("label")
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, (int, float))
            or not isinstance(y, (int, float))
            or not math.isfinite(float(x))
            or not math.isfinite(float(y))
        ):
            raise _PointPromptError(
                "invalid_points",
                "point coordinates must be finite numbers",
            )
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise _PointPromptError(
                "invalid_points",
                "point label must be 1 for foreground or 0 for background",
            )
        normalised.append({"x": float(x), "y": float(y), "label": label})

    if not any(point["label"] == 1 for point in normalised):
        raise _PointPromptError(
            "invalid_points",
            "at least one foreground point with label=1 is required",
        )
    return normalised


def _points_within_image(
    points: list[dict[str, Any]],
    image_size: tuple[int, int],
) -> bool:
    width, height = image_size
    return all(
        0.0 <= point["x"] < width and 0.0 <= point["y"] < height
        for point in points
    )


def _build_point_detections_and_artifacts(
    *,
    masks: Any,
    scores: Any,
    points: list[dict[str, Any]],
    image: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import numpy as np
    from PIL import Image

    normalised_masks = _normalise_masks(masks)
    normalised_scores = _scores_to_list(scores)
    image_width, image_height = image.size
    if normalised_masks.shape != (
        _POINT_CANDIDATE_COUNT,
        image_height,
        image_width,
    ):
        raise _InconsistentSam3Output("point masks have unexpected shape or count")
    if len(normalised_scores) != _POINT_CANDIDATE_COUNT or not all(
        math.isfinite(score) for score in normalised_scores
    ):
        raise _InconsistentSam3Output("point scores have unexpected count or values")

    detections: list[dict[str, Any]] = []
    for backend_index, mask in enumerate(normalised_masks):
        if not mask.any():
            raise _InconsistentSam3Output("empty mask in point-prompt output")
        detections.append(
            {
                "label": "point_prompt",
                "score": normalised_scores[backend_index],
                "bbox_xyxy": _mask_bbox_xyxy(mask, np=np),
                "mask": {
                    "format": "png",
                    "base64": _image_to_base64(
                        Image.fromarray(mask.astype(np.uint8) * 255, mode="L"),
                        fmt="png",
                    ),
                },
                "area_px": int(mask.sum()),
                "backend_index": backend_index,
            }
        )

    detections = _rank_sam3_detections(detections)
    artifacts: list[dict[str, Any]] = []
    for detection in detections:
        backend_index = detection["backend_index"]
        overlay = _build_point_candidate_overlay(
            image=image,
            mask=normalised_masks[backend_index],
            points=points,
            detection=detection,
        )
        artifacts.append(
            {
                "artifact_type": "candidate_overlay",
                "rank": detection["rank"],
                "backend_index": backend_index,
                "format": "png",
                "base64": _image_to_base64(overlay, fmt="png"),
            }
        )
    return detections, artifacts


def _mask_bbox_xyxy(mask: Any, *, np: Any) -> list[int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise _InconsistentSam3Output("cannot compute a box for an empty mask")
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]


def _build_point_candidate_overlay(
    *,
    image: Any,
    mask: Any,
    points: list[dict[str, Any]],
    detection: dict[str, Any],
) -> Any:
    import numpy as np
    from PIL import Image, ImageDraw

    overlay = np.asarray(image.convert("RGBA")).copy()
    mask_color = np.array([0, 128, 255, 110], dtype=np.uint8)
    overlay[mask] = (
        0.55 * overlay[mask].astype(np.float32)
        + 0.45 * mask_color.astype(np.float32)
    ).astype(np.uint8)
    rendered = Image.fromarray(overlay, mode="RGBA")
    draw = ImageDraw.Draw(rendered)
    radius = max(5, round(min(image.size) / 80))
    for point in points:
        x, y = round(point["x"]), round(point["y"])
        color = "#7CFC00" if point["label"] == 1 else "#FF3030"
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=color,
            width=max(2, radius // 3),
        )
        draw.line((x - radius, y, x + radius, y), fill=color, width=2)
        draw.line((x, y - radius, x, y + radius), fill=color, width=2)
    annotation = (
        f"rank={detection['rank']} score={detection['score']:.4f} "
        f"area={detection['area_px']}"
    )
    text_box = draw.textbbox((8, 8), annotation)
    draw.rectangle(
        (text_box[0] - 4, text_box[1] - 3, text_box[2] + 4, text_box[3] + 3),
        fill=(0, 0, 0, 190),
    )
    draw.text((8, 8), annotation, fill="white")
    return rendered


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
    import numpy as np

    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    if masks is None:
        return np.zeros((0, 0, 0), dtype=bool)
    if torch is not None and isinstance(masks, torch.Tensor):
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
    import numpy as np

    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    if boxes is None:
        return []
    if torch is not None and isinstance(boxes, torch.Tensor):
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
    import numpy as np

    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    if scores is None:
        return []
    if torch is not None and isinstance(scores, torch.Tensor):
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
    confidence_threshold: float | None,
) -> dict[str, Any]:
    version = backend_version or _git_commit(REPO_ROOT)
    if version and not version.startswith("sam3@"):
        version = f"sam3@{version}"
    metadata = {"backend_version": version}
    if confidence_threshold is not None:
        metadata["confidence_threshold"] = confidence_threshold
    return metadata


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
            "prompt_type": "text",
            "detection_count": 0,
            "detections": [],
            "artifacts": [],
            "reason": reason,
            "metadata": dict(metadata or {}),
        },
    }


def _point_failure_result(
    *,
    points: list[dict[str, Any]],
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
            "prompt_type": "points",
            "points": points,
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


class _Sam3ModelLoadFailed(Exception):
    """SAM3 model or processor construction failed."""


class _PointPromptError(ValueError):
    """A point prompt failed structural validation."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
