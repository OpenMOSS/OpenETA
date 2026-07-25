"""MolmoPoint image-grounding backend for the OpenETA MCP server."""

from __future__ import annotations

import base64
import binascii
import gc
import io
import math
import numbers
import time
from pathlib import Path
from typing import Any, Callable


TOOL_NAME = "molmopoint"
BACKEND_NAME = "molmopoint_mcp"
DEFAULT_MODEL_ID = "allenai/MolmoPoint-8B"
DEFAULT_MODEL_REVISION = "188130f961c8e0888a34e11121a1423c461a01ba"
DEFAULT_MAX_NEW_TOKENS = 200
MIN_IMAGE_COUNT = 1
MAX_IMAGE_COUNT = 4
MAX_IMAGE_SIDE = 8192
MAX_IMAGE_PIXELS = 32_000_000
MAX_TOTAL_IMAGE_PIXELS = 64_000_000
MAX_PROMPT_CHARS = 1024
COORDINATE_CONVENTION = {
    "origin": "top_left",
    "x_direction": "right",
    "y_direction": "down",
    "units": "pixels",
}


class MolmoPointInputError(ValueError):
    def __init__(
        self,
        reason: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.metadata = metadata or {}


class MolmoPointOutputError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class MolmoPointBackend:
    """Run deterministic MolmoPoint inference while unloading weights per call."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        model_id: str = DEFAULT_MODEL_ID,
        model_revision: str = DEFAULT_MODEL_REVISION,
        processor_loader: Callable[[Path], Any] | None = None,
        model_loader: Callable[[Path], Any] | None = None,
        runtime_loader: Callable[[], tuple[Any, Any]] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_id = model_id
        self.model_revision = model_revision
        self._processor_loader = processor_loader or _load_processor
        self._model_loader = model_loader or _load_model
        self._runtime_loader = runtime_loader or _load_runtime
        self._processor: Any | None = None

    def point_images(
        self,
        *,
        images: list[dict[str, Any]] | None,
        prompt: str | None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        metadata = self._metadata_base()
        try:
            parsed_prompt = validate_prompt(prompt)
            decoded_images, image_metadata = decode_images(images)
            metadata["images"] = image_metadata
        except MolmoPointInputError as exc:
            metadata.update(exc.metadata)
            return failure_result(
                reason=exc.reason,
                metadata=_with_total_seconds(metadata, started),
                model_id=self.model_id,
            )

        try:
            processor = self._get_processor()
        except Exception as exc:  # noqa: BLE001 - third-party load boundary.
            return failure_result(
                reason="model_load_failed",
                metadata=_with_total_seconds(
                    {**metadata, "error_type": type(exc).__name__},
                    started,
                ),
                model_id=self.model_id,
            )

        model = None
        try:
            torch, _Image = self._runtime_loader()
            load_started = time.perf_counter()
            try:
                model = self._model_loader(self.model_path)
                model.eval()
            except Exception as exc:  # noqa: BLE001 - classify backend failures.
                reason = "out_of_memory" if _is_cuda_oom(exc, torch) else "model_load_failed"
                return failure_result(
                    reason=reason,
                    metadata=_with_total_seconds(
                        {**metadata, "error_type": type(exc).__name__},
                        started,
                    ),
                    model_id=self.model_id,
                )
            metadata["model_load_seconds"] = round(time.perf_counter() - load_started, 4)

            inference_started = time.perf_counter()
            try:
                generated_text, raw_points = _run_inference(
                    torch=torch,
                    model=model,
                    processor=processor,
                    images=decoded_images,
                    prompt=parsed_prompt,
                )
                del generated_text
            except Exception as exc:  # noqa: BLE001 - classify backend failures.
                reason = "out_of_memory" if _is_cuda_oom(exc, torch) else "inference_failed"
                return failure_result(
                    reason=reason,
                    metadata=_with_total_seconds(
                        {**metadata, "error_type": type(exc).__name__},
                        started,
                    ),
                    model_id=self.model_id,
                )
            metadata["inference_seconds"] = round(
                time.perf_counter() - inference_started,
                4,
            )

            try:
                points = normalise_points(raw_points, image_metadata=image_metadata)
            except MolmoPointOutputError as exc:
                return failure_result(
                    reason=exc.reason,
                    metadata=_with_total_seconds(metadata, started),
                    model_id=self.model_id,
                )

            return {
                "success": True,
                "content": "MolmoPoint image pointing completed.",
                "details": {
                    "tool": TOOL_NAME,
                    "backend": BACKEND_NAME,
                    "model": self.model_id,
                    "point_count": len(points),
                    "points": points,
                    "coordinate_convention": dict(COORDINATE_CONVENTION),
                    "artifacts": [],
                    "metadata": _with_total_seconds(metadata, started),
                },
            }
        finally:
            if model is not None:
                del model
            _release_runtime_memory()

    def _get_processor(self) -> Any:
        if self._processor is None:
            self._processor = self._processor_loader(self.model_path)
        return self._processor

    def _metadata_base(self) -> dict[str, Any]:
        return {"model_revision": self.model_revision}


def validate_prompt(prompt: Any) -> str:
    if not isinstance(prompt, str):
        raise MolmoPointInputError("invalid_prompt")
    parsed = prompt.strip()
    if not parsed or len(parsed) > MAX_PROMPT_CHARS:
        raise MolmoPointInputError("invalid_prompt")
    return parsed


def decode_images(
    payloads: Any,
    *,
    max_image_side: int = MAX_IMAGE_SIDE,
    max_image_pixels: int = MAX_IMAGE_PIXELS,
    max_total_image_pixels: int = MAX_TOTAL_IMAGE_PIXELS,
) -> tuple[list[Any], list[dict[str, Any]]]:
    if not isinstance(payloads, list) or not MIN_IMAGE_COUNT <= len(payloads) <= MAX_IMAGE_COUNT:
        raise MolmoPointInputError("invalid_image_count")

    Image = _load_image_dependency()
    decoded: list[Any] = []
    metadata: list[dict[str, Any]] = []
    total_pixels = 0
    for image_index, payload in enumerate(payloads):
        try:
            image, image_format, source_mode = _open_image_payload(payload, Image=Image)
        except MolmoPointInputError as exc:
            raise MolmoPointInputError(
                exc.reason,
                metadata={"image_index": image_index},
            ) from exc

        width, height = image.size
        pixels = width * height
        total_pixels += pixels
        if (
            width <= 0
            or height <= 0
            or max(width, height) > max_image_side
            or pixels > max_image_pixels
            or total_pixels > max_total_image_pixels
        ):
            image.close()
            raise MolmoPointInputError(
                "image_too_large",
                metadata={
                    "image_index": image_index,
                    "image_width": width,
                    "image_height": height,
                    "total_image_pixels": total_pixels,
                },
            )
        try:
            image.load()
            decoded_image = image.convert("RGB")
        except Exception as exc:  # noqa: BLE001 - Pillow decoding boundary.
            raise MolmoPointInputError(
                "decode_failed",
                metadata={"image_index": image_index},
            ) from exc
        finally:
            image.close()
        decoded.append(decoded_image)
        metadata.append(
            {
                "image_index": image_index,
                "format": image_format,
                "width": width,
                "height": height,
                "source_image_mode": source_mode,
                "model_image_mode": "RGB",
            }
        )
    return decoded, metadata


def _open_image_payload(payload: Any, *, Image: Any) -> tuple[Any, str, str]:
    if not isinstance(payload, dict):
        raise MolmoPointInputError("invalid_image_payload")
    declared_format = payload.get("format")
    encoded = payload.get("base64")
    if not isinstance(declared_format, str) or not isinstance(encoded, str) or not encoded:
        raise MolmoPointInputError("invalid_image_payload")
    normalised_format = declared_format.strip().lower()
    if normalised_format == "jpg":
        normalised_format = "jpeg"
    if normalised_format not in {"png", "jpeg"}:
        raise MolmoPointInputError("unsupported_image_format")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise MolmoPointInputError("invalid_image_payload") from exc
    image = None
    try:
        image = Image.open(io.BytesIO(raw))
        detected_format = str(image.format or "").lower()
        if detected_format == "jpg":
            detected_format = "jpeg"
        if detected_format != normalised_format:
            raise MolmoPointInputError("decode_failed")
        if int(getattr(image, "n_frames", 1)) != 1:
            raise MolmoPointInputError("decode_failed")
        orientation = image.getexif().get(274, 1)
        if orientation not in (None, 1):
            raise MolmoPointInputError("unsupported_image_orientation")
        source_mode = str(image.mode)
        return image, normalised_format, source_mode
    except MolmoPointInputError:
        if image is not None:
            image.close()
        raise
    except Exception as exc:  # noqa: BLE001 - Pillow decoding boundary.
        if image is not None:
            image.close()
        raise MolmoPointInputError("decode_failed") from exc


def normalise_points(
    raw_points: Any,
    *,
    image_metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        iterable = raw_points.tolist() if hasattr(raw_points, "tolist") else list(raw_points)
    except Exception as exc:  # noqa: BLE001 - third-party output boundary.
        raise MolmoPointOutputError("invalid_model_output") from exc

    points: list[dict[str, Any]] = []
    for point_index, raw_point in enumerate(iterable):
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 4:
            raise MolmoPointOutputError("invalid_model_output")
        _object_id, image_index, pixel_x, pixel_y = raw_point
        if isinstance(image_index, bool) or not isinstance(image_index, numbers.Integral):
            raise MolmoPointOutputError("invalid_model_output")
        parsed_image_index = int(image_index)
        if not 0 <= parsed_image_index < len(image_metadata):
            raise MolmoPointOutputError("invalid_model_output")
        if (
            isinstance(pixel_x, bool)
            or isinstance(pixel_y, bool)
            or not isinstance(pixel_x, numbers.Real)
            or not isinstance(pixel_y, numbers.Real)
        ):
            raise MolmoPointOutputError("invalid_model_output")
        x = float(pixel_x)
        y = float(pixel_y)
        if not math.isfinite(x) or not math.isfinite(y):
            raise MolmoPointOutputError("invalid_model_output")
        image_info = image_metadata[parsed_image_index]
        if not (0.0 <= x < image_info["width"] and 0.0 <= y < image_info["height"]):
            raise MolmoPointOutputError("point_out_of_bounds")
        points.append(
            {
                "id": f"point_{point_index:03d}",
                "image_index": parsed_image_index,
                "pixel_x": x,
                "pixel_y": y,
            }
        )
    return points


def failure_result(
    *,
    reason: str,
    metadata: dict[str, Any],
    model_id: str = DEFAULT_MODEL_ID,
) -> dict[str, Any]:
    return {
        "success": False,
        "content": f"MolmoPoint image pointing failed: {reason}.",
        "details": {
            "tool": TOOL_NAME,
            "backend": BACKEND_NAME,
            "model": model_id,
            "point_count": 0,
            "points": [],
            "coordinate_convention": dict(COORDINATE_CONVENTION),
            "artifacts": [],
            "reason": reason,
            "metadata": metadata,
        },
    }


def _run_inference(
    *,
    torch: Any,
    model: Any,
    processor: Any,
    images: list[Any],
    prompt: str,
) -> tuple[str, Any]:
    content = [{"type": "text", "text": prompt}]
    content.extend({"type": "image", "image": image} for image in images)
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
        return_pointing_metadata=True,
    )
    pointing_metadata = inputs.pop("metadata")
    inputs = {
        key: value.to("cuda:0") if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.generate(
            **inputs,
            logits_processor=model.build_logit_processor_from_inputs(inputs),
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            do_sample=False,
        )
    torch.cuda.synchronize()
    generated_tokens = output[:, inputs["input_ids"].size(1) :]
    generated_text = processor.post_process_image_text_to_text(
        generated_tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    raw_points = model.extract_image_points(
        generated_text,
        pointing_metadata["token_pooling"],
        pointing_metadata["subpatch_mapping"],
        pointing_metadata["image_sizes"],
    )
    return generated_text, raw_points


def _load_processor(model_path: Path) -> Any:
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
        local_files_only=True,
        use_fast=False,
    )


def _load_model(model_path: Path) -> Any:
    import torch
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
    )


def _load_runtime() -> tuple[Any, Any]:
    import torch
    from PIL import Image

    return torch, Image


def _load_image_dependency() -> Any:
    from PIL import Image

    return Image


def _release_runtime_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_cuda_oom(exc: Exception, torch: Any) -> bool:
    oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    return bool(oom_type is not None and isinstance(exc, oom_type))


def _with_total_seconds(metadata: dict[str, Any], started: float) -> dict[str, Any]:
    return {**metadata, "total_seconds": round(time.perf_counter() - started, 4)}
