from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from tools import molmopoint_core as core


def _image_payload(
    *,
    size: tuple[int, int] = (32, 24),
    mode: str = "RGB",
    fmt: str = "PNG",
    declared_format: str | None = None,
    exif_orientation: int | None = None,
) -> dict[str, str]:
    color = (10, 20, 30, 128) if mode == "RGBA" else (10 if mode == "L" else (10, 20, 30))
    image = Image.new(mode, size, color)
    buffer = io.BytesIO()
    kwargs = {}
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[274] = exif_orientation
        kwargs["exif"] = exif
    image.save(buffer, format=fmt, **kwargs)
    return {
        "format": declared_format or fmt.lower(),
        "base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


@pytest.mark.parametrize("prompt", [None, 1, "", "   ", "x" * 1025])
def test_validate_prompt_rejects_invalid_values(prompt) -> None:
    with pytest.raises(core.MolmoPointInputError, match="invalid_prompt"):
        core.validate_prompt(prompt)


def test_validate_prompt_trims_only_outer_whitespace() -> None:
    assert core.validate_prompt("  Point to any can.  ") == "Point to any can."


@pytest.mark.parametrize("payloads", [None, {}, [], [None] * 5])
def test_decode_images_rejects_invalid_count(payloads) -> None:
    with pytest.raises(core.MolmoPointInputError, match="invalid_image_count"):
        core.decode_images(payloads)


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({}, "invalid_image_payload"),
        ({"format": "webp", "base64": "eA=="}, "unsupported_image_format"),
        ({"format": "png", "base64": "not-base64"}, "invalid_image_payload"),
        (_image_payload(fmt="JPEG", declared_format="png"), "decode_failed"),
        (_image_payload(fmt="JPEG", exif_orientation=6), "unsupported_image_orientation"),
    ],
)
def test_decode_images_returns_indexed_structured_failures(payload, reason) -> None:
    with pytest.raises(core.MolmoPointInputError) as caught:
        core.decode_images([payload])
    assert caught.value.reason == reason
    assert caught.value.metadata == {"image_index": 0}


@pytest.mark.parametrize(
    ("payload", "source_mode", "image_format"),
    [
        (_image_payload(mode="RGB", fmt="PNG"), "RGB", "png"),
        (_image_payload(mode="RGBA", fmt="PNG"), "RGBA", "png"),
        (_image_payload(mode="L", fmt="PNG"), "L", "png"),
        (_image_payload(mode="RGB", fmt="JPEG", declared_format="jpg"), "RGB", "jpeg"),
    ],
)
def test_decode_images_converts_supported_images_to_rgb(payload, source_mode, image_format) -> None:
    images, metadata = core.decode_images([payload])
    assert images[0].mode == "RGB"
    assert metadata[0] == {
        "image_index": 0,
        "format": image_format,
        "width": 32,
        "height": 24,
        "source_image_mode": source_mode,
        "model_image_mode": "RGB",
    }


def test_decode_images_enforces_per_image_and_aggregate_limits() -> None:
    payload = _image_payload(size=(10, 10))
    with pytest.raises(core.MolmoPointInputError) as per_image:
        core.decode_images([payload], max_image_pixels=99)
    assert per_image.value.reason == "image_too_large"

    with pytest.raises(core.MolmoPointInputError) as aggregate:
        core.decode_images([payload, payload], max_total_image_pixels=199)
    assert aggregate.value.reason == "image_too_large"
    assert aggregate.value.metadata["image_index"] == 1


def test_normalise_points_preserves_order_and_image_index_but_hides_object_id() -> None:
    metadata = [
        {"width": 100, "height": 80},
        {"width": 200, "height": 160},
    ]
    points = core.normalise_points(
        [[91, 1, 30.25, 40.5], [12, 0, 7.0, 8.0]],
        image_metadata=metadata,
    )
    assert points == [
        {"id": "point_000", "image_index": 1, "pixel_x": 30.25, "pixel_y": 40.5},
        {"id": "point_001", "image_index": 0, "pixel_x": 7.0, "pixel_y": 8.0},
    ]


@pytest.mark.parametrize(
    ("raw_points", "reason"),
    [
        ([[1, 2, 10.0, 10.0]], "invalid_model_output"),
        ([[1, 0, float("nan"), 10.0]], "invalid_model_output"),
        ([[1, 0, 100.0, 10.0]], "point_out_of_bounds"),
        ([[1, 0, 10.0]], "invalid_model_output"),
    ],
)
def test_normalise_points_rejects_invalid_outputs_atomically(raw_points, reason) -> None:
    with pytest.raises(core.MolmoPointOutputError) as caught:
        core.normalise_points(raw_points, image_metadata=[{"width": 100, "height": 80}])
    assert caught.value.reason == reason


class _Model:
    def __init__(self) -> None:
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True


class _CudaOOM(Exception):
    pass


def _fake_torch() -> SimpleNamespace:
    return SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=_CudaOOM))


def _backend(monkeypatch: pytest.MonkeyPatch, *, raw_points) -> tuple[core.MolmoPointBackend, list[str]]:
    releases: list[str] = []
    monkeypatch.setattr(core, "_release_runtime_memory", lambda: releases.append("release"))
    monkeypatch.setattr(
        core,
        "_run_inference",
        lambda **_kwargs: ("<generated>", raw_points),
    )
    backend = core.MolmoPointBackend(
        model_path="/model",
        processor_loader=lambda _path: object(),
        model_loader=lambda _path: _Model(),
        runtime_loader=lambda: (_fake_torch(), Image),
    )
    return backend, releases


def test_backend_returns_zero_points_as_success_and_releases_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, releases = _backend(monkeypatch, raw_points=[])
    result = backend.point_images(images=[_image_payload()], prompt="Find a can.")
    assert result["success"] is True
    assert result["details"]["point_count"] == 0
    assert result["details"]["points"] == []
    assert result["details"]["artifacts"] == []
    assert releases == ["release"]


def test_backend_rejects_one_invalid_point_without_partial_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, releases = _backend(
        monkeypatch,
        raw_points=[[1, 0, 5.0, 5.0], [2, 0, 1000.0, 5.0]],
    )
    result = backend.point_images(images=[_image_payload()], prompt="Find cans.")
    assert result["success"] is False
    assert result["details"]["reason"] == "point_out_of_bounds"
    assert result["details"]["point_count"] == 0
    assert result["details"]["points"] == []
    assert releases == ["release"]


def test_backend_classifies_cuda_oom_and_releases_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    releases: list[str] = []
    monkeypatch.setattr(core, "_release_runtime_memory", lambda: releases.append("release"))
    backend = core.MolmoPointBackend(
        model_path="/model",
        processor_loader=lambda _path: object(),
        model_loader=lambda _path: (_ for _ in ()).throw(_CudaOOM("oom")),
        runtime_loader=lambda: (_fake_torch(), Image),
    )
    result = backend.point_images(images=[_image_payload()], prompt="Find a can.")
    assert result["success"] is False
    assert result["details"]["reason"] == "out_of_memory"
    assert releases == ["release"]


def test_backend_classifies_inference_failure_and_releases_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    releases: list[str] = []
    monkeypatch.setattr(core, "_release_runtime_memory", lambda: releases.append("release"))
    monkeypatch.setattr(
        core,
        "_run_inference",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("generation failed")),
    )
    backend = core.MolmoPointBackend(
        model_path="/model",
        processor_loader=lambda _path: object(),
        model_loader=lambda _path: _Model(),
        runtime_loader=lambda: (_fake_torch(), Image),
    )
    result = backend.point_images(images=[_image_payload()], prompt="Find a can.")
    assert result["success"] is False
    assert result["details"]["reason"] == "inference_failed"
    assert releases == ["release"]


def test_backend_caches_processor_but_loads_model_each_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor_loads: list[str] = []
    model_loads: list[str] = []
    monkeypatch.setattr(core, "_release_runtime_memory", lambda: None)
    monkeypatch.setattr(core, "_run_inference", lambda **_kwargs: ("", []))
    backend = core.MolmoPointBackend(
        model_path="/model",
        processor_loader=lambda _path: processor_loads.append("processor") or object(),
        model_loader=lambda _path: model_loads.append("model") or _Model(),
        runtime_loader=lambda: (_fake_torch(), Image),
    )
    for _ in range(2):
        assert backend.point_images(images=[_image_payload()], prompt="Find a can.")["success"]
    assert processor_loads == ["processor"]
    assert model_loads == ["model", "model"]


def test_failure_result_never_leaks_points_or_artifacts() -> None:
    result = core.failure_result(reason="invalid_prompt", metadata={"image_index": 1})
    assert result["success"] is False
    assert result["details"]["points"] == []
    assert result["details"]["artifacts"] == []
    assert result["details"]["reason"] == "invalid_prompt"
