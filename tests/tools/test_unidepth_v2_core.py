from __future__ import annotations

import base64
import io
from contextlib import nullcontext

import numpy as np
from PIL import Image

from tools.unidepth_v2_core import (
    CONFIDENCE_SEMANTICS,
    DEFAULT_MODEL_ID,
    UniDepthV2Backend,
)


def _rgb_payload() -> dict[str, str]:
    image = Image.fromarray(np.full((3, 4, 3), 127, dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {
        "format": "png",
        "base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def _decode_npy(value: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(value)), allow_pickle=False)


class _FakeTensor:
    def permute(self, *_args):
        return self

    def contiguous(self):
        return self

    def to(self, _device):
        return self


class _FakeTorch:
    float32 = "float32"

    @staticmethod
    def from_numpy(_array):
        return _FakeTensor()

    @staticmethod
    def tensor(_value, **_kwargs):
        return _FakeTensor()

    @staticmethod
    def inference_mode():
        return nullcontext()


class _FakeModel:
    resolution_level = 0

    def infer(self, _rgb, _camera):
        return {
            "depth": np.full((1, 1, 3, 4), 1.25, dtype=np.float32),
            "confidence": np.linspace(0.1, 0.9, 12, dtype=np.float32).reshape(
                1, 1, 3, 4
            ),
        }


def test_backend_returns_compact_metric_npy_payload(monkeypatch) -> None:
    backend = UniDepthV2Backend()
    monkeypatch.setattr(
        backend,
        "_get_model",
        lambda: (_FakeModel(), _FakeTorch(), "cpu"),
    )

    result = backend.estimate_depth(
        rgb=_rgb_payload(),
        intrinsics={"fx": 100.0, "fy": 101.0, "cx": 2.0, "cy": 1.5},
        camera_id="wrist",
        calibration_profile_id="cal-v3",
    )

    assert result["success"] is True
    details = result["details"]
    assert details["model"] == DEFAULT_MODEL_ID
    assert details["confidence_semantics"] == CONFIDENCE_SEMANTICS
    assert np.allclose(_decode_npy(details["depth_npy_base64"]), 1.25)
    assert _decode_npy(details["confidence_npy_base64"]).shape == (3, 4)
    assert details["metadata"]["used_calibrated_camera"] is True


def test_backend_rejects_non_pinhole_before_model_load() -> None:
    backend = UniDepthV2Backend()

    result = backend.estimate_depth(
        rgb=_rgb_payload(),
        intrinsics={"fx": 100.0, "fy": 100.0, "cx": 2.0, "cy": 1.0},
        camera_model="fisheye",
    )

    assert result["success"] is False
    assert result["details"]["reason"] == "unsupported_camera_model"


def test_backend_rejects_invalid_resolution_level() -> None:
    backend = UniDepthV2Backend()

    result = backend.estimate_depth(
        rgb=_rgb_payload(),
        intrinsics={"fx": 100.0, "fy": 100.0, "cx": 2.0, "cy": 1.0},
        resolution_level=10,
    )

    assert result["success"] is False
    assert result["details"]["reason"] == "invalid_resolution_level"
