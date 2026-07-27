from __future__ import annotations

import base64
import io
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import tools.sam3_core as sam3_core
from tools.sam3_core import (
    _rank_sam3_detections,
    _run_point_inference,
    _run_text_inference,
    segment_image_points,
)


FIXTURE_IMAGE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "sam3" / "sam_test.png"
)


def _encoded_fixture() -> str:
    return base64.b64encode(FIXTURE_IMAGE.read_bytes()).decode("ascii")


def test_rank_sam3_detections_sorts_scores_and_preserves_backend_index() -> None:
    detections = [
        {"score": 0.42, "backend_index": 0},
        {"score": 0.91, "backend_index": 1},
        {"score": None, "backend_index": 2},
    ]

    ranked = _rank_sam3_detections(detections)

    assert [item["score"] for item in ranked] == [0.91, 0.42, None]
    assert [item["backend_index"] for item in ranked] == [1, 0, 2]
    assert [item["rank"] for item in ranked] == [0, 1, 2]


def test_point_prompt_requires_decodable_image() -> None:
    missing = segment_image_points(
        image_base64="",
        points=[{"x": 1, "y": 2, "label": 1}],
    )
    invalid = segment_image_points(
        image_base64="not-base64",
        points=[{"x": 1, "y": 2, "label": 1}],
    )

    assert missing["details"]["reason"] == "missing_image"
    assert invalid["details"]["reason"] == "image_decode_failed"


@pytest.mark.parametrize(
    ("points", "reason"),
    [
        (None, "missing_points"),
        ([], "missing_points"),
        ([{"x": 1, "y": 2, "label": 2}], "invalid_points"),
        ([{"x": 1, "y": 2, "label": 0}], "invalid_points"),
        ([{"x": float("nan"), "y": 2, "label": 1}], "invalid_points"),
        ([{"x": 1, "y": 2, "label": True}], "invalid_points"),
    ],
)
def test_point_prompt_rejects_invalid_structures(points, reason: str) -> None:
    result = segment_image_points(image_base64=_encoded_fixture(), points=points)

    assert result["success"] is False
    assert result["details"]["reason"] == reason
    assert result["details"]["detection_count"] == 0
    assert result["details"]["detections"] == []
    assert result["details"]["artifacts"] == []


def test_point_prompt_rejects_more_than_64_points() -> None:
    points = [{"x": 1, "y": 2, "label": 1}] * 65

    result = segment_image_points(image_base64=_encoded_fixture(), points=points)

    assert result["success"] is False
    assert result["details"]["reason"] == "invalid_points"


def test_point_prompt_rejects_out_of_bounds_point() -> None:
    width, height = Image.open(FIXTURE_IMAGE).size
    result = segment_image_points(
        image_base64=_encoded_fixture(),
        points=[{"x": width, "y": 10, "label": 1}],
    )

    assert result["success"] is False
    assert result["details"]["reason"] == "point_out_of_bounds"
    assert result["details"]["points"] == [
        {"x": float(width), "y": 10.0, "label": 1}
    ]
    assert result["details"]["metadata"]["image_size"] == [width, height]


def test_point_prompt_returns_three_ranked_candidates_and_overlays(monkeypatch) -> None:
    image = Image.open(FIXTURE_IMAGE)
    width, height = image.size
    masks = np.zeros((3, height, width), dtype=bool)
    masks[0, 10:20, 30:40] = True
    masks[1, 5:25, 20:45] = True
    masks[2, 7:10, 11:13] = True

    monkeypatch.setattr(
        sam3_core,
        "_run_point_inference",
        lambda **_kwargs: (masks, np.asarray([0.2, 0.9, 0.4])),
    )
    result = segment_image_points(
        image_base64=_encoded_fixture(),
        image_format="png",
        points=[
            {"x": 35, "y": 15, "label": 1},
            {"x": 100, "y": 80, "label": 0},
        ],
        backend_version="test",
    )

    assert result["success"] is True
    details = result["details"]
    assert details["prompt_type"] == "points"
    assert details["detection_count"] == 3
    assert details["ranking"] == "score_descending"
    assert [item["backend_index"] for item in details["detections"]] == [1, 2, 0]
    assert [item["rank"] for item in details["detections"]] == [0, 1, 2]
    assert details["detections"][0]["bbox_xyxy"] == [20, 5, 45, 25]
    assert details["detections"][0]["area_px"] == 500
    for detection in details["detections"]:
        assert detection["label"] == "point_prompt"
        assert Image.open(
            io.BytesIO(base64.b64decode(detection["mask"]["base64"]))
        ).size == (width, height)
    assert [item["rank"] for item in details["artifacts"]] == [0, 1, 2]
    assert all(
        item["artifact_type"] == "candidate_overlay"
        for item in details["artifacts"]
    )
    assert details["metadata"]["coordinate_units"] == "pixels"
    assert details["metadata"]["coordinate_origin"] == "top_left"
    assert details["metadata"]["point_count"] == 2
    assert details["metadata"]["foreground_point_count"] == 1
    assert details["metadata"]["background_point_count"] == 1
    assert details["metadata"]["candidate_count"] == 3
    assert details["metadata"]["backend_version"] == "sam3@test"


@pytest.mark.parametrize(
    ("mask_count", "scores"),
    [
        (2, [0.9, 0.8]),
        (3, [0.9, float("nan"), 0.7]),
    ],
)
def test_point_prompt_fails_atomically_on_inconsistent_candidates(
    monkeypatch,
    mask_count: int,
    scores: list[float],
) -> None:
    width, height = Image.open(FIXTURE_IMAGE).size
    masks = np.ones((mask_count, height, width), dtype=bool)
    monkeypatch.setattr(
        sam3_core,
        "_run_point_inference",
        lambda **_kwargs: (masks, np.asarray(scores)),
    )

    result = segment_image_points(
        image_base64=_encoded_fixture(),
        points=[{"x": 35, "y": 15, "label": 1}],
    )

    assert result["success"] is False
    assert result["details"]["reason"] == "inconsistent_detection_outputs"
    assert result["details"]["detection_count"] == 0
    assert result["details"]["detections"] == []
    assert result["details"]["artifacts"] == []


def test_point_prompt_fails_atomically_on_empty_candidate(monkeypatch) -> None:
    width, height = Image.open(FIXTURE_IMAGE).size
    masks = np.ones((3, height, width), dtype=bool)
    masks[1] = False
    monkeypatch.setattr(
        sam3_core,
        "_run_point_inference",
        lambda **_kwargs: (masks, np.asarray([0.9, 0.8, 0.7])),
    )

    result = segment_image_points(
        image_base64=_encoded_fixture(),
        points=[{"x": 35, "y": 15, "label": 1}],
    )

    assert result["success"] is False
    assert result["details"]["reason"] == "inconsistent_detection_outputs"
    assert result["details"]["detections"] == []


def test_point_prompt_structures_inference_exception(monkeypatch) -> None:
    def fail(**_kwargs):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(sam3_core, "_run_point_inference", fail)
    result = segment_image_points(
        image_base64=_encoded_fixture(),
        points=[{"x": 35, "y": 15, "label": 1}],
    )

    assert result["success"] is False
    assert result["details"]["reason"] == "model_inference_failed"
    assert result["details"]["detection_count"] == 0


class _FakeCuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def synchronize() -> None:
        return None

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakeTorch:
    Tensor = ()
    bfloat16 = object()

    def __init__(self) -> None:
        self.cuda = _FakeCuda()

    @staticmethod
    def autocast(**_kwargs):
        return nullcontext()


class _FakeModel:
    inst_interactive_predictor = None

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def predict_inst(self, *_args, **_kwargs):
        if self.fail:
            raise RuntimeError("inference failed")
        return (
            np.ones((3, 2, 2)),
            np.asarray([0.3, 0.2, 0.1]),
            np.zeros((3, 2, 2)),
        )


class _FakeProcessor:
    def __init__(self, *, fail: bool = False) -> None:
        self.model = _FakeModel(fail=fail)

    @staticmethod
    def set_image(_image):
        return {"state": True}

    def set_text_prompt(self, *, state, prompt):
        if self.model.fail:
            raise RuntimeError("text inference failed")
        return {"masks": [], "boxes": [], "scores": []}


@pytest.mark.parametrize("fail", [False, True])
def test_point_inference_empties_cuda_cache_on_success_and_failure(
    monkeypatch,
    fail: bool,
) -> None:
    torch = _FakeTorch()
    monkeypatch.setattr(
        sam3_core,
        "_get_processor",
        lambda **_kwargs: _FakeProcessor(fail=fail),
    )
    if fail:
        with pytest.raises(RuntimeError, match="inference failed"):
            _run_point_inference(
                image=object(),
                points=[{"x": 1.0, "y": 1.0, "label": 1}],
                np=np,
                torch=torch,
            )
    else:
        masks, scores = _run_point_inference(
            image=object(),
            points=[{"x": 1.0, "y": 1.0, "label": 1}],
            np=np,
            torch=torch,
        )
        assert masks.shape == (3, 2, 2)
        assert scores.tolist() == pytest.approx([0.3, 0.2, 0.1])
    assert torch.cuda.empty_cache_calls == 1


@pytest.mark.parametrize("fail", [False, True])
def test_text_inference_empties_cuda_cache(monkeypatch, fail: bool) -> None:
    torch = _FakeTorch()
    monkeypatch.setattr(
        sam3_core,
        "_get_processor",
        lambda **_kwargs: _FakeProcessor(fail=fail),
    )

    if fail:
        with pytest.raises(RuntimeError, match="text inference failed"):
            _run_text_inference(
                image=object(),
                prompt="shoe",
                confidence_threshold=0.5,
                torch=torch,
            )
    else:
        output = _run_text_inference(
            image=object(),
            prompt="shoe",
            confidence_threshold=0.5,
            torch=torch,
        )
        assert output == {"masks": [], "boxes": [], "scores": []}
    assert torch.cuda.empty_cache_calls == 1
