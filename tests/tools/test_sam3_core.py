from __future__ import annotations

import pytest

from tools.sam3_core import _box_xyxy_to_normalized_cxcywh, _rank_sam3_detections


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


def test_box_prompt_is_clipped_and_normalized_to_sam3_cxcywh() -> None:
    assert _box_xyxy_to_normalized_cxcywh(
        [-10, 20, 60, 120], image_size=(100, 200)
    ) == pytest.approx([0.3, 0.35, 0.6, 0.5])


@pytest.mark.parametrize(
    "box",
    ([1, 2, 1, 4], [1, 2, 3], [float("nan"), 0, 1, 1]),
)
def test_box_prompt_rejects_invalid_geometry(box) -> None:
    with pytest.raises(ValueError):
        _box_xyxy_to_normalized_cxcywh(box, image_size=(100, 100))
