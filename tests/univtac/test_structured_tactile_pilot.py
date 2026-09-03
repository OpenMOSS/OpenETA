from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from sim.envs.univtac.structured_tactile_pilot import (
    ACTIVE_THRESHOLD,
    CONDITION_ORDER,
    PILOT_PROMPT,
    SALIENCY_PERCENTILE,
    SEEDS,
    model_visible_descriptor,
    reference_stronger_side,
    stage_condition,
    structured_metrics,
    summarize_predictions,
    validate_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _prediction(left: str, right: str, stronger: str = "balanced") -> dict[str, str]:
    return {
        "left_change": "increased",
        "right_change": "increased",
        "left_change_region": left,
        "right_change_region": right,
        "stronger_change_side": stronger,
        "evidence_summary": "Visible marker-image changes support this judgment.",
        "uncertainty": "Image change is not a force measurement.",
    }


def _reference(left: str, right: str, stronger: str = "balanced") -> dict:
    return {
        "reference_stronger_side": stronger,
        "sensors": {
            "left_tactile": {"structured_metrics": {"reference_region": left}},
            "right_tactile": {"structured_metrics": {"reference_region": right}},
        },
    }


def test_config_and_prompt_fix_zero_simulator_protocol() -> None:
    path = REPO_ROOT / "configs/univtac/pull_out_key_structured_tactile_pilot.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert validate_config(payload)["seeds"] == list(SEEDS)
    assert payload["condition_order"] == {str(k): list(v) for k, v in CONDITION_ORDER.items()}
    assert ACTIVE_THRESHOLD == 12 and SALIENCY_PERCENTILE == 90
    assert "do not propose or execute an action" in PILOT_PROMPT.lower()
    runner = (REPO_ROOT / "scripts/univtac/run_structured_tactile_pilot.py").read_text()
    assert "run_pull_out_key_gate" not in runner
    assert "play_once" not in runner and "check_success" not in runner
    assert '"simulator_invocation_count": 0' in runner


def test_saliency_formula_threshold_and_horizontal_thirds() -> None:
    baseline = np.zeros((1, 18, 3), dtype=np.uint8)
    current = np.repeat(np.arange(18, dtype=np.uint8)[None, :, None], 3, axis=2)
    metrics = structured_metrics(baseline, current)
    assert metrics["saliency_threshold"] == pytest.approx(15.3)
    assert metrics["active_pixel_ratio"] == pytest.approx(5 / 18)
    assert metrics["saliency_energy_by_third"][:2] == [0.0, 0.0]
    assert metrics["normalized_horizontal_saliency"] == pytest.approx([0.0, 0.0, 1.0])
    assert metrics["reference_region"] == "right"
    low = np.zeros((1, 30, 3), dtype=np.uint8)
    low[:, -1, :] = 13
    low_metrics = structured_metrics(baseline=np.zeros_like(low), current=low)
    assert low_metrics["saliency_threshold"] == 12
    assert low_metrics["salient_mass"] == pytest.approx(1 / 30)


def test_model_descriptor_rounds_only_rgb_derived_metrics() -> None:
    metrics = {
        sensor: {
            "normalized_horizontal_saliency": [0.12345, 0.5, 0.37655],
            "active_pixel_ratio": 0.23456,
            "mean_absolute_change": 18.9876,
            "p95_absolute_change": 103.3333,
            "salient_mass": 1.81234,
        }
        for sensor in ("left_tactile", "right_tactile")
    }
    descriptor = model_visible_descriptor(metrics)
    assert descriptor["vector_order"] == ["left", "center", "right"]
    assert descriptor["left_tactile"]["normalized_horizontal_saliency"] == [
        0.123,
        0.5,
        0.377,
    ]
    serialized = json.dumps(descriptor)
    assert not any(word in serialized for word in ("reference", "press_depth", "contact"))


def test_all_conditions_stage_identical_images_and_only_swap_text(tmp_path: Path) -> None:
    paths: dict[str, str] = {}
    for index, name in enumerate(("head", "wrist", "lc", "ld", "rc", "rd"), start=1):
        path = tmp_path / "source" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((4, 5, 3), index, dtype=np.uint8)).save(path)
        paths[name] = path.relative_to(tmp_path).as_posix()
    legacy = {
        "cameras": {"head": paths["head"], "wrist": paths["wrist"]},
        "sensors": {
            "left_tactile": {
                "current_rgb_marker": paths["lc"],
                "difference_visual": paths["ld"],
            },
            "right_tactile": {
                "current_rgb_marker": paths["rc"],
                "difference_visual": paths["rd"],
            },
        },
    }
    descriptor = {
        "vector_order": ["left", "center", "right"],
        "left_tactile": {"marker": "left"},
        "right_tactile": {"marker": "right"},
    }
    reference = {"model_visible_descriptor": descriptor}
    manifests = {}
    for condition in CONDITION_ORDER[SEEDS[0]]:
        episode = tmp_path / condition
        episode.mkdir()
        manifests[condition] = stage_condition(
            condition=condition,
            source_root=tmp_path,
            legacy_reference=legacy,
            structured_reference=reference,
            episode_root=episode,
            simulator_root=tmp_path,
        )
    image_records = [
        [(item["label"], item["source_path"]) for item in manifest["images"]]
        for manifest in manifests.values()
    ]
    assert image_records[0] == image_records[1] == image_records[2]
    assert "model_visible_structured_tactile_summary" not in manifests["difference_only"]
    correct = manifests["correct_structured_guidance"][
        "model_visible_structured_tactile_summary"
    ]
    swapped = manifests["swapped_structured_guidance"][
        "model_visible_structured_tactile_summary"
    ]
    assert correct["left_tactile"]["marker"] == "left"
    assert swapped["left_tactile"]["marker"] == "right"
    assert swapped["right_tactile"]["marker"] == "left"


def test_stronger_side_and_preregistered_signal() -> None:
    assert reference_stronger_side(1.0, 0.95) == "balanced"
    assert reference_stronger_side(2.0, 1.0) == "left"
    references = {str(seed): _reference("left", "right", "left") for seed in SEEDS}
    rows = []
    for seed in SEEDS:
        for condition, prediction in (
            ("difference_only", _prediction("center", "center", "balanced")),
            ("correct_structured_guidance", _prediction("left", "right", "left")),
            ("swapped_structured_guidance", _prediction("right", "left", "right")),
        ):
            rows.append({"seed": seed, "condition": condition, **prediction})
    summary = summarize_predictions(rows, references)
    assert summary["structured_guidance_region_gain"] == 6
    assert summary["swapped_structure_follow_rate"]["followed"] == 6
    assert summary["structured_signal"] == "helpful_but_text_susceptible"
