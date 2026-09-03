from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from sim.envs.univtac.tactile_difference_pilot import (
    ACTIVE_THRESHOLD,
    CONDITION_ORDER,
    DIFFERENCE_GAIN,
    PILOT_PROMPT,
    SEEDS,
    difference_visual,
    image_difference_metrics,
    parse_prediction,
    stage_condition,
    summarize_predictions,
    validate_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _prediction(region: str = "left", stronger: str = "left") -> dict[str, str]:
    return {
        "left_change": "increased",
        "right_change": "increased",
        "left_change_region": region,
        "right_change_region": region,
        "stronger_change_side": stronger,
        "evidence_summary": "Localized marker-image changes are visible.",
        "uncertainty": "The images do not provide exact force.",
    }


def test_config_and_prompt_fix_protocol() -> None:
    path = REPO_ROOT / "configs/univtac/pull_out_key_tactile_difference_pilot.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert validate_config(payload)["seeds"] == list(SEEDS)
    assert payload["condition_order"] == {str(k): list(v) for k, v in CONDITION_ORDER.items()}
    assert DIFFERENCE_GAIN == 4 and ACTIVE_THRESHOLD == 12
    assert "condition" not in PILOT_PROMPT.lower()
    assert "do not propose or execute an action" in PILOT_PROMPT.lower()


def test_fixed_difference_gain_has_no_normalization() -> None:
    baseline = np.array([[[10, 20, 30], [100, 100, 100]]], dtype=np.uint8)
    current = np.array([[[13, 10, 40], [200, 0, 100]]], dtype=np.uint8)
    raw, visual = difference_visual(baseline, current)
    assert raw.tolist() == [[[3, 10, 10], [100, 100, 0]]]
    assert visual.tolist() == [[[12, 40, 40], [255, 255, 0]]]


def test_metrics_use_threshold_and_weighted_centroid() -> None:
    raw = np.zeros((3, 6, 3), dtype=np.int16)
    raw[:, 4:, :] = 24
    metrics = image_difference_metrics(raw)
    assert metrics["active_pixel_ratio"] == pytest.approx(1 / 3)
    assert metrics["centroid_region"] == "right"
    assert metrics["p95_absolute_difference"] == 24
    inactive = image_difference_metrics(np.zeros_like(raw))
    assert inactive["centroid_region"] == "inactive"
    assert inactive["active_pixel_weighted_centroid_xy"] is None


def test_three_condition_mapping_stages_six_images_and_swaps_only_difference(
    tmp_path: Path,
) -> None:
    colors = {
        "head": 1,
        "wrist": 2,
        "left_base": 3,
        "left_current": 4,
        "left_diff": 5,
        "right_base": 6,
        "right_current": 7,
        "right_diff": 8,
    }
    paths = {}
    for name, value in colors.items():
        path = tmp_path / "source" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((4, 5, 3), value, dtype=np.uint8)).save(path)
        paths[name] = path.relative_to(tmp_path).as_posix()
    pair = {
        "cameras": {"head": paths["head"], "wrist": paths["wrist"]},
        "sensors": {
            "left_tactile": {
                "baseline_rgb_marker": paths["left_base"],
                "current_rgb_marker": paths["left_current"],
                "difference_visual": paths["left_diff"],
            },
            "right_tactile": {
                "baseline_rgb_marker": paths["right_base"],
                "current_rgb_marker": paths["right_current"],
                "difference_visual": paths["right_diff"],
            },
        },
    }
    episode = tmp_path / "episode"
    episode.mkdir()
    manifest = stage_condition(
        condition="swapped_difference",
        pair=pair,
        episode_root=episode,
        output_root=tmp_path,
        simulator_root=tmp_path,
    )
    assert len(manifest["images"]) == 6
    assert [
        np.asarray(Image.open(episode / item["path"]))[0, 0, 0] for item in manifest["images"]
    ] == [
        1,
        2,
        4,
        8,
        7,
        5,
    ]


def test_prediction_parser_and_positive_signal() -> None:
    assert parse_prediction(json.dumps(_prediction())) == _prediction()
    references = {}
    rows = []
    for seed in SEEDS:
        references[str(seed)] = {
            "reference_stronger_side": "left",
            "press_depth_delta_stronger_side_secondary": "left",
            "sensors": {
                "left_tactile": {"metrics": {"centroid_region": "left"}},
                "right_tactile": {"metrics": {"centroid_region": "right"}},
            },
        }
        raw = _prediction("center", "right")
        raw["right_change_region"] = "center"
        explicit = _prediction("left", "left")
        explicit["right_change_region"] = "right"
        swapped = _prediction("right", "right")
        swapped["right_change_region"] = "left"
        for condition, payload in (
            ("raw_pair", raw),
            ("explicit_difference", explicit),
            ("swapped_difference", swapped),
        ):
            rows.append({"seed": seed, "condition": condition, **payload})
    summary = summarize_predictions(rows, references)
    assert summary["difference_region_gain"] == 1.0
    assert summary["swapped_difference_equivariance"]["equivariant"] == 6
    assert summary["difference_signal"] == "positive_difference_signal"


def test_runner_and_probe_have_zero_action_and_ordered_pair_capture() -> None:
    runner = (REPO_ROOT / "scripts/univtac/run_tactile_difference_pilot.py").read_text()
    probe = (REPO_ROOT / "scripts/univtac/probe_pull_out_key_seed.py").read_text()
    assert "for seed in SEEDS" in runner and "for condition in CONDITION_ORDER[seed]" in runner
    assert "play_once" not in runner and "check_success" not in runner
    assert "no_parallel_runtime" not in runner and "hashlib" not in runner
    assert probe.index('stages.record("pre_grasp_baseline_capture", "enter")') < probe.index(
        "return original_pre_move"
    )
    assert probe.index("task.reset(seed=gate_config") < probe.index(
        'stages.record("get_observations", "enter")'
    )
