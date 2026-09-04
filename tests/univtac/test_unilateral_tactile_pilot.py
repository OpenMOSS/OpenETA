from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from sim.envs.univtac.unilateral_tactile_pilot import (
    CONDITION_ORDER,
    EXPECTED_CHANGE_SIDE,
    IMAGE_SOURCE_MAPPING,
    PILOT_PROMPT,
    SEEDS,
    build_source_mapping,
    parse_prediction,
    stage_condition,
    summarize_predictions,
    validate_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _prediction(
    *,
    changed_side: str,
    left_state: str,
    right_state: str,
    access: str = "available",
    overall: str = "bilateral",
) -> dict[str, str]:
    return {
        "tactile_image_access": access,
        "left_tactile_state": left_state,
        "right_tactile_state": right_state,
        "tactile_changed_side": changed_side,
        "overall_contact_state": overall,
        "visual_tactile_consistency": "conflicting",
        "evidence_summary": "One tactile image shows more marker change.",
        "uncertainty": "This is image change, not force.",
    }


def _source_fixture(tmp_path: Path, seed: int) -> tuple[dict, dict]:
    paths = {}
    for sensor_index, sensor in enumerate(("left_tactile", "right_tactile"), start=1):
        sensor_paths = {}
        for stage, value in (("baseline", 10), ("current", 10 + sensor_index * 20)):
            path = tmp_path / "pairs" / sensor / f"{stage}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((4, 6, 3), value, dtype=np.uint8)).save(path)
            sensor_paths[stage] = path.relative_to(tmp_path).as_posix()
        paths[sensor] = sensor_paths
    cameras = {}
    for name, value in (("head", 3), ("wrist", 4)):
        path = tmp_path / "pairs" / f"{name}.png"
        Image.fromarray(np.full((4, 6, 3), value, dtype=np.uint8)).save(path)
        cameras[name] = path.relative_to(tmp_path).as_posix()
    legacy = {
        "cameras": cameras,
        "sensors": {
            sensor: {
                "baseline_rgb_marker": paths[sensor]["baseline"],
                "current_rgb_marker": paths[sensor]["current"],
            }
            for sensor in paths
        },
    }
    structured = {
        "sensors": {
            "left_tactile": {
                "structured_metrics": {
                    "active_pixel_ratio": 0.2,
                    "mean_absolute_change": 20.0,
                    "p95_absolute_change": 20.0,
                    "salient_mass": 8.0,
                }
            },
            "right_tactile": {
                "structured_metrics": {
                    "active_pixel_ratio": 0.3,
                    "mean_absolute_change": 40.0,
                    "p95_absolute_change": 40.0,
                    "salient_mass": 18.0,
                }
            },
        }
    }
    return legacy, structured


def test_fixed_mapping_latin_square_prompt_and_zero_simulator_runner() -> None:
    path = REPO_ROOT / "configs/univtac/pull_out_key_unilateral_tactile_pilot.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert validate_config(payload)["seeds"] == list(SEEDS)
    assert payload["condition_order"] == {str(k): list(v) for k, v in CONDITION_ORDER.items()}
    assert payload["image_source_mapping"] == {
        str(k): dict(v) for k, v in IMAGE_SOURCE_MAPPING.items()
    }
    assert EXPECTED_CHANGE_SIDE == {1_000_000: "left", 1_000_001: "right", 1_000_002: "left"}
    assert "do not propose or execute an action" in PILOT_PROMPT.lower()
    runner = (REPO_ROOT / "scripts/univtac/run_unilateral_tactile_pilot.py").read_text()
    assert "run_pull_out_key_gate" not in runner and "AppLauncher" not in runner
    assert "play_once" not in runner and "check_success" not in runner
    assert '"simulator_invocation_count": 0' in runner


def test_source_mapping_uses_current_and_rgb_derived_zero_baseline(tmp_path: Path) -> None:
    legacy, structured = _source_fixture(tmp_path, 1_000_000)
    mapping = build_source_mapping(
        seed=1_000_000,
        source_root=tmp_path,
        legacy_reference=legacy,
        structured_reference=structured,
    )
    assert mapping["image_source_mapping"] == {
        "left_tactile": "current",
        "right_tactile": "baseline",
    }
    assert mapping["expected_image_change_side"] == "left"
    assert mapping["correct_structured_summary"]["left_tactile"][
        "mean_absolute_change"
    ] == 20.0
    assert set(mapping["correct_structured_summary"]["right_tactile"].values()) == {0.0}
    assert mapping["swapped_structured_summary"]["left_tactile"] == mapping[
        "correct_structured_summary"
    ]["right_tactile"]


def test_three_conditions_stage_identical_four_images_and_swap_only_text(
    tmp_path: Path,
) -> None:
    legacy, structured = _source_fixture(tmp_path, 1_000_000)
    mapping = build_source_mapping(
        seed=1_000_000,
        source_root=tmp_path,
        legacy_reference=legacy,
        structured_reference=structured,
    )
    manifests = {}
    arrays = {}
    for condition in CONDITION_ORDER[1_000_000]:
        episode = tmp_path / condition
        episode.mkdir()
        manifest = stage_condition(
            condition=condition,
            source_root=tmp_path,
            legacy_reference=legacy,
            source_mapping=mapping,
            episode_root=episode,
            simulator_root=tmp_path,
        )
        manifests[condition] = manifest
        arrays[condition] = [np.asarray(Image.open(episode / item["path"])) for item in manifest["images"]]
    assert all(len(manifest["images"]) == 4 for manifest in manifests.values())
    labels = [item["label"] for item in manifests["image_only"]["images"]]
    assert labels == [
        "camera/head/rgb",
        "camera/wrist/rgb",
        "tactile/left_tactile/rgb_marker",
        "tactile/right_tactile/rgb_marker",
    ]
    for index in range(4):
        assert np.array_equal(arrays["image_only"][index], arrays["correct_structured_guidance"][index])
        assert np.array_equal(arrays["image_only"][index], arrays["swapped_structured_guidance"][index])
    assert "model_visible_structured_tactile_summary" not in manifests["image_only"]
    correct = manifests["correct_structured_guidance"]["model_visible_structured_tactile_summary"]
    swapped = manifests["swapped_structured_guidance"]["model_visible_structured_tactile_summary"]
    assert swapped["left_tactile"] == correct["right_tactile"]
    assert swapped["right_tactile"] == correct["left_tactile"]


def test_parser_and_preregistered_helpful_image_grounded_signal() -> None:
    rows = []
    mappings = {}
    for seed in SEEDS:
        expected = EXPECTED_CHANGE_SIDE[seed]
        opposite = "right" if expected == "left" else "left"
        mappings[str(seed)] = {"image_source_mapping": IMAGE_SOURCE_MAPPING[seed]}
        aligned_states = (
            {"left_state": "clear_change", "right_state": "little_or_no_change"}
            if expected == "left"
            else {"left_state": "little_or_no_change", "right_state": "clear_change"}
        )
        wrong_states = {
            "left_state": aligned_states["right_state"],
            "right_state": aligned_states["left_state"],
        }
        for condition, prediction in (
            ("image_only", _prediction(changed_side=opposite, **wrong_states)),
            ("correct_structured_guidance", _prediction(changed_side=expected, **aligned_states)),
            ("swapped_structured_guidance", _prediction(changed_side=expected, **aligned_states)),
        ):
            text = json.dumps(prediction)
            rows.append({"seed": seed, "condition": condition, **parse_prediction(text)})
    summary = summarize_predictions(rows, mappings)
    assert summary["image_only_changed_side_accuracy"]["correct"] == 0
    assert summary["correct_guidance_changed_side_accuracy"]["correct"] == 3
    assert summary["structured_guidance_gain"] == 3
    assert summary["unilateral_exact_match"]["correct_structured_guidance"]["matched"] == 3
    assert summary["swapped_follow_image_rate"]["followed"] == 3
    assert summary["swapped_follow_text_rate"]["followed"] == 0
    assert summary["unilateral_signal"] == "helpful_and_image_grounded"
