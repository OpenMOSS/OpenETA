"""Pure helpers for the R0.9.17 unilateral tactile guidance-conflict pilot."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sim.envs.univtac.structured_tactile_pilot import structured_metrics

SEEDS = (1_000_000, 1_000_001, 1_000_002)
CONDITIONS = (
    "image_only",
    "correct_structured_guidance",
    "swapped_structured_guidance",
)
CONDITION_ORDER = {
    1_000_000: CONDITIONS,
    1_000_001: (
        "correct_structured_guidance",
        "swapped_structured_guidance",
        "image_only",
    ),
    1_000_002: (
        "swapped_structured_guidance",
        "image_only",
        "correct_structured_guidance",
    ),
}
IMAGE_SOURCE_MAPPING = {
    1_000_000: {"left_tactile": "current", "right_tactile": "baseline"},
    1_000_001: {"left_tactile": "baseline", "right_tactile": "current"},
    1_000_002: {"left_tactile": "current", "right_tactile": "baseline"},
}
EXPECTED_CHANGE_SIDE = {1_000_000: "left", 1_000_001: "right", 1_000_002: "left"}
MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "medium"
SENSORS = ("left_tactile", "right_tactile")
SCALAR_FIELDS = (
    "active_pixel_ratio",
    "mean_absolute_change",
    "p95_absolute_change",
    "salient_mass",
)

PILOT_PROMPT = """You are a read-only multimodal tactile observer for one UniVTAC Pull Out Key pre-action scene.

Call `observe` exactly once.

The head and wrist images show the external scene. The left and right tactile images are GelSight
rgb_marker views assigned to the corresponding fingertips.

Some observations may also include a structured image-change summary computed relative to each
sensor's calibration reference. These values describe visible rgb_marker change, not force,
pressure, grip stability, or task success.

Assess the tactile images themselves first, then use any structured summary as additional evidence.

Return exactly one JSON object:
{
  "tactile_image_access": "available|unavailable|uncertain",
  "left_tactile_state": "clear_change|little_or_no_change|uncertain",
  "right_tactile_state": "clear_change|little_or_no_change|uncertain",
  "tactile_changed_side": "left|right|both|neither|uncertain",
  "overall_contact_state": "left_only|right_only|bilateral|none|uncertain",
  "visual_tactile_consistency": "consistent|conflicting|uncertain",
  "evidence_summary": "...",
  "uncertainty": "..."
}

Do not propose or execute an action. Do not output robot control commands. Do not claim exact force,
friction, grip stability, object pose, or task success.
"""

_ENUMS = {
    "tactile_image_access": {"available", "unavailable", "uncertain"},
    "left_tactile_state": {"clear_change", "little_or_no_change", "uncertain"},
    "right_tactile_state": {"clear_change", "little_or_no_change", "uncertain"},
    "tactile_changed_side": {"left", "right", "both", "neither", "uncertain"},
    "overall_contact_state": {"left_only", "right_only", "bilateral", "none", "uncertain"},
    "visual_tactile_consistency": {"consistent", "conflicting", "uncertain"},
}
_TEXT_FIELDS = {"evidence_summary", "uncertainty"}


def validate_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_order = {str(seed): list(order) for seed, order in CONDITION_ORDER.items()}
    expected_mapping = {
        str(seed): dict(mapping) for seed, mapping in IMAGE_SOURCE_MAPPING.items()
    }
    if tuple(payload.get("seeds", ())) != SEEDS:
        raise ValueError(f"unilateral pilot seeds must be exactly {list(SEEDS)}")
    if payload.get("condition_order") != expected_order:
        raise ValueError("unilateral pilot condition order must use the fixed Latin square")
    if payload.get("image_source_mapping") != expected_mapping:
        raise ValueError("unilateral pilot image mapping is fixed")
    if payload.get("model") != MODEL or payload.get("reasoning_effort") != REASONING_EFFORT:
        raise ValueError("unilateral pilot model and reasoning effort are fixed")
    return dict(payload)


def parse_prediction(text: str) -> dict[str, str]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("invalid fenced JSON response")
        stripped = "\n".join(lines[1:-1])
        if stripped.lstrip().startswith("json"):
            stripped = stripped.lstrip()[4:].lstrip("\n")
    payload = json.loads(stripped)
    if not isinstance(payload, dict) or set(payload) != set(_ENUMS) | _TEXT_FIELDS:
        raise ValueError("prediction JSON fields do not match the unilateral pilot schema")
    for field, allowed in _ENUMS.items():
        if payload[field] not in allowed:
            raise ValueError(f"invalid {field}: {payload[field]!r}")
    for field in _TEXT_FIELDS:
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"{field} must be non-empty text")
    return {str(key): str(value) for key, value in payload.items()}


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return array


def _scalar_summary(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {field: round(float(metrics[field]), 3) for field in SCALAR_FIELDS}


def build_source_mapping(
    *,
    seed: int,
    source_root: Path,
    legacy_reference: Mapping[str, Any],
    structured_reference: Mapping[str, Any],
) -> dict[str, Any]:
    image_sources = IMAGE_SOURCE_MAPPING[seed]
    summaries: dict[str, dict[str, float]] = {}
    images: dict[str, str] = {}
    for sensor in SENSORS:
        source = image_sources[sensor]
        legacy_sensor = legacy_reference["sensors"][sensor]
        path_key = "current_rgb_marker" if source == "current" else "baseline_rgb_marker"
        images[sensor] = str(legacy_sensor[path_key])
        if source == "current":
            metrics = structured_reference["sensors"][sensor]["structured_metrics"]
        else:
            baseline = _read_rgb(source_root / str(legacy_sensor["baseline_rgb_marker"]))
            metrics = structured_metrics(baseline, baseline)
        summaries[sensor] = _scalar_summary(metrics)
    return {
        "seed": seed,
        "counterfactual_multimodal_input": True,
        "image_source_mapping": dict(image_sources),
        "image_paths": images,
        "expected_image_change_side": EXPECTED_CHANGE_SIDE[seed],
        "correct_structured_summary": summaries,
        "swapped_structured_summary": {
            "left_tactile": dict(summaries["right_tactile"]),
            "right_tactile": dict(summaries["left_tactile"]),
        },
        "structured_source_mapping": {
            "correct_structured_guidance": {
                "left_tactile": "left_tactile",
                "right_tactile": "right_tactile",
            },
            "swapped_structured_guidance": {
                "left_tactile": "right_tactile",
                "right_tactile": "left_tactile",
            },
        },
    }


def stage_condition(
    *,
    condition: str,
    source_root: Path,
    legacy_reference: Mapping[str, Any],
    source_mapping: Mapping[str, Any],
    episode_root: Path,
    simulator_root: Path,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown unilateral condition: {condition}")
    images_dir = episode_root / "images"
    images_dir.mkdir(parents=True, exist_ok=False)
    mappings = [
        ("camera/head/rgb", legacy_reference["cameras"]["head"], "head_rgb.png"),
        ("camera/wrist/rgb", legacy_reference["cameras"]["wrist"], "wrist_rgb.png"),
        (
            "tactile/left_tactile/rgb_marker",
            source_mapping["image_paths"]["left_tactile"],
            "left_tactile_rgb_marker.png",
        ),
        (
            "tactile/right_tactile/rgb_marker",
            source_mapping["image_paths"]["right_tactile"],
            "right_tactile_rgb_marker.png",
        ),
    ]
    records = []
    for label, relative_source, filename in mappings:
        source = source_root / str(relative_source)
        destination = images_dir / filename
        shutil.copyfile(source, destination)
        with Image.open(destination) as image:
            width, height = image.size
            dtype = str(np.asarray(image).dtype)
        records.append(
            {
                "label": label,
                "path": destination.relative_to(episode_root).as_posix(),
                "source_path": str(relative_source),
                "shape": [height, width, 3],
                "dtype": dtype,
                "media_type": "image/png",
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": "openeta.univtac.unilateral_tactile_condition.v1",
        "condition": condition,
        "simulator_root": str(simulator_root.resolve(strict=True)),
        "images": records,
        "counterfactual_multimodal_input": True,
        "image_source_mapping": dict(source_mapping["image_source_mapping"]),
        "expected_image_change_side": source_mapping["expected_image_change_side"],
    }
    if condition == "correct_structured_guidance":
        manifest["model_visible_structured_tactile_summary"] = source_mapping[
            "correct_structured_summary"
        ]
        manifest["structured_source_mapping"] = source_mapping[
            "structured_source_mapping"
        ][condition]
    elif condition == "swapped_structured_guidance":
        manifest["model_visible_structured_tactile_summary"] = source_mapping[
            "swapped_structured_summary"
        ]
        manifest["structured_source_mapping"] = source_mapping[
            "structured_source_mapping"
        ][condition]
    (episode_root / "condition.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _count_accuracy(
    predictions: Mapping[tuple[int, str], Mapping[str, Any]], condition: str
) -> dict[str, Any]:
    correct = uncertain = 0
    for seed in SEEDS:
        predicted = predictions[(seed, condition)]["tactile_changed_side"]
        correct += int(predicted == EXPECTED_CHANGE_SIDE[seed])
        uncertain += int(predicted == "uncertain")
    return {"correct": correct, "evaluated": 3, "uncertain": uncertain, "rate": correct / 3}


def summarize_predictions(
    predictions: list[Mapping[str, Any]], source_mappings: Mapping[str, Any]
) -> dict[str, Any]:
    by_pair = {(int(row["seed"]), str(row["condition"])): row for row in predictions}
    if any((seed, condition) not in by_pair for seed in SEEDS for condition in CONDITIONS):
        raise ValueError("all nine unilateral predictions are required for summary")
    image_accuracy = _count_accuracy(by_pair, "image_only")
    correct_accuracy = _count_accuracy(by_pair, "correct_structured_guidance")
    per_sensor: dict[str, Any] = {}
    exact_matches: dict[str, Any] = {}
    for condition in CONDITIONS:
        totals = {"aligned": 0, "uncertain": 0, "contradictory": 0}
        by_source = {
            "current": {"aligned": 0, "uncertain": 0, "contradictory": 0},
            "baseline": {"aligned": 0, "uncertain": 0, "contradictory": 0},
        }
        exact = 0
        for seed in SEEDS:
            row = by_pair[(seed, condition)]
            mapping = source_mappings[str(seed)]["image_source_mapping"]
            side_ok = {}
            for side in ("left", "right"):
                source = mapping[f"{side}_tactile"]
                target = "clear_change" if source == "current" else "little_or_no_change"
                predicted = row[f"{side}_tactile_state"]
                outcome = (
                    "uncertain"
                    if predicted == "uncertain"
                    else "aligned"
                    if predicted == target
                    else "contradictory"
                )
                totals[outcome] += 1
                by_source[source][outcome] += 1
                side_ok[side] = outcome == "aligned"
            exact += int(
                side_ok["left"]
                and side_ok["right"]
                and row["tactile_changed_side"] == EXPECTED_CHANGE_SIDE[seed]
            )
        per_sensor[condition] = {"total": totals, "by_source": by_source}
        exact_matches[condition] = {"matched": exact, "evaluated": 3, "rate": exact / 3}
    swapped_image = swapped_text = 0
    for seed in SEEDS:
        predicted = by_pair[(seed, "swapped_structured_guidance")]["tactile_changed_side"]
        expected = EXPECTED_CHANGE_SIDE[seed]
        opposite = "right" if expected == "left" else "left"
        swapped_image += int(predicted == expected)
        swapped_text += int(predicted == opposite)
    comparison_fields = (
        "left_tactile_state",
        "right_tactile_state",
        "tactile_changed_side",
        "overall_contact_state",
        "visual_tactile_consistency",
    )
    structured_changed = wrong_changed = 0
    for seed in SEEDS:
        image = by_pair[(seed, "image_only")]
        correct = by_pair[(seed, "correct_structured_guidance")]
        swapped = by_pair[(seed, "swapped_structured_guidance")]
        structured_changed += sum(image[field] != correct[field] for field in comparison_fields)
        wrong_changed += sum(correct[field] != swapped[field] for field in comparison_fields)
    override = override_evaluated = 0
    unavailable = []
    for row in predictions:
        if row["tactile_changed_side"] in {"left", "right"}:
            override_evaluated += 1
            override += int(row["overall_contact_state"] == "bilateral")
        if row["tactile_image_access"] == "unavailable":
            unavailable.append({"seed": int(row["seed"]), "condition": str(row["condition"])})
    gain = correct_accuracy["correct"] - image_accuracy["correct"]
    correct_helpful = correct_accuracy["correct"] > image_accuracy["correct"]
    image_grounded = swapped_image >= 2
    text_susceptible = swapped_text >= 2
    if correct_helpful and image_grounded:
        signal = "helpful_and_image_grounded"
    elif correct_helpful and text_susceptible:
        signal = "helpful_but_text_susceptible"
    elif correct_helpful:
        signal = "helpful_but_conflict_unresolved"
    elif image_accuracy["correct"] >= 2 and image_grounded:
        signal = "raw_image_grounded"
    elif text_susceptible:
        signal = "text_sensitive_without_accuracy_gain"
    else:
        signal = "no_detectable_unilateral_tactile_signal"
    return {
        "classification": "unilateral_tactile_guidance_conflict_pilot_completed",
        "image_only_changed_side_accuracy": image_accuracy,
        "correct_guidance_changed_side_accuracy": correct_accuracy,
        "structured_guidance_gain": gain,
        "per_sensor_source_state_alignment": per_sensor,
        "unilateral_exact_match": exact_matches,
        "swapped_follow_image_rate": {
            "followed": swapped_image,
            "evaluated": 3,
            "rate": swapped_image / 3,
        },
        "swapped_follow_text_rate": {
            "followed": swapped_text,
            "evaluated": 3,
            "rate": swapped_text / 3,
        },
        "structured_changes_prediction_rate": {
            "changed": structured_changed,
            "evaluated": 15,
            "rate": structured_changed / 15,
        },
        "wrong_guidance_changes_prediction_rate": {
            "changed": wrong_changed,
            "evaluated": 15,
            "rate": wrong_changed / 15,
        },
        "visual_override_rate": {
            "overrides": override,
            "evaluated": override_evaluated,
            "rate": override / override_evaluated if override_evaluated else None,
        },
        "tactile_image_unavailable_count": len(unavailable),
        "tactile_image_unavailable_trials": unavailable,
        "unilateral_signal": signal,
        "signal_flags": {
            "correct_helpful": correct_helpful,
            "image_grounded": image_grounded,
            "text_susceptible": text_susceptible,
        },
    }


__all__ = [
    "CONDITIONS",
    "CONDITION_ORDER",
    "EXPECTED_CHANGE_SIDE",
    "IMAGE_SOURCE_MAPPING",
    "MODEL",
    "PILOT_PROMPT",
    "REASONING_EFFORT",
    "SEEDS",
    "build_source_mapping",
    "parse_prediction",
    "stage_condition",
    "summarize_predictions",
    "validate_config",
]
