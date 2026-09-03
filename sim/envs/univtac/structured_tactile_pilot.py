"""Pure helpers for the R0.9.16 RGB-marker structured-guidance pilot."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sim.envs.univtac.tactile_difference_pilot import parse_prediction

SEEDS = (1_000_000, 1_000_001, 1_000_002)
CONDITIONS = (
    "difference_only",
    "correct_structured_guidance",
    "swapped_structured_guidance",
)
CONDITION_ORDER = {
    1_000_000: CONDITIONS,
    1_000_001: (
        "correct_structured_guidance",
        "swapped_structured_guidance",
        "difference_only",
    ),
    1_000_002: (
        "swapped_structured_guidance",
        "difference_only",
        "correct_structured_guidance",
    ),
}
MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "medium"
ACTIVE_THRESHOLD = 12.0
SALIENCY_PERCENTILE = 90.0
SENSORS = ("left_tactile", "right_tactile")

PILOT_PROMPT = """You are a read-only tactile state observer for one UniVTAC Pull Out Key pre-action state.

Call `observe` exactly once.

You will receive external camera images and temporal tactile difference images. The observation may
also contain a structured summary derived from the same baseline/current rgb_marker images.

When a structured summary is present:

- normalized_horizontal_saliency is ordered as [left, center, right];
- larger values mean that more of the salient image change lies in that horizontal region;
- active_pixel_ratio and change magnitudes describe visible image change, not force or pressure.

Use all available evidence conservatively.

Return exactly one JSON object:
{
  "left_change": "increased|little_change|uncertain",
  "right_change": "increased|little_change|uncertain",
  "left_change_region": "left|center|right|diffuse|uncertain",
  "right_change_region": "left|center|right|diffuse|uncertain",
  "stronger_change_side": "left|right|balanced|uncertain",
  "evidence_summary": "...",
  "uncertainty": "..."
}

Do not propose or execute an action. Do not claim exact force, friction, grip stability, object pose,
or task success.
"""


def validate_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_order = {str(seed): list(order) for seed, order in CONDITION_ORDER.items()}
    if tuple(payload.get("seeds", ())) != SEEDS:
        raise ValueError(f"structured pilot seeds must be exactly {list(SEEDS)}")
    if payload.get("condition_order") != expected_order:
        raise ValueError("structured pilot condition order must use the fixed Latin square")
    if payload.get("model") != MODEL or payload.get("reasoning_effort") != REASONING_EFFORT:
        raise ValueError("structured pilot model and reasoning effort are fixed")
    if float(payload.get("active_threshold", -1)) != ACTIVE_THRESHOLD:
        raise ValueError("active threshold must be exactly 12")
    if float(payload.get("saliency_percentile", -1)) != SALIENCY_PERCENTILE:
        raise ValueError("saliency percentile must be exactly 90")
    if payload.get("source_round") != "R0.9.15":
        raise ValueError("structured pilot must reuse the R0.9.15 capture")
    return dict(payload)


def structured_metrics(baseline: np.ndarray, current: np.ndarray) -> dict[str, Any]:
    before = np.asarray(baseline)
    after = np.asarray(current)
    if before.shape != after.shape or before.ndim != 3 or before.shape[-1] != 3:
        raise ValueError("baseline/current rgb_marker arrays must have the same HxWx3 shape")
    if before.dtype != np.uint8 or after.dtype != np.uint8:
        raise ValueError("baseline/current rgb_marker arrays must be uint8")
    magnitude = np.abs(after.astype(np.int16) - before.astype(np.int16)).mean(axis=2)
    percentile_value = float(np.percentile(magnitude, SALIENCY_PERCENTILE))
    threshold = max(ACTIVE_THRESHOLD, percentile_value)
    saliency = np.maximum(magnitude - threshold, 0.0)
    energy = [float(part.sum()) for part in np.array_split(saliency, 3, axis=1)]
    total = float(sum(energy))
    normalized = [value / total for value in energy] if total else [0.0, 0.0, 0.0]
    if total == 0:
        region = "uncertain"
    else:
        ranked = sorted(normalized, reverse=True)
        if ranked[0] >= 0.45 and ranked[0] - ranked[1] >= 0.10:
            region = ("left", "center", "right")[int(np.argmax(normalized))]
        else:
            region = "diffuse"
    return {
        "normalized_horizontal_saliency": normalized,
        "saliency_energy_by_third": energy,
        "active_pixel_ratio": float((magnitude > ACTIVE_THRESHOLD).mean()),
        "mean_absolute_change": float(magnitude.mean()),
        "p95_absolute_change": float(np.percentile(magnitude, 95)),
        "salient_mass": float(saliency.mean()),
        "active_threshold": ACTIVE_THRESHOLD,
        "saliency_percentile": SALIENCY_PERCENTILE,
        "saliency_threshold": threshold,
        "reference_region": region,
    }


def reference_stronger_side(left_mass: float, right_mass: float) -> str:
    scale = max(left_mass, right_mass)
    if scale <= 0 or abs(left_mass - right_mass) / scale < 0.10:
        return "balanced"
    return "left" if left_mass > right_mass else "right"


def model_visible_descriptor(metrics_by_sensor: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    descriptor: dict[str, Any] = {"vector_order": ["left", "center", "right"]}
    for sensor in SENSORS:
        metrics = metrics_by_sensor[sensor]
        descriptor[sensor] = {
            "normalized_horizontal_saliency": [
                round(float(value), 3)
                for value in metrics["normalized_horizontal_saliency"]
            ],
            "active_pixel_ratio": round(float(metrics["active_pixel_ratio"]), 3),
            "mean_absolute_change": round(float(metrics["mean_absolute_change"]), 3),
            "p95_absolute_change": round(float(metrics["p95_absolute_change"]), 3),
            "salient_mass": round(float(metrics["salient_mass"]), 3),
        }
    return descriptor


def build_structured_reference(
    *, seed: int, source_root: Path, legacy_reference: Mapping[str, Any]
) -> dict[str, Any]:
    sensors: dict[str, Any] = {}
    model_metrics: dict[str, Mapping[str, Any]] = {}
    for sensor in SENSORS:
        legacy_sensor = legacy_reference["sensors"][sensor]
        baseline_path = source_root / legacy_sensor["baseline_rgb_marker"]
        current_path = source_root / legacy_sensor["current_rgb_marker"]
        with Image.open(baseline_path) as image:
            baseline = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(current_path) as image:
            current = np.asarray(image.convert("RGB"), dtype=np.uint8)
        metrics = structured_metrics(baseline, current)
        model_metrics[sensor] = metrics
        old = legacy_sensor["metrics"]
        sensors[sensor] = {
            "baseline_rgb_marker": legacy_sensor["baseline_rgb_marker"],
            "current_rgb_marker": legacy_sensor["current_rgb_marker"],
            "difference_visual": legacy_sensor["difference_visual"],
            "structured_metrics": metrics,
            "old_global_centroid_xy": old["active_pixel_weighted_centroid_xy"],
            "old_global_centroid_region": old["centroid_region"],
            "old_and_new_region_agree": old["centroid_region"]
            == metrics["reference_region"],
            "press_depth_delta_secondary": legacy_sensor.get(
                "press_depth_delta_secondary"
            ),
        }
    left_mass = float(model_metrics["left_tactile"]["salient_mass"])
    right_mass = float(model_metrics["right_tactile"]["salient_mass"])
    return {
        "seed": seed,
        "sensors": sensors,
        "model_visible_descriptor": model_visible_descriptor(model_metrics),
        "reference_stronger_side": reference_stronger_side(left_mass, right_mass),
        "old_global_reference_stronger_side": legacy_reference.get(
            "reference_stronger_side"
        ),
        "press_depth_delta_stronger_side_secondary": legacy_reference.get(
            "press_depth_delta_stronger_side_secondary"
        ),
        "metric_construct_note": (
            "global_active_mass_centroid_does_not_measure_dominant_local_saliency"
        ),
    }


def stage_condition(
    *,
    condition: str,
    source_root: Path,
    legacy_reference: Mapping[str, Any],
    structured_reference: Mapping[str, Any],
    episode_root: Path,
    simulator_root: Path,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown structured condition: {condition}")
    images_dir = episode_root / "images"
    images_dir.mkdir(parents=True, exist_ok=False)
    mappings = [
        ("camera/head/rgb", legacy_reference["cameras"]["head"], "head_rgb.png"),
        ("camera/wrist/rgb", legacy_reference["cameras"]["wrist"], "wrist_rgb.png"),
    ]
    for sensor in SENSORS:
        sensor_source = legacy_reference["sensors"][sensor]
        mappings.extend(
            [
                (
                    f"tactile/{sensor}/current_rgb_marker",
                    sensor_source["current_rgb_marker"],
                    f"{sensor}_current.png",
                ),
                (
                    f"tactile/{sensor}/difference",
                    sensor_source["difference_visual"],
                    f"{sensor}_difference.png",
                ),
            ]
        )
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
    descriptor = structured_reference["model_visible_descriptor"]
    source_mapping = {"left_tactile": "left_tactile", "right_tactile": "right_tactile"}
    if condition == "swapped_structured_guidance":
        descriptor = {
            "vector_order": list(descriptor["vector_order"]),
            "left_tactile": dict(descriptor["right_tactile"]),
            "right_tactile": dict(descriptor["left_tactile"]),
        }
        source_mapping = {"left_tactile": "right_tactile", "right_tactile": "left_tactile"}
    manifest: dict[str, Any] = {
        "schema_version": "openeta.univtac.structured_tactile_condition.v1",
        "condition": condition,
        "simulator_root": str(simulator_root.resolve(strict=True)),
        "images": records,
        "structured_source_mapping": source_mapping,
    }
    if condition != "difference_only":
        manifest["model_visible_structured_tactile_summary"] = descriptor
    (episode_root / "condition.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def summarize_predictions(
    predictions: list[Mapping[str, Any]], references: Mapping[str, Any]
) -> dict[str, Any]:
    by_pair = {(int(row["seed"]), str(row["condition"])): row for row in predictions}

    def region_alignment(condition: str) -> dict[str, Any]:
        aligned = evaluated = uncertain = 0
        for seed in SEEDS:
            row = by_pair.get((seed, condition))
            reference = references.get(str(seed))
            if not row or not reference:
                continue
            for side in ("left", "right"):
                prediction = row[f"{side}_change_region"]
                target = reference["sensors"][f"{side}_tactile"]["structured_metrics"][
                    "reference_region"
                ]
                if prediction == "uncertain" or target == "uncertain":
                    uncertain += 1
                else:
                    evaluated += 1
                    aligned += int(prediction == target)
        return {
            "aligned": aligned,
            "evaluated": evaluated,
            "uncertain": uncertain,
            "rate": aligned / evaluated if evaluated else None,
        }

    def stronger_alignment(condition: str) -> dict[str, Any]:
        aligned = evaluated = 0
        for seed in SEEDS:
            row = by_pair.get((seed, condition))
            reference = references.get(str(seed))
            if row and reference and row["stronger_change_side"] != "uncertain":
                evaluated += 1
                aligned += int(
                    row["stronger_change_side"] == reference["reference_stronger_side"]
                )
        return {
            "aligned": aligned,
            "evaluated": evaluated,
            "rate": aligned / evaluated if evaluated else None,
        }

    difference_alignment = region_alignment("difference_only")
    correct_alignment = region_alignment("correct_structured_guidance")
    correct_stronger = stronger_alignment("correct_structured_guidance")
    swapped_follow = swapped_evaluated = 0
    image_grounded = image_evaluated = 0
    changed = changed_evaluated = 0
    wrong_changed = wrong_changed_evaluated = 0
    for seed in SEEDS:
        difference = by_pair.get((seed, "difference_only"))
        correct = by_pair.get((seed, "correct_structured_guidance"))
        swapped = by_pair.get((seed, "swapped_structured_guidance"))
        reference = references.get(str(seed))
        if not difference or not correct or not swapped or not reference:
            continue
        for side in ("left", "right"):
            other = "right" if side == "left" else "left"
            image_target = reference["sensors"][f"{side}_tactile"]["structured_metrics"][
                "reference_region"
            ]
            swapped_target = reference["sensors"][f"{other}_tactile"][
                "structured_metrics"
            ]["reference_region"]
            swapped_prediction = swapped[f"{side}_change_region"]
            if swapped_target in {"left", "right"}:
                swapped_evaluated += 1
                swapped_follow += int(swapped_prediction == swapped_target)
            if image_target in {"left", "right"}:
                image_evaluated += 1
                image_grounded += int(swapped_prediction == image_target)
            changed_evaluated += 1
            changed += int(
                difference[f"{side}_change_region"] != correct[f"{side}_change_region"]
            )
            wrong_changed_evaluated += 1
            wrong_changed += int(
                correct[f"{side}_change_region"] != swapped[f"{side}_change_region"]
            )
        changed_evaluated += 1
        changed += int(
            difference["stronger_change_side"] != correct["stronger_change_side"]
        )
        wrong_changed_evaluated += 1
        wrong_changed += int(
            correct["stronger_change_side"] != swapped["stronger_change_side"]
        )
    gain = correct_alignment["aligned"] - difference_alignment["aligned"]
    correct_gain = gain >= 2
    wrong_follow = swapped_follow >= 3
    grounded = image_grounded >= 3
    if correct_gain and grounded:
        signal = "helpful_and_image_grounded"
    elif correct_gain and wrong_follow:
        signal = "helpful_but_text_susceptible"
    elif not correct_gain and wrong_follow:
        signal = "text_sensitive_without_accuracy_gain"
    else:
        signal = "no_detectable_structured_guidance_signal"
    return {
        "classification": "tactile_structured_guidance_pilot_completed",
        "difference_only_region_alignment": difference_alignment,
        "correct_structure_region_alignment": correct_alignment,
        "structured_guidance_region_gain": gain,
        "correct_structure_stronger_side_alignment": correct_stronger,
        "swapped_structure_follow_rate": {
            "followed": swapped_follow,
            "evaluated": swapped_evaluated,
            "rate": swapped_follow / swapped_evaluated if swapped_evaluated else None,
        },
        "image_grounding_under_conflict_rate": {
            "grounded": image_grounded,
            "evaluated": image_evaluated,
            "rate": image_grounded / image_evaluated if image_evaluated else None,
        },
        "structured_changes_prediction_rate": {
            "changed": changed,
            "evaluated": changed_evaluated,
            "rate": changed / changed_evaluated if changed_evaluated else None,
        },
        "wrong_guidance_changes_prediction_rate": {
            "changed": wrong_changed,
            "evaluated": wrong_changed_evaluated,
            "rate": wrong_changed / wrong_changed_evaluated
            if wrong_changed_evaluated
            else None,
        },
        "structured_signal": signal,
        "signal_flags": {
            "correct_gain": correct_gain,
            "wrong_follow": wrong_follow,
            "image_grounded": grounded,
        },
    }


__all__ = [
    "ACTIVE_THRESHOLD",
    "CONDITIONS",
    "CONDITION_ORDER",
    "MODEL",
    "PILOT_PROMPT",
    "REASONING_EFFORT",
    "SALIENCY_PERCENTILE",
    "SEEDS",
    "build_structured_reference",
    "model_visible_descriptor",
    "parse_prediction",
    "reference_stronger_side",
    "stage_condition",
    "structured_metrics",
    "summarize_predictions",
    "validate_config",
]
