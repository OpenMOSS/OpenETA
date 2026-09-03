"""Pure capture, staging, and analysis helpers for the R0.9.15 pilot."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SEEDS = (1_000_000, 1_000_001, 1_000_002)
CONDITIONS = ("raw_pair", "explicit_difference", "swapped_difference")
CONDITION_ORDER = {
    1_000_000: CONDITIONS,
    1_000_001: ("explicit_difference", "swapped_difference", "raw_pair"),
    1_000_002: ("swapped_difference", "raw_pair", "explicit_difference"),
}
MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "medium"
DIFFERENCE_GAIN = 4
ACTIVE_THRESHOLD = 12.0
SENSORS = ("left_tactile", "right_tactile")

PILOT_PROMPT = """You are a read-only tactile state observer for one UniVTAC Pull Out Key pre-action state.

Call `observe` exactly once.

The tool returns two external camera images and, for each fingertip, either:

- a baseline/current pair of GelSight rgb_marker images, or
- a current rgb_marker image and a precomputed temporal difference map.

A difference map represents fixed-scale absolute pixel change from the pre-grasp baseline to the
post-pre-move state. Brighter regions mean larger visible marker-image change.

Report only what is supported by the images.

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

Do not propose or execute an action. Do not claim exact force, friction, grip stability, object
pose, or task success.
"""

_ENUMS = {
    "left_change": {"increased", "little_change", "uncertain"},
    "right_change": {"increased", "little_change", "uncertain"},
    "left_change_region": {"left", "center", "right", "diffuse", "uncertain"},
    "right_change_region": {"left", "center", "right", "diffuse", "uncertain"},
    "stronger_change_side": {"left", "right", "balanced", "uncertain"},
}
_TEXT_FIELDS = {"evidence_summary", "uncertainty"}


def validate_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_order = {str(seed): list(order) for seed, order in CONDITION_ORDER.items()}
    if tuple(payload.get("seeds", ())) != SEEDS:
        raise ValueError(f"difference pilot seeds must be exactly {list(SEEDS)}")
    if payload.get("condition_order") != expected_order:
        raise ValueError("difference pilot condition order must use the fixed Latin square")
    if payload.get("model") != MODEL or payload.get("reasoning_effort") != REASONING_EFFORT:
        raise ValueError("difference pilot model and reasoning effort are fixed")
    if payload.get("difference_gain") != DIFFERENCE_GAIN:
        raise ValueError("difference gain must be exactly 4")
    if float(payload.get("active_threshold", -1)) != ACTIVE_THRESHOLD:
        raise ValueError("active threshold must be exactly 12")
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
        raise ValueError("prediction JSON fields do not match the difference pilot schema")
    for field, allowed in _ENUMS.items():
        if payload[field] not in allowed:
            raise ValueError(f"invalid {field}: {payload[field]!r}")
    for field in _TEXT_FIELDS:
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"{field} must be non-empty text")
    return {str(key): str(value) for key, value in payload.items()}


def difference_visual(baseline: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    before = np.asarray(baseline)
    after = np.asarray(current)
    if before.shape != after.shape or before.ndim != 3 or before.shape[-1] != 3:
        raise ValueError("baseline/current rgb_marker arrays must have the same HxWx3 shape")
    if before.dtype != np.uint8 or after.dtype != np.uint8:
        raise ValueError("baseline/current rgb_marker arrays must be uint8")
    raw = np.abs(after.astype(np.int16) - before.astype(np.int16))
    visual = np.clip(raw * DIFFERENCE_GAIN, 0, 255).astype(np.uint8)
    return raw, visual


def image_difference_metrics(raw_difference: np.ndarray) -> dict[str, Any]:
    raw = np.asarray(raw_difference)
    if raw.ndim != 3 or raw.shape[-1] != 3:
        raise ValueError("raw difference must be HxWx3")
    magnitude = raw.astype(np.float64).mean(axis=-1)
    active = magnitude > ACTIVE_THRESHOLD
    active_ratio = float(active.mean())
    weights = np.where(active, magnitude, 0.0)
    if float(weights.sum()) == 0:
        centroid = None
        region = "inactive"
    else:
        yy, xx = np.indices(magnitude.shape)
        centroid_x = float((xx * weights).sum() / weights.sum())
        centroid_y = float((yy * weights).sum() / weights.sum())
        centroid = [centroid_x, centroid_y]
        normalized_x = centroid_x / max(magnitude.shape[1] - 1, 1)
        region = "left" if normalized_x < 1 / 3 else "right" if normalized_x > 2 / 3 else "center"
    return {
        "mean_absolute_difference": float(magnitude.mean()),
        "p95_absolute_difference": float(np.percentile(magnitude, 95)),
        "active_pixel_ratio": active_ratio,
        "active_threshold": ACTIVE_THRESHOLD,
        "active_pixel_weighted_centroid_xy": centroid,
        "centroid_region": region,
        "rgb_aggregation": "mean",
    }


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected RGB uint8 image: {path}")
    return array


def _write_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def build_pair_artifacts(
    *,
    seed: int,
    baseline_snapshot: Mapping[str, Any],
    current_snapshot: Mapping[str, Any],
    simulator_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    pair_root = output_root / "pairs" / f"seed_{seed}"
    pair_root.mkdir(parents=True, exist_ok=False)
    cameras: dict[str, str] = {}
    for name in ("head", "wrist"):
        source = (
            simulator_root / current_snapshot["operator_visible"]["cameras"][name]["rgb"]["path"]
        )
        destination = pair_root / "camera" / f"{name}_rgb.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        cameras[name] = destination.relative_to(output_root).as_posix()
    sensors: dict[str, Any] = {}
    for sensor in SENSORS:
        before_path = (
            simulator_root
            / baseline_snapshot["operator_visible"]["tactile"][sensor]["rgb_marker"]["path"]
        )
        after_path = (
            simulator_root
            / current_snapshot["operator_visible"]["tactile"][sensor]["rgb_marker"]["path"]
        )
        before = _read_rgb(before_path)
        after = _read_rgb(after_path)
        raw, visual = difference_visual(before, after)
        sensor_root = pair_root / sensor
        sensor_root.mkdir()
        baseline_out = sensor_root / "baseline_rgb_marker.png"
        current_out = sensor_root / "current_rgb_marker.png"
        difference_npy = sensor_root / "raw_difference.npy"
        difference_png = sensor_root / "difference_visual.png"
        shutil.copyfile(before_path, baseline_out)
        shutil.copyfile(after_path, current_out)
        np.save(difference_npy, raw, allow_pickle=False)
        _write_rgb(difference_png, visual)
        sensors[sensor] = {
            "baseline_rgb_marker": baseline_out.relative_to(output_root).as_posix(),
            "current_rgb_marker": current_out.relative_to(output_root).as_posix(),
            "raw_difference_npy": difference_npy.relative_to(output_root).as_posix(),
            "difference_visual": difference_png.relative_to(output_root).as_posix(),
            "difference_gain": DIFFERENCE_GAIN,
        }
    return {
        "seed": seed,
        "baseline_stage": "pre_grasp_baseline",
        "current_stage": "post_pre_move",
        "difference_formula": "clip(abs(current.astype(int16)-baseline.astype(int16))*4,0,255)",
        "difference_gain": DIFFERENCE_GAIN,
        "active_threshold": ACTIVE_THRESHOLD,
        "cameras": cameras,
        "sensors": sensors,
    }


def add_press_depth_reference(
    *,
    pair: Mapping[str, Any],
    baseline_snapshot: Mapping[str, Any],
    current_snapshot: Mapping[str, Any],
    simulator_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    enriched = json.loads(json.dumps(pair))
    deltas: dict[str, float] = {}
    difference_means: dict[str, float] = {}
    for sensor in SENSORS:
        raw_difference = np.load(
            output_root / enriched["sensors"][sensor]["raw_difference_npy"], allow_pickle=False
        )
        metrics = image_difference_metrics(raw_difference)
        enriched["sensors"][sensor]["metrics"] = metrics
        difference_means[sensor] = float(metrics["mean_absolute_difference"])
        before = np.load(
            simulator_root
            / baseline_snapshot["host_only"]["tactile"][sensor]["press_depth"]["path"],
            allow_pickle=False,
        )
        after = np.load(
            simulator_root
            / current_snapshot["host_only"]["tactile"][sensor]["press_depth"]["path"],
            allow_pickle=False,
        )
        baseline_mean = float(np.asarray(before).mean())
        current_mean = float(np.asarray(after).mean())
        delta = current_mean - baseline_mean
        enriched["sensors"][sensor]["press_depth_delta_secondary"] = {
            "baseline_mean": baseline_mean,
            "current_mean": current_mean,
            "mean_delta": delta,
        }
        deltas[sensor] = delta
    left_difference = difference_means["left_tactile"]
    right_difference = difference_means["right_tactile"]
    difference_scale = max(left_difference, right_difference, 1e-12)
    enriched["reference_stronger_side"] = (
        "balanced"
        if abs(left_difference - right_difference) / difference_scale < 0.10
        else "left"
        if left_difference > right_difference
        else "right"
    )
    left_delta, right_delta = deltas["left_tactile"], deltas["right_tactile"]
    scale = max(abs(left_delta), abs(right_delta), 1e-12)
    enriched["press_depth_delta_stronger_side_secondary"] = (
        "balanced"
        if abs(left_delta - right_delta) / scale < 0.10
        else "left"
        if left_delta > right_delta
        else "right"
    )
    return enriched


def stage_condition(
    *,
    condition: str,
    pair: Mapping[str, Any],
    episode_root: Path,
    output_root: Path,
    simulator_root: Path,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown difference condition: {condition}")
    images_dir = episode_root / "images"
    images_dir.mkdir(parents=True, exist_ok=False)
    sources = pair["sensors"]
    camera_sources = pair["cameras"]
    mappings: list[tuple[str, str, str]] = [
        ("camera/head/rgb", camera_sources["head"], "head_rgb.png"),
        ("camera/wrist/rgb", camera_sources["wrist"], "wrist_rgb.png"),
    ]
    if condition == "raw_pair":
        for side in SENSORS:
            mappings.extend(
                [
                    (
                        f"tactile/{side}/baseline_rgb_marker",
                        sources[side]["baseline_rgb_marker"],
                        f"{side}_baseline.png",
                    ),
                    (
                        f"tactile/{side}/current_rgb_marker",
                        sources[side]["current_rgb_marker"],
                        f"{side}_current.png",
                    ),
                ]
            )
    else:
        difference_source = {
            "left_tactile": "right_tactile"
            if condition == "swapped_difference"
            else "left_tactile",
            "right_tactile": "left_tactile"
            if condition == "swapped_difference"
            else "right_tactile",
        }
        for side in SENSORS:
            mappings.extend(
                [
                    (
                        f"tactile/{side}/current_rgb_marker",
                        sources[side]["current_rgb_marker"],
                        f"{side}_current.png",
                    ),
                    (
                        f"tactile/{side}/difference",
                        sources[difference_source[side]]["difference_visual"],
                        f"{side}_difference.png",
                    ),
                ]
            )
    records = []
    for label, relative_source, filename in mappings:
        source = output_root / relative_source
        destination = images_dir / filename
        shutil.copyfile(source, destination)
        array = _read_rgb(destination)
        records.append(
            {
                "label": label,
                "path": destination.relative_to(episode_root).as_posix(),
                "source_path": relative_source,
                "shape": list(array.shape),
                "dtype": "uint8",
                "media_type": "image/png",
            }
        )
    if len(records) != 6:
        raise RuntimeError("difference condition must stage exactly six images")
    manifest = {
        "schema_version": "openeta.univtac.tactile_difference_condition.v1",
        "condition": condition,
        "simulator_root": str(simulator_root.resolve(strict=True)),
        "images": records,
    }
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
            ref = references.get(str(seed))
            if not row or not ref:
                continue
            for side in ("left", "right"):
                prediction = row[f"{side}_change_region"]
                target = ref["sensors"][f"{side}_tactile"]["metrics"]["centroid_region"]
                if prediction == "uncertain" or target == "inactive":
                    uncertain += 1
                else:
                    evaluated += 1
                    aligned += int(prediction == target)
        return {
            "aligned": aligned,
            "evaluated": evaluated,
            "uncertain_or_inactive": uncertain,
            "rate": aligned / evaluated if evaluated else None,
        }

    def stronger_alignment(condition: str, reference_key: str) -> dict[str, Any]:
        aligned = evaluated = 0
        for seed in SEEDS:
            row = by_pair.get((seed, condition))
            ref = references.get(str(seed))
            if row and ref and row["stronger_change_side"] != "uncertain":
                evaluated += 1
                aligned += int(row["stronger_change_side"] == ref[reference_key])
        return {
            "aligned": aligned,
            "evaluated": evaluated,
            "rate": aligned / evaluated if evaluated else None,
        }

    raw_region = region_alignment("raw_pair")
    explicit_region = region_alignment("explicit_difference")
    raw_stronger = stronger_alignment("raw_pair", "reference_stronger_side")
    explicit_stronger = stronger_alignment("explicit_difference", "reference_stronger_side")
    press_secondary = stronger_alignment(
        "explicit_difference", "press_depth_delta_stronger_side_secondary"
    )
    equivariant = equiv_evaluated = equiv_uncertain = 0
    side_flips = side_flip_evaluated = 0
    for seed in SEEDS:
        explicit = by_pair.get((seed, "explicit_difference"))
        swapped = by_pair.get((seed, "swapped_difference"))
        if not explicit or not swapped:
            continue
        for explicit_field, swapped_field in (
            ("left_change_region", "right_change_region"),
            ("right_change_region", "left_change_region"),
        ):
            values = (explicit[explicit_field], swapped[swapped_field])
            if "uncertain" in values:
                equiv_uncertain += 1
            else:
                equiv_evaluated += 1
                equivariant += int(values[0] == values[1])
        expected = {"left": "right", "right": "left"}.get(explicit["stronger_change_side"])
        if expected is not None:
            side_flip_evaluated += 1
            side_flips += int(swapped["stronger_change_side"] == expected)
    raw_rate, explicit_rate = raw_region["rate"], explicit_region["rate"]
    gain = explicit_rate - raw_rate if raw_rate is not None and explicit_rate is not None else None
    region_improved = gain is not None and gain > 0
    swap_pass = equivariant >= 4
    if region_improved and swap_pass:
        signal = "positive_difference_signal"
    elif region_improved != swap_pass:
        signal = "mixed_difference_signal"
    elif gain is not None and gain < 0 and equivariant == 0:
        signal = "contradictory_difference_signal"
    else:
        signal = "no_detectable_difference_signal"
    return {
        "classification": "tactile_temporal_difference_pilot_completed",
        "difference_signal": signal,
        "raw_pair_region_alignment": raw_region,
        "explicit_difference_region_alignment": explicit_region,
        "difference_region_gain": gain,
        "raw_pair_stronger_side_alignment": raw_stronger,
        "explicit_difference_stronger_side_alignment": explicit_stronger,
        "swapped_difference_equivariance": {
            "equivariant": equivariant,
            "evaluated": equiv_evaluated,
            "uncertain": equiv_uncertain,
            "rate": equivariant / equiv_evaluated if equiv_evaluated else None,
        },
        "swapped_stronger_side_flip_rate": side_flips / side_flip_evaluated
        if side_flip_evaluated
        else None,
        "swapped_stronger_side_flip_count": {
            "flipped": side_flips,
            "evaluated": side_flip_evaluated,
        },
        "press_depth_delta_alignment_secondary": press_secondary,
    }
