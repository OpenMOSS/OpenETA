"""Pure contracts and analysis for the R0.9.14 tactile causal pilot."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SEEDS = (1_000_000, 1_000_001, 1_000_002)
CONDITIONS = ("visual_only", "correct_tactile", "swapped_tactile")
CONDITION_ORDER = {
    1_000_000: CONDITIONS,
    1_000_001: ("correct_tactile", "swapped_tactile", "visual_only"),
    1_000_002: ("swapped_tactile", "visual_only", "correct_tactile"),
}
MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "medium"
CAMERA_LABELS = ("camera/head/rgb", "camera/wrist/rgb")
TACTILE_LABELS = (
    "tactile/left_tactile/rgb_marker",
    "tactile/right_tactile/rgb_marker",
)

PILOT_PROMPT = """You are the read-only operator for one UniVTAC Pull Out Key episode.

Call the MCP tool `observe` exactly once before answering. Inspect every returned image. Tactile
rgb_marker images may or may not be available. Base your answer only on the task text, proprio,
and images returned by `observe`.

Return exactly one JSON object with these fields:
{
  "scene_summary": "...",
  "left_contact": "likely_contact | likely_no_contact | uncertain",
  "right_contact": "likely_contact | likely_no_contact | uncertain",
  "bilateral_contact": "yes | no | uncertain",
  "left_marker_region": "left | center | right | diffuse | uncertain | not_available",
  "right_marker_region": "left | center | right | diffuse | uncertain | not_available",
  "stronger_contact_side": "left | right | balanced | uncertain",
  "evidence_source": "visual_only | visual_and_tactile",
  "evidence_summary": "...",
  "uncertainty": "..."
}

Do not execute an action. Do not emit numeric robot commands. Do not claim access to force,
depth, exact object pose, success labels, or host-only metadata.
"""

_ENUMS = {
    "left_contact": {"likely_contact", "likely_no_contact", "uncertain"},
    "right_contact": {"likely_contact", "likely_no_contact", "uncertain"},
    "bilateral_contact": {"yes", "no", "uncertain"},
    "left_marker_region": {"left", "center", "right", "diffuse", "uncertain", "not_available"},
    "right_marker_region": {"left", "center", "right", "diffuse", "uncertain", "not_available"},
    "stronger_contact_side": {"left", "right", "balanced", "uncertain"},
    "evidence_source": {"visual_only", "visual_and_tactile"},
}
_TEXT_FIELDS = {"scene_summary", "evidence_summary", "uncertainty"}


def validate_pilot_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    if tuple(payload.get("seeds", ())) != SEEDS:
        raise ValueError(f"pilot seeds must be exactly {list(SEEDS)}")
    if payload.get("model") != MODEL or payload.get("reasoning_effort") != REASONING_EFFORT:
        raise ValueError("pilot model and reasoning effort are fixed")
    if payload.get("condition_order") != {str(k): list(v) for k, v in CONDITION_ORDER.items()}:
        raise ValueError("pilot condition order must use the fixed Latin square")
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
        raise ValueError("prediction JSON fields do not match the pilot schema")
    for field, allowed in _ENUMS.items():
        if payload[field] not in allowed:
            raise ValueError(f"invalid {field}: {payload[field]!r}")
    for field in _TEXT_FIELDS:
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"{field} must be non-empty text")
    return {str(key): str(value) for key, value in payload.items()}


def stage_condition(
    *,
    condition: str,
    context_images: Mapping[str, Path],
    episode_root: Path,
    simulator_root: Path,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown pilot condition: {condition}")
    root = episode_root.resolve()
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=False)
    source_labels = list(CAMERA_LABELS)
    if condition == "correct_tactile":
        source_labels.extend(TACTILE_LABELS)
    elif condition == "swapped_tactile":
        source_labels.extend(reversed(TACTILE_LABELS))
    target_labels = list(CAMERA_LABELS)
    if condition != "visual_only":
        target_labels.extend(TACTILE_LABELS)
    records: list[dict[str, str]] = []
    names = (
        "head_rgb.png",
        "wrist_rgb.png",
        "left_tactile_rgb_marker.png",
        "right_tactile_rgb_marker.png",
    )
    for target, source, name in zip(
        target_labels, source_labels, names[: len(target_labels)], strict=True
    ):
        source_path = context_images[source].resolve(strict=True)
        destination = images_dir / name
        shutil.copyfile(source_path, destination)
        with Image.open(source_path) as before, Image.open(destination) as after:
            if not np.array_equal(np.asarray(before), np.asarray(after)):
                raise RuntimeError("staged condition image differs from its source pixels")
        records.append({"label": target, "path": f"images/{name}", "source_label": source})
    manifest = {
        "schema_version": "openeta.univtac.tactile_causal_condition.v1",
        "condition": condition,
        "simulator_root": str(simulator_root.resolve(strict=True)),
        "images": records,
    }
    (root / "condition.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def press_depth_reference(contact_summary: Mapping[str, Any]) -> dict[str, Any]:
    sensors = contact_summary["sensors"]
    left = sensors["left_tactile"]
    right = sensors["right_tactile"]
    left_mean, right_mean = float(left["mean"]), float(right["mean"])
    scale = max(abs(left_mean), abs(right_mean), 1e-12)
    if abs(left_mean - right_mean) / scale < 0.10:
        dominant = "balanced"
    else:
        dominant = "left" if left_mean > right_mean else "right"
    return {
        "left_contact_candidate": float(left["maximum"]) > 0,
        "right_contact_candidate": float(right["maximum"]) > 0,
        "bilateral_contact_candidate": bool(
            float(left["maximum"]) > 0 and float(right["maximum"]) > 0
        ),
        "left_mean_press_depth": left_mean,
        "left_max_press_depth": float(left["maximum"]),
        "left_positive_pixel_ratio": float(left["positive_pixel_ratio"]),
        "right_mean_press_depth": right_mean,
        "right_max_press_depth": float(right["maximum"]),
        "right_positive_pixel_ratio": float(right["positive_pixel_ratio"]),
        "press_depth_dominant_side_proxy": dominant,
    }


def summarize_pilot(
    predictions: list[Mapping[str, Any]], references: Mapping[str, Any]
) -> dict[str, Any]:
    by_pair = {(int(row["seed"]), str(row["condition"])): row for row in predictions}
    contact = {"aligned": 0, "uncertain": 0, "contradictory": 0}
    dominant_aligned = 0
    dominant_opposite = 0
    dominant_total = 0
    flips = 0
    flip_total = 0
    region_equivariant = 0
    region_uncertain = 0
    region_total = 0
    visual_uncertain = 0
    visual_total = 0
    tactile_changes = 0
    tactile_change_total = 0
    for seed in SEEDS:
        ref = references.get(str(seed))
        visual = by_pair.get((seed, "visual_only"))
        correct = by_pair.get((seed, "correct_tactile"))
        swapped = by_pair.get((seed, "swapped_tactile"))
        if correct and ref:
            for side in ("left", "right"):
                pred = correct[f"{side}_contact"]
                truth = bool(ref[f"{side}_contact_candidate"])
                if pred == "uncertain":
                    contact["uncertain"] += 1
                elif (pred == "likely_contact") == truth:
                    contact["aligned"] += 1
                else:
                    contact["contradictory"] += 1
            side = correct["stronger_contact_side"]
            if side != "uncertain":
                dominant_total += 1
                dominant_aligned += int(side == ref["press_depth_dominant_side_proxy"])
                dominant_opposite += int(
                    side in {"left", "right"}
                    and ref["press_depth_dominant_side_proxy"] in {"left", "right"}
                    and side != ref["press_depth_dominant_side_proxy"]
                )
        if correct and swapped:
            expected = {"left": "right", "right": "left"}.get(str(correct["stronger_contact_side"]))
            flip_total += 1
            flips += int(expected is not None and swapped["stronger_contact_side"] == expected)
            for correct_field, swapped_field in (
                ("left_marker_region", "right_marker_region"),
                ("right_marker_region", "left_marker_region"),
            ):
                region_total += 1
                values = (correct[correct_field], swapped[swapped_field])
                if any(v in {"uncertain", "not_available"} for v in values):
                    region_uncertain += 1
                else:
                    region_equivariant += int(values[0] == values[1])
        if visual:
            for field in ("left_contact", "right_contact", "stronger_contact_side"):
                visual_total += 1
                visual_uncertain += int(visual[field] == "uncertain")
        if visual and correct:
            for field in (
                "left_contact",
                "right_contact",
                "bilateral_contact",
                "stronger_contact_side",
            ):
                tactile_change_total += 1
                tactile_changes += int(visual[field] != correct[field])
    metrics = {
        "correct_tactile_contact_alignment": contact,
        "correct_tactile_dominant_side_alignment": {
            "aligned": dominant_aligned,
            "opposite": dominant_opposite,
            "evaluated": dominant_total,
            "rate": dominant_aligned / dominant_total if dominant_total else None,
        },
        "swapped_tactile_side_flip_rate": flips / flip_total if flip_total else None,
        "swapped_tactile_side_flip_count": {"flipped": flips, "evaluated": flip_total},
        "marker_region_swap_equivariance": {
            "equivariant": region_equivariant,
            "uncertain_or_not_available": region_uncertain,
            "evaluated": region_total - region_uncertain,
            "total_directions": region_total,
            "rate": (
                region_equivariant / (region_total - region_uncertain)
                if region_total > region_uncertain
                else None
            ),
        },
        "visual_only_uncertainty_rate": visual_uncertain / visual_total if visual_total else None,
        "tactile_changes_prediction_rate": (
            tactile_changes / tactile_change_total if tactile_change_total else None
        ),
    }
    valid_all = len(by_pair) == 9
    dominant_rate = metrics["correct_tactile_dominant_side_alignment"]["rate"]
    flip_rate = metrics["swapped_tactile_side_flip_rate"]
    dominant_pass = dominant_rate is not None and dominant_rate >= 2 / 3
    flip_pass = flip_rate is not None and flip_rate >= 2 / 3
    many_uncertain = contact["uncertain"] >= 3 or dominant_total < 2
    if valid_all and dominant_pass and flip_pass:
        signal = "positive_causal_signal"
    elif dominant_total == 3 and dominant_opposite == 3 and flips == 0:
        signal = "contradictory_signal"
    elif dominant_pass != flip_pass or many_uncertain:
        signal = "mixed_signal"
    else:
        signal = "no_detectable_signal"
    return {
        "classification": "pull_out_key_tactile_causal_pilot_completed",
        "pilot_signal": signal,
        "metrics": metrics,
    }
