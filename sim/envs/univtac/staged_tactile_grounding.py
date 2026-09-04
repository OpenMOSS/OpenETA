"""Pure staging, trace, and metrics helpers for the R0.9.18 grounding pilot."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

SEEDS = (1_000_000, 1_000_001, 1_000_002)
CONDITIONS = ("correct_guidance", "swapped_guidance")
CONDITION_ORDER = {
    1_000_000: CONDITIONS,
    1_000_001: ("swapped_guidance", "correct_guidance"),
    1_000_002: CONDITIONS,
}
EXPECTED_CHANGE_SIDE = {1_000_000: "left", 1_000_001: "right", 1_000_002: "left"}
MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "medium"
IMAGE_JUDGMENT_FIELDS = (
    "tactile_image_access",
    "left_tactile_state",
    "right_tactile_state",
    "tactile_changed_side",
    "left_visual_cue",
    "right_visual_cue",
    "evidence_summary",
)
IMAGE_REPEAT_FIELDS = IMAGE_JUDGMENT_FIELDS[:-1]

PILOT_PROMPT = """You are a read-only multimodal tactile observer.

Use the tools in this exact order:

1. Call `observe_images`.
2. Inspect the tactile images themselves.
3. Call `record_image_judgment` to commit your image-only reading.
4. Call `observe_structured_guidance`.
5. Compare the structured guidance with your already committed image reading.
6. Return the final JSON object.

The structured guidance is derived from tactile imagery, but it may be noisy or inconsistent. Do
not automatically overwrite your image reading. Explicitly report whether the two evidence sources
agree.

Return exactly one JSON object:
{
  "guidance_consistency": "consistent|conflicting|uncertain",
  "final_left_tactile_state": "clear_change|little_or_no_change|uncertain",
  "final_right_tactile_state": "clear_change|little_or_no_change|uncertain",
  "final_changed_side": "left|right|both|neither|uncertain",
  "final_evidence_basis": "image|structured_guidance|both|uncertain",
  "evidence_summary": "...",
  "uncertainty": "..."
}

Do not propose or execute an action. Do not output robot commands. Do not claim exact force,
pressure, friction, grip stability, object pose, or task success.
"""

_FINAL_ENUMS = {
    "guidance_consistency": {"consistent", "conflicting", "uncertain"},
    "final_left_tactile_state": {"clear_change", "little_or_no_change", "uncertain"},
    "final_right_tactile_state": {"clear_change", "little_or_no_change", "uncertain"},
    "final_changed_side": {"left", "right", "both", "neither", "uncertain"},
    "final_evidence_basis": {"image", "structured_guidance", "both", "uncertain"},
}
_FINAL_TEXT_FIELDS = {"evidence_summary", "uncertainty"}
_IMAGE_ENUMS = {
    "tactile_image_access": {"available", "unavailable", "uncertain"},
    "left_tactile_state": {"clear_change", "little_or_no_change", "uncertain"},
    "right_tactile_state": {"clear_change", "little_or_no_change", "uncertain"},
    "tactile_changed_side": {"left", "right", "both", "neither", "uncertain"},
    "left_visual_cue": {
        "localized_colored_disturbance",
        "regular_grid",
        "diffuse_or_ambiguous",
        "unavailable",
    },
    "right_visual_cue": {
        "localized_colored_disturbance",
        "regular_grid",
        "diffuse_or_ambiguous",
        "unavailable",
    },
}


def validate_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_order = {str(seed): list(order) for seed, order in CONDITION_ORDER.items()}
    if tuple(payload.get("seeds", ())) != SEEDS:
        raise ValueError(f"grounding pilot seeds must be exactly {list(SEEDS)}")
    if payload.get("condition_order") != expected_order:
        raise ValueError("grounding pilot condition order is fixed")
    if payload.get("model") != MODEL or payload.get("reasoning_effort") != REASONING_EFFORT:
        raise ValueError("grounding pilot model and reasoning effort are fixed")
    return dict(payload)


def _strip_fenced_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("invalid fenced JSON response")
        stripped = "\n".join(lines[1:-1])
        if stripped.lstrip().startswith("json"):
            stripped = stripped.lstrip()[4:].lstrip("\n")
    return stripped


def parse_final_prediction(text: str) -> dict[str, str]:
    payload = json.loads(_strip_fenced_json(text))
    if not isinstance(payload, dict) or set(payload) != set(_FINAL_ENUMS) | _FINAL_TEXT_FIELDS:
        raise ValueError("final prediction fields do not match the grounding schema")
    for field, allowed in _FINAL_ENUMS.items():
        if payload[field] not in allowed:
            raise ValueError(f"invalid {field}: {payload[field]!r}")
    for field in _FINAL_TEXT_FIELDS:
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"{field} must be non-empty text")
    return {str(key): str(value) for key, value in payload.items()}


def validate_image_judgment(payload: Mapping[str, Any]) -> dict[str, str]:
    if set(payload) != set(IMAGE_JUDGMENT_FIELDS):
        raise ValueError("image judgment fields do not match the grounding schema")
    for field, allowed in _IMAGE_ENUMS.items():
        if payload[field] not in allowed:
            raise ValueError(f"invalid {field}: {payload[field]!r}")
    if not isinstance(payload["evidence_summary"], str) or not payload["evidence_summary"].strip():
        raise ValueError("image judgment evidence_summary must be non-empty")
    return {str(key): str(value) for key, value in payload.items()}


def stage_condition(
    *,
    seed: int,
    condition: str,
    source_root: Path,
    source_mapping: Mapping[str, Any],
    episode_root: Path,
    simulator_root: Path,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown grounding condition: {condition}")
    source_images = source_root / "runs" / f"seed_{seed}" / "image_only" / "images"
    mappings = [
        ("camera/head/rgb", "head_rgb.png"),
        ("camera/wrist/rgb", "wrist_rgb.png"),
        ("tactile/left_tactile/rgb_marker", "left_tactile_rgb_marker.png"),
        ("tactile/right_tactile/rgb_marker", "right_tactile_rgb_marker.png"),
    ]
    images_dir = episode_root / "images"
    images_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for label, filename in mappings:
        source = source_images / filename
        destination = images_dir / filename
        shutil.copyfile(source, destination)
        with Image.open(destination) as image:
            width, height = image.size
        records.append(
            {
                "label": label,
                "path": destination.relative_to(episode_root).as_posix(),
                "source_path": source.relative_to(source_root).as_posix(),
                "shape": [height, width, 3],
                "dtype": "uint8",
                "media_type": "image/png",
            }
        )
    summary_key = (
        "correct_structured_summary"
        if condition == "correct_guidance"
        else "swapped_structured_summary"
    )
    manifest = {
        "schema_version": "openeta.univtac.staged_grounding_condition.v1",
        "condition": condition,
        "simulator_root": str(simulator_root.resolve(strict=True)),
        "images": records,
        "model_visible_structured_tactile_summary": source_mapping[summary_key],
        "counterfactual_multimodal_input": True,
        "image_source_mapping": source_mapping["image_source_mapping"],
        "expected_image_side": source_mapping["expected_image_change_side"],
        "text_indicated_side": source_mapping["expected_image_change_side"]
        if condition == "correct_guidance"
        else (
            "right"
            if source_mapping["expected_image_change_side"] == "left"
            else "left"
        ),
    }
    (episode_root / "condition.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_staged_trace(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    expected_tools = [
        "observe_images",
        "record_image_judgment",
        "observe_structured_guidance",
    ]
    if [row.get("tool") for row in rows] != expected_tools:
        raise ValueError("grounding tool trace order is invalid")
    if [row.get("seq") for row in rows] != [1, 2, 3]:
        raise ValueError("grounding tool trace sequence is invalid")
    if len(rows[0].get("response_image_paths", [])) != 4:
        raise ValueError("observe_images must return exactly four images")
    if rows[1].get("response_image_paths") or rows[2].get("response_image_paths"):
        raise ValueError("judgment and guidance tools must not return images")
    image_payload = json.loads(rows[0]["response_text_blocks"][0])
    expected_labels = [
        "camera/head/rgb",
        "camera/wrist/rgb",
        "tactile/left_tactile/rgb_marker",
        "tactile/right_tactile/rgb_marker",
    ]
    if [item.get("label") for item in image_payload.get("images", [])] != expected_labels:
        raise ValueError("observe_images labels are invalid")
    if "tactile_change_summary" in image_payload:
        raise ValueError("structured guidance leaked before image judgment")
    guidance_payload = json.loads(rows[2]["response_text_blocks"][0])
    if set(guidance_payload) != {"tactile_change_summary"}:
        raise ValueError("guidance tool response is invalid")
    validate_image_judgment(rows[1].get("arguments", {}))
    return rows


def summarize_predictions(
    image_judgments: list[Mapping[str, Any]],
    final_predictions: list[Mapping[str, Any]],
    r0917_summary: Mapping[str, Any],
) -> dict[str, Any]:
    images = {(int(row["seed"]), str(row["condition"])): row for row in image_judgments}
    finals = {(int(row["seed"]), str(row["condition"])): row for row in final_predictions}
    if any((seed, condition) not in images or (seed, condition) not in finals for seed in SEEDS for condition in CONDITIONS):
        raise ValueError("all six grounding trials are required for summary")
    access_available = sum(
        row["tactile_image_access"] == "available" for row in image_judgments
    )
    image_correct = sum(
        images[(seed, condition)]["tactile_changed_side"] == EXPECTED_CHANGE_SIDE[seed]
        for seed in SEEDS
        for condition in CONDITIONS
    )
    repeated = 0
    for seed in SEEDS:
        repeated += int(
            all(
                images[(seed, CONDITIONS[0])][field]
                == images[(seed, CONDITIONS[1])][field]
                for field in IMAGE_REPEAT_FIELDS
            )
        )
    correct_final = sum(
        finals[(seed, "correct_guidance")]["final_changed_side"]
        == EXPECTED_CHANGE_SIDE[seed]
        for seed in SEEDS
    )
    conflict_detected = follow_image = follow_text = overrides = repairs = 0
    for seed in SEEDS:
        expected = EXPECTED_CHANGE_SIDE[seed]
        opposite = "right" if expected == "left" else "left"
        swapped_image = images[(seed, "swapped_guidance")]["tactile_changed_side"]
        swapped_final = finals[(seed, "swapped_guidance")]
        correct_image = images[(seed, "correct_guidance")]["tactile_changed_side"]
        correct_final_row = finals[(seed, "correct_guidance")]
        conflict_detected += int(swapped_final["guidance_consistency"] == "conflicting")
        follow_image += int(swapped_final["final_changed_side"] == expected)
        follow_text += int(swapped_final["final_changed_side"] == opposite)
        overrides += int(swapped_image == expected and swapped_final["final_changed_side"] == opposite)
        repairs += int(
            correct_image != expected and correct_final_row["final_changed_side"] == expected
        )
    image_grounding_established = access_available >= 5 and image_correct >= 4
    if image_grounding_established and follow_image >= 2 and conflict_detected >= 2:
        signal = "grounding_improves_conflict_resistance"
    elif image_grounding_established and follow_text >= 2:
        signal = "image_grounded_but_text_override_persists"
    elif not image_grounding_established:
        signal = "image_consumption_or_interpretation_unreliable"
    else:
        signal = "mixed_grounding_signal"
    old_follow_image = int(
        r0917_summary.get("swapped_follow_image_rate", {}).get("followed", 0)
    )
    old_follow_text = int(
        r0917_summary.get("swapped_follow_text_rate", {}).get("followed", 0)
    )
    return {
        "classification": "staged_tactile_grounding_pilot_completed",
        "image_stage_access_rate": {
            "available": access_available,
            "evaluated": 6,
            "rate": access_available / 6,
        },
        "image_stage_changed_side_accuracy": {
            "correct": image_correct,
            "evaluated": 6,
            "rate": image_correct / 6,
        },
        "image_stage_repeat_consistency": {
            "consistent": repeated,
            "evaluated": 3,
            "rate": repeated / 3,
        },
        "correct_final_changed_side_accuracy": {
            "correct": correct_final,
            "evaluated": 3,
            "rate": correct_final / 3,
        },
        "swapped_conflict_detection_rate": {
            "detected": conflict_detected,
            "evaluated": 3,
            "rate": conflict_detected / 3,
        },
        "swapped_final_follow_image_rate": {
            "followed": follow_image,
            "evaluated": 3,
            "rate": follow_image / 3,
        },
        "swapped_final_follow_text_rate": {
            "followed": follow_text,
            "evaluated": 3,
            "rate": follow_text / 3,
        },
        "guidance_override_count": overrides,
        "correct_guidance_repair_count": repairs,
        "r0917_comparison": {
            "swapped_follow_image_baseline": old_follow_image,
            "swapped_follow_image_change": follow_image - old_follow_image,
            "swapped_follow_text_baseline": old_follow_text,
            "swapped_follow_text_change": follow_text - old_follow_text,
        },
        "grounding_signal": signal,
        "signal_flags": {"image_grounding_established": image_grounding_established},
    }


__all__ = [
    "CONDITIONS",
    "CONDITION_ORDER",
    "EXPECTED_CHANGE_SIDE",
    "IMAGE_JUDGMENT_FIELDS",
    "MODEL",
    "PILOT_PROMPT",
    "REASONING_EFFORT",
    "SEEDS",
    "load_staged_trace",
    "parse_final_prediction",
    "stage_condition",
    "summarize_predictions",
    "validate_config",
    "validate_image_judgment",
]
