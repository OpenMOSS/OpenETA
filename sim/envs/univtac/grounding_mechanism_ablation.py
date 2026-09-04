"""Pure staging, trace, and metrics helpers for the R0.9.19 mechanism ablation."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sim.envs.univtac.staged_tactile_grounding import (
    parse_final_prediction,
    validate_image_judgment,
)

SEEDS = (1_000_000, 1_000_001, 1_000_002)
PROTOCOLS = (
    "simultaneous_fusion",
    "image_first_no_commit",
    "image_first_with_commit",
)
PROTOCOL_ORDER = {
    1_000_000: PROTOCOLS,
    1_000_001: (
        "image_first_no_commit",
        "image_first_with_commit",
        "simultaneous_fusion",
    ),
    1_000_002: (
        "image_first_with_commit",
        "simultaneous_fusion",
        "image_first_no_commit",
    ),
}
EXPECTED_IMAGE_SIDE = {1_000_000: "left", 1_000_001: "right", 1_000_002: "left"}
MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "medium"

COMMON_SEMANTIC_PROMPT = """The left and right tactile images are GelSight rgb_marker views.
The structured summary contains image-change measurements assigned to the two tactile sides.

Assess whether the visual tactile evidence and the structured summary agree or conflict. Use all
evidence available in the current protocol.

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

Do not propose or execute an action. Do not claim force, pressure, friction, grip stability,
object pose, or task success.
"""

_PROCEDURES = {
    "simultaneous_fusion": "Call `observe` exactly once, then return the final JSON object.",
    "image_first_no_commit": (
        "Call `observe_images`, then call `observe_structured_guidance`, then return the final "
        "JSON object."
    ),
    "image_first_with_commit": (
        "Call `observe_images`, call `record_image_judgment` with your image reading, then call "
        "`observe_structured_guidance`, then return the final JSON object."
    ),
}

MCP_MODES = {
    "simultaneous_fusion": "simultaneous_fusion",
    "image_first_no_commit": "image_first_no_commit",
    "image_first_with_commit": "image_first_with_commit",
}

MCP_TOOLS = {
    "simultaneous_fusion": ["observe"],
    "image_first_no_commit": ["observe_images", "observe_structured_guidance"],
    "image_first_with_commit": [
        "observe_images",
        "record_image_judgment",
        "observe_structured_guidance",
    ],
}

_IMAGE_LABELS = [
    "camera/head/rgb",
    "camera/wrist/rgb",
    "tactile/left_tactile/rgb_marker",
    "tactile/right_tactile/rgb_marker",
]


def prompt_for(protocol: str) -> str:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown grounding mechanism protocol: {protocol}")
    return f"You are a read-only multimodal tactile observer.\n\n{_PROCEDURES[protocol]}\n\n{COMMON_SEMANTIC_PROMPT}"


def validate_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_order = {str(seed): list(order) for seed, order in PROTOCOL_ORDER.items()}
    if tuple(payload.get("seeds", ())) != SEEDS:
        raise ValueError(f"mechanism ablation seeds must be exactly {list(SEEDS)}")
    if payload.get("protocol_order") != expected_order:
        raise ValueError("mechanism ablation Latin square is fixed")
    if payload.get("model") != MODEL or payload.get("reasoning_effort") != REASONING_EFFORT:
        raise ValueError("mechanism ablation model and reasoning effort are fixed")
    return dict(payload)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def stage_protocol(
    *, seed: int, protocol: str, source_root: Path, episode_root: Path
) -> dict[str, Any]:
    if seed not in SEEDS or protocol not in PROTOCOLS:
        raise ValueError("unknown mechanism ablation seed or protocol")
    source_condition = source_root / "runs" / f"seed_{seed}" / "swapped_guidance"
    source_manifest = _load_json(source_condition / "condition.json")
    source_images = source_condition / "images"
    records: list[dict[str, Any]] = []
    images_dir = episode_root / "images"
    images_dir.mkdir(parents=True, exist_ok=False)
    for item in source_manifest.get("images", []):
        if item.get("label") not in _IMAGE_LABELS:
            raise ValueError("source manifest does not contain the four canonical images")
        filename = Path(str(item["path"])).name
        source = source_images / filename
        destination = images_dir / filename
        shutil.copyfile(source, destination)
        records.append(
            {
                "label": item["label"],
                "path": destination.relative_to(episode_root).as_posix(),
                "source_path": source.relative_to(source_root).as_posix(),
                "shape": list(item["shape"]),
                "dtype": item["dtype"],
                "media_type": item.get("media_type", "image/png"),
            }
        )
    if [item["label"] for item in records] != _IMAGE_LABELS:
        raise ValueError("source manifest image order is invalid")
    expected = EXPECTED_IMAGE_SIDE[seed]
    manifest = {
        "schema_version": "openeta.univtac.grounding_mechanism_condition.v1",
        "protocol": protocol,
        "simulator_root": source_manifest["simulator_root"],
        "images": records,
        "model_visible_structured_tactile_summary": source_manifest[
            "model_visible_structured_tactile_summary"
        ],
        "counterfactual_multimodal_input": True,
        "expected_image_side": expected,
        "text_indicated_side": "right" if expected == "left" else "left",
    }
    (episode_root / "condition.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def assert_paired_inputs_equal(output_root: Path) -> None:
    for seed in SEEDS:
        manifests = [
            _load_json(output_root / "runs" / f"seed_{seed}" / protocol / "condition.json")
            for protocol in PROTOCOLS
        ]
        summaries = [item["model_visible_structured_tactile_summary"] for item in manifests]
        if summaries[1:] != summaries[:-1]:
            raise ValueError(f"structured guidance differs across protocols for seed {seed}")
        for filename in (
            "head_rgb.png",
            "wrist_rgb.png",
            "left_tactile_rgb_marker.png",
            "right_tactile_rgb_marker.png",
        ):
            arrays = [
                np.asarray(
                    Image.open(
                        output_root / "runs" / f"seed_{seed}" / protocol / "images" / filename
                    )
                )
                for protocol in PROTOCOLS
            ]
            if not all(np.array_equal(arrays[0], array) for array in arrays[1:]):
                raise ValueError(f"protocol image inputs differ for seed {seed}: {filename}")


def load_protocol_trace(path: Path, protocol: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    expected_tools = MCP_TOOLS[protocol]
    if [row.get("tool") for row in rows] != expected_tools:
        raise ValueError(f"tool trace order is invalid for {protocol}")
    if [row.get("seq") for row in rows] != list(range(1, len(expected_tools) + 1)):
        raise ValueError(f"tool trace sequence is invalid for {protocol}")
    image_row = rows[0]
    if len(image_row.get("response_image_paths", [])) != 4:
        raise ValueError("the image observation must return exactly four images")
    payload = json.loads(image_row["response_text_blocks"][0])
    if [item.get("label") for item in payload.get("images", [])] != _IMAGE_LABELS:
        raise ValueError("image labels are invalid")
    guidance_in_first = "tactile_change_summary" in payload
    if protocol == "simultaneous_fusion":
        if not guidance_in_first:
            raise ValueError("simultaneous fusion omitted structured guidance")
    else:
        if guidance_in_first:
            raise ValueError("structured guidance leaked into the image-first observation")
        guidance_payload = json.loads(rows[-1]["response_text_blocks"][0])
        if set(guidance_payload) != {"tactile_change_summary"}:
            raise ValueError("structured guidance response is invalid")
        if rows[-1].get("response_image_paths"):
            raise ValueError("structured guidance must not return images")
    if protocol == "image_first_with_commit":
        validate_image_judgment(rows[1].get("arguments", {}))
        if rows[1].get("response_image_paths"):
            raise ValueError("image judgment must not return images")
    return rows


def summarize_predictions(
    final_predictions: list[Mapping[str, Any]],
    image_judgments: list[Mapping[str, Any]],
    r0918_summary: Mapping[str, Any],
) -> dict[str, Any]:
    finals = {(int(row["seed"]), str(row["protocol"])): row for row in final_predictions}
    expected_pairs = {(seed, protocol) for seed in SEEDS for protocol in PROTOCOLS}
    if set(finals) != expected_pairs:
        raise ValueError("all nine mechanism ablation trials are required")
    judgments = {int(row["seed"]): row for row in image_judgments}
    if set(judgments) != set(SEEDS):
        raise ValueError("three committed image judgments are required")
    per_protocol: dict[str, Any] = {}
    for protocol in PROTOCOLS:
        rows = [finals[(seed, protocol)] for seed in SEEDS]
        conflict = sum(row["guidance_consistency"] == "conflicting" for row in rows)
        follow_image = sum(
            row["final_changed_side"] == EXPECTED_IMAGE_SIDE[seed]
            for seed, row in zip(SEEDS, rows, strict=True)
        )
        follow_text = sum(
            row["final_changed_side"]
            == ("right" if EXPECTED_IMAGE_SIDE[seed] == "left" else "left")
            for seed, row in zip(SEEDS, rows, strict=True)
        )
        image_basis = sum(row["final_evidence_basis"] == "image" for row in rows)
        uncertain = sum(row["final_changed_side"] == "uncertain" for row in rows)
        per_protocol[protocol] = {
            "conflict_detection_rate": {
                "detected": conflict,
                "evaluated": 3,
                "rate": conflict / 3,
            },
            "final_follow_image_rate": {
                "followed": follow_image,
                "evaluated": 3,
                "rate": follow_image / 3,
            },
            "final_follow_text_rate": {
                "followed": follow_text,
                "evaluated": 3,
                "rate": follow_text / 3,
            },
            "image_evidence_basis_rate": {
                "selected": image_basis,
                "evaluated": 3,
                "rate": image_basis / 3,
            },
            "uncertain_rate": {
                "uncertain": uncertain,
                "evaluated": 3,
                "rate": uncertain / 3,
            },
        }
    provisional_correct = sum(
        judgments[seed]["tactile_changed_side"] == EXPECTED_IMAGE_SIDE[seed] for seed in SEEDS
    )
    overrides = sum(
        judgments[seed]["tactile_changed_side"] == EXPECTED_IMAGE_SIDE[seed]
        and finals[(seed, "image_first_with_commit")]["final_changed_side"]
        == ("right" if EXPECTED_IMAGE_SIDE[seed] == "left" else "left")
        for seed in SEEDS
    )
    follow = {
        protocol: per_protocol[protocol]["final_follow_image_rate"]["followed"]
        for protocol in PROTOCOLS
    }
    if follow["simultaneous_fusion"] >= 2:
        mechanism = "simultaneous_fusion_grounded_in_this_run"
    elif follow["image_first_no_commit"] >= 2:
        mechanism = "image_first_ordering_sufficient"
    elif follow["image_first_with_commit"] >= 2:
        mechanism = "explicit_commitment_required"
    elif int(
        r0918_summary.get("swapped_final_follow_image_rate", {}).get("followed", 0)
    ) == 3:
        mechanism = "explicit_conflict_aware_instruction_required"
    else:
        mechanism = "grounding_mechanism_unresolved"
    return {
        "classification": "tactile_grounding_mechanism_ablation_completed",
        "protocol_metrics": per_protocol,
        "image_first_with_commit": {
            "provisional_image_accuracy": {
                "correct": provisional_correct,
                "evaluated": 3,
                "rate": provisional_correct / 3,
            },
            "guidance_override_count": overrides,
        },
        "r0918_reference": {
            "conflict_aware_committed_protocol": {
                "follow_image": r0918_summary.get("swapped_final_follow_image_rate"),
                "follow_text": r0918_summary.get("swapped_final_follow_text_rate"),
            }
        },
        "mechanism_result": mechanism,
    }


__all__ = [
    "COMMON_SEMANTIC_PROMPT",
    "EXPECTED_IMAGE_SIDE",
    "MCP_MODES",
    "MCP_TOOLS",
    "MODEL",
    "PROTOCOLS",
    "PROTOCOL_ORDER",
    "REASONING_EFFORT",
    "SEEDS",
    "assert_paired_inputs_equal",
    "load_protocol_trace",
    "parse_final_prediction",
    "prompt_for",
    "stage_protocol",
    "summarize_predictions",
    "validate_config",
    "validate_image_judgment",
]
