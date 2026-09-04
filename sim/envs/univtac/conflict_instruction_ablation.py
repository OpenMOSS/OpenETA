"""Pure helpers for the R0.9.20 conflict-instruction interaction ablation."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sim.envs.univtac.grounding_mechanism_ablation import (
    EXPECTED_IMAGE_SIDE,
    load_protocol_trace,
    parse_final_prediction,
)

SEEDS = (1_000_000, 1_000_001, 1_000_002)
PROTOCOLS = (
    "simultaneous_conflict_aware",
    "image_first_no_commit_conflict_aware",
)
PROTOCOL_ORDER = {
    1_000_000: PROTOCOLS,
    1_000_001: (PROTOCOLS[1], PROTOCOLS[0]),
    1_000_002: PROTOCOLS,
}
MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "medium"
MCP_MODES = {
    "simultaneous_conflict_aware": "simultaneous_fusion",
    "image_first_no_commit_conflict_aware": "image_first_no_commit",
}
MCP_TOOLS = {
    "simultaneous_conflict_aware": ["observe"],
    "image_first_no_commit_conflict_aware": [
        "observe_images",
        "observe_structured_guidance",
    ],
}

COMMON_CONFLICT_INSTRUCTION = """The left and right tactile images are GelSight rgb_marker views.
The structured summary contains image-change measurements assigned to the two tactile sides.

The structured tactile summary is derived evidence and may be incorrect or assigned to the wrong
tactile side.

Inspect the tactile images themselves. When image evidence and structured guidance conflict,
explicitly report the conflict. Do not replace a clear image-supported judgment merely because the
structured summary disagrees. When the tactile images are genuinely ambiguous or unavailable,
report uncertainty rather than blindly choosing either source.

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

_PROCEDURES = {
    PROTOCOLS[0]: "Call `observe` exactly once, then return the final JSON object.",
    PROTOCOLS[1]: (
        "Call `observe_images`, then call `observe_structured_guidance`, then return the final "
        "JSON object."
    ),
}


def prompt_for(protocol: str) -> str:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown conflict-aware protocol: {protocol}")
    return f"You are a read-only multimodal tactile observer.\n\n{_PROCEDURES[protocol]}\n\n{COMMON_CONFLICT_INSTRUCTION}"


def validate_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_order = {str(seed): list(order) for seed, order in PROTOCOL_ORDER.items()}
    if tuple(payload.get("seeds", ())) != SEEDS:
        raise ValueError(f"conflict ablation seeds must be exactly {list(SEEDS)}")
    if payload.get("protocol_order") != expected_order:
        raise ValueError("conflict ablation protocol order is fixed")
    if payload.get("model") != MODEL or payload.get("reasoning_effort") != REASONING_EFFORT:
        raise ValueError("conflict ablation model and reasoning effort are fixed")
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
        raise ValueError("unknown conflict ablation seed or protocol")
    source = source_root / "runs" / f"seed_{seed}" / "simultaneous_fusion"
    source_manifest = _load_json(source / "condition.json")
    images_dir = episode_root / "images"
    images_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for item in source_manifest["images"]:
        filename = Path(str(item["path"])).name
        source_image = source / "images" / filename
        destination = images_dir / filename
        shutil.copyfile(source_image, destination)
        records.append(
            {
                "label": item["label"],
                "path": destination.relative_to(episode_root).as_posix(),
                "source_path": source_image.relative_to(source_root).as_posix(),
                "shape": list(item["shape"]),
                "dtype": item["dtype"],
                "media_type": item.get("media_type", "image/png"),
            }
        )
    expected = EXPECTED_IMAGE_SIDE[seed]
    manifest = {
        "schema_version": "openeta.univtac.conflict_instruction_condition.v1",
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
        roots = [output_root / "runs" / f"seed_{seed}" / protocol for protocol in PROTOCOLS]
        manifests = [_load_json(root / "condition.json") for root in roots]
        if (
            manifests[0]["model_visible_structured_tactile_summary"]
            != manifests[1]["model_visible_structured_tactile_summary"]
        ):
            raise ValueError(f"structured guidance differs across protocols for seed {seed}")
        for filename in (
            "head_rgb.png",
            "wrist_rgb.png",
            "left_tactile_rgb_marker.png",
            "right_tactile_rgb_marker.png",
        ):
            arrays = [np.asarray(Image.open(root / "images" / filename)) for root in roots]
            if not np.array_equal(arrays[0], arrays[1]):
                raise ValueError(f"protocol image inputs differ for seed {seed}: {filename}")


def validate_trace(path: Path, protocol: str) -> list[dict[str, Any]]:
    return load_protocol_trace(path, MCP_MODES[protocol])


def summarize_predictions(
    predictions: list[Mapping[str, Any]],
    r0919_summary: Mapping[str, Any],
    r0918_summary: Mapping[str, Any],
) -> dict[str, Any]:
    rows = {(int(row["seed"]), str(row["protocol"])): row for row in predictions}
    expected_pairs = {(seed, protocol) for seed in SEEDS for protocol in PROTOCOLS}
    if set(rows) != expected_pairs:
        raise ValueError("all six conflict-aware trials are required")
    metrics: dict[str, Any] = {}
    for protocol in PROTOCOLS:
        selected = [rows[(seed, protocol)] for seed in SEEDS]
        conflict = sum(row["guidance_consistency"] == "conflicting" for row in selected)
        follow_image = sum(
            row["final_changed_side"] == EXPECTED_IMAGE_SIDE[seed]
            for seed, row in zip(SEEDS, selected, strict=True)
        )
        follow_text = sum(
            row["final_changed_side"]
            == ("right" if EXPECTED_IMAGE_SIDE[seed] == "left" else "left")
            for seed, row in zip(SEEDS, selected, strict=True)
        )
        image_basis = sum(row["final_evidence_basis"] == "image" for row in selected)
        both_basis = sum(row["final_evidence_basis"] == "both" for row in selected)
        uncertain = sum(row["final_changed_side"] == "uncertain" for row in selected)
        fused_both = sum(row["final_changed_side"] == "both" for row in selected)
        pilot_sufficient = follow_image >= 2 and follow_text == 0
        robust = follow_image == 3 and follow_text == 0 and conflict == 3
        metrics[protocol] = {
            "conflict_detection_rate": {"detected": conflict, "evaluated": 3, "rate": conflict / 3},
            "final_follow_image_rate": {"followed": follow_image, "evaluated": 3, "rate": follow_image / 3},
            "final_follow_text_rate": {"followed": follow_text, "evaluated": 3, "rate": follow_text / 3},
            "final_image_basis_rate": {"selected": image_basis, "evaluated": 3, "rate": image_basis / 3},
            "final_both_basis_rate": {"selected": both_basis, "evaluated": 3, "rate": both_basis / 3},
            "uncertain_rate": {"uncertain": uncertain, "evaluated": 3, "rate": uncertain / 3},
            "fused_both_side_rate": {"both": fused_both, "evaluated": 3, "rate": fused_both / 3},
            "pilot_sufficient": pilot_sufficient,
            "robust_on_current_three_seeds": robust,
        }
    old_metrics = r0919_summary["protocol_metrics"]
    new_sim = metrics[PROTOCOLS[0]]["final_follow_image_rate"]["followed"]
    new_order = metrics[PROTOCOLS[1]]["final_follow_image_rate"]["followed"]
    old_sim = old_metrics["simultaneous_fusion"]["final_follow_image_rate"]["followed"]
    old_order = old_metrics["image_first_no_commit"]["final_follow_image_rate"]["followed"]
    r0918_image = r0918_summary["swapped_final_follow_image_rate"]["followed"]
    r0918_text = r0918_summary["swapped_final_follow_text_rate"]["followed"]
    r0918_conflict = r0918_summary["swapped_conflict_detection_rate"]["detected"]
    if metrics[PROTOCOLS[0]]["robust_on_current_three_seeds"]:
        candidate = "conflict_aware_instruction_without_staged_ordering"
    elif metrics[PROTOCOLS[1]]["robust_on_current_three_seeds"]:
        candidate = "image_first_ordering_plus_conflict_aware_instruction"
    elif r0918_image == 3 and r0918_text == 0 and r0918_conflict == 3:
        candidate = "image_first_commitment_plus_conflict_aware_instruction"
    else:
        candidate = "grounding_skill_mechanism_unresolved"
    return {
        "classification": "tactile_conflict_instruction_interaction_ablation_completed",
        "protocol_metrics": metrics,
        "instruction_gain_simultaneous": new_sim - old_sim,
        "instruction_gain_image_first": new_order - old_order,
        "ordering_gain_under_same_instruction": new_order - new_sim,
        "r0919_reference": {
            "simultaneous_neutral": old_metrics["simultaneous_fusion"],
            "image_first_no_commit_neutral": old_metrics["image_first_no_commit"],
            "image_first_commit_neutral": r0919_summary["image_first_with_commit"],
        },
        "r0918_reference": r0918_summary,
        "minimal_skill_candidate": candidate,
    }


__all__ = [
    "COMMON_CONFLICT_INSTRUCTION",
    "MCP_MODES",
    "MCP_TOOLS",
    "MODEL",
    "PROTOCOLS",
    "PROTOCOL_ORDER",
    "REASONING_EFFORT",
    "SEEDS",
    "assert_paired_inputs_equal",
    "parse_final_prediction",
    "prompt_for",
    "stage_protocol",
    "summarize_predictions",
    "validate_config",
    "validate_trace",
]
