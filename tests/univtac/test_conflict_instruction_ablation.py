from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from sim.envs.univtac.conflict_instruction_ablation import (
    COMMON_CONFLICT_INSTRUCTION,
    MCP_TOOLS,
    PROTOCOL_ORDER,
    PROTOCOLS,
    SEEDS,
    assert_paired_inputs_equal,
    prompt_for,
    stage_protocol,
    summarize_predictions,
    validate_config,
)
from sim.envs.univtac.grounding_mechanism_ablation import EXPECTED_IMAGE_SIDE


def _source(tmp_path: Path) -> Path:
    root = tmp_path / "r0919"
    labels = (
        "camera/head/rgb",
        "camera/wrist/rgb",
        "tactile/left_tactile/rgb_marker",
        "tactile/right_tactile/rgb_marker",
    )
    filenames = (
        "head_rgb.png",
        "wrist_rgb.png",
        "left_tactile_rgb_marker.png",
        "right_tactile_rgb_marker.png",
    )
    for seed in SEEDS:
        run = root / "runs" / f"seed_{seed}" / "simultaneous_fusion"
        (run / "images").mkdir(parents=True)
        images = []
        for index, (label, filename) in enumerate(zip(labels, filenames, strict=True)):
            Image.new("RGB", (8, 6), (index, 20, 30)).save(run / "images" / filename)
            images.append(
                {
                    "label": label,
                    "path": f"images/{filename}",
                    "shape": [6, 8, 3],
                    "dtype": "uint8",
                    "media_type": "image/png",
                }
            )
        metrics = {
            field: 18.0
            for field in (
                "active_pixel_ratio",
                "mean_absolute_change",
                "p95_absolute_change",
                "salient_mass",
            )
        }
        (run / "condition.json").write_text(
            json.dumps(
                {
                    "simulator_root": str(tmp_path),
                    "images": images,
                    "model_visible_structured_tactile_summary": {
                        "left_tactile": metrics,
                        "right_tactile": {key: 0.0 for key in metrics},
                    },
                }
            ),
            encoding="utf-8",
        )
    return root


def test_config_order_and_shared_conflict_instruction() -> None:
    payload = {
        "seeds": list(SEEDS),
        "protocol_order": {str(seed): list(PROTOCOL_ORDER[seed]) for seed in SEEDS},
        "model": "gpt-5.6-terra",
        "reasoning_effort": "medium",
    }
    assert validate_config(payload) == payload
    for protocol in PROTOCOLS:
        prompt = prompt_for(protocol)
        assert prompt.endswith(COMMON_CONFLICT_INSTRUCTION)
        assert protocol not in prompt
    assert "record_image_judgment" not in " ".join(prompt_for(p) for p in PROTOCOLS)
    assert MCP_TOOLS[PROTOCOLS[0]] == ["observe"]
    assert MCP_TOOLS[PROTOCOLS[1]] == ["observe_images", "observe_structured_guidance"]


def test_two_protocols_share_images_and_guidance(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "out"
    for seed in SEEDS:
        for protocol in PROTOCOLS:
            episode = output / "runs" / f"seed_{seed}" / protocol
            episode.mkdir(parents=True)
            stage_protocol(
                seed=seed, protocol=protocol, source_root=source, episode_root=episode
            )
    assert_paired_inputs_equal(output)
    assert all(
        json.loads(
            (output / "runs" / f"seed_{seed}" / protocol / "condition.json").read_text()
        )["text_indicated_side"]
        != EXPECTED_IMAGE_SIDE[seed]
        for seed in SEEDS
        for protocol in PROTOCOLS
    )


def _prediction(seed: int, protocol: str, follow: str) -> dict:
    expected = EXPECTED_IMAGE_SIDE[seed]
    text = "right" if expected == "left" else "left"
    side = expected if follow == "image" else text if follow == "text" else "uncertain"
    return {
        "seed": seed,
        "protocol": protocol,
        "guidance_consistency": "conflicting",
        "final_left_tactile_state": "clear_change",
        "final_right_tactile_state": "little_or_no_change",
        "final_changed_side": side,
        "final_evidence_basis": "image" if follow == "image" else "structured_guidance",
        "evidence_summary": "The evidence conflicts.",
        "uncertainty": "Low.",
    }


def _references() -> tuple[dict, dict]:
    neutral = {
        "protocol_metrics": {
            "simultaneous_fusion": {"final_follow_image_rate": {"followed": 0}},
            "image_first_no_commit": {"final_follow_image_rate": {"followed": 2}},
        },
        "image_first_with_commit": {"provisional_image_accuracy": {"correct": 3}},
    }
    conflict = {
        "swapped_final_follow_image_rate": {"followed": 3},
        "swapped_final_follow_text_rate": {"followed": 0},
        "swapped_conflict_detection_rate": {"detected": 3},
    }
    return neutral, conflict


@pytest.mark.parametrize(
    ("sim_follow", "ordered_follow", "candidate"),
    [
        ("image", "image", "conflict_aware_instruction_without_staged_ordering"),
        ("text", "image", "image_first_ordering_plus_conflict_aware_instruction"),
        ("text", "text", "image_first_commitment_plus_conflict_aware_instruction"),
    ],
)
def test_minimal_skill_candidate_priority(
    sim_follow: str, ordered_follow: str, candidate: str
) -> None:
    predictions = [
        _prediction(seed, protocol, sim_follow if protocol == PROTOCOLS[0] else ordered_follow)
        for seed in SEEDS
        for protocol in PROTOCOLS
    ]
    neutral, conflict = _references()
    summary = summarize_predictions(predictions, neutral, conflict)
    assert summary["minimal_skill_candidate"] == candidate
    assert summary["classification"] == (
        "tactile_conflict_instruction_interaction_ablation_completed"
    )
