from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from sim.envs.univtac.grounding_mechanism_ablation import (
    COMMON_SEMANTIC_PROMPT,
    EXPECTED_IMAGE_SIDE,
    MCP_TOOLS,
    PROTOCOL_ORDER,
    PROTOCOLS,
    SEEDS,
    assert_paired_inputs_equal,
    load_protocol_trace,
    prompt_for,
    stage_protocol,
    summarize_predictions,
    validate_config,
)


def _scalar(left: float, right: float) -> dict:
    fields = (
        "active_pixel_ratio",
        "mean_absolute_change",
        "p95_absolute_change",
        "salient_mass",
    )
    return {
        "left_tactile": {field: left for field in fields},
        "right_tactile": {field: right for field in fields},
    }


def _source_root(tmp_path: Path) -> Path:
    root = tmp_path / "r0918"
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
        run = root / "runs" / f"seed_{seed}" / "swapped_guidance"
        images = run / "images"
        images.mkdir(parents=True)
        records = []
        for index, (label, filename) in enumerate(zip(labels, filenames, strict=True)):
            path = images / filename
            Image.new("RGB", (8, 6), (index + seed % 255, 20, 30)).save(path)
            records.append(
                {
                    "label": label,
                    "path": f"images/{filename}",
                    "shape": [6, 8, 3],
                    "dtype": "uint8",
                    "media_type": "image/png",
                }
            )
        expected = EXPECTED_IMAGE_SIDE[seed]
        summary = _scalar(0.0, 18.0) if expected == "left" else _scalar(18.0, 0.0)
        (run / "condition.json").write_text(
            json.dumps(
                {
                    "simulator_root": str(tmp_path),
                    "images": records,
                    "model_visible_structured_tactile_summary": summary,
                }
            ),
            encoding="utf-8",
        )
    return root


def test_config_latin_square_and_neutral_prompts() -> None:
    payload = {
        "seeds": list(SEEDS),
        "protocol_order": {str(seed): list(PROTOCOL_ORDER[seed]) for seed in SEEDS},
        "model": "gpt-5.6-terra",
        "reasoning_effort": "medium",
    }
    assert validate_config(payload) == payload
    assert sorted(value for order in PROTOCOL_ORDER.values() for value in order) == sorted(
        PROTOCOLS * 3
    )
    banned = (
        "do not automatically overwrite",
        "prioritize the committed image reading",
        "guidance may be noisy",
        "retain the image judgment",
    )
    for protocol in PROTOCOLS:
        prompt = prompt_for(protocol)
        assert prompt.endswith(COMMON_SEMANTIC_PROMPT)
        assert not any(term in prompt.lower() for term in banned)
        assert protocol not in prompt


def test_staged_protocols_share_images_and_swapped_summary(tmp_path: Path) -> None:
    source = _source_root(tmp_path)
    output = tmp_path / "output"
    for seed in SEEDS:
        for protocol in PROTOCOLS:
            episode = output / "runs" / f"seed_{seed}" / protocol
            episode.mkdir(parents=True)
            stage_protocol(
                seed=seed,
                protocol=protocol,
                source_root=source,
                episode_root=episode,
            )
    assert_paired_inputs_equal(output)
    for seed in SEEDS:
        manifests = [
            json.loads(
                (output / "runs" / f"seed_{seed}" / protocol / "condition.json").read_text()
            )
            for protocol in PROTOCOLS
        ]
        assert all(item["text_indicated_side"] != EXPECTED_IMAGE_SIDE[seed] for item in manifests)
        assert all(len(item["images"]) == 4 for item in manifests)


def _trace_rows(protocol: str) -> list[dict]:
    image_payload = {
        "images": [
            {"label": label}
            for label in (
                "camera/head/rgb",
                "camera/wrist/rgb",
                "tactile/left_tactile/rgb_marker",
                "tactile/right_tactile/rgb_marker",
            )
        ]
    }
    guidance = {"tactile_change_summary": _scalar(0.0, 18.0)}
    judgment = {
        "tactile_image_access": "available",
        "left_tactile_state": "clear_change",
        "right_tactile_state": "little_or_no_change",
        "tactile_changed_side": "left",
        "left_visual_cue": "localized_colored_disturbance",
        "right_visual_cue": "regular_grid",
        "evidence_summary": "The left image has a localized disturbance.",
    }
    rows = []
    for index, tool in enumerate(MCP_TOOLS[protocol], 1):
        text = image_payload if index == 1 else guidance
        arguments = judgment if tool == "record_image_judgment" else {}
        if protocol == "simultaneous_fusion":
            text = {**image_payload, **guidance}
        elif tool == "record_image_judgment":
            text = {"status": "image_judgment_committed"}
        rows.append(
            {
                "seq": index,
                "tool": tool,
                "arguments": arguments,
                "response_text_blocks": [json.dumps(text)],
                "response_image_paths": ["a", "b", "c", "d"] if index == 1 else [],
            }
        )
    return rows


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_trace_validation_matches_each_protocol(tmp_path: Path, protocol: str) -> None:
    path = tmp_path / f"{protocol}.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in _trace_rows(protocol)) + "\n",
        encoding="utf-8",
    )
    assert [row["tool"] for row in load_protocol_trace(path, protocol)] == MCP_TOOLS[protocol]


def _final(seed: int, protocol: str, side: str) -> dict:
    return {
        "seed": seed,
        "protocol": protocol,
        "guidance_consistency": "conflicting",
        "final_left_tactile_state": "clear_change",
        "final_right_tactile_state": "little_or_no_change",
        "final_changed_side": side,
        "final_evidence_basis": "image" if side == EXPECTED_IMAGE_SIDE[seed] else "structured_guidance",
        "evidence_summary": "Evidence differs.",
        "uncertainty": "Low.",
    }


def _judgments() -> list[dict]:
    return [
        {
            "seed": seed,
            "protocol": "image_first_with_commit",
            "tactile_image_access": "available",
            "left_tactile_state": "clear_change",
            "right_tactile_state": "little_or_no_change",
            "tactile_changed_side": EXPECTED_IMAGE_SIDE[seed],
            "left_visual_cue": "localized_colored_disturbance",
            "right_visual_cue": "regular_grid",
            "evidence_summary": "Image-only evidence.",
        }
        for seed in SEEDS
    ]


@pytest.mark.parametrize(
    ("image_following_protocol", "expected_result"),
    [
        ("simultaneous_fusion", "simultaneous_fusion_grounded_in_this_run"),
        ("image_first_no_commit", "image_first_ordering_sufficient"),
        ("image_first_with_commit", "explicit_commitment_required"),
        (None, "explicit_conflict_aware_instruction_required"),
    ],
)
def test_mechanism_result_uses_preregistered_priority(
    image_following_protocol: str | None, expected_result: str
) -> None:
    finals = []
    for seed in SEEDS:
        expected = EXPECTED_IMAGE_SIDE[seed]
        opposite = "right" if expected == "left" else "left"
        for protocol in PROTOCOLS:
            side = expected if protocol == image_following_protocol else opposite
            finals.append(_final(seed, protocol, side))
    r0918 = {"swapped_final_follow_image_rate": {"followed": 3}}
    summary = summarize_predictions(finals, _judgments(), r0918)
    assert summary["mechanism_result"] == expected_result
    assert summary["classification"] == "tactile_grounding_mechanism_ablation_completed"
    assert summary["image_first_with_commit"]["provisional_image_accuracy"]["correct"] == 3
