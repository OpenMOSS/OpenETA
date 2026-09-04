from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from sim.envs.univtac.codex_readonly import build_codex_exec_command
from sim.envs.univtac.staged_tactile_grounding import (
    CONDITION_ORDER,
    EXPECTED_CHANGE_SIDE,
    PILOT_PROMPT,
    SEEDS,
    load_staged_trace,
    parse_final_prediction,
    stage_condition,
    summarize_predictions,
    validate_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _image_judgment(side: str) -> dict[str, str]:
    return {
        "tactile_image_access": "available",
        "left_tactile_state": "clear_change" if side == "left" else "little_or_no_change",
        "right_tactile_state": "clear_change" if side == "right" else "little_or_no_change",
        "tactile_changed_side": side,
        "left_visual_cue": "localized_colored_disturbance" if side == "left" else "regular_grid",
        "right_visual_cue": "localized_colored_disturbance" if side == "right" else "regular_grid",
        "evidence_summary": f"The {side} image has the clearer disturbance.",
    }


def _final(side: str, consistency: str) -> dict[str, str]:
    return {
        "guidance_consistency": consistency,
        "final_left_tactile_state": "clear_change" if side == "left" else "little_or_no_change",
        "final_right_tactile_state": "clear_change" if side == "right" else "little_or_no_change",
        "final_changed_side": side,
        "final_evidence_basis": "both",
        "evidence_summary": "The final judgment compares both evidence sources.",
        "uncertainty": "This does not measure force.",
    }


def test_config_prompt_command_and_zero_simulator_runner() -> None:
    path = REPO_ROOT / "configs/univtac/pull_out_key_staged_grounding_pilot.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert validate_config(payload)["seeds"] == list(SEEDS)
    assert payload["condition_order"] == {str(key): list(value) for key, value in CONDITION_ORDER.items()}
    assert "record_image_judgment" in PILOT_PROMPT
    command = build_codex_exec_command(
        codex_bin="codex",
        model="gpt-test",
        reasoning_effort="medium",
        workspace=Path("/tmp/workspace"),
        repo_root=REPO_ROOT,
        episode_root=Path("/tmp/episode"),
        snapshot_path=Path("/tmp/snapshot.json"),
        final_response_path=Path("/tmp/final.md"),
        condition_manifest=Path("/tmp/condition.json"),
        mcp_mode="staged",
    )
    assert "--mode" in " ".join(command) and "staged" in " ".join(command)
    runner = (REPO_ROOT / "scripts/univtac/run_staged_tactile_grounding_pilot.py").read_text()
    assert "run_pull_out_key_gate" not in runner and "AppLauncher" not in runner
    assert "play_once" not in runner and "check_success" not in runner
    assert '"simulator_invocation_count": 0' in runner


def test_stage_reuses_same_four_images_and_changes_only_guidance(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source_images = source / "runs/seed_1000000/image_only/images"
    source_images.mkdir(parents=True)
    names = (
        "head_rgb.png",
        "wrist_rgb.png",
        "left_tactile_rgb_marker.png",
        "right_tactile_rgb_marker.png",
    )
    for index, name in enumerate(names):
        Image.fromarray(np.full((4, 5, 3), index, dtype=np.uint8)).save(source_images / name)
    scalar = {
        "active_pixel_ratio": 0.2,
        "mean_absolute_change": 18.0,
        "p95_absolute_change": 103.0,
        "salient_mass": 1.8,
    }
    mapping = {
        "image_source_mapping": {"left_tactile": "current", "right_tactile": "baseline"},
        "expected_image_change_side": "left",
        "correct_structured_summary": {
            "left_tactile": scalar,
            "right_tactile": {key: 0.0 for key in scalar},
        },
        "swapped_structured_summary": {
            "left_tactile": {key: 0.0 for key in scalar},
            "right_tactile": scalar,
        },
    }
    manifests = {}
    arrays = {}
    for condition in ("correct_guidance", "swapped_guidance"):
        episode = tmp_path / condition
        episode.mkdir()
        manifest = stage_condition(
            seed=1_000_000,
            condition=condition,
            source_root=source,
            source_mapping=mapping,
            episode_root=episode,
            simulator_root=tmp_path,
        )
        manifests[condition] = manifest
        arrays[condition] = [np.asarray(Image.open(episode / item["path"])) for item in manifest["images"]]
    assert len(manifests["correct_guidance"]["images"]) == 4
    for index in range(4):
        assert np.array_equal(arrays["correct_guidance"][index], arrays["swapped_guidance"][index])
    assert manifests["correct_guidance"]["model_visible_structured_tactile_summary"]["left_tactile"] == scalar
    assert manifests["swapped_guidance"]["model_visible_structured_tactile_summary"]["right_tactile"] == scalar


def test_trace_parser_and_final_parser_enforce_three_stages(tmp_path: Path) -> None:
    judgment = _image_judgment("left")
    image_text = json.dumps(
        {
            "images": [
                {"label": "camera/head/rgb"},
                {"label": "camera/wrist/rgb"},
                {"label": "tactile/left_tactile/rgb_marker"},
                {"label": "tactile/right_tactile/rgb_marker"},
            ]
        }
    )
    rows = [
        {
            "seq": 1,
            "tool": "observe_images",
            "arguments": {},
            "response_text_blocks": [image_text],
            "response_image_paths": ["a", "b", "c", "d"],
        },
        {
            "seq": 2,
            "tool": "record_image_judgment",
            "arguments": judgment,
            "response_text_blocks": [json.dumps({"status": "image_judgment_committed"})],
            "response_image_paths": [],
        },
        {
            "seq": 3,
            "tool": "observe_structured_guidance",
            "arguments": {},
            "response_text_blocks": [json.dumps({"tactile_change_summary": {}})],
            "response_image_paths": [],
        },
    ]
    path = tmp_path / "operator_context.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    assert load_staged_trace(path) == rows
    parsed = parse_final_prediction(json.dumps(_final("left", "consistent")))
    assert parsed["final_changed_side"] == "left"


def test_grounding_metrics_classify_image_grounded_conflict_resistance() -> None:
    image_rows = []
    final_rows = []
    for seed in SEEDS:
        expected = EXPECTED_CHANGE_SIDE[seed]
        for condition in ("correct_guidance", "swapped_guidance"):
            image_rows.append({"seed": seed, "condition": condition, **_image_judgment(expected)})
            final_rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    **_final(expected, "consistent" if condition == "correct_guidance" else "conflicting"),
                }
            )
    r0917 = {
        "swapped_follow_image_rate": {"followed": 0},
        "swapped_follow_text_rate": {"followed": 3},
    }
    summary = summarize_predictions(image_rows, final_rows, r0917)
    assert summary["image_stage_access_rate"]["available"] == 6
    assert summary["image_stage_changed_side_accuracy"]["correct"] == 6
    assert summary["image_stage_repeat_consistency"]["consistent"] == 3
    assert summary["swapped_conflict_detection_rate"]["detected"] == 3
    assert summary["swapped_final_follow_image_rate"]["followed"] == 3
    assert summary["swapped_final_follow_text_rate"]["followed"] == 0
    assert summary["grounding_signal"] == "grounding_improves_conflict_resistance"
