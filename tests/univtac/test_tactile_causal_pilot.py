from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.univtac.run_tactile_causal_pilot import _archive_failed_attempt, _finish_outputs
from sim.envs.univtac.tactile_causal_pilot import (
    CONDITION_ORDER,
    PILOT_PROMPT,
    SEEDS,
    parse_prediction,
    press_depth_reference,
    summarize_pilot,
    validate_pilot_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _prediction(side: str = "left") -> dict[str, str]:
    return {
        "scene_summary": "A gripper and key are visible.",
        "left_contact": "likely_contact",
        "right_contact": "likely_contact",
        "bilateral_contact": "yes",
        "left_marker_region": "left",
        "right_marker_region": "right",
        "stronger_contact_side": side,
        "evidence_source": "visual_and_tactile",
        "evidence_summary": "Both marker images show local patterns.",
        "uncertainty": "Contact strength is only qualitative.",
    }


def test_config_fixes_three_seeds_latin_square_and_model() -> None:
    path = REPO_ROOT / "configs/univtac/pull_out_key_tactile_causal_pilot.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert validate_pilot_config(payload)["seeds"] == list(SEEDS)
    assert payload["condition_order"] == {str(k): list(v) for k, v in CONDITION_ORDER.items()}
    with pytest.raises(ValueError, match="seeds"):
        validate_pilot_config(dict(payload, seeds=[1_000_000, 1_000_001, 1_000_003]))


def test_prediction_parser_accepts_plain_json_or_one_fence() -> None:
    payload = _prediction()
    assert parse_prediction(json.dumps(payload)) == payload
    assert parse_prediction("```json\n" + json.dumps(payload) + "\n```") == payload
    with pytest.raises(ValueError, match="fields"):
        parse_prediction(json.dumps({"scene_summary": "x"}))


def test_prompt_is_fixed_readonly_and_does_not_disclose_condition() -> None:
    lowered = PILOT_PROMPT.lower()
    assert "observe` exactly once" in PILOT_PROMPT
    assert "swapped_tactile" not in lowered and "experimental condition" not in lowered
    assert "do not execute an action" in lowered


def test_runner_uses_codex_exec_and_has_no_action_or_http_backend() -> None:
    source = (REPO_ROOT / "scripts/univtac/run_tactile_causal_pilot.py").read_text()
    assert "build_codex_exec_command" in source
    assert "chat/completions" not in source
    assert "play_once" not in source and "check_success" not in source
    assert "no_parallel_runtime" not in source
    assert "hashlib" not in source
    assert "for seed in SEEDS[1:]" in source
    assert "for condition in CONDITION_ORDER[seed]" in source


def test_resume_archives_failed_trace_but_keeps_staged_condition(tmp_path: Path) -> None:
    root = tmp_path / "condition"
    (root / "images").mkdir(parents=True)
    (root / "condition.json").write_text("{}", encoding="utf-8")
    (root / "episode.json").write_text("{}", encoding="utf-8")
    (root / "codex_exec.jsonl").write_text("failed\n", encoding="utf-8")
    _archive_failed_attempt(root)
    assert (root / "condition.json").is_file()
    assert (root / "images").is_dir()
    assert (root / "attempts/attempt_1/episode.json").is_file()
    assert (root / "attempts/attempt_1/codex_exec.jsonl").read_text() == "failed\n"


def test_incomplete_results_are_not_classified_as_completed(tmp_path: Path) -> None:
    pilot = {"new_simulator_invocation_count": 2, "codex_process_count": 9}
    summary = _finish_outputs(
        output_root=tmp_path,
        predictions=[],
        references={},
        sources=[],
        condition_errors=[{"error": "service unavailable"}],
        pilot=pilot,
    )
    assert summary["classification"] == "pull_out_key_tactile_causal_pilot_incomplete"
    assert summary["pilot_signal"] == "unavailable_incomplete"
    assert pilot["status"] == "incomplete"


def test_press_depth_reference_and_pilot_metrics() -> None:
    contact = {
        "sensors": {
            "left_tactile": {"mean": 2.0, "maximum": 3.0, "positive_pixel_ratio": 0.2},
            "right_tactile": {"mean": 1.0, "maximum": 2.0, "positive_pixel_ratio": 0.1},
        }
    }
    reference = press_depth_reference(contact)
    assert reference["press_depth_dominant_side_proxy"] == "left"
    rows = []
    for seed in SEEDS:
        visual = _prediction("uncertain")
        visual.update(
            {
                "left_contact": "uncertain",
                "right_contact": "uncertain",
                "bilateral_contact": "uncertain",
                "evidence_source": "visual_only",
            }
        )
        correct = _prediction("left")
        swapped = _prediction("right")
        swapped["left_marker_region"] = "right"
        swapped["right_marker_region"] = "left"
        for condition, payload in (
            ("visual_only", visual),
            ("correct_tactile", correct),
            ("swapped_tactile", swapped),
        ):
            rows.append({"seed": seed, "condition": condition, **payload})
    summary = summarize_pilot(rows, {str(seed): reference for seed in SEEDS})
    assert summary["pilot_signal"] == "positive"
    assert summary["metrics"]["swapped_tactile_side_flip_rate"] == 1.0
    assert summary["metrics"]["correct_tactile_contact_alignment"]["aligned"] == 6


def test_balanced_to_balanced_is_not_a_side_flip() -> None:
    reference = {
        "left_contact_candidate": True,
        "right_contact_candidate": True,
        "press_depth_dominant_side_proxy": "right",
    }
    rows = []
    for seed in SEEDS:
        for condition in ("visual_only", "correct_tactile", "swapped_tactile"):
            payload = _prediction("balanced")
            if condition == "visual_only":
                payload["evidence_source"] = "visual_only"
            rows.append({"seed": seed, "condition": condition, **payload})
    summary = summarize_pilot(rows, {str(seed): reference for seed in SEEDS})
    assert summary["metrics"]["swapped_tactile_side_flip_rate"] == 0.0
    assert summary["metrics"]["swapped_tactile_side_flip_count"] == {
        "flipped": 0,
        "evaluated": 3,
    }
