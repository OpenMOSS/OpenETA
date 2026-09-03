from __future__ import annotations

import json
from pathlib import Path

from scripts.univtac.serve_experiment_dashboard import (
    DETAIL_HTML,
    PILOT_HTML,
    PROGRESS_HTML,
    build_timeline,
    discover_pilots,
    discover_runs,
    load_pilot_detail,
    load_project_progress,
    load_run_detail,
)


def _episode(tmp_path: Path) -> Path:
    root = tmp_path / "outputs/r0913"
    root.mkdir(parents=True)
    episode = {
        "round": "R0.9.13",
        "task": "pull_out_key",
        "seed": 1_000_000,
        "model": "gpt-test",
        "status": "completed",
        "started_at": "2026-09-03T00:00:00Z",
        "ended_at": "2026-09-03T00:00:02Z",
        "duration_seconds": 2.0,
        "tool_call_count": 1,
    }
    (root / "episode.json").write_text(json.dumps(episode), encoding="utf-8")
    text = json.dumps(
        {
            "task_instruction": "Pull the key out of the slot.",
            "step_identifiers": {"phase": "pre_action"},
            "proprio": {"joint": [], "ee": []},
            "images": [
                {"label": label, "shape": [1, 1, 3], "dtype": "uint8"}
                for label in (
                    "camera/head/rgb",
                    "camera/wrist/rgb",
                    "tactile/left_tactile/rgb_marker",
                    "tactile/right_tactile/rgb_marker",
                )
            ],
        }
    )
    row = {
        "seq": 1,
        "timestamp_s": 1.0,
        "tool": "observe",
        "arguments": {},
        "response_text_blocks": [text],
        "response_image_paths": ["a.png", "b.png", "c.png", "d.png"],
    }
    (root / "operator_context.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (root / "codex_exec.jsonl").write_text(
        json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "final"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "agent_final.md").write_text("final", encoding="utf-8")
    return root


def test_dashboard_discovers_run_and_agent_saw_comes_from_operator_trace(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    runs = discover_runs(tmp_path / "outputs")
    assert runs[0]["directory"] == "r0913"
    assert runs[0]["observe_count"] == 1
    detail = load_run_detail(episode)
    assert detail["operator_context"][0]["response_image_paths"] == [
        "a.png",
        "b.png",
        "c.png",
        "d.png",
    ]
    assert detail["agent_final"] == "final"
    assert "ACTUAL MCP CONTEXT SEEN BY CODEX" in DETAIL_HTML
    assert r"join('\n')" in DETAIL_HTML


def test_timeline_contains_codex_observe_and_completion_events() -> None:
    episode = {"codex_started_at": "start", "codex_ended_at": "end"}
    codex = [
        {
            "type": "item.started",
            "item": {"type": "mcp_tool_call", "tool": "observe", "arguments": {}},
        }
    ]
    operator = [
        {
            "tool": "observe",
            "arguments": {},
            "timestamp_s": 1.0,
            "response_text_blocks": ["{}"],
            "response_image_paths": ["a", "b", "c", "d"],
        }
    ]
    kinds = [row["event_type"] for row in build_timeline(episode, codex, operator)]
    assert kinds == [
        "Codex started",
        "observe tool requested",
        "observe tool returned",
        "Codex completed",
    ]


def test_dashboard_has_no_control_post_route() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/univtac/serve_experiment_dashboard.py").read_text()
    assert "def do_POST" not in source
    assert "move_to" not in source and "set_gripper" not in source


def test_causal_pilot_dashboard_loads_three_by_three_actual_context(tmp_path: Path) -> None:
    root = tmp_path / "outputs/r0914"
    (root / "runs").mkdir(parents=True)
    (root / "pilot.json").write_text(
        json.dumps({"round": "R0.9.14", "model": "gpt-5.6-terra"}), encoding="utf-8"
    )
    (root / "summary.json").write_text(
        json.dumps({"pilot_signal": "mixed", "completed_call_count": 9, "metrics": {}}),
        encoding="utf-8",
    )
    (root / "host_reference.json").write_text("{}", encoding="utf-8")
    for seed in (1_000_000, 1_000_001, 1_000_002):
        for condition in ("visual_only", "correct_tactile", "swapped_tactile"):
            run = root / "runs" / f"seed_{seed}" / condition
            run.mkdir(parents=True)
            (run / "episode.json").write_text(
                json.dumps(
                    {
                        "round": "R0.9.14",
                        "seed": seed,
                        "condition": condition,
                        "tool_call_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            paths = ["images/head.png", "images/wrist.png"]
            if condition != "visual_only":
                paths += ["images/left.png", "images/right.png"]
            (run / "operator_context.jsonl").write_text(
                json.dumps({"tool": "observe", "response_image_paths": paths}) + "\n",
                encoding="utf-8",
            )
            (run / "agent_final.md").write_text("{}", encoding="utf-8")
            (run / "condition.json").write_text(
                json.dumps({"condition": condition}), encoding="utf-8"
            )
    pilots = discover_pilots(tmp_path / "outputs")
    assert pilots == [{"directory": "r0914", "signal": "mixed", "completed_call_count": 9}]
    detail = load_pilot_detail(root)
    assert len(detail["cells"]) == 9
    assert [cell["condition"] for cell in detail["cells"][:3]] == [
        "visual_only",
        "correct_tactile",
        "swapped_tactile",
    ]
    assert sum(len(cell["image_paths"]) for cell in detail["cells"]) == 30
    assert "HOST-ONLY EVALUATION — NOT SHOWN TO CODEX" in PILOT_HTML
    assert "LEFT/RIGHT TACTILE ASSIGNMENT SWAPPED" in PILOT_HTML


def test_project_progress_uses_human_summaries_and_links_pilots(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    progress = root / "univtac-project-dashboard/progress.jsonl"
    progress.parent.mkdir(parents=True)
    row = {
        "round": "R0.9.13",
        "status": "completed",
        "pro_instruction_summary": "接通只读触觉观察。",
        "codex_work_summary": "调用一次 observe 并展示四张图。",
        "result_summary": "真实 Codex 完成观察。",
    }
    progress.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    payload = load_project_progress(root)
    assert payload["entries"] == [row]
    assert payload["pilots"] == []
    assert "Pro 想弄清楚什么" in PROGRESS_HTML
    assert "我们怎么验证的" in PROGRESS_HTML
    assert "实际看到了什么" in PROGRESS_HTML
    assert "这说明什么" in PROGRESS_HTML
    assert "一句话结论" in PROGRESS_HTML
    assert "技术证据（commit、测试、产物）" in PROGRESS_HTML
    assert "x.pro_question||x.pro_instruction_summary" in PROGRESS_HTML


def test_difference_pilot_dashboard_orders_conditions_and_builds_pair_viewer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "outputs/r0915"
    (root / "runs").mkdir(parents=True)
    (root / "pilot.json").write_text(
        json.dumps({"round": "R0.9.15", "model": "gpt-5.6-terra"}), encoding="utf-8"
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "difference_signal": "mixed_difference_signal",
                "completed_semantic_trials": 9,
            }
        ),
        encoding="utf-8",
    )
    (root / "difference_metrics.json").write_text("{}", encoding="utf-8")
    for seed in (1_000_000, 1_000_001, 1_000_002):
        for condition in ("raw_pair", "explicit_difference", "swapped_difference"):
            run = root / "runs" / f"seed_{seed}" / condition
            (run / "images").mkdir(parents=True)
            (run / "episode.json").write_text(
                json.dumps({"round": "R0.9.15", "seed": seed, "condition": condition}),
                encoding="utf-8",
            )
            (run / "operator_context.jsonl").write_text(
                json.dumps({"tool": "observe", "response_image_paths": []}) + "\n",
                encoding="utf-8",
            )
            (run / "condition.json").write_text("{}", encoding="utf-8")
    detail = load_pilot_detail(root)
    assert [cell["condition"] for cell in detail["cells"][:3]] == [
        "raw_pair",
        "explicit_difference",
        "swapped_difference",
    ]
    assert len(detail["pairs"]) == 6
    assert [image["label"] for image in detail["pairs"][0]["images"]] == [
        "Baseline",
        "Current",
        "Difference",
    ]


def test_structured_pilot_dashboard_preserves_actual_text_and_host_references(
    tmp_path: Path,
) -> None:
    root = tmp_path / "outputs/r0916"
    (root / "runs").mkdir(parents=True)
    (root / "pilot.json").write_text(
        json.dumps({"round": "R0.9.16", "model": "gpt-5.6-terra"}), encoding="utf-8"
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "structured_signal": "text_sensitive_without_accuracy_gain",
                "completed_semantic_trials": 9,
            }
        ),
        encoding="utf-8",
    )
    reference = {
        "seeds": {
            "1000000": {
                "seed": 1000000,
                "sensors": {
                    "left_tactile": {
                        "structured_metrics": {
                            "reference_region": "center",
                            "normalized_horizontal_saliency": [0.2, 0.6, 0.2],
                        },
                        "old_global_centroid_region": "center",
                        "old_and_new_region_agree": True,
                    }
                },
            }
        }
    }
    (root / "structured_reference.json").write_text(
        json.dumps(reference), encoding="utf-8"
    )
    conditions = (
        "difference_only",
        "correct_structured_guidance",
        "swapped_structured_guidance",
    )
    summary = {
        "vector_order": ["left", "center", "right"],
        "left_tactile": {"salient_mass": 1.8},
        "right_tactile": {"salient_mass": 1.9},
    }
    for seed in (1_000_000, 1_000_001, 1_000_002):
        for condition in conditions:
            run = root / "runs" / f"seed_{seed}" / condition
            run.mkdir(parents=True)
            (run / "episode.json").write_text(
                json.dumps({"round": "R0.9.16", "seed": seed, "condition": condition}),
                encoding="utf-8",
            )
            text_payload = {"images": []}
            if condition != "difference_only":
                text_payload["tactile_change_summary"] = summary
            (run / "operator_context.jsonl").write_text(
                json.dumps(
                    {
                        "tool": "observe",
                        "response_text_blocks": [json.dumps(text_payload)],
                        "response_image_paths": [f"images/{index}.png" for index in range(6)],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run / "agent_final.md").write_text("{}", encoding="utf-8")
            (run / "condition.json").write_text("{}", encoding="utf-8")
    pilots = discover_pilots(tmp_path / "outputs")
    assert pilots == [
        {
            "directory": "r0916",
            "signal": "text_sensitive_without_accuracy_gain",
            "completed_call_count": 9,
        }
    ]
    detail = load_pilot_detail(root)
    assert len(detail["cells"]) == 9
    assert [cell["condition"] for cell in detail["cells"][:3]] == list(conditions)
    assert all(len(cell["image_paths"]) == 6 for cell in detail["cells"])
    assert detail["cells"][0]["structured_summary"] is None
    assert detail["cells"][1]["structured_summary"] == summary
    assert detail["structured_reference"] == reference
    assert "Actual structured text seen by Codex" in PILOT_HTML
    assert "old centroid" in PILOT_HTML
    assert "LEFT/RIGHT DIFFERENCE MAP ASSIGNMENT SWAPPED" in PILOT_HTML
    assert 'class="pair-images"' in PILOT_HTML
