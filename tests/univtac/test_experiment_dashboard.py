from __future__ import annotations

import json
from pathlib import Path

from scripts.univtac.serve_experiment_dashboard import (
    DETAIL_HTML,
    build_timeline,
    discover_runs,
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
