from __future__ import annotations

import json
from pathlib import Path

from scripts.univtac.run_codex_readonly_observation import (
    _final_answer_covers_requested_views,
)
from sim.envs.univtac.codex_readonly import (
    CODEX_OPERATOR_PROMPT,
    build_codex_exec_command,
    load_operator_context,
    summarize_codex_exec,
)


def test_codex_command_uses_exec_readonly_and_only_univtac_mcp(tmp_path: Path) -> None:
    command = build_codex_exec_command(
        codex_bin="codex",
        model="gpt-test",
        reasoning_effort="medium",
        workspace=tmp_path / "workspace",
        repo_root=tmp_path / "repo",
        episode_root=tmp_path / "episode",
        snapshot_path=tmp_path / "episode/simulator/seed/snapshot_pre.json",
        final_response_path=tmp_path / "episode/agent_final.md",
    )
    assert command[0] == "codex"
    assert "exec" in command and "--json" in command
    assert command[command.index("-s") + 1] == "read-only"
    rendered = " ".join(command)
    assert "tools.univtac_operator_mcp_server" in rendered
    assert "mcp_servers.univtac.required=true" in command
    assert "chat/completions" not in rendered
    assert "move_to" not in rendered and "set_gripper" not in rendered
    assert command[-1] == CODEX_OPERATOR_PROMPT


def test_codex_trace_parser_extracts_visible_message_and_usage_only() -> None:
    rows = [
        {"type": "thread.started", "thread_id": "t1"},
        {
            "type": "item.started",
            "item": {"type": "mcp_tool_call", "tool": "observe", "arguments": {}},
        },
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "Visible final answer"},
        },
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 4}},
    ]
    summary = summarize_codex_exec(rows)
    assert summary["visible_assistant_messages"] == ["Visible final answer"]
    assert summary["usage"] == {"input_tokens": 10, "output_tokens": 4}
    assert "reasoning" not in json.dumps(summary).lower()


def test_operator_context_parser_requires_one_text_and_four_images(tmp_path: Path) -> None:
    path = tmp_path / "operator_context.jsonl"
    row = {
        "seq": 1,
        "tool": "observe",
        "arguments": {},
        "response_text_blocks": ["{}"],
        "response_image_paths": ["a.png", "b.png", "c.png", "d.png"],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert load_operator_context(path) == [row]
    row["tool"] = "move_to"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with __import__("pytest").raises(ValueError, match="non-observe"):
        load_operator_context(path)


def test_launcher_source_has_no_http_model_backend_or_action_path() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/univtac/run_codex_readonly_observation.py").read_text()
    assert "chat/completions" not in source
    assert "OpenAICompatible" not in source
    assert "EnvAction" not in source
    assert "play_once" not in source
    assert "check_success" not in source
    assert "no_parallel_runtime" not in source


def test_final_answer_accepts_plain_uncertainty_synonyms() -> None:
    answer = (
        "The head scene and wrist view show the gripper. "
        "The left tactile contact differs from the right tactile contact. "
        "It remains unclear whether the grasp is secure."
    )
    assert _final_answer_covers_requested_views(answer) is True
