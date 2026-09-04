from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

from sim.envs.univtac.live_operation import (
    CODEX_OPERATION_PROMPT,
    MCP_TOOL_NAMES,
    PullOutKeySkillBudget,
    build_live_codex_command,
    validate_agent_attempt_boundary,
    validate_live_operation_config,
    validate_worker_projection,
)
from tools.univtac_operation_mcp_server import build_server

REPO_ROOT = Path(__file__).resolve().parents[2]


def _projection() -> dict:
    return {
        "task_instruction": "Pull the key out of the slot.",
        "visible_phase": "pre_action",
        "step_identifiers": {
            "snapshot_id": "snapshot",
            "action_id": "action",
            "phase": "pre_action",
            "simulator_step": 1,
            "take_action_count": 0,
        },
        "proprio": {"joint": [0.0] * 9, "ee": [0.0] * 7},
        "images": [
            {"label": "camera/head/rgb", "path": "head.png", "shape": [1, 1, 3], "dtype": "uint8"},
            {"label": "camera/wrist/rgb", "path": "wrist.png", "shape": [1, 1, 3], "dtype": "uint8"},
            {
                "label": "tactile/left_tactile/rgb_marker",
                "path": "left.png",
                "shape": [1, 1, 3],
                "dtype": "uint8",
            },
            {
                "label": "tactile/right_tactile/rgb_marker",
                "path": "right.png",
                "shape": [1, 1, 3],
                "dtype": "uint8",
            },
        ],
        "skill_execution": {"skill": None, "status": "not_started"},
        "remaining_skills": ["align_key", "settle_alignment", "pull_key_out"],
    }


def test_live_config_and_skill_budget_are_bounded() -> None:
    payload = yaml.safe_load(
        (REPO_ROOT / "configs/univtac/pull_out_key_live_operation.yaml").read_text()
    )
    assert validate_live_operation_config(payload)["max_world_changing_skills"] == 3
    budget = PullOutKeySkillBudget()
    assert budget.reserve("pull_key_out") == 1
    with pytest.raises(ValueError, match="already executed"):
        budget.reserve("pull_key_out")
    with pytest.raises(ValueError, match="unknown"):
        budget.reserve("move_to")
    budget.reserve("align_key")
    budget.reserve("settle_alignment")
    assert budget.remaining == []


def test_worker_projection_contains_only_operator_visible_fields() -> None:
    projection = _projection()
    assert validate_worker_projection(projection)["visible_phase"] == "pre_action"
    leaked = dict(projection)
    leaked["skill_execution"] = {"skill": None, "status": "ok", "check_success": True}
    with pytest.raises(ValueError, match="privileged"):
        validate_worker_projection(leaked)


def test_codex_command_uses_cli_and_only_live_mcp(tmp_path: Path) -> None:
    command = build_live_codex_command(
        codex_bin="codex",
        workspace=tmp_path / "workspace",
        repo_root=REPO_ROOT,
        episode_root=tmp_path / "episode",
        worker_url="http://127.0.0.1:12345",
        final_response_path=tmp_path / "final.md",
    )
    joined = " ".join(command)
    assert command[:4] == ["codex", "-m", "gpt-5.6-terra", "exec"]
    assert "tools.univtac_operation_mcp_server" in joined
    assert "/v1/chat/completions" not in joined
    assert 'default_tools_approval_mode="approve"' in joined
    assert "mcp_servers.univtac.enabled_tools" in joined
    assert "execute_skill" in CODEX_OPERATION_PROMPT
    assert "align_key" not in CODEX_OPERATION_PROMPT


def test_mcp_exposes_exact_three_tools_and_one_skill_argument(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    server = build_server(episode_root=tmp_path, worker_url="http://127.0.0.1:1")
    assert tuple(sorted(server._tool_manager._tools)) == tuple(sorted(MCP_TOOL_NAMES))
    execute = server._tool_manager._tools["execute_skill"].fn
    assert list(inspect.signature(execute).parameters) == ["skill"]


def test_live_worker_never_calls_expert_play_once() -> None:
    source = (
        REPO_ROOT / "scripts/univtac/serve_pull_out_key_live_worker.py"
    ).read_text(encoding="utf-8")
    assert "task.play_once(" not in source
    assert "task.move(actions, **kwargs)" in source
    assert "task.rng.uniform(0.09, 0.16)" in source
    assert "task.delay(post_delay)" in source


def test_agent_retry_is_allowed_only_before_any_world_action(tmp_path: Path) -> None:
    assert validate_agent_attempt_boundary(tmp_path, 1) == []
    first = tmp_path / "agent/seed_1000000"
    first.mkdir(parents=True)
    (first / "episode.json").write_text(
        '{"attempt":1,"agent_action_count":0}', encoding="utf-8"
    )
    assert validate_agent_attempt_boundary(tmp_path, 2) == [
        {"attempt": 1, "action_count": 0}
    ]
    with pytest.raises(ValueError, match="next fresh attempt 2"):
        validate_agent_attempt_boundary(tmp_path, 3)

    (first / "action_trace.jsonl").write_text('{"skill":"align_key"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="world-changing action already ran"):
        validate_agent_attempt_boundary(tmp_path, 2)
