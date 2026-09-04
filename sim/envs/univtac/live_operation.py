"""Contracts for the first bounded live UniVTAC Codex operation episode."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sim.envs.univtac.contract import UniVTACContractError
from sim.envs.univtac.native_operation import EXPERT_SEGMENTS, TASK_INSTRUCTION, TASK_NAME

LIVE_SEED = 1_000_000
MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "medium"
MCP_TOOL_NAMES = ("observe", "execute_skill", "finish_episode")
MAX_WORLD_CHANGING_SKILLS = 3

CODEX_OPERATION_PROMPT = """You control one live UniVTAC Pull Out Key episode through a
bounded manipulation-skill library.

First call `observe`.

Choose only from the skills exposed by `execute_skill`.
After every executed skill, inspect the newly returned camera
and tactile observations before selecting the next skill.

Use at most three world-changing skill calls.
Call `finish_episode` when you believe the task is complete,
or when no appropriate skill remains.

Do not invent joint values, poses, displacements, or low-level
trajectories.

Native task success is evaluated privately after the episode.
"""


def validate_live_operation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "task",
        "seed",
        "task_config",
        "mode",
        "device",
        "task_instruction",
        "model",
        "reasoning_effort",
        "skills",
        "max_world_changing_skills",
        "worker_timeout_seconds",
        "codex_timeout_seconds",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise UniVTACContractError(f"live operation config is missing: {missing}")
    expected = {
        "task": TASK_NAME,
        "seed": LIVE_SEED,
        "task_config": "demo",
        "mode": "collect",
        "device": "cuda:0",
        "task_instruction": TASK_INSTRUCTION,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "skills": list(EXPERT_SEGMENTS),
        "max_world_changing_skills": MAX_WORLD_CHANGING_SKILLS,
    }
    for key, value in expected.items():
        if config[key] != value:
            raise UniVTACContractError(
                f"live operation requires {key}={value!r}, got {config[key]!r}"
            )
    validated = dict(config)
    for key in ("worker_timeout_seconds", "codex_timeout_seconds"):
        value = float(config[key])
        if value <= 0:
            raise UniVTACContractError(f"{key} must be positive")
        validated[key] = value
    return validated


def validate_agent_attempt_boundary(
    output_root: Path, requested_attempt: int
) -> list[dict[str, int]]:
    """Allow only the next attempt, and never one after physical motion began."""
    if requested_attempt < 1:
        raise UniVTACContractError("Agent attempt must be positive")
    attempts: list[dict[str, int]] = []
    for episode_path in (output_root / "agent").rglob("episode.json"):
        payload = json.loads(episode_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise UniVTACContractError(f"invalid Agent episode: {episode_path}")
        attempt = int(payload.get("attempt", 1))
        action_count = int(payload.get("agent_action_count", 0))
        trace_path = episode_path.parent / "action_trace.jsonl"
        if trace_path.is_file() and trace_path.read_text(encoding="utf-8").strip():
            action_count = max(action_count, 1)
        attempts.append({"attempt": attempt, "action_count": action_count})
    attempts.sort(key=lambda row: row["attempt"])
    if any(row["action_count"] > 0 for row in attempts):
        raise UniVTACContractError(
            "an Agent world-changing action already ran; further attempts are forbidden"
        )
    expected_attempt = attempts[-1]["attempt"] + 1 if attempts else 1
    if requested_attempt != expected_attempt:
        raise UniVTACContractError(
            f"Agent attempt must be the next fresh attempt {expected_attempt}, "
            f"got {requested_attempt}"
        )
    return attempts


@dataclass
class PullOutKeySkillBudget:
    """Reserve each bounded world-changing skill at most once."""

    executed: list[str] = field(default_factory=list)
    max_calls: int = MAX_WORLD_CHANGING_SKILLS

    @property
    def remaining(self) -> list[str]:
        return [name for name in EXPERT_SEGMENTS if name not in self.executed]

    def reserve(self, skill: str) -> int:
        if skill not in EXPERT_SEGMENTS:
            raise UniVTACContractError(f"unknown Pull Out Key skill: {skill!r}")
        if skill in self.executed:
            raise UniVTACContractError(f"Pull Out Key skill already executed: {skill}")
        if len(self.executed) >= self.max_calls:
            raise UniVTACContractError("world-changing skill budget is exhausted")
        self.executed.append(skill)
        return len(self.executed)


def build_live_codex_command(
    *,
    codex_bin: str,
    workspace: Path,
    repo_root: Path,
    episode_root: Path,
    worker_url: str,
    final_response_path: Path,
    model: str = MODEL,
    reasoning_effort: str = REASONING_EFFORT,
) -> list[str]:
    mcp_args = [
        f"PYTHONPATH={repo_root}",
        "uv",
        "run",
        "--no-project",
        "--python",
        str(repo_root / ".venv/bin/python"),
        "-m",
        "tools.univtac_operation_mcp_server",
        "--episode-root",
        str(episode_root),
        "--worker-url",
        worker_url,
    ]
    config = [
        "features.memories=false",
        "features.enable_request_compression=false",
        "memories.use_memories=false",
        "memories.generate_memories=false",
        'history.persistence="none"',
        f"model_reasoning_effort={json.dumps(reasoning_effort)}",
        'mcp_servers.univtac.command="env"',
        f"mcp_servers.univtac.args={json.dumps(mcp_args)}",
        "mcp_servers.univtac.required=true",
        f"mcp_servers.univtac.enabled_tools={json.dumps(list(MCP_TOOL_NAMES))}",
        'mcp_servers.univtac.default_tools_approval_mode="approve"',
        "mcp_servers.univtac.tool_timeout_sec=600",
    ]
    command = [
        codex_bin,
        "-m",
        model,
        "exec",
        "-C",
        str(workspace),
        "-s",
        "read-only",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--json",
        "--output-last-message",
        str(final_response_path),
    ]
    for value in config:
        command.extend(("-c", value))
    command.append(CODEX_OPERATION_PROMPT)
    return command


def validate_worker_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "task_instruction",
        "visible_phase",
        "step_identifiers",
        "proprio",
        "images",
        "skill_execution",
        "remaining_skills",
    }
    if set(payload) != allowed:
        raise UniVTACContractError(
            f"live worker projection fields differ from allowlist: {sorted(payload)}"
        )
    images = payload["images"]
    labels = [item.get("label") for item in images] if isinstance(images, list) else []
    expected_labels = [
        "camera/head/rgb",
        "camera/wrist/rgb",
        "tactile/left_tactile/rgb_marker",
        "tactile/right_tactile/rgb_marker",
    ]
    if labels != expected_labels:
        raise UniVTACContractError(f"live worker image labels are invalid: {labels}")
    forbidden = {
        "actor_state",
        "press_depth",
        "depth",
        "plan_success",
        "check_success",
        "reward",
        "target_pose",
        "over_rotate",
        "key_rotation",
    }

    def walk(value: Any):
        if isinstance(value, Mapping):
            for key, child in value.items():
                yield str(key)
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)

    leaked = sorted(forbidden & set(walk(payload)))
    if leaked:
        raise UniVTACContractError(f"privileged live worker fields leaked: {leaked}")
    return dict(payload)
