"""Persistent human-interaction records for paused parallel episodes."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from adapter.protocol import JsonDict
from agent.runtime.episode import (
    DEFAULT_EPISODE_TIMEOUT_S,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MAX_TOTAL_TOKENS,
    EpisodeResult,
)


DEFAULT_INTERACTION_ROOT = Path(".openeta_memory") / "parallel_interactions"


@dataclass(slots=True)
class PausedEpisodeRecord:
    batch_id: str
    episode_id: str
    session_id: str
    interaction_id: str
    question: str
    task: str
    env_id: str
    seed: int
    max_turns: int
    turn_index: int
    memory_root: str
    artifact_root: str
    workspace_root: str = ""
    skills_root: str = ""
    sandbox_root: str = ""
    supervision_profile: str = "standard"
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    timeout_s: float = DEFAULT_EPISODE_TIMEOUT_S
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS
    tool_call_count: int = 0
    total_tokens: int = 0
    token_usage_sources: JsonDict = field(default_factory=dict)
    resume_mode: str = "restart_environment"
    human_intervention_count: int = 0
    interaction_history: list[JsonDict] = field(default_factory=list)
    created_at_s: float = field(default_factory=time.time)

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": "openeta.paused_episode.v2",
            "batch_id": self.batch_id,
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "interaction_id": self.interaction_id,
            "question": self.question,
            "task": self.task,
            "env_id": self.env_id,
            "seed": self.seed,
            "max_turns": self.max_turns,
            "turn_index": self.turn_index,
            "memory_root": self.memory_root,
            "artifact_root": self.artifact_root,
            "workspace_root": self.workspace_root,
            "skills_root": self.skills_root,
            "sandbox_root": self.sandbox_root,
            "supervision_profile": self.supervision_profile,
            "max_tool_calls": self.max_tool_calls,
            "timeout_s": self.timeout_s,
            "max_total_tokens": self.max_total_tokens,
            "tool_call_count": self.tool_call_count,
            "total_tokens": self.total_tokens,
            "token_usage_sources": self.token_usage_sources,
            "resume_mode": self.resume_mode,
            "human_intervention_count": self.human_intervention_count,
            "interaction_history": self.interaction_history,
            "created_at_s": self.created_at_s,
        }

    @classmethod
    def from_dict(cls, payload: JsonDict) -> "PausedEpisodeRecord":
        resume_mode = str(payload.get("resume_mode") or "restart_environment")
        if resume_mode != "restart_environment":
            raise ValueError(f"unsupported paused episode resume_mode: {resume_mode}")
        return cls(
            batch_id=str(payload["batch_id"]),
            episode_id=str(payload["episode_id"]),
            session_id=str(payload["session_id"]),
            interaction_id=str(payload["interaction_id"]),
            question=str(payload.get("question") or ""),
            task=str(payload["task"]),
            env_id=str(payload["env_id"]),
            seed=int(payload.get("seed", 0)),
            max_turns=int(payload["max_turns"]),
            turn_index=int(payload.get("turn_index", 0)),
            memory_root=str(payload["memory_root"]),
            artifact_root=str(payload["artifact_root"]),
            workspace_root=str(payload.get("workspace_root") or ""),
            skills_root=str(payload.get("skills_root") or ""),
            sandbox_root=str(payload.get("sandbox_root") or ""),
            supervision_profile=str(payload.get("supervision_profile") or "standard"),
            max_tool_calls=int(payload.get("max_tool_calls", DEFAULT_MAX_TOOL_CALLS)),
            timeout_s=float(payload.get("timeout_s", DEFAULT_EPISODE_TIMEOUT_S)),
            max_total_tokens=int(
                payload.get("max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS)
            ),
            tool_call_count=int(payload.get("tool_call_count", 0)),
            total_tokens=int(payload.get("total_tokens", 0)),
            token_usage_sources=dict(payload.get("token_usage_sources") or {}),
            resume_mode=resume_mode,
            human_intervention_count=int(payload.get("human_intervention_count", 0)),
            interaction_history=list(payload.get("interaction_history") or []),
            created_at_s=float(payload.get("created_at_s") or time.time()),
        )


class PausedEpisodeStore:
    def __init__(self, root: str | Path = DEFAULT_INTERACTION_ROOT) -> None:
        self.root = Path(root)

    def save(self, record: PausedEpisodeRecord) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(record.session_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def load(self, session_id: str) -> PausedEpisodeRecord:
        path = self.path_for(session_id)
        if not path.exists():
            raise FileNotFoundError(f"paused session not found: {session_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"invalid paused session record: {session_id}")
        return PausedEpisodeRecord.from_dict(payload)

    def delete(self, session_id: str) -> None:
        self.path_for(session_id).unlink(missing_ok=True)

    def path_for(self, session_id: str) -> Path:
        safe = "".join(ch for ch in session_id if ch.isalnum() or ch in "-_")
        if not safe:
            raise ValueError("session_id has no safe path characters")
        return self.root / f"{safe}.json"


def question_from_episode(result: EpisodeResult) -> str:
    if not result.steps:
        return "Human guidance is required."
    request = result.steps[-1].action.command.get("request")
    parameters = request.get("parameters") if isinstance(request, dict) else {}
    if not isinstance(parameters, dict):
        parameters = {}
    return str(parameters.get("question") or parameters.get("message") or "").strip() or (
        "Human guidance is required."
    )


def new_interaction_id() -> str:
    return f"interaction-{uuid4().hex[:16]}"
