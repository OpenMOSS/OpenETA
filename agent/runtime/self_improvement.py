"""Post-episode self-improvement review hooks for OpenETA."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from adapter.protocol import JsonDict
from agent.runtime.skills import BUILTIN_SKILL_DIR, SkillRegistry


SELF_IMPROVEMENT_SCHEMA_VERSION = "openeta.self_improvement.v1"
DEFAULT_SKILL_REVIEW_ROOT = Path(".openeta_memory") / "skill_reviews" / "pending"
VALID_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class SelfImprovementConfig:
    """Runtime policy for post-episode self-improvement reviews."""

    enabled: bool = True
    min_tool_calls: int = 5
    review_on_positive_reward: bool = True
    review_on_failure: bool = True
    review_on_truncation: bool = True
    proposal_root: Path | str = DEFAULT_SKILL_REVIEW_ROOT


@dataclass(frozen=True, slots=True)
class SkillReviewContext:
    """Bounded context passed to a skill-review subagent."""

    task: str
    session_id: str | None
    summary: JsonDict
    available_skills: tuple[str, ...]
    loaded_skills: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SkillReviewProposal:
    """A proposed skill-library update, staged for human approval."""

    proposal_id: str
    action: str
    skill_name: str
    rationale: str
    suggested_markdown: str
    signals: JsonDict = field(default_factory=dict)
    status: str = "pending"
    created_at_s: float = field(default_factory=time.time)
    schema_version: str = SELF_IMPROVEMENT_SCHEMA_VERSION

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "proposal_id": self.proposal_id,
            "status": self.status,
            "action": self.action,
            "skill_name": self.skill_name,
            "rationale": self.rationale,
            "suggested_markdown": self.suggested_markdown,
            "signals": self.signals,
            "created_at_s": self.created_at_s,
        }


class SkillReviewSubagent(Protocol):
    """Subagent boundary for turning episode traces into skill proposals."""

    def review(self, context: SkillReviewContext) -> list[SkillReviewProposal]:
        """Return staged skill update proposals."""
        ...


class HeuristicSkillReviewSubagent:
    """Deterministic first-pass review agent.

    This intentionally does not rewrite skill files. It stages conservative
    proposals so the TUI/CLI can later show diffs and ask for approval.
    """

    def review(self, context: SkillReviewContext) -> list[SkillReviewProposal]:
        target = _select_target_skill(context)
        if not target:
            return []
        signals = dict(context.summary.get("signals", {}))
        if not _has_real_skill_signal(signals):
            return []
        return [
            SkillReviewProposal(
                proposal_id=f"skill-review-{uuid4().hex[:12]}",
                action="patch",
                skill_name=target,
                rationale=_proposal_rationale(context),
                suggested_markdown=_suggested_markdown(context),
                signals=signals,
            )
        ]


class SkillReviewProposalStore:
    """Filesystem-backed pending proposal store."""

    def __init__(self, root: Path | str = DEFAULT_SKILL_REVIEW_ROOT) -> None:
        self.root = Path(root)

    def save(self, proposal: SkillReviewProposal) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{proposal.proposal_id}.json"
        _atomic_write_json(path, proposal.to_dict())
        return path

    def list(self, *, status: str | None = "pending") -> list[JsonDict]:
        if not self.root.exists():
            return []
        proposals: list[JsonDict] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                proposal = self.load(path.stem)
            except (FileNotFoundError, ValueError):
                continue
            if status is not None and proposal.get("status") != status:
                continue
            proposals.append(proposal)
        proposals.sort(key=lambda item: float(item.get("created_at_s") or 0.0))
        return proposals

    def load(self, proposal_id: str) -> JsonDict:
        path = self._proposal_path(proposal_id)
        if not path.exists():
            raise FileNotFoundError(f"Unknown skill review proposal: {proposal_id}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid skill review proposal JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid skill review proposal payload: {path}")
        payload.setdefault("path", str(path))
        return payload

    def approve(
        self,
        proposal_id: str,
        *,
        reviewer: str = "cli-user",
        skill_dir: Path | str = BUILTIN_SKILL_DIR,
    ) -> JsonDict:
        proposal = self.load(proposal_id)
        if proposal.get("status") != "pending":
            raise ValueError(f"Skill review proposal is not pending: {proposal_id}")
        application = apply_skill_review_proposal(proposal, skill_dir=skill_dir)
        return self._set_status(
            proposal,
            "approved",
            reviewer=reviewer,
            resolution={"application": application},
        )

    def reject(
        self,
        proposal_id: str,
        *,
        reviewer: str = "cli-user",
        reason: str = "",
    ) -> JsonDict:
        proposal = self.load(proposal_id)
        if proposal.get("status") != "pending":
            raise ValueError(f"Skill review proposal is not pending: {proposal_id}")
        resolution: JsonDict = {}
        if reason:
            resolution["reason"] = reason
        return self._set_status(
            proposal,
            "rejected",
            reviewer=reviewer,
            resolution=resolution,
        )

    def _set_status(
        self,
        proposal: JsonDict,
        status: str,
        *,
        reviewer: str,
        resolution: JsonDict,
    ) -> JsonDict:
        path = self._proposal_path(str(proposal["proposal_id"]))
        updated = dict(proposal)
        updated["status"] = status
        updated["resolved_at_s"] = time.time()
        updated["resolved_by"] = reviewer
        updated["resolution"] = resolution
        updated["path"] = str(path)
        _atomic_write_json(path, updated)
        return updated

    def _proposal_path(self, proposal_id: str) -> Path:
        normalized = proposal_id.strip()
        if normalized.endswith(".json"):
            normalized = normalized[:-5]
        if not normalized or "/" in normalized or "\\" in normalized:
            raise ValueError(f"Invalid skill review proposal id: {proposal_id}")
        return self.root / f"{normalized}.json"


class SelfImprovementReviewer:
    """Runs post-episode skill review through a restricted subagent."""

    def __init__(
        self,
        *,
        config: SelfImprovementConfig | None = None,
        subagent: SkillReviewSubagent | None = None,
        store: SkillReviewProposalStore | None = None,
    ) -> None:
        self.config = config or SelfImprovementConfig()
        self.subagent = subagent or HeuristicSkillReviewSubagent()
        self.store = store or SkillReviewProposalStore(self.config.proposal_root)

    def maybe_review(
        self,
        result: Any,
        *,
        skills: SkillRegistry,
    ) -> JsonDict:
        """Run a bounded review if the episode has useful learning signal."""

        summary = summarize_episode_for_review(result)
        trigger = should_review_episode(summary, config=self.config)
        if not trigger["should_review"]:
            return {"reviewed": False, "trigger": trigger, "proposals": []}

        context = SkillReviewContext(
            task=result.task,
            session_id=result.session_id,
            summary=summary,
            available_skills=tuple(skill.name for skill in skills.list()),
            loaded_skills=tuple(summary.get("loaded_skills", ())),
        )
        proposals = self.subagent.review(context)
        saved: list[JsonDict] = []
        for proposal in proposals:
            path = self.store.save(proposal)
            saved.append({"path": str(path), **proposal.to_dict()})
        return {
            "reviewed": True,
            "trigger": trigger,
            "schema_version": SELF_IMPROVEMENT_SCHEMA_VERSION,
            "subagent": type(self.subagent).__name__,
            "proposals": saved,
        }


def summarize_episode_for_review(result: Any) -> JsonDict:
    """Build a compact review context from an episode result."""

    tool_calls: list[JsonDict] = []
    skill_names: list[str] = []
    rewards: list[float] = []
    failure_count = 0
    for step in result.steps:
        step_reward = step.step_result.reward
        if isinstance(step_reward, (int, float)):
            rewards.append(float(step_reward))
        action = step.action.command if isinstance(step.action.command, dict) else {}
        skill_call = action.get("skill_call")
        if isinstance(skill_call, dict):
            name = str(skill_call.get("name", "")).strip()
            if name:
                skill_names.append(name)
        for call in action.get("tool_calls", []) or []:
            if not isinstance(call, dict):
                continue
            compact_call = {
                "name": call.get("name"),
                "status": call.get("status"),
            }
            result_payload = call.get("result")
            if isinstance(result_payload, dict):
                success = result_payload.get("success")
                compact_call["success"] = success
                if success is False:
                    failure_count += 1
            elif call.get("status") == "failed":
                failure_count += 1
            tool_calls.append(compact_call)

    max_reward = max(rewards) if rewards else None
    positive_reward = bool(max_reward is not None and max_reward > 0)
    return {
        "task": result.task,
        "session_id": result.session_id,
        "num_steps": len(result.steps),
        "metadata": result.metadata,
        "loaded_skills": sorted(set(skill_names)),
        "tool_calls": tool_calls,
        "signals": {
            "tool_call_count": len(tool_calls),
            "failure_count": failure_count,
            "positive_reward": positive_reward,
            "max_reward": max_reward,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "stop_reason": result.metadata.get("stop_reason"),
        },
    }


def should_review_episode(
    summary: JsonDict,
    *,
    config: SelfImprovementConfig | None = None,
) -> JsonDict:
    """Return whether the episode should spawn a review subagent."""

    cfg = config or SelfImprovementConfig()
    if not cfg.enabled:
        return {"should_review": False, "reason": "disabled"}
    signals = summary.get("signals", {})
    if not isinstance(signals, dict):
        signals = {}
    tool_count = _int_signal(signals.get("tool_call_count"))
    failure_count = _int_signal(signals.get("failure_count"))
    if tool_count >= cfg.min_tool_calls:
        return {"should_review": True, "reason": "tool_call_threshold"}
    if cfg.review_on_positive_reward and signals.get("positive_reward") is True:
        return {"should_review": True, "reason": "positive_reward"}
    if cfg.review_on_failure and failure_count > 0:
        return {"should_review": True, "reason": "tool_failure"}
    if cfg.review_on_truncation and signals.get("truncated") is True:
        return {"should_review": True, "reason": "episode_truncated"}
    return {"should_review": False, "reason": "no_signal"}


def apply_skill_review_proposal(
    proposal: JsonDict,
    *,
    skill_dir: Path | str = BUILTIN_SKILL_DIR,
) -> JsonDict:
    """Apply one approved skill proposal to `agent/skills`.

    This is intentionally narrow: proposals may only create or append markdown
    under the configured skill directory, and callers should invoke it only
    after explicit human approval.
    """

    action = str(proposal.get("action", "")).strip()
    skill_name = _validated_skill_name(str(proposal.get("skill_name", "")))
    suggested_markdown = str(proposal.get("suggested_markdown", "")).strip()
    if not suggested_markdown:
        raise ValueError("skill review proposal has empty suggested_markdown")
    target = _skill_markdown_path(skill_name, skill_dir=Path(skill_dir))
    proposal_id = str(proposal.get("proposal_id") or "unknown")
    if action == "patch":
        if not target.exists():
            raise FileNotFoundError(f"Cannot patch missing skill markdown: {target}")
        addition = _format_skill_patch(proposal_id, suggested_markdown)
        existing = target.read_text(encoding="utf-8")
        separator = "" if existing.endswith("\n") else "\n"
        target.write_text(f"{existing}{separator}{addition}", encoding="utf-8")
        return {
            "action": action,
            "skill_name": skill_name,
            "target_path": str(target),
            "bytes_written": len(addition.encode("utf-8")),
        }
    if action == "create":
        if target.exists():
            raise FileExistsError(f"Cannot create existing skill markdown: {target}")
        content = _format_created_skill_markdown(proposal, skill_name, suggested_markdown)
        target.write_text(content, encoding="utf-8")
        return {
            "action": action,
            "skill_name": skill_name,
            "target_path": str(target),
            "bytes_written": len(content.encode("utf-8")),
        }
    raise ValueError(f"Unsupported skill review proposal action: {action}")


def _select_target_skill(context: SkillReviewContext) -> str:
    for name in context.loaded_skills:
        if name in context.available_skills:
            return name
    task = context.task.lower()
    for name in context.available_skills:
        pattern = rf"\b{re.escape(name.lower())}\b"
        if re.search(pattern, task):
            return name
    return ""


def _validated_skill_name(skill_name: str) -> str:
    normalized = skill_name.strip().lower()
    if not VALID_SKILL_NAME_RE.fullmatch(normalized):
        raise ValueError(f"Invalid skill name: {skill_name}")
    return normalized


def _skill_markdown_path(skill_name: str, *, skill_dir: Path) -> Path:
    root = skill_dir.resolve()
    target = (root / f"{skill_name}.md").resolve()
    if target.parent != root:
        raise ValueError("skill review target must stay under agent/skills")
    return target


def _format_skill_patch(proposal_id: str, suggested_markdown: str) -> str:
    return (
        "\n"
        f"<!-- openeta-skill-review:{proposal_id} -->\n"
        f"{suggested_markdown.strip()}\n"
    )


def _format_created_skill_markdown(
    proposal: JsonDict,
    skill_name: str,
    suggested_markdown: str,
) -> str:
    description = str(proposal.get("description") or skill_name).strip()
    return (
        "---\n"
        f"name: {skill_name}\n"
        f"description: {description}\n"
        "version: v1\n"
        "editable: true\n"
        "---\n\n"
        "Use this skill as text guidance only. Do not treat it as an executable "
        "macro action.\n\n"
        f"{suggested_markdown.strip()}\n"
    )


def _has_real_skill_signal(signals: JsonDict) -> bool:
    return bool(
        _int_signal(signals.get("tool_call_count")) > 0
        or _int_signal(signals.get("failure_count")) > 0
        or signals.get("positive_reward") is True
        or signals.get("truncated") is True
    )


def _proposal_rationale(context: SkillReviewContext) -> str:
    signals = context.summary.get("signals", {})
    return (
        "Post-episode review found reusable procedural signal: "
        f"tool_calls={signals.get('tool_call_count', 0)}, "
        f"failures={signals.get('failure_count', 0)}, "
        f"positive_reward={signals.get('positive_reward', False)}, "
        f"stop_reason={signals.get('stop_reason') or 'unknown'}."
    )


def _suggested_markdown(context: SkillReviewContext) -> str:
    signals = context.summary.get("signals", {})
    lines = [
        "## Review Notes",
        "",
        f"- Task: {context.task}",
        f"- Tool calls observed: {signals.get('tool_call_count', 0)}",
        f"- Failures observed: {signals.get('failure_count', 0)}",
        f"- Max reward: {signals.get('max_reward')}",
        "- Approval required before merging this into the skill document.",
    ]
    return "\n".join(lines)


def _int_signal(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _atomic_write_json(path: Path, payload: JsonDict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
