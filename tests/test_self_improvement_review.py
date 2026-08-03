from __future__ import annotations

import json
from pathlib import Path

from agent.runtime.episode import DummyEpisodeEnvironment, OpenEtaEpisodeRunner
from agent.runtime.planner import BasePlanner, PlannerDecision
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.self_improvement import (
    HeuristicSkillReviewSubagent,
    SelfImprovementConfig,
    SelfImprovementReviewer,
    SkillReviewContext,
    SkillReviewProposal,
    SkillReviewProposalStore,
    should_review_episode,
)


class SaveMemoryPlanner(BasePlanner):
    def plan(self, observation, *, memory, tools, skills):
        del observation, memory, tools, skills
        return PlannerDecision(
            action_type="tool_call",
            action="save_memory",
            parameters={"key": "probe", "content": {"ok": True}},
            reasoning="exercise one executable bookkeeping tool",
        )


def test_should_review_episode_triggers_on_tool_call_threshold() -> None:
    summary = {
        "signals": {
            "tool_call_count": 3,
            "failure_count": 0,
            "positive_reward": False,
            "truncated": False,
        }
    }

    trigger = should_review_episode(
        summary,
        config=SelfImprovementConfig(min_tool_calls=3),
    )

    assert trigger == {"should_review": True, "reason": "tool_call_threshold"}


def test_heuristic_skill_review_subagent_targets_loaded_skill() -> None:
    context = SkillReviewContext(
        task="pick up the book",
        session_id="session-1",
        available_skills=("pick", "place"),
        loaded_skills=("pick",),
        summary={
            "signals": {
                "tool_call_count": 5,
                "failure_count": 0,
                "positive_reward": False,
                "max_reward": 0.0,
                "stop_reason": "environment",
            }
        },
    )

    proposals = HeuristicSkillReviewSubagent().review(context)

    assert len(proposals) == 1
    assert proposals[0].action == "patch"
    assert proposals[0].skill_name == "pick"
    assert "Tool calls observed: 5" in proposals[0].suggested_markdown


def test_episode_runner_writes_pending_skill_review_proposal(tmp_path: Path) -> None:
    reviewer = SelfImprovementReviewer(
        config=SelfImprovementConfig(min_tool_calls=1, proposal_root=tmp_path / "pending"),
        store=SkillReviewProposalStore(tmp_path / "pending"),
    )
    runtime = OpenEtaAgentRuntime(
        planner=SaveMemoryPlanner(),
        self_improvement_reviewer=reviewer,
    )
    runner = OpenEtaEpisodeRunner(
        runtime=runtime,
        environment=DummyEpisodeEnvironment(max_steps=1),
    )

    result = runner.run(task="pick up the book", max_turns=1)

    review = result.metadata["self_improvement_review"]
    assert review["reviewed"] is True
    assert review["trigger"]["reason"] == "tool_call_threshold"
    assert len(review["proposals"]) == 1
    proposal_path = Path(review["proposals"][0]["path"])
    assert proposal_path.exists()
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    assert proposal["status"] == "pending"
    assert proposal["skill_name"] == "pick"
    assert proposal["action"] == "patch"
    assert "Approval required" in proposal["suggested_markdown"]
    assert runtime.skills.get("pick").source != str(proposal_path)


def test_episode_runner_skips_review_without_signal(tmp_path: Path) -> None:
    reviewer = SelfImprovementReviewer(
        config=SelfImprovementConfig(min_tool_calls=99, proposal_root=tmp_path / "pending"),
        store=SkillReviewProposalStore(tmp_path / "pending"),
    )
    runtime = OpenEtaAgentRuntime(
        planner=SaveMemoryPlanner(),
        self_improvement_reviewer=reviewer,
    )
    runner = OpenEtaEpisodeRunner(
        runtime=runtime,
        environment=DummyEpisodeEnvironment(max_steps=1),
    )

    result = runner.run(task="pick up the book", max_turns=1)

    review = result.metadata["self_improvement_review"]
    assert review["reviewed"] is False
    assert review["trigger"]["reason"] == "no_signal"
    assert not (tmp_path / "pending").exists()


def test_skill_review_store_lists_loads_and_rejects_pending_proposals(tmp_path: Path) -> None:
    store = SkillReviewProposalStore(tmp_path / "pending")
    proposal = SkillReviewProposal(
        proposal_id="skill-review-test",
        action="patch",
        skill_name="pick",
        rationale="capture reusable lesson",
        suggested_markdown="## Review Notes\n\n- Keep retry bounded.",
    )

    path = store.save(proposal)

    pending = store.list()
    assert [item["proposal_id"] for item in pending] == ["skill-review-test"]
    assert store.load("skill-review-test")["path"] == str(path)
    rejected = store.reject("skill-review-test", reviewer="unit-test", reason="too vague")
    assert rejected["status"] == "rejected"
    assert rejected["resolved_by"] == "unit-test"
    assert store.list() == []
    assert store.list(status="rejected")[0]["resolution"]["reason"] == "too vague"


def test_skill_review_store_approve_applies_patch_to_skill_markdown(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()
    skill_path = skill_dir / "pick.md"
    skill_path.write_text(
        "---\nname: pick\ndescription: Pick guidance\n---\n\nExisting guidance.\n",
        encoding="utf-8",
    )
    store = SkillReviewProposalStore(tmp_path / "pending")
    proposal = SkillReviewProposal(
        proposal_id="skill-review-apply",
        action="patch",
        skill_name="pick",
        rationale="capture reusable lesson",
        suggested_markdown="## Review Notes\n\n- Prefer a short retreat after grasp.",
    )
    store.save(proposal)

    approved = store.approve("skill-review-apply", reviewer="unit-test", skill_dir=skill_dir)

    content = skill_path.read_text(encoding="utf-8")
    assert "<!-- openeta-skill-review:skill-review-apply -->" in content
    assert "Prefer a short retreat after grasp." in content
    assert approved["status"] == "approved"
    assert approved["resolved_by"] == "unit-test"
    application = approved["resolution"]["application"]
    assert application["target_path"] == str(skill_path.resolve())
