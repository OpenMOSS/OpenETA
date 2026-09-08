from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import threading
import time

import pytest

from agent.backends.planner import StaticPlannerBackend
from agent.runtime.episode import DummyEpisodeEnvironment, EpisodeResult, OpenEtaEpisodeRunner
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.self_improvement import (
    BackendReviewedSkillAutoApplier, SelfImprovementConfig, SelfImprovementReviewer,
    SkillReviewContext, SkillReviewProposal,
)
from agent.runtime.skills import SkillRegistry, SkillSpec


PROPOSAL = SkillReviewProposal("fixture-review", "patch", "pick", "fixture", "new guidance")
SKILL = SkillSpec("pick", "fixture", "old guidance")


class Proposer:
    def review(self, context):
        return [deepcopy(PROPOSAL)]


class Blocker:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def wait(self):
        self.entered.set()
        assert self.release.wait(3), "test did not release preparation"


def make_runner(tmp_path, *, subagent=None, applier=None, timeout=0.08):
    reviewer = SelfImprovementReviewer(config=SelfImprovementConfig(
        min_tool_calls=0, proposal_root=tmp_path / "pending", skill_dir=tmp_path / "skills",
        preparation_timeout_s=timeout, auto_apply_reviewed=applier is not None,
    ), subagent=subagent or Proposer(), auto_applier=applier)
    skills = SkillRegistry()
    skills.register(deepcopy(SKILL))
    runtime = OpenEtaAgentRuntime(
        planner=ToolCallingPlanner(StaticPlannerBackend({
            "kind": "response", "name": "talk", "parameters": {"message": "fixture report"},
        })), skills=skills, self_improvement_reviewer=reviewer,
    )
    return OpenEtaEpisodeRunner(runtime=runtime, environment=DummyEpisodeEnvironment())


@pytest.mark.parametrize("value", [True, False, 0, -1, "1", None, float("nan"), float("inf"), 10**1000])
def test_invalid_review_preparation_budget_is_rejected(value):
    with pytest.raises(ValueError, match="positive and finite"):
        SelfImprovementConfig(preparation_timeout_s=value)


def test_proposal_timeout_returns_task_without_files_and_retains_busy_worker(tmp_path):
    blocker = Blocker()

    class SlowProposer:
        def review(self, context):
            blocker.wait()
            context.summary["metadata"].clear()
            return [deepcopy(PROPOSAL)]

    runner = make_runner(tmp_path, subagent=SlowProposer())
    started = time.monotonic()
    try:
        result = runner.run(task="fixture", max_turns=1)
        assert blocker.entered.is_set()
        assert time.monotonic() - started < 1
        assert result.terminated and len(result.steps) == 1
        report = result.metadata["self_improvement_review"]
        assert report["error"]["type"] == "ReviewPreparationTimeout"
        assert report["preparation_budget"]["commit_started"] is False
        assert report["preparation_budget"]["worker_pending"] is True
        assert not list(tmp_path.rglob("*.json"))
        assert not runner.wait_for_idle(timeout_s=0)
        with pytest.raises(RuntimeError, match="still running"):
            runner.run(task="must not start", max_turns=1)
        before = deepcopy(result.to_dict())
    finally:
        blocker.release.set()
        assert runner.wait_for_idle(timeout_s=1)
    assert result.to_dict() == before
    assert not list(tmp_path.rglob("*.json"))
    assert runner.runtime.skills.get("pick") == SKILL


@pytest.mark.parametrize("stage", ["author", "review"])
def test_late_author_or_reviewer_has_no_write_path_and_no_next_provider_dispatch(tmp_path, stage):
    blocker, calls = Blocker(), []

    class Author:
        def author(self, request):
            calls.append("author")
            if stage == "author":
                blocker.wait()
            return SimpleNamespace(skill=replace(SKILL, content="new guidance", version="v2"), details={})

    class Approver:
        def review(self, **kwargs):
            calls.append("review")
            if stage == "review":
                blocker.wait()
            return SimpleNamespace(approved=True, decision="approve", reason="fixture", details={})

    applier = BackendReviewedSkillAutoApplier(author=Author(), reviewer=Approver(), executable_tools=())
    runner = make_runner(tmp_path, applier=applier)
    try:
        result = runner.run(task="fixture", max_turns=1)
        assert result.metadata["self_improvement_review"]["preparation_budget"]["commit_started"] is False
        assert blocker.entered.is_set()
    finally:
        blocker.release.set()
        assert runner.wait_for_idle(timeout_s=1)
    assert calls == (["author"] if stage == "author" else ["author", "review"])
    assert not list(tmp_path.rglob("*.json")) and not list(tmp_path.rglob("*.md"))
    assert runner.runtime.skills.get("pick") == SKILL


def test_preparation_is_write_free_and_commit_applies_without_provider_calls(tmp_path):
    calls = []
    author = SimpleNamespace(author=lambda request: calls.append("author") or SimpleNamespace(
        skill=replace(SKILL, content="new guidance", version="v2"), details={}))
    approver = SimpleNamespace(review=lambda **kwargs: calls.append("review") or SimpleNamespace(
        approved=True, decision="approve", reason="fixture", details={}))
    applier = BackendReviewedSkillAutoApplier(author=author, reviewer=approver, executable_tools=())
    runner = make_runner(tmp_path, applier=applier, timeout=1)
    reviewer = runner.runtime.self_improvement_reviewer
    plan = reviewer.prepare_review(EpisodeResult("fixture", "fixture"), skills=runner.runtime.skills)
    assert calls == ["author", "review"]
    assert not list(tmp_path.rglob("*.json")) and not list(tmp_path.rglob("*.md"))
    assert runner.runtime.skills.get("pick") == SKILL
    report = reviewer.commit_review(plan, skills=runner.runtime.skills)
    assert report["proposals"][0]["auto_apply"]["applied"] is True
    assert calls == ["author", "review"]
    assert runner.runtime.skills.get("pick").version == "v2"
    assert "new guidance" in (tmp_path / "skills" / "pick.md").read_text()


@pytest.mark.parametrize("changed", [replace(SKILL, version="elsewhere"), replace(SKILL, editable=False)])
def test_stale_or_now_locked_skill_is_not_overwritten_at_commit(tmp_path, changed):
    applier = BackendReviewedSkillAutoApplier(
        author=SimpleNamespace(author=lambda request: SimpleNamespace(
            skill=replace(SKILL, version="v2"), details={})),
        reviewer=SimpleNamespace(review=lambda **kwargs: SimpleNamespace(
            approved=True, decision="approve", reason="fixture", details={})), executable_tools=(),
    )
    skills = SkillRegistry()
    skills.register(SKILL)
    plan = applier.prepare(SkillReviewContext("fixture", "fixture", {}, ("pick",)), PROPOSAL, skills=skills)
    skills.update(changed)
    assert applier.commit(plan, skills=skills, skill_dir=tmp_path)["applied"] is False
    assert skills.get("pick") == changed
    assert not list(tmp_path.iterdir())


def test_timely_runner_commits_proposals_with_independent_budget_metadata(tmp_path):
    runner = make_runner(tmp_path, timeout=1)
    result = runner.run(task="fixture", max_turns=1)
    report = result.metadata["self_improvement_review"]
    assert report["reviewed"] is True
    assert report["preparation_budget"] == {"timeout_s": 1.0, "completed_in_time": True}
    assert (tmp_path / "pending" / "fixture-review.json").is_file()
    assert runner.wait_for_idle(timeout_s=0)


def test_reconfigured_reviewer_cannot_commit_old_plan(tmp_path):
    runner = make_runner(tmp_path)
    reviewer = runner.runtime.self_improvement_reviewer
    plan = reviewer.prepare_review(EpisodeResult("fixture", "fixture"), skills=runner.runtime.skills)
    reviewer.config = replace(reviewer.config, proposal_root=tmp_path / "replacement")
    with pytest.raises(RuntimeError, match="configuration changed"):
        reviewer.commit_review(plan, skills=runner.runtime.skills)
    assert not list(tmp_path.rglob("*.json"))


def test_session_replacement_during_preparation_discards_plan_without_writes(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    blocker = Blocker()

    class SlowProposer:
        def review(self, context):
            blocker.wait()
            return [deepcopy(PROPOSAL)]

    runner = make_runner(tmp_path, subagent=SlowProposer(), timeout=2)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(runner.run, task="fixture", max_turns=1)
            assert blocker.entered.wait(1)
            runner.runtime.start_session(task="replacement", session_id="replacement")
            result = future.result(timeout=1)
            assert result.metadata["self_improvement_review"]["memory_publication"]["recorded"] is False
    finally:
        blocker.release.set()
        assert runner.wait_for_idle(timeout_s=1)
    assert not list(tmp_path.rglob("*.json"))
    assert all(event.event_type != "self_improvement_review" for event in runner.runtime.memory.events)


def test_legacy_custom_applier_remains_synchronous_and_is_not_claimed_bounded(tmp_path):
    calls = []
    applier = SimpleNamespace(apply=lambda *args, **kwargs: calls.append("apply") or {"applied": False})
    runner = make_runner(tmp_path, applier=applier)
    assert not runner.runtime.self_improvement_reviewer.supports_bounded_preparation
    result = runner.run(task="fixture", max_turns=1)
    assert calls == ["apply"]
    assert "preparation_budget" not in result.metadata["self_improvement_review"]


def test_multiple_updates_use_virtual_order_then_commit_in_same_order(tmp_path):
    versions = []

    class TwoProposals:
        def review(self, context):
            return [replace(PROPOSAL, proposal_id="first"), replace(PROPOSAL, proposal_id="second")]

    class Author:
        def author(self, request):
            versions.append(request.current_skill.version)
            return SimpleNamespace(skill=replace(request.current_skill, version=f"v{len(versions) + 1}"), details={})

    applier = BackendReviewedSkillAutoApplier(
        author=Author(), reviewer=SimpleNamespace(review=lambda **kwargs: SimpleNamespace(
            approved=True, decision="approve", reason="fixture", details={})), executable_tools=(),
    )
    runner = make_runner(tmp_path, subagent=TwoProposals(), applier=applier, timeout=1)
    result = runner.run(task="fixture", max_turns=1)
    report = result.metadata["self_improvement_review"]
    assert report["preparation_budget"]["completed_in_time"] is True
    assert versions == ["v1", "v2"]
    assert len(report["proposals"]) == 2
    assert all(p["auto_apply"]["applied"] for p in report["proposals"])
    assert runner.runtime.skills.get("pick").version == "v3"
