from __future__ import annotations

import json
from pathlib import Path

from agent.backends.planner import StaticPlannerBackend
from adapter.protocol import EnvAction, EnvObservation, RobotState, StepResult
from agent.runtime.episode import (
    DummyEpisodeEnvironment,
    EpisodeResult,
    EpisodeStep,
    OpenEtaEpisodeRunner,
)
from agent.runtime.planner import BasePlanner, PlannerDecision
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.self_improvement import (
    BackendReviewedSkillAutoApplier,
    HeuristicSkillReviewSubagent,
    SelfImprovementConfig,
    SelfImprovementReviewer,
    SkillReviewContext,
    SkillReviewProposal,
    SkillReviewProposalStore,
    should_review_episode,
)
from agent.runtime.skill_authoring import (
    BackendSkillAuthoringSubagent,
    BackendSkillChangeReviewer,
)
from agent.runtime.skills import SkillRegistry, load_skill_markdown
from agent.tools.registry import build_default_tool_registry


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


def test_positive_episode_extracts_reviewed_task_playbook(tmp_path: Path) -> None:
    session_id = "session-success"
    rollout = (
        tmp_path
        / "rollouts"
        / "sessions"
        / session_id
        / "rollout"
        / "tool_calls.jsonl"
    )
    rollout.parent.mkdir(parents=True)
    rollout.write_text(
        json.dumps(
            {
                "event": {
                    "phase": "start",
                    "name": "retrieve_asset_reference",
                    "parameters": {"target_object": "test bowl"},
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    observation = EnvObservation(
        task="pick up test bowl",
        cameras=[],
        robot=RobotState(end_effector_pose={"xyz": [0.0, 0.0, 0.5]}),
        objects=[],
        metadata={
            "episode_id": "episode-success",
            "env_id": "openeta/test-v0",
            "suite": "test_suite",
            "task_index": 1,
        },
    )
    action = EnvAction(
        action_type="tool_call",
        command={
            "tool_calls": [
                {
                    "name": "observe",
                    "status": "executed",
                    "result": {"success": True},
                }
            ]
        },
    )
    result = EpisodeResult(
        task="pick up test bowl",
        session_id=session_id,
        steps=[
            EpisodeStep(
                turn_index=0,
                observation=observation,
                action=action,
                step_result=StepResult(observation=observation, reward=1.0),
            )
        ],
        terminated=True,
        metadata={"assistance": {"agent_assisted": False}},
    )
    reviewer = SelfImprovementReviewer(
        config=SelfImprovementConfig(
            min_tool_calls=99,
            proposal_root=tmp_path / "reviews",
            task_playbook_candidate_root=tmp_path / "playbooks",
            rollout_root=tmp_path / "rollouts",
        ),
        store=SkillReviewProposalStore(tmp_path / "reviews"),
    )

    review = reviewer.maybe_review(result, skills=SkillRegistry())

    task_candidate = review["task_playbook_candidate"]
    assert task_candidate["created"] is True
    assert task_candidate["review"]["reviewer"] == "objective_evidence"
    assert Path(task_candidate["path"]).is_file()


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


def test_reviewed_auto_applier_uses_author_and_second_reviewer_in_session_dir(
    tmp_path: Path,
) -> None:
    skill_dir = tmp_path / "workspace" / "skills"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "pick.md"
    skill_path.write_text(
        "---\nname: pick\ndescription: Pick guidance\neditable: true\n---\n\n"
        "Existing guidance.\n",
        encoding="utf-8",
    )
    skills = SkillRegistry()
    skills.register(load_skill_markdown(skill_path))
    tools = build_default_tool_registry()
    tools.bind_handler("observe", lambda _context: {"success": True})
    applier = BackendReviewedSkillAutoApplier(
        author=BackendSkillAuthoringSubagent(
            StaticPlannerBackend(
                {
                    "name": "pick",
                    "description": "Pick objects after a failed or uncertain attempt.",
                    "content": "Observe again before retrying a failed grasp.",
                    "task_patterns": ["pick <object>"],
                    "allowed_tools": ["observe"],
                    "version": "v2",
                }
            )
        ),
        reviewer=BackendSkillChangeReviewer(
            StaticPlannerBackend(
                {"decision": "approve", "reason": "bounded and evidence-based"}
            )
        ),
        executable_tools=(tools.get("observe"),),
    )
    context = SkillReviewContext(
        task="pick the cup",
        session_id="session-a",
        summary={"signals": {"failure_count": 1}},
        available_skills=("pick",),
        loaded_skills=("pick",),
    )
    proposal = SkillReviewProposal(
        proposal_id="reviewed-session-update",
        action="patch",
        skill_name="pick",
        rationale="The previous grasp failed.",
        suggested_markdown="Observe again before retrying.",
    )

    result = applier.apply(context, proposal, skills=skills, skill_dir=skill_dir)

    assert result["applied"] is True
    assert result["application"]["session_local"] is True
    assert skills.get("pick").version == "v2"
    assert "Observe again before retrying" in skill_path.read_text(encoding="utf-8")
