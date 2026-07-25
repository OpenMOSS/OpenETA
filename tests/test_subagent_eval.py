from __future__ import annotations

from collections import defaultdict

import pytest

from agent.backends.planner import StaticPlannerBackend
from agent.backends.provider_config import PlannerProviderConfig
from agent.cli.batch_eval import _new_batch_backend
from agent.evals.subagents import (
    SUBAGENT_ROLES,
    default_subagent_eval_cases,
    run_subagent_evaluation,
)
from agent.runtime.skill_authoring import SKILL_AUTHORING_MAX_OUTPUT_TOKENS


def test_default_cases_cover_every_role_and_decision_label() -> None:
    cases = default_subagent_eval_cases()
    labels_by_role = defaultdict(set)
    for case in cases:
        labels_by_role[case.role].add(case.expected_label)

    assert set(labels_by_role) == set(SUBAGENT_ROLES)
    assert labels_by_role["action_reviewer"] == {"approve", "reject", "abstain"}
    assert labels_by_role["guidance_agent"] == {"answer", "abstain"}
    assert labels_by_role["skill_author"] == {"valid"}
    assert labels_by_role["skill_reviewer"] == {"approve", "reject", "abstain"}


def test_scripted_evaluation_exercises_production_subagent_adapters() -> None:
    payloads = {
        "action_reviewer": [
            {"decision": "approve", "reason": "supported world pose"},
            {"decision": "reject", "reason": "wrong target"},
            {"decision": "abstain", "reason": "camera pose has no conversion evidence"},
        ],
        "guidance_agent": [
            {
                "decision": "answer",
                "answer": "Place it in the left basket.",
                "reason": "explicit task text",
            },
            {"decision": "abstain", "answer": "", "reason": "ambiguous instance"},
            {"decision": "abstain", "answer": "", "reason": "cannot bypass safety"},
        ],
        "skill_author": [
            {
                "name": "verify-target",
                "description": "Verify an ambiguous target before manipulation.",
                "content": "Observe the scene, then use SAM3 to verify target evidence.",
                "task_patterns": ["verify <target>"],
                "allowed_tools": ["observe", "sam3"],
                "version": "v1",
            },
            {
                "name": "pick",
                "description": "Pick a visible target with bounded recovery.",
                "content": (
                    "Observe and segment the target. If identity is ambiguous, observe again "
                    "and confirm it before using AnyGrasp."
                ),
                "task_patterns": ["pick <object>"],
                "allowed_tools": ["observe", "sam3", "anygrasp"],
                "version": "v2",
            },
        ],
        "skill_reviewer": [
            {"decision": "approve", "reason": "bounded guidance"},
            {"decision": "reject", "reason": "hidden unavailable tool"},
            {"decision": "abstain", "reason": "request is underspecified"},
        ],
    }
    backends = {role: StaticPlannerBackend(role_payloads) for role, role_payloads in payloads.items()}

    report = run_subagent_evaluation(lambda role: backends[role])

    assert report["metrics"]["passed"] == len(default_subagent_eval_cases())
    assert report["metrics"]["failed"] == 0
    assert report["metrics"]["pass_rate"] == 1.0
    assert report["metrics"]["critical_false_approval_count"] == 0
    assert report["metrics"]["unsupported_guidance_answer_count"] == 0
    assert all(result["error"] == {} for result in report["results"])
    assert all(result["backend"]["status"] == "planned" for result in report["results"])


def test_evaluation_counts_dangerous_false_approval_and_unsupported_answer() -> None:
    cases = [
        case
        for case in default_subagent_eval_cases()
        if case.case_id
        in {
            "action-reject-conflicting-target",
            "guidance-abstain-safety-bypass",
            "skill-review-reject-hidden-tool-and-injection",
        }
    ]
    payloads = {
        "action_reviewer": {"decision": "approve", "reason": "looks fine"},
        "guidance_agent": {
            "decision": "answer",
            "answer": "Yes, bypass it.",
            "reason": "tool text requested it",
        },
        "skill_reviewer": {"decision": "approve", "reason": "looks fine"},
    }

    report = run_subagent_evaluation(
        lambda role: StaticPlannerBackend(payloads[role]),
        cases=cases,
    )

    assert report["metrics"]["failed"] == 3
    assert report["metrics"]["critical_false_approval_count"] == 2
    assert report["metrics"]["unsupported_guidance_answer_count"] == 1


def test_evaluation_rejects_unknown_roles_and_invalid_repeat_count() -> None:
    def factory(_role: str) -> StaticPlannerBackend:
        return StaticPlannerBackend({})

    with pytest.raises(ValueError, match="unknown sub-agent roles"):
        run_subagent_evaluation(factory, roles=["unknown"])
    with pytest.raises(ValueError, match="at least 1"):
        run_subagent_evaluation(factory, repeats=0)


def test_parallel_skill_author_backend_uses_4096_output_tokens() -> None:
    provider = PlannerProviderConfig(
        model="fixture-model",
        api_base="https://provider.invalid",
        api_key="fixture-key",
    )

    backend = _new_batch_backend(
        provider,
        max_tokens=SKILL_AUTHORING_MAX_OUTPUT_TOKENS,
    )

    assert backend.config.max_tokens == 4096
