from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import agent.runtime.runtime_assembly as runtime_assembly
from agent.backends.planner import CallablePlannerBackend
from agent.cli.openeta_cli import OpenEtaCli
from agent.runtime.skill_authoring import (
    SKILL_AUTHORING_SYSTEM_PROMPT,
    SKILL_REVIEW_SYSTEM_PROMPT,
    BackendSkillAuthoringSubagent,
    SkillAuthoringRequest,
    SkillAuthoringResult,
    validate_authored_skill,
)
from agent.runtime.skills import SkillSpec
from agent.tools.registry import build_default_tool_registry


def _authored_payload(*, name: str = "inspect-object") -> dict:
    return {
        "name": name,
        "description": (
            "Inspect a target object before manipulation. Use for ambiguous or "
            "partially occluded tabletop targets."
        ),
        "content": (
            "# Inspect Object\n\n"
            "1. Observe the current scene.\n"
            "2. Compare visible evidence before selecting the target."
        ),
        "task_patterns": ["inspect <object>"],
        "allowed_tools": ["observe"],
        "version": "v1",
    }


def test_skill_prompts_include_authoring_and_review_examples() -> None:
    assert "- register:" in SKILL_AUTHORING_SYSTEM_PROMPT
    assert "- update:" in SKILL_AUTHORING_SYSTEM_PROMPT
    for label in ("approve:", "reject:", "abstain:"):
        assert label in SKILL_REVIEW_SYSTEM_PROMPT


def test_skill_authoring_subagent_receives_clean_bounded_context() -> None:
    captured = {}

    def author(request):
        captured["request"] = request
        return _authored_payload()

    observe = build_default_tool_registry().get("observe")
    result = BackendSkillAuthoringSubagent(
        CallablePlannerBackend(author, provider="fixture", model="author-model")
    ).author(
        SkillAuthoringRequest(
            operation="register",
            parameters={
                "name": "inspect-object",
                "goal": "inspect ambiguous objects",
                "conversation_history": "must not leak",
            },
            executable_tools=(observe,),
        )
    )

    request = captured["request"]
    assert request.system_prompt == SKILL_AUTHORING_SYSTEM_PROMPT
    assert request.metadata["isolated_context"] is True
    assert request.tool_context["role"] == "skill_document_author"
    assert request.tool_context["immutable_boundaries"]["tools_mutable"] is False
    assert request.tool_context["requested_skill"] == {
        "name": "inspect-object",
        "goal": "inspect ambiguous objects",
    }
    assert "memory" not in request.tool_context
    assert "observation" not in request.tool_context
    assert "conversation_history" not in request.tool_context["requested_skill"]
    assert result.skill.source == "skill_authoring_subagent"
    assert result.provider == "fixture"
    assert result.model == "author-model"


def test_skill_authoring_rejects_tool_mutation_fields() -> None:
    payload = _authored_payload()
    payload["tool_updates"] = [{"name": "observe", "parameters": {}}]

    with pytest.raises(ValueError, match="forbidden fields: tool_updates"):
        validate_authored_skill(
            payload,
            operation="register",
            requested_name="inspect-object",
            current_skill=None,
            executable_tool_names={"observe"},
        )


def test_skill_authoring_rejects_unavailable_tools_and_skill_rename() -> None:
    unavailable = _authored_payload()
    unavailable["allowed_tools"] = ["observe", "invented-motion-tool"]
    with pytest.raises(ValueError, match="unavailable or unbound tools"):
        validate_authored_skill(
            unavailable,
            operation="register",
            requested_name="inspect-object",
            current_skill=None,
            executable_tool_names={"observe"},
        )

    renamed = _authored_payload(name="renamed-skill")
    with pytest.raises(ValueError, match="changed requested name"):
        validate_authored_skill(
            renamed,
            operation="update",
            requested_name="inspect-object",
            current_skill=SkillSpec(
                name="inspect-object",
                description="Existing description",
                content="Existing guidance",
            ),
            executable_tool_names={"observe"},
        )


def test_skill_authoring_update_preserves_legacy_existing_name() -> None:
    payload = _authored_payload(name="sim_mcp")

    skill = validate_authored_skill(
        payload,
        operation="update",
        requested_name="sim_mcp",
        current_skill=SkillSpec(
            name="sim_mcp",
            description="Existing simulator guidance",
            content="Existing guidance",
        ),
        executable_tool_names={"observe"},
    )

    assert skill.name == "sim_mcp"


def test_tool_specs_are_host_owned_and_have_no_update_surface() -> None:
    tools = build_default_tool_registry()
    observe = tools.get("observe")

    assert not hasattr(tools, "update")
    with pytest.raises(FrozenInstanceError):
        observe.description = "mutated"  # type: ignore[misc]
    assert "never" in tools.get("register_skill").description.lower()
    assert "tools" in tools.get("update_skill").description.lower()


def test_cli_skill_tools_require_authoring_subagent(monkeypatch) -> None:
    calls: list[SkillAuthoringRequest] = []

    class FakeAuthor:
        def author(self, request: SkillAuthoringRequest) -> SkillAuthoringResult:
            calls.append(request)
            description = "Updated guidance" if request.operation == "update" else "New guidance"
            return SkillAuthoringResult(
                skill=SkillSpec(
                    name="inspect-object",
                    description=description,
                    content=f"# Inspect Object\n\n{description}.",
                    task_patterns=("inspect <object>",),
                    allowed_tools=("observe",),
                    source="skill_authoring_subagent",
                    metadata={"isolated_context": True},
                ),
                provider="fixture",
                model="author-model",
                details={"isolated_context": True},
            )

    monkeypatch.setattr(
        runtime_assembly,
        "BackendSkillAuthoringSubagent",
        lambda _backend: FakeAuthor(),
    )
    runtime = OpenEtaCli()._require_runtime()

    registered = runtime.tools.call(
        "register_skill",
        {"name": "inspect-object", "goal": "inspect ambiguous objects"},
    )
    updated = runtime.tools.call(
        "update_skill",
        {"name": "inspect-object", "requested_changes": "clarify target comparison"},
    )

    assert registered.success is True
    assert updated.success is True
    assert [request.operation for request in calls] == ["register", "update"]
    assert calls[0].current_skill is None
    assert calls[1].current_skill is not None
    assert calls[1].current_skill.description == "New guidance"
    assert runtime.skills.get("inspect-object").description == "Updated guidance"
    assert updated.details["outputs"]["authoring"]["isolated_context"] is True
