from __future__ import annotations

from pathlib import Path

from agent.backends.planner import (
    PlannerBackendRequest,
    _partition_planner_tool_context,
)
from agent.runtime.planner import (
    _agent_owned_tool_planner_system_prompt,
    _contract_driven_tool_references,
)
from agent.runtime.planner_prompts import compose_main_planner_prompt
from agent.runtime.skills import load_skill_markdown
from agent.tools.registry import build_default_tool_registry


SKILL_ROOT = Path(__file__).resolve().parents[1] / "agent" / "skills"


def test_main_planner_prompt_is_compact_and_task_agnostic() -> None:
    prompt, _metadata = compose_main_planner_prompt(
        _agent_owned_tool_planner_system_prompt()
    )

    assert len(prompt) < 5_000
    assert "You own task decomposition" in prompt
    assert "current evidence outranks" in prompt.lower()
    assert "positive official reward" in prompt
    for task_specific_term in (
        "sam3",
        "anygrasp",
        "anyplace",
        "grasp_pose_estimate",
        "source_packet_id",
        "ik_receipt_id",
        "bundle_id",
        "milk carton",
        "libero",
    ):
        assert task_specific_term not in prompt.lower()


def test_task_skills_do_not_restate_tool_interface_fields() -> None:
    interface_only_terms = (
        "source_packet_id",
        "camera_frame_id",
        "bundle_id",
        "sam3_result_id",
        "grasp_result_id",
        "result_id",
        "detection_id",
        "candidate_id",
        "compiled_grasp_id",
        "ik_receipt_id",
        "receipt_id",
        "probe_id",
        "waypoint_index",
        "target_pose",
        "num_steps",
        "repair_call",
    )
    for name in ("pick", "place", "pull", "push", "stack", "sim_mcp"):
        content = load_skill_markdown(SKILL_ROOT / f"{name}.md").content.lower()
        for term in interface_only_terms:
            assert term not in content, f"{name} duplicates interface field {term}"


def test_python_artifact_api_is_documented_by_tool_schema() -> None:
    spec = build_default_tool_registry().get("python_exec")
    code_contract = str(spec.parameters["code"])

    for helper in (
        "describe()",
        "list_files()",
        "list_images()",
        "read_json()",
        "read_text()",
        "grep_text()",
    ):
        assert helper in code_contract


def test_agent_tool_projection_has_five_owned_fields_and_short_descriptions() -> None:
    references, _audit = _contract_driven_tool_references(
        build_default_tool_registry().list()
    )

    assert references
    for reference in references:
        assert set(reference) == {
            "name",
            "description",
            "parameters",
            "returns",
            "semantic_limits",
        }
        assert len(reference["description"]) <= 240
        assert isinstance(reference["returns"]["outcomes"], list)
        assert isinstance(reference["returns"]["fields"], list)
        assert reference["semantic_limits"]


def test_simulator_skill_does_not_duplicate_tool_permission_contracts() -> None:
    content = load_skill_markdown(SKILL_ROOT / "sim_mcp.md").content.lower()

    for interface_text in (
        "has no simulator mcp or network authority",
        "only agent-facing environment creation path",
        "only agent-facing environment cleanup path",
        "ik_receipt_id",
        "source_packet_id",
    ):
        assert interface_text not in content


def test_pick_skill_fits_default_projection_and_has_no_exact_task_evidence() -> None:
    skill = load_skill_markdown(SKILL_ROOT / "pick.md")
    lowered = skill.content.lower()

    assert len(skill.content) <= 8_000
    assert "context_char_limit" not in skill.metadata
    for exact_task_term in (
        "libero",
        "milk carton",
        "alphabet soup",
        "same-task canary",
        "10.59",
    ):
        assert exact_task_term not in lowered


def test_exact_task_playbook_uses_cache_stable_context_layer() -> None:
    request = PlannerBackendRequest(
        tool_context={
            "schema_version": "openeta.agent_context.v2",
            "task_playbook": {
                "schema_version": "openeta.task_playbook.v1",
                "playbook_id": "exact-task",
            },
            "current_observation": {"status": "available"},
        }
    )

    stable, dynamic = _partition_planner_tool_context(request)

    assert stable["task_playbook"]["playbook_id"] == "exact-task"
    assert "task_playbook" not in dynamic
    assert "current_observation" in dynamic
