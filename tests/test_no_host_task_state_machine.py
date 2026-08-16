from __future__ import annotations

import inspect
from pathlib import Path

from agent.runtime.memory import AgentMemory
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import PlannerContextConfig


RUNTIME_ROOT = Path(__file__).parents[1] / "agent"
FORBIDDEN_RUNTIME_TOKENS = (
    "grasp_execution",
    "grasp_candidate_policy",
    "grasp_recovery",
    "grasp_lift_probe",
    "placement_release",
    "required_action",
    "host_task_policy_enabled",
    "task_execution_gate_enabled",
    "task_state_tracking_enabled",
    "activate_final_grasp_candidate",
    "grasp_stage",
    "placement_stage",
    "successful_stage_sequence",
    "selection_obligation",
    "reference_localization_obligation",
    "attachment_gate",
    "grasp_compile_obligation",
    "candidate_fallback",
    "final_refinable_fallback",
)


def test_runtime_contains_no_host_task_progress_state_machine() -> None:
    offenders: dict[str, list[str]] = {}
    for path in RUNTIME_ROOT.rglob("*"):
        if path.suffix not in {".py", ".md"} or "README" in path.name:
            continue
        if path.name == "memory_migrations.py":
            continue
        text = path.read_text(encoding="utf-8")
        found = [token for token in FORBIDDEN_RUNTIME_TOKENS if token in text]
        if found:
            offenders[str(path.relative_to(RUNTIME_ROOT.parent))] = found
    assert offenders == {}


def test_runtime_components_have_no_legacy_policy_switches() -> None:
    assert "task_state_tracking_enabled" not in inspect.signature(AgentMemory).parameters
    assert "task_execution_gate_enabled" not in inspect.signature(ActionPipeline).parameters
    assert "host_task_policy_enabled" not in inspect.signature(
        PlannerContextConfig
    ).parameters


def test_planning_context_contains_evidence_not_task_phases() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    context = memory.planning_context()

    for key in (
        "grasp_execution",
        "grasp_candidate_policy",
        "grasp_recovery",
        "grasp_lift_probe",
        "placement_release",
    ):
        assert key not in context
    assert context["object_scene_epoch"] == 0
    assert context["robot_motion_epoch"] == 0
    assert "provenance_evidence_graph" in context
    assert "motion_reconciliation" in context
