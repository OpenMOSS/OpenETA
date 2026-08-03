"""Dry-run a model-backed OpenETA planner without a simulator or tool execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapter.protocol import CameraFrame, EnvObservation, RobotState
from agent.backends.planner import (
    OpenAICompatiblePlannerBackend,
    OpenAICompatiblePlannerBackendConfig,
)
from agent.backends.provider_config import load_planner_provider_config
from agent.runtime.memory import AgentMemory
from agent.runtime.planner import (
    PlannerContextConfig,
    ToolCallingPlanner,
    build_tool_context,
)
from agent.runtime.skills import build_default_skill_registry
from agent.tools.handlers import bind_dummy_tool_handlers
from agent.tools.registry import build_default_tool_registry


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Call the configured OpenAI-compatible planner backend on a static "
            "OpenETA observation. No simulator step or tool handler is executed."
        )
    )
    parser.add_argument("--model", default="", help="Override OPENETA_LLM_MODEL.")
    parser.add_argument("--task", default=_default_task())
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--print-context", action="store_true")
    args = parser.parse_args()

    provider_config = load_planner_provider_config()
    if args.model:
        provider_config.model = args.model
    if args.timeout_s is not None:
        provider_config.timeout_s = args.timeout_s

    missing = provider_config.missing_fields()
    if missing:
        raise SystemExit(
            "Missing planner provider fields: "
            + ", ".join(missing)
            + ". Configure .env with /provider first."
        )

    memory = AgentMemory()
    memory.start_session(task=args.task, metadata={"source": "model_backed_planner_dry_run"})
    observation = _observation(args.task)
    tools = bind_dummy_tool_handlers(build_default_tool_registry())
    skills = build_default_skill_registry()
    context_config = PlannerContextConfig(
        context_window_tokens=provider_config.context_window_tokens,
        token_estimator_model=provider_config.model,
    )
    tool_context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=tools,
        skills=skills,
        config=context_config,
    )
    if args.print_context:
        print_json({"tool_context": tool_context})

    backend_config = OpenAICompatiblePlannerBackendConfig.from_provider_config(provider_config)
    planner = ToolCallingPlanner(
        OpenAICompatiblePlannerBackend(backend_config),
        context_config=context_config,
    )
    decision = planner.plan(
        observation,
        memory=memory,
        tools=tools,
        skills=skills,
    )

    print_json(
        {
            "dry_run": True,
            "executed_tool": False,
            "provider": provider_config.redacted(),
            "session_id": memory.session_id,
            "context_budget": tool_context.get("context_budget", {}),
            "decision": {
                "kind": decision.action_type,
                "name": decision.action,
                "parameters": decision.parameters,
                "reasoning": decision.reasoning,
                "metadata": {
                    "backend_status": decision.metadata.get("backend_status"),
                    "backend_provider": decision.metadata.get("backend_provider"),
                    "backend_model": decision.metadata.get("backend_model"),
                    "backend_details": decision.metadata.get("backend_details", {}),
                    "validation_attempts": decision.metadata.get("validation_attempts"),
                },
            },
        }
    )


def _observation(task: str) -> EnvObservation:
    return EnvObservation(
        task=task,
        cameras=[
            CameraFrame(
                frame_id="front",
                rgb=[[[0, 0, 0]]],
                depth=[[1.0]],
            )
        ],
        robot=RobotState(end_effector_pose={"xyz": [0.0, 0.0, 0.5]}),
        objects=[{"name": "cube", "position": [0.2, 0.0, 0.0]}],
        metadata={"step_idx": 1, "dry_run": True},
    )


def _default_task() -> str:
    return (
        "Dry-run planning only. Choose exactly one next OpenETA command for "
        "inspecting the cube. Prefer a read-only or planning tool."
    )


def print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
