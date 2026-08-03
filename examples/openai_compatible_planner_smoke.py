"""Smoke test an OpenAI-compatible planner backend against the dummy runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapter.protocol import CameraFrame, EnvObservation, RobotState
from agent.backends.planner import (
    OpenAICompatiblePlannerBackend,
    OpenAICompatiblePlannerBackendConfig,
    list_openai_compatible_models,
)
from agent.backends.provider_config import load_planner_provider_config
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.tools.handlers import bind_dummy_tool_handlers
from agent.tools.registry import build_default_tool_registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", help="Override OPENETA_LLM_MODEL.")
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    provider_config = load_planner_provider_config()
    if args.model:
        provider_config.model = args.model
    if args.timeout_s is not None:
        provider_config.timeout_s = args.timeout_s

    backend_config = OpenAICompatiblePlannerBackendConfig.from_provider_config(provider_config)
    if args.list_models:
        for model in list_openai_compatible_models(backend_config):
            print(model)
        return

    missing = provider_config.missing_fields()
    if missing:
        raise SystemExit(
            "Missing planner provider fields: "
            + ", ".join(missing)
            + ". Set them in .env or pass --model."
        )

    tools = build_default_tool_registry()
    bind_dummy_tool_handlers(tools)

    planner = ToolCallingPlanner(
        OpenAICompatiblePlannerBackend(backend_config)
    )
    runtime = OpenEtaAgentRuntime(planner=planner, tools=tools)
    runtime.start_session(task=_task())

    action = runtime.act(_observation())
    print("provider_config:")
    pprint(provider_config.redacted())
    print("action_summary:")
    pprint(_action_summary(action.command))


def _observation() -> EnvObservation:
    return EnvObservation(
        task=_task(),
        cameras=[
            CameraFrame(
                frame_id="front",
                rgb=[[[0, 0, 0]]],
                depth=[[1.0]],
            )
        ],
        robot=RobotState(end_effector_pose={"xyz": [0.0, 0.0, 0.5]}),
        objects=[{"name": "cube", "position": [0.2, 0.0, 0.0]}],
        metadata={"step_idx": 1},
    )


def _task() -> str:
    return (
        "Call exactly one available read-only tool to inspect the cube. "
        "Prefer sam3 with image='front' and prompt='cube'."
    )


def _action_summary(command):
    request = command.get("request", {})
    tool_calls = command.get("tool_calls", [])
    return {
        "status": command.get("status"),
        "request": request,
        "tool_calls": [
            {
                "name": call.get("name"),
                "status": call.get("status"),
                "result": call.get("result"),
            }
            for call in tool_calls
        ],
        "backend": {
            "provider": command.get("metadata", {})
            .get("planner_metadata", {})
            .get("backend_provider"),
            "model": command.get("metadata", {})
            .get("planner_metadata", {})
            .get("backend_model"),
            "usage": command.get("metadata", {})
            .get("planner_metadata", {})
            .get("backend_details", {})
            .get("usage", {}),
        },
    }


if __name__ == "__main__":
    main()
