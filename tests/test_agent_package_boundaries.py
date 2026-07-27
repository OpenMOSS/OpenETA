from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from adapter.dummy_agent import DummyAgentAdapter
from adapter.dummy_sim import DummySimulatorAdapter
from adapter.openeta_agent import OpenEtaAgentAdapter
from agent.backends.planner import StaticPlannerBackend
from agent.tools.coding import PythonExecRuntime
from agent.tools.registry import ToolRegistry, build_default_tool_registry
from agent.tools.sim_mcp import SseSimulatorMcpTransport


def test_agent_backend_and_tool_packages_are_primary_import_paths() -> None:
    tools = build_default_tool_registry()

    assert isinstance(tools, ToolRegistry)
    assert StaticPlannerBackend({"kind": "response", "name": "talk"}).descriptor()["name"] == (
        "StaticPlannerBackend"
    )
    assert PythonExecRuntime.__name__ == "PythonExecRuntime"
    assert SseSimulatorMcpTransport.__name__ == "SseSimulatorMcpTransport"


def test_dummy_adapter_uses_current_response_schema() -> None:
    simulator = DummySimulatorAdapter()
    agent = DummyAgentAdapter()
    observation = simulator.reset(task="test current command schema")
    agent.start_session(task=observation.task)

    action = agent.act(observation)

    assert action.action_type == "response"
    assert action.command["schema_version"] == "openeta.agent_command.v1"
    assert action.command["request"]["kind"] == "response"
    assert action.command["request"]["name"] == "task_complete"


def test_openeta_agent_adapter_imports_from_runtime_owner() -> None:
    assert OpenEtaAgentAdapter.__name__ == "OpenEtaAgentAdapter"


@pytest.mark.parametrize(
    "script",
    [
        "examples/model_backed_planner_dry_run.py",
        "examples/openai_compatible_planner_smoke.py",
        "examples/parallel_simulator_capacity_smoke.py",
    ],
)
def test_documented_planner_examples_expose_help(script: str) -> None:
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=repo,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
