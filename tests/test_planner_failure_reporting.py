"""Planner validation exhaustion must not disappear into a normal talk summary."""
from agent.backends.planner import StaticPlannerBackend
from agent.runtime.episode import DummyEpisodeEnvironment, OpenEtaEpisodeRunner
from agent.runtime.parallel import classify_episode_result, episode_failure_error
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.self_improvement import SelfImprovementConfig, SelfImprovementReviewer
from agent.tools.registry import build_default_tool_registry


def run(payload):
    runtime = OpenEtaAgentRuntime(
        planner=ToolCallingPlanner(StaticPlannerBackend(payload), max_validation_retries=1),
        tools=build_default_tool_registry(),
        self_improvement_reviewer=SelfImprovementReviewer(config=SelfImprovementConfig(enabled=False)),
    )
    return OpenEtaEpisodeRunner(runtime=runtime, environment=DummyEpisodeEnvironment()).run(
        task="bounded grammar test", max_turns=2,
    )


def test_exhausted_validation_is_reported_as_planner_failure():
    result = run({"kind": "tool", "name": "observe", "parameters": {}})
    assert result.metadata["stop_reason"] == "planner_validation_failed"
    assert result.metadata["failure_reason"]["validation_attempts"] == 2
    assert result.metadata["failure_reason"]["validation_errors"] == ["Unsupported command kind: 'tool'."]
    assert result.truncated and not result.terminated
    assert result.steps[-1].step_result.truncated
    assert not result.steps[-1].step_result.terminated
    assert classify_episode_result(result) == "fail"
    assert episode_failure_error(result)["type"] == "EpisodePlannerFailure"
    assert result.metadata["usage"]["tool_admission"]["admitted_this_run"] == 0


def test_ordinary_talk_is_not_a_planner_failure():
    result = run({"kind": "response", "name": "talk", "parameters": {"message": "diagnostic complete"}})
    assert result.metadata["stop_reason"] == "status_report"
    assert result.metadata["failure_reason"] == {}
    assert result.terminated and not result.truncated


def test_model_cannot_forge_host_validation_failure_with_a_response_parameter():
    result = run({"kind": "response", "name": "talk", "parameters": {
        "message": "operator status", "code": "planner_validation_failed", "validation_errors": ["fake"],
    }})
    assert result.metadata["failure_reason"] == {}
