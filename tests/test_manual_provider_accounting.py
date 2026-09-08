from __future__ import annotations

import pytest

from agent.backends.planner import OpenAICompatiblePlannerBackend, OpenAICompatiblePlannerBackendConfig
from agent.backends.provider_interaction import manual_provider_interaction
from agent.runtime.episode import DummyEpisodeEnvironment, OpenEtaEpisodeRunner
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.tools.registry import build_default_tool_registry
from tools import manual_vlm_proxy as proxy
from tools.manual_vlm_openeta import serialize_decision_xml


def marker(request_id="r1", wait_s=5.0):
    return {"schema_version": "manual_vlm.provider_interaction.v1", "mode": "manual_console",
            "request_id": request_id, "wait_s": wait_s}


@pytest.mark.parametrize("value", [None, {}, [], "manual", marker(wait_s=True), marker(wait_s=-1),
    marker(wait_s=float("nan")), marker(wait_s=float("inf")), marker(wait_s="5"),
    marker(wait_s=10**1000), marker(request_id=""), marker(request_id="x" * 129),
    {**marker(), "mode": "autonomous"}, {**marker(), "schema_version": "future"}])
def test_invalid_provider_timing_is_not_evidence(value):
    assert manual_provider_interaction(value) == {}


def test_console_timing_uses_monotonic_clock_and_is_not_operator_supplied(monkeypatch):
    store = proxy.RequestStore()
    request = store.add({"model": "fixture", "messages": []})
    request.received_monotonic_s = 100
    monkeypatch.setattr(proxy.time, "monotonic", lambda: 107.5)
    monkeypatch.setattr(proxy.time, "time", lambda: request.received_at - 1000)
    assert proxy._provider_interaction(request) == {}
    store.respond(request.request_id, {"content": "fixture", "provider_interaction": marker(wait_s=999)})
    assert proxy._provider_interaction(request) == marker(request.request_id, 7.5)


def make_runner(responses):
    def transport(url, body, headers, timeout_s):
        return responses.pop(0)

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            api_base="http://fixture.invalid/v1", api_key="fixture", model="fixture",
            enable_vision=False,
        ), transport=transport,
    )
    return OpenEtaEpisodeRunner(
        runtime=OpenEtaAgentRuntime(planner=ToolCallingPlanner(backend, max_validation_retries=1),
                                   tools=build_default_tool_registry()),
        environment=DummyEpisodeEnvironment(),
    )


def completion(payload, interaction=None):
    result = {"choices": [{"message": {"role": "assistant", "content": serialize_decision_xml(payload)},
                            "finish_reason": "stop"}]}
    if interaction is not None:
        result["provider_interaction"] = interaction
    return result


DECISION = {"kind": "tool_call", "name": "get_memory", "parameters": {}}


def test_full_backend_planner_episode_preserves_validation_retry_wait_without_changing_budget_clock():
    runner = make_runner([
        completion({"kind": "tool_call", "name": "unknown", "parameters": {}}, marker("r1", 4)),
        completion(DECISION, marker("r2", 6)),
    ])
    result = runner.run(task="fixture", max_turns=1)
    usage = result.metadata["usage"]
    assert usage["manual_provider"] == {
        "scope": "recorded_main_planner_responses_this_run", "response_count": 2,
        "wait_s": 10.0, "source": "provider_reported",
    }
    assert usage["human_wait_s"] == 0
    assert usage["elapsed_s"] >= 0
    assert result.metadata["assistance"]["manual_provider_assisted"] is True
    assert result.metadata["assistance"]["human_assisted"] is True
    assert result.metadata["assistance"]["agent_assisted"] is False


@pytest.mark.parametrize("interaction", [None, marker(wait_s=0), marker(wait_s=float("nan"))])
def test_mode_is_explicit_not_guessed_from_model_name_or_elapsed_time(interaction):
    runner = make_runner([completion(DECISION, interaction)])
    result = runner.run(task="fixture", max_turns=1)
    expected = bool(manual_provider_interaction(interaction))
    assert result.metadata["assistance"]["manual_provider_assisted"] is expected
    assert result.metadata["assistance"]["human_assisted"] is expected


def test_duplicate_receipt_is_counted_once_and_new_run_clears_current_run_ledger():
    responses = [
        completion({"kind": "tool_call", "name": "unknown", "parameters": {}}, marker()),
        completion(DECISION, marker()), completion(DECISION),
    ]
    runner = make_runner(responses)
    first = runner.run(task="fixture", max_turns=1)
    assert first.metadata["usage"]["manual_provider"]["response_count"] == 1
    assert first.metadata["usage"]["manual_provider"]["wait_s"] == 5
    second = runner.run(task="fixture", max_turns=1)
    assert second.metadata["usage"]["manual_provider"]["response_count"] == 0
    assert second.metadata["assistance"]["manual_provider_assisted"] is False
