from __future__ import annotations

import io
import json
import urllib.error

import pytest

from agent.backends import planner as module
from agent.backends.planner import (
    OpenAICompatiblePlannerBackend, OpenAICompatiblePlannerBackendConfig,
    PlannerBackendRequest, ProviderHttpError,
)
from agent.backends.provider_config import ProviderEndpointConfig
from agent.runtime.actions import PipelineStatus
from agent.runtime.episode import DummyEpisodeEnvironment, OpenEtaEpisodeRunner
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.tools.registry import build_default_tool_registry


def envelope(**updates):
    return json.dumps({"error": {
        "type": "human_cancelled", "code": "human_cancelled",
        "request_id": "fixture-request", "message": "operator cancelled",
        **updates,
    }})


def backend(transport, *, fallback=True, sleep=lambda seconds: None):
    return OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="fixture", api_base="http://primary.invalid/v1", api_key="fixture",
            max_attempts=4, retry_backoff_s=0,
            fallback=ProviderEndpointConfig(
                provider="fixture", model="fallback", api_base="http://fallback.invalid/v1",
                api_key="fixture", timeout_s=1,
            ) if fallback else None,
            enable_vision=False,
        ), transport=transport, sleep=sleep,
    )


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("message", ["operator cancelled", "Human response timed out.", "capacity quota overloaded"])
def test_explicit_cancellation_never_retries_sleeps_or_fails_over(fallback, message):
    calls, sleeps = [], []

    def transport(url, *args):
        calls.append(url)
        raise ProviderHttpError(503, envelope(message=message))

    result = backend(transport, fallback=fallback, sleep=sleeps.append).decide(
        PlannerBackendRequest(tool_context={}, system_prompt="fixture"),
    )
    assert result.status == PipelineStatus.FAILED
    assert result.details["provider_error_code"] == "manual_provider_cancelled"
    assert result.details["retryable"] is False
    assert result.details["provider_attempts"] == 1
    assert result.details["provider_failover"] is False
    assert len(calls) == 1 and not sleeps


@pytest.mark.parametrize("message", [
    "human_cancelled", "human_cancelled: overloaded",
    json.dumps({"message": envelope()}), "[]", "null", "{broken",
    envelope(type="busy"), envelope(code="busy"), envelope(request_id=""),
    envelope(request_id=True), envelope(request_id="x" * 129),
    envelope(message="x" * 16_384),
])
def test_ordinary_or_unrecognized_503_keeps_retry_and_failover_policy(message):
    exc = ProviderHttpError(503, message)
    assert module._is_transient_provider_error(exc)
    assert module._is_provider_failover_error(exc)
    assert module._provider_failure_code(exc) != "manual_provider_cancelled"


def test_real_http_error_conversion_preserves_terminal_envelope_and_does_not_send_again(monkeypatch):
    calls = []

    def urlopen(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 503, "Unavailable", {},
                                     io.BytesIO(envelope().encode()))

    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
    result = backend(module._post_json).decide(PlannerBackendRequest(tool_context={}, system_prompt="fixture"))
    assert result.details["provider_error_code"] == "manual_provider_cancelled"
    assert len(calls) == 1


def test_planner_validation_and_episode_do_not_reissue_cancelled_provider_call():
    calls = []

    def transport(*args):
        calls.append(1)
        raise ProviderHttpError(503, envelope())

    runner = OpenEtaEpisodeRunner(
        runtime=OpenEtaAgentRuntime(
            planner=ToolCallingPlanner(backend(transport), max_validation_retries=3),
            tools=build_default_tool_registry(),
        ), environment=DummyEpisodeEnvironment(),
    )
    result = runner.run(task="fixture", max_turns=8)
    assert len(calls) == 1
    request = result.steps[0].action.command["request"]
    assert request["name"] == "ask_human"
    assert request["parameters"]["provider_error_code"] == "manual_provider_cancelled"
    assert request["parameters"]["retryable"] is False
