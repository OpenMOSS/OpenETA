from __future__ import annotations

import http.client
import json
from pathlib import Path

from agent.backends.planner import (
    OpenAICompatiblePlannerBackend,
    OpenAICompatiblePlannerBackendConfig,
    PlannerBackendRequest,
    PROVIDER_CAPACITY_MIN_RETRY_BACKOFF_S,
    ProviderHttpError,
    _planner_user_prompt,
)
from agent.backends.provider_config import (
    DEFAULT_PLANNER_PROVIDER_TIMEOUT_S,
    PlannerProviderConfig,
    ProviderEndpointConfig,
    load_planner_provider_config,
    write_env_file,
)


def _success_response() -> dict[str, object]:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": '{"kind":"response","name":"talk"}'},
            }
        ],
        "usage": {"total_tokens": 8},
    }


def _request() -> PlannerBackendRequest:
    return PlannerBackendRequest(tool_context={"task": "test"}, system_prompt="json")


def _fallback() -> ProviderEndpointConfig:
    return ProviderEndpointConfig(
        provider="fallback-compatible",
        model="fallback-model",
        api_base="https://fallback.example.test/v1",
        api_key="fallback-key",
        timeout_s=9.0,
    )


def test_planner_retry_prompt_marks_previous_candidate_rejected_and_requires_change() -> None:
    prompt = _planner_user_prompt(
        PlannerBackendRequest(
            tool_context={
                "decision_state": {
                    "current_observation_packet": {"source_packet_id": "obs-0000"}
                }
            },
            system_prompt="json",
            attempt=2,
            validation_errors=["source_packet_id is required"],
        )
    )

    payload = json.loads(prompt)
    assert payload["attempt"] == 2
    assert payload["validation_feedback"] == {
        "status": "previous_attempt_rejected",
        "rejected_attempt": 1,
        "must_change_rejected_action": True,
        "errors": ["source_packet_id is required"],
    }
    assert "requested first attempt is now complete" in payload["instruction"]
    assert "do not repeat the same rejected" in payload["instruction"]


def test_provider_timeout_defaults_allow_reasoning_provider_latency(
    tmp_path: Path,
) -> None:
    loaded = load_planner_provider_config(
        env={
            "OPENETA_LLM_PROVIDER": "primary-compatible",
            "OPENETA_LLM_MODEL": "primary-model",
            "OPENETA_LLM_API_BASE": "https://primary.example.test",
            "OPENETA_LLM_API_KEY": "primary-key",
            "OPENETA_LLM_FALLBACK_PROVIDER": "fallback-compatible",
            "OPENETA_LLM_FALLBACK_MODEL": "fallback-model",
            "OPENETA_LLM_FALLBACK_API_BASE": "https://fallback.example.test",
            "OPENETA_LLM_FALLBACK_API_KEY": "fallback-key",
        },
        dotenv_path=tmp_path / "missing.env",
        apikey_path=tmp_path / "missing.md",
    )

    assert DEFAULT_PLANNER_PROVIDER_TIMEOUT_S == 180.0
    assert loaded.timeout_s == DEFAULT_PLANNER_PROVIDER_TIMEOUT_S
    assert loaded.fallback is not None
    assert loaded.fallback.timeout_s == DEFAULT_PLANNER_PROVIDER_TIMEOUT_S
    assert OpenAICompatiblePlannerBackendConfig().timeout_s == 180.0


def test_provider_config_roundtrips_fallback_endpoint(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    write_env_file(
        PlannerProviderConfig(
            provider="primary-compatible",
            model="primary-model",
            api_base="https://primary.example.test",
            api_key="primary-key",
            timeout_s=5.0,
            max_attempts=4,
            retry_backoff_s=0.25,
            fallback=_fallback(),
        ),
        env_path,
    )

    loaded = load_planner_provider_config(
        env={},
        dotenv_path=env_path,
        apikey_path=tmp_path / "missing.md",
    )

    assert loaded.fallback is not None
    assert loaded.fallback.provider == "fallback-compatible"
    assert loaded.fallback.model == "fallback-model"
    assert loaded.fallback.api_base == "https://fallback.example.test/v1"
    assert loaded.fallback.api_key == "fallback-key"
    assert loaded.fallback.timeout_s == 9.0
    redacted_fallback = loaded.redacted()["fallback"]
    assert isinstance(redacted_fallback, dict)
    assert redacted_fallback["api_key"] != "fallback-key"


def test_backend_alternates_providers_after_consecutive_timeouts() -> None:
    calls: list[dict[str, object]] = []

    def flaky_transport(url, body, headers, timeout_s):
        calls.append(
            {
                "url": url,
                "model": body["model"],
                "authorization": headers["Authorization"],
                "timeout_s": timeout_s,
            }
        )
        if len(calls) < 3:
            raise TimeoutError("provider read timed out")
        return _success_response()

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            provider="primary-compatible",
            model="primary-model",
            api_base="https://primary.example.test",
            api_key="primary-key",
            timeout_s=5.0,
            max_attempts=3,
            retry_backoff_s=0,
            fallback=_fallback(),
        ),
        transport=flaky_transport,
    )

    result = backend.decide(_request())

    assert result.status.value == "planned"
    assert [call["url"] for call in calls] == [
        "https://primary.example.test/v1/chat/completions",
        "https://fallback.example.test/v1/chat/completions",
        "https://primary.example.test/v1/chat/completions",
    ]
    assert [call["model"] for call in calls] == [
        "primary-model",
        "fallback-model",
        "primary-model",
    ]
    assert [call["authorization"] for call in calls] == [
        "Bearer primary-key",
        "Bearer fallback-key",
        "Bearer primary-key",
    ]
    assert [call["timeout_s"] for call in calls] == [5.0, 9.0, 5.0]
    assert result.provider == "primary-compatible"
    assert result.model == "primary-model"
    assert result.details["provider_role"] == "primary"
    assert result.details["provider_failover"] is True
    assert result.details["provider_switch_count"] == 2
    assert result.details["retry_errors"][0]["failover_next"] is True
    assert result.details["retry_errors"][1]["failover_next"] is False
    assert result.details["retry_errors"][0]["next_provider_role"] == "fallback"
    assert result.details["retry_errors"][1]["next_provider_role"] == "primary"


def test_backend_fails_over_after_api_key_rejection() -> None:
    urls: list[str] = []

    def rejected_primary_transport(url, body, headers, timeout_s):
        del body, headers, timeout_s
        urls.append(url)
        if len(urls) == 1:
            raise ProviderHttpError(401, "invalid api key")
        return _success_response()

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="primary-model",
            api_base="https://primary.example.test",
            api_key="primary-key",
            max_attempts=2,
            retry_backoff_s=0,
            fallback=_fallback(),
        ),
        transport=rejected_primary_transport,
    )

    result = backend.decide(_request())

    assert result.status.value == "planned"
    assert urls == [
        "https://primary.example.test/v1/chat/completions",
        "https://fallback.example.test/v1/chat/completions",
    ]
    assert result.details["provider_failover"] is True


def test_backend_tries_each_provider_once_for_persistent_quota_failure() -> None:
    urls: list[str] = []

    def quota_exhausted_transport(url, body, headers, timeout_s):
        del body, headers, timeout_s
        urls.append(url)
        raise ProviderHttpError(
            403,
            '{"error":{"code":"insufficient_user_quota",'
            '"message":"insufficient balance"}}',
        )

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="primary-model",
            api_base="https://primary.example.test",
            api_key="primary-key",
            max_attempts=3,
            retry_backoff_s=0,
            fallback=_fallback(),
        ),
        transport=quota_exhausted_transport,
    )

    result = backend.decide(_request())

    assert result.status.value == "failed"
    assert urls == [
        "https://primary.example.test/v1/chat/completions",
        "https://fallback.example.test/v1/chat/completions",
    ]
    assert result.details["provider_error_code"] == "insufficient_provider_quota"
    assert result.details["retryable"] is False
    assert result.details["provider_switch_count"] == 1
    assert result.payload["parameters"]["provider_error_code"] == (
        "insufficient_provider_quota"
    )
    assert result.payload["parameters"]["retryable"] is False


def test_backend_fails_over_after_provider_overload() -> None:
    urls: list[str] = []

    def overloaded_primary_transport(url, body, headers, timeout_s):
        del body, headers, timeout_s
        urls.append(url)
        if len(urls) == 1:
            raise ProviderHttpError(503, "system cpu overloaded")
        return _success_response()

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="primary-model",
            api_base="https://primary.example.test",
            api_key="primary-key",
            max_attempts=2,
            retry_backoff_s=0,
            fallback=_fallback(),
        ),
        transport=overloaded_primary_transport,
        sleep=lambda _delay_s: None,
    )

    result = backend.decide(_request())

    assert result.status.value == "planned"
    assert urls == [
        "https://primary.example.test/v1/chat/completions",
        "https://fallback.example.test/v1/chat/completions",
    ]
    assert result.details["provider_role"] == "fallback"
    assert result.details["provider_failover"] is True
    assert result.details["retry_errors"][0]["next_provider_role"] == "fallback"


def test_backend_keeps_successful_fallback_as_next_call_preference() -> None:
    urls: list[str] = []

    def primary_timeout_transport(url, body, headers, timeout_s):
        del body, headers, timeout_s
        urls.append(url)
        if len(urls) == 1:
            raise TimeoutError("primary timed out")
        return _success_response()

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="primary-model",
            api_base="https://primary.example.test",
            api_key="primary-key",
            max_attempts=3,
            retry_backoff_s=0,
            fallback=_fallback(),
        ),
        transport=primary_timeout_transport,
    )

    first = backend.decide(_request())
    second = backend.decide(_request())

    assert first.status.value == "planned"
    assert first.details["provider_role"] == "fallback"
    assert first.details["provider_switch_count"] == 1
    assert second.status.value == "planned"
    assert second.details["provider_role"] == "fallback"
    assert second.details["provider_attempts"] == 1
    assert second.details["provider_switch_count"] == 0
    assert urls == [
        "https://primary.example.test/v1/chat/completions",
        "https://fallback.example.test/v1/chat/completions",
        "https://fallback.example.test/v1/chat/completions",
    ]


def test_backend_keeps_primary_for_generic_server_error() -> None:
    urls: list[str] = []

    def server_error_transport(url, body, headers, timeout_s):
        del body, headers, timeout_s
        urls.append(url)
        if len(urls) == 1:
            raise ProviderHttpError(500, "internal server error")
        return _success_response()

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="primary-model",
            api_base="https://primary.example.test",
            api_key="primary-key",
            max_attempts=2,
            retry_backoff_s=0,
            fallback=_fallback(),
        ),
        transport=server_error_transport,
    )

    result = backend.decide(_request())

    assert result.status.value == "planned"
    assert urls == [
        "https://primary.example.test/v1/chat/completions",
        "https://primary.example.test/v1/chat/completions",
    ]
    assert result.details["provider_role"] == "primary"
    assert result.details["provider_failover"] is False
    assert result.details["retry_errors"][0]["failover_next"] is False


def test_backend_fails_over_for_capacity_error_reported_as_http_500() -> None:
    urls: list[str] = []

    def capacity_error_transport(url, body, headers, timeout_s):
        del body, headers, timeout_s
        urls.append(url)
        if len(urls) == 1:
            raise ProviderHttpError(500, "model concurrency capacity overloaded")
        return _success_response()

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="primary-model",
            api_base="https://primary.example.test",
            api_key="primary-key",
            max_attempts=2,
            retry_backoff_s=0,
            fallback=_fallback(),
        ),
        transport=capacity_error_transport,
        sleep=lambda _delay_s: None,
    )

    result = backend.decide(_request())

    assert result.status.value == "planned"
    assert urls == [
        "https://primary.example.test/v1/chat/completions",
        "https://fallback.example.test/v1/chat/completions",
    ]
    assert result.details["provider_role"] == "fallback"
    assert result.details["provider_failover"] is True
    assert result.details["retry_errors"][0]["next_provider_role"] == "fallback"


def test_backend_fails_over_for_temporarily_unavailable_http_500() -> None:
    urls: list[str] = []

    def unavailable_transport(url, body, headers, timeout_s):
        del body, headers, timeout_s
        urls.append(url)
        if len(urls) == 1:
            raise ProviderHttpError(500, "Service temporarily unavailable")
        return _success_response()

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="primary-model",
            api_base="https://primary.example.test",
            api_key="primary-key",
            max_attempts=2,
            retry_backoff_s=0,
            fallback=_fallback(),
        ),
        transport=unavailable_transport,
        sleep=lambda _delay_s: None,
    )

    result = backend.decide(_request())

    assert result.status.value == "planned"
    assert urls == [
        "https://primary.example.test/v1/chat/completions",
        "https://fallback.example.test/v1/chat/completions",
    ]
    assert result.details["provider_role"] == "fallback"
    assert result.details["provider_failover"] is True
    assert result.details["retry_errors"][0]["next_provider_role"] == "fallback"


def test_backend_applies_capacity_cooldown_floor_before_retrying() -> None:
    delays: list[float] = []
    calls = 0

    def overloaded_transport(url, body, headers, timeout_s):
        nonlocal calls
        del url, body, headers, timeout_s
        calls += 1
        if calls < 3:
            raise ProviderHttpError(500, "model concurrency capacity overloaded")
        return _success_response()

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="primary-model",
            api_base="https://primary.example.test",
            api_key="primary-key",
            max_attempts=3,
            retry_backoff_s=0.5,
            fallback=_fallback(),
        ),
        transport=overloaded_transport,
        sleep=delays.append,
    )

    result = backend.decide(_request())

    assert result.status.value == "planned"
    assert delays == [
        PROVIDER_CAPACITY_MIN_RETRY_BACKOFF_S,
        PROVIDER_CAPACITY_MIN_RETRY_BACKOFF_S * 2,
    ]
    assert [entry["retry_delay_s"] for entry in result.details["retry_errors"]] == delays


def test_backend_retries_and_fails_over_for_incomplete_http_body() -> None:
    urls: list[str] = []

    def incomplete_transport(url, body, headers, timeout_s):
        del body, headers, timeout_s
        urls.append(url)
        if len(urls) == 1:
            raise http.client.IncompleteRead(b"partial", 100)
        return _success_response()

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="primary-model",
            api_base="https://primary.example.test",
            api_key="primary-key",
            max_attempts=3,
            retry_backoff_s=0,
            fallback=_fallback(),
        ),
        transport=incomplete_transport,
    )

    result = backend.decide(_request())

    assert result.status.value == "planned"
    assert urls == [
        "https://primary.example.test/v1/chat/completions",
        "https://fallback.example.test/v1/chat/completions",
    ]
    assert result.details["provider_attempts"] == 2
    assert result.details["provider_role"] == "fallback"
    assert result.details["retry_errors"][0]["error_type"] == "IncompleteRead"


def test_backend_fails_over_for_empty_success_response() -> None:
    urls: list[str] = []

    def empty_response_transport(url, body, headers, timeout_s):
        del body, headers, timeout_s
        urls.append(url)
        if len(urls) == 1:
            return {
                "choices": [
                    {"message": {"content": ""}, "finish_reason": "stop"},
                ]
            }
        return _success_response()

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="primary-model",
            api_base="https://primary.example.test",
            api_key="primary-key",
            max_attempts=2,
            retry_backoff_s=0,
            fallback=_fallback(),
        ),
        transport=empty_response_transport,
    )

    result = backend.decide(_request())

    assert result.status.value == "planned"
    assert urls == [
        "https://primary.example.test/v1/chat/completions",
        "https://fallback.example.test/v1/chat/completions",
    ]
    assert result.details["provider_role"] == "fallback"
    assert result.details["provider_attempts"] == 2
    assert result.details["provider_failover"] is True
    assert result.details["retry_errors"][0]["error_type"] == "ProviderProtocolError"
    assert result.details["retry_errors"][0]["provider_response"] == {
        "choice_count": 1,
        "finish_reason": "stop",
        "content_type": "str",
        "content_chars": 0,
        "refusal_present": False,
    }
