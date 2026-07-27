from __future__ import annotations

from pathlib import Path

from agent.backends.planner import (
    OpenAICompatiblePlannerBackend,
    OpenAICompatiblePlannerBackendConfig,
    PlannerBackendRequest,
    ProviderHttpError,
)
from agent.backends.provider_config import (
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
