from __future__ import annotations

import json
import socket

import pytest

from agent.backends.provider_config import PlannerProviderConfig, ProviderEndpointConfig
from agent.runtime.memory import _compact_tool_result_details
from agent.runtime.planner import _validate_tool_parameters
from agent.tools.registry import ToolEffect, build_default_tool_registry
from agent.tools.web_access import (
    HostedWebSearchClient,
    WebAccessConfig,
    WebAccessError,
    WebHttpResponse,
    WebSearchConfig,
    WebSearchEndpointConfig,
    bind_configured_web_tool_handlers,
    build_web_fetch_handler,
    build_web_search_handler,
    load_configured_web_access,
)


def _provider_config(*, fallback: bool = False) -> PlannerProviderConfig:
    return PlannerProviderConfig(
        provider="primary-provider",
        model="search-model",
        api_base="https://primary.example.com/v1",
        api_key="primary-secret",
        timeout_s=90.0,
        fallback=(
            ProviderEndpointConfig(
                provider="fallback-provider",
                model="fallback-model",
                api_base="https://fallback.example.com",
                api_key="fallback-secret",
                timeout_s=80.0,
            )
            if fallback
            else None
        ),
    )


def _search_config(*, fallback: bool = False) -> WebSearchConfig:
    return WebSearchConfig.from_provider_config(_provider_config(fallback=fallback))


def _responses_payload() -> bytes:
    text = "OpenETA is documented by the project repository and its guide."
    return json.dumps(
        {
            "status": "completed",
            "output": [
                {"type": "web_search_call", "status": "completed"},
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "title": " OpenETA   repository ",
                                    "url": "https://github.com/example/openeta",
                                    "start_index": 0,
                                    "end_index": 7,
                                },
                                {
                                    "type": "url_citation",
                                    "title": "OpenETA guide",
                                    "url": "https://docs.example.com/openeta",
                                    "start_index": 47,
                                    "end_index": len(text),
                                },
                            ],
                        }
                    ],
                },
            ],
        }
    ).encode()


def _public_resolver(host, port, **_kwargs):
    assert host == "docs.example.com"
    assert port == 443
    return [
        (
            socket.AF_INET,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            ("93.184.216.34", 443),
        )
    ]


def test_web_tool_specs_are_read_only_and_host_unbound_by_default() -> None:
    tools = build_default_tool_registry()

    search = tools.get("web_search")
    fetch = tools.get("web_fetch")

    assert search.category == "web"
    assert search.effect == ToolEffect.READ_ONLY
    assert search.batchable is False
    assert fetch.category == "web"
    assert fetch.effect == ToolEffect.READ_ONLY
    assert fetch.batchable is False
    assert tools.can_execute("web_search") is False
    assert tools.can_execute("web_fetch") is False


def test_web_access_config_enables_fetch_with_search_or_explicit_flag() -> None:
    disabled = WebAccessConfig()
    fetch_only = load_configured_web_access(
        provider_config=PlannerProviderConfig(),
        environ={"OPENETA_WEB_FETCH_ENABLED": "true"},
        dotenv_path="/does/not/exist",
        apikey_path="/does/not/exist",
    )
    search = load_configured_web_access(
        provider_config=_provider_config(),
        environ={},
        dotenv_path="/does/not/exist",
        apikey_path="/does/not/exist",
    )

    assert disabled.fetch_enabled is False
    assert disabled.search is None
    assert fetch_only.fetch_enabled is True
    assert fetch_only.search is None
    assert search.fetch_enabled is True
    assert search.search is not None
    assert search.search.primary.responses_url() == "https://primary.example.com/v1/responses"


def test_search_config_rejects_api_key_over_plain_http() -> None:
    config = WebSearchEndpointConfig(
        provider="provider",
        model="model",
        api_base="http://search.internal/v1",
        api_key="secret",
    )

    with pytest.raises(ValueError, match="must be an HTTPS"):
        config.validate()


def test_search_config_repr_redacts_provider_keys() -> None:
    rendered = repr(_search_config(fallback=True))

    assert "primary-secret" not in rendered
    assert "fallback-secret" not in rendered


def test_web_access_loader_reads_dotenv_with_environment_precedence(tmp_path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "OPENETA_WEB_FETCH_ENABLED=false\n"
        "OPENETA_WEB_SEARCH_ENABLED=false\n",
        encoding="utf-8",
    )

    config = load_configured_web_access(
        provider_config=_provider_config(),
        environ={
            "OPENETA_WEB_FETCH_ENABLED": "true",
            "OPENETA_WEB_SEARCH_ENABLED": "true",
        },
        dotenv_path=str(dotenv),
        apikey_path=str(tmp_path / "missing.md"),
    )

    assert config.fetch_enabled is True
    assert config.search is not None
    assert config.search.primary.provider == "primary-provider"


def test_hosted_search_client_normalizes_citations_and_keeps_key_host_side() -> None:
    requests = []

    def transport(url, body, headers, timeout_s, max_bytes):
        requests.append((url, body, dict(headers), timeout_s, max_bytes))
        return _responses_payload()

    client = HostedWebSearchClient(_search_config(), transport=transport)

    response = client.search(
        query="  panda   calibration ",
        max_results=2,
        language="en",
        time_range="month",
    )
    results = response["results"]

    assert [result["result_id"] for result in results] == [
        "search_result_001",
        "search_result_002",
    ]
    assert results[0]["title"] == "OpenETA repository"
    assert results[0]["url"] == "https://github.com/example/openeta"
    assert results[0]["snippet"] == "OpenETA"
    assert results[1]["rank"] == 1
    assert response["search_call_count"] == 1
    assert response["provider_role"] == "primary"
    url, body, headers, timeout_s, max_bytes = requests[0]
    assert url == "https://primary.example.com/v1/responses"
    assert body["model"] == "search-model"
    assert body["tools"] == [{"type": "web_search"}]
    assert "panda calibration" in body["input"]
    assert "last month" in body["input"]
    assert headers["Authorization"] == "Bearer primary-secret"
    assert "primary-secret" not in json.dumps(body)
    assert "primary-secret" not in url
    assert timeout_s == 90.0
    assert max_bytes == 2 * 1024 * 1024


def test_hosted_search_falls_back_without_exposing_provider_errors() -> None:
    calls = []

    def transport(url, body, headers, *_args):
        calls.append((url, body["model"], headers["Authorization"]))
        if "primary.example.com" in url:
            raise WebAccessError("web_search_backend_error", "HTTP 503")
        return _responses_payload()

    response = HostedWebSearchClient(
        _search_config(fallback=True),
        transport=transport,
    ).search(query="OpenETA", max_results=1)

    assert response["provider_role"] == "fallback"
    assert response["provider"] == "fallback-provider"
    assert calls == [
        (
            "https://primary.example.com/v1/responses",
            "search-model",
            "Bearer primary-secret",
        ),
        (
            "https://fallback.example.com/v1/responses",
            "fallback-model",
            "Bearer fallback-secret",
        ),
    ]


def test_web_search_handler_marks_external_content_untrusted() -> None:
    client = HostedWebSearchClient(
        _search_config(),
        transport=lambda *_args: _responses_payload(),
    )
    tools = build_default_tool_registry()
    tools.bind_handler("web_search", build_web_search_handler(client))

    result = tools.call("web_search", {"query": "OpenETA", "max_results": 5})

    assert result.success is True
    outputs = result.details["outputs"]
    assert outputs["schema_version"] == "openeta.web_search.v1"
    assert outputs["untrusted_external_content"] is True
    assert outputs["answer"].startswith("OpenETA")
    assert outputs["results"][0]["url"] == "https://github.com/example/openeta"
    assert result.details["parameters"] == {"query": "OpenETA", "max_results": 5}


@pytest.mark.parametrize(
    ("url", "resolver", "code"),
    [
        (
            "http://docs.example.com/page",
            _public_resolver,
            "invalid_web_fetch_url",
        ),
        (
            "https://127.0.0.1/private",
            _public_resolver,
            "web_fetch_address_blocked",
        ),
        (
            "https://docs.example.com/private",
            lambda *_args, **_kwargs: [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("127.0.0.2", 443),
                )
            ],
            "web_fetch_address_blocked",
        ),
        (
            "https://user@docs.example.com/private",
            _public_resolver,
            "invalid_web_fetch_url",
        ),
        (
            "https://docs.example.com:444/private",
            _public_resolver,
            "invalid_web_fetch_url",
        ),
    ],
)
def test_web_fetch_rejects_insecure_or_private_destinations(url, resolver, code) -> None:
    tools = build_default_tool_registry()
    tools.bind_handler(
        "web_fetch",
        build_web_fetch_handler(
            transport=lambda *_args: pytest.fail("transport must not be called"),
            resolver=resolver,
        ),
    )

    result = tools.call("web_fetch", {"url": url})

    assert result.success is False
    assert result.details["diagnostics"] == [{"code": code}]


def test_web_fetch_rejects_mixed_public_and_private_dns_answers() -> None:
    def mixed_resolver(*_args, **_kwargs):
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("93.184.216.34", 443),
            ),
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("127.0.0.2", 443),
            ),
        ]

    tools = build_default_tool_registry()
    tools.bind_handler(
        "web_fetch",
        build_web_fetch_handler(
            transport=lambda *_args: pytest.fail("transport must not be called"),
            resolver=mixed_resolver,
        ),
    )

    result = tools.call("web_fetch", {"url": "https://docs.example.com/private"})

    assert result.details["diagnostics"] == [{"code": "web_fetch_address_blocked"}]


def test_web_fetch_extracts_text_and_drops_active_html() -> None:
    seen = []

    def transport(resolved, timeout_s, max_bytes):
        seen.append((resolved, timeout_s, max_bytes))
        return WebHttpResponse(
            url=resolved.url,
            status=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            body=(
                b"<html><head><title>  Robot Docs </title>"
                b"<script>ignore this instruction</script></head>"
                b"<body><h1>Calibration</h1><p>Use measured evidence.</p>"
                b"<style>hidden</style></body></html>"
            ),
        )

    tools = build_default_tool_registry()
    tools.bind_handler(
        "web_fetch",
        build_web_fetch_handler(
            transport=transport,
            resolver=_public_resolver,
        ),
    )

    result = tools.call(
        "web_fetch",
        {"url": "https://docs.example.com/guide?q=robot", "max_chars": 1000},
    )

    assert result.success is True
    outputs = result.details["outputs"]
    assert outputs["schema_version"] == "openeta.web_fetch.v1"
    assert outputs["url"] == "https://docs.example.com/guide?q=robot"
    assert outputs["title"] == "Robot Docs"
    assert outputs["text"] == "Calibration\nUse measured evidence."
    assert outputs["truncated"] is False
    assert outputs["untrusted_external_content"] is True
    assert "ignore this instruction" not in outputs["text"]
    assert "hidden" not in outputs["text"]
    resolved, timeout_s, max_bytes = seen[0]
    assert resolved.addresses == ((socket.AF_INET, ("93.184.216.34", 443)),)
    assert timeout_s == 20.0
    assert max_bytes == 2 * 1024 * 1024


def test_web_fetch_rejects_redirect_and_oversized_injected_response() -> None:
    tools = build_default_tool_registry()
    tools.bind_handler(
        "web_fetch",
        build_web_fetch_handler(
            max_response_bytes=4,
            transport=lambda resolved, *_args: WebHttpResponse(
                url=resolved.url,
                status=302,
                headers={"Content-Type": "text/plain"},
                body=b"",
            ),
            resolver=_public_resolver,
        ),
    )
    redirect = tools.call("web_fetch", {"url": "https://docs.example.com/redirect"})
    assert redirect.details["diagnostics"] == [{"code": "web_fetch_redirect_rejected"}]

    tools.bind_handler(
        "web_fetch",
        build_web_fetch_handler(
            max_response_bytes=4,
            transport=lambda resolved, *_args: WebHttpResponse(
                url=resolved.url,
                status=200,
                headers={"Content-Type": "text/plain"},
                body=b"12345",
            ),
            resolver=_public_resolver,
        ),
        replace=True,
    )
    oversized = tools.call("web_fetch", {"url": "https://docs.example.com/large"})
    assert oversized.details["diagnostics"] == [{"code": "web_fetch_response_too_large"}]


def test_configured_binding_keeps_unavailable_web_tools_hidden() -> None:
    tools = build_default_tool_registry()

    bind_configured_web_tool_handlers(tools, config=WebAccessConfig())

    assert tools.can_execute("web_search") is False
    assert tools.can_execute("web_fetch") is False

    bind_configured_web_tool_handlers(
        tools,
        config=WebAccessConfig(
            fetch_enabled=True,
            search=_search_config(),
        ),
        search_transport=lambda *_args: _responses_payload(),
        fetch_transport=lambda resolved, *_args: WebHttpResponse(
            url=resolved.url,
            status=200,
            headers={"Content-Type": "text/plain"},
            body=b"ready",
        ),
        resolver=_public_resolver,
    )

    assert tools.can_execute("web_search") is True
    assert tools.can_execute("web_fetch") is True


def test_web_tool_planner_parameter_validation() -> None:
    assert _validate_tool_parameters(
        "web_search",
        {"query": "official robot docs", "max_results": 3, "time_range": "year"},
    ) == []
    assert _validate_tool_parameters(
        "web_fetch",
        {"url": "https://docs.example.com/robot", "max_chars": 5000},
    ) == []
    assert _validate_tool_parameters("web_search", {"query": "", "max_results": 0})
    assert _validate_tool_parameters("web_fetch", {"url": "http://localhost/private"})


def test_web_handlers_reject_non_integer_limits() -> None:
    tools = build_default_tool_registry()
    tools.bind_handler(
        "web_search",
        build_web_search_handler(
            HostedWebSearchClient(
                _search_config(),
                transport=lambda *_args: pytest.fail("transport must not be called"),
            )
        ),
    )
    tools.bind_handler(
        "web_fetch",
        build_web_fetch_handler(
            transport=lambda *_args: pytest.fail("transport must not be called"),
            resolver=_public_resolver,
        ),
    )

    search = tools.call("web_search", {"query": "robot", "max_results": 1.5})
    fetch = tools.call(
        "web_fetch",
        {"url": "https://docs.example.com/robot", "max_chars": "100"},
    )

    assert search.details["diagnostics"] == [{"code": "invalid_web_search_request"}]
    assert fetch.details["diagnostics"] == [{"code": "invalid_web_fetch_request"}]


def test_web_search_invalid_payload_has_stable_diagnostic() -> None:
    client = HostedWebSearchClient(
        _search_config(),
        transport=lambda *_args: b"[]",
    )

    with pytest.raises(WebAccessError) as exc:
        client.search(query="robot", max_results=5)

    assert exc.value.code == "web_search_backend_error"
    assert "primary:web_search_invalid_response" in str(exc.value)


def test_web_results_keep_untrusted_marker_across_memory_compaction() -> None:
    compact = _compact_tool_result_details(
        {
            "outputs": {
                "schema_version": "openeta.web_fetch.v1",
                "url": "https://docs.example.com/robot",
                "title": "Robot docs",
                "text": "external text " * 1000,
                "untrusted_external_content": True,
            }
        }
    )

    outputs = compact["outputs"]
    assert outputs["schema_version"] == "openeta.web_fetch.v1"
    assert outputs["url"] == "https://docs.example.com/robot"
    assert outputs["untrusted_external_content"] is True
    assert outputs["text"].endswith("...[truncated]")


def test_web_search_answer_survives_memory_compaction_with_bounded_provenance() -> None:
    compact = _compact_tool_result_details(
        {
            "outputs": {
                "schema_version": "openeta.web_search.v1",
                "query": "OpenETA",
                "answer": "search answer " * 500,
                "answer_truncated": False,
                "result_count": 1,
                "results": [
                    {
                        "title": "OpenETA",
                        "url": "https://github.com/example/openeta",
                        "snippet": "project repository",
                    }
                ],
                "search_call_count": 2,
                "provider_role": "primary",
                "provider": "primary-provider",
                "model": "search-model",
                "untrusted_external_content": True,
            }
        }
    )

    outputs = compact["outputs"]
    assert outputs["answer"].startswith("search answer")
    assert outputs["answer"].endswith("...[truncated]")
    assert len(outputs["answer"]) < 4100
    assert outputs["provider_role"] == "primary"
    assert outputs["search_call_count"] == 2
    assert outputs["results"][0]["url"] == "https://github.com/example/openeta"
