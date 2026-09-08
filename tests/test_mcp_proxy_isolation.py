"""Client-scoped MCP proxy routing; all requests below are local fixtures."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
import threading

import httpx
import mcp
import mcp.client.sse
from mcp.types import CallToolResult, ListToolsResult, TextContent
import pytest

from agent.tools import sim_mcp


PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
              "http_proxy", "https_proxy", "all_proxy", "no_proxy")


@pytest.fixture
def proxy_env(monkeypatch):
    for key in PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("NO_PROXY", "baseline.invalid")
    return {key: os.environ.get(key) for key in PROXY_KEYS}


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:18766/sse", "http://localhost:18766/sse",
    "http://[::1]:18766/sse", "https://MCP.EXAMPLE/sse", "https://例子.测试/sse",
])
def test_only_target_host_bypasses_proxy_without_disabling_tls_environment(proxy_env, url):
    async def check():
        async with sim_mcp._mcp_host_http_client_factory(url)() as client:
            target = httpx.URL(url)
            assert client.trust_env is True
            assert client._transport_for_url(target) is client._transport
            # Preserve the old target-host bypass across ports, not other hosts.
            assert client._transport_for_url(target.copy_with(port=9999)) is client._transport
            assert client._transport_for_url(httpx.URL("http://other.invalid/")) is not client._transport
            assert client._transport_for_url(httpx.URL("https://other.invalid/")) is not client._transport
            assert client._transport_for_url(httpx.URL("http://baseline.invalid/")) is client._transport
            assert {key: os.environ.get(key) for key in PROXY_KEYS} == proxy_env
        assert client.is_closed

    asyncio.run(check())
    assert {key: os.environ.get(key) for key in PROXY_KEYS} == proxy_env


def test_two_mcp_clients_and_unrelated_client_have_independent_routes(proxy_env):
    async def check():
        async with (
            sim_mcp._mcp_host_http_client_factory("http://a.invalid/sse")() as a,
            sim_mcp._mcp_host_http_client_factory("http://b.invalid/sse")() as b,
            httpx.AsyncClient() as unrelated,
        ):
            for client, own, other in ((a, "a", "b"), (b, "b", "a")):
                assert client._transport_for_url(httpx.URL(f"http://{own}.invalid/")) is client._transport
                assert client._transport_for_url(httpx.URL(f"http://{other}.invalid/")) is not client._transport
            assert unrelated._transport_for_url(httpx.URL("http://a.invalid/")) is not unrelated._transport
            assert unrelated._transport_for_url(httpx.URL("http://b.invalid/")) is not unrelated._transport
    asyncio.run(check())


def test_redirect_to_other_host_still_uses_environment_proxy(proxy_env, monkeypatch):
    requests = []

    async def request(transport, request):
        requests.append((transport, str(request.url)))
        if request.url.host == "mcp.invalid":
            return httpx.Response(302, headers={"Location": "https://other.invalid/final"})
        return httpx.Response(200, json={"fixture": True})

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", request)

    async def check():
        async with sim_mcp._mcp_host_http_client_factory("http://mcp.invalid/sse")() as client:
            result = await client.get("http://mcp.invalid/start")
            assert result.json() == {"fixture": True}
            assert requests[0][0] is client._transport
            assert requests[1][0] is not client._transport
    asyncio.run(check())
    assert [url for _, url in requests] == ["http://mcp.invalid/start", "https://other.invalid/final"]


def test_factory_preserves_sdk_headers_timeout_auth_and_ca_loading(proxy_env, monkeypatch):
    seen = []
    original_init = httpx.AsyncHTTPTransport.__init__

    def initialize(self, *args, **kwargs):
        seen.append({key: kwargs.get(key) for key in ("verify", "trust_env")})
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "__init__", initialize)
    auth = httpx.BasicAuth("fixture", "fixture")
    timeout = httpx.Timeout(5.0, read=405.0)

    async def check():
        async with sim_mcp._mcp_host_http_client_factory("http://mcp.invalid/sse")(
            headers={"X-Fixture": "yes"}, timeout=timeout, auth=auth,
        ) as client:
            assert client.headers["X-Fixture"] == "yes"
            assert client.timeout == timeout
            assert client.auth is auth
            assert client.follow_redirects is True
    asyncio.run(check())
    assert seen and all(row == {"verify": True, "trust_env": True} for row in seen)


def test_invalid_environment_ca_file_still_fails_closed(proxy_env, monkeypatch, tmp_path):
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing-ca.pem"))
    with pytest.raises(FileNotFoundError):
        sim_mcp._mcp_host_http_client_factory("https://mcp.invalid/sse")()


@pytest.mark.parametrize("operation", ["list", "call"])
@pytest.mark.parametrize("outcome", ["success", "failure", "timeout"])
def test_sdk_entry_points_use_scoped_factory_and_close_clients(proxy_env, monkeypatch, operation, outcome):
    clients = []
    received = []

    @asynccontextmanager
    async def sse(url, *, sse_read_timeout, httpx_client_factory):
        client = httpx_client_factory(timeout=httpx.Timeout(5.0, read=sse_read_timeout))
        clients.append(client)
        async with client:
            assert client._transport_for_url(httpx.URL(url)) is client._transport
            assert {key: os.environ.get(key) for key in PROXY_KEYS} == proxy_env
            yield "read", "write"

    class Session:
        def __init__(self, read, write):
            assert (read, write) == ("read", "write")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def initialize(self):
            if outcome == "failure":
                raise RuntimeError("fixture failure")
            if outcome == "timeout":
                await asyncio.Event().wait()

        async def list_tools(self):
            return ListToolsResult(tools=[])

        async def call_tool(self, name, arguments, **kwargs):
            received.append((name, arguments))
            return CallToolResult(content=[TextContent(type="text", text='{"success":true}')])

    monkeypatch.setattr(mcp.client.sse, "sse_client", sse)
    monkeypatch.setattr(mcp, "ClientSession", Session)
    transport = sim_mcp.SseSimulatorMcpTransport("http://mcp.invalid/sse")

    def execute():
        if operation == "list":
            return transport.list_tools(timeout_s=0.2)
        return transport.call_tool("fixture", {"value": 1}, timeout_s=0.2)

    if outcome == "success":
        result = execute()
        assert result.get("tool_count") == 0 if operation == "list" else result["success"] is True
        if operation == "call":
            assert received == [("fixture", {"value": 1})]
    else:
        with pytest.raises(sim_mcp.SimulatorMcpTransportError) as error:
            execute()
        if outcome == "timeout":
            assert error.value.code == "simulator_mcp_transport_timeout"
    assert clients and all(client.is_closed for client in clients)
    assert {key: os.environ.get(key) for key in PROXY_KEYS} == proxy_env


def test_overlapping_calls_do_not_remove_other_bypass_or_restore_stale_environment(proxy_env, monkeypatch):
    entered = {name: threading.Event() for name in ("a", "b")}
    release = {name: threading.Event() for name in ("a", "b")}
    errors = []

    async def listing(*, url, timeout_s):
        name = httpx.URL(url).host.split(".")[0]
        entered[name].set()
        assert release[name].wait(2)
        return {"tools": [], "tool_count": 0}

    def run(name):
        try:
            sim_mcp.SseSimulatorMcpTransport(f"http://{name}.invalid/sse").list_tools(timeout_s=3)
        except Exception as error:
            errors.append(error)

    monkeypatch.setattr(sim_mcp, "_list_sse_mcp_tools", listing)
    threads = [threading.Thread(target=run, args=(name,)) for name in ("a", "b")]
    try:
        for thread in threads:
            thread.start()
        assert all(event.wait(1) for event in entered.values())
        assert {key: os.environ.get(key) for key in PROXY_KEYS} == proxy_env
        monkeypatch.setenv("NO_PROXY", "new-owner.invalid")
        release["a"].set()
        threads[0].join(1)
        assert not threads[0].is_alive() and threads[1].is_alive()
        assert os.environ["NO_PROXY"] == "new-owner.invalid"
    finally:
        for event in release.values():
            event.set()
        for thread in threads:
            thread.join(2)
    assert not errors
    assert not any(thread.is_alive() for thread in threads)
    assert os.environ["NO_PROXY"] == "new-owner.invalid"
    assert "no_proxy" not in os.environ
