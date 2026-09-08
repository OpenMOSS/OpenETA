"""Real SDK + loopback HTTP transport fixture, not a harness task experiment."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import socket
import threading

from mcp.server.fastmcp import FastMCP
import pytest
import uvicorn

from agent.tools.sim_mcp import SseSimulatorMcpTransport


def test_real_sse_roundtrips_bypass_proxy_without_changing_environment(monkeypatch):
    # Only fixed test strings are sent to this ephemeral in-process server.
    for name in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "baseline.invalid")
    names = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "no_proxy")
    baseline = {name: os.environ.get(name) for name in names}
    mcp = FastMCP("proxy-loopback-fixture", log_level="ERROR")

    @mcp.tool()
    def echo_fixture(value: str) -> dict:
        return {"success": True, "value": value}

    listener = socket.socket()
    try:
        listener.bind(("127.0.0.1", 0))
    except PermissionError:
        listener.close()
        pytest.skip("Loopback sockets require an explicitly allowed integration run")
    listener.listen()
    port = listener.getsockname()[1]
    ready = threading.Event()

    class Server(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            ready.set()

    server = Server(uvicorn.Config(mcp.sse_app(), log_level="error", lifespan="off"))
    errors = []

    def serve():
        try:
            server.run(sockets=[listener])
        except BaseException as error:
            errors.append(error)
            ready.set()

    worker = threading.Thread(target=serve)
    worker.start()
    try:
        assert ready.wait(5) and not errors and server.started
        transport = SseSimulatorMcpTransport(f"http://127.0.0.1:{port}/sse")
        catalog = transport.list_tools(timeout_s=5)
        assert "echo_fixture" in {tool["name"] for tool in catalog["tools"]}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(transport.call_tool, "echo_fixture", {"value": value}, timeout_s=5)
                       for value in ("fixture-a", "fixture-b")]
            assert [future.result(timeout=7)["value"] for future in futures] == ["fixture-a", "fixture-b"]
        assert {name: os.environ.get(name) for name in names} == baseline
    finally:
        server.should_exit = True
        worker.join(7)
        if worker.is_alive():
            server.force_exit = True
            worker.join(2)
        listener.close()
    assert not worker.is_alive()
    assert not errors
