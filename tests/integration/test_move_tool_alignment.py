"""Integration test: simulator backends and move tools must be frame-aligned.

Runs the atomic contract checks from
``scripts/embodied/verify_move_tool_alignment.py`` against a locally launched
simulator MCP server, once per backend.  Env ids are discovered through
``list_envs`` at runtime, so a backend is covered automatically as soon as its
bench venv exists on the machine; otherwise that backend is skipped.

Run a local server first to get real coverage::

    PYTHONPATH=. .venv/bin/python sim/mcp_server/server.py --port 8769 &
    OPENETA_TEST_SIMULATOR_URL=http://127.0.0.1:8769/sse \
      .venv/bin/python -m pytest tests/integration/test_move_tool_alignment.py
"""

from __future__ import annotations

import argparse
import os
import socket
import urllib.parse

import pytest

from agent.tools.sim_mcp import SseSimulatorMcpTransport
from scripts.embodied.verify_move_tool_alignment import run_alignment_checks

DEFAULT_URL = "http://127.0.0.1:8769/sse"

# Backend -> preferred env-id prefixes, most specific first.
BACKENDS: dict[str, tuple[str, ...]] = {
    "libero": ("openeta/libero_libero_object_task0-v0", "openeta/libero_"),
    "metaworld": ("openeta/metaworld_50_assembly", "openeta/metaworld_50_"),
    "maniskill": ("openeta/maniskill_PickCube-v1-v0", "openeta/maniskill_Pick"),
    "robocasa": ("openeta/robocasa_target_", "openeta/robocasa_"),
}


def _server_reachable(url: str, timeout_s: float = 2.0) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _discover_env(
    transport: SseSimulatorMcpTransport, prefixes: tuple[str, ...]
) -> tuple[str, str] | None:
    """Return (env_id, task) for the first registered env matching a prefix."""
    payload = transport.call_tool("list_envs", {}, timeout_s=30.0)
    envs = payload.get("envs", []) if isinstance(payload, dict) else []
    for prefix in prefixes:
        if prefix.endswith("-v0"):
            for env in envs:
                if isinstance(env, dict) and env.get("id") == prefix:
                    return prefix, str(env.get("description") or "")
            continue
        for env in envs:
            if isinstance(env, dict) and str(env.get("id", "")).startswith(prefix):
                return str(env["id"]), str(env.get("description") or "")
    return None


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_move_tool_alignment_backend(backend: str) -> None:
    url = os.environ.get("OPENETA_TEST_SIMULATOR_URL", DEFAULT_URL)
    if not _server_reachable(url):
        pytest.skip(f"no simulator MCP server reachable at {url}")

    transport = SseSimulatorMcpTransport(url)
    discovered = _discover_env(transport, BACKENDS[backend])
    if discovered is None:
        pytest.skip(f"backend {backend} has no registered env on {url}")
    env_id, task = discovered

    args = argparse.Namespace(
        simulator_url=url,
        env_id=env_id,
        task=task,
        seed=0,
        delta_m=0.02,
        tolerance_m=0.01,
    )
    results = run_alignment_checks(args)
    if any(r.name == "create_env" and not r.ok for r in results):
        pytest.skip(f"backend {backend} env {env_id} could not be created")
    failures = [r for r in results if not r.ok]
    assert not failures, f"move-tool alignment failures on {env_id}: " + "; ".join(
        f"{r.name} ({r.detail})" for r in failures
    )
