"""Live camera-view page and SSE live-streaming endpoints.

This server is agent-facing: environment control happens through the MCP
tools, not over HTTP.  The only HTTP surface here is the read-only live
camera view that ``create_env`` points users at, plus the SSE endpoints
that push rendered camera frames to that page.  Each handler is a
Starlette-compatible async function.
"""

from __future__ import annotations

import asyncio
import json

from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse

from sim.mcp_server.dashboard_html import _SESSION_DASHBOARD_HTML
from sim.mcp_server.session import (
    _session_envs,
    _session_stream_tasks,
    _session_streams,
    _session_stream_interval,
)
from sim.mcp_server.worker_mgr import (
    _live_stream_loop,
)


# ══════════════════════════════════════════════════════════════════════
# Session env listing (read-only, used by the live camera view page)
# ══════════════════════════════════════════════════════════════════════

async def session_envs(request):
    """Return the list of active envs for a session."""
    sid = request.path_params.get("sid", "")
    envs = _session_envs.get(sid, {})
    entries: list[dict] = []
    for h, meta in envs.items():
        entries.append({
            "handle": h,
            "remote_handle": meta.get("remote_handle", ""),
            "env_id": meta.get("env_id", "unknown"),
            "backend": meta.get("backend", "unknown"),
        })
    return JSONResponse({"session_id": sid, "count": len(entries), "envs": entries})


# ══════════════════════════════════════════════════════════════════════
# Live-stream SSE endpoints
# ══════════════════════════════════════════════════════════════════════

async def session_stream(request):
    """SSE endpoint: push rendered frames for all envs at the configured interval."""
    sid = request.path_params.get("sid", "")
    if not sid or sid not in _session_envs:
        return JSONResponse({"error": "Unknown session"}, 404)

    interval_s = _session_stream_interval.get(sid, 0.05)

    async def event_stream():
        q: asyncio.Queue = asyncio.Queue(maxsize=4)
        streams = _session_streams.setdefault(sid, set())
        streams.add(q)

        # Start render loop if not already running
        if sid not in _session_stream_tasks or _session_stream_tasks[sid].done():
            _session_stream_tasks[sid] = asyncio.create_task(
                _live_stream_loop(sid, interval_s, stream_key=sid)
            )

        try:
            yield f"event: config\ndata: {json.dumps({'interval_ms': int(interval_s * 1000)})}\n\n"
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=30)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            streams.discard(q)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def session_env_stream(request):
    """SSE endpoint: push rendered frames for a specific env handle."""
    sid = request.path_params.get("sid", "")
    handle = request.path_params.get("handle", "")
    if not sid or sid not in _session_envs:
        return JSONResponse({"error": "Unknown session"}, 404)
    if handle not in _session_envs.get(sid, {}):
        return JSONResponse({"error": f"Unknown env handle: {handle}"}, 404)

    interval_s = _session_stream_interval.get(sid, 0.05)
    stream_key = f"{sid}/{handle}"

    async def event_stream():
        q: asyncio.Queue = asyncio.Queue(maxsize=4)
        streams = _session_streams.setdefault(stream_key, set())
        streams.add(q)

        if stream_key not in _session_stream_tasks or _session_stream_tasks[stream_key].done():
            _session_stream_tasks[stream_key] = asyncio.create_task(
                _live_stream_loop(sid, interval_s, stream_key=stream_key, handle=handle)
            )

        try:
            yield f"event: config\ndata: {json.dumps({'interval_ms': int(interval_s * 1000), 'handle': handle})}\n\n"
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=30)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            streams.discard(q)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def session_dashboard(request):
    """Per-session linked live-rendering page."""
    sid = request.path_params.get("sid", "")
    return HTMLResponse(_SESSION_DASHBOARD_HTML.replace("__SESSION_ID__", sid))
