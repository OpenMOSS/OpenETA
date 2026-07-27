"""OpenETA MCP Server package.

Split from the monolithic ``sim/mcp_server.py`` into:

  session.py       — session state storage and lifecycle
  worker_mgr.py    — per-bench subprocess worker management + proxy helpers
  rest_api.py      — REST API handlers + SSE streaming endpoints
  dashboard_html.py — HTML templates (standalone + per-session)
  server.py        — FastMCP server, MCP tools, ASGI assembly, CLI entry
"""
