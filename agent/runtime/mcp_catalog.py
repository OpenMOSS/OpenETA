"""Shared discovery and compaction for simulator MCP tool documentation."""

from __future__ import annotations

from pathlib import Path

from adapter.protocol import JsonDict
from agent.runtime.response_artifacts import materialize_json_response
from agent.tools.sim_mcp import SimulatorMcpTransport, mcp_server_url_from_endpoint


def discover_mcp_tool_catalog(
    transport: SimulatorMcpTransport | None,
    *,
    endpoint_url: str,
    output_root: str | Path,
    timeout_s: float = 10.0,
) -> JsonDict:
    """Return planner-safe MCP docs and persist the complete catalog."""

    if transport is None or not hasattr(transport, "list_tools"):
        return {}
    try:
        catalog = transport.list_tools(timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001 - discovery is best effort.
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "url": endpoint_url,
        }
    if not isinstance(catalog, dict):
        return {
            "available": False,
            "error_type": "TypeError",
            "message": "MCP list_tools returned a non-object response.",
            "url": endpoint_url,
        }
    artifact = materialize_json_response(
        catalog,
        output_root=output_root,
        bundle_id="mcp-list_tools",
        name="response",
    )
    return compact_mcp_tool_catalog(
        catalog,
        response_path=artifact.path,
        grep_hint=artifact.grep_hint,
        url=endpoint_url,
    )


def compact_mcp_tool_catalog(
    catalog: JsonDict,
    *,
    response_path: str,
    grep_hint: str,
    url: str,
) -> JsonDict:
    """Keep tool names and bounded schemas in planner context."""

    tools = catalog.get("tools", [])
    if not isinstance(tools, list):
        tools = []
    return {
        "available": True,
        "url": url,
        "mcp_server_url": mcp_server_url_from_endpoint(url),
        "tool_count": catalog.get("tool_count", len(tools)),
        "response_path": response_path,
        "grep_hint": grep_hint,
        "tools": [
            _compact_mcp_tool_doc(tool)
            for tool in tools[:32]
            if isinstance(tool, dict)
        ],
    }


def _compact_mcp_tool_doc(tool: JsonDict) -> JsonDict:
    schema = tool.get("input_schema")
    if not isinstance(schema, dict):
        schema = {}
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required = schema.get("required")
    return {
        "name": str(tool.get("name") or ""),
        "description": _truncate(str(tool.get("description") or ""), 700),
        "required": list(required) if isinstance(required, list) else [],
        "parameters": {
            str(name): _compact_mcp_schema_property(value)
            for name, value in list(properties.items())[:24]
            if isinstance(value, dict)
        },
    }


def _compact_mcp_schema_property(value: JsonDict) -> JsonDict:
    compact: JsonDict = {}
    for key in ("type", "description", "default", "enum"):
        if key not in value:
            continue
        field = value[key]
        if isinstance(field, str):
            compact[key] = _truncate(field, 300)
        elif isinstance(field, (int, float, bool)) or field is None:
            compact[key] = field
        elif isinstance(field, list):
            compact[key] = field[:20]
    return compact


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."
