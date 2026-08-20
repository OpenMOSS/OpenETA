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
    contract_diagnostics = simulator_mcp_contract_diagnostics(tools)
    return {
        "available": True,
        "contract_compatible": not contract_diagnostics,
        "contract_diagnostics": contract_diagnostics,
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


def simulator_mcp_contract_diagnostics(tools: list[object]) -> list[JsonDict]:
    """Report host/private arguments that an older simulator will ignore.

    This is intentionally a schema compatibility check, not a task-state gate.
    The controller remains usable, while both the Agent and experiment audit can
    see which safety/evidence receipts cannot be expected from this deployment.
    """

    schemas: dict[str, set[str]] = {}
    for item in tools:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        schema = item.get("input_schema")
        schema = schema if isinstance(schema, dict) else {}
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        schemas[name] = {str(key) for key in properties}

    expected = {
        "gripper_close": {
            "contact_authorization": (
                "attachment_proxy_contract_outdated",
                "Gripper close cannot bind the host-selected target or return a "
                "trustworthy attachment-proxy receipt.",
            )
        },
        "move_to": {
            "contact_authorization": (
                "motion_contact_authorization_unsupported",
                "Contact/lift motion cannot bind host provenance to collision geometry.",
            ),
            "ik_execution_seed": (
                "motion_ik_seed_unsupported",
                "A previewed IK solution cannot be bound to the subsequent execution.",
            ),
        },
    }
    diagnostics: list[JsonDict] = []
    for tool_name, requirements in expected.items():
        if tool_name not in schemas:
            continue
        for parameter, (code, message) in requirements.items():
            if parameter in schemas[tool_name]:
                continue
            diagnostics.append(
                {
                    "code": code,
                    "severity": "warning",
                    "tool": tool_name,
                    "missing_parameter": parameter,
                    "message": message,
                    "recovery": (
                        "Restart or redeploy the simulator MCP from the current "
                        "OpenETA repository before relying on this contract."
                    ),
                }
            )
    return diagnostics


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
