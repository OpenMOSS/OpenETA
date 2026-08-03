"""Helpers for reading OpenETA MCP server registrations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapter.protocol import JsonDict


DEFAULT_MCP_CONFIG_PATH = ".mcp.json"


@dataclass(frozen=True, slots=True)
class McpServerConfig:
    """One MCP server entry loaded from `.mcp.json`."""

    name: str
    url: str
    transport: str = "sse"
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "name": self.name,
            "url": self.url,
            "transport": self.transport,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


def load_mcp_server_configs(path: str | Path = DEFAULT_MCP_CONFIG_PATH) -> dict[str, McpServerConfig]:
    """Load MCP server configs from a Codex/Claude-compatible `.mcp.json` file."""

    config_path = Path(path)
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    servers = payload.get("mcpServers")
    if not isinstance(servers, dict):
        return {}

    configs: dict[str, McpServerConfig] = {}
    for raw_name, raw_server in servers.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        config = _coerce_mcp_server_config(raw_name.strip(), raw_server)
        if config is not None:
            configs[config.name] = config
    return configs


def load_mcp_server_url(
    name: str,
    *,
    aliases: tuple[str, ...] = (),
    path: str | Path = DEFAULT_MCP_CONFIG_PATH,
) -> str:
    """Return the first matching MCP server URL by canonical name or alias."""

    configs = load_mcp_server_configs(path)
    for candidate in (name, *aliases):
        config = configs.get(candidate)
        if config is not None and config.url:
            return config.url
    return ""


def compact_mcp_registry(path: str | Path = DEFAULT_MCP_CONFIG_PATH) -> JsonDict:
    """Return a bounded registry summary suitable for planner memory."""

    configs = load_mcp_server_configs(path)
    return {
        "schema_version": "openeta.mcp_registry.v1",
        "source": str(path),
        "server_count": len(configs),
        "servers": [config.to_dict() for config in configs.values()],
    }


def _coerce_mcp_server_config(name: str, raw_server: Any) -> McpServerConfig | None:
    if isinstance(raw_server, str):
        url = raw_server.strip()
        return McpServerConfig(name=name, url=url) if url else None
    if not isinstance(raw_server, dict):
        return None
    raw_url = raw_server.get("url")
    if not isinstance(raw_url, str) or not raw_url.strip():
        return None
    transport = raw_server.get("transport", "sse")
    if not isinstance(transport, str) or not transport.strip():
        transport = "sse"
    metadata = {
        str(key): value
        for key, value in raw_server.items()
        if key not in {"url", "transport"} and isinstance(key, str)
    }
    return McpServerConfig(
        name=name,
        url=raw_url.strip(),
        transport=transport.strip(),
        metadata=metadata,
    )
