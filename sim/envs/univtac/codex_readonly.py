"""Codex-native read-only UniVTAC observation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CODEX_OPERATOR_PROMPT = """You are the read-only operator for one UniVTAC Pull Out Key episode.

Call the MCP tool `observe` exactly once before answering.

Inspect the head camera, wrist camera, left tactile rgb_marker image,
and right tactile rgb_marker image. Report:

1. What scene and object/gripper geometry is actually visible.
2. What the left tactile image appears to indicate.
3. What the right tactile image appears to indicate.
4. Whether the two tactile observations agree or differ.
5. Important uncertainties or ambiguities.
6. One qualitative next decision you would consider.

Do not execute an action. Do not emit numeric robot commands.
Do not claim access to force, depth, exact object pose, success labels,
or any host-only metadata.
"""


def build_codex_exec_command(
    *,
    codex_bin: str,
    model: str,
    reasoning_effort: str,
    workspace: Path,
    repo_root: Path,
    episode_root: Path,
    snapshot_path: Path,
    final_response_path: Path,
) -> list[str]:
    """Build one isolated Codex exec command with only the UniVTAC MCP."""

    mcp_args = [
        f"PYTHONPATH={repo_root}",
        "uv",
        "run",
        "--no-project",
        "--python",
        str(repo_root / ".venv/bin/python"),
        "-m",
        "tools.univtac_operator_mcp_server",
        "--episode-root",
        str(episode_root),
        "--snapshot-pre",
        str(snapshot_path),
    ]
    config = [
        "features.memories=false",
        "features.enable_request_compression=false",
        "memories.use_memories=false",
        "memories.generate_memories=false",
        'history.persistence="none"',
        f"model_reasoning_effort={json.dumps(reasoning_effort)}",
        'mcp_servers.univtac.command="env"',
        f"mcp_servers.univtac.args={json.dumps(mcp_args)}",
        "mcp_servers.univtac.required=true",
        "mcp_servers.univtac.tool_timeout_sec=120",
    ]
    command = [
        codex_bin,
        "-m",
        model,
        "exec",
        "-C",
        str(workspace),
        "-s",
        "read-only",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--json",
        "--output-last-message",
        str(final_response_path),
    ]
    for value in config:
        command.extend(("-c", value))
    command.append(CODEX_OPERATOR_PROMPT)
    return command


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _event_item(event: dict[str, Any]) -> dict[str, Any]:
    item = event.get("item")
    if isinstance(item, dict):
        return item
    params = event.get("params")
    if isinstance(params, dict) and isinstance(params.get("item"), dict):
        return params["item"]
    return {}


def summarize_codex_exec(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract visible messages and lifecycle facts without hidden reasoning."""

    visible_messages: list[str] = []
    event_types: list[str] = []
    errors: list[str] = []
    usage: dict[str, Any] | None = None
    for event in rows:
        event_type = str(event.get("type") or event.get("method") or "unknown")
        event_types.append(event_type)
        item = _event_item(event)
        item_type = str(item.get("type", "")).lower()
        if "agent" in item_type and "message" in item_type:
            text = item.get("text") or item.get("content")
            if isinstance(text, str) and text.strip():
                visible_messages.append(text)
        if "error" in event_type.lower() or "failed" in event_type.lower():
            errors.append(str(event.get("message") or event.get("error") or event_type))
        candidate_usage = event.get("usage")
        if isinstance(candidate_usage, dict):
            usage = dict(candidate_usage)
    return {
        "event_count": len(rows),
        "event_types": event_types,
        "visible_assistant_messages": visible_messages,
        "errors": errors,
        "usage": usage,
    }


def load_operator_context(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for row in rows:
        if row.get("tool") != "observe" or row.get("arguments") != {}:
            raise ValueError("operator context contains a non-observe tool call")
        if len(row.get("response_text_blocks", [])) != 1:
            raise ValueError("observe must return exactly one text block")
        if len(row.get("response_image_paths", [])) != 4:
            raise ValueError("observe must return exactly four image paths")
    return rows
