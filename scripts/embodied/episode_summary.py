"""Summarize the durable fields needed by public Operator evaluation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _load_context_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = root / "operator_context.jsonl"
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _response_payload(row: dict[str, Any]) -> dict[str, Any]:
    blocks = row.get("response_text_blocks")
    if not isinstance(blocks, list) or not blocks:
        return {}
    try:
        value = json.loads(blocks[0])
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _infrastructure_diagnostics(
    root: Path,
    *,
    episode_status: str | None,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    stream_disconnect_count = 0
    terminal_stream_error_count = 0
    system_error_count = 0
    app_trace = root / "operator_app_server.jsonl"
    if app_trace.is_file():
        for line in app_trace.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            method = event.get("method")
            params = event.get("params")
            params = params if isinstance(params, dict) else {}
            error = params.get("error")
            error = error if isinstance(error, dict) else {}
            message = str(error.get("message") or "")
            info = error.get("codexErrorInfo")
            info_text = (
                json.dumps(info, ensure_ascii=False)
                if isinstance(info, (dict, list))
                else str(info or "")
            )
            if (
                "responseStreamDisconnected" in info_text
                or "stream disconnected" in message.lower()
            ):
                stream_disconnect_count += 1
            if method == "error" and params.get("willRetry") is False:
                terminal_stream_error_count += 1
            status = params.get("status")
            if (
                method == "thread/status/changed"
                and isinstance(status, dict)
                and status.get("type") == "systemError"
            ):
                system_error_count += 1

    manual_abort_count = sum(
        row.get("tool") == "manual_abort" for row in rows
    )
    finish_episode_count = sum(
        row.get("tool") == "finish_episode" for row in rows
    )
    reasons: list[str] = []
    if episode_status == "aborted":
        reasons.append("episode_aborted")
    if manual_abort_count:
        reasons.append("manual_abort")
    if terminal_stream_error_count or system_error_count:
        reasons.append("operator_stream_system_error")
    return {
        "infrastructure_valid": not reasons,
        "infrastructure_invalid_reasons": reasons,
        "manual_abort_count": manual_abort_count,
        "finish_episode_count": finish_episode_count,
        "stream_disconnect_count": stream_disconnect_count,
        "terminal_stream_error_count": terminal_stream_error_count,
        "operator_stream_system_error_count": system_error_count,
    }


def summarize_episode(root: Path) -> dict[str, Any]:
    rows = _load_context_rows(root)
    final = _response_payload(rows[-1]) if rows else {}
    durable = _load_json(root / "episode.json") or _load_json(
        root / "current.json"
    )
    contract = _load_json(root / "operator_context_contract.json")
    context_profile = contract.get("context_profile")
    context_profile = (
        context_profile if isinstance(context_profile, dict) else {}
    )
    episode_status = durable.get("status", durable.get("episode_status"))
    if episode_status is None:
        episode_status = final.get("episode_status")
    episode_success = durable.get("success")
    if episode_success is None:
        episode_success = durable.get(
            "task_success", durable.get("episode_success")
        )
    if episode_success is None:
        episode_success = final.get("episode_success")
    diagnostics = _infrastructure_diagnostics(
        root,
        episode_status=(
            str(episode_status) if episode_status is not None else None
        ),
        rows=rows,
    )
    calls_by_tool = Counter(str(row.get("tool")) for row in rows)
    return {
        "episode_root": str(root),
        "profile": context_profile.get("label"),
        "composition_sha256": context_profile.get("composition_sha256"),
        "resolved_context_sha256": contract.get("resolved_context_sha256"),
        "episode_status": episode_status,
        "episode_success": episode_success,
        **diagnostics,
        "tool_calls": len(rows),
        "calls_by_tool": dict(sorted(calls_by_tool.items())),
    }
