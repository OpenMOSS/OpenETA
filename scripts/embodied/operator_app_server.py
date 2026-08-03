#!/usr/bin/env python3
"""Drive a persistent Codex Operator thread through app-server JSONL."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, TextIO


TERMINAL_EPISODE_STATES = {"completed", "failed", "aborted", "stopped"}
VISIBLE_VALUE_LIMIT = 320
IMAGE_VALUE_KEYS = {
    "base64",
    "b64",
    "b64json",
    "data",
    "image",
    "imagebase64",
    "imagedata",
    "imageurl",
    "inputimage",
    "url",
}
IMAGE_BASE64_PREFIXES = ("/9j/", "iVBORw0KGgo", "R0lGOD", "UklGR", "PHN2Zy")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _episode_projection(root: Path) -> dict[str, Any]:
    return _read_json(root / "current.json") or _read_json(root / "episode.json")


def _operator_tool_count(root: Path) -> int:
    path = root / "operator_context.jsonl"
    try:
        with path.open("rb") as stream:
            return sum(1 for line in stream if line.strip())
    except OSError:
        return 0


def _continuation_prompt(root: Path, *, previous_count: int, turn_number: int) -> str:
    projection = _episode_projection(root)
    current_count = _operator_tool_count(root)
    latest_issue = projection.get("latest_issue")
    issue_text = ""
    if isinstance(latest_issue, dict):
        code = latest_issue.get("code")
        message = latest_issue.get("message")
        if code or message:
            issue_text = f" Latest retained issue: {code or 'unknown'}: {message or ''}."
    progress = current_count - previous_count
    return (
        f"Continue the running embodied episode (turn {turn_number}). "
        "Use current evidence, take the next useful action, verify with check_task, "
        "and call finish_episode only at a true terminal state. "
        f"The previous turn made {progress} tool call(s).{issue_text}"
    )


def _turn_status(message: dict[str, Any]) -> str:
    """Return the app-server turn status without trusting missing metadata."""

    params = message.get("params")
    if not isinstance(params, dict):
        return ""
    turn = params.get("turn")
    return str(turn.get("status") or "") if isinstance(turn, dict) else ""


def _short_value(value: Any, *, limit: int = VISIBLE_VALUE_LIMIT) -> str:
    rendered = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 1] + "…"


def _scrub_event(value: Any, *, parent_type: str | None = None) -> Any:
    """Remove binary image payloads while retaining useful app-server evidence."""

    if isinstance(value, list):
        return [_scrub_event(item, parent_type=parent_type) for item in value]
    if not isinstance(value, dict):
        return value

    value_type = value.get("type") if isinstance(value.get("type"), str) else parent_type
    image_context = isinstance(value_type, str) and value_type.lower() in {
        "image",
        "inputimage",
    }
    scrubbed: dict[str, Any] = {}
    for key, item in value.items():
        normalized_key = key.replace("_", "").lower()
        image_value = normalized_key in IMAGE_VALUE_KEYS or "base64" in normalized_key
        if isinstance(item, str) and (
            (image_context and image_value)
            or normalized_key in {"image", "imagebase64", "imagedata", "imageurl", "inputimage"}
            or "base64" in normalized_key
            or item.startswith(IMAGE_BASE64_PREFIXES)
        ):
            scrubbed[key] = f"<omitted {len(item)} chars>"
            continue
        if isinstance(item, str) and item.startswith("data:image/"):
            scrubbed[key] = f"<omitted image data URL: {len(item)} chars>"
            continue
        if isinstance(item, str) and item.lstrip()[:1] in {"{", "["}:
            try:
                nested = json.loads(item)
            except json.JSONDecodeError:
                nested = None
            if isinstance(nested, (dict, list)):
                scrubbed_nested = _scrub_event(nested)
                if scrubbed_nested != nested:
                    scrubbed[key] = json.dumps(
                        scrubbed_nested,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    continue
        if isinstance(item, str) and len(item) > 100_000:
            scrubbed[key] = f"<omitted large string: {len(item)} chars>"
            continue
        child_type = "image" if normalized_key in {"image", "images", "inputimage"} else value_type
        scrubbed[key] = _scrub_event(item, parent_type=child_type)
    return scrubbed


def _mcp_result_summary(item: dict[str, Any]) -> str:
    result = item.get("result")
    content = result.get("content") if isinstance(result, dict) else None
    text_blocks = [
        block.get("text")
        for block in content or []
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ]
    if not text_blocks:
        error = item.get("error")
        return _short_value(error) if error else "no text result"

    text = text_blocks[0]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text if len(text) <= VISIBLE_VALUE_LIMIT else text[: VISIBLE_VALUE_LIMIT - 1] + "…"
    if not isinstance(payload, dict):
        return _short_value(payload)

    summary: dict[str, Any] = {}
    for key in (
        "kind",
        "success",
        "observation_id",
        "scene_mode",
        "selected_detection_id",
        "selected_grasp_id",
        "pose_id",
        "inspected_grasp_id",
        "stage",
        "contact_status",
        "required_next_tool",
        "task_success",
        "episode_status",
        "issue_code",
        "gripper_requested",
        "gripper_executed",
        "gripper_skip_reason",
        "error",
        "message",
    ):
        if key in payload and payload[key] is not None:
            summary[key] = payload[key]
    count_keys = {
        "detection_ids": "detection_count",
        "candidate_ids": "candidate_count",
        "displayed_pose_ids": "displayed_pose_count",
    }
    for key, count_key in count_keys.items():
        values = payload.get(key)
        if isinstance(values, list):
            summary[count_key] = len(values)
    return _short_value(summary or payload)


def _compact_event(message: dict[str, Any]) -> str | None:
    method = message.get("method")
    params = message.get("params")
    if method == "item/agentMessage/delta":
        return None
    if method in {"thread/started", "turn/started", "turn/completed"}:
        status = None
        if isinstance(params, dict):
            turn = params.get("turn")
            thread = params.get("thread")
            subject = turn if isinstance(turn, dict) else thread
            status = subject.get("status") if isinstance(subject, dict) else None
        suffix = f" status={status}" if status else ""
        return f"\n[app-server] {method}{suffix}\n"
    if method in {"item/started", "item/completed"} and isinstance(params, dict):
        item = params.get("item")
        if not isinstance(item, dict):
            return None
        item_type = item.get("type")
        if item_type == "agentMessage":
            text = item.get("text")
            if method == "item/completed" and isinstance(text, str) and text.strip():
                return f"\n[operator] {text.strip()}\n"
            return None
        if item_type == "mcpToolCall":
            tool = item.get("tool") or "unknown"
            if method == "item/started":
                return f"\n[tool] {tool} {_short_value(item.get('arguments') or {})}\n"
            status = item.get("status") or "completed"
            return f"[tool] {tool} -> {status}: {_mcp_result_summary(item)}\n"
        if item_type == "commandExecution":
            command = item.get("command") or item.get("aggregatedOutput") or "command"
            status = item.get("status") or method.removeprefix("item/")
            return f"\n[command] {status}: {_short_value(command)}\n"
        return None
    if method == "error":
        return f"\n[app-server] error {_short_value(params)}\n"
    if "error" in message and "id" in message:
        return f"\n[app-server] response error {_short_value(message.get('error'))}\n"
    return None


def _persistent_event(message: dict[str, Any]) -> dict[str, Any] | None:
    """Return the compact, scrubbed event suitable for the durable JSONL log."""

    method = message.get("method")
    if method == "item/agentMessage/delta":
        return None

    scrubbed = _scrub_event(message)
    params = scrubbed.get("params")
    if method not in {"item/started", "item/completed"} or not isinstance(params, dict):
        return scrubbed

    item = params.get("item")
    if not isinstance(item, dict):
        return scrubbed
    item_type = item.get("type")
    if item_type == "agentMessage":
        if method != "item/completed":
            return None
        keep = {key: item[key] for key in ("id", "type", "status", "text") if key in item}
    elif item_type == "mcpToolCall":
        keep = {
            key: item[key]
            for key in ("id", "type", "server", "tool", "status")
            if key in item
        }
        keep.setdefault("status", method.removeprefix("item/"))
        if method == "item/completed":
            keep["resultSummary"] = _mcp_result_summary(item)
    else:
        return scrubbed

    compact = dict(scrubbed)
    compact["params"] = dict(params)
    compact["params"]["item"] = keep
    return compact


def _write_persistent_event(stream: TextIO, message: dict[str, Any]) -> bool:
    event = _persistent_event(message)
    if event is None:
        return False
    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    stream.flush()
    return True


def _thread_start_result(
    message: dict[str, Any], request_id: int | None
) -> tuple[str | None, str | None]:
    if message.get("id") != request_id:
        return None, None
    result = message.get("result")
    thread = result.get("thread") if isinstance(result, dict) else None
    thread_id = thread.get("id") if isinstance(thread, dict) else None
    effort = result.get("reasoningEffort") if isinstance(result, dict) else None
    return (
        thread_id if isinstance(thread_id, str) else None,
        effort if isinstance(effort, str) else None,
    )


def run(args: argparse.Namespace) -> int:
    root = args.episode_root.expanduser().resolve()
    raw_events = root / "operator_app_server.jsonl"
    raw_events.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [args.codex_bin, "app-server", "--listen", "stdio://"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
        cwd=args.cwd,
    )
    assert process.stdin is not None and process.stdout is not None

    def stop_child(_signum: int | None = None, _frame: Any = None) -> None:
        if process.poll() is None:
            process.terminate()

    signal.signal(signal.SIGTERM, stop_child)
    signal.signal(signal.SIGINT, stop_child)

    next_id = 1
    thread_request_id: int | None = None
    thread_id: str | None = None
    turn_number = 0
    turn_in_flight = False
    previous_tool_count = 0
    no_progress_failures = 0

    def send(method: str, params: dict[str, Any], *, request: bool = True) -> int | None:
        nonlocal next_id
        message: dict[str, Any] = {"method": method, "params": params}
        request_id: int | None = None
        if request:
            request_id = next_id
            next_id += 1
            message["id"] = request_id
        process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        process.stdin.flush()
        return request_id

    def start_turn(prompt: str) -> None:
        nonlocal turn_number, turn_in_flight, previous_tool_count
        assert thread_id is not None
        turn_number += 1
        previous_tool_count = _operator_tool_count(root)
        send(
            "turn/start",
            {
                "threadId": thread_id,
                "input": [{"type": "text", "text": prompt}],
                "model": args.model,
                "cwd": str(args.cwd),
                "approvalPolicy": "never",
                "sandboxPolicy": {"type": "dangerFullAccess"},
            },
        )
        turn_in_flight = True
        print(f"\n[persistent-operator] started turn {turn_number}\n", flush=True)

    send(
        "initialize",
        {
            "clientInfo": {
                "name": "openeta_embodied_operator",
                "title": "OpenETA Persistent Embodied Operator",
                "version": "0.1.0",
            }
        },
    )
    send("initialized", {}, request=False)
    thread_request_id = send(
        "thread/start",
        {
            "model": args.model,
            "cwd": str(args.cwd),
            "approvalPolicy": "never",
            "sandbox": "danger-full-access",
            "ephemeral": True,
        },
    )

    with raw_events.open("a", encoding="utf-8") as raw:
        for line in process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                raw.write(line)
                raw.flush()
                print(f"[app-server/raw] {line.rstrip()}", flush=True)
                continue
            _write_persistent_event(raw, message)
            rendered = _compact_event(_scrub_event(message))
            if rendered:
                print(rendered, end="", flush=True)

            if message.get("id") == thread_request_id:
                value, observed_effort = _thread_start_result(
                    message, thread_request_id
                )
                if not isinstance(value, str):
                    print("[persistent-operator] thread/start returned no thread id", flush=True)
                    return 2
                if observed_effort != args.reasoning_effort:
                    print(
                        "[persistent-operator] reasoning effort mismatch: "
                        f"expected={args.reasoning_effort!r} "
                        f"observed={observed_effort!r}",
                        flush=True,
                    )
                    return 2
                thread_id = value
                start_turn(args.prompt)
                continue

            if message.get("method") == "turn/completed":
                turn_in_flight = False
                time.sleep(0.2)
                projection = _episode_projection(root)
                status = str(projection.get("status") or "running")
                if status in TERMINAL_EPISODE_STATES:
                    print(
                        f"\n[persistent-operator] episode terminal: {status}; stopping app-server\n",
                        flush=True,
                    )
                    break
                if turn_number >= args.max_turns:
                    print(
                        f"\n[persistent-operator] max turns reached while episode is {status}\n",
                        flush=True,
                    )
                    return 3
                # A disconnected stream can leave the episode live while no
                # Operator decision is possible.  Do not spin forever through
                # empty continuation turns; the launcher will record an
                # external abort before tearing down the Gateway.
                progress = _operator_tool_count(root) - previous_tool_count
                if _turn_status(message) == "failed" and progress <= 0:
                    no_progress_failures += 1
                else:
                    no_progress_failures = 0
                if no_progress_failures >= 3:
                    print(
                        "\n[persistent-operator] stopping after "
                        f"{no_progress_failures} consecutive failed turns "
                        "with no new tool evidence\n",
                        flush=True,
                    )
                    return 5
                start_turn(
                    _continuation_prompt(
                        root,
                        previous_count=previous_tool_count,
                        turn_number=turn_number + 1,
                    )
                )

    # An app-server EOF is not an operator decision.  Keep the episode
    # projection untouched and make the unexpected child exit visible to the
    # caller instead of silently turning a live episode into "stopped".
    if process.poll() is not None:
        exit_code = process.returncode
        projection = _episode_projection(root)
        status = str(projection.get("status") or "running")
        if status not in TERMINAL_EPISODE_STATES:
            print(
                f"\n[persistent-operator] app-server exited unexpectedly "
                f"(returncode={exit_code}) while episode is {status}; "
                "leaving the episode open for recovery\n",
                flush=True,
            )
            return 4
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--max-turns", type=int, default=48)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
