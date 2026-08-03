"""Tail an observability episode for the Logger Kitty pane."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _frame_summary(root: Path, event: dict[str, Any]) -> str:
    frames = event.get("payload", {}).get("frames", [])
    refs: list[str] = []
    for frame in frames:
        camera = frame.get("camera_id", "camera")
        rgb = frame.get("rgb_path") or "-"
        depth = frame.get("depth_path") or "-"
        refs.append(f"{camera}: rgb={rgb} depth={depth}")
    return "; ".join(refs) if refs else "no media"


def format_event(root: Path, event: dict[str, Any]) -> str:
    kind = event.get("kind", "?")
    seq = event.get("seq", "?")
    if kind == "observation":
        payload = event.get("payload", {})
        return (
            f"[event {seq}] observation id={payload.get('observation_id')} "
            f"source={payload.get('source')} step={payload.get('sim_step')} "
            f"{_frame_summary(root, event)}"
        )
    if kind == "action":
        return f"[event {seq}] action id={event.get('action_id')} request={event.get('payload', {}).get('request')}"
    if kind == "tool_result":
        payload = event.get("payload", {})
        return f"[event {seq}] tool_result tool={payload.get('tool')} success={payload.get('success')} post={event.get('frame_refs', {}).get('post', [])}"
    if kind == "episode_end":
        payload = event.get("payload", {})
        return f"[event {seq}] episode_end status={payload.get('status')} success={payload.get('success')}"
    if kind == "failure_case":
        payload = event.get("payload", {})
        return (
            f"[event {seq}] FAILURE case={payload.get('failure_case_id')} "
            f"component={payload.get('component')} code={payload.get('code')} "
            f"message={payload.get('message')}"
        )
    return f"[event {seq}] {kind}"


def read_events(root: Path) -> list[dict[str, Any]]:
    path = root / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_status(root: Path) -> str | None:
    """Read lifecycle status, preferring the canonical episode manifest."""
    episode = root / "episode.json"
    if episode.exists():
        payload = json.loads(episode.read_text(encoding="utf-8"))
        status = payload.get("status")
        if isinstance(status, str) and status:
            return status
    current = root / "current.json"
    if current.exists():
        payload = json.loads(current.read_text(encoding="utf-8"))
        status = payload.get("status")
        if isinstance(status, str) and status:
            return status
    return None


def print_snapshot(root: Path, *, start: int = 0) -> int:
    events = read_events(root)
    new_events = events[start:]
    for event in new_events:
        print(format_event(root, event), flush=True)
    current = root / "current.json"
    # The projection is rewritten frequently.  Only print its summary when a
    # new event is present (or on the first snapshot), otherwise the logger
    # pane becomes a wall of identical lines every polling interval.
    if current.exists() and (new_events or start == 0):
        payload = json.loads(current.read_text(encoding="utf-8"))
        viewer = payload.get("viewer_url") or "-"
        print(
            f"[current] status={read_status(root)} sim_step={payload.get('sim_step')} "
            f"viewer={viewer} root={root}",
            flush=True,
        )
        print(
            f"[artifacts] open with: kitten icat --transfer-mode=stream "
            f"{root}/media/frames/<frame>.rgb.png",
            flush=True,
        )
    return len(events)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    seen = 0
    waiting_notice_printed = False
    while True:
        if not (args.root / "events.jsonl").exists():
            if not waiting_notice_printed:
                print(f"[logger] waiting for {args.root}/events.jsonl", flush=True)
                waiting_notice_printed = True
        else:
            waiting_notice_printed = False
            seen = print_snapshot(args.root, start=seen)
            status = read_status(args.root)
            if status in {"completed", "stopped", "failed", "terminated", "error"}:
                return 0
        if args.once:
            return 0
        time.sleep(max(0.05, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
