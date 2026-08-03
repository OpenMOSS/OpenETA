#!/usr/bin/env python3
"""One read-only-capable index and HTTP surface for every embodied episode."""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

from scripts.embodied.episode_web_dashboard import (
    Handler as EpisodeHandler,
    ThreadingHTTPServer,
    render_dashboard_html,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TERMINAL_STATUSES = {
    "completed",
    "failed",
    "aborted",
    "stopped",
    "terminated",
    "error",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _episode_roots(episodes_root: Path) -> list[Path]:
    return sorted(
        {path.parent.resolve() for path in episodes_root.rglob("episode.json")},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _observability_roots(episodes_root: Path) -> list[Path]:
    """Include launch-aborted roots that never produced episode.json."""

    roots = {path.parent.resolve() for path in episodes_root.rglob("services.tsv")}
    roots.update(_episode_roots(episodes_root))
    return sorted(roots, key=lambda path: path.stat().st_mtime, reverse=True)


def _registered_urls(root: Path) -> tuple[str, str]:
    viser_url = ""
    control_url = ""
    registry = root / "services.tsv"
    try:
        lines = registry.read_text(encoding="utf-8").splitlines()
    except OSError:
        return viser_url, control_url
    for line in lines:
        fields = line.split("\t")
        if len(fields) < 5:
            continue
        role, raw_pid, _pgid, _start, ports = fields[:5]
        try:
            pid = int(raw_pid)
            os.kill(pid, 0)
        except (OSError, ValueError):
            continue
        numeric_ports = [value for value in ports.split(",") if value.isdigit()]
        if role == "viser-supervisor" and numeric_ports:
            viser_url = f"http://127.0.0.1:{numeric_ports[0]}"
        if "gateway" in role and numeric_ports:
            control_url = f"http://127.0.0.1:{numeric_ports[-1]}"
    return viser_url, control_url


def _episode_status(root: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
    episode = _read_json(root / "episode.json")
    current = _read_json(root / "current.json")
    status = str(current.get("status") or episode.get("status") or "unknown")
    return status, episode, current


def _terminal_finished_at(root: Path) -> float | None:
    status, episode, current = _episode_status(root)
    if status not in TERMINAL_STATUSES:
        return None
    for value in (
        episode.get("finished_at_s"),
        current.get("finished_at_s"),
    ):
        if isinstance(value, (int, float)):
            return float(value)
    candidates = [root / "episode.json", root / "current.json"]
    mtimes = [path.stat().st_mtime for path in candidates if path.exists()]
    return max(mtimes) if mtimes else root.stat().st_mtime


def _has_registered_observability(root: Path) -> bool:
    registry = root / "services.tsv"
    try:
        return any(
            line.split("\t", 1)[0] in {"dashboard", "viser-supervisor"}
            for line in registry.read_text(encoding="utf-8").splitlines()
            if line
        )
    except OSError:
        return False


def _has_live_execution_service(root: Path) -> bool:
    registry = root / "services.tsv"
    try:
        lines = registry.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in lines:
        fields = line.split("\t")
        if len(fields) < 2 or fields[0] in {"dashboard", "viser-supervisor"}:
            continue
        try:
            os.kill(int(fields[1]), 0)
        except (OSError, ValueError):
            continue
        return True
    return False


def _last_episode_activity(root: Path) -> float:
    candidates = [root / "episode.json", root / "current.json", root / "trace.jsonl"]
    mtimes = [path.stat().st_mtime for path in candidates if path.exists()]
    return max(mtimes) if mtimes else root.stat().st_mtime


def sweep_terminal_replays(
    episodes_root: Path,
    *,
    ttl_seconds: float,
    orphan_ttl_seconds: float | None = None,
    now: float | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """Stop expired per-episode viewers while retaining every artifact.

    Interrupted runs can remain marked ``running`` forever. Those viewers are
    reclaimed only after their longer orphan TTL and only when no registered
    execution service is alive.
    """

    current_time = time.time() if now is None else float(now)
    stopped: list[Path] = []
    stop_script = REPO_ROOT / "scripts/embodied/stop_episode_observability.sh"
    for root in _observability_roots(episodes_root):
        if (root / ".replay-pin").exists() or not _has_registered_observability(root):
            continue
        lease = root / ".replay-lease-until"
        try:
            lease_until = float(lease.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            lease_until = 0.0
        if lease_until > current_time:
            continue
        finished_at = _terminal_finished_at(root)
        terminal_expired = (
            finished_at is not None and current_time - finished_at >= ttl_seconds
        )
        orphan_expired = (
            finished_at is None
            and orphan_ttl_seconds is not None
            and current_time - _last_episode_activity(root) >= orphan_ttl_seconds
            and not _has_live_execution_service(root)
        )
        if not terminal_expired and not orphan_expired:
            continue
        stopped.append(root)
        if not dry_run:
            subprocess.run(
                ["bash", str(stop_script), str(root)],
                cwd=REPO_ROOT,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    return stopped


class ReplayHubHandler(EpisodeHandler):
    episodes_root: Path

    def _episode_route(self) -> tuple[Path, str, str] | None:
        request_path = unquote(urlparse(self.path).path)
        prefix = "/episodes/"
        if not request_path.startswith(prefix):
            return None
        raw = request_path.removeprefix(prefix)
        marker_positions = [
            position
            for marker in ("/api/", "/artifact/")
            if (position := raw.find(marker)) >= 0
        ]
        if marker_positions:
            split_at = min(marker_positions)
            relative_text = raw[:split_at].rstrip("/")
            forwarded = raw[split_at:]
        else:
            relative_text = raw.rstrip("/")
            forwarded = "/"
        if not relative_text:
            return None
        root = (self.episodes_root / relative_text).resolve()
        base = self.episodes_root.resolve()
        if (root != base and base not in root.parents) or not (
            root / "episode.json"
        ).is_file():
            return None
        public_prefix = "/episodes/" + quote(
            root.relative_to(base).as_posix(), safe="/"
        )
        return root, public_prefix, forwarded

    def _index(self) -> bytes:
        rows = []
        base = self.episodes_root.resolve()
        for root in _episode_roots(base):
            status, episode, current = _episode_status(root)
            relative = root.relative_to(base).as_posix()
            href = "/episodes/" + quote(relative, safe="/") + "/"
            success = episode.get("success")
            result = episode.get("result") if isinstance(episode.get("result"), dict) else {}
            started = episode.get("started_at_s")
            started_text = (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(started)))
                if isinstance(started, (int, float))
                else "unknown"
            )
            pinned = (root / ".replay-pin").exists()
            rows.append(
                "<tr>"
                f"<td><a href='{html.escape(href)}'>{html.escape(relative)}</a></td>"
                f"<td>{html.escape(status)}</td>"
                f"<td>{html.escape(str(success))}</td>"
                f"<td>{html.escape(str(episode.get('seed', '')))}</td>"
                f"<td>{html.escape(str(result.get('sim_step', current.get('sim_step', ''))))}</td>"
                f"<td>{html.escape(started_text)}</td>"
                f"<td>{'PINNED' if pinned else ''}</td>"
                "</tr>"
            )
        body = "".join(rows) or "<tr><td colspan=7>No episodes found.</td></tr>"
        document = f"""<!doctype html><meta charset=utf-8><title>OpenETA Replay Hub</title>
<style>body{{font:14px sans-serif;background:#111;color:#ddd;margin:24px}}a{{color:#8fd3ff}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #333;padding:8px;text-align:left}}th{{position:sticky;top:0;background:#191919}}.hint{{color:#aaa}}</style>
<h1>OpenETA Replay Hub</h1>
<p class=hint>One process serves durable artifacts from {html.escape(str(base))}. Per-episode Dashboard and Viser processes are temporary and reclaimed after their terminal TTL.</p>
<table><thead><tr><th>episode</th><th>status</th><th>success</th><th>seed</th><th>sim step</th><th>started</th><th>retention</th></tr></thead><tbody>{body}</tbody></table>"""
        return document.encode()

    def _bind_episode(self, root: Path) -> None:
        viser_url, control_url = _registered_urls(root)
        self.root = root
        self.viser_url = viser_url or "#"
        self.control_url = control_url.rstrip("/")

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path == "/":
            return self._send(self._index(), "text/html; charset=utf-8")
        routed = self._episode_route()
        if routed is None:
            return self.send_error(404)
        root, public_prefix, forwarded = routed
        self._bind_episode(root)
        if forwarded == "/":
            return self._send(
                render_dashboard_html(public_prefix).encode(),
                "text/html; charset=utf-8",
            )
        original = self.path
        self.path = forwarded
        try:
            return super().do_GET()
        finally:
            self.path = original

    def do_POST(self) -> None:  # noqa: N802
        routed = self._episode_route()
        if routed is None:
            return self.send_error(404)
        root, _public_prefix, forwarded = routed
        self._bind_episode(root)
        original = self.path
        self.path = forwarded
        try:
            return super().do_POST()
        finally:
            self.path = original


def _sweeper(
    episodes_root: Path,
    ttl_seconds: float,
    orphan_ttl_seconds: float,
    interval_seconds: float,
) -> None:
    while True:
        sweep_terminal_replays(
            episodes_root,
            ttl_seconds=ttl_seconds,
            orphan_ttl_seconds=orphan_ttl_seconds,
        )
        time.sleep(interval_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9295)
    parser.add_argument("--terminal-ttl-seconds", type=float, default=900.0)
    parser.add_argument("--orphan-ttl-seconds", type=float, default=3600.0)
    parser.add_argument("--sweep-interval-seconds", type=float, default=30.0)
    parser.add_argument("--sweep-once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    episodes_root = args.episodes_root.resolve()
    episodes_root.mkdir(parents=True, exist_ok=True)
    if args.sweep_once:
        for root in sweep_terminal_replays(
            episodes_root,
            ttl_seconds=args.terminal_ttl_seconds,
            orphan_ttl_seconds=args.orphan_ttl_seconds,
            dry_run=args.dry_run,
        ):
            print(root)
        return 0
    ReplayHubHandler.episodes_root = episodes_root
    thread = threading.Thread(
        target=_sweeper,
        args=(
            episodes_root,
            args.terminal_ttl_seconds,
            args.orphan_ttl_seconds,
            args.sweep_interval_seconds,
        ),
        daemon=True,
    )
    thread.start()
    ThreadingHTTPServer((args.host, args.port), ReplayHubHandler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
