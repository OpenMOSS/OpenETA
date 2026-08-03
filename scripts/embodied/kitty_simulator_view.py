"""Show the latest embodied simulator frame inside a Kitty pane.

The simulator MCP server is owned by the parent shell.  This process is only
the human-facing view: it watches the host-side projection and redraws the
latest agentview RGB frame with Kitty's graphics protocol.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _latest_rgb(root: Path, payload: dict[str, Any]) -> Path | None:
    record = payload.get("record")
    frames = record.get("frames", []) if isinstance(record, dict) else []
    preferred = [
        frame for frame in frames
        if isinstance(frame, dict) and frame.get("camera_id") == "agentview"
    ]
    candidates = preferred or [
        frame for frame in frames if isinstance(frame, dict) and frame.get("rgb_path")
    ]
    if not candidates:
        return None
    path = root / str(candidates[0].get("rgb_path", ""))
    return path if path.is_file() else None


def _visualization(root: Path, payload: dict[str, Any] | None) -> tuple[str, Path | None]:
    if not payload:
        return "observation", None
    value = payload.get("visualization")
    if isinstance(value, dict):
        raw = value.get("image_path")
        if isinstance(raw, str) and raw:
            path = Path(raw)
            if not path.is_absolute():
                path = root / path
            if path.is_file():
                return str(value.get("kind") or "visualization"), path
    return "observation", _latest_rgb(root, payload)


def _image_key(view_kind: str, image: Path | None) -> str:
    """Return a redraw key that changes when an artifact is replaced in place."""

    if image is None:
        return view_kind
    try:
        stat = image.stat()
        return f"{view_kind}:{image}:{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        return f"{view_kind}:{image}"


def _header(
    root: Path,
    payload: dict[str, Any] | None,
    *,
    server_alive: bool,
    dashboard: str,
    view_kind: str,
) -> str:
    if payload is None:
        return (
            "LIBERO simulator view\n"
            f"server={'UP' if server_alive else 'DOWN'}  waiting for Operator observation\n"
            f"view={view_kind}\n"
            f"dashboard={dashboard}\n"
            f"root={root}"
        )
    return (
        "LIBERO simulator view\n"
        f"server={'UP' if server_alive else 'DOWN'}  "
        f"status={payload.get('status', '?')}  sim_step={payload.get('sim_step', '?')}  view={view_kind}\n"
        f"task={payload.get('task', '')}\n"
        f"dashboard={payload.get('viewer_url') or dashboard}"
        + (
            f"\nFAILURE={payload.get('failure_case', {}).get('component')}"
            f"/{payload.get('failure_case', {}).get('code')}"
            if isinstance(payload.get('failure_case'), dict)
            else ""
        )
    )


def _render(
    root: Path,
    *,
    server_alive: bool,
    dashboard: str,
    clear_images: bool = False,
) -> None:
    payload = _read_json(root / "current.json")
    view_kind, image = _visualization(root, payload)
    columns, lines = shutil.get_terminal_size((100, 32))
    # Redraw only the four text rows.  Clearing the whole terminal on every
    # artifact update caused visible blinking, especially when Kitty had to
    # remove and recreate a large graphics image.
    header_lines = _header(
        root,
        payload,
        server_alive=server_alive,
        dashboard=dashboard,
        view_kind=view_kind,
    ).splitlines()
    header_rows = max(4, len(header_lines))
    image_top = header_rows + 1
    image_rows = max(8, lines - image_top)
    sys.stdout.write("\033[H")
    for index in range(header_rows):
        line = header_lines[index] if index < len(header_lines) else ""
        sys.stdout.write(f"\033[2K{line}\033[1B\r")
    sys.stdout.write(f"\033[{image_top};1H")
    # Clear the old text/image region before drawing the new frame.  Without
    # this, the initial "no RGB frame yet" placeholder remains visible even
    # after current.json points at a valid segmentation/contact-sheet image.
    sys.stdout.write("\033[J")
    sys.stdout.flush()
    if image is None or shutil.which("kitten") is None:
        sys.stdout.write("\n(no RGB frame yet; waiting for observe)\n")
        sys.stdout.flush()
        return

    command = [
        "kitten", "icat",
        "--transfer-mode=stream",
        "--scale-up",
        "--image-id=1",
        "--place", f"{columns}x{image_rows}@0x{image_top}",
        str(image),
    ]
    if clear_images:
        command.insert(2, "--clear")
    # A graphics-protocol failure must not make the simulator pane look like
    # the simulator itself failed.  In particular, Kitty launched from an
    # older tmux can report passthrough errors; the launcher strips TMUX, but
    # keep this viewer quiet and alive if an inherited environment slips
    # through or a terminal does not support icat.
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        # The artifact is still useful even when the terminal transport is
        # unavailable (for example, a pane with no controlling tty).  Keep a
        # truthful diagnostic instead of the stale pre-observation message.
        sys.stdout.write(
            f"\n(image transmission failed; artifact={image})\n"
        )
        sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=float, default=0.25)
    args = parser.parse_args()
    dashboard = f"http://127.0.0.1:{args.port}/"

    # Render once immediately so an idle simulator pane still explains its
    # state instead of looking like a dead/empty terminal.
    last_state: str | None = None
    first_render = True
    while True:
        try:
            payload = _read_json(args.root / "current.json")
            view_kind, image = _visualization(args.root, payload)
            image_key = _image_key(view_kind, image)
            columns, lines = shutil.get_terminal_size((100, 32))
            server_alive = subprocess.run(
                ["kill", "-0", str(args.server_pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode == 0
            state = f"{image_key}:{server_alive}:{columns}x{lines}"
            if state != last_state:
                _render(
                    args.root,
                    server_alive=server_alive,
                    dashboard=dashboard,
                    clear_images=first_render,
                )
                first_render = False
                last_state = state
            time.sleep(max(0.05, args.interval))
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
