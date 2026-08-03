"""Mirror the exact Operator-visible MCP history and images in Kitty."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _history(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    output: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            output.append(value)
    return output


def _summary(event: dict[str, Any]) -> str:
    texts = event.get("response_text_blocks", [])
    if not texts:
        return ""
    try:
        payload = json.loads(str(texts[0]))
    except json.JSONDecodeError:
        return str(texts[0])[:120]
    if not isinstance(payload, dict):
        return str(payload)[:120]
    parts = [str(payload.get("kind") or "result")]
    if "success" in payload:
        parts.append(f"success={payload['success']}")
    for key in ("pose_id", "selected_grasp_id", "stage", "failure_code"):
        if payload.get(key) is not None:
            parts.append(f"{key}={payload[key]}")
    return " ".join(parts)


def _contact_sheet(paths: list[str], target: Path) -> Path | None:
    images: list[tuple[str, Image.Image]] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            continue
        try:
            image = Image.open(path).convert("RGB")
        except OSError:
            continue
        image.thumbnail((620, 500))
        images.append((path.name, image.copy()))
    if not images:
        return None
    margin = 12
    label_height = 26
    width = max(image.width for _name, image in images) + margin * 2
    height = sum(image.height + label_height + margin for _name, image in images) + margin
    canvas = Image.new("RGB", (width, height), (20, 23, 28))
    draw = ImageDraw.Draw(canvas)
    y = margin
    for name, image in images:
        draw.text((margin, y), name, fill=(235, 238, 245))
        y += label_height
        canvas.paste(image, (margin, y))
        y += image.height + margin
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)
    return target


def _render(root: Path, contract: dict[str, Any], history: list[dict[str, Any]]) -> None:
    columns, lines = shutil.get_terminal_size((120, 42))
    latest = history[-1] if history else {}
    text_lines = [
        "EXACT OPERATOR MCP CONTEXT MIRROR",
        "This mirrors completed MCP calls only; Operator reasoning remains in its own Kitty.",
        f"model={contract.get('model', '?')}  task={contract.get('task', '?')}",
        f"clean_context={contract.get('clean_context', {})}",
        "",
        "START PROMPT:",
    ]
    prompt = str(contract.get("prompt") or "waiting for operator contract")
    text_lines.extend(textwrap.wrap(prompt, max(40, columns - 2))[:5])
    text_lines.append("")
    text_lines.append(f"MCP HISTORY ({len(history)} calls, latest 8):")
    for event in history[-8:]:
        text_lines.append(
            f"#{event.get('seq', '?'):>3} {event.get('tool', '?')} "
            f"args={json.dumps(event.get('arguments', {}), ensure_ascii=False, default=str)}"
        )
        text_lines.append(f"      -> {_summary(event)}")
    if latest:
        text_lines.extend(["", "LATEST EXACT RESPONSE TEXT:"])
        for block in latest.get("response_text_blocks", []):
            text_lines.extend(
                textwrap.wrap(str(block), max(40, columns - 2))[:8]
            )
        text_lines.append(
            "LATEST RESPONSE IMAGES: "
            + json.dumps(latest.get("response_image_paths", []), ensure_ascii=False)
        )
    max_text_rows = min(max(18, lines // 2), len(text_lines))
    sys.stdout.write("\033[H")
    for index in range(max_text_rows):
        sys.stdout.write(f"\033[2K{text_lines[index][:columns]}\033[1B\r")
    sys.stdout.flush()
    sheet = _contact_sheet(
        [str(value) for value in latest.get("response_image_paths", [])],
        root / "operator-context" / "latest-images.png",
    )
    if sheet is None or shutil.which("kitten") is None:
        return
    image_top = max_text_rows + 1
    image_rows = max(8, lines - image_top)
    subprocess.run(
        [
            "kitten",
            "icat",
            "--transfer-mode=stream",
            "--scale-up",
            "--image-id=3",
            "--place",
            f"{columns}x{image_rows}@0x{image_top}",
            str(sheet),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.2)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    last_key = ""
    while True:
        try:
            contract_path = root / "operator_context_contract.json"
            history_path = root / "operator_context.jsonl"
            key = ":".join(
                str(path.stat().st_mtime_ns) if path.exists() else "0"
                for path in (contract_path, history_path)
            )
            if key != last_key:
                _render(root, _read_json(contract_path), _history(history_path))
                last_key = key
            time.sleep(max(0.05, args.interval))
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
