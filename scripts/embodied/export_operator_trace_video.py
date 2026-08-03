#!/usr/bin/env python3
"""Export a successful embodied Operator episode as a presentation MP4.

The video is driven by the exact MCP context delivered to the Operator.  Each
card shows the current visual evidence, the tool call, a compact response, and
the other images returned by that call.  It is deliberately separate from the
simulator logger replay: this surface explains the Operator workflow rather
than pretending that sparse post-action observations are continuous physics
footage.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Iterable

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


CANVAS = (1600, 900)
BACKGROUND = "#0b0f14"
PANEL = "#121923"
MUTED = "#8fa1b5"
TEXT = "#ecf2f8"
ACCENT = "#57b7ff"
SUCCESS = "#4fd18b"
WARNING = "#ffbd5c"


def _font(size: int, *, mono: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        [Path.home() / ".local/share/fonts/FiraCodeNerdFontMono-Regular.ttf"]
        if mono
        else [
            Path("/usr/share/fonts/truetype/lato/Lato-Medium.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


TITLE_FONT = _font(30)
HEADING_FONT = _font(23)
BODY_FONT = _font(19)
SMALL_FONT = _font(16)
MONO_FONT = _font(17, mono=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _response(row: dict[str, Any]) -> dict[str, Any]:
    blocks = row.get("response_text_blocks")
    if not isinstance(blocks, list):
        return {}
    for block in blocks:
        if not isinstance(block, str):
            continue
        try:
            value = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _compact_response(response: dict[str, Any]) -> dict[str, Any]:
    preferred = (
        "kind",
        "success",
        "status",
        "task_success",
        "outcome",
        "observation_id",
        "point_id",
        "xyz_m",
        "measurement_semantics",
        "motion_status",
        "actual_grip_site_xyz_m",
        "gripper",
        "reason",
        "message",
    )
    compact = {key: response[key] for key in preferred if key in response}
    motion = response.get("motion")
    if isinstance(motion, dict):
        compact["motion"] = {
            key: motion[key]
            for key in ("status", "position_error_mm", "endpoint_reached")
            if key in motion
        }
    return compact


def _wrap_json(value: Any, width: int = 56, max_lines: int = 13) -> list[str]:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    lines: list[str] = []
    for logical in text.splitlines() or [""]:
        lines.extend(
            textwrap.wrap(
                logical,
                width=width,
                break_long_words=True,
                break_on_hyphens=False,
                replace_whitespace=False,
            )
            or [""]
        )
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + ["…"]
    return lines


def _existing_images(row: dict[str, Any]) -> list[Path]:
    paths = row.get("response_image_paths")
    if not isinstance(paths, list):
        return []
    return [Path(path) for path in paths if isinstance(path, str) and Path(path).is_file()]


def _is_agentview(path: Path) -> bool:
    name = path.name.lower()
    return "agentview" in name or ("frame-" in name and "agent" in name)


def _fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    try:
        image = Image.open(path).convert("RGB")
    except OSError:
        image = Image.new("RGB", size, "#1b2633")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    result = Image.new("RGB", size, "#05080c")
    result.paste(
        image,
        ((size[0] - image.width) // 2, (size[1] - image.height) // 2),
    )
    return result


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    lines: Iterable[str],
    *,
    xy: tuple[int, int],
    font: ImageFont.ImageFont,
    fill: str,
    line_height: int,
) -> int:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def render_card(
    *,
    row: dict[str, Any],
    index: int,
    total: int,
    task: str,
    latest_agentview: Path | None,
    final_success: bool,
) -> tuple[Image.Image, Path | None]:
    frame = Image.new("RGB", CANVAS, BACKGROUND)
    draw = ImageDraw.Draw(frame)
    images = _existing_images(row)
    agentviews = [path for path in images if _is_agentview(path)]
    if agentviews:
        latest_agentview = agentviews[0]
    tool = str(row.get("tool") or "tool")
    response = _response(row)
    visual = (
        images[0]
        if tool == "mark_point" and images
        else latest_agentview or (images[0] if images else None)
    )

    draw.rounded_rectangle((24, 20, 1576, 80), 12, fill=PANEL)
    draw.text((48, 34), "OpenETA · LIBERO Operator Trace", font=TITLE_FONT, fill=TEXT)
    badge = "NATIVE SUCCESS" if final_success else "TRACE"
    badge_color = SUCCESS if final_success else WARNING
    badge_box = draw.textbbox((0, 0), badge, font=HEADING_FONT)
    badge_width = badge_box[2] - badge_box[0] + 30
    draw.rounded_rectangle(
        (1548 - badge_width, 31, 1548, 70),
        10,
        fill=badge_color,
    )
    draw.text(
        (1563 - badge_width, 38),
        badge,
        font=SMALL_FONT,
        fill="#08120d",
    )

    draw.rounded_rectangle((24, 100, 960, 842), 16, fill=PANEL)
    draw.text((48, 120), "Visual context received by Operator", font=HEADING_FONT, fill=TEXT)
    if visual is not None:
        primary = _fit_image(visual, (880, 590))
        frame.paste(primary, (52, 166))
        draw.text((52, 762), visual.name, font=SMALL_FONT, fill=MUTED)
    else:
        draw.text((52, 190), "No image returned by this call", font=BODY_FONT, fill=MUTED)

    thumbnails = [path for path in images if path != visual][:4]
    for thumb_index, path in enumerate(thumbnails):
        x = 52 + thumb_index * 218
        thumb = _fit_image(path, (202, 92))
        frame.paste(thumb, (x, 798))
        # Keep the filename compact and visible over the thumbnail.
        label = path.name[:24]
        draw.rectangle((x, 866, x + 202, 890), fill="#05080ccc")
        draw.text((x + 5, 870), label, font=SMALL_FONT, fill=TEXT)

    draw.rounded_rectangle((984, 100, 1576, 842), 16, fill=PANEL)
    draw.text(
        (1012, 122),
        f"{index + 1:02d}/{total:02d}  {tool}",
        font=TITLE_FONT,
        fill=ACCENT,
    )
    y = 174
    draw.text((1012, y), "CALL", font=SMALL_FONT, fill=MUTED)
    y = _draw_lines(
        draw,
        _wrap_json(row.get("arguments", {}), max_lines=14),
        xy=(1012, y + 25),
        font=MONO_FONT,
        fill=TEXT,
        line_height=25,
    )
    y += 18
    draw.text((1012, y), "RESPONSE", font=SMALL_FONT, fill=MUTED)
    response_color = SUCCESS if response.get("success") is True else WARNING
    _draw_lines(
        draw,
        _wrap_json(_compact_response(response), max_lines=13),
        xy=(1012, y + 25),
        font=MONO_FONT,
        fill=response_color,
        line_height=25,
    )

    draw.rounded_rectangle((24, 854, 1576, 888), 10, fill="#192331")
    progress = (index + 1) / max(1, total)
    draw.rounded_rectangle(
        (24, 854, 24 + int(1552 * progress), 888),
        10,
        fill=ACCENT if index + 1 < total else SUCCESS,
    )
    task_line = textwrap.shorten(task, width=115, placeholder="…")
    draw.text((48, 860), task_line, font=SMALL_FONT, fill="#07121c")
    return frame, latest_agentview


def export_video(
    episode_root: Path,
    output: Path,
    *,
    fps: int = 6,
    seconds_per_call: float = 1.6,
) -> Path:
    episode = json.loads((episode_root / "episode.json").read_text(encoding="utf-8"))
    if episode.get("success") is not True or episode.get("status") != "completed":
        raise ValueError(
            "Refusing to label an incomplete/failed episode as a success video"
        )
    rows = _read_jsonl(episode_root / "operator_context.jsonl")
    if not rows:
        raise ValueError("operator_context.jsonl contains no tool calls")
    task = str(episode.get("task") or "LIBERO manipulation")
    output.parent.mkdir(parents=True, exist_ok=True)
    latest_agentview: Path | None = None
    hold_frames = max(1, round(float(seconds_per_call) * int(fps)))
    with imageio.get_writer(
        output,
        fps=max(1, int(fps)),
        codec="libx264",
        quality=8,
        macro_block_size=None,
        ffmpeg_log_level="error",
    ) as writer:
        for index, row in enumerate(rows):
            card, latest_agentview = render_card(
                row=row,
                index=index,
                total=len(rows),
                task=task,
                latest_agentview=latest_agentview,
                final_success=True,
            )
            pixels = np.asarray(card, dtype=np.uint8)
            duration = hold_frames + (fps if row.get("tool") == "move_to" else 0)
            for _ in range(duration):
                writer.append_data(pixels)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--seconds-per-call", type=float, default=1.6)
    args = parser.parse_args()
    root = args.episode_root.resolve()
    output = (
        args.output.resolve()
        if args.output
        else root / "presentation" / "operator-trace-success.mp4"
    )
    print(
        export_video(
            root,
            output,
            fps=args.fps,
            seconds_per_call=args.seconds_per_call,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
