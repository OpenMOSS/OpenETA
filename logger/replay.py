"""Offline episode timeline reconstruction and video export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
from PIL import Image, ImageDraw


def load_episode(episode_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory = Path(episode_dir)
    metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (directory / "steps.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return metadata, events


def build_timeline(episode_dir: str | Path) -> list[dict[str, Any]]:
    """Return the compact observation→decision→verdict→result timeline."""
    _, events = load_episode(episode_dir)
    return [
        {
            "step_index": event.get("step_index"),
            "task": event.get("observation", {}).get("task", ""),
            "camera_refs": event.get("observation", {}).get("camera_refs", []),
            "plan": event.get("plan", {}),
            "safety_verdict": event.get("safety_verdict", {}),
            "action": event.get("action", {}),
            "step_result": event.get("step_result"),
            "failure_verdict": event.get("failure_verdict", {}),
        }
        for event in events
    ]


def _camera_rgb(refs: list[dict[str, Any]], camera: str | None) -> str | None:
    for ref in refs:
        if camera is None or ref.get("frame_id") == camera:
            path = ref.get("rgb")
            if path:
                return str(path)
    return None


def _label(event: dict[str, Any]) -> str:
    action = event.get("action", {})
    command = action.get("command", {}) if isinstance(action, dict) else {}
    request = command.get("request", {}) if isinstance(command, dict) else {}
    name = request.get("name") or action.get("action_type", "")
    result = event.get("step_result")
    reward = result.get("reward") if isinstance(result, dict) else None
    safety = event.get("safety_verdict", {})
    suffix = ""
    if safety:
        suffix = f" safety={safety.get('approved', safety.get('status', 'recorded'))}"
    return f"step={event.get('step_index')} action={name} reward={reward}{suffix}"


def export_video(
    episode_dir: str | Path,
    output: str | Path,
    *,
    camera: str | None = None,
    fps: int = 10,
    overlay: bool = True,
) -> Path:
    """Export logged RGB frames as MP4/GIF, optionally with decision overlay."""
    directory = Path(episode_dir)
    metadata, events = load_episode(directory)
    frames: list[Image.Image] = []
    labels: list[str] = []
    for event in events:
        relative = _camera_rgb(event.get("observation", {}).get("camera_refs", []), camera)
        if relative:
            frames.append(Image.open(directory / relative).convert("RGB"))
            labels.append(_label(event))
    final_relative = _camera_rgb(metadata.get("final_camera_refs", []), camera)
    if final_relative:
        frames.append(Image.open(directory / final_relative).convert("RGB"))
        labels.append(f"final status={metadata.get('status')} success={metadata.get('success')}")
    if not frames:
        raise ValueError(f"No RGB frames found for camera {camera!r}")

    width, height = frames[0].size
    rendered = []
    for frame, label in zip(frames, labels):
        if frame.size != (width, height):
            frame = frame.resize((width, height), Image.Resampling.BILINEAR)
        if overlay:
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, width, min(24, height)), fill=(0, 0, 0))
            draw.text((4, 4), label, fill=(255, 255, 255))
        rendered.append(frame)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=max(1, int(fps)), macro_block_size=None) as writer:
        for frame in rendered:
            writer.append_data(__import__("numpy").asarray(frame))
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args(argv)
    export_video(
        args.episode_dir,
        args.output,
        camera=args.camera,
        fps=args.fps,
        overlay=not args.no_overlay,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
