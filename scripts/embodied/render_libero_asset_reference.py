#!/usr/bin/env python3
"""Render a canonical four-view LIBERO MJCF asset reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw


def render_reference(xml_path: Path, output: Path, *, size: int = 480) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=size, width=size)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[1] = 1
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = model.stat.center
    camera.distance = max(float(model.stat.extent) * 2.4, 0.16)
    views: list[Image.Image] = []
    for label, azimuth, elevation in (
        ("front", 90.0, -10.0),
        ("left", 180.0, -10.0),
        ("back", 270.0, -10.0),
        ("top-oblique", 35.0, -35.0),
    ):
        camera.azimuth = azimuth
        camera.elevation = elevation
        renderer.update_scene(data, camera=camera, scene_option=option)
        image = Image.fromarray(renderer.render()).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 150, 28), fill=(0, 0, 0))
        draw.text((8, 6), label, fill=(255, 255, 255))
        views.append(image)
    renderer.close()

    sheet = Image.new("RGB", (size * 2, size * 2), (24, 24, 24))
    for index, image in enumerate(views):
        sheet.paste(image, ((index % 2) * size, (index // 2) * size))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=480)
    args = parser.parse_args()
    render_reference(args.xml.expanduser().resolve(), args.output.expanduser().resolve(), size=args.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
