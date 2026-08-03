#!/usr/bin/env python3
"""Minimal Chrome/Viser viewer for a retained world-frame point cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8924)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--point-size", type=float, default=0.0018)
    parser.add_argument("--per-view", action="store_true")
    parser.add_argument("--gray", action="store_true", help="show geometry in neutral gray")
    args = parser.parse_args()

    import viser

    payload = np.load(args.npz)
    points = np.asarray(payload["points_world"], dtype=np.float32)
    colors = np.asarray(payload["colors_rgb"], dtype=np.uint8)
    if args.gray:
        colors = np.repeat(np.asarray([[185, 185, 185]], dtype=np.uint8), len(points), axis=0)
    server = viser.ViserServer(host=args.host, port=args.port, label="OpenETA point cloud probe")
    server.scene.set_up_direction("+z")
    server.scene.add_point_cloud(
        "/pointcloud/fused",
        points=points,
        colors=colors,
        point_size=float(args.point_size),
        point_shape="rounded",
    )
    # If the probe directory is provided, expose each virtual view as an
    # independently toggleable layer.  This makes registration errors visible
    # instead of hiding them in one blended cloud.
    probe_root = args.npz.parent
    per_view = sorted(probe_root.glob("virtual-*.workspace.world.npz")) if args.per_view else []
    for path in per_view:
        item = np.load(path)
        handle = server.scene.add_point_cloud(
            f"/pointcloud/{path.stem}",
            points=np.asarray(item["points_world"], dtype=np.float32),
            colors=np.asarray(item["colors_rgb"], dtype=np.uint8),
            point_size=0.0012,
            point_shape="rounded",
            visible=False,
        )
        checkbox = server.gui.add_checkbox(f"Show {path.stem}", False)
        checkbox.on_update(lambda event, h=handle, c=checkbox: setattr(h, "visible", bool(c.value)))
    server.gui.add_markdown(
        "# OpenETA multiview point cloud\n"
        "This is the raw fused RGB-D cloud. Orbit with left-drag, pan with right-drag, zoom with wheel.\n"
        f"points: {len(points)}\n"
        f"source: `{args.npz}`"
    )
    print(json.dumps({"url": f"http://127.0.0.1:{args.port}", "points": int(len(points))}), flush=True)
    try:
        import time
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
