#!/usr/bin/env python3
"""Run the pure libuipc two-tetrahedron CUDA probe without Isaac or TacEx."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np


EVENT_PATH: Path | None = None


def _event(events: list[dict[str, Any]], name: str) -> None:
    events.append({"event": name, "monotonic_seconds": time.monotonic()})
    print(json.dumps(events[-1], sort_keys=True), flush=True)
    if EVENT_PATH is not None:
        with EVENT_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(events[-1], sort_keys=True) + "\n")


def _stage(events: list[dict[str, Any]], name: str, operation: Callable[[], Any]) -> Any:
    _event(events, f"{name}_started")
    value = operation()
    _event(events, f"{name}_completed")
    return value


def run_probe(workspace: Path) -> dict[str, Any]:
    from uipc import Engine, Scene, Vector3, World, builtin, view
    from uipc.constitution import AffineBodyConstitution
    from uipc.geometry import label_surface, label_triangle_orient, tetmesh
    from uipc.unit import GPa, MPa

    events: list[dict[str, Any]] = []
    workspace.mkdir(parents=True, exist_ok=False)
    engine = _stage(events, "engine_construct", lambda: Engine("cuda", str(workspace)))

    def make_scene():
        config = Scene.default_config()
        config["dt"] = 0.02
        config["gravity"] = [[0.0], [-9.8], [0.0]]
        return Scene(config)

    scene = _stage(events, "scene_construct", make_scene)

    def make_geometry():
        constitution = AffineBodyConstitution()
        scene.constitution_tabular().insert(constitution)
        scene.contact_tabular().default_model(0.5, 1.0 * GPa)
        contact = scene.contact_tabular().default_element()
        vertices = np.array(
            [[0, 1, 0], [0, 0, 1], [-math.sqrt(3) / 2, 0, -0.5], [math.sqrt(3) / 2, 0, -0.5]],
            dtype=np.float64,
        )
        tetrahedra = np.array([[0, 1, 2, 3]], dtype=np.int32)
        base = tetmesh(vertices, tetrahedra)
        constitution.apply_to(base, 100 * MPa)
        contact.apply_to(base)
        label_surface(base)
        label_triangle_orient(base)
        upper = base.copy()
        view(upper.positions())[:] += Vector3.UnitY() * 1.5
        lower = base.copy()
        view(lower.instances().find(builtin.is_fixed))[:] = 1
        scene.objects().create("upper_tet").geometries().create(upper)
        scene.objects().create("lower_tet").geometries().create(lower)
        return {"vertex_count": 8, "tetrahedron_count": 2}

    geometry = _stage(events, "geometry_construct", make_geometry)
    world = _stage(events, "world_construct", lambda: World(engine))
    _stage(events, "world_init", lambda: world.init(scene))
    _stage(events, "initial_retrieve", world.retrieve)
    _stage(events, "first_advance", world.advance)
    _stage(events, "post_advance_retrieve", world.retrieve)
    _stage(events, "probe", lambda: None)
    return {
        "schema_version": "openeta.univtac.libuipc_core_probe.v1",
        "runtime_variant": "blackwell_compat_adaptation_v1",
        "success": True,
        "classification": "passed",
        "workspace": str(workspace.resolve()),
        "geometry": geometry,
        "world_frame": int(world.frame()),
        "events": events,
        "forbidden_modules_loaded": sorted(
            name for name in sys.modules if name.split(".")[0] in {"isaacsim", "isaaclab", "omni", "carb", "tacex_uipc", "openeta"}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    args = parser.parse_args()
    global EVENT_PATH
    EVENT_PATH = args.events
    EVENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = run_probe(args.workspace.resolve())
        if payload["forbidden_modules_loaded"]:
            raise RuntimeError(f"forbidden modules loaded: {payload['forbidden_modules_loaded']}")
    except BaseException as exc:
        payload = {
            "schema_version": "openeta.univtac.libuipc_core_probe.v1",
            "runtime_variant": "blackwell_compat_adaptation_v1",
            "success": False,
            "classification": "libuipc_core_sm120_native_error",
            "error_class": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "events": [json.loads(line) for line in EVENT_PATH.read_text(encoding="utf-8").splitlines()] if EVENT_PATH.exists() else [],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(0 if payload.get("success") else 1)


if __name__ == "__main__":
    main()
