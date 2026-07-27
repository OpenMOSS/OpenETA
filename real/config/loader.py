"""Build a :class:`~real.env.RealRobotEnv` from a declarative config file.

A config is a JSON document describing the cameras and (optional) robot arm of
one deployment cell. Drivers are referenced by their registry key (see
``real.registry``) so no user config imports a vendor SDK. Intrinsics and
extrinsics are carried verbatim from the file — leave them ``{}`` until
calibration produces them.

Schema (see ``real/config/ur5e_bench.example.json`` for a template)::

    {
      "task": "",                      # optional default task string
      "home_on_reset": false,          # optional; keep false for observation-first
      "robot": {                       # optional; omit for camera-only cells
        "driver": "ur5e", "name": "arm", "ip": "ROBOT_CONTROLLER_IP", ...
      },
      "cameras": [
        {"driver": "webcam", "name": "wrist", "device": "/dev/cam_left", ...},
        {"driver": "realsense_d435", "name": "wrist_camera", "serial": null,
         "enabled": true, ...}
      ]
    }

A camera entry with ``"enabled": false`` is skipped — handy for hardware that
is wired into the config but not currently plugged in (e.g. an L515 that is not
connected yet).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from real.cameras.base import CameraConfig
from real.env import RealRobotEnv
from real.registry import make_camera, make_robot
from real.robots.base import RobotConfig


def _split_driver(entry: dict[str, Any], *, what: str) -> tuple[str, dict[str, Any]]:
    """Pop the required ``driver`` key, returning it and the remaining fields."""
    fields = dict(entry)
    driver = fields.pop("driver", None)
    if not driver:
        raise ValueError(f"{what} config entry is missing required 'driver' key: {entry!r}")
    return driver, fields


def _build_config(cls: type, fields: dict[str, Any], *, what: str):
    """Instantiate a config dataclass, rejecting unknown keys with a clear error."""
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = set(fields) - valid
    if unknown:
        raise ValueError(
            f"{what} config has unknown key(s) {sorted(unknown)}; "
            f"valid keys are {sorted(valid)}."
        )
    return cls(**fields)


def build_env_from_spec(spec: dict[str, Any], *, task: str | None = None) -> RealRobotEnv:
    """Construct a :class:`RealRobotEnv` from an already-parsed config dict."""
    if "cameras" not in spec or not isinstance(spec["cameras"], list):
        raise ValueError("config must contain a 'cameras' list.")

    cameras = []
    for entry in spec["cameras"]:
        if entry.get("enabled", True) is False:
            continue  # wired in config but not currently connected
        # 'enabled' is a loader-level flag, not a CameraConfig field.
        entry = {k: v for k, v in entry.items() if k != "enabled"}
        driver, fields = _split_driver(entry, what="camera")
        cameras.append(make_camera(driver, _build_config(CameraConfig, fields, what="camera")))

    if not cameras:
        raise ValueError("config produced no enabled cameras.")

    robot = None
    robot_spec = spec.get("robot")
    if robot_spec:
        driver, fields = _split_driver(dict(robot_spec), what="robot")
        robot = make_robot(driver, _build_config(RobotConfig, fields, what="robot"))

    return RealRobotEnv(
        cameras=cameras,
        robot=robot,
        task=task if task is not None else spec.get("task", ""),
        home_on_reset=bool(spec.get("home_on_reset", False)),
    )


def load_spec(path: str | Path) -> dict[str, Any]:
    """Read and parse a JSON config file."""
    text = Path(path).read_text(encoding="utf-8")
    spec = json.loads(text)
    if not isinstance(spec, dict):
        raise ValueError(f"config at {path} must be a JSON object, got {type(spec).__name__}.")
    return spec


def build_env_from_file(path: str | Path, *, task: str | None = None) -> RealRobotEnv:
    """Read a JSON config file and build the described env."""
    return build_env_from_spec(load_spec(path), task=task)
