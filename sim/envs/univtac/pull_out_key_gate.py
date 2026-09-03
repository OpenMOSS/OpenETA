"""Pure contracts for the Isaac51 Pull Out Key pre-action gate."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from sim.envs.univtac.contract import UniVTACContractError
from sim.envs.univtac.observation import array_summary, observation_key_tree, to_numpy

EXPECTED_TASK = "pull_out_key"
EXPECTED_SEED = 1_000_000
ALLOWED_SEEDS = (1_000_000, 1_000_001, 1_000_002)
EXPECTED_SEED_LABEL = "legacy_ftp1_eval_seed_index_aligned"
EXPECTED_INSTRUCTION = "Pull the key out of the slot."
EXPECTED_TACTILE_SENSORS = ("left_tactile", "right_tactile")
REQUIRED_ROOTS = frozenset({"step", "atom", "observation", "embodiment", "tactile", "actor"})
REQUIRED_TACTILE_FIELDS = frozenset({"rgb", "rgb_marker", "marker", "depth", "press_depth", "pose"})


def validate_gate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "task",
        "seed",
        "seed_label",
        "task_config",
        "sensor_type",
        "optical_backend",
        "device",
        "task_instruction",
        "save_host_only",
        "strict_two_tactile_sensors",
        "require_press_depth",
        "require_any_positive_press_depth",
        "run_play_once",
        "call_success_checks",
        "capture_post_action",
        "timeout_seconds",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise UniVTACContractError(f"Pull Out Key gate config is missing: {missing}")
    expected = {
        "task": EXPECTED_TASK,
        "seed_label": EXPECTED_SEED_LABEL,
        "task_config": "demo",
        "sensor_type": "gsmini",
        "optical_backend": "taxim",
        "device": "cuda:0",
        "task_instruction": EXPECTED_INSTRUCTION,
        "save_host_only": True,
        "strict_two_tactile_sensors": True,
        "require_press_depth": True,
        "require_any_positive_press_depth": True,
        "run_play_once": False,
        "call_success_checks": False,
        "capture_post_action": False,
        "timeout_seconds": 1200,
    }
    for key, value in expected.items():
        if config[key] != value:
            raise UniVTACContractError(
                f"Pull Out Key gate requires {key}={value!r}, got {config[key]!r}"
            )
    if config["seed"] not in ALLOWED_SEEDS:
        raise UniVTACContractError(
            f"Pull Out Key gate seed must be one of {ALLOWED_SEEDS}, got {config['seed']!r}"
        )
    return dict(config)


def seed_dir_name(seed: int) -> str:
    return f"pull_out_key_seed{seed}"


def success_classification(seed: int) -> str:
    return f"scoped_launcher_and_pull_out_key_seed{seed}_gate_passed"


def _require_array(
    value: Any,
    *,
    path: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    array = to_numpy(value)
    summary = array_summary(array)
    if array.shape != shape:
        raise UniVTACContractError(f"{path} expected shape {shape}, got {tuple(array.shape)}")
    if dtype is not None and array.dtype != dtype:
        raise UniVTACContractError(f"{path} expected dtype {dtype}, got {array.dtype}")
    return array, summary


def summarize_pull_out_key_observation(
    observation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the one native observation and summarize contact candidates."""

    missing_roots = sorted(REQUIRED_ROOTS - observation.keys())
    if missing_roots:
        raise UniVTACContractError(f"native observation is missing roots: {missing_roots}")
    cameras = observation["observation"]
    embodiment = observation["embodiment"]
    tactile = observation["tactile"]
    actors = observation["actor"]
    if not all(isinstance(value, Mapping) for value in (cameras, embodiment, tactile, actors)):
        raise UniVTACContractError("native observation groups must be mappings")
    summary: dict[str, Any] = {
        "key_tree": observation_key_tree(observation),
        "step": int(observation["step"]),
        "atom": {"id": int(observation["atom"]["id"]), "tag": str(observation["atom"]["tag"])},
        "cameras": {},
        "embodiment": {},
        "tactile": {},
        "actor": {},
    }
    for name in ("head", "wrist"):
        if name not in cameras or "rgb" not in cameras[name]:
            raise UniVTACContractError(f"observation.{name}.rgb is missing")
        _, metadata = _require_array(
            cameras[name]["rgb"],
            path=f"observation.{name}.rgb",
            shape=(270, 480, 3),
            dtype=np.dtype("uint8"),
        )
        summary["cameras"][name] = {"rgb": metadata}
    for name, shape in (("joint", (9,)), ("ee", (7,))):
        if name not in embodiment:
            raise UniVTACContractError(f"embodiment.{name} is missing")
        _, metadata = _require_array(embodiment[name], path=f"embodiment.{name}", shape=shape)
        summary["embodiment"][name] = metadata
    actual_sensors = tuple(sorted(str(name) for name in tactile))
    if actual_sensors != EXPECTED_TACTILE_SENSORS:
        raise UniVTACContractError(
            f"expected tactile sensors {EXPECTED_TACTILE_SENSORS}, got {actual_sensors}"
        )
    contact_sensors: dict[str, Any] = {}
    for sensor_name in EXPECTED_TACTILE_SENSORS:
        packet = tactile[sensor_name]
        if not isinstance(packet, Mapping):
            raise UniVTACContractError(f"{sensor_name} packet must be a mapping")
        missing = sorted(REQUIRED_TACTILE_FIELDS - packet.keys())
        if missing:
            raise UniVTACContractError(f"{sensor_name} packet is missing: {missing}")
        sensor_summary: dict[str, Any] = {}
        for name in ("rgb", "rgb_marker"):
            _, metadata = _require_array(
                packet[name],
                path=f"tactile.{sensor_name}.{name}",
                shape=(240, 320, 3),
                dtype=np.dtype("uint8"),
            )
            sensor_summary[name] = metadata
        for name, shape in (
            ("marker", (2, 63, 2)),
            ("depth", (240, 320)),
            ("press_depth", (240, 320)),
            ("pose", (7,)),
        ):
            array, metadata = _require_array(
                packet[name], path=f"tactile.{sensor_name}.{name}", shape=shape
            )
            if name == "press_depth" and float(array.min()) < 0:
                raise UniVTACContractError(f"{sensor_name}.press_depth must be non-negative")
            sensor_summary[name] = metadata
        press_depth = to_numpy(packet["press_depth"])
        positive_count = int(np.count_nonzero(press_depth > 0))
        contact_sensors[sensor_name] = {
            "minimum": float(press_depth.min()),
            "maximum": float(press_depth.max()),
            "mean": float(press_depth.mean()),
            "positive_pixel_count": positive_count,
            "positive_pixel_ratio": float(positive_count / press_depth.size),
            "sensor_contact_candidate": bool(float(press_depth.max()) > 0),
        }
        summary["tactile"][sensor_name] = sensor_summary
    for actor_name in ("slot", "key"):
        if actor_name not in actors:
            raise UniVTACContractError(f"actor.{actor_name} is missing")
        _, metadata = _require_array(actors[actor_name], path=f"actor.{actor_name}", shape=(7,))
        summary["actor"][actor_name] = metadata
    contacts = [item["sensor_contact_candidate"] for item in contact_sensors.values()]
    contact_summary = {
        "sensors": contact_sensors,
        "any_contact_candidate": any(contacts),
        "bilateral_contact_candidate": all(contacts),
        "interpretation": "positive press_depth contact candidate only",
    }
    return summary, contact_summary
