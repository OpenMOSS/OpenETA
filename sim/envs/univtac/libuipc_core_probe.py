"""Stage contract and result classification for the pure libuipc probe."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


STAGES = (
    "engine_construct",
    "scene_construct",
    "geometry_construct",
    "world_construct",
    "world_init",
    "initial_retrieve",
    "first_advance",
    "post_advance_retrieve",
    "probe",
)


def expected_stage_events() -> tuple[str, ...]:
    return tuple(f"{stage}_{suffix}" for stage in STAGES for suffix in ("started", "completed"))


def classify_core_probe(payload: Mapping[str, Any], *, timed_out: bool = False) -> str:
    if timed_out:
        return "libuipc_core_sm120_timeout"
    events = payload.get("events", [])
    names = [item.get("event") for item in events if isinstance(item, Mapping)]
    if payload.get("success") is True and all(name in names for name in expected_stage_events()):
        return "passed"
    return "libuipc_core_sm120_native_error"


def core_run_policy(results: Sequence[Mapping[str, Any]]) -> str | None:
    if not results:
        return "run1"
    if results[0].get("classification") != "passed":
        return None
    if len(results) == 1:
        return "run2"
    return None
