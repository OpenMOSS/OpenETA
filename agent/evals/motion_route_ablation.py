"""Experiment-local metrics for the direct-vs-waypoint motion ablation."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Iterable

from adapter.protocol import JsonDict
from agent.evals.store import EvaluationRunStore


SCHEMA_VERSION = "openeta.motion_route_ablation_metrics.v1"
_MOTION_TOOLS = {"move_to", "follow_eef_trajectory"}


def summarize_motion_route_job(
    *,
    job: JsonDict,
    final_result: JsonDict,
    tool_rows: Iterable[JsonDict],
    model_rows: Iterable[JsonDict],
) -> JsonDict:
    """Summarize one rollout without adding experiment fields to universal eval."""

    episode = _dict(job.get("episode"))
    metadata = _dict(episode.get("metadata"))
    method = str(metadata.get("route_method") or "unknown")
    model_rows = list(model_rows)
    tool_ends = [
        _dict(row.get("event"))
        for row in tool_rows
        if _dict(row.get("event")).get("phase") == "end"
    ]
    previews = [event for event in tool_ends if event.get("name") == "ik_preview_check"]
    direct = [event for event in tool_ends if event.get("name") == "move_to"]
    trajectories = [
        event for event in tool_ends if event.get("name") == "follow_eef_trajectory"
    ]
    motion_events = [event for event in tool_ends if event.get("name") in _MOTION_TOOLS]
    final_motion_event = motion_events[-1] if motion_events else {}
    outputs = _dict(_dict(final_motion_event.get("details")).get("outputs"))
    motion = _dict(outputs.get("motion_summary"))
    full_response = _load_full_response(outputs)
    collision = _dict(motion.get("collision") or full_response.get("collision"))
    target_xyz = _xyz(metadata.get("target_xyz"))
    end_xyz = _xyz(_dict(motion.get("end")).get("xyz"))
    position_error_m = _euclidean(target_xyz, end_xyz)
    max_axis_error_m = _max_axis_error(target_xyz, end_xyz)
    tolerance = _float(metadata.get("position_tolerance_m")) or 0.01
    selected_waypoint_count = 0
    if trajectories:
        params = _dict(trajectories[-1].get("parameters"))
        receipt_ids = params.get("ik_receipt_ids")
        trajectory = params.get("trajectory")
        if isinstance(receipt_ids, list):
            selected_waypoint_count = len(receipt_ids)
        elif isinstance(trajectory, list):
            selected_waypoint_count = len(trajectory)

    extra_tools = sorted(
        {
            str(event.get("name") or "")
            for event in tool_ends
            if str(event.get("name") or "")
            not in {"ik_preview_check", "move_to", "follow_eef_trajectory"}
        }
    )
    preview_success_count = sum(event.get("success") is True for event in previews)
    if method == "direct":
        method_adherent = (
            len(previews) == 1
            and len(direct) == 1
            and not trajectories
        )
    elif method == "agent_waypoints":
        method_adherent = (
            2 <= len(previews) <= 5
            and not direct
            and len(trajectories) == 1
            and 2 <= selected_waypoint_count <= 5
            and selected_waypoint_count <= preview_success_count
        )
    else:
        method_adherent = False

    usage = _sum_usage(model_rows)
    execution_model_rows = _model_rows_through_motion(model_rows)
    execution_usage = _sum_usage(execution_model_rows)
    reached = (motion.get("reached_target") if "reached_target" in motion else full_response.get("reached_target")) is True
    collision_detected = collision.get("detected") is True
    within_tolerance = max_axis_error_m is not None and max_axis_error_m <= tolerance
    execution_success = reached and not collision_detected and within_tolerance
    outcome = _dict(final_result.get("outcome"))
    compact_controller = _dict(motion.get("controller_receipt"))
    full_controller = _dict(full_response.get("controller_receipt"))
    controller = full_controller or compact_controller
    motion_profile = _dict(
        motion.get("motion_execution_profile")
        or full_response.get("motion_execution_profile")
        or full_controller.get("motion_execution_profile")
        or compact_controller.get("motion_execution_profile")
    )
    parameters = _dict(final_motion_event.get("parameters"))
    route_points = _route_points(parameters)
    start_xyz = _xyz(_dict(motion.get("start") or full_response.get("start")).get("xyz"))
    route_length_m = _polyline_length(start_xyz, route_points)
    straight_line_m = _euclidean(start_xyz, target_xyz)
    route_detour_ratio = (
        route_length_m / straight_line_m
        if route_length_m is not None and straight_line_m not in {None, 0.0}
        else None
    )
    waypoints_requested = full_response.get("waypoints_requested")
    waypoints_completed = full_response.get("waypoints_completed")
    waypoint_results = full_response.get("waypoint_results")
    if not isinstance(waypoint_results, list):
        waypoint_results = motion.get("waypoint_results")
    failed_waypoint_index = None
    if isinstance(waypoint_results, list):
        failed_waypoint_index = next(
            (
                index
                for index, value in enumerate(waypoint_results)
                if isinstance(value, dict) and value.get("reached_target") is not True
            ),
            None,
        )
    waypoint_attainment = [
        {
            "index": index,
            "reached_target": value.get("reached_target") is True,
            "stop_reason": value.get("stop_reason"),
            "steps_executed": value.get("steps_executed"),
            "stable_steps_completed": _dict(
                value.get("controller_receipt")
            ).get("stable_steps_completed"),
            "ik_execution_seed_receipt_id": _dict(
                value.get("controller_receipt")
            ).get("ik_execution_seed_receipt_id"),
        }
        for index, value in enumerate(waypoint_results or [])
        if isinstance(value, dict)
    ]
    sequential = _dict(
        motion.get("sequential_route_preview")
        or full_response.get("sequential_route_preview")
    )
    motion_condition = str(motion_profile.get("condition") or "")
    if not motion_condition:
        if sequential:
            motion_condition = "C"
        elif any(
            isinstance(item.get("stable_steps_completed"), int | float)
            for item in waypoint_attainment
        ):
            # Compatibility for artifacts recorded before the compact motion
            # summary carried the profile. Stable-arrival receipts distinguish
            # condition B from the deliberately legacy-compatible A behavior.
            motion_condition = "B"
        else:
            motion_condition = "A"
    controller_failure = _dict(
        motion.get("controller_failure") or full_response.get("controller_failure")
    )
    minimum_distance = _float(collision.get("minimum_distance_m"))
    hard_stop_distance = _float(collision.get("hard_stop_distance_m"))
    path_safety_stop = bool(
        collision_detected
        or (
            minimum_distance is not None
            and hard_stop_distance is not None
            and minimum_distance < hard_stop_distance
        )
        or str(controller_failure.get("code") or "")
        in {
            "constraint_escape_preview_rejected",
            "all_qp_variants_infeasible",
        }
    )
    return {
        "job_id": job.get("job_id"),
        "source_episode_id": job.get("source_episode_id"),
        "repeat_index": job.get("repeat_index"),
        "ablation_pair_id": metadata.get("ablation_pair_id"),
        "method": method,
        "target_xyz": target_xyz,
        "position_tolerance_m": tolerance,
        "terminal_status": final_result.get("status"),
        "episode_status": outcome.get("status"),
        "method_adherent": method_adherent,
        "instruction_exact": method_adherent and not extra_tools,
        "extra_tool_names": extra_tools,
        "preview_call_count": len(previews),
        "preview_success_count": preview_success_count,
        "preview_rejection_count": len(previews) - preview_success_count,
        "move_to_call_count": len(direct),
        "trajectory_call_count": len(trajectories),
        "selected_waypoint_count": selected_waypoint_count,
        "tool_call_count": len(tool_ends),
        "planner_call_count": len(model_rows),
        "planner_call_count_to_motion": len(execution_model_rows),
        "post_motion_planner_call_count": len(model_rows) - len(execution_model_rows),
        "execution_success": execution_success,
        "reached_target": reached,
        "within_position_tolerance": within_tolerance,
        "collision_detected": collision_detected,
        "path_safety_stop": path_safety_stop,
        "minimum_distance_m": minimum_distance,
        "motion_condition": motion_condition,
        "motion_execution_profile": motion_profile,
        "end_xyz": end_xyz,
        "position_error_m": position_error_m,
        "max_axis_position_error_m": max_axis_error_m,
        "steps_executed": motion.get("steps_executed") or full_response.get("steps_executed"),
        "stop_reason": motion.get("stop_reason") or controller.get("stop_reason"),
        "waypoints_requested": waypoints_requested,
        "waypoints_completed": waypoints_completed,
        "failed_waypoint_index": failed_waypoint_index,
        "waypoint_attainment": waypoint_attainment,
        "stable_waypoint_count": sum(
            isinstance(item.get("stable_steps_completed"), int | float)
            and item.get("stable_steps_completed") > 0
            for item in waypoint_attainment
        ),
        "sequential_waypoints_previewed": sequential.get("waypoints_previewed"),
        "sequential_waypoints_authorized": sequential.get("waypoints_authorized"),
        "controller_failure_code": controller_failure.get("code"),
        "convergence_stall_kind": controller.get("convergence_stall_kind"),
        "route_points": route_points,
        "route_length_m": route_length_m,
        "straight_line_distance_m": straight_line_m,
        "route_detour_ratio": route_detour_ratio,
        "duration_s": final_result.get("duration_s"),
        "usage": usage,
        "usage_to_motion": execution_usage,
    }


def extract_motion_route_ablation(store: EvaluationRunStore) -> JsonDict:
    """Extract completed jobs and persist an experiment-local report."""

    compiled = store.compiled_plan()
    jobs = {
        str(item.get("job_id") or ""): item
        for item in compiled.get("jobs", [])
        if isinstance(item, dict)
    }
    rows: list[JsonDict] = []
    for final_result in store.final_results():
        job_id = str(final_result.get("job_id") or "")
        job = jobs.get(job_id)
        if not isinstance(job, dict):
            continue
        attempt = int(final_result.get("attempt") or 1)
        rollout = _rollout_dir(store.attempt_dir(job_id, attempt))
        tool_rows = _jsonl(rollout / "tool_calls.jsonl")
        model_rows = _jsonl(rollout / "model_calls.jsonl")
        rows.append(
            summarize_motion_route_job(
                job=job,
                final_result=final_result,
                tool_rows=tool_rows,
                model_rows=model_rows,
            )
        )
    rows.sort(
        key=lambda row: (
            str(row.get("ablation_pair_id") or ""),
            int(row.get("repeat_index") or 0),
            str(row.get("method") or ""),
        )
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": store.run_id,
        "job_count": len(rows),
        "jobs": rows,
        "aggregate": _aggregate(rows),
    }
    output_dir = store.root / "extractors" / "motion_route_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _aggregate(rows: list[JsonDict]) -> JsonDict:
    methods: JsonDict = {}
    for method in sorted({str(row.get("method") or "") for row in rows}):
        selected = [row for row in rows if row.get("method") == method]
        methods[method] = _aggregate_rows(selected)
    pairs: JsonDict = {}
    for pair in sorted({str(row.get("ablation_pair_id") or "") for row in rows}):
        selected = [row for row in rows if row.get("ablation_pair_id") == pair]
        pairs[pair] = {
            method: _aggregate_rows(
                [row for row in selected if row.get("method") == method]
            )
            for method in sorted({str(row.get("method") or "") for row in selected})
        }
    conditions = {
        condition: _aggregate_rows(
            [row for row in rows if row.get("motion_condition") == condition]
        )
        for condition in sorted(
            {str(row.get("motion_condition") or "unknown") for row in rows}
        )
    }
    return {"by_method": methods, "by_pair": pairs, "by_condition": conditions}


def _aggregate_rows(rows: list[JsonDict]) -> JsonDict:
    return {
        "count": len(rows),
        "method_adherent_count": sum(row.get("method_adherent") is True for row in rows),
        "instruction_exact_count": sum(row.get("instruction_exact") is True for row in rows),
        "execution_success_count": sum(row.get("execution_success") is True for row in rows),
        "collision_count": sum(row.get("collision_detected") is True for row in rows),
        "path_safety_stop_count": sum(
            row.get("path_safety_stop") is True for row in rows
        ),
        "mean_selected_waypoint_count": _mean(rows, "selected_waypoint_count"),
        "mean_stable_waypoint_count": _mean(rows, "stable_waypoint_count"),
        "mean_steps_executed": _mean(rows, "steps_executed"),
        "mean_planner_calls": _mean(rows, "planner_call_count"),
        "mean_planner_calls_to_motion": _mean(rows, "planner_call_count_to_motion"),
        "mean_total_tokens": _mean_nested(rows, "usage", "total_tokens"),
        "mean_tokens_to_motion": _mean_nested(rows, "usage_to_motion", "total_tokens"),
        "mean_max_axis_position_error_m": _mean(rows, "max_axis_position_error_m"),
        "mean_route_detour_ratio": _mean(rows, "route_detour_ratio"),
        "mean_duration_s": _mean(rows, "duration_s"),
    }


def _rollout_dir(attempt_dir: Path) -> Path:
    index = json.loads((attempt_dir / "session_index.json").read_text(encoding="utf-8"))
    sessions = index.get("sessions")
    if not isinstance(sessions, dict) or not sessions:
        raise ValueError(f"no recorded session in {attempt_dir}")
    session = next(iter(sessions.values()))
    if not isinstance(session, dict):
        raise ValueError(f"invalid session record in {attempt_dir}")
    workspace = _dict(_dict(session.get("metadata")).get("workspace"))
    root = workspace.get("root")
    if root:
        return Path(str(root)) / "rollout"
    return attempt_dir / "sessions" / str(session.get("session_id") or "") / "rollout"


def _jsonl(path: Path) -> list[JsonDict]:
    if not path.is_file():
        return []
    return [
        payload
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for payload in [json.loads(line)]
        if isinstance(payload, dict)
    ]


def _sum_usage(rows: Iterable[JsonDict]) -> JsonDict:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 0,
    }
    for row in rows:
        usage = _dict(_dict(row.get("result")).get("details")).get("usage")
        usage = _dict(usage)
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int | float):
                totals[key] += int(value)
    return totals


def _model_rows_through_motion(rows: list[JsonDict]) -> list[JsonDict]:
    accepted_motion_indexes = [
        index
        for index, row in enumerate(rows)
        if str(_dict(row.get("parsed_decision")).get("name") or "") in _MOTION_TOOLS
        and _dict(row.get("validation")).get("accepted") is True
    ]
    if not accepted_motion_indexes:
        return rows
    return rows[: accepted_motion_indexes[-1] + 1]


def _load_full_response(outputs: JsonDict) -> JsonDict:
    response = _dict(outputs.get("response"))
    path = response.get("response_path")
    if not isinstance(path, str) or not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _route_points(parameters: JsonDict) -> list[list[float]]:
    trajectory = parameters.get("trajectory")
    if isinstance(trajectory, list):
        return [
            point
            for value in trajectory
            if isinstance(value, dict)
            for point in [_xyz(value.get("xyz", value.get("translation_xyz")))]
            if point is not None
        ]
    target = parameters.get("target_pose")
    if isinstance(target, dict):
        point = _xyz(target.get("xyz", target.get("translation_xyz")))
        return [point] if point is not None else []
    return []


def _polyline_length(
    start: list[float] | None,
    points: list[list[float]],
) -> float | None:
    if start is None or not points:
        return None
    total = 0.0
    previous = start
    for point in points:
        segment = _euclidean(previous, point)
        if segment is None:
            return None
        total += segment
        previous = point
    return total


def _mean(rows: list[JsonDict], key: str) -> float | None:
    values = [_float(row.get(key)) for row in rows]
    selected = [value for value in values if value is not None]
    return round(mean(selected), 6) if selected else None


def _mean_nested(rows: list[JsonDict], parent: str, key: str) -> float | None:
    values = [_float(_dict(row.get(parent)).get(key)) for row in rows]
    selected = [value for value in values if value is not None]
    return round(mean(selected), 6) if selected else None


def _dict(value: object) -> JsonDict:
    return value if isinstance(value, dict) else {}


def _float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _xyz(value: object) -> list[float] | None:
    if not isinstance(value, list | tuple) or len(value) < 3:
        return None
    if not all(isinstance(item, int | float) for item in value[:3]):
        return None
    return [float(item) for item in value[:3]]


def _euclidean(left: list[float] | None, right: list[float] | None) -> float | None:
    if left is None or right is None:
        return None
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True)) ** 0.5


def _max_axis_error(left: list[float] | None, right: list[float] | None) -> float | None:
    if left is None or right is None:
        return None
    return max(abs(a - b) for a, b in zip(left, right, strict=True))
