"""Experiment-specific extraction from durable visual-history eval rollouts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from adapter.protocol import JsonDict
from agent.evals.store import EvaluationRunStore


VISUAL_HISTORY_ROLLOUT_METRICS_SCHEMA_VERSION = (
    "openeta.visual_history_rollout_metrics.v1"
)
VISUAL_HISTORY_STATE_CASE_SCHEMA_VERSION = "openeta.visual_history_state_case.v1"


def extract_visual_history_rollouts(store: EvaluationRunStore) -> JsonDict:
    """Extract ABC-specific metrics without modifying the universal evaluator."""

    compiled = store.compiled_plan()
    raw_jobs = compiled.get("jobs") if isinstance(compiled.get("jobs"), list) else []
    jobs = {
        str(item.get("job_id") or ""): item
        for item in raw_jobs
        if isinstance(item, dict)
    }
    rows: list[JsonDict] = []
    probe_cases: list[JsonDict] = []
    for final in store.final_results():
        job_id = str(final.get("job_id") or "")
        job = jobs.get(job_id, {})
        outcome = final.get("outcome")
        outcome = outcome if isinstance(outcome, dict) else {}
        episode = outcome.get("episode")
        episode = episode if isinstance(episode, dict) else {}
        metadata = episode.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        rollout_dir = _rollout_dir(
            store=store,
            job_id=job_id,
            final=final,
            metadata=metadata,
        )
        model_calls = _jsonl(rollout_dir / "model_calls.jsonl")
        transitions = _jsonl(rollout_dir / "transitions.jsonl")
        planner_calls = [row for row in model_calls if _is_main_planner_call(row)]
        vdm_calls = [row for row in model_calls if _call_role(row) == "visual_differencing"]
        raw_counts: list[int] = []
        compressed_counts: list[int] = []
        for call in planner_calls:
            context = _tool_context(call)
            history = context.get("visual_history")
            history = history if isinstance(history, dict) else {}
            raw = history.get("raw_evidence")
            compressed = history.get("compressed_deltas")
            raw_counts.append(len(raw) if isinstance(raw, list) else 0)
            compressed_counts.append(len(compressed) if isinstance(compressed, list) else 0)
        signatures = [_action_signature(row) for row in transitions]
        repeated_actions = sum(
            current == previous and bool(current)
            for previous, current in zip(signatures, signatures[1:])
        )
        vdm_success = sum(
            bool((row.get("validation") or {}).get("accepted")) for row in vdm_calls
        )
        vdm_durations = [
            float(row.get("duration_s"))
            for row in vdm_calls
            if isinstance(row.get("duration_s"), (int, float))
        ]
        vdm_tokens = sum(_usage_tokens(row) for row in vdm_calls)
        objective_success = _objective_success(episode)
        false_completion = (
            metadata.get("stop_reason") == "task_complete" and not objective_success
        )
        variant_id = str(job.get("variant_id") or "")
        row = {
            "job_id": job_id,
            "pair_id": job.get("pair_id"),
            "source_episode_id": job.get("source_episode_id"),
            "variant_id": variant_id,
            "repeat_index": job.get("repeat_index"),
            "seed": outcome.get("seed"),
            "status": outcome.get("status"),
            "objective_success": objective_success,
            "false_task_completion": false_completion,
            "planner_turn_count": len(planner_calls),
            "transition_count": len(transitions),
            "consecutive_repeated_action_count": repeated_actions,
            "mean_raw_visual_evidence_per_turn": _mean(raw_counts),
            "max_raw_visual_evidence_per_turn": max(raw_counts, default=0),
            "mean_compressed_delta_count_per_turn": _mean(compressed_counts),
            "vdm_call_count": len(vdm_calls),
            "vdm_success_count": vdm_success,
            "vdm_failure_count": len(vdm_calls) - vdm_success,
            "vdm_total_tokens": vdm_tokens,
            "vdm_total_duration_s": round(sum(vdm_durations), 3),
            "rollout_dir": str(rollout_dir),
        }
        rows.append(row)
        probe_cases.extend(
            _state_probe_cases(
                job=job,
                outcome=outcome,
                planner_calls=planner_calls,
                transitions=transitions,
            )
        )

    variants: JsonDict = {}
    for variant_id in sorted({str(row.get("variant_id") or "") for row in rows}):
        selected = [row for row in rows if row.get("variant_id") == variant_id]
        variants[variant_id] = {
            "job_count": len(selected),
            "objective_success_count": sum(bool(row["objective_success"]) for row in selected),
            "false_task_completion_count": sum(
                bool(row["false_task_completion"]) for row in selected
            ),
            "mean_planner_turn_count": _mean(
                [int(row["planner_turn_count"]) for row in selected]
            ),
            "mean_consecutive_repeated_action_count": _mean(
                [int(row["consecutive_repeated_action_count"]) for row in selected]
            ),
            "mean_raw_visual_evidence_per_turn": _mean(
                [float(row["mean_raw_visual_evidence_per_turn"]) for row in selected]
            ),
            "vdm_call_count": sum(int(row["vdm_call_count"]) for row in selected),
            "vdm_failure_count": sum(int(row["vdm_failure_count"]) for row in selected),
            "vdm_total_tokens": sum(int(row["vdm_total_tokens"]) for row in selected),
            "vdm_total_duration_s": round(
                sum(float(row["vdm_total_duration_s"]) for row in selected), 3
            ),
        }
    pair_coverage = Counter(str(row.get("pair_id") or "") for row in rows)
    report = {
        "schema_version": VISUAL_HISTORY_ROLLOUT_METRICS_SCHEMA_VERSION,
        "run_id": store.run_id,
        "plan_sha256": compiled.get("plan_sha256"),
        "job_count": len(rows),
        "state_probe_case_count": len(probe_cases),
        "complete_pair_count": sum(count == len(variants) for count in pair_coverage.values()),
        "variants": variants,
        "jobs": rows,
    }
    output = store.root / "extractors" / "visual_history"
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "metrics.json", report)
    _write_jsonl(output / "state_probe_cases.jsonl", probe_cases)
    report["state_probe_cases_path"] = str(output / "state_probe_cases.jsonl")
    report["metrics_path"] = str(output / "metrics.json")
    return report


def _rollout_dir(
    *,
    store: EvaluationRunStore,
    job_id: str,
    final: JsonDict,
    metadata: JsonDict,
) -> Path:
    """Resolve a rollout from durable evaluator ownership, with legacy fallback.

    Batch run metadata is recorded in ``session_index.json``.  The reduced
    episode outcome deliberately does not duplicate that workspace payload, so
    extractors must not rely on ``outcome.episode.metadata.workspace`` being
    present.
    """

    attempt = int(final.get("attempt") or 0)
    if attempt > 0:
        attempt_root = store.attempt_dir(job_id, attempt)
        index_path = attempt_root / "session_index.json"
        if index_path.is_file():
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                payload = {}
            sessions = payload.get("sessions") if isinstance(payload, dict) else {}
            if isinstance(sessions, dict):
                entries = list(sessions.items())
            elif isinstance(sessions, list):
                entries = [
                    (str(item.get("session_id") or ""), item)
                    for item in sessions
                    if isinstance(item, dict)
                ]
            else:
                entries = []
            matching = []
            for session_id, entry in entries:
                entry = entry if isinstance(entry, dict) else {}
                session_metadata = entry.get("metadata")
                session_metadata = (
                    session_metadata if isinstance(session_metadata, dict) else {}
                )
                evaluation = session_metadata.get("evaluation")
                evaluation = evaluation if isinstance(evaluation, dict) else {}
                if evaluation.get("job_id") == job_id or session_metadata.get(
                    "episode_id"
                ) == job_id:
                    matching.append((session_id, entry))
            candidates = matching or entries
            for session_id, entry in candidates:
                if session_id:
                    derived = attempt_root / "sessions" / session_id / "rollout"
                    if derived.is_dir():
                        return derived
                session_path = str(entry.get("session_path") or "")
                if session_path:
                    stored = Path(session_path).parent / "rollout"
                    if stored.is_dir():
                        return stored
        discovered = sorted((attempt_root / "sessions").glob("*/rollout"))
        if len(discovered) == 1:
            return discovered[0]

    workspace = metadata.get("workspace")
    workspace = workspace if isinstance(workspace, dict) else {}
    return Path(str(workspace.get("root") or "")) / "rollout"


def _state_probe_cases(
    *,
    job: JsonDict,
    outcome: JsonDict,
    planner_calls: list[JsonDict],
    transitions: list[JsonDict],
) -> list[JsonDict]:
    cases: list[JsonDict] = []
    for index, call in enumerate(planner_calls):
        context = _tool_context(call)
        transition = transitions[index] if index < len(transitions) else {}
        cases.append(
            {
                "schema_version": VISUAL_HISTORY_STATE_CASE_SCHEMA_VERSION,
                "case_id": f"{job.get('job_id')}:turn-{index:03d}",
                "job_id": job.get("job_id"),
                "pair_id": job.get("pair_id"),
                "variant_id": job.get("variant_id"),
                "turn_index": index,
                "agent_context": context,
                "agent_decision": call.get("parsed_decision"),
                "environment_label": {
                    "reward": transition.get("reward"),
                    "terminated": transition.get("terminated"),
                    "truncated": transition.get("truncated"),
                    "info": transition.get("info"),
                    "final_status": outcome.get("status"),
                },
            }
        )
    return cases


def _is_main_planner_call(row: JsonDict) -> bool:
    context = _tool_context(row)
    return context.get("schema_version") == "openeta.agent_context.v2"


def _call_role(row: JsonDict) -> str:
    request = row.get("semantic_request")
    request = request if isinstance(request, dict) else {}
    metadata = request.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return str(metadata.get("role") or "")


def _tool_context(row: JsonDict) -> JsonDict:
    request = row.get("semantic_request")
    request = request if isinstance(request, dict) else {}
    value = request.get("tool_context")
    return dict(value) if isinstance(value, dict) else {}


def _action_signature(row: JsonDict) -> str:
    action = row.get("action")
    action = action if isinstance(action, dict) else {}
    command = action.get("command")
    if not isinstance(command, dict):
        return ""
    return json.dumps(command, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _usage_tokens(row: JsonDict) -> int:
    result = row.get("result")
    result = result if isinstance(result, dict) else {}
    details = result.get("details")
    details = details if isinstance(details, dict) else {}
    usage = details.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    total = usage.get("total_tokens")
    if isinstance(total, int) and not isinstance(total, bool) and total > 0:
        return total
    return sum(
        int(usage.get(key) or 0) for key in ("prompt_tokens", "completion_tokens")
    )


def _objective_success(episode: JsonDict) -> bool:
    for step in episode.get("steps") or []:
        result = step.get("step_result") if isinstance(step, dict) else None
        if not isinstance(result, dict):
            continue
        reward = result.get("reward")
        if isinstance(reward, (int, float)) and not isinstance(reward, bool) and reward > 0:
            return True
        info = result.get("info")
        if isinstance(info, dict) and any(
            info.get(key) is True
            for key in (
                "task_success",
                "environment_success",
                "checker_success",
                "benchmark_success",
            )
        ):
            return True
    return False


def _jsonl(path: Path) -> list[JsonDict]:
    if not path.is_file():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _mean(values: list[int | float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _write_json(path: Path, payload: JsonDict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)
