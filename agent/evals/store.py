"""Crash-tolerant filesystem store for evaluation runs and attempts."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from adapter.protocol import JsonDict
from agent.evals.plan import EvaluationJob


EVALUATION_RUN_SCHEMA_VERSION = "openeta.evaluation_run.v1"
EVALUATION_ATTEMPT_SCHEMA_VERSION = "openeta.evaluation_attempt.v1"
DEFAULT_EVALUATION_ROOT = Path(".openeta_eval") / "runs"


@dataclass(frozen=True, slots=True)
class EvaluationAttemptRef:
    job_id: str
    attempt: int
    path: Path


class EvaluationRunStore:
    """Own immutable inputs and atomically committed per-job results."""

    def __init__(self, run_id: str, *, root: str | Path = DEFAULT_EVALUATION_ROOT) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
            raise ValueError("evaluation run_id contains unsupported characters")
        self.run_id = run_id
        self.root = Path(root) / run_id
        self._lock = threading.RLock()

    @classmethod
    def create(
        cls,
        run_id: str,
        *,
        compiled_plan: JsonDict,
        root: str | Path = DEFAULT_EVALUATION_ROOT,
        provenance: JsonDict | None = None,
    ) -> "EvaluationRunStore":
        store = cls(run_id, root=root)
        store.root.mkdir(parents=True, exist_ok=True)
        expected_hash = str(compiled_plan.get("plan_sha256") or "")
        compiled_path = store.root / "compiled_plan.json"
        if compiled_path.exists():
            existing = _read_json(compiled_path)
            if str(existing.get("plan_sha256") or "") != expected_hash:
                raise ValueError(
                    f"evaluation run {run_id} already exists with a different plan"
                )
        else:
            _atomic_write_json(compiled_path, compiled_plan)
        run_path = store.root / "run.json"
        if not run_path.exists():
            now = time.time()
            _atomic_write_json(
                run_path,
                {
                    "schema_version": EVALUATION_RUN_SCHEMA_VERSION,
                    "run_id": run_id,
                    "plan_id": (compiled_plan.get("plan") or {}).get("plan_id"),
                    "plan_sha256": expected_hash,
                    "status": "created",
                    "created_at_s": now,
                    "updated_at_s": now,
                    "provenance": provenance or {},
                },
            )
        return store

    def exists(self) -> bool:
        return (self.root / "run.json").is_file()

    def compiled_plan(self) -> JsonDict:
        return _read_json(self.root / "compiled_plan.json")

    def run_metadata(self) -> JsonDict:
        return _read_json(self.root / "run.json")

    def set_run_status(self, status: str, **changes: object) -> None:
        with self._lock:
            payload = self.run_metadata()
            payload.update(changes)
            payload.update({"status": status, "updated_at_s": time.time()})
            _atomic_write_json(self.root / "run.json", payload)
            self._journal("run_status", {"status": status, **changes})

    def next_attempt(self, job_id: str) -> int:
        attempts = self.job_dir(job_id) / "attempts"
        observed = [
            int(path.parent.name)
            for path in attempts.glob("*/state.json")
            if path.parent.name.isdigit()
        ] if attempts.exists() else []
        return max(observed, default=0) + 1

    def settled_attempt_count(self, job_id: str) -> int:
        attempts = self.job_dir(job_id) / "attempts"
        return len(list(attempts.glob("*/result.json"))) if attempts.exists() else 0

    def start_attempt(
        self,
        job: EvaluationJob,
        *,
        attempt: int,
        spec: JsonDict,
    ) -> EvaluationAttemptRef:
        path = self.attempt_dir(job.job_id, attempt)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": EVALUATION_ATTEMPT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "job_id": job.job_id,
            "attempt": attempt,
            "status": "running",
            "started_at_s": time.time(),
            "job": job.to_dict(),
            "scheduled_episode": spec,
        }
        _atomic_write_json(path / "state.json", payload)
        self._journal(
            "attempt_started",
            {"job_id": job.job_id, "attempt": attempt},
        )
        return EvaluationAttemptRef(job.job_id, attempt, path)

    def abandon_incomplete_attempts(self) -> list[JsonDict]:
        """Close attempt records left running by a dead evaluator process."""

        abandoned: list[JsonDict] = []
        jobs_root = self.root / "jobs"
        if not jobs_root.exists():
            return abandoned
        for state_path in sorted(jobs_root.glob("*/attempts/*/state.json")):
            state = _read_json(state_path)
            if state.get("status") != "running":
                continue
            state.update(
                {
                    "status": "abandoned",
                    "completed_at_s": time.time(),
                    "failure": {
                        "class": "infrastructure",
                        "stage": "scheduler",
                        "code": "scheduler_interrupted",
                        "retryable": True,
                    },
                }
            )
            _atomic_write_json(state_path, state)
            abandoned.append(
                {
                    "job_id": state.get("job_id"),
                    "attempt": state.get("attempt"),
                }
            )
        if abandoned:
            self._journal("attempts_abandoned", {"attempts": abandoned})
        return abandoned

    def record_attempt_result(
        self,
        job: EvaluationJob,
        *,
        attempt: int,
        outcome: JsonDict,
        failure: JsonDict,
        retryable: bool,
        max_attempts: int,
    ) -> JsonDict:
        with self._lock:
            path = self.attempt_dir(job.job_id, attempt)
            state = _read_json(path / "state.json")
            completed_at = time.time()
            result = {
                "schema_version": EVALUATION_ATTEMPT_SCHEMA_VERSION,
                "run_id": self.run_id,
                "job_id": job.job_id,
                "attempt": attempt,
                "status": outcome.get("status"),
                "started_at_s": state.get("started_at_s"),
                "completed_at_s": completed_at,
                "duration_s": outcome.get("duration_s"),
                "failure": failure,
                "outcome": outcome,
            }
            _atomic_write_json(path / "result.json", result)
            state.update(
                {
                    "status": "completed",
                    "completed_at_s": completed_at,
                    "outcome_status": outcome.get("status"),
                    "failure": failure,
                }
            )
            _atomic_write_json(path / "state.json", state)
            attempts_used = self.settled_attempt_count(job.job_id)
            terminal = not retryable or attempts_used >= max_attempts
            if terminal:
                _atomic_write_json(
                    self.job_dir(job.job_id) / "final.json",
                    {
                        **result,
                        "terminal": True,
                        "attempts_used": attempts_used,
                    },
                )
            self._journal(
                "attempt_completed",
                {
                    "job_id": job.job_id,
                    "attempt": attempt,
                    "status": outcome.get("status"),
                    "failure": failure,
                    "terminal": terminal,
                },
            )
            return {"terminal": terminal, "result": result}

    def final_result(self, job_id: str) -> JsonDict | None:
        path = self.job_dir(job_id) / "final.json"
        return _read_json(path) if path.is_file() else None

    def final_results(self) -> list[JsonDict]:
        jobs_root = self.root / "jobs"
        if not jobs_root.exists():
            return []
        return [_read_json(path) for path in sorted(jobs_root.glob("*/final.json"))]

    def attempt_results(self, job_id: str) -> list[JsonDict]:
        attempts = self.job_dir(job_id) / "attempts"
        if not attempts.exists():
            return []
        return [
            _read_json(path)
            for path in sorted(attempts.glob("*/result.json"))
        ]

    def write_report(self, payload: JsonDict) -> Path:
        path = self.root / "report.json"
        _atomic_write_json(path, payload)
        return path

    def read_report(self) -> JsonDict | None:
        path = self.root / "report.json"
        return _read_json(path) if path.is_file() else None

    def job_dir(self, job_id: str) -> Path:
        return self.root / "jobs" / job_id

    def attempt_dir(self, job_id: str, attempt: int) -> Path:
        return self.job_dir(job_id) / "attempts" / f"{attempt:03d}"

    def _journal(self, event: str, payload: JsonDict) -> None:
        with self._lock:
            path = self.root / "journal.jsonl"
            row = {
                "schema_version": "openeta.evaluation_journal.v1",
                "timestamp_s": time.time(),
                "event": event,
                **payload,
            }
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
