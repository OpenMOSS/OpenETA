"""Lossless, training-oriented rollout recording for OpenETA sessions."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import mimetypes
import os
import platform
import re
import subprocess
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from adapter.protocol import EnvAction, EnvObservation, JsonDict, StepResult


ROLLOUT_SCHEMA_VERSION = "openeta.rollout.v1"
MODEL_CALL_SCHEMA_VERSION = "openeta.rollout.model_call.v1"
TOOL_EVENT_SCHEMA_VERSION = "openeta.rollout.tool_event.v1"
TRANSITION_SCHEMA_VERSION = "openeta.rollout.transition.v1"
EPISODE_EVENT_SCHEMA_VERSION = "openeta.rollout.episode_event.v1"
ARTIFACT_SCHEMA_VERSION = "openeta.rollout.artifact.v1"
ROLLOUT_VALIDATION_SCHEMA_VERSION = "openeta.rollout.validation.v1"

_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[A-Za-z0-9.+_-]+/[A-Za-z0-9.+_-]+);base64,(?P<data>.+)$",
    re.DOTALL,
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "refresh_token",
    "secret",
    "set_cookie",
    "x_api_key",
    "access_token",
}
_ARTIFACT_PATH_HINTS = (
    "artifact",
    "attachment",
    "crop",
    "depth",
    "image",
    "mask",
    "overlay",
    "response",
    "rgb",
)


class RolloutRecorder:
    """Persist raw model, tool, and environment evidence beside session memory.

    Recording is deliberately best-effort: an I/O or serialization failure is
    retained in ``errors`` but never interrupts robot or simulator execution.
    """

    def __init__(self, root: str | Path = ".openeta_memory") -> None:
        self.root = Path(root)
        self.session_id = ""
        self.rollout_dir: Path | None = None
        self.errors: list[str] = []
        self._lock = threading.RLock()
        self._next_sequences: dict[str, int] = {}
        self._artifacts: dict[str, JsonDict] = {}

    @property
    def enabled(self) -> bool:
        return self.rollout_dir is not None and bool(self.session_id)

    def start_session(
        self,
        *,
        session_id: str,
        task: str,
        metadata: JsonDict | None = None,
        provenance: JsonDict | None = None,
        resumed: bool = False,
    ) -> None:
        """Create or reopen one session-local rollout bundle."""

        try:
            self._start_session(
                session_id=session_id,
                task=task,
                metadata=metadata,
                provenance=provenance,
                resumed=resumed,
            )
        except Exception as exc:  # noqa: BLE001 - recording cannot fail execution.
            self._remember_error("start_session", exc)

    def record_model_call(
        self,
        *,
        request: Any,
        result: Any,
        decision: Any,
        validation_errors: list[str],
        backend: JsonDict,
        started_at_s: float,
        completed_at_s: float,
    ) -> None:
        """Record one planner attempt before host validation retries continue."""

        if not self.enabled:
            return
        try:
            details = dict(getattr(result, "details", {}) or {})
            exchange = getattr(result, "rollout_exchange", None)
            if not exchange:
                exchange = details.pop("_rollout_exchange", None)
            raw_request: JsonDict = {
                "tool_context": getattr(request, "tool_context", {}),
                "system_prompt": getattr(request, "system_prompt", ""),
                "conversation_messages": getattr(request, "conversation_messages", []),
                "conversation_summary": getattr(request, "conversation_summary", ""),
                "attempt": getattr(request, "attempt", 1),
                "validation_errors": getattr(request, "validation_errors", []),
                "metadata": getattr(request, "metadata", {}),
            }
            payload: JsonDict = {
                "call_id": str(uuid4()),
                "session_id": self.session_id,
                "started_at_s": started_at_s,
                "completed_at_s": completed_at_s,
                "duration_s": max(0.0, completed_at_s - started_at_s),
                "backend": backend,
                "semantic_request": raw_request,
                "provider_exchange": exchange,
                "result": {
                    "payload": getattr(result, "payload", None),
                    "status": _enum_value(getattr(result, "status", None)),
                    "provider": getattr(result, "provider", ""),
                    "model": getattr(result, "model", ""),
                    "details": details,
                },
                "parsed_decision": _decision_payload(decision),
                "validation": {
                    "accepted": not validation_errors,
                    "errors": list(validation_errors),
                },
            }
            self._append_prepared("model_calls.jsonl", MODEL_CALL_SCHEMA_VERSION, payload)
        except Exception as exc:  # noqa: BLE001
            self._remember_error("record_model_call", exc)

    def record_tool_event(self, event: JsonDict) -> None:
        """Record one exact tool boundary event emitted by ``ToolRegistry``."""

        if not self.enabled:
            return
        try:
            self._append_prepared(
                "tool_calls.jsonl",
                TOOL_EVENT_SCHEMA_VERSION,
                {
                    "session_id": self.session_id,
                    "timestamp_s": time.time(),
                    "event": event,
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._remember_error("record_tool_event", exc)

    def record_episode_start(
        self,
        *,
        episode_id: str,
        task: str,
        observation: EnvObservation,
        metadata: JsonDict,
    ) -> None:
        """Persist the initial state even when an episode never reaches a step."""

        if not self.enabled:
            return
        try:
            payload = {
                "session_id": self.session_id,
                "episode_id": episode_id,
                "event": "start",
                "timestamp_s": time.time(),
                "task": task,
                "metadata": metadata,
                "initial_observation": self._observation_payload(
                    observation,
                    role=f"episode/{episode_id}/initial",
                ),
            }
            self._append_prepared(
                "episodes.jsonl",
                EPISODE_EVENT_SCHEMA_VERSION,
                payload,
            )
        except Exception as exc:  # noqa: BLE001
            self._remember_error("record_episode_start", exc)

    def record_transition(
        self,
        *,
        episode_id: str,
        turn_index: int,
        observation: EnvObservation,
        action: EnvAction,
        step_result: StepResult,
        started_at_s: float,
        completed_at_s: float,
    ) -> None:
        """Persist one lossless observation-action-next-observation transition."""

        if not self.enabled:
            return
        try:
            transition_id = f"{episode_id}:{turn_index}"
            payload = {
                "transition_id": transition_id,
                "session_id": self.session_id,
                "episode_id": episode_id,
                "turn_index": turn_index,
                "started_at_s": started_at_s,
                "completed_at_s": completed_at_s,
                "duration_s": max(0.0, completed_at_s - started_at_s),
                "observation": self._observation_payload(
                    observation,
                    role=f"transition/{transition_id}/observation",
                ),
                "action": action.to_dict(),
                "next_observation": self._observation_payload(
                    step_result.observation,
                    role=f"transition/{transition_id}/next_observation",
                ),
                "reward": step_result.reward,
                "terminated": step_result.terminated,
                "truncated": step_result.truncated,
                "info": step_result.info,
            }
            self._append_prepared(
                "transitions.jsonl",
                TRANSITION_SCHEMA_VERSION,
                payload,
            )
        except Exception as exc:  # noqa: BLE001
            self._remember_error("record_transition", exc)

    def record_episode_result(
        self,
        *,
        episode_id: str,
        result: JsonDict,
    ) -> None:
        """Record episode completion and update the session manifest."""

        if not self.enabled:
            return
        try:
            self._append_prepared(
                "episodes.jsonl",
                EPISODE_EVENT_SCHEMA_VERSION,
                {
                    "session_id": self.session_id,
                    "episode_id": episode_id,
                    "event": "result",
                    "timestamp_s": time.time(),
                    "result": result,
                },
            )
            self._update_manifest(
                {
                    "updated_at_s": time.time(),
                    "last_episode": {
                        "episode_id": episode_id,
                        "terminated": result.get("terminated"),
                        "truncated": result.get("truncated"),
                        "num_steps": result.get("num_steps"),
                        "metadata": result.get("metadata"),
                    },
                    "recording_errors": list(self.errors),
                }
            )
        except Exception as exc:  # noqa: BLE001
            self._remember_error("record_episode_result", exc)

    def _start_session(
        self,
        *,
        session_id: str,
        task: str,
        metadata: JsonDict | None,
        provenance: JsonDict | None,
        resumed: bool,
    ) -> None:
        with self._lock:
            self.session_id = session_id
            self.rollout_dir = self.root / "sessions" / session_id / "rollout"
            (self.rollout_dir / "artifacts").mkdir(parents=True, exist_ok=True)
            self._load_artifacts()
            for name in (
                "model_calls.jsonl",
                "tool_calls.jsonl",
                "transitions.jsonl",
                "episodes.jsonl",
                "artifacts.jsonl",
            ):
                self._next_sequences[name] = self._load_next_sequence(
                    self.rollout_dir / name
                )
            now = time.time()
            existing = self._read_manifest()
            manifest = {
                **existing,
                "schema_version": ROLLOUT_SCHEMA_VERSION,
                "session_id": session_id,
                "task": task,
                "status": "active",
                "created_at_s": existing.get("created_at_s", now),
                "updated_at_s": now,
                "resume_count": int(existing.get("resume_count") or 0) + int(resumed),
                "metadata": _sanitize(metadata or {}),
                "provenance": _sanitize(provenance or {}),
                "files": {
                    "model_calls": "model_calls.jsonl",
                    "tool_calls": "tool_calls.jsonl",
                    "transitions": "transitions.jsonl",
                    "episodes": "episodes.jsonl",
                    "artifacts": "artifacts.jsonl",
                    "artifact_root": "artifacts",
                },
            }
            self._write_manifest(manifest)

    def _observation_payload(self, observation: EnvObservation, *, role: str) -> JsonDict:
        cameras: list[JsonDict] = []
        for index, camera in enumerate(observation.cameras):
            camera_role = f"{role}/camera/{camera.frame_id or index}"
            rgb_ref = None
            if camera.rgb:
                array = np.asarray(camera.rgb, dtype=np.uint8)
                image = Image.fromarray(array)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                rgb_ref = self._record_bytes(
                    buffer.getvalue(),
                    suffix=".png",
                    mime_type="image/png",
                    role=f"{camera_role}/rgb",
                    metadata={
                        "encoding": "png",
                        "dtype": str(array.dtype),
                        "shape": list(array.shape),
                    },
                )
            depth_ref = None
            if camera.depth is not None:
                array = np.asarray(camera.depth)
                buffer = io.BytesIO()
                np.save(buffer, array, allow_pickle=False)
                depth_ref = self._record_bytes(
                    buffer.getvalue(),
                    suffix=".npy",
                    mime_type="application/x-npy",
                    role=f"{camera_role}/depth",
                    metadata={
                        "encoding": "npy",
                        "dtype": str(array.dtype),
                        "shape": list(array.shape),
                        "unit": _depth_unit(camera.intrinsics),
                        "scale": camera.intrinsics.get("scale"),
                    },
                )
            cameras.append(
                {
                    "frame_id": camera.frame_id,
                    "rgb": rgb_ref,
                    "depth": depth_ref,
                    "intrinsics": camera.intrinsics,
                    "extrinsics": camera.extrinsics,
                    "timestamp_s": camera.timestamp_s,
                }
            )
        return {
            "task": observation.task,
            "cameras": cameras,
            "robot": observation.robot.to_dict(),
            "objects": observation.objects,
            "metadata": observation.metadata,
        }

    def _append_prepared(
        self,
        filename: str,
        schema_version: str,
        payload: JsonDict,
    ) -> None:
        prepared, artifact_refs = self._prepare(payload)
        if artifact_refs:
            prepared["artifact_refs"] = artifact_refs
        with self._lock:
            if self.rollout_dir is None:
                return
            sequence = self._next_sequences.get(filename, 1)
            self._next_sequences[filename] = sequence + 1
            row = {
                "schema_version": schema_version,
                "seq": sequence,
                **prepared,
            }
            path = self.rollout_dir / filename
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def _prepare(self, value: Any) -> tuple[JsonDict, list[JsonDict]]:
        artifact_refs: list[JsonDict] = []

        def visit(item: Any, *, key: str = "", role: str = "payload") -> Any:
            if isinstance(item, str):
                match = _DATA_URL_RE.match(item)
                if match:
                    raw = base64.b64decode(match.group("data"), validate=True)
                    mime_type = match.group("mime").lower()
                    ref = self._record_bytes(
                        raw,
                        suffix=_mime_suffix(mime_type),
                        mime_type=mime_type,
                        role=role,
                        metadata={"source_encoding": "data_url"},
                    )
                    artifact_refs.append(ref)
                    return {
                        "artifact_id": ref["artifact_id"],
                        "bundle_path": ref["bundle_path"],
                        "mime_type": mime_type,
                        "source_encoding": "data_url",
                    }
                if _is_artifact_path_key(key):
                    path = Path(item)
                    if path.is_file():
                        ref = self._record_file(path, role=role)
                        artifact_refs.append(ref)
                return _redact_string(item)
            if isinstance(item, dict):
                result: JsonDict = {}
                for child_key, child in item.items():
                    normalized_key = str(child_key)
                    if _is_secret_key(normalized_key):
                        result[normalized_key] = "<redacted>"
                    else:
                        result[normalized_key] = visit(
                            child,
                            key=normalized_key,
                            role=f"{role}/{normalized_key}",
                        )
                return result
            if isinstance(item, (list, tuple)):
                return [
                    visit(child, key=key, role=f"{role}/{index}")
                    for index, child in enumerate(item)
                ]
            if isinstance(item, Path):
                return visit(str(item), key=key, role=role)
            if is_dataclass(item):
                return visit(asdict(item), key=key, role=role)
            if hasattr(item, "item") and callable(item.item):
                try:
                    return visit(item.item(), key=key, role=role)
                except (TypeError, ValueError):
                    pass
            if hasattr(item, "tolist") and callable(item.tolist):
                return visit(item.tolist(), key=key, role=role)
            if item is None or isinstance(item, (bool, int, float)):
                return item
            return f"<non_json:{type(item).__name__}>"

        prepared = visit(value)
        return (prepared if isinstance(prepared, dict) else {"value": prepared}, artifact_refs)

    def _record_file(self, path: Path, *, role: str) -> JsonDict:
        data = path.read_bytes()
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return self._record_bytes(
            data,
            suffix=path.suffix or ".bin",
            mime_type=mime_type,
            role=role,
            metadata={"source_path": str(path.resolve())},
        )

    def _record_bytes(
        self,
        data: bytes,
        *,
        suffix: str,
        mime_type: str,
        role: str,
        metadata: JsonDict | None = None,
    ) -> JsonDict:
        if self.rollout_dir is None:
            raise RuntimeError("rollout session has not started")
        digest = hashlib.sha256(data).hexdigest()
        artifact_id = f"sha256:{digest}"
        safe_suffix = suffix.lower() if re.fullmatch(r"\.[a-z0-9]{1,10}", suffix.lower()) else ".bin"
        relative = Path("artifacts") / digest[:2] / f"{digest}{safe_suffix}"
        destination = self.rollout_dir / relative
        with self._lock:
            if artifact_id not in self._artifacts:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    destination.write_bytes(data)
                entry = {
                    "artifact_id": artifact_id,
                    "sha256": digest,
                    "bundle_path": relative.as_posix(),
                    "byte_size": len(data),
                    "mime_type": mime_type,
                    "created_at_s": time.time(),
                    "metadata": _sanitize(metadata or {}),
                }
                self._artifacts[artifact_id] = entry
                self._append_artifact_entry(entry)
        return {
            "artifact_id": artifact_id,
            "bundle_path": relative.as_posix(),
            "role": role,
            "mime_type": mime_type,
            "byte_size": len(data),
        }

    def _append_artifact_entry(self, entry: JsonDict) -> None:
        if self.rollout_dir is None:
            return
        filename = "artifacts.jsonl"
        sequence = self._next_sequences.get(filename, 1)
        self._next_sequences[filename] = sequence + 1
        row = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "seq": sequence,
            **entry,
        }
        with (self.rollout_dir / filename).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _load_artifacts(self) -> None:
        self._artifacts.clear()
        if self.rollout_dir is None:
            return
        path = self.rollout_dir / "artifacts.jsonl"
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            artifact_id = entry.get("artifact_id")
            if isinstance(artifact_id, str):
                self._artifacts[artifact_id] = entry

    def _read_manifest(self) -> JsonDict:
        if self.rollout_dir is None:
            return {}
        path = self.rollout_dir / "manifest.json"
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _write_manifest(self, manifest: JsonDict) -> None:
        if self.rollout_dir is None:
            return
        path = self.rollout_dir / "manifest.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _update_manifest(self, changes: JsonDict) -> None:
        with self._lock:
            manifest = self._read_manifest()
            manifest.update(_sanitize(changes))
            self._write_manifest(manifest)

    @staticmethod
    def _load_next_sequence(path: Path) -> int:
        if not path.exists():
            return 1
        maximum = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
                maximum = max(maximum, int(value.get("seq") or 0))
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
        return maximum + 1

    def _remember_error(self, operation: str, exc: Exception) -> None:
        message = f"{operation}: {type(exc).__name__}: {exc}"
        with self._lock:
            self.errors = [*self.errors, message][-16:]


def build_rollout_provenance(
    *,
    planner: Any,
    tools: Any,
    skills: Any,
    metadata: JsonDict | None = None,
) -> JsonDict:
    """Build a reproducibility snapshot without retaining credentials."""

    repository_root = Path(__file__).resolve().parents[2]
    git = _git_provenance(repository_root)
    skill_rows = []
    for skill in skills.list():
        content = str(getattr(skill, "content", ""))
        skill_rows.append(
            {
                "name": getattr(skill, "name", ""),
                "description": getattr(skill, "description", ""),
                "version": getattr(skill, "version", ""),
                "source": getattr(skill, "source", ""),
                "editable": getattr(skill, "editable", False),
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "content": content,
                "metadata": getattr(skill, "metadata", {}),
            }
        )
    tool_rows = []
    executable = set(tools.handler_names())
    for spec in tools.list():
        tool_rows.append(
            {
                "name": spec.name,
                "description": spec.description,
                "category": spec.category,
                "parameters": spec.parameters,
                "effect": _enum_value(spec.effect),
                "batchable": spec.allows_batched_observation,
                "executable": spec.name in executable,
            }
        )
    return _sanitize(
        {
            "captured_at_s": time.time(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git": git,
            "planner": {
                "type": type(planner).__name__,
                "prompt": getattr(planner, "prompt_metadata", {}),
                "backend": getattr(getattr(planner, "backend", None), "descriptor", lambda: {})(),
            },
            "tools": tool_rows,
            "skills": skill_rows,
            "session_metadata": metadata or {},
        }
    )


def public_backend_details(details: JsonDict) -> JsonDict:
    """Remove recorder-only raw exchanges from model-visible trace metadata."""

    return {
        str(key): value
        for key, value in details.items()
        if not str(key).startswith("_rollout_")
    }


def validate_rollout_bundle(bundle_dir: str | Path) -> JsonDict:
    """Validate stream ordering and content-addressed artifacts before export."""

    bundle = Path(bundle_dir)
    errors: list[JsonDict] = []
    stream_rows: JsonDict = {}
    manifest_path = bundle / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema_version": ROLLOUT_VALIDATION_SCHEMA_VERSION,
            "valid": False,
            "bundle_dir": str(bundle),
            "errors": [
                {
                    "code": "manifest_unreadable",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            ],
        }
    if manifest.get("schema_version") != ROLLOUT_SCHEMA_VERSION:
        errors.append(
            {
                "code": "manifest_schema_mismatch",
                "observed": manifest.get("schema_version"),
                "expected": ROLLOUT_SCHEMA_VERSION,
            }
        )
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    for logical_name in ("model_calls", "tool_calls", "transitions", "episodes", "artifacts"):
        relative = files.get(logical_name)
        if not isinstance(relative, str) or not relative:
            errors.append({"code": "stream_missing_from_manifest", "stream": logical_name})
            continue
        path = bundle / relative
        rows, stream_errors = _validate_jsonl_stream(path)
        stream_rows[logical_name] = rows
        errors.extend(
            {"stream": logical_name, **stream_error}
            for stream_error in stream_errors
        )

    artifact_count = 0
    artifact_bytes = 0
    artifacts_path = bundle / str(files.get("artifacts") or "artifacts.jsonl")
    if artifacts_path.exists():
        for line_number, line in enumerate(
            artifacts_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            relative = entry.get("bundle_path")
            expected = entry.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected, str):
                errors.append(
                    {
                        "code": "artifact_metadata_invalid",
                        "line": line_number,
                    }
                )
                continue
            path = bundle / relative
            if not path.is_file():
                errors.append(
                    {
                        "code": "artifact_missing",
                        "line": line_number,
                        "bundle_path": relative,
                    }
                )
                continue
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            if observed != expected:
                errors.append(
                    {
                        "code": "artifact_hash_mismatch",
                        "line": line_number,
                        "bundle_path": relative,
                        "expected": expected,
                        "observed": observed,
                    }
                )
                continue
            artifact_count += 1
            artifact_bytes += path.stat().st_size
    return {
        "schema_version": ROLLOUT_VALIDATION_SCHEMA_VERSION,
        "valid": not errors,
        "bundle_dir": str(bundle.resolve()),
        "session_id": manifest.get("session_id"),
        "streams": stream_rows,
        "artifact_count": artifact_count,
        "artifact_bytes": artifact_bytes,
        "errors": errors,
    }


def _decision_payload(decision: Any) -> JsonDict:
    return {
        "kind": getattr(decision, "action_type", ""),
        "name": getattr(decision, "action", ""),
        "parameters": getattr(decision, "parameters", {}),
        "reasoning": getattr(decision, "reasoning", ""),
        "skill": getattr(decision, "skill", None),
        "code": getattr(decision, "code", None),
        "metadata": getattr(decision, "metadata", {}),
    }


def _validate_jsonl_stream(path: Path) -> tuple[int, list[JsonDict]]:
    if not path.exists():
        return 0, [{"code": "stream_file_missing", "path": str(path)}]
    errors: list[JsonDict] = []
    expected_sequence = 1
    rows = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(
                {
                    "code": "stream_json_invalid",
                    "line": line_number,
                    "message": str(exc),
                }
            )
            continue
        rows += 1
        observed_sequence = row.get("seq")
        if observed_sequence != expected_sequence:
            errors.append(
                {
                    "code": "stream_sequence_invalid",
                    "line": line_number,
                    "expected": expected_sequence,
                    "observed": observed_sequence,
                }
            )
        expected_sequence += 1
    return rows, errors


def _git_provenance(root: Path) -> JsonDict:
    def run(*args: str) -> bytes:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                timeout=5,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return b""

    commit = run("rev-parse", "HEAD").decode("utf-8", errors="replace").strip()
    status = run("status", "--porcelain=v1", "-z")
    diff = run("diff", "--binary", "--no-ext-diff", "HEAD")
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        result: JsonDict = {}
        for key, item in value.items():
            normalized = str(key)
            result[normalized] = (
                "<redacted>" if _is_secret_key(normalized) else _sanitize(item)
            )
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _sanitize(asdict(value))
    if hasattr(value, "item") and callable(value.item):
        try:
            return _sanitize(value.item())
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return f"<non_json:{type(value).__name__}>"


def _redact_string(value: str) -> str:
    value = _BEARER_RE.sub("Bearer <redacted>", value)
    return _OPENAI_KEY_RE.sub("<redacted-api-key>", value)


def _is_secret_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    return normalized in _SECRET_KEYS


def _is_artifact_path_key(key: str) -> bool:
    normalized = key.lower()
    if "apikey" in normalized or "api_key" in normalized:
        return False
    return normalized == "path" or any(hint in normalized for hint in _ARTIFACT_PATH_HINTS)


def _mime_suffix(mime_type: str) -> str:
    guessed = mimetypes.guess_extension(mime_type)
    return ".jpg" if guessed == ".jpe" else (guessed or ".bin")


def _depth_unit(intrinsics: JsonDict) -> str:
    unit = intrinsics.get("depth_unit") or intrinsics.get("unit")
    return str(unit) if unit else "metre"


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)
