"""Exact-task experience records backed by objective rollout evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from adapter.protocol import JsonDict


TASK_PLAYBOOK_SCHEMA_VERSION = "openeta.task_playbook.v1"
DEFAULT_TASK_PLAYBOOK_ROOT = Path(__file__).resolve().parents[1] / "task_playbooks"
_FORBIDDEN_GUIDANCE_KEYS = {
    "absolute_position",
    "candidate_rank",
    "move_to_parameters",
    "target_pose",
    "world_xyz",
}


class TaskPlaybookError(ValueError):
    """Raised when exact-task experience is malformed or unsupported."""


def normalize_task_text(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return " ".join(text.split())


def task_text_sha256(value: str) -> str:
    return hashlib.sha256(normalize_task_text(value).encode("utf-8")).hexdigest()


def validate_task_playbook(value: Mapping[str, Any]) -> JsonDict:
    playbook = json.loads(json.dumps(dict(value)))
    if playbook.get("schema_version") != TASK_PLAYBOOK_SCHEMA_VERSION:
        raise TaskPlaybookError("unsupported task playbook schema")
    allowed = {
        "schema_version",
        "status",
        "playbook_id",
        "revision",
        "scope",
        "compatibility",
        "guidance",
        "evidence",
        "lifecycle",
        "_source_path",
    }
    unknown = set(playbook) - allowed
    if unknown:
        raise TaskPlaybookError(
            "task playbook has forbidden fields: " + ", ".join(sorted(unknown))
        )
    if playbook.get("status") not in {"candidate", "validated"}:
        raise TaskPlaybookError("task playbook status must be candidate or validated")
    playbook_id = str(playbook.get("playbook_id") or "").strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,127}", playbook_id):
        raise TaskPlaybookError("playbook_id is invalid")
    revision = playbook.get("revision", 1)
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise TaskPlaybookError("revision must be a positive integer")
    playbook["revision"] = revision
    scope = _mapping(playbook.get("scope"), "scope")
    environment_id = str(scope.get("environment_id") or "").strip()
    suite = str(scope.get("suite") or "").strip()
    task_hash = str(scope.get("task_text_sha256") or "").strip().lower()
    task_index = scope.get("task_index")
    if not environment_id or not suite:
        raise TaskPlaybookError("scope environment_id and suite are required")
    if isinstance(task_index, bool) or not isinstance(task_index, int) or task_index < 0:
        raise TaskPlaybookError("scope.task_index must be non-negative")
    if not re.fullmatch(r"[0-9a-f]{64}", task_hash):
        raise TaskPlaybookError("scope.task_text_sha256 must be a SHA-256 digest")
    compatibility = _mapping(playbook.get("compatibility", {}), "compatibility")
    calibration_ids = compatibility.get("calibration_ids", [])
    if not isinstance(calibration_ids, list) or not all(
        isinstance(item, str) and item.strip() for item in calibration_ids
    ):
        raise TaskPlaybookError("compatibility.calibration_ids must be strings")
    guidance = _mapping(playbook.get("guidance"), "guidance")
    _reject_executable_guidance(guidance)
    evidence = _mapping(playbook.get("evidence"), "evidence")
    sessions = evidence.get("source_session_ids")
    rewards = evidence.get("official_rewards")
    if not isinstance(sessions, list) or not sessions or not all(
        isinstance(item, str) and item.strip() for item in sessions
    ):
        raise TaskPlaybookError("evidence.source_session_ids must be non-empty strings")
    if not isinstance(rewards, list) or not rewards or not all(
        isinstance(item, int | float)
        and not isinstance(item, bool)
        and math.isfinite(float(item))
        and float(item) > 0
        for item in rewards
    ):
        raise TaskPlaybookError("evidence.official_rewards must be positive finite numbers")
    return playbook


def load_task_playbooks(
    root: str | Path = DEFAULT_TASK_PLAYBOOK_ROOT,
) -> list[JsonDict]:
    playbooks: list[JsonDict] = []
    seen: set[str] = set()
    for status in ("validated", "candidate"):
        for path in sorted((Path(root) / status).rglob("*.json")):
            playbook = validate_task_playbook(json.loads(path.read_text(encoding="utf-8")))
            playbook_id = str(playbook["playbook_id"])
            if playbook_id in seen:
                continue
            seen.add(playbook_id)
            playbook["_source_path"] = str(path.resolve())
            playbooks.append(playbook)
    return playbooks


def select_task_playbook(
    playbooks: list[Mapping[str, Any]],
    *,
    environment_id: str,
    suite: str,
    task_index: int,
    task: str,
    calibration_id: str = "",
) -> JsonDict | None:
    task_hash = task_text_sha256(task)
    matches: list[JsonDict] = []
    for raw in playbooks:
        playbook = validate_task_playbook(raw)
        scope = _mapping(playbook["scope"], "scope")
        if (
            scope.get("environment_id") != environment_id
            or scope.get("suite") != suite
            or scope.get("task_index") != task_index
            or scope.get("task_text_sha256") != task_hash
        ):
            continue
        compatibility = _mapping(playbook.get("compatibility", {}), "compatibility")
        calibration_ids = compatibility.get("calibration_ids", [])
        if calibration_ids and calibration_id not in calibration_ids:
            continue
        matches.append(playbook)
    if not matches:
        return None
    matches.sort(
        key=lambda item: (
            0 if item.get("status") == "validated" else 1,
            -int(item.get("revision") or 1),
            str(item.get("playbook_id") or ""),
        )
    )
    selected = matches[0]
    return {
        "schema_version": selected["schema_version"],
        "status": selected["status"],
        "playbook_id": selected["playbook_id"],
        "revision": selected["revision"],
        "guidance": selected["guidance"],
        "evidence_summary": {
            "support_count": len(selected["evidence"]["source_session_ids"]),
            "official_reward_count": len(selected["evidence"]["official_rewards"]),
        },
        "usage_contract": (
            "Treat this as a localization and strategy prior. Re-observe and verify the "
            "current scene; never replay stored coordinates or bypass safety, attachment, "
            "placement, or official-reward checks."
        ),
    }


def extract_task_playbook_candidate(
    *,
    outcome: Mapping[str, Any],
    rollout_tool_calls: str | Path,
) -> JsonDict:
    episode = _mapping(outcome.get("episode"), "outcome.episode")
    task = str(episode.get("task") or "").strip()
    session_id = str(outcome.get("session_id") or episode.get("session_id") or "").strip()
    episode_id = str(outcome.get("episode_id") or "").strip()
    rewards = _positive_rewards(episode)
    if outcome.get("status") != "success" or not rewards:
        raise TaskPlaybookError("candidate extraction requires objective episode success")
    metadata = _episode_scope_metadata(episode)
    environment_id = str(outcome.get("env_id") or metadata.get("env_id") or "").strip()
    suite = str(metadata.get("suite") or "").strip()
    task_index = metadata.get("task_index")
    if isinstance(task_index, bool) or not isinstance(task_index, int):
        raise TaskPlaybookError("successful episode is missing task_index metadata")
    queries: list[str] = []
    grasp_signatures: list[JsonDict] = []
    stage_sequence: list[str] = []
    for record in _jsonl_records(Path(rollout_tool_calls)):
        event = record.get("event")
        if not isinstance(event, dict):
            continue
        name = str(event.get("name") or "")
        phase = str(event.get("phase") or "")
        parameters = event.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        if phase == "start" and name == "retrieve_asset_reference":
            _append_unique(queries, str(parameters.get("target_object") or ""))
        if phase == "start" and name == "sam3":
            _append_unique(queries, str(parameters.get("prompt") or ""))
        if phase == "start" and name == "compile_grasp_seed":
            camera_pose = parameters.get("camera_pose")
            camera_pose = camera_pose if isinstance(camera_pose, dict) else {}
            signature: JsonDict = {
                "geometry_family": parameters.get("target_geometry_family")
                or parameters.get("target_class"),
                "strategy_id": parameters.get("strategy_id"),
                "gripper_width_m": camera_pose.get("width"),
                "backend": camera_pose.get("source_backend"),
            }
            grasp_signatures.append(
                {key: value for key, value in signature.items() if value not in (None, "")}
            )
        if phase == "start" and name == "move_to":
            pose = parameters.get("target_pose")
            pose = pose if isinstance(pose, dict) else {}
            stage = pose.get("grasp_stage") or pose.get("placement_stage") or pose.get("probe_type")
            if isinstance(stage, str) and stage:
                stage_sequence.append(stage)
        if phase == "start" and name == "gripper_control":
            position = parameters.get("position")
            if position in {0, 1}:
                stage_sequence.append("gripper_close" if position == 0 else "gripper_open")
    scope_slug = re.sub(
        r"[^a-z0-9]+",
        "-",
        f"{environment_id}-{suite}-task-{task_index}".lower(),
    ).strip("-")
    candidate = {
        "schema_version": TASK_PLAYBOOK_SCHEMA_VERSION,
        "status": "candidate",
        "playbook_id": f"auto-{scope_slug}-{task_text_sha256(task)[:12]}",
        "revision": 1,
        "scope": {
            "environment_id": environment_id,
            "suite": suite,
            "task_index": task_index,
            "task_text_sha256": task_text_sha256(task),
        },
        "compatibility": {
            "calibration_ids": [str(metadata.get("calibration_profile_id"))]
            if metadata.get("calibration_profile_id")
            else []
        },
        "guidance": {
            "task_summary": task,
            "observed_object_queries": queries,
            "successful_grasp_signatures": grasp_signatures,
            "successful_stage_sequence": stage_sequence,
            "rules": [
                "Use object identity and geometry hints only after fresh visual verification.",
                "Treat estimator poses as seeds and keep attachment and reward gates active.",
            ],
        },
        "evidence": {
            "source_session_ids": [session_id],
            "supporting_episode_ids": [episode_id],
            "official_rewards": rewards,
            "rollout_tool_calls": [str(Path(rollout_tool_calls))],
            "autonomous": not _outcome_assisted(outcome),
        },
    }
    reviewed = review_task_playbook_candidate(candidate, outcome=outcome)
    if not reviewed["approved"]:
        raise TaskPlaybookError(str(reviewed["reason"]))
    return validate_task_playbook(candidate)


def review_task_playbook_candidate(
    playbook: Mapping[str, Any], *, outcome: Mapping[str, Any]
) -> JsonDict:
    try:
        validated = validate_task_playbook(playbook)
    except (TaskPlaybookError, TypeError, ValueError) as exc:
        return {"approved": False, "reason": str(exc), "reviewer": "objective_evidence"}
    episode = outcome.get("episode")
    if not isinstance(episode, dict) or not _positive_rewards(episode):
        return {
            "approved": False,
            "reason": "no same-episode objective reward evidence",
            "reviewer": "objective_evidence",
        }
    session_id = str(outcome.get("session_id") or episode.get("session_id") or "")
    if session_id not in validated["evidence"]["source_session_ids"]:
        return {
            "approved": False,
            "reason": "source session does not match successful outcome",
            "reviewer": "objective_evidence",
        }
    metadata = _episode_scope_metadata(episode)
    scope = validated["scope"]
    expected_scope = {
        "environment_id": str(outcome.get("env_id") or metadata.get("env_id") or ""),
        "suite": str(metadata.get("suite") or ""),
        "task_index": metadata.get("task_index"),
        "task_text_sha256": task_text_sha256(str(episode.get("task") or "")),
    }
    if any(scope.get(key) != value for key, value in expected_scope.items()):
        return {
            "approved": False,
            "reason": "task scope does not match the successful outcome",
            "reviewer": "objective_evidence",
        }
    return {
        "approved": True,
        "reason": "schema, non-executable guidance, and objective reward evidence verified",
        "reviewer": "objective_evidence",
    }


def write_task_playbook(playbook: Mapping[str, Any], *, root: str | Path) -> Path:
    validated = validate_task_playbook(playbook)
    destination = Path(root) / str(validated["status"]) / f"{validated['playbook_id']}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in validated.items() if not key.startswith("_")}
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _reject_executable_guidance(value: Any, *, path: str = "guidance") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in _FORBIDDEN_GUIDANCE_KEYS:
                raise TaskPlaybookError(f"{path}.{key} is forbidden")
            _reject_executable_guidance(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_executable_guidance(child, path=f"{path}[{index}]")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TaskPlaybookError(f"{label} must be an object")
    return value


def _jsonl_records(path: Path) -> list[JsonDict]:
    records: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _positive_rewards(episode: Mapping[str, Any]) -> list[float]:
    rewards: list[float] = []
    for step in episode.get("steps") or []:
        if not isinstance(step, dict):
            continue
        result = step.get("step_result")
        reward = result.get("reward") if isinstance(result, dict) else None
        if (
            isinstance(reward, int | float)
            and not isinstance(reward, bool)
            and math.isfinite(float(reward))
            and float(reward) > 0
        ):
            rewards.append(float(reward))
    return rewards


def _episode_scope_metadata(episode: Mapping[str, Any]) -> JsonDict:
    metadata = episode.get("metadata")
    merged = dict(metadata) if isinstance(metadata, dict) else {}
    for step in episode.get("steps") or []:
        if not isinstance(step, dict):
            continue
        observation = step.get("observation")
        observation_metadata = observation.get("metadata") if isinstance(observation, dict) else None
        if isinstance(observation_metadata, dict):
            merged = {**merged, **observation_metadata}
            break
    return merged


def _outcome_assisted(outcome: Mapping[str, Any]) -> bool:
    assistance = outcome.get("assistance")
    return isinstance(assistance, dict) and bool(assistance.get("assisted"))


def _append_unique(items: list[str], value: str) -> None:
    normalized = value.strip()
    if normalized and normalized not in items:
        items.append(normalized)
