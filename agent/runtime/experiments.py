"""Generation-owned workspaces and skill lineage for unattended experiments."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from adapter.protocol import JsonDict
from agent.runtime.artifact_paths import safe_artifact_component
from agent.runtime.parallel import ParallelEpisodeSpec
from agent.runtime.skills import BUILTIN_SKILL_DIR
from agent.runtime.task_playbooks import (
    DEFAULT_TASK_PLAYBOOK_ROOT,
    review_task_playbook_candidate,
    validate_task_playbook,
)
from agent.runtime.grasp_strategy_lifecycle import (
    GRASP_STRATEGY_EVIDENCE_SCHEMA_VERSION,
)
from agent.tools.grasp_strategies import (
    DEFAULT_GRASP_STRATEGY_ROOT,
    grasp_strategy_sha256,
    grasp_strategy_tree_sha256,
    validate_grasp_strategy,
)


DEFAULT_EXPERIMENT_ROOT = Path(".openeta_memory") / "experiments"
EXPERIMENT_SCHEMA_VERSION = "openeta.skill_experiment.v1"


@dataclass(frozen=True, slots=True)
class ExperimentWorkspace:
    """Stable filesystem ownership boundary for one iterative experiment."""

    experiment_id: str
    root: Path

    @classmethod
    def create(
        cls,
        experiment_id: str,
        *,
        root: str | Path = DEFAULT_EXPERIMENT_ROOT,
    ) -> "ExperimentWorkspace":
        safe_id = safe_artifact_component(experiment_id, fallback="experiment")
        workspace = cls(experiment_id=safe_id, root=Path(root) / safe_id)
        workspace.root.mkdir(parents=True, exist_ok=True)
        metadata_path = workspace.root / "experiment.json"
        if not metadata_path.exists():
            _atomic_write_json(
                metadata_path,
                {
                    "schema_version": EXPERIMENT_SCHEMA_VERSION,
                    "experiment_id": safe_id,
                    "created_at_s": time.time(),
                },
            )
        return workspace

    def generation_dir(self, index: int) -> Path:
        if index < 0:
            raise ValueError("generation index must be non-negative")
        return self.root / "generations" / f"{index:03d}"

    def initialize_generation(
        self,
        index: int,
        *,
        source_skills: str | Path = BUILTIN_SKILL_DIR,
        source_grasp_strategies: str | Path = DEFAULT_GRASP_STRATEGY_ROOT,
        source_task_playbooks: str | Path = DEFAULT_TASK_PLAYBOOK_ROOT,
    ) -> Path:
        generation = self.generation_dir(index)
        baseline = generation / "baseline_skills"
        baseline.mkdir(parents=True, exist_ok=True)
        if not any(baseline.glob("*.md")):
            _copy_skill_tree(Path(source_skills), baseline)
        strategy_baseline = generation / "baseline_grasp_strategies"
        if not any(strategy_baseline.rglob("*.json")):
            _copy_strategy_tree(Path(source_grasp_strategies), strategy_baseline)
        playbook_baseline = generation / "baseline_task_playbooks"
        if not any(playbook_baseline.rglob("*.json")):
            _copy_task_playbook_tree(Path(source_task_playbooks), playbook_baseline)
        _atomic_write_json(
            generation / "generation.json",
            {
                "schema_version": EXPERIMENT_SCHEMA_VERSION,
                "experiment_id": self.experiment_id,
                "generation": index,
                "baseline_skills": str(baseline),
                "baseline_hash": skill_tree_hash(baseline),
                "baseline_grasp_strategies": str(strategy_baseline),
                "baseline_grasp_strategy_hash": grasp_strategy_tree_sha256(
                    strategy_baseline
                ),
                "baseline_task_playbooks": str(playbook_baseline),
                "baseline_task_playbook_hash": task_playbook_tree_sha256(
                    playbook_baseline
                ),
            },
        )
        return baseline

    def prepare_specs(
        self,
        specs: Iterable[ParallelEpisodeSpec],
        *,
        generation: int,
        phase: str,
        skills_root: str | Path,
        grasp_strategies_root: str | Path | None = None,
        task_playbooks_root: str | Path | None = None,
        on_need_human: str = "fail",
    ) -> list[ParallelEpisodeSpec]:
        phase_name = safe_artifact_component(phase, fallback="run")
        sessions_root = self.generation_dir(generation) / phase_name / "sessions"
        sessions_root.mkdir(parents=True, exist_ok=True)
        prepared: list[ParallelEpisodeSpec] = []
        for spec in specs:
            metadata = {
                **spec.metadata,
                "experiment_id": self.experiment_id,
                "generation": generation,
                "experiment_phase": phase_name,
                "workspace_parent": str(sessions_root),
                "source_skills_root": str(skills_root),
                "source_grasp_strategies_root": str(
                    grasp_strategies_root
                    or self.generation_dir(generation)
                    / "baseline_grasp_strategies"
                ),
                "source_task_playbooks_root": str(
                    task_playbooks_root
                    or self.generation_dir(generation)
                    / "baseline_task_playbooks"
                ),
                "on_need_human": on_need_human,
            }
            prepared.append(replace(spec, metadata=metadata))
        return prepared

    def grasp_strategy_baseline(self, index: int) -> Path:
        return self.generation_dir(index) / "baseline_grasp_strategies"

    def task_playbook_baseline(self, index: int) -> Path:
        return self.generation_dir(index) / "baseline_task_playbooks"

    def write_phase_result(
        self,
        generation: int,
        phase: str,
        payload: JsonDict,
    ) -> Path:
        phase_name = safe_artifact_component(phase, fallback="run")
        path = self.generation_dir(generation) / phase_name / "result.json"
        _atomic_write_json(path, payload)
        return path


def skill_tree_hash(root: str | Path) -> str:
    """Hash skill names and bytes for reproducible generation lineage."""

    digest = hashlib.sha256()
    for path in sorted(Path(root).glob("*.md")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def task_playbook_tree_sha256(root: str | Path) -> str:
    """Hash validated task-playbook records for generation lineage."""

    digest = hashlib.sha256()
    for status in ("candidate", "validated"):
        for path in sorted((Path(root) / status).rglob("*.json")):
            playbook = validate_task_playbook(
                json.loads(path.read_text(encoding="utf-8"))
            )
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(
                json.dumps(
                    playbook,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(b"\0")
    return digest.hexdigest()


def objective_success_evidence(outcome: JsonDict) -> list[JsonDict]:
    """Return environment-owned success evidence, excluding agent declarations."""

    if outcome.get("status") != "success":
        return []
    episode = outcome.get("episode")
    if not isinstance(episode, dict):
        return []
    evidence: list[JsonDict] = []
    for index, step in enumerate(episode.get("steps") or []):
        if not isinstance(step, dict):
            continue
        result = step.get("step_result")
        if not isinstance(result, dict):
            continue
        reward = result.get("reward")
        if isinstance(reward, (int, float)) and not isinstance(reward, bool) and reward > 0:
            evidence.append({"kind": "positive_reward", "step": index, "value": reward})
        info = result.get("info")
        if not isinstance(info, dict):
            continue
        for key in (
            "task_success",
            "environment_success",
            "checker_success",
            "benchmark_success",
        ):
            if info.get(key) is True:
                evidence.append({"kind": key, "step": index, "value": True})
    return evidence


def objective_batch_metrics(payload: JsonDict) -> JsonDict:
    outcomes = [item for item in payload.get("outcomes", []) if isinstance(item, dict)]
    objective_ids = [
        str(item.get("episode_id") or "") for item in outcomes if objective_success_evidence(item)
    ]
    count = len(outcomes)
    return {
        "episode_count": count,
        "objective_success_count": len(objective_ids),
        "objective_success_rate": round(len(objective_ids) / count, 6) if count else 0.0,
        "objective_success_episode_ids": objective_ids,
        "runtime_success_count": int(payload.get("success_count") or 0),
        "fail_count": int(payload.get("fail_count") or 0),
        "need_human_count": int(payload.get("need_human_count") or 0),
    }


def collect_skill_candidates(
    *,
    generation_dir: str | Path,
    phase: str,
    batch_payload: JsonDict,
) -> JsonDict:
    """Collect changed session skills supported by objective task success."""

    generation = Path(generation_dir)
    baseline = generation / "baseline_skills"
    sessions = generation / safe_artifact_component(phase, fallback="train") / "sessions"
    candidate_root = generation / "candidates"
    if candidate_root.exists():
        shutil.rmtree(candidate_root)
    candidate_root.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], JsonDict] = {}
    for outcome in batch_payload.get("outcomes", []) or []:
        if not isinstance(outcome, dict):
            continue
        evidence = objective_success_evidence(outcome)
        if not evidence:
            continue
        session_id = safe_artifact_component(
            str(outcome.get("session_id") or ""), fallback="session"
        )
        episode_id = str(outcome.get("episode_id") or "")
        skill_dir = sessions / session_id / "skills"
        for skill_path in sorted(skill_dir.glob("*.md")):
            baseline_path = baseline / skill_path.name
            content = skill_path.read_bytes()
            if baseline_path.exists() and baseline_path.read_bytes() == content:
                continue
            digest = hashlib.sha256(content).hexdigest()
            key = (skill_path.stem, digest)
            entry = grouped.setdefault(
                key,
                {
                    "skill_name": skill_path.stem,
                    "sha256": digest,
                    "source_path": str(skill_path),
                    "supporting_episode_ids": [],
                    "evidence": [],
                },
            )
            entry["supporting_episode_ids"].append(episode_id)
            entry["evidence"].append({"episode_id": episode_id, "items": evidence})
    candidates: list[JsonDict] = []
    for (_, digest), entry in sorted(grouped.items()):
        destination = candidate_root / entry["skill_name"] / f"{digest}.md"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry["source_path"], destination)
        entry["candidate_path"] = str(destination)
        entry["support_count"] = len(entry["supporting_episode_ids"])
        candidates.append(entry)
    payload = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "baseline_hash": skill_tree_hash(baseline),
        "objective_metrics": objective_batch_metrics(batch_payload),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    _atomic_write_json(candidate_root / "manifest.json", payload)
    return payload


def collect_grasp_strategy_candidates(
    *,
    generation_dir: str | Path,
    phase: str,
    batch_payload: JsonDict,
) -> JsonDict:
    """Collect reviewed session proposals backed by objective reward."""

    generation = Path(generation_dir)
    sessions = generation / safe_artifact_component(phase, fallback="train") / "sessions"
    candidate_root = generation / "grasp_strategy_candidates"
    if candidate_root.exists():
        shutil.rmtree(candidate_root)
    candidate_root.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], JsonDict] = {}
    for outcome in batch_payload.get("outcomes", []) or []:
        if not isinstance(outcome, dict):
            continue
        evidence = objective_success_evidence(outcome)
        if not evidence:
            continue
        session_id = safe_artifact_component(
            str(outcome.get("session_id") or ""),
            fallback="session",
        )
        episode_id = str(outcome.get("episode_id") or "")
        staged = (
            sessions
            / session_id
            / "memory"
            / "grasp_strategy_lifecycle"
            / "staged"
            / "candidate"
        )
        for path in sorted(staged.glob("*.json")):
            strategy = validate_grasp_strategy(
                json.loads(path.read_text(encoding="utf-8"))
            )
            digest = grasp_strategy_sha256(strategy)
            key = (str(strategy["strategy_id"]), digest)
            entry = grouped.setdefault(
                key,
                {
                    "strategy_id": strategy["strategy_id"],
                    "strategy_sha256": digest,
                    "source_path": str(path),
                    "supporting_episode_ids": [],
                    "evidence": [],
                },
            )
            entry["supporting_episode_ids"].append(episode_id)
            entry["evidence"].append({"episode_id": episode_id, "items": evidence})
    candidates: list[JsonDict] = []
    for (_, digest), entry in sorted(grouped.items()):
        destination = (
            candidate_root / str(entry["strategy_id"]) / f"{digest}.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry["source_path"], destination)
        entry["candidate_path"] = str(destination)
        entry["support_count"] = len(entry["supporting_episode_ids"])
        candidates.append(entry)
    payload = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "baseline_grasp_strategy_hash": grasp_strategy_tree_sha256(
            generation / "baseline_grasp_strategies"
        ),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    _atomic_write_json(candidate_root / "manifest.json", payload)
    return payload


def collect_task_playbook_candidates(
    *,
    generation_dir: str | Path,
    phase: str,
    batch_payload: JsonDict,
) -> JsonDict:
    """Collect exact-task playbooks produced by objectively successful sessions."""

    generation = Path(generation_dir)
    sessions = generation / safe_artifact_component(phase, fallback="train") / "sessions"
    candidate_root = generation / "task_playbook_candidates"
    if candidate_root.exists():
        shutil.rmtree(candidate_root)
    candidate_root.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], JsonDict] = {}
    for outcome in batch_payload.get("outcomes", []) or []:
        if not isinstance(outcome, dict):
            continue
        evidence = objective_success_evidence(outcome)
        if not evidence:
            continue
        session_id = safe_artifact_component(
            str(outcome.get("session_id") or ""), fallback="session"
        )
        episode_id = str(outcome.get("episode_id") or "")
        staged = (
            sessions
            / session_id
            / "memory"
            / "task_playbook_reviews"
            / "candidate"
        )
        for path in sorted(staged.rglob("*.json")):
            playbook = validate_task_playbook(
                json.loads(path.read_text(encoding="utf-8"))
            )
            review = review_task_playbook_candidate(playbook, outcome=outcome)
            if review.get("approved") is not True:
                continue
            content = json.dumps(
                playbook,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest = hashlib.sha256(content).hexdigest()
            key = (str(playbook["playbook_id"]), digest)
            entry = grouped.setdefault(
                key,
                {
                    "playbook_id": playbook["playbook_id"],
                    "task_playbook_sha256": digest,
                    "source_path": str(path),
                    "supporting_episode_ids": [],
                    "evidence": [],
                },
            )
            entry["supporting_episode_ids"].append(episode_id)
            entry["evidence"].append({"episode_id": episode_id, "items": evidence})
    candidates: list[JsonDict] = []
    for (_, digest), entry in sorted(grouped.items()):
        destination = (
            candidate_root / str(entry["playbook_id"]) / f"{digest}.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry["source_path"], destination)
        entry["candidate_path"] = str(destination)
        entry["support_count"] = len(entry["supporting_episode_ids"])
        candidates.append(entry)
    payload = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "baseline_task_playbook_hash": task_playbook_tree_sha256(
            generation / "baseline_task_playbooks"
        ),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    _atomic_write_json(candidate_root / "manifest.json", payload)
    return payload


def build_proposed_task_playbook_tree(
    *,
    baseline_task_playbooks: str | Path,
    destination: str | Path,
    candidates: Iterable[JsonDict],
) -> Path:
    """Overlay reviewed candidate playbooks without promoting their status."""

    target = Path(destination)
    if target.exists():
        shutil.rmtree(target)
    _copy_task_playbook_tree(Path(baseline_task_playbooks), target)
    for item in candidates:
        source = Path(str(item.get("candidate_path") or ""))
        if not source.is_file():
            raise ValueError("task playbook candidate file is unavailable")
        playbook = validate_task_playbook(
            json.loads(source.read_text(encoding="utf-8"))
        )
        if playbook["status"] != "candidate":
            raise ValueError("generated task playbooks must remain candidates")
        existing_path, existing = _find_task_playbook_for_scope(
            target,
            scope=playbook["scope"],
        )
        if existing is not None and existing.get("status") == "validated":
            continue
        if existing is not None and existing_path is not None:
            playbook = _merge_task_playbooks(existing, playbook)
            output = existing_path
        else:
            output = target / "candidate" / f"{playbook['playbook_id']}.json"
        _atomic_write_json(output, playbook)
    return target


def select_supported_task_playbook_candidates(
    manifest: JsonDict,
) -> list[JsonDict]:
    """Select a uniquely best-supported variant for each exact task."""

    grouped: dict[str, list[JsonDict]] = {}
    for item in manifest.get("candidates", []) or []:
        if isinstance(item, dict):
            grouped.setdefault(str(item.get("playbook_id") or ""), []).append(item)
    selected: list[JsonDict] = []
    for playbook_id in sorted(grouped):
        variants = grouped[playbook_id]
        max_support = max(int(item.get("support_count") or 0) for item in variants)
        leaders = [
            item
            for item in variants
            if int(item.get("support_count") or 0) == max_support
        ]
        if len(leaders) == 1:
            selected.append(leaders[0])
    return selected


def select_supported_grasp_strategy_candidate(
    manifest: JsonDict,
) -> JsonDict | None:
    """Select one unique best-supported strategy variant."""

    candidates = [
        item for item in manifest.get("candidates", []) if isinstance(item, dict)
    ]
    if not candidates:
        return None
    max_support = max(int(item.get("support_count") or 0) for item in candidates)
    leaders = [
        item
        for item in candidates
        if int(item.get("support_count") or 0) == max_support
    ]
    return leaders[0] if len(leaders) == 1 else None


def build_proposed_grasp_strategy_tree(
    *,
    baseline_strategies: str | Path,
    destination: str | Path,
    candidate_path: str | Path,
) -> Path:
    target = Path(destination)
    if target.exists():
        shutil.rmtree(target)
    _copy_strategy_tree(Path(baseline_strategies), target)
    candidate = validate_grasp_strategy(
        json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    )
    output = target / "candidate" / f"{candidate['strategy_id']}.json"
    _atomic_write_json(output, candidate)
    return target


def write_grasp_strategy_evidence(
    path: str | Path,
    *,
    split: str,
    strategy_sha256: str,
    calibration_profile_sha256: str,
    baseline: JsonDict,
    candidate: JsonDict,
    expected_strategy_tree_sha256: str = "",
) -> Path:
    """Write host-derived paired objective evidence for strategy promotion."""

    if split not in {"canary", "held_out"}:
        raise ValueError("strategy evidence split must be canary or held_out")
    filtered_baseline, baseline_excluded = _without_infrastructure_failures(
        baseline
    )
    filtered_candidate, candidate_excluded = _without_infrastructure_failures(
        candidate
    )
    baseline_metrics = objective_batch_metrics(filtered_baseline)
    candidate_metrics = objective_batch_metrics(filtered_candidate)
    baseline_ids = set(baseline_metrics["objective_success_episode_ids"])
    candidate_ids = set(candidate_metrics["objective_success_episode_ids"])
    outcomes = [
        item
        for item in filtered_candidate.get("outcomes", [])
        if isinstance(item, dict)
    ]
    payload = {
        "schema_version": GRASP_STRATEGY_EVIDENCE_SCHEMA_VERSION,
        "producer": "openeta_experiment_host",
        "strategy_sha256": strategy_sha256,
        "calibration_profile_sha256": calibration_profile_sha256,
        "split": split,
        "attempts": len(outcomes),
        "successes": candidate_metrics["objective_success_count"],
        "baseline_attempts": len(
            [
                item
                for item in filtered_baseline.get("outcomes", [])
                if isinstance(item, dict)
            ]
        ),
        "baseline_successes": baseline_metrics["objective_success_count"],
        "task_ids": sorted(
            {
                _outcome_task_id(item)
                for item in outcomes
                if _outcome_task_id(item)
            }
        ),
        "seeds": sorted(
            {
                int(item.get("seed"))
                for item in outcomes
                if isinstance(item.get("seed"), int)
                and not isinstance(item.get("seed"), bool)
                and int(item.get("seed")) >= 0
            }
        ),
        "regressed_episode_ids": sorted(baseline_ids - candidate_ids),
        "safety_violations": sum(
            _outcome_violation_count(item, kind="safety") for item in outcomes
        ),
        "contract_violations": sum(
            _outcome_violation_count(item, kind="contract") for item in outcomes
        )
        + sum(
            int(
                not _outcome_has_strategy_provenance(
                    item,
                    strategy_tree_sha256=expected_strategy_tree_sha256,
                    calibration_sha256=calibration_profile_sha256,
                )
            )
            for item in outcomes
        ),
        "human_interventions": sum(
            int(
                (
                    item.get("assistance")
                    if isinstance(item.get("assistance"), dict)
                    else {}
                ).get("human_intervention_count")
                or 0
            )
            for item in outcomes
        ),
        "excluded_infrastructure": {
            "baseline": baseline_excluded,
            "candidate": candidate_excluded,
        },
    }
    target = Path(path)
    _atomic_write_json(target, payload)
    return target


def compact_strategy_rollout_summary(payload: JsonDict) -> JsonDict:
    """Bound training results for the isolated strategy author."""

    rows: list[JsonDict] = []
    for outcome in payload.get("outcomes", []) or []:
        if not isinstance(outcome, dict):
            continue
        episode = outcome.get("episode")
        episode = episode if isinstance(episode, dict) else {}
        metadata = episode.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        rows.append(
            {
                "episode_id": outcome.get("episode_id"),
                "env_id": outcome.get("env_id"),
                "seed": outcome.get("seed"),
                "status": outcome.get("status"),
                "objective_success": bool(objective_success_evidence(outcome)),
                "failure_reason": metadata.get("failure_reason"),
                "assistance": outcome.get("assistance"),
                "grasp_events": _compact_grasp_events(episode),
            }
        )
    return {
        "objective_metrics": objective_batch_metrics(payload),
        "episodes": rows[:64],
        "truncated": len(rows) > 64,
    }


def select_supported_candidates(candidate_manifest: JsonDict) -> list[JsonDict]:
    """Choose a unique best-supported variant per skill before review.

    Conflicting variants with equal support fail closed instead of using a hash
    or filesystem order as a semantic tie-breaker.
    """

    grouped: dict[str, list[JsonDict]] = {}
    for candidate in candidate_manifest.get("candidates", []) or []:
        if not isinstance(candidate, dict):
            continue
        name = str(candidate.get("skill_name") or "")
        grouped.setdefault(name, []).append(candidate)
    selected: list[JsonDict] = []
    for name in sorted(grouped):
        variants = grouped[name]
        max_support = max(int(item.get("support_count") or 0) for item in variants)
        leaders = [item for item in variants if int(item.get("support_count") or 0) == max_support]
        if len(leaders) == 1:
            selected.append(leaders[0])
    return selected


def build_proposed_skill_tree(
    *,
    baseline_skills: str | Path,
    destination: str | Path,
    approved_candidates: Iterable[JsonDict],
) -> Path:
    target = Path(destination)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    _copy_skill_tree(Path(baseline_skills), target)
    for candidate in approved_candidates:
        name = str(candidate.get("skill_name") or "").strip()
        source = Path(str(candidate.get("candidate_path") or ""))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name) or not source.is_file():
            raise ValueError("approved candidate has no valid skill name or file")
        shutil.copy2(source, target / f"{name}.md")
    return target


def validation_has_no_regression(baseline: JsonDict, candidate: JsonDict) -> JsonDict:
    baseline_metrics = objective_batch_metrics(baseline)
    candidate_metrics = objective_batch_metrics(candidate)
    baseline_ids = set(baseline_metrics["objective_success_episode_ids"])
    candidate_ids = set(candidate_metrics["objective_success_episode_ids"])
    regressed_ids = sorted(baseline_ids - candidate_ids)
    passed = (
        not regressed_ids
        and candidate_metrics["objective_success_count"] > 0
        and candidate_metrics["objective_success_count"]
        >= baseline_metrics["objective_success_count"]
        and candidate_metrics["runtime_success_count"] >= baseline_metrics["runtime_success_count"]
        and candidate_metrics["fail_count"] <= baseline_metrics["fail_count"]
        and candidate_metrics["need_human_count"] <= baseline_metrics["need_human_count"]
    )
    return {
        "passed": passed,
        "candidate_has_objective_success": (
            candidate_metrics["objective_success_count"] > 0
        ),
        "regressed_episode_ids": regressed_ids,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
    }


def strategy_validation_has_no_regression(
    baseline: JsonDict,
    candidate: JsonDict,
) -> JsonDict:
    """Compare paired strategy runs without charging infrastructure failures."""

    filtered_baseline, baseline_excluded = _without_infrastructure_failures(
        baseline
    )
    filtered_candidate, candidate_excluded = _without_infrastructure_failures(
        candidate
    )
    comparison = validation_has_no_regression(
        filtered_baseline,
        filtered_candidate,
    )
    comparison["excluded_infrastructure"] = {
        "baseline": baseline_excluded,
        "candidate": candidate_excluded,
    }
    return comparison


def _copy_skill_tree(source: Path, destination: Path) -> None:
    files = sorted(source.glob("*.md"))
    if not files:
        raise ValueError(f"skill baseline contains no markdown files: {source}")
    for path in files:
        shutil.copy2(path, destination / path.name)


def _copy_strategy_tree(source: Path, destination: Path) -> None:
    files = [
        path
        for status in ("candidate", "validated")
        for path in sorted((source / status).glob("*.json"))
    ]
    if not files:
        raise ValueError(f"grasp strategy baseline contains no JSON files: {source}")
    for path in files:
        target = destination / path.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _copy_task_playbook_tree(source: Path, destination: Path) -> None:
    files = [
        path
        for status in ("candidate", "validated")
        for path in sorted((source / status).rglob("*.json"))
    ]
    destination.mkdir(parents=True, exist_ok=True)
    for status in ("candidate", "validated"):
        (destination / status).mkdir(parents=True, exist_ok=True)
    for path in files:
        playbook = validate_task_playbook(
            json.loads(path.read_text(encoding="utf-8"))
        )
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(target, playbook)


def _find_task_playbook_for_scope(
    root: Path,
    *,
    scope: JsonDict,
) -> tuple[Path | None, JsonDict | None]:
    for status in ("validated", "candidate"):
        for path in sorted((root / status).rglob("*.json")):
            playbook = validate_task_playbook(
                json.loads(path.read_text(encoding="utf-8"))
            )
            if playbook.get("scope") == scope:
                return path, playbook
    return None, None


def _merge_task_playbooks(existing: JsonDict, incoming: JsonDict) -> JsonDict:
    merged = dict(existing)
    merged["revision"] = max(
        int(existing.get("revision") or 1),
        int(incoming.get("revision") or 1),
    ) + 1
    merged["guidance"] = _merge_json_guidance(
        dict(existing.get("guidance") or {}),
        dict(incoming.get("guidance") or {}),
    )
    old_evidence = dict(existing.get("evidence") or {})
    new_evidence = dict(incoming.get("evidence") or {})
    for key, value in new_evidence.items():
        if isinstance(value, list):
            old_evidence[key] = _stable_unique_json(
                list(old_evidence.get(key) or []) + value
            )
        elif key not in old_evidence:
            old_evidence[key] = value
    merged["evidence"] = old_evidence
    return validate_task_playbook(merged)


def _merge_json_guidance(existing: JsonDict, incoming: JsonDict) -> JsonDict:
    merged = dict(existing)
    for key, value in incoming.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_json_guidance(current, value)
        elif isinstance(current, list) and isinstance(value, list):
            merged[key] = _stable_unique_json(current + value)
        elif key not in merged:
            merged[key] = value
    return merged


def _stable_unique_json(values: list[object]) -> list[object]:
    seen: set[str] = set()
    result: list[object] = []
    for value in values:
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def _outcome_task_id(outcome: JsonDict) -> str:
    episode = outcome.get("episode")
    if isinstance(episode, dict) and str(episode.get("task") or "").strip():
        return str(episode["task"]).strip()
    return str(outcome.get("env_id") or "").strip()


def _outcome_violation_count(outcome: JsonDict, *, kind: str) -> int:
    terms = {
        "safety": ("safety", "collision", "ik_rejected"),
        "contract": ("contract", "schema", "frame_mismatch"),
    }[kind]
    serialized = json.dumps(
        {
            "error": outcome.get("error"),
            "episode_metadata": (
                outcome.get("episode", {}).get("metadata")
                if isinstance(outcome.get("episode"), dict)
                else {}
            ),
        },
        ensure_ascii=True,
    ).lower()
    return int(any(term in serialized for term in terms))


def _without_infrastructure_failures(payload: JsonDict) -> tuple[JsonDict, list[str]]:
    outcomes: list[JsonDict] = []
    excluded: list[str] = []
    for item in payload.get("outcomes", []) or []:
        if not isinstance(item, dict):
            continue
        if _is_infrastructure_failure(item):
            excluded.append(str(item.get("episode_id") or ""))
            continue
        outcomes.append(item)
    return (
        {
            **payload,
            "outcomes": outcomes,
            "success_count": sum(item.get("status") == "success" for item in outcomes),
            "fail_count": sum(item.get("status") == "fail" for item in outcomes),
            "need_human_count": sum(
                item.get("status") == "need_human" for item in outcomes
            ),
        },
        excluded,
    )


def _is_infrastructure_failure(outcome: JsonDict) -> bool:
    error = outcome.get("error")
    if not isinstance(error, dict):
        return False
    value = (
        f"{str(error.get('code') or '').strip().lower()} "
        f"{str(error.get('type') or '').strip().lower()}"
    )
    return any(
        marker in value
        for marker in (
            "provider_timeout",
            "provider_queue_timeout",
            "provider_unavailable",
            "simulator_mcp",
            "environment_create",
            "model_load_failed",
            "out_of_memory",
            "cuda_oom",
            "infrastructure",
        )
    )


def _outcome_has_strategy_provenance(
    outcome: JsonDict,
    *,
    strategy_tree_sha256: str,
    calibration_sha256: str,
) -> bool:
    episode = outcome.get("episode")
    metadata = episode.get("metadata") if isinstance(episode, dict) else None
    if not isinstance(metadata, dict):
        return False
    return (
        (not strategy_tree_sha256)
        or str(metadata.get("grasp_strategy_tree_sha256") or "")
        == strategy_tree_sha256
    ) and str(metadata.get("calibration_profile_sha256") or "") == calibration_sha256


def _compact_grasp_events(episode: JsonDict) -> list[JsonDict]:
    events: list[JsonDict] = []
    relevant = {
        "grasp_pose_estimate",
        "compile_grasp_seed",
        "compute_wrist_alignment",
        "move_to",
        "gripper_control",
    }
    for step in episode.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        if not isinstance(action, dict):
            continue
        command = action.get("command")
        calls = (
            command.get("tool_calls", [])
            if isinstance(command, dict)
            else action.get("tool_calls", [])
        )
        for call in calls or []:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or "")
            if name not in relevant:
                continue
            result = call.get("result")
            result = result if isinstance(result, dict) else {}
            details = result.get("details")
            details = details if isinstance(details, dict) else {}
            outputs = details.get("outputs")
            if not isinstance(outputs, dict):
                outputs = result.get("outputs")
            outputs = outputs if isinstance(outputs, dict) else {}
            events.append(
                {
                    "tool": name,
                    "status": call.get("status"),
                    "success": result.get("success"),
                    "reason": outputs.get("reason"),
                    "candidate_id": outputs.get("candidate_id"),
                    "strategy_id": outputs.get("strategy_id"),
                    "strategy_selection": outputs.get("strategy_selection"),
                    "gripper_width_m": outputs.get("gripper_width_m"),
                    "orientation_clamped": outputs.get("orientation_clamped"),
                    "reached_target": outputs.get("reached_target"),
                }
            )
    return events[-24:]


def _atomic_write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
