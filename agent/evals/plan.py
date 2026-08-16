"""Declarative, experiment-agnostic evaluation plans and compiled jobs."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

from adapter.protocol import JsonDict
from agent.runtime.parallel import (
    DEFAULT_PARALLEL_EPISODES,
    MAX_PARALLEL_EPISODES,
    ParallelEpisodeSpec,
)


EVALUATION_PLAN_SCHEMA_VERSION = "openeta.evaluation_plan.v1"
COMPILED_EVALUATION_SCHEMA_VERSION = "openeta.compiled_evaluation.v1"


@dataclass(frozen=True, slots=True)
class EvaluationVariant:
    """One host-owned runtime configuration in an evaluation matrix."""

    variant_id: str
    runtime: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: JsonDict, *, index: int) -> "EvaluationVariant":
        variant_id = _required_identifier(value, "variant_id", location=f"variants[{index}]")
        runtime = value.get("runtime")
        metadata = value.get("metadata")
        if runtime is not None and not isinstance(runtime, dict):
            raise ValueError(f"variants[{index}].runtime must be an object")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"variants[{index}].metadata must be an object")
        return cls(
            variant_id=variant_id,
            runtime=dict(runtime or {}),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> JsonDict:
        return {
            "variant_id": self.variant_id,
            "runtime": self.runtime,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class EvaluationExecution:
    """Generic scheduling policy; it contains no experiment-specific metrics."""

    concurrency: int = DEFAULT_PARALLEL_EPISODES
    provider_concurrency: int = 2
    provider_queue_timeout_s: float = 180.0
    supervision_profile: str = "standard"
    max_attempts: int = 2

    @classmethod
    def from_dict(cls, value: object) -> "EvaluationExecution":
        payload = dict(value) if isinstance(value, dict) else {}
        execution = cls(
            concurrency=int(payload.get("concurrency", DEFAULT_PARALLEL_EPISODES)),
            provider_concurrency=int(payload.get("provider_concurrency", 2)),
            provider_queue_timeout_s=float(
                payload.get("provider_queue_timeout_s", 180.0)
            ),
            supervision_profile=str(
                payload.get("supervision_profile") or "standard"
            ).strip(),
            max_attempts=int(payload.get("max_attempts", 2)),
        )
        if not 1 <= execution.concurrency <= MAX_PARALLEL_EPISODES:
            raise ValueError(
                f"execution.concurrency must be between 1 and {MAX_PARALLEL_EPISODES}"
            )
        if execution.provider_concurrency < 1:
            raise ValueError("execution.provider_concurrency must be positive")
        if execution.provider_queue_timeout_s <= 0:
            raise ValueError("execution.provider_queue_timeout_s must be positive")
        if execution.max_attempts < 1:
            raise ValueError("execution.max_attempts must be positive")
        if execution.supervision_profile not in {
            "human_gated",
            "standard",
            "reviewed_autonomy",
        }:
            raise ValueError("execution.supervision_profile is unsupported")
        return execution

    def to_dict(self) -> JsonDict:
        return {
            "concurrency": self.concurrency,
            "provider_concurrency": self.provider_concurrency,
            "provider_queue_timeout_s": self.provider_queue_timeout_s,
            "supervision_profile": self.supervision_profile,
            "max_attempts": self.max_attempts,
        }


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    """Validated source plan before episode/variant expansion."""

    plan_id: str
    description: str
    episodes: tuple[ParallelEpisodeSpec, ...]
    variants: tuple[EvaluationVariant, ...]
    repeats: int = 1
    shuffle_seed: int = 0
    execution: EvaluationExecution = field(default_factory=EvaluationExecution)
    metadata: JsonDict = field(default_factory=dict)
    source_path: str = ""
    source_payload: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": EVALUATION_PLAN_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "description": self.description,
            "episodes": [_episode_spec_to_dict(spec) for spec in self.episodes],
            "variants": [variant.to_dict() for variant in self.variants],
            "repeats": self.repeats,
            "schedule": {"shuffle_seed": self.shuffle_seed},
            "execution": self.execution.to_dict(),
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class EvaluationJob:
    """Stable compiled unit scheduled by the durable evaluator."""

    index: int
    job_id: str
    pair_id: str
    source_episode_id: str
    variant_id: str
    repeat_index: int
    runtime: JsonDict
    spec: ParallelEpisodeSpec

    def to_dict(self) -> JsonDict:
        return {
            "index": self.index,
            "job_id": self.job_id,
            "pair_id": self.pair_id,
            "source_episode_id": self.source_episode_id,
            "variant_id": self.variant_id,
            "repeat_index": self.repeat_index,
            "runtime": self.runtime,
            "episode": _episode_spec_to_dict(self.spec),
        }


def load_evaluation_plan(path: str | Path) -> EvaluationPlan:
    """Load a plan and resolve a referenced episode manifest relative to it."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evaluation plan must contain one JSON object")
    observed_schema = payload.get("schema_version")
    if observed_schema != EVALUATION_PLAN_SCHEMA_VERSION:
        raise ValueError(
            f"evaluation plan schema must be {EVALUATION_PLAN_SCHEMA_VERSION}, "
            f"got {observed_schema!r}"
        )
    plan_id = _required_identifier(payload, "plan_id", location="plan")
    episodes = _load_plan_episodes(payload, source.parent)
    raw_variants = payload.get("variants")
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("evaluation plan requires a non-empty variants list")
    variants = tuple(
        EvaluationVariant.from_dict(_object(item, f"variants[{index}]"), index=index)
        for index, item in enumerate(raw_variants)
    )
    _require_unique((variant.variant_id for variant in variants), "variant_id")
    repeats = int(payload.get("repeats", 1))
    if repeats < 1:
        raise ValueError("evaluation plan repeats must be positive")
    schedule = payload.get("schedule")
    schedule = dict(schedule) if isinstance(schedule, dict) else {}
    metadata = payload.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("evaluation plan metadata must be an object")
    return EvaluationPlan(
        plan_id=plan_id,
        description=str(payload.get("description") or ""),
        episodes=episodes,
        variants=variants,
        repeats=repeats,
        shuffle_seed=int(schedule.get("shuffle_seed", 0)),
        execution=EvaluationExecution.from_dict(payload.get("execution")),
        metadata=dict(metadata or {}),
        source_path=str(source),
        source_payload=dict(payload),
    )


def compile_evaluation_plan(plan: EvaluationPlan) -> tuple[EvaluationJob, ...]:
    """Expand a plan into deterministically shuffled, paired jobs."""

    jobs: list[EvaluationJob] = []
    for repeat_index in range(plan.repeats):
        for source_spec in plan.episodes:
            pair_id = _safe_id(f"{source_spec.episode_id}-r{repeat_index:03d}")
            for variant in plan.variants:
                job_id = _safe_id(f"{pair_id}-{variant.variant_id}")
                evaluation_metadata: JsonDict = {
                    "plan_id": plan.plan_id,
                    "job_id": job_id,
                    "pair_id": pair_id,
                    "source_episode_id": source_spec.episode_id,
                    "variant_id": variant.variant_id,
                    "repeat_index": repeat_index,
                    "variant_metadata": variant.metadata,
                }
                spec = replace(
                    source_spec,
                    episode_id=job_id,
                    metadata={
                        **source_spec.metadata,
                        "evaluation": evaluation_metadata,
                        "evaluation_runtime": variant.runtime,
                    },
                )
                jobs.append(
                    EvaluationJob(
                        index=len(jobs),
                        job_id=job_id,
                        pair_id=pair_id,
                        source_episode_id=source_spec.episode_id,
                        variant_id=variant.variant_id,
                        repeat_index=repeat_index,
                        runtime=variant.runtime,
                        spec=spec,
                    )
                )
    random.Random(plan.shuffle_seed).shuffle(jobs)
    return tuple(replace(job, index=index) for index, job in enumerate(jobs))


def plan_sha256(plan: EvaluationPlan, jobs: Iterable[EvaluationJob]) -> str:
    """Hash resolved inputs, not just the possibly indirect source file."""

    payload = {
        "plan": plan.to_dict(),
        "jobs": [job.to_dict() for job in jobs],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compiled_plan_payload(
    plan: EvaluationPlan,
    jobs: Iterable[EvaluationJob],
) -> JsonDict:
    selected = tuple(jobs)
    return {
        "schema_version": COMPILED_EVALUATION_SCHEMA_VERSION,
        "plan": plan.to_dict(),
        "plan_sha256": plan_sha256(plan, selected),
        "job_count": len(selected),
        "jobs": [job.to_dict() for job in selected],
    }


def evaluation_job_from_dict(value: JsonDict) -> EvaluationJob:
    """Restore one compiled job from an immutable run snapshot."""

    episode = _object(value.get("episode"), "compiled job episode")
    spec = ParallelEpisodeSpec.from_dict(episode, index=int(value.get("index") or 0))
    runtime = value.get("runtime")
    if runtime is not None and not isinstance(runtime, dict):
        raise ValueError("compiled job runtime must be an object")
    return EvaluationJob(
        index=int(value.get("index") or 0),
        job_id=_required_identifier(value, "job_id", location="compiled job"),
        pair_id=_required_identifier(value, "pair_id", location="compiled job"),
        source_episode_id=_required_identifier(
            value, "source_episode_id", location="compiled job"
        ),
        variant_id=_required_identifier(value, "variant_id", location="compiled job"),
        repeat_index=int(value.get("repeat_index") or 0),
        runtime=dict(runtime or {}),
        spec=spec,
    )


def _load_plan_episodes(payload: JsonDict, base_dir: Path) -> tuple[ParallelEpisodeSpec, ...]:
    inline = payload.get("episodes")
    reference = str(payload.get("episode_manifest") or "").strip()
    if inline is not None and reference:
        raise ValueError("use either episodes or episode_manifest, not both")
    if reference:
        reference_path = Path(reference)
        if reference_path.is_absolute() or ".." in reference_path.parts:
            raise ValueError("episode_manifest must be relative to the evaluation plan")
        manifest = json.loads((base_dir / reference_path).read_text(encoding="utf-8"))
        rows = manifest.get("episodes") if isinstance(manifest, dict) else manifest
    else:
        rows = inline
    if not isinstance(rows, list) or not rows:
        raise ValueError("evaluation plan requires episodes or episode_manifest")
    specs = tuple(
        ParallelEpisodeSpec.from_dict(_object(item, f"episodes[{index}]"), index=index)
        for index, item in enumerate(rows)
    )
    _require_unique((spec.episode_id for spec in specs), "episode_id")
    return specs


def _episode_spec_to_dict(spec: ParallelEpisodeSpec) -> JsonDict:
    return {
        "episode_id": spec.episode_id,
        "task": spec.task,
        "env_id": spec.env_id,
        "seed": spec.seed,
        "max_turns": spec.max_turns,
        "recovery_turns_per_branch": spec.recovery_turns_per_branch,
        "max_recovery_turns": spec.max_recovery_turns,
        "max_tool_calls": spec.max_tool_calls,
        "timeout_s": spec.timeout_s,
        "max_total_tokens": spec.max_total_tokens,
        "metadata": spec.metadata,
    }


def _required_identifier(value: JsonDict, key: str, *, location: str) -> str:
    result = str(value.get(key) or "").strip()
    if not result:
        raise ValueError(f"{location}.{key} is required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", result):
        raise ValueError(f"{location}.{key} contains unsupported characters")
    return result


def _safe_id(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    if not result:
        raise ValueError("compiled evaluation id is empty")
    if len(result) <= 220:
        return result
    suffix = hashlib.sha256(result.encode("utf-8")).hexdigest()[:12]
    return f"{result[:207]}-{suffix}"


def _object(value: object, location: str) -> JsonDict:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    return dict(value)


def _require_unique(values: Iterable[str], field_name: str) -> None:
    selected = list(values)
    if len(selected) != len(set(selected)):
        raise ValueError(f"evaluation plan {field_name} values must be unique")
