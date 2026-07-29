"""RoboCasa365 benchmark manifests and result aggregation.

RoboCasa's public protocol evaluates a selected task set and split on 50
randomly sampled scenarios per task.  This module makes that sampling explicit
and serialisable so an interrupted OpenETA run can be resumed without silently
changing the benchmark instances.

All RoboCasa imports are lazy; manifest files can be loaded and inspected from
the lightweight agent environment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


MANIFEST_SCHEMA = "openeta.robocasa_benchmark_manifest.v1"
RESULT_SCHEMA = "openeta.robocasa_benchmark_result.v1"
DEFAULT_SCENARIOS_PER_TASK = 50
VALID_SPLITS = ("pretrain", "target")
PARALLEL_BATCH_SCHEMA = "openeta.parallel_episode_batch.v2"
PARALLEL_EPISODE_LIMIT_KEYS = frozenset(
    {"max_turns", "max_tool_calls", "timeout_s", "max_total_tokens"}
)


@dataclass(frozen=True, slots=True)
class RoboCasaScenario:
    task: str
    split: str
    scenario_index: int
    seed: int
    horizon: int

    @property
    def scenario_id(self) -> str:
        return f"{self.task}:{self.split}:{self.scenario_index:02d}:{self.seed}"

    @property
    def env_id(self) -> str:
        return f"openeta/robocasa_{self.split}_{self.task}-v0"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scenario_id"] = self.scenario_id
        data["env_id"] = self.env_id
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoboCasaScenario":
        return cls(
            task=str(data["task"]),
            split=str(data["split"]),
            scenario_index=int(data["scenario_index"]),
            seed=int(data["seed"]),
            horizon=int(data["horizon"]),
        )


@dataclass(slots=True)
class RoboCasaBenchmarkManifest:
    task_set: str
    split: str
    master_seed: int
    scenarios_per_task: int
    scenarios: list[RoboCasaScenario]
    robocasa_version: str = "1.0.1"
    robocasa_commit: str = ""
    schema_version: str = MANIFEST_SCHEMA

    @property
    def task_count(self) -> int:
        return len({scenario.task for scenario in self.scenarios})

    @property
    def rollout_count(self) -> int:
        return len(self.scenarios)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "benchmark": "robocasa365",
            "robocasa_version": self.robocasa_version,
            "robocasa_commit": self.robocasa_commit,
            "task_set": self.task_set,
            "split": self.split,
            "master_seed": self.master_seed,
            "scenarios_per_task": self.scenarios_per_task,
            "task_count": self.task_count,
            "rollout_count": self.rollout_count,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoboCasaBenchmarkManifest":
        schema = str(data.get("schema_version", ""))
        if schema != MANIFEST_SCHEMA:
            raise ValueError(f"Unsupported RoboCasa manifest schema {schema!r}")
        manifest = cls(
            task_set=str(data["task_set"]),
            split=str(data["split"]),
            master_seed=int(data["master_seed"]),
            scenarios_per_task=int(data["scenarios_per_task"]),
            scenarios=[RoboCasaScenario.from_dict(item) for item in data["scenarios"]],
            robocasa_version=str(data.get("robocasa_version", "")),
            robocasa_commit=str(data.get("robocasa_commit", "")),
        )
        if manifest.split not in VALID_SPLITS:
            raise ValueError(f"Invalid RoboCasa split {manifest.split!r}")
        expected = manifest.task_count * manifest.scenarios_per_task
        if manifest.rollout_count != expected:
            raise ValueError(
                f"Manifest has {manifest.rollout_count} scenarios; expected {expected}"
            )
        expected_hash = data.get("manifest_sha256")
        if expected_hash and expected_hash != manifest.to_dict()["manifest_sha256"]:
            raise ValueError("RoboCasa manifest checksum mismatch")
        return manifest

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def to_parallel_dict(
        self,
        *,
        task_resolver: Callable[[RoboCasaScenario], str] | None = None,
        episode_limits: Mapping[str, int | float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Adapt this manifest to the shared ``ParallelEpisodeSpec`` input.

        The native RoboCasa manifest remains the source of truth for scenario
        sampling and checksums.  This additive view can be passed directly to
        ``openeta-batch`` without changing the existing manifest/result schema.
        """

        return build_parallel_episode_manifest(
            self,
            task_resolver=task_resolver,
            episode_limits=episode_limits,
            metadata=metadata,
        )

    def write_parallel_json(
        self,
        path: str | Path,
        *,
        task_resolver: Callable[[RoboCasaScenario], str] | None = None,
        episode_limits: Mapping[str, int | float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        payload = self.to_parallel_dict(
            task_resolver=task_resolver,
            episode_limits=episode_limits,
            metadata=metadata,
        )
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read_json(cls, path: str | Path) -> "RoboCasaBenchmarkManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _task_class_fallback_instruction(scenario: RoboCasaScenario) -> str:
    """Return a readable fallback when scenario-specific language is unavailable."""

    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", scenario.task)
    words = re.sub(r"[_\s]+", " ", words).strip().lower()
    return f"RoboCasa365 {scenario.split} task: {words or scenario.task}"


def build_parallel_episode_manifest(
    manifest: RoboCasaBenchmarkManifest,
    *,
    task_resolver: Callable[[RoboCasaScenario], str] | None = None,
    episode_limits: Mapping[str, int | float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared ``{episodes: [...]}`` batch-harness representation.

    ``task_resolver`` may provide the exact natural-language instruction for a
    sampled scenario.  Without one, the task class is humanised as an explicit
    fallback; the authoritative task class is always retained in metadata.

    Official RoboCasa success is deliberately fail-closed: every emitted
    episode forces ``require_official_reward=True`` so the common harness must
    observe a trusted positive simulator receipt.
    """

    resolver = task_resolver or _task_class_fallback_instruction
    limits = dict(episode_limits or {})
    unsupported_limits = sorted(set(limits) - PARALLEL_EPISODE_LIMIT_KEYS)
    if unsupported_limits:
        raise ValueError(
            "Unsupported ParallelEpisodeSpec limit keys: "
            + ", ".join(unsupported_limits)
        )

    manifest_payload = manifest.to_dict()
    manifest_sha256 = str(manifest_payload["manifest_sha256"])
    common_metadata = dict(metadata or {})
    episodes: list[dict[str, Any]] = []
    for scenario in manifest.scenarios:
        task = str(resolver(scenario) or "").strip()
        if not task:
            raise ValueError(
                f"Parallel task resolver returned an empty instruction for "
                f"{scenario.scenario_id}"
            )
        episode_metadata = {
            **common_metadata,
            "benchmark": "robocasa365",
            "task_class": scenario.task,
            "task_set": manifest.task_set,
            "split": scenario.split,
            "scenario_index": scenario.scenario_index,
            "horizon": scenario.horizon,
            "manifest_sha256": manifest_sha256,
            # This field is consumed by ParallelEpisodeHarness.classify_episode_result.
            # Keep it last so a caller cannot accidentally weaken benchmark scoring.
            "require_official_reward": True,
        }
        episodes.append(
            {
                "episode_id": scenario.scenario_id,
                "task": task,
                "env_id": scenario.env_id,
                "seed": scenario.seed,
                **limits,
                "metadata": episode_metadata,
            }
        )

    return {
        "benchmark": "robocasa365",
        "source_schema_version": manifest.schema_version,
        "source_manifest_sha256": manifest_sha256,
        "task_set": manifest.task_set,
        "split": manifest.split,
        "episodes": episodes,
    }


def _official_registry() -> tuple[Mapping[str, Sequence[str]], Callable[[str], int], str, str]:
    import robocasa
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    version = str(getattr(robocasa, "__version__", "1.0.1"))
    commit = ""
    try:
        import subprocess

        package_root = Path(robocasa.__path__[0]).parent
        commit = subprocess.check_output(
            ["git", "-C", str(package_root), "rev-parse", "HEAD"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        pass
    return TASK_SET_REGISTRY, get_task_horizon, version, commit


def list_task_sets(
    task_registry: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, int]:
    if task_registry is None:
        task_registry, _, _, _ = _official_registry()
    return {name: len(tasks) for name, tasks in task_registry.items()}


def build_manifest(
    task_set: str,
    split: str,
    *,
    scenarios_per_task: int = DEFAULT_SCENARIOS_PER_TASK,
    master_seed: int = 0,
    task_registry: Mapping[str, Sequence[str]] | None = None,
    horizon_lookup: Callable[[str], int] | None = None,
    robocasa_version: str | None = None,
    robocasa_commit: str | None = None,
) -> RoboCasaBenchmarkManifest:
    """Create a deterministic manifest of randomly sampled scenarios.

    A separate ``SeedSequence`` child is used per task.  Adding or reordering a
    different task set therefore cannot change the 50 seeds assigned to a task
    already present in the manifest.
    """

    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
    if scenarios_per_task <= 0:
        raise ValueError("scenarios_per_task must be positive")

    if task_registry is None or horizon_lookup is None:
        official_registry, official_horizon, version, commit = _official_registry()
        if task_registry is None:
            task_registry = official_registry
        if horizon_lookup is None:
            horizon_lookup = official_horizon
        if robocasa_version is None:
            robocasa_version = version
        if robocasa_commit is None:
            robocasa_commit = commit

    if task_set not in task_registry:
        raise KeyError(
            f"Unknown RoboCasa task set {task_set!r}; "
            f"available: {', '.join(sorted(task_registry))}"
        )

    scenarios: list[RoboCasaScenario] = []
    for task in task_registry[task_set]:
        # Derive task entropy from its stable UTF-8 name, not registry order.
        digest = hashlib.sha256(task.encode("utf-8")).digest()
        task_entropy = int.from_bytes(digest[:8], "little")
        seed_sequence = np.random.SeedSequence([int(master_seed), task_entropy])
        rng = np.random.default_rng(seed_sequence)
        seeds = rng.choice(
            np.iinfo(np.int32).max,
            size=int(scenarios_per_task),
            replace=False,
        )
        horizon = int(horizon_lookup(task))
        for index, scenario_seed in enumerate(seeds.tolist()):
            scenarios.append(
                RoboCasaScenario(
                    task=str(task),
                    split=split,
                    scenario_index=index,
                    seed=int(scenario_seed),
                    horizon=horizon,
                )
            )

    return RoboCasaBenchmarkManifest(
        task_set=task_set,
        split=split,
        master_seed=int(master_seed),
        scenarios_per_task=int(scenarios_per_task),
        scenarios=scenarios,
        robocasa_version=robocasa_version or "1.0.1",
        robocasa_commit=robocasa_commit or "",
    )


@dataclass(frozen=True, slots=True)
class RoboCasaRolloutResult:
    scenario_id: str
    task: str
    split: str
    seed: int
    success: bool
    steps: int
    status: str = "completed"
    error: str = ""
    video_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoboCasaRolloutResult":
        return cls(
            scenario_id=str(data["scenario_id"]),
            task=str(data["task"]),
            split=str(data["split"]),
            seed=int(data["seed"]),
            success=bool(data["success"]),
            steps=int(data["steps"]),
            status=str(data.get("status", "completed")),
            error=str(data.get("error", "")),
            video_path=str(data.get("video_path", "")),
            metadata=dict(data.get("metadata", {})),
        )


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _validate_rollout_identity(
    scenario: RoboCasaScenario,
    result: RoboCasaRolloutResult,
) -> None:
    expected = {
        "scenario_id": scenario.scenario_id,
        "task": scenario.task,
        "split": scenario.split,
        "seed": scenario.seed,
    }
    actual = {
        "scenario_id": result.scenario_id,
        "task": result.task,
        "split": result.split,
        "seed": result.seed,
    }
    mismatches = [
        f"{key}={actual[key]!r} (expected {value!r})"
        for key, value in expected.items()
        if actual[key] != value
    ]
    if mismatches:
        raise ValueError(
            f"Runner result identity mismatch for {scenario.scenario_id}: "
            + "; ".join(mismatches)
        )
    if result.steps < 0:
        raise ValueError(
            f"Runner returned negative steps for {scenario.scenario_id}: {result.steps}"
        )


def evaluate_manifest(
    manifest: RoboCasaBenchmarkManifest,
    rollout_fn: Callable[[RoboCasaScenario], RoboCasaRolloutResult | Mapping[str, Any]],
    *,
    output_path: str | Path,
    resume: bool = True,
    fail_fast: bool = False,
    max_rollouts: int | None = None,
) -> dict[str, Any]:
    """Run pending scenarios sequentially and durably checkpoint every result.

    ``rollout_fn`` is deliberately policy-agnostic: an OpenETA agent, a learned
    vector-action policy, or a remote service can own one complete rollout.  It
    receives the immutable task/split/seed/horizon tuple and must return a
    :class:`RoboCasaRolloutResult` or an equivalent mapping.
    """

    output = Path(output_path)
    results: list[RoboCasaRolloutResult] = []
    if resume and output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        expected_hash = manifest.to_dict()["manifest_sha256"]
        if previous.get("manifest_sha256") != expected_hash:
            raise ValueError("Existing result file belongs to a different manifest")
        results = [RoboCasaRolloutResult.from_dict(item) for item in previous["rollouts"]]

    completed_ids = {result.scenario_id for result in results}
    if len(completed_ids) != len(results):
        raise ValueError("Existing result file contains duplicate scenario ids")

    attempted = 0
    for scenario in manifest.scenarios:
        if scenario.scenario_id in completed_ids:
            continue
        if max_rollouts is not None and attempted >= max_rollouts:
            break
        attempted += 1
        try:
            raw_result = rollout_fn(scenario)
            result = (
                raw_result
                if isinstance(raw_result, RoboCasaRolloutResult)
                else RoboCasaRolloutResult.from_dict(raw_result)
            )
            _validate_rollout_identity(scenario, result)
        except Exception as exc:
            result = RoboCasaRolloutResult(
                scenario_id=scenario.scenario_id,
                task=scenario.task,
                split=scenario.split,
                seed=scenario.seed,
                success=False,
                steps=0,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
            results.append(result)
            _atomic_write_json(output, aggregate_results(manifest, results))
            if fail_fast:
                raise
            completed_ids.add(scenario.scenario_id)
            continue

        results.append(result)
        completed_ids.add(scenario.scenario_id)
        _atomic_write_json(output, aggregate_results(manifest, results))

    summary = aggregate_results(manifest, results)
    _atomic_write_json(output, summary)
    return summary


def aggregate_results(
    manifest: RoboCasaBenchmarkManifest,
    results: Iterable[RoboCasaRolloutResult],
) -> dict[str, Any]:
    result_list = list(results)
    by_id = {result.scenario_id: result for result in result_list}
    if len(by_id) != len(result_list):
        raise ValueError("Results contain duplicate scenario ids")
    expected_ids = {scenario.scenario_id for scenario in manifest.scenarios}
    unexpected = sorted(set(by_id) - expected_ids)
    if unexpected:
        raise ValueError(f"Results contain scenarios outside the manifest: {unexpected[:3]}")
    scenario_by_id = {
        scenario.scenario_id: scenario for scenario in manifest.scenarios
    }
    for result in result_list:
        _validate_rollout_identity(scenario_by_id[result.scenario_id], result)

    by_task: dict[str, dict[str, Any]] = {}
    for task in sorted({scenario.task for scenario in manifest.scenarios}):
        task_scenarios = [scenario for scenario in manifest.scenarios if scenario.task == task]
        task_results = [by_id[s.scenario_id] for s in task_scenarios if s.scenario_id in by_id]
        successes = sum(int(result.success) for result in task_results)
        by_task[task] = {
            "completed": len(task_results),
            "expected": len(task_scenarios),
            "successes": successes,
            "success_rate": successes / len(task_results) if task_results else None,
        }

    successes = sum(int(result.success) for result in result_list)
    failures = sum(int(not result.success) for result in result_list)
    return {
        "schema_version": RESULT_SCHEMA,
        "benchmark": "robocasa365",
        "manifest_sha256": manifest.to_dict()["manifest_sha256"],
        "task_set": manifest.task_set,
        "split": manifest.split,
        "completed_rollouts": len(result_list),
        "expected_rollouts": manifest.rollout_count,
        "successes": successes,
        "failures": failures,
        "success_rate": successes / len(result_list) if result_list else None,
        "complete": len(result_list) == manifest.rollout_count,
        "by_task": by_task,
        "rollouts": [result.to_dict() for result in result_list],
    }


def _parallel_outcome_has_trusted_success(outcome: Mapping[str, Any]) -> bool:
    episode = outcome.get("episode")
    if not isinstance(episode, Mapping):
        return False
    episode_metadata = episode.get("metadata")
    execution_id = (
        str(episode_metadata.get("execution_id") or "")
        if isinstance(episode_metadata, Mapping)
        else ""
    )
    steps = episode.get("steps")
    if not isinstance(steps, list):
        return False
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        step_result = step.get("step_result")
        if not isinstance(step_result, Mapping):
            continue
        reward = step_result.get("reward")
        if (
            not isinstance(reward, (int, float))
            or isinstance(reward, bool)
            or not math.isfinite(float(reward))
            or float(reward) <= 0.0
        ):
            continue
        info = step_result.get("info")
        if not isinstance(info, Mapping):
            continue
        receipt = info.get("environment_receipt")
        if not (
            info.get("environment_receipt_trusted") is True
            and info.get("official_reward") is True
            and isinstance(receipt, Mapping)
            and (
                not execution_id
                or str(receipt.get("execution_id") or "") == execution_id
            )
        ):
            continue
        receipt_schema = str(receipt.get("schema_version") or "")
        if receipt_schema:
            if receipt_schema == "openeta.environment_receipt.v1":
                return True
            continue
        # EpisodeResult.to_dict intentionally compacts nested metadata and can
        # omit schema_version after the live harness has already validated the
        # full receipt. Require the stable receipt identity fields retained by
        # that serializer before accepting the already-classified success.
        if all(
            isinstance(receipt.get(key), str) and bool(receipt.get(key))
            for key in ("receipt_id", "backend", "agent_tool", "remote_tool")
        ):
            return True
    return False


def _parallel_outcome_step_count(outcome: Mapping[str, Any]) -> int:
    """Prefer RoboCasa's simulator counter; fall back to agent turn count."""

    episode = outcome.get("episode")
    if not isinstance(episode, Mapping):
        return 0
    elapsed_steps: list[int] = []
    steps = episode.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            step_result = step.get("step_result")
            observation = (
                step_result.get("observation")
                if isinstance(step_result, Mapping)
                else None
            )
            observation_metadata = (
                observation.get("metadata")
                if isinstance(observation, Mapping)
                else None
            )
            benchmark = (
                observation_metadata.get("benchmark")
                if isinstance(observation_metadata, Mapping)
                else None
            )
            elapsed = (
                benchmark.get("elapsed_steps")
                if isinstance(benchmark, Mapping)
                else None
            )
            if isinstance(elapsed, int) and not isinstance(elapsed, bool) and elapsed >= 0:
                elapsed_steps.append(elapsed)
    if elapsed_steps:
        return max(elapsed_steps)
    num_steps = episode.get("num_steps")
    if isinstance(num_steps, int) and not isinstance(num_steps, bool):
        return max(0, num_steps)
    return len(steps) if isinstance(steps, list) else 0


def aggregate_parallel_batch_results(
    manifest: RoboCasaBenchmarkManifest,
    batch_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert a shared batch-harness v2 result to the native RoboCasa summary.

    A common-harness ``success`` is accepted only when its serialised episode
    contains the same trusted positive environment receipt required during
    execution.  This prevents an edited or incorrectly classified batch result
    from turning a planner declaration into benchmark success.
    """

    schema = str(batch_result.get("schema_version") or "")
    if schema != PARALLEL_BATCH_SCHEMA:
        raise ValueError(
            f"Unsupported parallel batch result schema {schema!r}; "
            f"expected {PARALLEL_BATCH_SCHEMA!r}"
        )
    outcomes = batch_result.get("outcomes")
    if not isinstance(outcomes, list):
        raise TypeError("Parallel batch result must contain an outcomes list")

    scenario_by_id = {
        scenario.scenario_id: scenario for scenario in manifest.scenarios
    }
    seen_ids: set[str] = set()
    results: list[RoboCasaRolloutResult] = []
    for index, outcome in enumerate(outcomes):
        if not isinstance(outcome, Mapping):
            raise TypeError(f"outcomes[{index}] must be an object")
        episode_id = str(outcome.get("episode_id") or "")
        if episode_id not in scenario_by_id:
            raise ValueError(
                f"Parallel outcome {episode_id!r} is outside the RoboCasa manifest"
            )
        if episode_id in seen_ids:
            raise ValueError(f"Duplicate parallel outcome episode_id: {episode_id}")
        seen_ids.add(episode_id)

        status = str(outcome.get("status") or "")
        if status not in {"success", "fail", "need_human"}:
            raise ValueError(
                f"Unsupported parallel outcome status for {episode_id}: {status!r}"
            )
        success = status == "success"
        if success and not _parallel_outcome_has_trusted_success(outcome):
            raise ValueError(
                f"Parallel success for {episode_id} has no trusted official reward receipt"
            )

        scenario = scenario_by_id[episode_id]
        error_payload = outcome.get("error")
        error = (
            json.dumps(error_payload, sort_keys=True)
            if isinstance(error_payload, Mapping) and error_payload
            else ""
        )
        results.append(
            RoboCasaRolloutResult(
                scenario_id=scenario.scenario_id,
                task=scenario.task,
                split=scenario.split,
                seed=scenario.seed,
                success=success,
                steps=_parallel_outcome_step_count(outcome),
                status=status,
                error=error,
                metadata={
                    "source": "openeta.parallel_episode_batch.v2",
                    "batch_id": str(batch_result.get("batch_id") or ""),
                    "parallel_status": status,
                    "cleanup": dict(outcome.get("cleanup") or {})
                    if isinstance(outcome.get("cleanup"), Mapping)
                    else {},
                    "assistance": dict(outcome.get("assistance") or {})
                    if isinstance(outcome.get("assistance"), Mapping)
                    else {},
                },
            )
        )
    return aggregate_results(manifest, results)
