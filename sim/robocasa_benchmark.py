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
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


MANIFEST_SCHEMA = "openeta.robocasa_benchmark_manifest.v1"
RESULT_SCHEMA = "openeta.robocasa_benchmark_result.v1"
DEFAULT_SCENARIOS_PER_TASK = 50
VALID_SPLITS = ("pretrain", "target")


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

    @classmethod
    def read_json(cls, path: str | Path) -> "RoboCasaBenchmarkManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


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
            if result.scenario_id != scenario.scenario_id:
                raise ValueError(
                    f"Runner returned scenario {result.scenario_id!r}; "
                    f"expected {scenario.scenario_id!r}"
                )
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
