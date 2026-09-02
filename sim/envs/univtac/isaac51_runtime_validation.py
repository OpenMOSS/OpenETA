"""Pure-Python contracts for the isolated UniVTAC Isaac Sim 5.1 gates."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "openeta.univtac.isaac51_runtime_validation.v1"
SOURCE_COMMIT = "371fac67917307026be8f00869fcc1b61c623a9f"
EXPECTED_GATE_ORDER = ("G0", "L0", "C0", "H0")
EXPECTED_GATE_CONTRACTS = {
    "G0": (
        "grasp_classify",
        0,
        "official_isaac51_phase1_collection_smoke",
        "grasp_classify_seed0",
    ),
    "L0": (
        "lift_can",
        1_000_000,
        "legacy_ftp1_eval_seed_index_aligned",
        "lift_can_seed1000000",
    ),
    "C0": (
        "pull_out_key",
        1_000_000,
        "legacy_ftp1_eval_seed_index_aligned",
        "pull_out_key_seed1000000",
    ),
    "H0": (
        "insert_hole",
        1_000_000,
        "legacy_ftp1_eval_seed_index_aligned",
        "insert_hole_seed1000000",
    ),
}
SEED_LINE = re.compile(r"\bSeed\s+(\d+)\s+(?:success|failed)\b")
START_LINE = re.compile(r"Starting from seed\s+(\d+)\.")


@dataclass(frozen=True)
class CollectionGate:
    gate_id: str
    task: str
    seed: int
    label: str
    output_dir: str
    conditional: bool = False
    requires: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deterministic_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def environment_spec_sha256(config: Mapping[str, Any]) -> str:
    selected = {
        "schema_version": config.get("schema_version"),
        "source": config.get("source"),
        "environment": config.get("environment"),
        "runtime": config.get("runtime"),
        "gates": config.get("gates"),
    }
    return hashlib.sha256(
        json.dumps(selected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_config(config: Mapping[str, Any]) -> tuple[CollectionGate, ...]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    source = config.get("source", {})
    if source.get("commit") != SOURCE_COMMIT:
        raise ValueError("Isaac51 source must use the pinned immutable commit")
    environment = config.get("environment", {})
    expected_environment = {
        "conda_name": "UniVTAC-isaac51-r07",
        "python": "3.11",
        "torch": "2.7.0",
        "torchvision": "0.22.0",
        "torch_cuda": "12.6",
        "cuda_arch": "89",
        "build_jobs": 4,
    }
    for key, expected in expected_environment.items():
        if environment.get(key) != expected:
            raise ValueError(f"environment.{key} must be {expected!r}")

    gates: list[CollectionGate] = []
    raw_gates = config.get("gates")
    if not isinstance(raw_gates, list):
        raise TypeError("gates must be a list")
    for raw in raw_gates:
        gate = CollectionGate(
            gate_id=str(raw["id"]),
            task=str(raw["task"]),
            seed=int(raw["seed"]),
            label=str(raw["label"]),
            output_dir=str(raw["output_dir"]),
            conditional=bool(raw.get("conditional", False)),
            requires=raw.get("requires"),
        )
        gates.append(gate)
    if tuple(gate.gate_id for gate in gates) != EXPECTED_GATE_ORDER:
        raise ValueError(f"gate order must be {EXPECTED_GATE_ORDER}")
    for gate in gates:
        actual = (gate.task, gate.seed, gate.label, gate.output_dir)
        if actual != EXPECTED_GATE_CONTRACTS[gate.gate_id]:
            raise ValueError(f"{gate.gate_id} protocol contract changed: {actual!r}")
    if gates[-1].conditional is not True or gates[-1].requires != "C0_runtime_and_tactile_valid":
        raise ValueError("H0 must remain conditional on C0 runtime and tactile validity")
    return tuple(gates)


def build_collect_command(
    *, python: Path, source_root: Path, gate: CollectionGate, collection_root: Path, gpu: str
) -> list[str]:
    """Build the exact one-seed official collection command."""
    return [
        str(python),
        str(source_root / "scripts" / "collect_data.py"),
        gate.task,
        "demo",
        "--start_seed",
        str(gate.seed),
        "--max_seed",
        str(gate.seed),
        "--headless",
        "--gpu",
        str(gpu),
        "--config-overrides",
        "collect_settings.episode_num=1",
        f"collect_settings.save_root_dir={collection_root.resolve()}",
    ]


def validate_collect_command(command: Sequence[str], gate: CollectionGate) -> None:
    tokens = list(command)
    for flag in ("--start_seed", "--max_seed"):
        if tokens.count(flag) != 1:
            raise ValueError(f"collection command must contain one {flag}")
        value = int(tokens[tokens.index(flag) + 1])
        if value != gate.seed:
            raise ValueError(f"{flag} must equal requested seed {gate.seed}")
    if str(gate.seed + 1) in tokens:
        raise ValueError("collection command must not request the next seed")


def inspect_fresh_output_root(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    entries = sorted(str(item.relative_to(path)) for item in path.rglob("*")) if path.exists() else []
    forbidden = [
        item
        for item in entries
        if Path(item).name == "suc_map.txt"
        or Path(item).suffix.lower() in {".hdf5", ".h5", ".mp4", ".avi"}
        or Path(item).name == "metadata.json"
    ]
    return {
        "fresh_output_root": not entries,
        "preexisting_suc_map": any(Path(item).name == "suc_map.txt" for item in entries),
        "preexisting_entries": entries,
        "preexisting_forbidden_entries": forbidden,
    }


def observed_attempted_seeds(log_text: str) -> list[int]:
    seeds = [int(match.group(1)) for match in START_LINE.finditer(log_text)]
    seeds.extend(int(match.group(1)) for match in SEED_LINE.finditer(log_text))
    return list(dict.fromkeys(seeds))


def _read_metadata(task_root: Path) -> dict[str, Any]:
    path = task_root / "metadata.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def summarize_collection_gate(
    *,
    gate: CollectionGate,
    collection_root: Path,
    log_text: str,
    process_result: Mapping[str, Any],
    freshness: Mapping[str, Any],
) -> dict[str, Any]:
    collection_root = collection_root.resolve()
    task_root = collection_root / gate.task / "demo"
    observed = observed_attempted_seeds(log_text)
    unexpected = [seed for seed in observed if seed != gate.seed]
    metadata = _read_metadata(task_root)
    episode_metadata = metadata.get(str(gate.seed))
    if not isinstance(episode_metadata, dict):
        episode_metadata = {}
    result_value = episode_metadata.get("result")
    normal_result = bool(SEED_LINE.search(log_text))
    reset_exception = "task.reset(seed=seed)" in log_text
    hdf5 = task_root / "hdf5" / f"{gate.seed}.hdf5"
    if not hdf5.is_file():
        hdf5_candidates = list(task_root.rglob(f"{gate.seed}.hdf5")) if task_root.exists() else []
        hdf5 = hdf5_candidates[0] if hdf5_candidates else hdf5
    seed_contract_valid = (
        bool(freshness.get("fresh_output_root"))
        and not bool(freshness.get("preexisting_suc_map"))
        and observed == [gate.seed]
        and not unexpected
    )
    seed_contract_violation = (
        not bool(freshness.get("fresh_output_root"))
        or bool(freshness.get("preexisting_suc_map"))
        or bool(unexpected)
        or (bool(observed) and observed != [gate.seed])
    )
    if seed_contract_violation:
        classification = "seed_contract_violation"
    elif process_result.get("timed_out"):
        classification = "timeout"
    elif process_result.get("returncode") != 0:
        classification = "runtime_error"
    elif result_value == "success" and hdf5.is_file():
        classification = "episode_saved"
    elif normal_result:
        classification = "task_unsuccessful"
    else:
        classification = "runtime_error"
    return {
        "schema_version": "openeta.univtac.isaac51_collection_gate.v1",
        "gate_id": gate.gate_id,
        "task": gate.task,
        "label": gate.label,
        "requested_seed": gate.seed,
        "observed_attempted_seeds": observed,
        "unexpected_seeds": unexpected,
        "fresh_output_root": bool(freshness.get("fresh_output_root")),
        "preexisting_suc_map": bool(freshness.get("preexisting_suc_map")),
        "seed_contract_valid": seed_contract_valid,
        "seed_contract_violation": seed_contract_violation,
        "reset_returned": True if normal_result else (False if reset_exception else None),
        "expert_trajectory_completed": normal_result,
        "task_success": result_value == "success",
        "episode_saved": result_value == "success" and hdf5.is_file(),
        "metadata_result": result_value,
        "collection_root": str(collection_root),
        "task_output_root": str(task_root),
        "hdf5_path": str(hdf5) if hdf5.is_file() else None,
        "process_returncode": process_result.get("returncode"),
        "process_timed_out": bool(process_result.get("timed_out")),
        "cleanup_success": bool(process_result.get("cleanup_complete")),
        "classification": classification,
    }


def gate_sequence_decision(
    completed: Mapping[str, Mapping[str, Any]], *, c0_tactile_valid: bool = False
) -> str | None:
    """Return the next gate, stopping at any failed or violated predecessor."""
    for gate_id in ("G0", "L0", "C0"):
        result = completed.get(gate_id)
        if result is None:
            return gate_id
        if not result.get("seed_contract_valid") or result.get("classification") in {
            "runtime_error",
            "timeout",
            "seed_contract_violation",
        }:
            return None
    if not c0_tactile_valid:
        return None
    return "H0" if "H0" not in completed else None
