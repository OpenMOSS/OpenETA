#!/usr/bin/env python3
"""Run the ordered, isolated, single-seed R0.9 expert-collection runtime gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sim.envs.univtac.isaac51_blackwell_runtime import clean_runtime_environment, load_json, managed
from sim.envs.univtac.isaac51_runtime_validation import (
    CollectionGate,
    build_collect_command,
    inspect_fresh_output_root,
    summarize_collection_gate,
    validate_collect_command,
)
from sim.envs.univtac.resource_sanitation import utc_now, write_json
from sim.envs.univtac.simulator_install_contract import RUNTIME_VARIANT, validate_config, validate_provenance_marker


def collection_gates(config: dict[str, Any]) -> tuple[CollectionGate, ...]:
    return tuple(
        CollectionGate(
            gate_id=str(item["id"]), task=str(item["task"]), seed=int(item["seed"]),
            label=str(item["label"]), output_dir=str(item["output_dir"]),
            conditional=bool(item.get("conditional", False)), requires=item.get("requires"),
        )
        for item in config["collection_gates"]
    )


def should_continue(gate_id: str, result: dict[str, Any]) -> bool:
    if not result.get("seed_contract_valid") or not result.get("cleanup_success"):
        return False
    if result.get("classification") in {"runtime_error", "timeout", "seed_contract_violation"}:
        return False
    if gate_id == "G0":
        return bool(result.get("episode_saved"))
    return True


def c0_allows_h0(c0: dict[str, Any], smoke: dict[str, Any]) -> bool:
    return bool(
        c0.get("seed_contract_valid")
        and c0.get("reset_returned")
        and c0.get("expert_trajectory_completed")
        and c0.get("cleanup_success")
        and smoke.get("status") == "completed"
        and smoke.get("cleanup_complete")
        and all(record.get("passed") for record in smoke.get("runs", {}).values())
    )


def classify(completed: dict[str, dict[str, Any]]) -> str:
    if any(item.get("seed_contract_violation") for item in completed.values()):
        return "seed_contract_violation"
    if any(not item.get("cleanup_success") for item in completed.values()):
        return "cleanup_incomplete"
    g0 = completed.get("G0")
    if not g0 or not g0.get("episode_saved"):
        return "isaac51_phase1_collection_failed"
    for gate_id, failure in (("L0", "isaac51_lift_runtime_failed"), ("C0", "isaac51_pull_out_key_runtime_failed")):
        result = completed.get(gate_id)
        if not result or result.get("classification") in {"runtime_error", "timeout"}:
            return failure
    if "H0" in completed and completed["H0"].get("classification") != "episode_saved":
        return "isaac51_insertion_planner_failure"
    return "isaac51_blackwell_runtime_viable" if "H0" in completed else "isaac51_contact_rich_runtime_viable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conda-exe", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    source = args.source_checkout.resolve()
    output = args.output_root.resolve()
    smoke = load_json(output / "simulator_smoke/run_manifest.json")
    if smoke.get("status") != "completed" or not smoke.get("cleanup_complete"):
        raise RuntimeError("sequential S0/S1 Taxim smoke gates have not passed cleanly")
    base = subprocess.run([str(args.conda_exe), "info", "--base"], check=True, capture_output=True, text=True).stdout.strip()
    prefix = Path(base) / "envs" / config["environment"]["conda_name"]
    validate_provenance_marker(
        load_json(prefix / ".univtac-r09-provenance.json"),
        source=config["environment"]["clone_from"], target=config["environment"]["conda_name"],
    )
    environment, _ = clean_runtime_environment(prefix, source, config["runtime"]["gpu"])
    gates_root = output / "task_gates"
    if gates_root.exists():
        raise FileExistsError("R0.9 task gate output already exists")
    gates_root.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema_version": "openeta.univtac.r09_task_gates.v1", "runtime_variant": RUNTIME_VARIANT,
        **config["claims"], "started_at": utc_now(), "status": "running", "classification": None,
        "ftp1_model_loaded": False, "openeta_imported": False, "agent_started": False,
        "parallel_isaac": False, "gates": {},
    }
    write_json(gates_root / "run_manifest.json", manifest)
    completed: dict[str, dict[str, Any]] = {}
    try:
        for gate in collection_gates(config):
            if gate.gate_id == "H0" and not c0_allows_h0(completed.get("C0", {}), smoke):
                manifest["h0_triggered"] = False
                break
            gate_root = gates_root / gate.output_dir
            collection_root = gate_root / "collection"
            freshness = inspect_fresh_output_root(collection_root)
            if not freshness["fresh_output_root"]:
                raise RuntimeError(f"{gate.gate_id} output is not fresh")
            gate_root.mkdir(parents=True, exist_ok=False)
            command = build_collect_command(
                python=prefix / "bin/python", source_root=source, gate=gate,
                collection_root=collection_root, gpu=config["runtime"]["gpu"],
            )
            validate_collect_command(command, gate)
            process = managed(
                command, cwd=source, output_root=gate_root, name="collect",
                timeout_seconds=float(config["timeouts_seconds"]["collection"]), environment=environment,
            )
            log_text = Path(process["log_path"]).read_text(encoding="utf-8", errors="replace")
            result = summarize_collection_gate(
                gate=gate, collection_root=collection_root, log_text=log_text,
                process_result=process, freshness=freshness,
            )
            if gate.gate_id == "C0":
                result["tactile_observation_legal"] = c0_allows_h0(result, smoke)
            completed[gate.gate_id] = result
            manifest["gates"][gate.gate_id] = result
            if gate.gate_id == "H0":
                manifest["h0_triggered"] = True
            write_json(gate_root / "gate_result.json", result)
            write_json(gates_root / "run_manifest.json", manifest)
            if not should_continue(gate.gate_id, result):
                break
        manifest["classification"] = classify(completed)
        manifest["status"] = "completed" if manifest["classification"] in {
            "isaac51_blackwell_runtime_viable", "isaac51_contact_rich_runtime_viable", "isaac51_insertion_planner_failure"
        } else "failed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["classification"] = manifest["classification"] or "blocked_by_external_resources"
        manifest["failure"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_json(gates_root / "failure.json", manifest["failure"])
        raise
    finally:
        manifest["ended_at"] = utc_now()
        write_json(gates_root / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
