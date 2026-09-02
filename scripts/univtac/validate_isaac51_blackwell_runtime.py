#!/usr/bin/env python3
"""Run two sequential official Taxim smokes in the installed R0.9 environment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.univtac.resume_blackwell_native_bridge import managed_cleanup_summary
from sim.envs.univtac.isaac51_blackwell_runtime import clean_runtime_environment, load_json, managed, process_ok
from sim.envs.univtac.resource_sanitation import utc_now, write_json
from sim.envs.univtac.simulator_install_contract import RUNTIME_VARIANT, validate_config, validate_provenance_marker


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
    install_manifest = load_json(output / "run_manifest.json")
    if install_manifest.get("status") != "install_validated":
        raise RuntimeError("R0.9 package/native installation has not passed")
    base = subprocess.run([str(args.conda_exe), "info", "--base"], check=True, capture_output=True, text=True).stdout.strip()
    prefix = Path(base) / "envs" / config["environment"]["conda_name"]
    marker = load_json(prefix / ".univtac-r09-provenance.json")
    validate_provenance_marker(marker, source=config["environment"]["clone_from"], target=config["environment"]["conda_name"])
    environment, _ = clean_runtime_environment(prefix, source, config["runtime"]["gpu"])
    root = output / "simulator_smoke"
    if root.exists():
        raise FileExistsError("R0.9 simulator smoke output already exists")
    root.mkdir(parents=True)
    manifest = {
        "schema_version": "openeta.univtac.r09_simulator_smoke.v1",
        "runtime_variant": RUNTIME_VARIANT, **config["claims"],
        "started_at": utc_now(), "status": "running", "classification": None,
        "backend": "taxim", "headless": True, "runs": {}, "parallel_isaac": False,
    }
    write_json(root / "run_manifest.json", manifest)
    try:
        for gate_id, timeout_key in (("S0", "smoke_first"), ("S1", "smoke_second")):
            run_root = root / gate_id.lower()
            smoke_output = run_root / "artifacts"
            result = managed(
                [str(prefix / "bin/python"), str(source / "scripts/smoke_isaac51.py"), "--backend", "taxim", "--output-dir", str(smoke_output), "--headless"],
                cwd=source, output_root=run_root, name="smoke",
                timeout_seconds=float(config["timeouts_seconds"][timeout_key]), environment=environment,
            )
            log_text = Path(result["log_path"]).read_text(encoding="utf-8", errors="replace")
            passed = process_ok(result) and "PASS backend=taxim" in log_text
            record = {"gate_id": gate_id, "backend": "taxim", "passed": passed, "process": result}
            manifest["runs"][gate_id] = record
            write_json(run_root / "result.json", record)
            write_json(root / "run_manifest.json", manifest)
            if not passed:
                manifest["classification"] = "isaac51_taxim_smoke_failed"
                raise RuntimeError(f"{gate_id} official Taxim smoke failed")
        manifest["status"] = "completed"
        manifest["classification"] = "isaac51_taxim_smoke_validated"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["classification"] = manifest["classification"] or "isaac51_taxim_smoke_failed"
        manifest["failure"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_json(root / "failure.json", manifest["failure"])
        raise
    finally:
        cleanup = managed_cleanup_summary(root)
        write_json(root / "final_cleanup.json", cleanup)
        manifest["cleanup_complete"] = cleanup["all_records_cleanup_complete"] and not cleanup["cleanup_residual"]
        if not manifest["cleanup_complete"]:
            manifest["status"] = "failed"
            manifest["classification"] = "cleanup_incomplete"
        manifest["ended_at"] = utc_now()
        write_json(root / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
