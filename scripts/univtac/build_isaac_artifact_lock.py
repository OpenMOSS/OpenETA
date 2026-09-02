#!/usr/bin/env python3
"""Build the R0.9.5 artifact lock from the accepted R0.9.4 P0A report."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml

ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "sim/envs/univtac/artifact_lock_contract.py"
SPEC = importlib.util.spec_from_file_location("univtac_artifact_lock_contract", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(HELPER_PATH)
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)
build_artifact_lock = HELPER.build_artifact_lock
deterministic_sha256 = HELPER.deterministic_sha256
failure_classification = HELPER.failure_classification
public_summary = HELPER.public_summary
sha256_file = HELPER.sha256_file
validate_config = HELPER.validate_config
validate_p0a_process = HELPER.validate_p0a_process


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--p0a-report", type=Path, required=True)
    parser.add_argument("--p0a-process", type=Path, required=True)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, mode=0o750)
    output.chmod(0o750)
    report = json.loads(args.p0a_report.read_text(encoding="utf-8"))
    if report.get("version") != "1" or report.get("pip_version") != config["source_report"]["pip_version"]:
        raise ValueError("P0A report identity changed")
    process = json.loads(args.p0a_process.read_text(encoding="utf-8"))
    validate_p0a_process(
        process,
        report_path=args.p0a_report,
        constraints_path=args.constraints,
        requirement=config["source_report"]["requirement"],
    )
    lock = build_artifact_lock(report, allowed_hosts=set(config["allowed_remote_hosts"]))
    flatdict = [record for record in lock["records"] if record["name"] == "flatdict"]
    if len(flatdict) != 1 or flatdict[0]["url_scheme"] != "file" or flatdict[0]["sha256"] != config["flatdict"]["sha256"]:
        raise ValueError("P0A flatdict artifact identity changed")
    flatdict_path = Path(unquote(urlparse(flatdict[0]["url"]).path))
    if not flatdict_path.is_file() or sha256_file(flatdict_path) != config["flatdict"]["sha256"]:
        raise ValueError("verified local flatdict wheel is missing or changed")
    lock["source_report"] = {
        "path": str(args.p0a_report.resolve()),
        "sha256": sha256_file(args.p0a_report),
        "version": report["version"],
        "pip_version": report["pip_version"],
        "requirement": config["source_report"]["requirement"],
        "constraints_path": str(args.constraints.resolve()),
        "constraints_sha256": sha256_file(args.constraints),
    }
    lock["artifact_lock_sha256"] = deterministic_sha256(lock)
    private = output / "artifact_lock.private.json"
    descriptor = os.open(private, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    summary = public_summary(lock)
    summary["artifact_lock_sha256"] = lock["artifact_lock_sha256"]
    summary["classification"] = failure_classification(lock)
    (output / "artifact_lock.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"success": lock["success"], "classification": summary["classification"], "record_count": lock["record_count"], "failures": lock["failures"]}, sort_keys=True))
    raise SystemExit(0 if lock["success"] else 2)


if __name__ == "__main__":
    main()
