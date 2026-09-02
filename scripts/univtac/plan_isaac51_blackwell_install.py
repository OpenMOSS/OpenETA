#!/usr/bin/env python3
"""Print the immutable R0.9 environment clone and simulator installation plan."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sim.envs.univtac.simulator_install_contract import clone_command, constraints_text, validate_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--conda-exe", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--curobo-checkout", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    conda_base = subprocess.run(
        [str(args.conda_exe), "info", "--base"], check=True, capture_output=True, text=True
    ).stdout.strip()
    target_prefix = Path(conda_base) / "envs" / config["environment"]["conda_name"]
    python = target_prefix / "bin/python"
    constraints = args.output_root.resolve() / "install/protected_constraints.txt"
    payload = {
        "runtime_variant": config["runtime_variant"],
        **config["claims"],
        "source_checkout": str(args.source_checkout.resolve()),
        "curobo_checkout": str(args.curobo_checkout.resolve()),
        "output_root": str(args.output_root.resolve()),
        "target_prefix": str(target_prefix),
        "target_exists": target_prefix.exists(),
        "gate_order": config["gate_order"],
        "clone_command": clone_command(
            args.conda_exe, config["environment"]["clone_from"], config["environment"]["conda_name"]
        ),
        "constraints_path": str(constraints),
        "constraints_content": constraints_text(config).splitlines(),
        "isaac_dry_run_command": [
            str(python), "-m", "pip", "install", "--dry-run", "--report", str(args.output_root.resolve() / "install/isaac_dry_run_report.json"),
            "--constraint", str(constraints), config["install"]["isaaclab_requirement"],
            "--extra-index-url", config["install"]["isaac_extra_index_url"],
        ],
        "official_install_script_executed": False,
        "sysctl_modified": False,
        "isaac_started": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
