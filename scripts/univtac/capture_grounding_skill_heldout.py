#!/usr/bin/env python3
"""Capture three held-out Pull Out Key tactile pairs without running an Agent."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.univtac.run_codex_readonly_observation import _utc_now
from scripts.univtac.run_tactile_difference_pilot import _load_json, _run_seed_capture
from sim.envs.univtac.tactile_difference_pilot import build_pair_artifacts
from sim.envs.univtac.trace import write_json

SEEDS = (1_000_003, 1_000_004, 1_000_005)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = yaml.safe_load(args.config.expanduser().resolve(strict=True).read_text())
    if not isinstance(config, dict) or tuple(config.get("seeds", ())) != SEEDS:
        raise ValueError(f"held-out capture seeds must be exactly {list(SEEDS)}")
    output = args.output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output root must be fresh: {output}")
    runtime_python = args.runtime_python.expanduser().resolve(strict=True)
    source_root = args.source_root.expanduser().resolve(strict=True)
    base_gate = yaml.safe_load(
        (REPO_ROOT / str(config["gate_config"])).resolve(strict=True).read_text()
    )
    output.mkdir(parents=True)
    pilot: dict[str, Any] = {
        "round": "R0.9.21",
        "task": "pull_out_key",
        "seeds": list(SEEDS),
        "status": "capturing",
        "started_at": _utc_now(),
        "skill_name": config["skill_name"],
        "skill_version": config["skill_version"],
        "simulator_invocation_count": 0,
        "reset_requested_count": 0,
        "play_once_count": 0,
        "agent_action_count": 0,
    }
    write_json(output / "pilot.json", pilot)
    sources: list[dict[str, Any]] = []
    pairs: dict[str, Any] = {}
    errors = []
    for seed in SEEDS:
        pilot["simulator_invocation_count"] += 1
        pilot["reset_requested_count"] += 1
        write_json(output / "pilot.json", pilot)
        try:
            source = _run_seed_capture(
                seed=seed,
                base_gate=base_gate,
                runtime_python=runtime_python,
                source_root=source_root,
                output_root=output,
                headless=bool(config["headless"]),
            )
        except Exception as exc:  # noqa: BLE001 - continue the fixed held-out seed set
            source = {"seed": seed, "valid": False, "error": str(exc)}
            errors.append(
                {
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        sources.append(source)
        if source.get("valid"):
            baseline = _load_json(Path(source["baseline_snapshot"]))
            current = _load_json(Path(source["current_snapshot"]))
            pairs[str(seed)] = build_pair_artifacts(
                seed=seed,
                baseline_snapshot=baseline,
                current_snapshot=current,
                simulator_root=Path(source["simulator_root"]),
                output_root=output,
            )
    write_json(
        output / "capture_summary.json",
        {
            "sources": sources,
            "valid_seed_count": sum(bool(source.get("valid")) for source in sources),
            "simulator_invocation_count": 3,
            "reset_requested_count": 3,
            "play_once_count": 0,
            "agent_action_count": 0,
            "errors": errors,
        },
    )
    write_json(output / "pair_artifacts.json", {"seeds": pairs})
    pilot.update(
        {
            "status": "captured" if len(pairs) == 3 else "capture_incomplete",
            "ended_at": _utc_now(),
            "valid_seed_count": len(pairs),
        }
    )
    write_json(output / "pilot.json", pilot)
    print(json.dumps(_load_json(output / "capture_summary.json"), indent=2))
    return 0 if len(pairs) == 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
