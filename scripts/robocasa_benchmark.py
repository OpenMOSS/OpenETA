"""CLI for reproducible RoboCasa365 manifests and resumable evaluations."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from sim.robocasa_benchmark import (
    RoboCasaBenchmarkManifest,
    RoboCasaRolloutResult,
    RoboCasaScenario,
    aggregate_parallel_batch_results,
    aggregate_results,
    build_manifest,
    evaluate_manifest,
    list_task_sets,
)


def _load_runner(
    spec: str,
) -> Callable[[RoboCasaScenario], RoboCasaRolloutResult | Mapping[str, Any]]:
    if spec == "noop":
        return _noop_rollout
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("runner must be 'noop' or 'python.module:callable'")
    runner = getattr(importlib.import_module(module_name), attribute)
    if not callable(runner):
        raise TypeError(f"Runner {spec!r} is not callable")
    return runner


def _noop_rollout(scenario: RoboCasaScenario) -> RoboCasaRolloutResult:
    """Physics/integration baseline that deliberately performs no task policy."""

    import gymnasium as gym
    from sim.env_registry import hot_activate

    hot_activate("robocasa")
    env = gym.make(
        scenario.env_id,
        seed=scenario.seed,
        render_mode="rgb_array",
    )
    steps = 0
    success = False
    try:
        _obs, info = env.reset(seed=scenario.seed)
        success = bool(info.get("success", False))
        action = np.zeros(12, dtype=np.float32)
        action[11] = -1.0
        while not success and steps < scenario.horizon:
            _obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            success = bool(info.get("success", False) or reward > 0)
            if terminated or truncated:
                break
    finally:
        env.close()
    return RoboCasaRolloutResult(
        scenario_id=scenario.scenario_id,
        task=scenario.task,
        split=scenario.split,
        seed=scenario.seed,
        success=success,
        steps=steps,
        metadata={"runner": "noop"},
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenETA RoboCasa365 benchmark harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("task-sets", help="list official RoboCasa task sets")

    manifest = subparsers.add_parser("manifest", help="create a deterministic scenario manifest")
    manifest.add_argument("--task-set", default="all_tasks")
    manifest.add_argument("--split", choices=("pretrain", "target"), default="target")
    manifest.add_argument("--scenarios-per-task", type=int, default=50)
    manifest.add_argument("--seed", type=int, default=0)
    manifest.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate", help="run or resume a manifest")
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("--runner", default="noop", help="noop or python.module:callable")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--no-resume", action="store_true")
    evaluate.add_argument("--fail-fast", action="store_true")
    evaluate.add_argument("--max-rollouts", type=int)

    parallel_manifest = subparsers.add_parser(
        "parallel-manifest",
        help="adapt a RoboCasa manifest for the shared openeta-batch harness",
    )
    parallel_manifest.add_argument("manifest", type=Path)
    parallel_manifest.add_argument("--output", type=Path, required=True)
    parallel_manifest.add_argument("--max-turns", type=int)
    parallel_manifest.add_argument("--max-tool-calls", type=int)
    parallel_manifest.add_argument("--timeout-s", type=float)
    parallel_manifest.add_argument("--max-total-tokens", type=int)

    parallel_summary = subparsers.add_parser(
        "parallel-summary",
        help="convert an openeta-batch v2 result to the RoboCasa result schema",
    )
    parallel_summary.add_argument("manifest", type=Path)
    parallel_summary.add_argument("batch_results", type=Path)
    parallel_summary.add_argument("--output", type=Path)

    summary = subparsers.add_parser("summary", help="recompute and print a result summary")
    summary.add_argument("manifest", type=Path)
    summary.add_argument("results", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "task-sets":
        print(json.dumps(list_task_sets(), indent=2, sort_keys=True))
        return 0
    if args.command == "manifest":
        manifest = build_manifest(
            args.task_set,
            args.split,
            scenarios_per_task=args.scenarios_per_task,
            master_seed=args.seed,
        )
        manifest.write_json(args.output)
        print(json.dumps({
            "output": str(args.output),
            "task_count": manifest.task_count,
            "rollout_count": manifest.rollout_count,
            "manifest_sha256": manifest.to_dict()["manifest_sha256"],
        }, indent=2))
        return 0
    if args.command == "evaluate":
        manifest = RoboCasaBenchmarkManifest.read_json(args.manifest)
        result = evaluate_manifest(
            manifest,
            _load_runner(args.runner),
            output_path=args.output,
            resume=not args.no_resume,
            fail_fast=args.fail_fast,
            max_rollouts=args.max_rollouts,
        )
        print(json.dumps({key: result[key] for key in (
            "completed_rollouts", "expected_rollouts", "success_rate", "complete"
        )}, indent=2))
        return 0
    if args.command == "parallel-manifest":
        manifest = RoboCasaBenchmarkManifest.read_json(args.manifest)
        episode_limits = {
            key: value
            for key, value in {
                "max_turns": args.max_turns,
                "max_tool_calls": args.max_tool_calls,
                "timeout_s": args.timeout_s,
                "max_total_tokens": args.max_total_tokens,
            }.items()
            if value is not None
        }
        manifest.write_parallel_json(
            args.output,
            episode_limits=episode_limits,
        )
        print(json.dumps({
            "output": str(args.output),
            "episode_count": manifest.rollout_count,
            "source_manifest_sha256": manifest.to_dict()["manifest_sha256"],
            "require_official_reward": True,
        }, indent=2))
        return 0
    if args.command == "parallel-summary":
        manifest = RoboCasaBenchmarkManifest.read_json(args.manifest)
        batch_payload = json.loads(args.batch_results.read_text(encoding="utf-8"))
        result = aggregate_parallel_batch_results(manifest, batch_payload)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    manifest = RoboCasaBenchmarkManifest.read_json(args.manifest)
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    results = [RoboCasaRolloutResult.from_dict(item) for item in payload["rollouts"]]
    print(json.dumps(aggregate_results(manifest, results), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
