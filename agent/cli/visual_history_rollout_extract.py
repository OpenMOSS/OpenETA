"""Extract visual-history-specific metrics from a completed eval run."""

from __future__ import annotations

import argparse
import json

from agent.evals.store import DEFAULT_EVALUATION_ROOT, EvaluationRunStore
from agent.evals.visual_history_rollout import extract_visual_history_rollouts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract ABC visual-history metrics from durable rollout bundles."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", default=str(DEFAULT_EVALUATION_ROOT))
    args = parser.parse_args(argv)
    store = EvaluationRunStore(args.run_id, root=args.root)
    if not store.exists():
        parser.error(f"unknown evaluation run: {args.run_id}")
    report = extract_visual_history_rollouts(store)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
