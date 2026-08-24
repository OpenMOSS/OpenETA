"""Extract direct-vs-waypoint motion metrics from a durable eval run."""

from __future__ import annotations

import argparse
import json

from agent.evals.motion_route_ablation import extract_motion_route_ablation
from agent.evals.store import DEFAULT_EVALUATION_ROOT, EvaluationRunStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", default=str(DEFAULT_EVALUATION_ROOT))
    args = parser.parse_args(argv)
    store = EvaluationRunStore(args.run_id, root=args.root)
    if not store.exists():
        parser.error(f"unknown evaluation run: {args.run_id}")
    print(
        json.dumps(
            extract_motion_route_ablation(store),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
