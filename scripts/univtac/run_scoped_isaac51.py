#!/usr/bin/env python3
"""Run an exact argv through the repository-scoped Isaac51 launcher."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.envs.univtac.scoped_isaac51_launcher import (
    ScopedIsaac51LaunchSpec,
    build_scoped_isaac51_dry_run,
    resolve_real_libcuda_driver,
    run_scoped_isaac51_command,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("child_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.child_argv[:1] == ["--"]:
        args.child_argv = args.child_argv[1:]
    if not args.child_argv:
        parser.error("an exact child argv is required after --")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = ScopedIsaac51LaunchSpec(
        python_executable=args.python.resolve(),
        command=tuple(args.child_argv),
        cwd=args.cwd.resolve(),
        output_root=args.output_root.resolve(),
        timeout_seconds=args.timeout,
    )
    if args.dry_run:
        probe = resolve_real_libcuda_driver(python_executable=spec.python_executable)
        print(json.dumps(build_scoped_isaac51_dry_run(spec, probe), indent=2, sort_keys=True))
        return 0
    result = run_scoped_isaac51_command(spec)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
