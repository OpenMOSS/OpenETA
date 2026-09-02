#!/usr/bin/env python3
"""Exercise the installed filelock package in a fresh private directory."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.univtac.prepare_filelock3131_wheel import exercise_lock, private_directory_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.work_root.resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True, mode=0o700)
    root.chmod(0o700)
    environment = {**os.environ, "PYTHONNOUSERSITE": "1"}
    probe = subprocess.run(
        [sys.executable, "-c", "import filelock, json; from filelock import FileLock, SoftFileLock, Timeout; print(json.dumps({'version': filelock.__version__, 'path': filelock.__file__, 'types': [FileLock.__name__, SoftFileLock.__name__, Timeout.__name__]}))"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    if probe.returncode:
        raise RuntimeError(probe.stderr)
    import_witness = json.loads(probe.stdout)
    if import_witness["version"] != "3.13.1":
        raise ValueError("installed filelock version mismatch")
    locks = {kind: exercise_lock(Path(sys.executable), environment, root, kind) for kind in ("file", "soft")}
    payload = {
        "schema_version": "openeta.univtac.installed_filelock_smoke.v1",
        "success": True,
        "distribution_version": importlib.metadata.version("filelock"),
        "import": import_witness,
        "private_directory": private_directory_record(root),
        "lock_smoke": locks,
        "exploit_or_symlink_attack_tested": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
