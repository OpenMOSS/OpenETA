#!/usr/bin/env python3
"""Fetch, verify, and exercise the fixed official filelock 3.13.1 wheel."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import yaml
from packaging.markers import default_environment
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.univtac.prepare_isaaclab_build_tool_wheels import verify_wheel

HELPER_PATH = ROOT / "sim/envs/univtac/filelock_compatibility.py"
SPEC = importlib.util.spec_from_file_location("univtac_filelock_compatibility", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(HELPER_PATH)
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)
private_directory_record = HELPER.private_directory_record
validate_config = HELPER.validate_config


def fetch(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "OpenETA-R0.9.4"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "OpenETA-R0.9.4"})
    with urllib.request.urlopen(request, timeout=120) as response:
        path.write_bytes(response.read())


HOLDER = r"""
import pathlib, sys, time
from filelock import FileLock, SoftFileLock
kind, lock_path, ready_path, release_path = sys.argv[1:]
cls = FileLock if kind == 'file' else SoftFileLock
with cls(lock_path):
    pathlib.Path(ready_path).touch()
    while not pathlib.Path(release_path).exists():
        time.sleep(0.02)
"""

CONTENDER = r"""
import json, sys
from filelock import FileLock, SoftFileLock, Timeout
kind, lock_path, timeout_text = sys.argv[1:]
cls = FileLock if kind == 'file' else SoftFileLock
try:
    with cls(lock_path, timeout=float(timeout_text)):
        outcome = 'acquired'
except Timeout:
    outcome = 'timeout'
print(json.dumps({'kind': kind, 'outcome': outcome, 'timeout_is_exception': issubclass(Timeout, Exception)}))
"""


def _wait_for(path: Path, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise TimeoutError(f"holder did not become ready: {path.name}")


def exercise_lock(python: Path, environment: dict[str, str], root: Path, kind: str) -> dict:
    lock_path = root / f"{kind}.lock"
    ready_path = root / f"{kind}.ready"
    release_path = root / f"{kind}.release"
    holder = subprocess.Popen(
        [str(python), "-c", HOLDER, kind, str(lock_path), str(ready_path), str(release_path)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for(ready_path, timeout=10)
        blocked = subprocess.run(
            [str(python), "-c", CONTENDER, kind, str(lock_path), "0.2"],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if blocked.returncode:
            raise RuntimeError(blocked.stderr)
        blocked_payload = json.loads(blocked.stdout)
        if blocked_payload != {"kind": kind, "outcome": "timeout", "timeout_is_exception": True}:
            raise ValueError(f"{kind} contender did not time out")
        release_path.touch()
        stdout, stderr = holder.communicate(timeout=10)
        if holder.returncode:
            raise RuntimeError(stderr)
        acquired = subprocess.run(
            [str(python), "-c", CONTENDER, kind, str(lock_path), "1.0"],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if acquired.returncode:
            raise RuntimeError(acquired.stderr)
        acquired_payload = json.loads(acquired.stdout)
        if acquired_payload != {"kind": kind, "outcome": "acquired", "timeout_is_exception": True}:
            raise ValueError(f"{kind} contender did not acquire after release")
        return {
            "kind": kind,
            "held_contention": blocked_payload,
            "post_release": acquired_payload,
            "holder_stdout": stdout,
        }
    finally:
        if holder.poll() is None:
            holder.terminate()
            try:
                holder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                holder.kill()
                holder.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    spec = config["wheel"]
    release = fetch(spec["pypi_json_url"])
    candidates = [
        item for item in release["urls"]
        if item["filename"] == spec["filename"] and item["packagetype"] == "bdist_wheel"
    ]
    if len(candidates) != 1:
        raise ValueError("expected exactly one official filelock wheel")
    candidate = candidates[0]
    if (
        urlparse(candidate["url"]).hostname != config["allowed_host"]
        or candidate.get("yanked")
        or candidate["size"] != spec["size"]
        or candidate["digests"]["sha256"] != spec["sha256"]
    ):
        raise ValueError("official filelock release identity mismatch")
    wheel = output / "wheelhouse" / spec["filename"]
    wheel.parent.mkdir(parents=True)
    download(candidate["url"], wheel)
    validation = verify_wheel(
        wheel,
        distribution="filelock",
        version=str(spec["version"]),
        size=int(spec["size"]),
        sha256=spec["sha256"],
    )
    marker_environment = default_environment()
    marker_environment["extra"] = ""
    active_requirements = [
        raw
        for raw in validation["requires_dist"]
        if Requirement(raw).marker is None or Requirement(raw).marker.evaluate(marker_environment)
    ]
    if active_requirements:
        raise ValueError("filelock 3.13.1 wheel unexpectedly has active runtime dependencies")
    with tempfile.TemporaryDirectory(prefix="univtac-filelock3131-") as temporary:
        private_root = Path(temporary)
        private_root.chmod(0o700)
        virtualenv = private_root / "venv"
        create = subprocess.run(
            [str(args.target_python), "-m", "venv", str(virtualenv)],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
            timeout=120,
        )
        if create.returncode:
            raise RuntimeError(create.stderr)
        isolated_python = virtualenv / "bin/python"
        install = subprocess.run(
            [str(isolated_python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
            timeout=120,
        )
        if install.returncode:
            raise RuntimeError(install.stderr)
        environment = {**os.environ, "PYTHONNOUSERSITE": "1"}
        import_probe = subprocess.run(
            [str(isolated_python), "-c", "import filelock, json; from filelock import FileLock, SoftFileLock, Timeout; print(json.dumps({'version': filelock.__version__, 'path': filelock.__file__, 'types': [FileLock.__name__, SoftFileLock.__name__, Timeout.__name__]}))"],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=30,
        )
        if import_probe.returncode:
            raise RuntimeError(import_probe.stderr)
        import_witness = json.loads(import_probe.stdout)
        if import_witness["version"] != str(spec["version"]) or not Path(import_witness["path"]).is_relative_to(virtualenv):
            raise ValueError("isolated filelock import identity mismatch")
        lock_root = private_root / "locks"
        lock_root.mkdir(mode=0o700)
        security = private_directory_record(lock_root)
        if not security["private"] or not security["owned_by_current_user"]:
            raise PermissionError("filelock smoke directory is not private and user-owned")
        lock_smoke = {kind: exercise_lock(isolated_python, environment, lock_root, kind) for kind in ("file", "soft")}
    if Path(temporary).exists():
        raise RuntimeError("isolated filelock smoke temporary directory was not removed")
    payload = {
        "schema_version": "openeta.univtac.filelock3131_wheel_preparation.v1",
        "success": True,
        "pypi": {key: candidate.get(key) for key in ("filename", "url", "size", "digests", "upload_time_iso_8601", "requires_python", "yanked")},
        "wheel": validation,
        "active_runtime_requirements": active_requirements,
        "wheel_path": str(wheel),
        "isolated_import": import_witness,
        "isolation_method": "fresh_venv",
        "lock_smoke": lock_smoke,
        "security": security,
        "temporary_root_removed": True,
        "exploit_or_symlink_attack_tested": False,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
