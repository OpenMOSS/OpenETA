#!/usr/bin/env python3
"""Materialize and validate the R0.9.5.1 derived all-wheel closure."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim.envs.univtac.resumable_download_contract import (
    ALLOWED_REMOTE_HOSTS,
    closed_wheelhouse,
    curl_download_command,
    disk_space_gate,
    expected_total,
    parse_headers,
    sha256_file,
    sidecar_matches,
    validate_wheel,
    validate_lock_pair,
)
from sim.envs.univtac.artifact_lock_contract import validate_config as validate_wheelhouse_config


class WheelhouseError(RuntimeError):
    def __init__(self, classification: str, message: str):
        super().__init__(message)
        self.classification = classification


def write_json(path: Path, payload: object, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode or 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if mode is not None:
        path.chmod(mode)


def run_curl(command: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)


def probe_remote(record: dict, root: Path) -> dict:
    headers_path = root / f"{record['index']:03d}-{record['filename']}.head.headers"
    headers_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["curl", "--head", "--location", "--fail", "--silent", "--show-error", "--http1.1", "--proto", "=https", "--proto-redir", "=https", "--connect-timeout", "60", "--max-time", "300", "--dump-header", str(headers_path), "--output", "/dev/null", "--write-out", "%{http_code}\n%{url_effective}\n", record["url"]]
    result = run_curl(command, timeout=360)
    if result.returncode:
        raise WheelhouseError("blocked_by_external_resources", f"remote HEAD failed: {record['filename']}: {result.stderr.strip()}")
    parsed = parse_headers(headers_path.read_text(encoding="utf-8", errors="replace"))
    lines = result.stdout.strip().splitlines()
    effective_url = lines[-1] if lines else record["url"]
    effective_host = urlparse(effective_url).hostname
    if urlparse(record["url"]).hostname not in ALLOWED_REMOTE_HOSTS or effective_host not in ALLOWED_REMOTE_HOSTS:
        raise WheelhouseError("remote_artifact_identity_changed", f"unapproved effective host: {effective_host}")
    parsed.update({"source_url": record["url"], "source_host": record["origin_host"], "effective_url": effective_url, "effective_host": effective_host, "tls_verification": "enabled_and_passed", "head_command": command})
    size = parsed.get("content_length")
    if not size or size > 256 * 1024 * 1024:
        range_headers = root / f"{record['index']:03d}-{record['filename']}.range.headers"
        range_command = ["curl", "--location", "--fail", "--silent", "--show-error", "--http1.1", "--proto", "=https", "--proto-redir", "=https", "--connect-timeout", "60", "--max-time", "300", "--range", "0-0", "--max-filesize", "1024", "--dump-header", str(range_headers), "--output", "/dev/null", "--write-out", "%{http_code}\n%{url_effective}\n", record["url"]]
        ranged = run_curl(range_command, timeout=360)
        if ranged.returncode:
            raise WheelhouseError("blocked_by_external_resources", f"remote range probe failed: {record['filename']}: {ranged.stderr.strip()}")
        range_parsed = parse_headers(range_headers.read_text(encoding="utf-8", errors="replace"))
        if range_parsed["http_status"] != 206 or range_parsed["range_start"] != 0 or range_parsed["range_end"] != 0 or not range_parsed["content_range_total"]:
            raise WheelhouseError("remote_artifact_identity_changed", f"range identity is invalid: {record['filename']}")
        parsed.update({key: value for key, value in range_parsed.items() if value is not None})
        parsed["range_command"] = range_command
    parsed["expected_total"] = expected_total(parsed)
    return parsed


def copy_local(record: dict, destination: Path) -> int:
    source = Path(unquote(urlparse(record["url"]).path))
    if not source.is_file() or sha256_file(source) != record["sha256"]:
        raise WheelhouseError("wheel_artifact_hash_mismatch", f"local wheel changed: {record['filename']}")
    shutil.copyfile(source, destination)
    destination.chmod(0o444)
    return destination.stat().st_size


def copy_source_input(record: dict, source_path: Path, target: Path) -> None:
    if not source_path.is_file():
        raise WheelhouseError("wheel_artifact_hash_mismatch", f"locked source input is missing: {record['filename']}")
    if not target.exists():
        shutil.copyfile(source_path, target)
    if not target.is_file() or sha256_file(target) != record["sha256"]:
        raise WheelhouseError("wheel_artifact_hash_mismatch", f"source input changed: {record['filename']}")
    target.chmod(0o444)


def download_remote(record: dict, metadata: dict, cache: Path, config: dict, completed: Path) -> tuple[dict, list[dict]]:
    artifacts = cache / "artifacts"
    partial_root = cache / "partial"
    quarantine = cache / "quarantined"
    final = artifacts / record["filename"]
    total = expected_total(metadata)
    if final.is_file():
        try:
            validation = validate_wheel(final, record, total)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise WheelhouseError(
                "wheel_artifact_hash_mismatch",
                f"cached wheel validation failed: {record['filename']}: {exc}",
            ) from exc
        return {**validation, "disposition": "reused"}, []
    part = partial_root / f"{record['filename']}.part"
    sidecar_path = partial_root / f"{record['filename']}.part.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.is_file() else None
    if part.exists() != bool(sidecar):
        raise WheelhouseError("remote_artifact_identity_changed", f"partial/sidecar pair is incomplete: {record['filename']}")
    if sidecar and not sidecar_matches(sidecar, record, metadata):
        raise WheelhouseError("remote_artifact_identity_changed", f"partial identity changed: {record['filename']}")
    if part.exists() and part.stat().st_size >= total:
        raise WheelhouseError("remote_artifact_identity_changed", f"partial size is invalid: {record['filename']}")
    if not sidecar:
        sidecar = {"schema_version": "openeta.univtac.wheel_partial.v1", "url": record["url"], "filename": record["filename"], "sha256": record["sha256"], "expected_size": total, "etag": metadata.get("etag"), "attempts": []}
        write_json(sidecar_path, sidecar, mode=0o600)
    attempts = []
    maximum = int(config["max_transport_attempts_per_artifact"])
    for index in range(len(sidecar.get("attempts", [])) + 1, maximum + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = partial_root / f"{record['filename']}.attempt-{index}.headers"
        command = curl_download_command(url=record["url"], output=part, headers=headers, offset=offset, config=config["download"])
        started = time.monotonic()
        result = run_curl(command, timeout=int(config["download"]["max_time_seconds"]) + 60)
        ended = part.stat().st_size if part.exists() else 0
        parsed = None
        try:
            parsed = parse_headers(headers.read_text(encoding="utf-8", errors="replace")) if headers.is_file() else None
        except ValueError:
            parsed = None
        write_lines = result.stdout.strip().splitlines()
        effective_url = write_lines[-3] if len(write_lines) >= 4 else record["url"]
        if urlparse(effective_url).hostname not in ALLOWED_REMOTE_HOSTS:
            raise WheelhouseError("remote_artifact_identity_changed", f"download redirected to unapproved host: {record['filename']}")
        if parsed and metadata.get("etag") and parsed.get("etag") != metadata.get("etag"):
            raise WheelhouseError("remote_artifact_identity_changed", f"download ETag changed: {record['filename']}")
        if offset and parsed and (parsed.get("http_status") != 206 or parsed.get("range_start") != offset):
            raise WheelhouseError("remote_artifact_identity_changed", f"resume range changed: {record['filename']}")
        record_attempt = {"attempt_index": index, "start_offset": offset, "end_offset": ended, "bytes_gained": ended - offset, "curl_returncode": result.returncode, "elapsed_seconds": time.monotonic() - started, "error": result.stderr.strip() or None, "resumable": ended > 0 and ended < total, "http": parsed}
        attempts.append(record_attempt)
        sidecar.setdefault("attempts", []).append(record_attempt)
        sidecar["current_bytes"] = ended
        write_json(sidecar_path, sidecar, mode=0o600)
        if ended > total:
            raise WheelhouseError("remote_artifact_identity_changed", f"partial exceeded expected size: {record['filename']}")
        if result.returncode == 0 and ended == total:
            break
        if index < maximum:
            time.sleep(int(config["download"]["retry_wait_seconds"]))
    if not part.is_file() or part.stat().st_size != total:
        raise WheelhouseError("wheel_download_attempts_exhausted", f"download incomplete after {maximum} attempts: {record['filename']}")
    digest = sha256_file(part)
    if digest != record["sha256"]:
        quarantine.mkdir(parents=True, exist_ok=True)
        quarantined = quarantine / f"{record['filename']}.sha256-mismatch"
        os.replace(part, quarantined)
        raise WheelhouseError("wheel_artifact_hash_mismatch", f"downloaded wheel SHA mismatch: {record['filename']}")
    validation = validate_wheel(part, record, total)
    with part.open("rb") as handle:
        os.fsync(handle.fileno())
    part.chmod(0o444)
    os.replace(part, final)
    directory = os.open(artifacts, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    with completed.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"filename": final.name, "size": total, "sha256": digest, "attempts": len(sidecar["attempts"])}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    sidecar_path.unlink()
    for header in partial_root.glob(f"{record['filename']}.attempt-*.headers"):
        header.unlink()
    return {**validation, "disposition": "resumed" if attempts and attempts[0]["start_offset"] else "fresh"}, attempts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--derived-lock", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--legacy-bridge-root", type=Path, required=True)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--wheel-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    derived = json.loads(args.derived_lock.read_text(encoding="utf-8"))
    source = json.loads(args.source_lock.read_text(encoding="utf-8"))
    validate_wheelhouse_config(config)
    validate_lock_pair(source, derived)
    cache = args.wheel_cache_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError("wheelhouse output must be fresh")
    output.mkdir(parents=True, mode=0o750)
    marker_payload = {"schema_version": "openeta.univtac.r095_wheelhouse_cache.v1", "derived_install_artifact_lock_sha256": derived["derived_install_artifact_lock_sha256"], "source_artifact_lock_sha256": source["artifact_lock_sha256"], "owner_uid": os.getuid(), "mode": "0700"}
    if cache.exists():
        if cache.stat().st_uid != os.getuid() or (cache.stat().st_mode & 0o777) != 0o700:
            raise PermissionError("existing wheel cache is not private and user-owned")
        existing_marker = json.loads((cache / ".univtac-r095-wheelhouse.json").read_text(encoding="utf-8"))
        if existing_marker != marker_payload:
            raise WheelhouseError("remote_artifact_identity_changed", "wheel cache provenance marker changed")
        required_directories = {"source_inputs", "derived_artifacts", "artifacts", "partial", "quarantined", "manifests"}
        if any(not (cache / name).is_dir() for name in required_directories):
            raise WheelhouseError("remote_artifact_identity_changed", "wheel cache directory layout is incomplete")
    else:
        cache.mkdir(parents=True, mode=0o700)
        cache.chmod(0o700)
        for name in ("source_inputs", "derived_artifacts", "artifacts", "partial", "quarantined", "manifests"):
            (cache / name).mkdir(mode=0o700)
    state = {"status": "running", "classification": None, "stages": {stage: "not_run_due_to_gate" for stage in ("L1", "L2", "D0", "D1", "D2")}, "transport_attempts": 0}
    write_json(output / "run_manifest.json", state)
    lock_handle = (cache / ".download.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (cache / ".univtac-r095-wheelhouse.json").exists():
            write_json(cache / ".univtac-r095-wheelhouse.json", marker_payload, mode=0o600)
        if not (cache / "manifests/derived_install_artifact_lock.private.json").exists():
            shutil.copyfile(args.derived_lock, cache / "manifests/derived_install_artifact_lock.private.json")
            (cache / "manifests/derived_install_artifact_lock.private.json").chmod(0o600)
        state["current_stage"] = "L1"
        metadata = {}
        for record in derived["records"]:
            if record["url_scheme"] == "https":
                metadata[record["filename"]] = probe_remote(record, output / "remote_metadata/private")
            else:
                local = Path(unquote(urlparse(record["url"]).path))
                metadata[record["filename"]] = {"expected_total": local.stat().st_size, "content_length": local.stat().st_size, "source_url": record["url"], "source_host": None, "effective_url": record["url"], "effective_host": None, "tls_verification": "not_applicable_local"}
        write_json(cache / "manifests/remote_metadata.private.json", metadata, mode=0o600)
        public_metadata = {name: {key: value for key, value in item.items() if key not in {"source_url", "effective_url", "head_command", "range_command"}} for name, item in metadata.items()}
        write_json(output / "remote_metadata/summary.json", public_metadata)
        state["stages"]["L1"] = "passed"
        for record in source["records"]:
            if record["artifact_type"] == "sdist":
                source_path = args.legacy_bridge_root / "legacy_sdists" / record["name"] / record["filename"]
                target = cache / "source_inputs" / record["filename"]
                copy_source_input(record, source_path, target)
        state["current_stage"] = "L2"
        sizes = {name: expected_total(item) for name, item in metadata.items()}
        total = sum(sizes.values())
        disk = disk_space_gate(total_bytes=total, remaining_bytes=total, largest_bytes=max(sizes.values()), cache_root=cache, environment_root=args.environment_root.resolve())
        write_json(output / "disk_space/summary.json", disk)
        if not disk["passed"]:
            raise WheelhouseError("insufficient_space_for_closed_wheelhouse", "disk space gate failed")
        state["stages"]["L2"] = "passed"

        state["current_stage"] = "D0"
        completed = cache / "manifests/completed.jsonl"
        validations = []
        all_attempts = []
        for record in derived["records"]:
            destination = cache / "artifacts" / record["filename"]
            if record["url_scheme"] == "file":
                if destination.is_file():
                    size = destination.stat().st_size
                    disposition = "reused"
                else:
                    size = copy_local(record, destination)
                    disposition = "local_copy"
                validation = validate_wheel(destination, record, size)
                validations.append({**validation, "disposition": disposition, "transport_attempts": 0})
            else:
                validation, attempts = download_remote(record, metadata[record["filename"]], cache, config, completed)
                validations.append({**validation, "transport_attempts": len(attempts)})
                all_attempts.extend({"filename": record["filename"], **item} for item in attempts)
                state["transport_attempts"] = len(all_attempts)
                write_json(output / "run_manifest.json", state)
        state["stages"]["D0"] = "passed"
        state["stages"]["D1"] = "passed"
        state["current_stage"] = "D2"
        write_json(output / "download_summary/artifacts.json", validations)
        write_json(output / "download_summary/attempts.json", all_attempts)
        closed = closed_wheelhouse(derived["records"], cache / "artifacts", sizes)
        write_json(output / "wheelhouse_validation/closed_wheelhouse.json", closed)
        if not closed["success"]:
            raise WheelhouseError("wheelhouse_validation_failed", "closed wheelhouse validation failed")
        state["stages"]["D2"] = "passed"
        state["status"] = "completed"
        state["classification"] = "closed_wheelhouse_validated"
        state["planned_total_bytes"] = total
        state["largest_artifact_bytes"] = max(sizes.values())
        state["wheelhouse_manifest_sha256"] = closed["wheelhouse_manifest_sha256"]
    except WheelhouseError as exc:
        current_stage = state.get("current_stage")
        if current_stage in state["stages"] and state["stages"][current_stage] == "not_run_due_to_gate":
            state["stages"][current_stage] = "failed"
        state["status"] = "failed"
        state["classification"] = exc.classification
        state["failure"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        write_json(output / "run_manifest.json", state)
        write_json(output / "summary.json", state)
        lock_handle.close()



if __name__ == "__main__":
    main()
