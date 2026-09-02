#!/usr/bin/env python3
"""Materialize and validate the R0.9.5.2 hash-locked all-wheel closure."""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import time
import traceback
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_helper(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DOWNLOAD_CONTRACT = _load_helper(
    "univtac_resumable_download_contract",
    "sim/envs/univtac/resumable_download_contract.py",
)
OFFLINE_CONTRACT = _load_helper(
    "univtac_offline_wheelhouse_contract",
    "sim/envs/univtac/offline_wheelhouse_contract.py",
)
ALLOWED_REMOTE_HOSTS = DOWNLOAD_CONTRACT.ALLOWED_REMOTE_HOSTS
closed_wheelhouse = DOWNLOAD_CONTRACT.closed_wheelhouse
curl_download_command = DOWNLOAD_CONTRACT.curl_download_command
disk_space_gate = DOWNLOAD_CONTRACT.disk_space_gate
expected_total = DOWNLOAD_CONTRACT.expected_total
parse_headers = DOWNLOAD_CONTRACT.parse_headers
sha256_file = DOWNLOAD_CONTRACT.sha256_file
sidecar_matches = DOWNLOAD_CONTRACT.sidecar_matches
validate_wheel = DOWNLOAD_CONTRACT.validate_wheel
validate_lock_pair = DOWNLOAD_CONTRACT.validate_lock_pair
classify_records = OFFLINE_CONTRACT.classify_records
select_canaries = OFFLINE_CONTRACT.select_canaries
validate_r0952_config = OFFLINE_CONTRACT.validate_config
validate_cached_metadata = OFFLINE_CONTRACT.validate_cached_metadata
validate_existing_cache = OFFLINE_CONTRACT.validate_existing_cache
verify_nvidia_canary_minimum = OFFLINE_CONTRACT.verify_nvidia_canary_minimum
SYSTEM_CURL = Path("/usr/bin/curl")


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
    if not command or Path(command[0]) != SYSTEM_CURL:
        raise ValueError("transport must use the fixed system curl executable")
    environment = dict(os.environ)
    environment.pop("LD_LIBRARY_PATH", None)
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            command,
            124,
            exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            f"TimeoutExpired after {timeout}s",
        )


def _probe_command(record: dict, headers: Path, config: dict, *, method: str) -> list[str]:
    command = [
        str(SYSTEM_CURL), "--location", "--fail", "--silent", "--show-error", "--http1.1",
        "--proto", "=https", "--proto-redir", "=https",
        "--connect-timeout", str(config["probe"]["connect_timeout_seconds"]),
        "--max-time", str(config["probe"]["max_time_seconds"]),
        "--dump-header", str(headers), "--output", "/dev/null",
        "--write-out", "%{http_code}\n%{url_effective}\n%{size_download}\n%{time_total}\n",
    ]
    if method == "head":
        command.append("--head")
    elif method == "range":
        command.extend(["--range", "0-0", "--max-filesize", "1024"])
    else:
        raise ValueError(f"unsupported probe method: {method}")
    command.append(record["url"])
    return command


def _effective_url(result: subprocess.CompletedProcess[str], record: dict) -> str:
    lines = result.stdout.strip().splitlines()
    return lines[-3] if len(lines) >= 4 else record["url"]


def _validate_effective_url(record: dict, effective_url: str) -> None:
    source = urlparse(record["url"])
    effective = urlparse(effective_url)
    filename = Path(unquote(effective.path)).name
    if (
        source.scheme != "https"
        or source.hostname not in ALLOWED_REMOTE_HOSTS
        or effective.scheme != "https"
        or effective.hostname not in ALLOWED_REMOTE_HOSTS
        or filename != record["filename"]
    ):
        raise WheelhouseError(
            "remote_artifact_identity_changed",
            f"remote effective URL identity changed: {record['filename']}",
        )


def _probe_attempt(record: dict, root: Path, config: dict, *, method: str, attempt: int) -> dict:
    headers = root / f"{record['index']:03d}-{record['filename']}.{method}-{attempt}.headers"
    headers.parent.mkdir(parents=True, exist_ok=True)
    command = _probe_command(record, headers, config, method=method)
    started = time.monotonic()
    result = run_curl(command, timeout=int(config["probe"]["max_time_seconds"]) + 60)
    effective = _effective_url(result, record)
    _validate_effective_url(record, effective)
    parsed = None
    try:
        parsed = parse_headers(headers.read_text(encoding="utf-8", errors="replace")) if headers.is_file() else None
    except ValueError:
        parsed = None
    return {
        "method": method,
        "attempt": attempt,
        "returncode": result.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "effective_url": effective,
        "effective_host": urlparse(effective).hostname,
        "error": result.stderr.strip() or None,
        "http": parsed,
        "command": command,
        "transport_environment": {"LD_LIBRARY_PATH": "unset", "system_curl": str(SYSTEM_CURL)},
    }


def probe_remote(record: dict, root: Path, config: dict, *, classification: str) -> dict:
    attempts = [_probe_attempt(record, root, config, method="head", attempt=1)]
    head = attempts[0]
    head_http = head.get("http") or {}
    head_size = head_http.get("content_length") if head["returncode"] == 0 else None
    threshold = int(config["large_artifact_threshold_bytes"])
    if isinstance(head_size, int) and 0 < head_size <= threshold:
        return {
            **head_http,
            "expected_total": head_size,
            "source_url": record["url"],
            "source_host": record["origin_host"],
            "effective_url": head["effective_url"],
            "effective_host": head["effective_host"],
            "tls_verification": "enabled_and_passed",
            "metadata_method": "head_succeeded",
            "attempts": attempts,
        }
    range_limit = int(config["metadata_range_attempts"])
    for index in range(1, range_limit + 1):
        ranged = _probe_attempt(record, root, config, method="range", attempt=index)
        attempts.append(ranged)
        parsed = ranged.get("http") or {}
        if (
            ranged["returncode"] == 0
            and parsed.get("http_status") == 206
            and parsed.get("range_start") == 0
            and parsed.get("range_end") == 0
            and isinstance(parsed.get("content_range_total"), int)
            and parsed["content_range_total"] > 0
        ):
            return {
                **parsed,
                "expected_total": parsed["content_range_total"],
                "source_url": record["url"],
                "source_host": record["origin_host"],
                "effective_url": ranged["effective_url"],
                "effective_host": ranged["effective_host"],
                "tls_verification": "enabled_and_passed",
                "metadata_method": "head_failed_range_succeeded" if head["returncode"] else "head_large_range_succeeded",
                "attempts": attempts,
            }
        if index < range_limit:
            time.sleep(int(config["retry_wait_seconds"]))
    detail = attempts[-1].get("error") or "no valid Content-Range"
    raise WheelhouseError(classification, f"metadata probes exhausted: {record['filename']}: {detail}")


def copy_local(record: dict, destination: Path, *, source: Path | None = None) -> int:
    source = source or Path(unquote(urlparse(record["url"]).path))
    if not source.is_file() or sha256_file(source) != record["sha256"]:
        raise WheelhouseError("wheel_artifact_hash_mismatch", f"local wheel changed: {record['filename']}")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise WheelhouseError("wheelhouse_validation_failed", f"stale local-copy temporary exists: {temporary.name}")
    shutil.copyfile(source, temporary)
    if sha256_file(temporary) != record["sha256"]:
        raise WheelhouseError("wheel_artifact_hash_mismatch", f"copied local wheel changed: {record['filename']}")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.chmod(0o444)
    os.replace(temporary, destination)
    directory = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
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
    if part.exists() and part.stat().st_size > total:
        raise WheelhouseError("remote_artifact_identity_changed", f"partial size is invalid: {record['filename']}")
    if not sidecar:
        sidecar = {"schema_version": "openeta.univtac.wheel_partial.v1", "url": record["url"], "filename": record["filename"], "sha256": record["sha256"], "expected_size": total, "etag": metadata.get("etag"), "attempts": []}
        write_json(sidecar_path, sidecar, mode=0o600)
    historical_attempts = len(sidecar.get("attempts", []))
    recovered_complete_part = part.exists() and part.stat().st_size == total
    attempts = []
    maximum = int(config["max_transport_attempts_per_artifact"])
    first_attempt = (
        maximum + 1
        if part.exists() and part.stat().st_size == total
        else len(sidecar.get("attempts", [])) + 1
    )
    for index in range(first_attempt, maximum + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = partial_root / f"{record['filename']}.attempt-{index}.headers"
        command = curl_download_command(url=record["url"], output=part, headers=headers, offset=offset, config=config["download"])
        command[0] = str(SYSTEM_CURL)
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
        effective = urlparse(effective_url)
        if effective.scheme != "https" or effective.hostname not in ALLOWED_REMOTE_HOSTS:
            raise WheelhouseError("remote_artifact_identity_changed", f"download redirected to unapproved host: {record['filename']}")
        if parsed and metadata.get("etag") and parsed.get("etag") != metadata.get("etag"):
            raise WheelhouseError("remote_artifact_identity_changed", f"download ETag changed: {record['filename']}")
        if parsed and parsed.get("content_range_total") not in (None, total):
            raise WheelhouseError("remote_artifact_identity_changed", f"download total changed: {record['filename']}")
        if offset and parsed and (parsed.get("http_status") != 206 or parsed.get("range_start") != offset):
            raise WheelhouseError("remote_artifact_identity_changed", f"resume range changed: {record['filename']}")
        record_attempt = {
            "artifact_index": record["index"],
            "distribution": record["name"],
            "version": record["version"],
            "attempt_index": index,
            "start_offset": offset,
            "end_offset": ended,
            "bytes_gained": ended - offset,
            "total_size": total,
            "curl_returncode": result.returncode,
            "elapsed_seconds": time.monotonic() - started,
            "effective_url": effective_url,
            "effective_host": effective.hostname,
            "tls_verification": "enabled",
            "error": result.stderr.strip() or None,
            "resumable": ended > 0 and ended < total,
            "http": parsed,
        }
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
    validation = validate_wheel(part, record, total, check_filename=False)
    with part.open("rb") as handle:
        os.fsync(handle.fileno())
    part.chmod(0o444)
    os.replace(part, final)
    directory = os.open(artifacts, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    validation = validate_wheel(final, record, total)
    with completed.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"filename": final.name, "size": total, "sha256": digest, "attempts": len(sidecar["attempts"])}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    sidecar_path.unlink()
    for header in partial_root.glob(f"{record['filename']}.attempt-*.headers"):
        header.unlink()
    disposition = "recovered_complete_part" if recovered_complete_part else "resumed" if attempts and attempts[0]["start_offset"] else "fresh"
    return {
        **validation,
        "disposition": disposition,
        "historical_transport_attempts": historical_attempts,
        "new_transport_attempts": len(attempts),
        "cumulative_transport_attempts": historical_attempts + len(attempts),
        "recovered_complete_part": recovered_complete_part,
    }, attempts


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
    validate_r0952_config(config)
    validate_lock_pair(source, derived)
    groups = classify_records(derived["records"])
    cache = args.wheel_cache_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError("wheelhouse output must be fresh")
    output.mkdir(parents=True, mode=0o750)
    state = {
        "status": "running",
        "classification": None,
        "stages": {stage: "not_run_due_to_gate" for stage in ("C0", "M0", "M1", "D0", "D1", "D2", "V0")},
        "remote_record_count": len(groups["remote"]),
        "derived_local_record_count": len(groups["derived"]),
        "verified_local_record_count": len(groups["verified_local"]),
        "transport_attempts": 0,
        "actual_install_invocations": 0,
    }
    write_json(output / "run_manifest.json", state)
    marker_payload = {
        "schema_version": "openeta.univtac.r095_wheelhouse_cache.v1",
        "derived_install_artifact_lock_sha256": derived["derived_install_artifact_lock_sha256"],
        "source_artifact_lock_sha256": source["artifact_lock_sha256"],
        "owner_uid": os.getuid(),
        "mode": "0700",
    }
    lock_handle = None
    try:
        if not cache.is_dir() or cache.stat().st_uid != os.getuid() or (cache.stat().st_mode & 0o777) != 0o700:
            raise WheelhouseError("existing_cache_identity_failed", "existing wheel cache is not private and user-owned")
        existing_marker = json.loads((cache / ".univtac-r095-wheelhouse.json").read_text(encoding="utf-8"))
        if existing_marker != marker_payload:
            raise WheelhouseError("existing_cache_identity_failed", "wheel cache provenance marker changed")
        required_directories = {"source_inputs", "derived_artifacts", "artifacts", "partial", "quarantined", "manifests"}
        if any(not (cache / name).is_dir() for name in required_directories):
            raise WheelhouseError("existing_cache_identity_failed", "wheel cache directory layout is incomplete")
        lock_handle = (cache / ".download.lock").open("w")
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            validate_existing_cache(cache, source, derived, config)
        except (OSError, TypeError, ValueError) as exc:
            raise WheelhouseError("existing_cache_identity_failed", str(exc)) from exc

        state["current_stage"] = "C0"
        canary_records = select_canaries(groups["remote"], config)
        canary_metadata = {}
        for label, record in canary_records.items():
            canary_metadata[record["filename"]] = probe_remote(
                record,
                output / "transport_canary/private" / label,
                config,
                classification="network_transport_preflight_failed",
            )
            config_key = "networkx_canary" if label == "pythonhost" else "nvidia_canary"
            if canary_metadata[record["filename"]]["expected_total"] != int(config[config_key]["expected_size"]):
                raise WheelhouseError(
                    "remote_artifact_identity_changed",
                    f"canary size changed: {record['filename']}",
                )
        write_json(output / "transport_canary/private/results.json", canary_metadata, mode=0o600)
        write_json(
            output / "transport_canary/summary.json",
            {
                "passed": True,
                "records": [
                    {
                        "label": label,
                        "index": record["index"],
                        "name": record["name"],
                        "filename": record["filename"],
                        "origin_host": record["origin_host"],
                        "metadata_method": canary_metadata[record["filename"]]["metadata_method"],
                        "expected_total": canary_metadata[record["filename"]]["expected_total"],
                    }
                    for label, record in canary_records.items()
                ],
            },
        )
        state["stages"]["C0"] = "passed"
        write_json(output / "run_manifest.json", state)

        state["current_stage"] = "M0"
        cached_metadata_path = cache / "manifests/remote_metadata.private.json"
        metadata_cache_reused = cached_metadata_path.is_file()
        if metadata_cache_reused:
            metadata = json.loads(cached_metadata_path.read_text(encoding="utf-8"))
            try:
                validate_cached_metadata(derived["records"], metadata, ALLOWED_REMOTE_HOSTS)
            except (OSError, TypeError, ValueError) as exc:
                raise WheelhouseError("existing_cache_identity_failed", str(exc)) from exc
            metadata.update(canary_metadata)
            refreshed_large = []
            threshold = int(config["large_artifact_threshold_bytes"])
            for record in sorted(groups["remote"], key=lambda item: int(item["index"])):
                cached_total = int(metadata[record["filename"]]["expected_total"])
                if cached_total <= threshold:
                    continue
                refreshed = probe_remote(
                    record,
                    output / "remote_metadata/system_curl_large_refresh",
                    config,
                    classification="blocked_by_external_resources",
                )
                if refreshed["expected_total"] != cached_total:
                    raise WheelhouseError("remote_artifact_identity_changed", f"large artifact total changed: {record['filename']}")
                metadata[record["filename"]] = refreshed
                refreshed_large.append({"index": record["index"], "filename": record["filename"], "expected_total": cached_total, "metadata_method": refreshed["metadata_method"]})
        else:
            refreshed_large = []
            metadata = dict(canary_metadata)
            for record in sorted(groups["remote"], key=lambda item: int(item["index"])):
                if record["filename"] not in metadata:
                    metadata[record["filename"]] = probe_remote(
                        record,
                        output / "remote_metadata/private",
                        config,
                        classification="blocked_by_external_resources",
                    )
            for record in groups["derived"] + groups["verified_local"]:
                local = Path(unquote(urlparse(record["url"]).path))
                if not local.is_file() or sha256_file(local) != record["sha256"]:
                    raise WheelhouseError("wheel_artifact_hash_mismatch", f"local wheel changed: {record['filename']}")
                metadata[record["filename"]] = {
                    "expected_total": local.stat().st_size,
                    "content_length": local.stat().st_size,
                    "source_url": record["url"],
                    "source_host": None,
                    "effective_url": record["url"],
                    "effective_host": None,
                    "tls_verification": "not_applicable_local",
                    "metadata_method": "local_locked_file",
                    "attempts": [],
                }
        canary_minimum = verify_nvidia_canary_minimum(groups["remote"], metadata, canary_records["nvidia"])
        if not canary_minimum["passed"]:
            raise WheelhouseError("r0952_resume_precondition_failed", "NVIDIA canary was not the smallest observed locked wheel")
        write_json(cache / "manifests/remote_metadata.private.json", metadata, mode=0o600)
        public_metadata = {
            name: {key: value for key, value in item.items() if key not in {"source_url", "effective_url", "attempts"}}
            for name, item in metadata.items()
        }
        write_json(output / "remote_metadata/summary.json", public_metadata)
        method_counts: dict[str, int] = {}
        for item in metadata.values():
            method = item["metadata_method"]
            method_counts[method] = method_counts.get(method, 0) + 1
        write_json(output / "remote_metadata/method_counts.json", method_counts)
        write_json(output / "remote_metadata/cache_reuse.json", {"reused": metadata_cache_reused, "large_artifacts_refreshed_with_system_curl": refreshed_large})
        write_json(output / "transport_canary/nvidia_minimum_check.json", canary_minimum)
        state["stages"]["M0"] = "passed"
        write_json(output / "run_manifest.json", state)

        state["current_stage"] = "M1"
        sizes = {name: expected_total(item) for name, item in metadata.items()}
        total = sum(sizes.values())
        partial_bytes = sum(path.stat().st_size for path in (cache / "partial").glob("*.part"))
        existing_bytes = sum(path.stat().st_size for path in (cache / "artifacts").glob("*.whl"))
        remaining = max(0, total - partial_bytes - existing_bytes)
        disk = disk_space_gate(
            total_bytes=total,
            remaining_bytes=remaining,
            largest_bytes=max(sizes.values()),
            cache_root=cache,
            environment_root=args.environment_root.resolve(),
        )
        host_totals = {
            host: sum(sizes[record["filename"]] for record in groups["remote"] if record["origin_host"] == host)
            for host in sorted(ALLOWED_REMOTE_HOSTS)
        }
        plan = {
            "remote_wheel_count": len(groups["remote"]),
            "derived_local_wheel_count": len(groups["derived"]),
            "verified_local_wheel_count": len(groups["verified_local"]),
            "final_artifact_count": len(derived["records"]),
            "total_planned_bytes": total,
            "remaining_download_bytes": remaining,
            "largest_artifact_bytes": max(sizes.values()),
            "remote_bytes_by_host": host_totals,
            "disk": disk,
        }
        write_json(output / "download_summary/plan.json", plan)
        if not disk["passed"]:
            raise WheelhouseError("insufficient_space_for_closed_wheelhouse", "disk space gate failed")
        state["stages"]["M1"] = "passed"
        write_json(output / "run_manifest.json", state)

        state["current_stage"] = "D0"
        completed = cache / "manifests/completed.jsonl"
        validations = []
        all_attempts = []
        for record in sorted(groups["remote"], key=lambda item: int(item["index"])):
            validation, attempts = download_remote(record, metadata[record["filename"]], cache, config, completed)
            validations.append({**validation, "transport_attempts": len(attempts), "origin": "remote_locked"})
            all_attempts.extend({"filename": record["filename"], **item} for item in attempts)
            state["transport_attempts"] = len(all_attempts)
            state["completed_remote_artifacts"] = len(validations)
            state["cumulative_downloaded_bytes"] = sum(item["bytes_gained"] for item in all_attempts)
            write_json(output / "download_summary/artifacts.json", validations)
            write_json(output / "download_summary/attempts.json", all_attempts)
            write_json(output / "run_manifest.json", state)
        state["stages"]["D0"] = "passed"
        state["stages"]["D1"] = "passed"

        state["current_stage"] = "D2"
        for record in sorted(groups["derived"], key=lambda item: int(item["index"])):
            source_path = Path(unquote(urlparse(record["url"]).path))
            derived_copy = cache / "derived_artifacts" / record["filename"]
            if not derived_copy.exists():
                copy_local(record, derived_copy, source=source_path)
            validate_wheel(derived_copy, record, sizes[record["filename"]])
            destination = cache / "artifacts" / record["filename"]
            if not destination.exists():
                copy_local(record, destination, source=derived_copy)
            validation = validate_wheel(destination, record, sizes[record["filename"]])
            validations.append({**validation, "disposition": "local_derived", "transport_attempts": 0, "origin": "local_derived"})
        for record in groups["verified_local"]:
            destination = cache / "artifacts" / record["filename"]
            if not destination.exists():
                copy_local(record, destination)
            validation = validate_wheel(destination, record, sizes[record["filename"]])
            validations.append({**validation, "disposition": "local_verified", "transport_attempts": 0, "origin": "local_verified"})
        state["stages"]["D2"] = "passed"

        state["current_stage"] = "V0"
        final_validations = []
        for record in sorted(derived["records"], key=lambda item: int(item["index"])):
            final_validations.append(validate_wheel(cache / "artifacts" / record["filename"], record, sizes[record["filename"]]))
        closed = closed_wheelhouse(derived["records"], cache / "artifacts", sizes)
        closed.update(
            {
                "remote_locked_wheels": len(groups["remote"]),
                "derived_local_wheels": len(groups["derived"]),
                "flatdict_local_wheels": len(groups["verified_local"]),
                "total_sdists": 0,
                "duplicate_records": len(derived["records"]) - len({record["filename"] for record in derived["records"]}),
            }
        )
        write_json(output / "download_summary/artifacts.json", validations)
        write_json(output / "download_summary/attempts.json", all_attempts)
        write_json(output / "artifact_validation/all_wheels.json", final_validations)
        write_json(output / "closed_wheelhouse/closed_wheelhouse.json", closed)
        write_json(output / "closed_wheelhouse/wheelhouse_manifest.json", closed["manifest"])
        if not closed["success"]:
            raise WheelhouseError("wheelhouse_validation_failed", "closed wheelhouse validation failed")
        state["stages"]["V0"] = "passed"
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
    except BaseException as exc:
        current_stage = state.get("current_stage")
        if current_stage in state["stages"] and state["stages"][current_stage] == "not_run_due_to_gate":
            state["stages"][current_stage] = "failed"
        state["status"] = "failed"
        state["classification"] = "wheelhouse_validation_failed"
        state["failure"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        write_json(output / "run_manifest.json", state)
        write_json(output / "summary.json", state)
        if lock_handle is not None:
            lock_handle.close()



if __name__ == "__main__":
    main()
