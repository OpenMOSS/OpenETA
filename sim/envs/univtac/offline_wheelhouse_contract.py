"""Pure contracts for the R0.9.5.2 all-wheel closure and offline plan."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


SCHEMA_VERSION = "openeta.univtac.isaac51_offline_wheelhouse_r0952.v1"
EXPECTED_HEAD = "6d9da7c49944ef57ee4319e6d4bcc3dba4db7978"
EXPECTED_SOURCE_REPORT_SHA256 = "081e2f2a27a0252cdb305abdaf9fb00add6557f42626aa04f91f8a8702f19e4b"
EXPECTED_SOURCE_LOCK_SHA256 = "816691fad4acfcf3d4e450f40c41858ec9ab75f9dd7de3233cf40547dbf95f7f"
EXPECTED_TRANSFORM_LOCK_SHA256 = "b2e9f58ddd17298c80ab95203a680f8e7553250da80698e3c4458bff0e29f01c"
EXPECTED_DERIVED_LOCK_SHA256 = "2be906ebd28183dee3621df59f12dcdd8266f253e8f45d597fdf1e7a37976fd8"
EXPECTED_TRANSFORMS = {"idna-ssl", "pyperclip", "antlr4-python3-runtime"}
PROTECTED_DISTRIBUTIONS = {
    "torch", "torchvision", "warp-lang", "pyuipc", "nvidia-curobo",
    "setuptools", "setuptools-scm", "wheel", "packaging", "filelock",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "installation_method_label": "isaac51_hash_locked_offline_wheelhouse_v1",
        "environment": "UniVTAC-isaac51-sm120-r09",
        "openeta_head": EXPECTED_HEAD,
        "requirement": "isaaclab[isaacsim,all]==2.3.0",
        "source_report_sha256": EXPECTED_SOURCE_REPORT_SHA256,
        "source_artifact_lock_sha256": EXPECTED_SOURCE_LOCK_SHA256,
        "transformation_lock_sha256": EXPECTED_TRANSFORM_LOCK_SHA256,
        "derived_install_lock_sha256": EXPECTED_DERIVED_LOCK_SHA256,
        "cache_mode": "0700",
        "artifact_mode": "0444",
        "large_artifact_threshold_bytes": 256 * 1024 * 1024,
        "canary_range_attempts": 3,
        "metadata_range_attempts": 3,
        "max_transport_attempts_per_artifact": 20,
        "retry_wait_seconds": 30,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("R0.9.5.2 configuration identity changed")
    if tuple(config.get("allowed_remote_hosts", ())) != (
        "files.pythonhosted.org", "pypi.nvidia.com"
    ):
        raise ValueError("remote host allowlist changed")
    if config.get("install_accounting") != {
        "prior_online_invocations": 1,
        "authorized_offline_retry_invocations": 0,
        "total_invocations": 1,
        "third_install_invocation": False,
    }:
        raise ValueError("R0.9.5.2 install accounting changed")
    if set(config.get("source_inputs", {})) != EXPECTED_TRANSFORMS:
        raise ValueError("source input set changed")
    if set(config.get("derived_wheels", {})) != EXPECTED_TRANSFORMS:
        raise ValueError("derived wheel set changed")
    if config.get("networkx_canary") != {
        "index": 92,
        "filename": "networkx-3.3-py3-none-any.whl",
        "expected_size": 1702396,
    }:
        raise ValueError("networkx canary changed")
    if config.get("nvidia_canary") != {
        "index": 75,
        "filename": "isaacsim_rl-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl",
        "expected_size": 17268,
        "selection": "minimum_observed_size_then_record_index_from_preserved_metadata_probe",
    }:
        raise ValueError("NVIDIA canary changed")


def classify_records(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    remote = [record for record in records if record.get("url_scheme") == "https"]
    local = [record for record in records if record.get("url_scheme") == "file"]
    derived = [record for record in local if record.get("source_transformation_sha256")]
    verified = [record for record in local if not record.get("source_transformation_sha256")]
    if (
        len(records) != 173
        or any(record.get("artifact_type") != "wheel" or not str(record.get("filename", "")).endswith(".whl") for record in records)
        or len(remote) != 169
        or len(derived) != 3
        or len(verified) != 1
        or {record["name"] for record in derived} != EXPECTED_TRANSFORMS
        or {record["name"] for record in verified} != {"flatdict"}
    ):
        raise ValueError("derived closure is not 169 remote + 3 derived + 1 verified local")
    return {"remote": remote, "derived": derived, "verified_local": verified}


def select_canaries(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    by_index = {int(record["index"]): record for record in records}
    networkx = by_index.get(int(config["networkx_canary"]["index"]))
    nvidia = by_index.get(int(config["nvidia_canary"]["index"]))
    if not networkx or networkx.get("filename") != config["networkx_canary"]["filename"]:
        raise ValueError("networkx canary identity changed")
    if networkx.get("origin_host") != "files.pythonhosted.org":
        raise ValueError("networkx canary host changed")
    if not nvidia or nvidia.get("filename") != config["nvidia_canary"]["filename"]:
        raise ValueError("NVIDIA canary identity changed")
    if nvidia.get("origin_host") != "pypi.nvidia.com":
        raise ValueError("NVIDIA canary host changed")
    return {"pythonhost": networkx, "nvidia": nvidia}


def verify_nvidia_canary_minimum(
    records: Sequence[Mapping[str, Any]], metadata: Mapping[str, Mapping[str, Any]], canary: Mapping[str, Any]
) -> dict[str, Any]:
    candidates = [record for record in records if record.get("origin_host") == "pypi.nvidia.com"]
    ranked = sorted(
        candidates,
        key=lambda record: (int(metadata[record["filename"]]["expected_total"]), int(record["index"])),
    )
    selected = ranked[0]
    return {
        "passed": selected["filename"] == canary["filename"],
        "selected_filename": canary["filename"],
        "observed_minimum_filename": selected["filename"],
        "observed_minimum_size": metadata[selected["filename"]]["expected_total"],
        "tie_break_index": selected["index"],
    }


def validate_cached_metadata(
    records: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    allowed_hosts: set[str],
) -> dict[str, Any]:
    expected = {record["filename"]: record for record in records}
    if set(metadata) != set(expected):
        raise ValueError("cached metadata record set changed")
    method_counts: dict[str, int] = {}
    for filename, record in expected.items():
        item = metadata[filename]
        total = item.get("expected_total")
        if not isinstance(total, int) or total <= 0:
            raise ValueError(f"cached metadata size is invalid: {filename}")
        method = item.get("metadata_method")
        method_counts[str(method)] = method_counts.get(str(method), 0) + 1
        if record.get("url_scheme") == "https":
            effective = urlparse(str(item.get("effective_url", "")))
            if (
                item.get("source_url") != record.get("url")
                or item.get("source_host") != record.get("origin_host")
                or effective.scheme != "https"
                or effective.hostname not in allowed_hosts
                or Path(unquote(effective.path)).name != filename
                or item.get("tls_verification") != "enabled_and_passed"
                or method not in {"head_succeeded", "head_failed_range_succeeded", "head_large_range_succeeded"}
            ):
                raise ValueError(f"cached remote metadata identity changed: {filename}")
        elif record.get("url_scheme") == "file":
            path = Path(unquote(urlparse(record["url"]).path))
            if (
                method != "local_locked_file"
                or not path.is_file()
                or path.stat().st_size != total
                or sha256_file(path) != record["sha256"]
            ):
                raise ValueError(f"cached local metadata identity changed: {filename}")
        else:
            raise ValueError(f"cached metadata has unsupported URL scheme: {filename}")
    return {"passed": True, "record_count": len(expected), "method_counts": method_counts}


def validate_system_curl_provenance(
    *,
    executable: Path,
    version_output: str,
    ldd_output: str,
    conda_prefixes: Sequence[Path],
) -> dict[str, Any]:
    realpath = executable.resolve()
    combined = f"{version_output}\n{ldd_output}"
    if realpath != Path("/usr/bin/curl"):
        raise ValueError("system curl realpath changed")
    if "no version information available" in combined or "not found" in ldd_output:
        raise ValueError("system curl loader provenance is invalid")
    forbidden = [str(prefix.resolve()) for prefix in conda_prefixes]
    forbidden.extend(["/anaconda", "/miniconda", "/conda/envs/"])
    if any(fragment in ldd_output for fragment in forbidden):
        raise ValueError("system curl loaded a Conda library")
    library_paths = []
    for line in ldd_output.splitlines():
        if "=>" not in line:
            continue
        candidate = line.split("=>", 1)[1].strip().split()[0]
        if candidate.startswith("/"):
            library_paths.append(candidate)
    if not library_paths or any(not path.startswith(("/lib/", "/usr/lib/")) for path in library_paths):
        raise ValueError("system curl loaded a non-system library")
    first_line = version_output.splitlines()[0] if version_output.splitlines() else ""
    if not first_line.startswith("curl "):
        raise ValueError("system curl version output is invalid")
    ssl_backend = next(
        (token for token in first_line.split() if token.startswith(("OpenSSL/", "GnuTLS/"))),
        None,
    )
    if ssl_backend is None:
        raise ValueError("system curl TLS backend is unavailable")
    return {
        "passed": True,
        "curl_executable": str(executable),
        "curl_realpath": str(realpath),
        "curl_version_output": version_output,
        "curl_ssl_backend": ssl_backend,
        "library_paths": library_paths,
    }


def validate_existing_cache(
    cache: Path,
    source_lock: Mapping[str, Any],
    derived_lock: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if not cache.is_dir() or cache.stat().st_uid != os.getuid():
        raise ValueError("wheel cache is absent or not user-owned")
    if stat.S_IMODE(cache.stat().st_mode) != 0o700:
        raise ValueError("wheel cache mode changed")
    directories = {"source_inputs", "derived_artifacts", "artifacts", "partial", "quarantined", "manifests"}
    root_entries = {path.name for path in cache.iterdir()}
    allowed_root = directories | {".univtac-r095-wheelhouse.json", ".download.lock"}
    if root_entries != allowed_root or any(not (cache / name).is_dir() for name in directories):
        raise ValueError("wheel cache layout contains missing or extra entries")
    if any(stat.S_IMODE((cache / name).stat().st_mode) != 0o700 for name in directories):
        raise ValueError("wheel cache directory mode changed")
    marker_path = cache / ".univtac-r095-wheelhouse.json"
    if stat.S_IMODE(marker_path.stat().st_mode) != 0o600:
        raise ValueError("wheel cache provenance marker mode changed")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker != {
        "schema_version": "openeta.univtac.r095_wheelhouse_cache.v1",
        "derived_install_artifact_lock_sha256": derived_lock["derived_install_artifact_lock_sha256"],
        "source_artifact_lock_sha256": source_lock["artifact_lock_sha256"],
        "owner_uid": os.getuid(),
        "mode": "0700",
    }:
        raise ValueError("wheel cache provenance marker changed")
    expected_sources = {item["filename"]: item for item in config["source_inputs"].values()}
    actual_sources = {path.name: path for path in (cache / "source_inputs").iterdir()}
    source_results = []
    if set(actual_sources) != set(expected_sources):
        raise ValueError("existing source input set changed")
    for filename, expected in sorted(expected_sources.items()):
        path = actual_sources[filename]
        mode = stat.S_IMODE(path.stat().st_mode)
        digest = sha256_file(path)
        if not path.is_file() or path.stat().st_size != expected["size"] or digest != expected["sha256"] or mode != 0o444:
            raise ValueError(f"existing source input identity changed: {filename}")
        source_results.append(
            {"filename": filename, "size": path.stat().st_size, "sha256": digest, "mode": f"{mode:04o}"}
        )
    manifest_files = {path.name for path in (cache / "manifests").iterdir()}
    allowed_manifests = {
        "derived_install_artifact_lock.private.json",
        "remote_metadata.private.json",
        "completed.jsonl",
    }
    if "derived_install_artifact_lock.private.json" not in manifest_files or not manifest_files.issubset(allowed_manifests):
        raise ValueError("existing cache manifest set changed")
    cached_lock = cache / "manifests/derived_install_artifact_lock.private.json"
    if stat.S_IMODE(cached_lock.stat().st_mode) != 0o600 or json.loads(cached_lock.read_text(encoding="utf-8")) != derived_lock:
        raise ValueError("cached derived lock changed")
    records = {record["filename"]: record for record in derived_lock.get("records", [])}
    transformed = {record["filename"] for record in derived_lock.get("records", []) if record.get("source_transformation_sha256")}
    artifact_results = []
    for directory_name in ("artifacts", "derived_artifacts"):
        for path in sorted((cache / directory_name).iterdir()):
            record = records.get(path.name)
            if not path.is_file() or record is None or (directory_name == "derived_artifacts" and path.name not in transformed):
                raise ValueError(f"unexpected preexisting cache artifact: {directory_name}/{path.name}")
            digest = sha256_file(path)
            mode = stat.S_IMODE(path.stat().st_mode)
            if digest != record["sha256"] or mode != 0o444:
                raise ValueError(f"preexisting wheel identity changed: {path.name}")
            artifact_results.append({"directory": directory_name, "filename": path.name, "size": path.stat().st_size, "sha256": digest, "mode": f"{mode:04o}"})
    partial_root = cache / "partial"
    partial_files = {path.name: path for path in partial_root.iterdir()}
    part_names = {name[:-5] for name in partial_files if name.endswith(".part")}
    sidecar_names = {name[:-10] for name in partial_files if name.endswith(".part.json")}
    if part_names != sidecar_names:
        raise ValueError("partial/sidecar set is incomplete")
    partial_results = []
    allowed_partial_files = set()
    for filename in sorted(part_names):
        record = records.get(filename)
        if record is None or record.get("url_scheme") != "https":
            raise ValueError(f"unknown partial artifact: {filename}")
        part = partial_files[f"{filename}.part"]
        sidecar_path = partial_files[f"{filename}.part.json"]
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        expected = {
            "url": record["url"],
            "filename": filename,
            "sha256": record["sha256"],
        }
        if any(sidecar.get(key) != value for key, value in expected.items()):
            raise ValueError(f"partial artifact identity changed: {filename}")
        expected_size = sidecar.get("expected_size")
        if not isinstance(expected_size, int) or expected_size <= 0 or part.stat().st_size > expected_size:
            raise ValueError(f"partial artifact size is invalid: {filename}")
        complete_hash_valid = part.stat().st_size == expected_size and sha256_file(part) == record["sha256"]
        if part.stat().st_size == expected_size and not complete_hash_valid:
            raise ValueError(f"complete partial hash changed: {filename}")
        allowed_partial_files.update({f"{filename}.part", f"{filename}.part.json"})
        header_pattern = re.compile(re.escape(filename) + r"\.attempt-\d+\.headers")
        headers = sorted(name for name in partial_files if header_pattern.fullmatch(name))
        allowed_partial_files.update(headers)
        partial_results.append({"filename": filename, "current_bytes": part.stat().st_size, "expected_size": expected_size, "complete_hash_valid": complete_hash_valid, "attempt_count": len(sidecar.get("attempts", [])), "headers": headers})
    if set(partial_files) != allowed_partial_files:
        raise ValueError("partial directory contains unknown files")
    if any((cache / "quarantined").iterdir()):
        raise ValueError("quarantined cache is not empty")
    remote_artifacts = [item for item in artifact_results if item["directory"] == "artifacts" and records[item["filename"]].get("url_scheme") == "https"]
    cache_state = "source_inputs_written_before_final_gate_order_fix" if not artifact_results and not partial_results else "verified_source_inputs_plus_resumable_remote_state"
    return {
        "passed": True,
        "cache_original_state": cache_state,
        "source_input_identity": "revalidated_against_source_lock",
        "remote_wheel_artifacts_present": bool(remote_artifacts),
        "partial_remote_wheels_present": bool(partial_results),
        "combined_status": "accepted_existing_verified_source_inputs" if not artifact_results and not partial_results else "accepted_existing_verified_source_inputs_and_resumable_state",
        "root": str(cache),
        "mode": "0700",
        "source_inputs": source_results,
        "artifacts": artifact_results,
        "partials": partial_results,
    }


def offline_dry_run_command(
    python: Path, wheelhouse: Path, constraints: Path, report: Path, requirement: str
) -> list[str]:
    return [
        str(python), "-m", "pip", "install", "--dry-run", "--report", str(report),
        "--no-index", "--no-cache-dir", "--find-links", str(wheelhouse),
        "--only-binary=:all:", "--constraint", str(constraints), requirement,
    ]


def validate_offline_command(command: Sequence[str]) -> None:
    required = {"--dry-run", "--report", "--no-index", "--no-cache-dir", "--find-links", "--only-binary=:all:", "--constraint", "isaaclab[isaacsim,all]==2.3.0"}
    prohibited = {"--no-deps", "--upgrade", "--ignore-installed", "--no-build-isolation"}
    if not required.issubset(command) or prohibited.intersection(command):
        raise ValueError("offline dry-run command contract changed")


def _normalize_metadata(item: Mapping[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata", {})
    requirements = []
    for raw in metadata.get("requires_dist", []) or []:
        requirements.append(str(Requirement(str(raw))))
    return {
        "name": canonicalize_name(str(metadata.get("name", ""))),
        "version": str(metadata.get("version", "")),
        "requires_python": metadata.get("requires_python"),
        "requires_dist": sorted(requirements),
        "provides_extra": sorted(metadata.get("provides_extra", []) or []),
        "requested": bool(item.get("requested")),
    }


def compare_offline_resolution(
    source_report: Mapping[str, Any],
    derived_lock: Mapping[str, Any],
    offline_report: Mapping[str, Any],
) -> dict[str, Any]:
    source_items = {_normalize_metadata(item)["name"]: item for item in source_report.get("install", [])}
    offline_items = {_normalize_metadata(item)["name"]: item for item in offline_report.get("install", [])}
    records = {record["name"]: record for record in derived_lock.get("records", [])}
    failures = []
    if not (len(source_items) == len(offline_items) == len(records) == 173):
        failures.append("record_count_or_duplicate_mismatch")
    if set(source_items) != set(offline_items) or set(source_items) != set(records):
        failures.append("dependency_closure_changed")
    transformed = {name for name, record in records.items() if record.get("source_transformation_sha256")}
    if transformed != EXPECTED_TRANSFORMS:
        failures.append("transformation_set_changed")
    artifact_differences = []
    metadata_differences = []
    for name in sorted(set(source_items) & set(offline_items) & set(records)):
        source_item = source_items[name]
        offline_item = offline_items[name]
        source_metadata = _normalize_metadata(source_item)
        offline_metadata = _normalize_metadata(offline_item)
        if source_metadata != offline_metadata:
            metadata_differences.append(name)
        download = offline_item.get("download_info", {})
        url = str(download.get("url", ""))
        filename = Path(unquote(urlparse(url).path)).name
        digest = download.get("archive_info", {}).get("hashes", {}).get("sha256")
        record = records[name]
        if urlparse(url).scheme != "file" or filename != record["filename"] or digest != record["sha256"] or not filename.endswith(".whl"):
            artifact_differences.append(name)
    planned_protected = sorted(set(offline_items) & PROTECTED_DISTRIBUTIONS)
    if planned_protected:
        failures.append("protected_packages_would_change")
    if "vcs-versioning" in offline_items:
        failures.append("vcs_versioning_present")
    if metadata_differences:
        failures.append("metadata_changed")
    if artifact_differences:
        failures.append("artifact_mapping_changed")
    return {
        "success": not failures,
        "source_record_count": len(source_items),
        "derived_record_count": len(records),
        "offline_record_count": len(offline_items),
        "transformed_packages": sorted(transformed),
        "metadata_differences": metadata_differences,
        "artifact_differences": artifact_differences,
        "protected_packages_planned_for_change": planned_protected,
        "offline_url_schemes": sorted({urlparse(str(item.get("download_info", {}).get("url", ""))).scheme for item in offline_items.values()}),
        "failures": sorted(set(failures)),
    }
