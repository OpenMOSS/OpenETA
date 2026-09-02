"""Artifact-lock contract for the R0.9.5 offline Isaac wheelhouse."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename


SCHEMA_VERSION = "openeta.univtac.isaac51_offline_wheelhouse.v1"
INSTALLATION_METHOD_LABEL = "isaac51_hash_locked_offline_wheelhouse_v1"
FLATDICT_SHA256 = "94614060d175f1acbff62a4bca3051e5145f4ed9c880a36861e47533f82ba212"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("offline wheelhouse schema changed")
    if config.get("installation_method_label") != INSTALLATION_METHOD_LABEL:
        raise ValueError("offline wheelhouse label changed")
    if config.get("environment") != "UniVTAC-isaac51-sm120-r09":
        raise ValueError("offline wheelhouse may only target R0.9")
    source = config.get("source_report", {})
    if source.get("pip_version") != "26.2.1" or source.get("requirement") != "isaaclab[isaacsim,all]==2.3.0":
        raise ValueError("P0A source identity changed")
    if tuple(config.get("allowed_remote_hosts", ())) != ("files.pythonhosted.org", "pypi.nvidia.com"):
        raise ValueError("artifact origin allowlist changed")
    if config.get("flatdict") != {"version": "4.0.1", "sha256": FLATDICT_SHA256}:
        raise ValueError("flatdict artifact identity changed")
    if config.get("cache_mode") != "0700" or config.get("artifact_mode") != "0444":
        raise ValueError("wheelhouse permissions changed")
    if int(config.get("max_transport_attempts_per_artifact", 0)) != 20:
        raise ValueError("transport attempt limit changed")
    if config.get("install_accounting") != {
        "prior_online_invocations": 1,
        "authorized_offline_retry_invocations": 1,
        "total_authorized_invocations": 2,
        "additional_install_retry_allowed": False,
    }:
        raise ValueError("Isaac install accounting changed")


def _artifact_type(filename: str) -> str:
    return "wheel" if filename.endswith(".whl") else "sdist" if filename.endswith((".tar.gz", ".zip")) else "unknown"


def build_artifact_lock(report: Mapping[str, Any], *, allowed_hosts: set[str]) -> dict[str, Any]:
    installs = report.get("install")
    if not isinstance(installs, list):
        raise TypeError("pip report install field must be a list")
    compatible = set(cpython_tags(python_version=(3, 11))) | set(
        compatible_tags(python_version=(3, 11), interpreter="cp311")
    )
    records: list[dict[str, Any]] = []
    missing_sha: list[str] = []
    nonwheels: list[str] = []
    yanked: list[str] = []
    invalid_origins: list[str] = []
    incompatible_wheels: list[str] = []
    seen: set[tuple[str, str]] = set()
    duplicates: list[str] = []
    for index, item in enumerate(installs):
        metadata = item.get("metadata", {})
        name = canonicalize_name(str(metadata.get("name", "")))
        version = str(metadata.get("version", ""))
        identity = (name, version)
        if identity in seen:
            duplicates.append(f"{name}=={version}")
        seen.add(identity)
        download = item.get("download_info", {})
        url = str(download.get("url", ""))
        parsed = urlparse(url)
        filename = Path(unquote(parsed.path)).name
        digest = download.get("archive_info", {}).get("hashes", {}).get("sha256")
        artifact_type = _artifact_type(filename)
        if not digest:
            missing_sha.append(filename or f"record-{index}")
        if artifact_type != "wheel":
            nonwheels.append(filename)
        if item.get("is_yanked"):
            yanked.append(filename)
        if parsed.scheme == "https":
            if parsed.hostname not in allowed_hosts:
                invalid_origins.append(filename)
        elif parsed.scheme == "file":
            if name != "flatdict":
                invalid_origins.append(filename)
        else:
            invalid_origins.append(filename)
        tags: list[str] = []
        tag_compatible: bool | None = None
        if artifact_type == "wheel":
            try:
                wheel_name, wheel_version, _, wheel_tags = parse_wheel_filename(filename)
                tags = sorted(map(str, wheel_tags))
                tag_compatible = bool(compatible.intersection(wheel_tags))
                if canonicalize_name(str(wheel_name)) != name or str(wheel_version) != version or not tag_compatible:
                    incompatible_wheels.append(filename)
            except ValueError:
                tag_compatible = False
                incompatible_wheels.append(filename)
        requested_extras = ["all", "isaacsim"] if name == "isaaclab" and item.get("requested") else []
        records.append(
            {
                "index": index,
                "name": name,
                "version": version,
                "filename": filename,
                "artifact_type": artifact_type,
                "url": url,
                "url_scheme": parsed.scheme,
                "origin_host": parsed.hostname,
                "sha256": digest,
                "requested": bool(item.get("requested")),
                "requested_extras": requested_extras,
                "yanked": bool(item.get("is_yanked")),
                "wheel_tags": tags,
                "wheel_tag_compatible": tag_compatible,
                "requires_python": metadata.get("requires_python"),
                "source_report_item_index": index,
            }
        )
    failures = {
        "missing_sha256": sorted(missing_sha),
        "nonwheel_artifacts": sorted(nonwheels),
        "yanked_artifacts": sorted(yanked),
        "invalid_origins": sorted(invalid_origins),
        "incompatible_wheels": sorted(set(incompatible_wheels)),
        "duplicate_name_versions": sorted(duplicates),
    }
    success = not any(failures.values()) and len(records) == len(installs)
    return {
        "schema_version": "openeta.univtac.isaac_artifact_lock.v1",
        "source_pip_version": report.get("pip_version"),
        "record_count": len(records),
        "records": records,
        "failures": failures,
        "success": success,
    }


def failure_classification(lock: Mapping[str, Any]) -> str | None:
    failures = lock.get("failures", {})
    if failures.get("missing_sha256"):
        return "artifact_lock_missing_sha256"
    if failures.get("nonwheel_artifacts"):
        return "artifact_lock_contains_nonwheel"
    if any(failures.get(key) for key in ("yanked_artifacts", "invalid_origins", "incompatible_wheels", "duplicate_name_versions")):
        return "wheelhouse_validation_failed"
    return None


def validate_p0a_process(
    process: Mapping[str, Any], *, report_path: Path, constraints_path: Path, requirement: str
) -> None:
    command = list(process.get("command", ()))
    if process.get("returncode") != 0 or process.get("cleanup_complete") is not True:
        raise ValueError("P0A process did not complete cleanly")
    required_tokens = {
        "--dry-run", "--report", "--constraint", "--find-links", "--only-binary=flatdict",
        "--extra-index-url", "https://pypi.nvidia.com", requirement,
    }
    if not required_tokens.issubset(command):
        raise ValueError("P0A command identity changed")
    for option, expected in (("--report", report_path.resolve()), ("--constraint", constraints_path.resolve())):
        try:
            observed = Path(command[command.index(option) + 1]).resolve()
        except (ValueError, IndexError) as exc:
            raise ValueError(f"P0A command missing {option}") from exc
        if observed != expected:
            raise ValueError(f"P0A {option} path changed")
    prohibited = {"--no-deps", "--upgrade", "--ignore-installed", "--no-build-isolation"}
    if prohibited.intersection(command):
        raise ValueError("P0A command contains prohibited options")


def public_summary(lock: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": lock["schema_version"],
        "source_pip_version": lock.get("source_pip_version"),
        "record_count": lock["record_count"],
        "records": [
            {
                key: record.get(key)
                for key in (
                    "index", "name", "version", "filename", "artifact_type", "origin_host",
                    "sha256", "requested", "requested_extras", "yanked", "wheel_tags",
                    "wheel_tag_compatible", "requires_python", "source_report_item_index",
                )
            }
            for record in lock["records"]
        ],
        "failures": lock["failures"],
        "success": lock["success"],
    }


def deterministic_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
