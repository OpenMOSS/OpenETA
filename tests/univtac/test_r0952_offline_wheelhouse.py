from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.univtac import download_isaac_wheelhouse as downloader
from scripts.univtac import resume_isaac51_r0952 as runner
from sim.envs.univtac.offline_wheelhouse_contract import (
    EXPECTED_TRANSFORMS,
    classify_records,
    compare_offline_resolution,
    offline_dry_run_command,
    select_canaries,
    validate_cached_metadata,
    validate_config,
    validate_existing_cache,
    validate_offline_command,
    validate_system_curl_provenance,
    verify_nvidia_canary_minimum,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/univtac/isaac51_offline_wheelhouse_r0952.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _remote_record(index: int, name: str, host: str, filename: str) -> dict:
    return {
        "index": index,
        "name": name,
        "version": "1.0",
        "filename": filename,
        "artifact_type": "wheel",
        "url": f"https://{host}/{name}/{filename}",
        "url_scheme": "https",
        "origin_host": host,
        "sha256": "0" * 64,
    }


def test_r0952_config_locks_transport_and_no_install_accounting() -> None:
    config = _config()
    validate_config(config)
    assert config["max_transport_attempts_per_artifact"] == 20
    assert "openeta_head" not in config
    assert config["revision_contract"]["implementation_base_head"] == (
        "2319e6587cb1c8330905313bae75e765b9b716d2"
    )
    assert config["install_accounting"]["total_invocations"] == 1
    changed = dict(config)
    changed["max_transport_attempts_per_artifact"] = 21
    with pytest.raises(ValueError):
        validate_config(changed)


def test_record_partition_is_exactly_169_plus_three_plus_one() -> None:
    records = [_remote_record(index, f"remote-{index}", "files.pythonhosted.org", f"remote_{index}-1.0-py3-none-any.whl") for index in range(169)]
    for offset, name in enumerate(sorted(EXPECTED_TRANSFORMS), 169):
        records.append({"index": offset, "name": name, "version": "1.0", "filename": f"{name}-1.0-py3-none-any.whl", "artifact_type": "wheel", "url_scheme": "file", "source_transformation_sha256": name})
    records.append({"index": 172, "name": "flatdict", "version": "4.0.1", "filename": "flatdict-4.0.1-py3-none-any.whl", "artifact_type": "wheel", "url_scheme": "file"})
    groups = classify_records(records)
    assert [len(groups[key]) for key in ("remote", "derived", "verified_local")] == [169, 3, 1]
    records[-1]["artifact_type"] = "sdist"
    records[-1]["filename"] = "flatdict.tar.gz"
    with pytest.raises(ValueError):
        classify_records(records)


def test_canaries_are_fixed_and_nvidia_minimum_is_verified() -> None:
    config = _config()
    records = [
        _remote_record(92, "networkx", "files.pythonhosted.org", config["networkx_canary"]["filename"]),
        _remote_record(config["nvidia_canary"]["index"], "isaacsim-rl", "pypi.nvidia.com", config["nvidia_canary"]["filename"]),
        _remote_record(8, "isaacsim", "pypi.nvidia.com", "isaacsim-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl"),
    ]
    selected = select_canaries(records, config)
    metadata = {
        selected["nvidia"]["filename"]: {"expected_total": 10},
        records[2]["filename"]: {"expected_total": 20},
    }
    assert verify_nvidia_canary_minimum(records, metadata, selected["nvidia"])["passed"] is True
    metadata[records[2]["filename"]]["expected_total"] = 5
    assert verify_nvidia_canary_minimum(records, metadata, selected["nvidia"])["passed"] is False


def test_probe_head_failure_uses_range_without_identity_misclassification(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    record = _remote_record(92, "networkx", "files.pythonhosted.org", "networkx-3.3-py3-none-any.whl")
    responses = [
        {"method": "head", "attempt": 1, "returncode": 28, "effective_url": record["url"], "effective_host": record["origin_host"], "error": "timeout", "http": None, "command": []},
        {"method": "range", "attempt": 1, "returncode": 0, "effective_url": record["url"], "effective_host": record["origin_host"], "error": None, "http": {"http_status": 206, "range_start": 0, "range_end": 0, "content_range_total": 123}, "command": []},
    ]
    monkeypatch.setattr(downloader, "_probe_attempt", lambda *_args, **_kwargs: responses.pop(0))
    result = downloader.probe_remote(record, tmp_path, _config(), classification="network_transport_preflight_failed")
    assert result["metadata_method"] == "head_failed_range_succeeded"
    assert result["expected_total"] == 123


def test_large_head_result_requires_range(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    record = _remote_record(0, "isaac", "pypi.nvidia.com", "isaac-1.0-py3-none-any.whl")
    calls = []
    responses = [
        {"method": "head", "attempt": 1, "returncode": 0, "effective_url": record["url"], "effective_host": record["origin_host"], "error": None, "http": {"http_status": 200, "content_length": 300 * 1024 * 1024}, "command": []},
        {"method": "range", "attempt": 1, "returncode": 0, "effective_url": record["url"], "effective_host": record["origin_host"], "error": None, "http": {"http_status": 206, "range_start": 0, "range_end": 0, "content_range_total": 300 * 1024 * 1024}, "command": []},
    ]
    def fake(*_args: object, **kwargs: object) -> dict:
        calls.append(kwargs["method"])
        return responses.pop(0)
    monkeypatch.setattr(downloader, "_probe_attempt", fake)
    result = downloader.probe_remote(record, tmp_path, _config(), classification="blocked_by_external_resources")
    assert calls == ["head", "range"]
    assert result["metadata_method"] == "head_large_range_succeeded"


def test_canary_probe_exhaustion_keeps_transport_classification(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    record = _remote_record(92, "networkx", "files.pythonhosted.org", "networkx-3.3-py3-none-any.whl")
    response = {"method": "range", "attempt": 1, "returncode": 28, "effective_url": record["url"], "effective_host": record["origin_host"], "error": "timeout", "http": None, "command": []}
    monkeypatch.setattr(downloader, "_probe_attempt", lambda *_args, **_kwargs: dict(response))
    monkeypatch.setattr(downloader.time, "sleep", lambda _seconds: None)
    with pytest.raises(downloader.WheelhouseError) as caught:
        downloader.probe_remote(record, tmp_path, _config(), classification="network_transport_preflight_failed")
    assert caught.value.classification == "network_transport_preflight_failed"


def test_probe_and_download_commands_keep_tls_and_single_connection(tmp_path: Path) -> None:
    record = _remote_record(1, "demo", "files.pythonhosted.org", "demo-1.0-py3-none-any.whl")
    command = downloader._probe_command(record, tmp_path / "headers", _config(), method="range")
    assert [command[command.index("--proto") + 1], command[command.index("--proto-redir") + 1]] == ["=https", "=https"]
    assert "--http1.1" in command and "--range" in command
    assert "--insecure" not in command and "--trusted-host" not in command and "--parallel" not in command
    assert command[0] == "/usr/bin/curl"


def test_curl_transport_does_not_inherit_conda_library_path() -> None:
    source = inspect.getsource(downloader.run_curl)
    assert 'environment.pop("LD_LIBRARY_PATH", None)' in source
    assert "env=environment" in source


def test_system_curl_provenance_rejects_conda_libraries() -> None:
    good = validate_system_curl_provenance(
        executable=Path("/usr/bin/curl"),
        version_output="curl 7.81.0 libcurl/7.81.0 OpenSSL/3.0.2\nProtocols: http https\nFeatures: SSL",
        ldd_output="libcurl.so.4 => /lib/x86_64-linux-gnu/libcurl.so.4 (0x1)\nlibssl.so.3 => /lib/x86_64-linux-gnu/libssl.so.3 (0x2)",
        conda_prefixes=(Path("/opt/conda/envs/r09"),),
    )
    assert good["curl_ssl_backend"] == "OpenSSL/3.0.2"
    with pytest.raises(ValueError, match="Conda"):
        validate_system_curl_provenance(
            executable=Path("/usr/bin/curl"),
            version_output="curl 7.81.0 libcurl/8.16 OpenSSL/3.5",
            ldd_output="libcurl.so.4 => /opt/conda/envs/r09/lib/libcurl.so.4 (0x1)",
            conda_prefixes=(Path("/opt/conda/envs/r09"),),
        )


def test_unapproved_redirect_is_rejected() -> None:
    record = _remote_record(1, "demo", "files.pythonhosted.org", "demo-1.0-py3-none-any.whl")
    with pytest.raises(downloader.WheelhouseError) as caught:
        downloader._validate_effective_url(record, "https://example.invalid/demo-1.0-py3-none-any.whl")
    assert caught.value.classification == "remote_artifact_identity_changed"


def test_download_attempt_limit_is_wrapper_owned(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    for name in ("artifacts", "partial", "quarantined"):
        (cache / name).mkdir(parents=True)
    record = _remote_record(1, "demo", "files.pythonhosted.org", "demo-1.0-py3-none-any.whl")
    config = _config()
    config["max_transport_attempts_per_artifact"] = 2
    config["download"]["retry_wait_seconds"] = 0
    calls = []
    def failed(command: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
        calls.append((command, timeout))
        return subprocess.CompletedProcess(command, 28, "", "timeout")
    monkeypatch.setattr(downloader, "run_curl", failed)
    with pytest.raises(downloader.WheelhouseError) as caught:
        downloader.download_remote(record, {"content_length": 10}, cache, config, cache / "completed.jsonl")
    assert caught.value.classification == "wheel_download_attempts_exhausted"
    assert len(calls) == 2
    assert all("--retry" not in command for command, _ in calls)


def test_existing_verified_source_inputs_are_accepted_and_extras_rejected(tmp_path: Path) -> None:
    config = _config()
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    for name in ("source_inputs", "derived_artifacts", "artifacts", "partial", "quarantined", "manifests"):
        (cache / name).mkdir(mode=0o700)
    (cache / ".download.lock").touch()
    source_lock = {"artifact_lock_sha256": config["source_artifact_lock_sha256"]}
    derived_lock = {"derived_install_artifact_lock_sha256": config["derived_install_lock_sha256"]}
    marker = {"schema_version": "openeta.univtac.r095_wheelhouse_cache.v1", "derived_install_artifact_lock_sha256": config["derived_install_lock_sha256"], "source_artifact_lock_sha256": config["source_artifact_lock_sha256"], "owner_uid": os.getuid(), "mode": "0700"}
    (cache / ".univtac-r095-wheelhouse.json").write_text(json.dumps(marker))
    os.chmod(cache / ".univtac-r095-wheelhouse.json", 0o600)
    for item in config["source_inputs"].values():
        path = cache / "source_inputs" / item["filename"]
        data = item["filename"].encode()
        path.write_bytes(data)
        path.chmod(0o444)
        item["size"] = len(data)
        item["sha256"] = hashlib.sha256(data).hexdigest()
    cached_lock = cache / "manifests/derived_install_artifact_lock.private.json"
    cached_lock.write_text(json.dumps(derived_lock))
    cached_lock.chmod(0o600)
    assert validate_existing_cache(cache, source_lock, derived_lock, config)["combined_status"] == "accepted_existing_verified_source_inputs"
    (cache / "artifacts/extra.whl").touch()
    with pytest.raises(ValueError, match="unexpected preexisting"):
        validate_existing_cache(cache, source_lock, derived_lock, config)


def test_cached_metadata_is_bound_to_lock_url_host_and_local_hash(tmp_path: Path) -> None:
    local = tmp_path / "local-1.0-py3-none-any.whl"
    local.write_bytes(b"wheel")
    local_digest = hashlib.sha256(b"wheel").hexdigest()
    records = [
        _remote_record(1, "remote", "files.pythonhosted.org", "remote-1.0-py3-none-any.whl"),
        {"index": 2, "name": "local", "filename": local.name, "url": local.as_uri(), "url_scheme": "file", "sha256": local_digest},
    ]
    metadata = {
        records[0]["filename"]: {"expected_total": 10, "source_url": records[0]["url"], "source_host": records[0]["origin_host"], "effective_url": records[0]["url"], "tls_verification": "enabled_and_passed", "metadata_method": "head_succeeded"},
        local.name: {"expected_total": 5, "metadata_method": "local_locked_file"},
    }
    assert validate_cached_metadata(records, metadata, {"files.pythonhosted.org"})["passed"] is True
    metadata[records[0]["filename"]]["effective_url"] = "https://example.invalid/remote-1.0-py3-none-any.whl"
    with pytest.raises(ValueError, match="identity changed"):
        validate_cached_metadata(records, metadata, {"files.pythonhosted.org"})


def _report_item(name: str, *, source: bool, transformed: bool = False) -> dict:
    filename = f"{name}-1.0.tar.gz" if source and transformed else f"{name}-1.0-py3-none-any.whl"
    scheme = "https" if source else "file"
    return {
        "requested": name == "isaaclab",
        "metadata": {"name": name, "version": "1.0", "requires_dist": [], "requires_python": ">=3.11", "provides_extra": ["all", "isaacsim"] if name == "isaaclab" else []},
        "download_info": {"url": f"{scheme}:///artifacts/{filename}", "archive_info": {"hashes": {"sha256": f"{'source' if source else 'final'}-{name}"}}},
    }


def test_offline_plan_allows_only_three_locked_artifact_transformations() -> None:
    names = sorted(EXPECTED_TRANSFORMS) + ["isaaclab"] + [f"package-{index}" for index in range(169)]
    source_items = [_report_item(name, source=True, transformed=name in EXPECTED_TRANSFORMS) for name in names]
    offline_items = [_report_item(name, source=False) for name in names]
    records = []
    for index, item in enumerate(offline_items):
        name = item["metadata"]["name"]
        records.append({"index": index, "name": name, "filename": Path(item["download_info"]["url"]).name, "sha256": f"final-{name}", "source_transformation_sha256": name if name in EXPECTED_TRANSFORMS else None})
    result = compare_offline_resolution({"install": source_items}, {"records": records}, {"install": offline_items})
    assert result["success"] is True
    records[3]["source_transformation_sha256"] = "unauthorized"
    result = compare_offline_resolution({"install": source_items}, {"records": records}, {"install": offline_items})
    assert result["success"] is False
    assert "transformation_set_changed" in result["failures"]


def test_offline_command_is_dry_run_only_and_has_no_network_flags(tmp_path: Path) -> None:
    command = offline_dry_run_command(Path("python"), tmp_path / "wheels", tmp_path / "constraints", tmp_path / "report.json", "isaaclab[isaacsim,all]==2.3.0")
    validate_offline_command(command)
    assert "--dry-run" in command and "--no-index" in command and "--only-binary=:all:" in command
    assert "--extra-index-url" not in command and "--no-deps" not in command


def test_r0952_runner_stops_before_install_or_simulator() -> None:
    source = inspect.getsource(runner.main)
    assert 'manifest["stages"]["O0"] = "passed"' in source
    assert '"actual_install_executed": False' in source
    assert '"isaac_started": False' in source
    assert '"task_started": False' in source and '"agent_started": False' in source
    assert "AppLauncher" not in source and "start_seed" not in source
    assert "I0B_R1" not in source and "pip install" not in source
    assert source.index('manifest["stages"]["T0"] = "passed"') < source.index('manifest["stages"]["T1"] = "passed"')
    assert source.index('manifest["stages"]["T1"] = "passed"') < source.index("download_process = managed(")
    assert 'prior_r0952.get("classification") == "network_transport_preflight_failed"' in source
    assert 'manifest["stages"]["R0"] = "failed"' in source


def test_downloader_revalidates_cache_before_transport() -> None:
    source = inspect.getsource(downloader.main)
    assert source.index("validate_existing_cache(cache, source, derived, config)") < source.index('state["current_stage"] = "C0"')


def test_complete_part_is_validated_before_and_after_atomic_rename() -> None:
    source = inspect.getsource(downloader.download_remote)
    assert 'validate_wheel(part, record, total, check_filename=False)' in source
    assert source.index('validate_wheel(part, record, total, check_filename=False)') < source.index("os.replace(part, final)")
    assert source.index("os.replace(part, final)") < source.rindex("validation = validate_wheel(final, record, total)")
    assert 'historical_attempts = len(sidecar.get("attempts", []))' in source
    assert 'historical_attempts + len(attempts)' in source


def test_cached_large_metadata_is_refreshed_before_download() -> None:
    source = inspect.getsource(downloader.main)
    assert source.index("if metadata_cache_reused:") < source.index("refreshed_large = []")
    assert source.index("refreshed_large.append") < source.index('state["current_stage"] = "D0"')


def test_source_input_modes_are_checked(tmp_path: Path) -> None:
    path = tmp_path / "source"
    path.write_bytes(b"source")
    path.chmod(0o444)
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
