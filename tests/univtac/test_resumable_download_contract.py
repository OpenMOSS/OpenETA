from __future__ import annotations

import copy
import io
import json
from pathlib import Path

import pytest

from scripts.univtac import build_reproducible_legacy_wheels as builder
from scripts.univtac import download_isaac_wheelhouse as downloader
from sim.envs.univtac.resumable_download_contract import (
    curl_download_command,
    deterministic_sha256,
    expected_total,
    parse_headers,
    sidecar_matches,
    validate_lock_pair,
)


CONFIG = {"connect_timeout_seconds": 60, "max_time_seconds": 3600, "speed_time_seconds": 180, "speed_limit_bytes_per_second": 1024}


def test_parse_range_identity_and_total() -> None:
    text = "HTTP/1.1 200 Connection established\r\n\r\nHTTP/1.1 206 Partial Content\r\nETag: abc\r\nContent-Range: bytes 0-0/3021300000\r\nContent-Length: 1\r\n\r\n"
    result = parse_headers(text)
    assert result["http_status"] == 206
    assert result["range_start"] == 0
    assert expected_total(result) == 3021300000


def test_resume_command_only_adds_continue_at_for_existing_partial(tmp_path: Path) -> None:
    first = curl_download_command(url="https://files.pythonhosted.org/x.whl", output=tmp_path / "x.part", headers=tmp_path / "h", offset=0, config=CONFIG)
    resumed = curl_download_command(url="https://files.pythonhosted.org/x.whl", output=tmp_path / "x.part", headers=tmp_path / "h", offset=42, config=CONFIG)
    assert "--continue-at" not in first
    assert resumed[resumed.index("--continue-at") + 1] == "-"
    assert "--retry" not in resumed and "--insecure" not in resumed and "--parallel" not in resumed


def test_partial_sidecar_binds_url_hash_size_and_etag() -> None:
    record = {"url": "https://pypi.nvidia.com/x.whl", "filename": "x.whl", "sha256": "abc"}
    metadata = {"content_length": 10, "etag": "tag"}
    sidecar = {"url": record["url"], "filename": "x.whl", "sha256": "abc", "expected_size": 10, "etag": "tag"}
    assert sidecar_matches(sidecar, record, metadata) is True
    sidecar["etag"] = "changed"
    assert sidecar_matches(sidecar, record, metadata) is False


def _lock_pair() -> tuple[dict, dict]:
    transformed = {"idna-ssl", "pyperclip", "antlr4-python3-runtime"}
    records = []
    for index in range(173):
        name = sorted(transformed)[index] if index < 3 else f"package-{index}"
        records.append(
            {
                "index": index,
                "name": name,
                "version": "1.0",
                "artifact_type": "sdist" if name in transformed else "wheel",
            }
        )
    source = {
        "schema_version": "openeta.univtac.isaac_artifact_lock.v1",
        "record_count": 173,
        "records": records,
    }
    source["artifact_lock_sha256"] = deterministic_sha256(source)
    derived_records = copy.deepcopy(records)
    for record in derived_records[:3]:
        record.update(
            artifact_type="wheel",
            url_scheme="file",
            source_transformation_sha256="transform",
        )
    derived = {
        "schema_version": "openeta.univtac.derived_install_artifact_lock.v1",
        "record_count": 173,
        "source_record_count": 173,
        "source_sdist_count": 3,
        "source_wheel_count": 170,
        "final_install_record_count": 173,
        "final_install_sdist_count": 0,
        "final_install_wheel_count": 173,
        "transformation_count": 3,
        "original_artifact_lock_sha256": source["artifact_lock_sha256"],
        "records": derived_records,
    }
    derived["derived_install_artifact_lock_sha256"] = deterministic_sha256(derived)
    return source, derived


def test_lock_pair_requires_exact_unique_indices() -> None:
    source, derived = _lock_pair()
    validate_lock_pair(source, derived)
    duplicate = copy.deepcopy(source)
    duplicate["records"][-1]["index"] = duplicate["records"][-2]["index"]
    duplicate["artifact_lock_sha256"] = deterministic_sha256(
        {key: value for key, value in duplicate.items() if key != "artifact_lock_sha256"}
    )
    duplicate_derived = copy.deepcopy(derived)
    duplicate_derived["original_artifact_lock_sha256"] = duplicate["artifact_lock_sha256"]
    duplicate_derived["derived_install_artifact_lock_sha256"] = deterministic_sha256(
        {
            key: value
            for key, value in duplicate_derived.items()
            if key != "derived_install_artifact_lock_sha256"
        }
    )
    with pytest.raises(ValueError, match="indices differ"):
        validate_lock_pair(duplicate, duplicate_derived)


def test_missing_source_input_is_classified(tmp_path: Path) -> None:
    record = {"filename": "legacy.tar.gz", "sha256": "0" * 64}
    with pytest.raises(downloader.WheelhouseError) as caught:
        downloader.copy_source_input(record, tmp_path / "missing", tmp_path / "target")
    assert caught.value.classification == "wheel_artifact_hash_mismatch"


def test_invalid_existing_cached_wheel_is_classified(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    (cache / "artifacts").mkdir(parents=True)
    (cache / "partial").mkdir()
    (cache / "quarantined").mkdir()
    final = cache / "artifacts" / "broken-1.0-py3-none-any.whl"
    final.write_bytes(b"not a wheel")
    record = {
        "filename": final.name,
        "name": "broken",
        "version": "1.0",
        "sha256": "0" * 64,
    }
    with pytest.raises(downloader.WheelhouseError) as caught:
        downloader.download_remote(record, {"content_length": final.stat().st_size}, cache, {}, tmp_path / "completed")
    assert caught.value.classification == "wheel_artifact_hash_mismatch"


class _RedirectedJsonResponse(io.BytesIO):
    headers = {}

    def __enter__(self) -> "_RedirectedJsonResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def geturl(self) -> str:
        return "http://example.invalid/project.json"


def test_metadata_fetch_rejects_unapproved_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builder.urllib.request, "urlopen", lambda *_args, **_kwargs: _RedirectedJsonResponse(json.dumps({}).encode()))
    with pytest.raises(builder.BridgeError, match="redirected to unapproved"):
        builder.fetch_json("https://pypi.org/pypi/example/1/json")
