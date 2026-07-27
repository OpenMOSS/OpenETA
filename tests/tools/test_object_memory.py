from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
import zipfile

import pytest
from PIL import Image

from agent.tools import object_memory as object_memory_module
from agent.tools.object_memory import (
    ObjectMemoryBankClient,
    ObjectMemoryBankConfig,
    ObjectMemoryResolutionError,
    object_memory_query_key,
)


def _png(color: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 12), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _bundle(*, key: str = "libero/alphabet_soup", unsafe: bool = False) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        root = "../escape" if unsafe else key
        archive.writestr(f"{root}/view_top.png", _png("blue"))
        archive.writestr(f"{root}/view_front.png", _png("red"))
        archive.writestr(f"{root}/view_side.png", _png("green"))
        archive.writestr(
            "manifest.json",
            json.dumps(
                [
                    {
                        "asset_id": key.split("/", 1)[1],
                        "label": "alphabet soup",
                        "namespace": key.split("/", 1)[0],
                        "key": key,
                        # The live service may report repository-relative paths
                        # that differ from member paths, so parsing uses ZIP members.
                        "image_paths": ["images/legacy/path.png"],
                    }
                ]
            ),
        )
    return buffer.getvalue()


def _multi_bundle(*, keys: list[str]) -> bytes:
    buffer = io.BytesIO()
    manifests = []
    with zipfile.ZipFile(buffer, "w") as archive:
        for index, key in enumerate(keys):
            archive.writestr(f"{key}/view_top.png", _png("blue"))
            archive.writestr(f"{key}/view_front.png", _png("red"))
            archive.writestr(f"{key}/view_side.png", _png("green"))
            namespace, asset_id = key.split("/", 1)
            manifests.append(
                {
                    "asset_id": asset_id,
                    "label": f"asset {index}",
                    "namespace": namespace,
                    "key": key,
                }
            )
        archive.writestr("manifest.json", json.dumps(manifests))
    return buffer.getvalue()


def _search_response(*candidates: dict) -> bytes:
    return json.dumps(
        {
            "schema_version": "openeta.object_memory.search.v1",
            "candidates": list(candidates),
        }
    ).encode()


def _candidate(
    key: str,
    *,
    score: float,
    match_type: str,
    aliases: list[str] | None = None,
) -> dict:
    return {
        "key": key,
        "label": key.rsplit("/", 1)[-1].replace("_", " "),
        "aliases": aliases or [],
        "score": score,
        "match_type": match_type,
    }


def test_object_memory_query_key_normalizes_environment_and_target() -> None:
    assert (
        object_memory_query_key(
            environment="openeta/libero_libero_object_task0-v0",
            target_object="Alphabet Soup",
        )
        == "libero/alphabet_soup"
    )


def test_object_memory_config_repr_redacts_api_key() -> None:
    config = ObjectMemoryBankConfig(
        base_url="https://memory.example.test",
        api_key="do-not-log-this",
    )

    assert "do-not-log-this" not in repr(config)
    assert (
        object_memory_query_key(environment="ignored", target_object="libero/red_coffee_mug")
        == "libero/red_coffee_mug"
    )


def test_object_memory_config_allows_private_network_http() -> None:
    ObjectMemoryBankConfig(
        base_url="http://127.0.0.2:8080",
        api_key="secret",
    ).validate()
    ObjectMemoryBankConfig(
        base_url="http://127.0.0.1:8080",
        api_key="secret",
    ).validate()


@pytest.mark.parametrize(
    "base_url",
    [
        "http://8.8.8.8:8080",
        "http://memory.example.test:8080",
        "ftp://127.0.0.2:8080",
        "http://user@127.0.0.2:8080",
    ],
)
def test_object_memory_config_rejects_unsafe_cleartext_urls(base_url: str) -> None:
    with pytest.raises(ValueError, match="object memory bank"):
        ObjectMemoryBankConfig(base_url=base_url, api_key="secret").validate()


def test_object_memory_private_network_download_bypasses_proxy(monkeypatch) -> None:
    captured_handlers = []

    class Response:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit):
            return b"response"

    class Opener:
        def open(self, _request, *, timeout):
            assert timeout == 3.0
            return Response()

    def build_opener(*handlers):
        captured_handlers.extend(handlers)
        return Opener()

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)

    result = object_memory_module._download_object_memory_bundle(
        "http://127.0.0.2:8080/search",
        {},
        3.0,
        1024,
    )

    assert result == b"response"
    assert any(
        isinstance(handler, urllib.request.ProxyHandler) and handler.proxies == {}
        for handler in captured_handlers
    )


def test_object_memory_client_downloads_and_orders_three_views() -> None:
    calls = []

    def download(url, headers, timeout, max_bytes):
        calls.append((url, dict(headers), timeout, max_bytes))
        return _bundle()

    client = ObjectMemoryBankClient(
        ObjectMemoryBankConfig(
            base_url="https://memory.example.test",
            api_key="secret",
        ),
        downloader=download,
    )

    bundle = client.retrieve(environment="libero", target_object="alphabet soup")

    assert bundle.query_key == "libero/alphabet_soup"
    assert bundle.asset_id == "alphabet_soup"
    assert [reference.view for reference in bundle.references] == ["front", "side", "top"]
    assert all(reference.image_bytes.startswith(b"\x89PNG") for reference in bundle.references)
    assert calls[0][0] == "https://memory.example.test/bundle?name=libero%2Falphabet_soup"
    assert calls[0][1] == {"X-API-Key": "secret"}


def test_object_memory_client_rejects_archive_path_traversal() -> None:
    client = ObjectMemoryBankClient(
        ObjectMemoryBankConfig(
            base_url="https://memory.example.test",
            api_key="secret",
        ),
        downloader=lambda *_args: _bundle(unsafe=True),
    )

    with pytest.raises(ValueError, match="unsafe path"):
        client.retrieve(environment="libero", target_object="alphabet soup")


def test_object_memory_client_selects_unique_exact_token_match() -> None:
    client = ObjectMemoryBankClient(
        ObjectMemoryBankConfig(
            base_url="https://memory.example.test",
            api_key="secret",
        ),
        downloader=lambda *_args: _multi_bundle(
            keys=[
                "libero/akita_black_bowl",
                "libero/red_bowl",
                "libero/white_bowl",
                "libero/bowl_drainer",
            ]
        ),
    )

    bundle = client.retrieve(environment="libero", target_object="black bowl")

    assert bundle.asset_id == "akita_black_bowl"
    assert all(
        reference.archive_path.startswith("libero/akita_black_bowl/")
        for reference in bundle.references
    )


def test_object_memory_client_rejects_ambiguous_exact_token_matches() -> None:
    client = ObjectMemoryBankClient(
        ObjectMemoryBankConfig(
            base_url="https://memory.example.test",
            api_key="secret",
        ),
        downloader=lambda *_args: _multi_bundle(
            keys=[
                "libero/akita_black_bowl",
                "libero/stone_black_bowl",
            ]
        ),
    )

    with pytest.raises(ValueError, match="uniquely matching"):
        client.retrieve(environment="libero", target_object="black bowl")


def test_object_memory_client_resolves_search_then_fetches_exact_bundle() -> None:
    calls: list[str] = []

    def download(url, *_args):
        calls.append(url)
        if "/search?" in url:
            return _search_response(
                _candidate(
                    "libero/akita_black_bowl",
                    score=0.96,
                    match_type="exact_alias",
                    aliases=["black bowl"],
                ),
                _candidate("libero/red_bowl", score=0.31, match_type="token"),
            )
        assert url.endswith("/bundle?name=libero%2Fakita_black_bowl")
        return _bundle(key="libero/akita_black_bowl")

    client = ObjectMemoryBankClient(
        ObjectMemoryBankConfig(
            base_url="https://memory.example.test",
            api_key="secret",
        ),
        downloader=download,
    )

    bundle = client.resolve(environment="openeta/libero_task0-v0", target_object="black bowl")

    assert calls[0].endswith("/search?namespace=libero&q=black+bowl&limit=5")
    assert bundle.query_key == "libero/black_bowl"
    assert bundle.resolved_key == "libero/akita_black_bowl"
    assert bundle.asset_id == "akita_black_bowl"
    assert bundle.resolution is not None
    assert bundle.resolution.to_dict() == {
        "requested_query_key": "libero/black_bowl",
        "resolved_asset_key": "libero/akita_black_bowl",
        "score": 0.96,
        "match_type": "exact_alias",
        "candidate_count": 2,
        "legacy_fallback": False,
    }


def test_object_memory_client_rejects_ambiguous_ranked_search() -> None:
    client = ObjectMemoryBankClient(
        ObjectMemoryBankConfig(
            base_url="https://memory.example.test",
            api_key="secret",
            search_min_margin=0.1,
        ),
        downloader=lambda *_args: _search_response(
            _candidate("libero/akita_black_bowl", score=0.90, match_type="fuzzy"),
            _candidate("libero/stone_black_bowl", score=0.85, match_type="fuzzy"),
        ),
    )

    with pytest.raises(ObjectMemoryResolutionError) as captured:
        client.resolve(environment="libero", target_object="black bowl")

    assert captured.value.code == "ambiguous_candidates"
    assert len(captured.value.candidates) == 2


def test_object_memory_client_rejects_low_confidence_ranked_search() -> None:
    client = ObjectMemoryBankClient(
        ObjectMemoryBankConfig(
            base_url="https://memory.example.test",
            api_key="secret",
            search_min_score=0.75,
        ),
        downloader=lambda *_args: _search_response(
            _candidate("libero/akita_black_bowl", score=0.62, match_type="semantic"),
        ),
    )

    with pytest.raises(ObjectMemoryResolutionError) as captured:
        client.resolve(environment="libero", target_object="black bowl")

    assert captured.value.code == "low_confidence"


def test_object_memory_client_allows_exact_key_below_rank_threshold() -> None:
    def download(url, *_args):
        if "/search?" in url:
            return _search_response(
                _candidate("libero/alphabet_soup", score=0.2, match_type="exact_key"),
            )
        return _bundle()

    client = ObjectMemoryBankClient(
        ObjectMemoryBankConfig(
            base_url="https://memory.example.test",
            api_key="secret",
            search_min_score=0.99,
        ),
        downloader=download,
    )

    bundle = client.resolve(environment="libero", target_object="alphabet soup")

    assert bundle.resolved_key == "libero/alphabet_soup"
    assert bundle.resolution is not None
    assert bundle.resolution.match_type == "exact_key"


def test_object_memory_client_falls_back_only_when_search_endpoint_is_unavailable() -> None:
    calls: list[str] = []

    def download(url, *_args):
        calls.append(url)
        if "/search?" in url:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return _multi_bundle(
            keys=[
                "libero/akita_black_bowl",
                "libero/red_bowl",
            ]
        )

    client = ObjectMemoryBankClient(
        ObjectMemoryBankConfig(
            base_url="https://memory.example.test",
            api_key="secret",
        ),
        downloader=download,
    )

    bundle = client.resolve(environment="libero", target_object="black bowl")

    assert len(calls) == 2
    assert bundle.resolved_key == "libero/akita_black_bowl"
    assert bundle.resolution is not None
    assert bundle.resolution.legacy_fallback is True
    assert bundle.resolution.match_type == "legacy_bundle"
