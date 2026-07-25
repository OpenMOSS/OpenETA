"""Bounded client for retrieving simulator asset views from object memory."""

from __future__ import annotations

import io
import ipaddress
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath

from adapter.protocol import JsonDict


OBJECT_MEMORY_BANK_URL_ENV = "OPENETA_OBJECT_MEMORY_BANK_URL"
OBJECT_MEMORY_BANK_API_KEY_ENV = "OPENETA_OBJECT_MEMORY_BANK_API_KEY"
# Public snapshot placeholder. Configure OPENETA_OBJECT_MEMORY_BANK_URL in a
# local, ignored environment before enabling object-memory access.
DEFAULT_OBJECT_MEMORY_BANK_URL = "https://memory.example.invalid"
DEFAULT_OBJECT_MEMORY_TIMEOUT_S = 30.0
DEFAULT_OBJECT_MEMORY_MAX_BUNDLE_BYTES = 32 * 1024 * 1024
DEFAULT_OBJECT_MEMORY_MAX_IMAGE_BYTES = 8 * 1024 * 1024
DEFAULT_OBJECT_MEMORY_MAX_UNCOMPRESSED_BYTES = 40 * 1024 * 1024
DEFAULT_OBJECT_MEMORY_MAX_SEARCH_BYTES = 256 * 1024
DEFAULT_OBJECT_MEMORY_SEARCH_LIMIT = 5
DEFAULT_OBJECT_MEMORY_SEARCH_MIN_SCORE = 0.75
DEFAULT_OBJECT_MEMORY_SEARCH_MIN_MARGIN = 0.10
OBJECT_MEMORY_SEARCH_SCHEMA_VERSION = "openeta.object_memory.search.v1"
OBJECT_MEMORY_SEARCH_MATCH_TYPES = frozenset(
    {"exact_key", "exact_alias", "token", "fuzzy", "semantic"}
)

ObjectMemoryDownloader = Callable[[str, Mapping[str, str], float, int], bytes]


@dataclass(frozen=True, slots=True)
class ObjectMemoryBankConfig:
    base_url: str
    api_key: str = field(repr=False)
    timeout_s: float = DEFAULT_OBJECT_MEMORY_TIMEOUT_S
    max_bundle_bytes: int = DEFAULT_OBJECT_MEMORY_MAX_BUNDLE_BYTES
    max_image_bytes: int = DEFAULT_OBJECT_MEMORY_MAX_IMAGE_BYTES
    max_uncompressed_bytes: int = DEFAULT_OBJECT_MEMORY_MAX_UNCOMPRESSED_BYTES
    max_search_bytes: int = DEFAULT_OBJECT_MEMORY_MAX_SEARCH_BYTES
    search_limit: int = DEFAULT_OBJECT_MEMORY_SEARCH_LIMIT
    search_min_score: float = DEFAULT_OBJECT_MEMORY_SEARCH_MIN_SCORE
    search_min_margin: float = DEFAULT_OBJECT_MEMORY_SEARCH_MIN_MARGIN
    allow_legacy_bundle_fallback: bool = True

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "ObjectMemoryBankConfig | None":
        source = environ if environ is not None else os.environ
        api_key = str(source.get(OBJECT_MEMORY_BANK_API_KEY_ENV) or "").strip()
        if not api_key:
            return None
        base_url = str(
            source.get(OBJECT_MEMORY_BANK_URL_ENV) or DEFAULT_OBJECT_MEMORY_BANK_URL
        ).strip()
        return cls(base_url=base_url, api_key=api_key)

    def validate(self) -> None:
        parsed = urllib.parse.urlparse(self.base_url)
        if not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("object memory bank URL must contain a host and no credentials")
        if parsed.scheme == "http":
            try:
                address = ipaddress.ip_address(parsed.hostname)
            except ValueError as exc:
                raise ValueError(
                    "object memory bank HTTP URL must use a private IP literal"
                ) from exc
            if not (address.is_private or address.is_loopback or address.is_link_local):
                raise ValueError(
                    "object memory bank HTTP URL must use a private, loopback, or "
                    "link-local IP"
                )
        elif parsed.scheme != "https":
            raise ValueError(
                "object memory bank URL must use https or private-network http"
            )
        if parsed.query or parsed.fragment:
            raise ValueError("object memory bank URL must not contain query or fragment")
        if not self.api_key:
            raise ValueError("object memory bank API key is missing")
        if self.max_search_bytes < 1:
            raise ValueError("object memory search byte limit must be positive")
        if not 1 <= self.search_limit <= 20:
            raise ValueError("object memory search limit must be between 1 and 20")
        if not 0.0 <= self.search_min_score <= 1.0:
            raise ValueError("object memory search minimum score must be between 0 and 1")
        if not 0.0 <= self.search_min_margin <= 1.0:
            raise ValueError("object memory search minimum margin must be between 0 and 1")


def load_configured_object_memory_bank(
    *,
    environ: Mapping[str, str] | None = None,
    dotenv_path: str = ".env",
    apikey_path: str = "apikey.md",
) -> ObjectMemoryBankConfig | None:
    """Load object-memory credentials with env > .env > local curl example precedence."""

    source = dict(environ if environ is not None else os.environ)
    dotenv = _read_simple_env(dotenv_path)
    example_url, example_key = _read_object_memory_curl_example(apikey_path)
    api_key = str(
        source.get(OBJECT_MEMORY_BANK_API_KEY_ENV)
        or dotenv.get(OBJECT_MEMORY_BANK_API_KEY_ENV)
        or example_key
        or ""
    ).strip()
    if not api_key:
        return None
    base_url = str(
        source.get(OBJECT_MEMORY_BANK_URL_ENV)
        or dotenv.get(OBJECT_MEMORY_BANK_URL_ENV)
        or example_url
        or DEFAULT_OBJECT_MEMORY_BANK_URL
    ).strip()
    return ObjectMemoryBankConfig(base_url=base_url, api_key=api_key)


@dataclass(frozen=True, slots=True)
class ObjectMemoryReference:
    view: str
    archive_path: str
    image_bytes: bytes


@dataclass(frozen=True, slots=True)
class ObjectMemorySearchCandidate:
    key: str
    namespace: str
    asset_id: str
    label: str
    aliases: tuple[str, ...]
    score: float
    match_type: str

    def to_dict(self) -> JsonDict:
        return {
            "key": self.key,
            "namespace": self.namespace,
            "asset_id": self.asset_id,
            "label": self.label,
            "aliases": list(self.aliases),
            "score": self.score,
            "match_type": self.match_type,
        }


@dataclass(frozen=True, slots=True)
class ObjectMemoryResolution:
    requested_query_key: str
    resolved_asset_key: str
    score: float | None
    match_type: str
    candidate_count: int
    legacy_fallback: bool = False

    def to_dict(self) -> JsonDict:
        return {
            "requested_query_key": self.requested_query_key,
            "resolved_asset_key": self.resolved_asset_key,
            "score": self.score,
            "match_type": self.match_type,
            "candidate_count": self.candidate_count,
            "legacy_fallback": self.legacy_fallback,
        }


class ObjectMemoryResolutionError(ValueError):
    """Raised when ranked search cannot safely select one canonical asset."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        candidates: tuple[ObjectMemorySearchCandidate, ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.candidates = candidates


@dataclass(frozen=True, slots=True)
class ObjectMemoryBundle:
    query_key: str
    namespace: str
    asset_id: str
    label: str
    references: tuple[ObjectMemoryReference, ...]
    manifest: JsonDict
    resolved_key: str = ""
    resolution: ObjectMemoryResolution | None = None


class ObjectMemoryBankClient:
    """Fetch and validate one three-view object-memory bundle."""

    def __init__(
        self,
        config: ObjectMemoryBankConfig,
        *,
        downloader: ObjectMemoryDownloader | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.downloader = downloader or _download_object_memory_bundle

    def retrieve(self, *, environment: str, target_object: str) -> ObjectMemoryBundle:
        query_key = object_memory_query_key(
            environment=environment,
            target_object=target_object,
        )
        return self._retrieve_exact(query_key)

    def search(
        self,
        *,
        environment: str,
        target_object: str,
    ) -> tuple[ObjectMemorySearchCandidate, ...]:
        query_key = object_memory_query_key(
            environment=environment,
            target_object=target_object,
        )
        namespace, _ = query_key.split("/", 1)
        endpoint = self.config.base_url.rstrip("/") + "/search?" + urllib.parse.urlencode(
            {
                "namespace": namespace,
                "q": target_object.strip(),
                "limit": self.config.search_limit,
            }
        )
        raw = self.downloader(
            endpoint,
            {"X-API-Key": self.config.api_key},
            self.config.timeout_s,
            self.config.max_search_bytes,
        )
        return _parse_object_memory_search(
            raw,
            namespace=namespace,
            limit=self.config.search_limit,
        )

    def resolve(self, *, environment: str, target_object: str) -> ObjectMemoryBundle:
        requested_query_key = object_memory_query_key(
            environment=environment,
            target_object=target_object,
        )
        try:
            candidates = self.search(
                environment=environment,
                target_object=target_object,
            )
        except urllib.error.HTTPError as exc:
            if not self.config.allow_legacy_bundle_fallback or exc.code not in {
                404,
                405,
                501,
            }:
                raise
            bundle = self._retrieve_exact(requested_query_key)
            resolved_key = str(bundle.manifest.get("key") or requested_query_key)
            return replace(
                bundle,
                query_key=requested_query_key,
                resolved_key=resolved_key,
                resolution=ObjectMemoryResolution(
                    requested_query_key=requested_query_key,
                    resolved_asset_key=resolved_key,
                    score=None,
                    match_type="legacy_bundle",
                    candidate_count=0,
                    legacy_fallback=True,
                ),
            )

        selected = _select_search_candidate(
            candidates,
            query_key=requested_query_key,
            min_score=self.config.search_min_score,
            min_margin=self.config.search_min_margin,
        )
        bundle = self._retrieve_exact(selected.key)
        return replace(
            bundle,
            query_key=requested_query_key,
            resolved_key=selected.key,
            resolution=ObjectMemoryResolution(
                requested_query_key=requested_query_key,
                resolved_asset_key=selected.key,
                score=selected.score,
                match_type=selected.match_type,
                candidate_count=len(candidates),
            ),
        )

    def _retrieve_exact(self, query_key: str) -> ObjectMemoryBundle:
        endpoint = self.config.base_url.rstrip("/") + "/bundle?" + urllib.parse.urlencode(
            {"name": query_key}
        )
        raw = self.downloader(
            endpoint,
            {"X-API-Key": self.config.api_key},
            self.config.timeout_s,
            self.config.max_bundle_bytes,
        )
        return _parse_object_memory_bundle(
            raw,
            query_key=query_key,
            max_image_bytes=self.config.max_image_bytes,
            max_uncompressed_bytes=self.config.max_uncompressed_bytes,
        )


def object_memory_query_key(*, environment: str, target_object: str) -> str:
    """Build a controlled `<namespace>/<asset_id>` lookup key."""

    target = target_object.strip().lower()
    if "/" in target:
        parts = target.split("/")
        if len(parts) != 2:
            raise ValueError("object memory target key must contain one namespace separator")
        namespace = _safe_key_component(parts[0])
        asset_id = _safe_key_component(parts[1])
    else:
        namespace = _environment_namespace(environment)
        asset_id = _safe_key_component(target)
    return f"{namespace}/{asset_id}"


def _environment_namespace(environment: str) -> str:
    normalized = environment.strip().lower()
    for namespace in ("libero", "robocasa", "maniskill", "robotwin"):
        if namespace in normalized:
            return namespace
    return _safe_key_component(normalized)


def _safe_key_component(value: str) -> str:
    component = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not component or len(component) > 128:
        raise ValueError("object memory lookup contains an invalid key component")
    return component


def _parse_object_memory_search(
    raw: bytes,
    *,
    namespace: str,
    limit: int,
) -> tuple[ObjectMemorySearchCandidate, ...]:
    if not raw:
        raise ValueError("object memory search response is empty")
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("object memory search response is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("object memory search response must be an object")
    if payload.get("schema_version") != OBJECT_MEMORY_SEARCH_SCHEMA_VERSION:
        raise ValueError("object memory search schema version is incompatible")
    rows = payload.get("candidates")
    if not isinstance(rows, list) or len(rows) > limit:
        raise ValueError("object memory search candidate count is invalid")

    candidates: list[ObjectMemorySearchCandidate] = []
    seen_keys: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("object memory search candidate must be an object")
        key = str(row.get("key") or "").strip().lower()
        if "/" not in key:
            raise ValueError("object memory search candidate key is invalid")
        candidate_namespace, asset_id = key.split("/", 1)
        if (
            candidate_namespace != namespace
            or _safe_key_component(candidate_namespace) != candidate_namespace
            or _safe_key_component(asset_id) != asset_id
        ):
            raise ValueError("object memory search candidate namespace or key is invalid")
        if key in seen_keys:
            raise ValueError("object memory search contains duplicate candidate keys")
        seen_keys.add(key)
        try:
            score = float(row.get("score"))
        except (TypeError, ValueError) as exc:
            raise ValueError("object memory search candidate score is invalid") from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("object memory search candidate score is invalid")
        match_type = str(row.get("match_type") or "").strip()
        if match_type not in OBJECT_MEMORY_SEARCH_MATCH_TYPES:
            raise ValueError("object memory search candidate match type is invalid")
        aliases_value = row.get("aliases") or []
        if (
            not isinstance(aliases_value, list)
            or len(aliases_value) > 32
            or not all(
                isinstance(alias, str) and 0 < len(alias.strip()) <= 256
                for alias in aliases_value
            )
        ):
            raise ValueError("object memory search candidate aliases are invalid")
        candidates.append(
            ObjectMemorySearchCandidate(
                key=key,
                namespace=candidate_namespace,
                asset_id=asset_id,
                label=str(row.get("label") or target_label(asset_id)).strip()[:256],
                aliases=tuple(alias.strip() for alias in aliases_value),
                score=score,
                match_type=match_type,
            )
        )
    return tuple(sorted(candidates, key=lambda candidate: (-candidate.score, candidate.key)))


def _select_search_candidate(
    candidates: tuple[ObjectMemorySearchCandidate, ...],
    *,
    query_key: str,
    min_score: float,
    min_margin: float,
) -> ObjectMemorySearchCandidate:
    exact = tuple(candidate for candidate in candidates if candidate.key == query_key)
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ObjectMemoryResolutionError(
            "object memory search returned duplicate exact matches",
            code="duplicate_exact_match",
            candidates=candidates,
        )
    if not candidates:
        raise ObjectMemoryResolutionError(
            "object memory search returned no candidates",
            code="no_candidates",
        )
    top = candidates[0]
    if top.score < min_score:
        raise ObjectMemoryResolutionError(
            "object memory search top candidate is below the confidence threshold",
            code="low_confidence",
            candidates=candidates,
        )
    if len(candidates) > 1 and top.score - candidates[1].score < min_margin:
        raise ObjectMemoryResolutionError(
            "object memory search top candidates are ambiguous",
            code="ambiguous_candidates",
            candidates=candidates,
        )
    return top


def _download_object_memory_bundle(
    url: str,
    headers: Mapping[str, str],
    timeout_s: float,
    max_bytes: int,
) -> bytes:
    request = urllib.request.Request(
        url,
        headers={**dict(headers), "User-Agent": "OpenETA-ObjectMemory/1.0"},
        method="GET",
    )
    handlers: list[urllib.request.BaseHandler] = [_RejectObjectMemoryRedirects()]
    if _is_private_ip_url(url):
        # Keep LAN object-memory traffic direct regardless of ambient proxy settings.
        handlers.insert(0, urllib.request.ProxyHandler({}))
    opener = urllib.request.build_opener(*handlers)
    with opener.open(request, timeout=timeout_s) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            raise ValueError("object memory bundle exceeds the configured byte limit")
        raw = response.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("object memory bundle exceeds the configured byte limit")
    return raw


def _is_private_ip_url(url: str) -> bool:
    hostname = urllib.parse.urlparse(url).hostname
    if not hostname:
        return False
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local


class _RejectObjectMemoryRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        del request, file_pointer, code, message, headers, new_url
        return None


def _parse_object_memory_bundle(
    raw: bytes,
    *,
    query_key: str,
    max_image_bytes: int,
    max_uncompressed_bytes: int,
) -> ObjectMemoryBundle:
    if not raw:
        raise ValueError("object memory bundle is empty")
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise ValueError("object memory response is not a valid ZIP bundle") from exc
    with archive:
        members = archive.infolist()
        if not members or len(members) > 32:
            raise ValueError("object memory bundle has an invalid member count")
        total_size = sum(max(0, member.file_size) for member in members)
        if total_size > max_uncompressed_bytes:
            raise ValueError("object memory bundle exceeds the uncompressed byte limit")
        for member in members:
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or member.is_dir():
                if member.is_dir():
                    continue
                raise ValueError("object memory bundle contains an unsafe path")
        manifest_member = next(
            (member for member in members if PurePosixPath(member.filename).name == "manifest.json"),
            None,
        )
        if manifest_member is None or manifest_member.file_size > 64 * 1024:
            raise ValueError("object memory bundle manifest is missing or too large")
        try:
            manifest_value = json.loads(archive.read(manifest_member))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("object memory bundle manifest is invalid") from exc
        manifest = _select_manifest_asset(manifest_value, query_key=query_key)
        all_image_members = [
            member
            for member in members
            if PurePosixPath(member.filename).suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        manifest_key = str(manifest.get("key") or "").strip().lower()
        image_members = [
            member
            for member in all_image_members
            if _member_belongs_to_asset(member, asset_key=manifest_key)
        ]
        if not image_members and len(manifest_value) == 1:
            image_members = all_image_members
        if len(image_members) < 3:
            raise ValueError("object memory bundle must contain three reference views")
        ordered = sorted(image_members, key=_reference_member_sort_key)[:3]
        references: list[ObjectMemoryReference] = []
        for member in ordered:
            if member.file_size <= 0 or member.file_size > max_image_bytes:
                raise ValueError("object memory reference exceeds the image byte limit")
            image_bytes = archive.read(member)
            _verify_image_bytes(image_bytes)
            references.append(
                ObjectMemoryReference(
                    view=_reference_view(member.filename),
                    archive_path=member.filename,
                    image_bytes=image_bytes,
                )
            )
    namespace, asset_id = query_key.split("/", 1)
    return ObjectMemoryBundle(
        query_key=query_key,
        namespace=str(manifest.get("namespace") or namespace),
        asset_id=str(manifest.get("asset_id") or asset_id),
        label=str(manifest.get("label") or target_label(asset_id)),
        references=tuple(references),
        manifest=dict(manifest),
    )


def _select_manifest_asset(value: object, *, query_key: str) -> JsonDict:
    if not isinstance(value, list) or not value:
        raise ValueError("object memory bundle manifest must contain assets")
    manifests: list[JsonDict] = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError("object memory bundle manifest entry must be an object")
        manifests.append(entry)

    exact = [
        manifest
        for manifest in manifests
        if str(manifest.get("key") or "").strip().lower() == query_key
    ]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError("object memory bundle contains duplicate exact assets")

    if len(manifests) == 1 and not str(manifests[0].get("key") or "").strip():
        return manifests[0]

    namespace, query_asset_id = query_key.split("/", 1)
    query_tokens = frozenset(query_asset_id.split("_"))
    token_matches: list[JsonDict] = []
    for manifest in manifests:
        manifest_key = str(manifest.get("key") or "").strip().lower()
        if "/" not in manifest_key:
            continue
        candidate_namespace, candidate_asset_id = manifest_key.split("/", 1)
        if candidate_namespace != namespace:
            continue
        candidate_tokens = frozenset(candidate_asset_id.split("_"))
        if query_tokens.issubset(candidate_tokens):
            token_matches.append(manifest)
    if len(token_matches) == 1:
        return token_matches[0]
    raise ValueError(
        "object memory bundle does not contain one uniquely matching exact-token asset"
    )


def _member_belongs_to_asset(member: zipfile.ZipInfo, *, asset_key: str) -> bool:
    if not asset_key:
        return False
    member_parts = PurePosixPath(member.filename).parts
    asset_parts = PurePosixPath(asset_key).parts
    return member_parts[: len(asset_parts)] == asset_parts


def _reference_member_sort_key(member: zipfile.ZipInfo) -> tuple[int, str]:
    view = _reference_view(member.filename)
    order = {"front": 0, "side": 1, "top": 2}
    return order.get(view, 9), member.filename


def _reference_view(path: str) -> str:
    stem = PurePosixPath(path).stem.lower()
    for view in ("front", "side", "top"):
        if view in stem:
            return view
    return stem


def _verify_image_bytes(raw: bytes) -> None:
    from PIL import Image

    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
    except Exception as exc:  # noqa: BLE001 - normalize Pillow decoder failures.
        raise ValueError("object memory bundle contains an invalid reference image") from exc


def target_label(asset_id: str) -> str:
    return " ".join(asset_id.replace("_", " ").split())


def _read_simple_env(path: str) -> dict[str, str]:
    env_path = os.path.abspath(path)
    if not os.path.isfile(env_path):
        return {}
    values: dict[str, str] = {}
    with open(env_path, encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    return values


def _read_object_memory_curl_example(path: str) -> tuple[str, str]:
    config_path = os.path.abspath(path)
    if not os.path.isfile(config_path):
        return "", ""
    with open(config_path, encoding="utf-8") as stream:
        text = stream.read()
    url_match = re.search(r"curl\s+['\"](https://[^'\"]+/bundle)(?:\?[^'\"]*)?['\"]", text)
    key_match = re.search(r"X-API-Key:\s*([^\s'\"]+)", text)
    if not url_match or not key_match:
        return "", ""
    parsed = urllib.parse.urlparse(url_match.group(1))
    return f"{parsed.scheme}://{parsed.netloc}", key_match.group(1)
