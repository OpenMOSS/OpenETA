"""Controlled simulator asset references for VLM-guided target localization."""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from fnmatch import fnmatchcase
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from adapter.protocol import JsonDict
from agent.runtime.artifact_paths import artifact_session_id, artifact_session_root
from agent.runtime.reference_localization import ReferencePointLocalizer
from agent.tools.object_memory import (
    OBJECT_MEMORY_BANK_API_KEY_ENV,
    OBJECT_MEMORY_BANK_SETUP_URL,
    OBJECT_MEMORY_BANK_URL_ENV,
    ObjectMemoryBankClient,
    ObjectMemoryResolutionError,
)
from agent.tools.registry import ToolExecutionContext, ToolHandler, make_tool_result


DEFAULT_ASSET_REFERENCE_OUTPUT_ROOT = Path("tmp") / "image" / "asset_reference"
DEFAULT_ASSET_REFERENCE_MAX_BYTES = 8 * 1024 * 1024
DEFAULT_ASSET_REFERENCE_TIMEOUT_S = 15.0
AssetDownloader = Callable[[str, float, int], bytes]
ASSET_REFERENCE_CATALOG_ENV = "OPENETA_ASSET_REFERENCE_CATALOG_PATH"


@dataclass(frozen=True, slots=True)
class ResolvedAssetReferences:
    environment: str
    target_object: str
    references: tuple[JsonDict, ...]


class AssetReferenceCatalog:
    """Resolve environment/object aliases to catalog-owned reference sources."""

    def __init__(self, *, path: Path, payload: JsonDict) -> None:
        self.path = path.resolve()
        self.payload = payload
        if payload.get("schema_version") != "openeta.asset_reference_catalog.v1":
            raise ValueError("unsupported asset reference catalog schema_version")
        environments = payload.get("environments")
        if not isinstance(environments, list):
            raise ValueError("asset reference catalog environments must be a list")

    @classmethod
    def load(cls, path: str | Path) -> "AssetReferenceCatalog":
        catalog_path = Path(path)
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("asset reference catalog must be a JSON object")
        return cls(path=catalog_path, payload=payload)

    def resolve(self, *, environment: str, target_object: str) -> ResolvedAssetReferences:
        environment_entry = self._environment_entry(environment)
        if environment_entry is None:
            raise LookupError(f"environment is not present in asset catalog: {environment}")
        object_entry = self._object_entry(environment_entry, target_object)
        if object_entry is None:
            raise LookupError(f"target object is not present in asset catalog: {target_object}")
        references = object_entry.get("references")
        if not isinstance(references, list) or not references:
            raise LookupError(f"target object has no asset references: {target_object}")
        parsed = tuple(dict(item) for item in references if isinstance(item, dict))
        if not parsed:
            raise LookupError(f"target object has no valid asset references: {target_object}")
        return ResolvedAssetReferences(
            environment=str(environment_entry.get("id") or environment),
            target_object=str(object_entry.get("id") or target_object),
            references=parsed,
        )

    def materialize_reference(
        self,
        reference: JsonDict,
        *,
        output_path: Path,
        downloader: AssetDownloader,
        timeout_s: float,
        max_bytes: int,
    ) -> Path:
        url = str(reference.get("url") or "").strip()
        local_path = str(reference.get("path") or "").strip()
        if bool(url) == bool(local_path):
            raise ValueError("each asset reference must define exactly one of url or path")
        if url:
            self._validate_url(url)
            raw = downloader(url, timeout_s, max_bytes)
        else:
            candidate = (self.path.parent / local_path).resolve()
            if not candidate.is_relative_to(self.path.parent):
                raise ValueError("asset reference path escapes the catalog directory")
            raw = candidate.read_bytes()
            if len(raw) > max_bytes:
                raise ValueError("asset reference exceeds the configured byte limit")
        return _normalize_reference_image(raw, output_path=output_path)

    def _environment_entry(self, environment: str) -> JsonDict | None:
        normalized = _normalize_alias(environment)
        for entry in self.payload.get("environments", []):
            if not isinstance(entry, dict):
                continue
            patterns = [entry.get("id"), *(entry.get("aliases") or [])]
            if any(
                isinstance(pattern, str) and fnmatchcase(normalized, _normalize_alias(pattern))
                for pattern in patterns
            ):
                return entry
        return None

    @staticmethod
    def _object_entry(environment: JsonDict, target_object: str) -> JsonDict | None:
        normalized = _normalize_alias(target_object)
        objects = environment.get("objects")
        if not isinstance(objects, list):
            return None
        for entry in objects:
            if not isinstance(entry, dict):
                continue
            aliases = [entry.get("id"), *(entry.get("aliases") or [])]
            if normalized in {
                _normalize_alias(alias) for alias in aliases if isinstance(alias, str)
            }:
                return entry
        return None

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("asset reference URLs must use https")
        allowed_hosts = self.payload.get("allowed_hosts")
        if not isinstance(allowed_hosts, list) or not allowed_hosts:
            raise ValueError("remote asset references require a non-empty allowed_hosts list")
        hostname = parsed.hostname.lower().rstrip(".")
        allowed = {
            str(host).lower().rstrip(".")
            for host in allowed_hosts
            if isinstance(host, str) and host.strip()
        }
        if hostname not in allowed:
            raise ValueError(f"asset reference host is not allowlisted: {hostname}")


def load_configured_asset_reference_catalog(
    path: str | Path | None = None,
) -> AssetReferenceCatalog | None:
    """Load the configured catalog, or return None when the feature is unconfigured."""

    configured = str(path or os.environ.get(ASSET_REFERENCE_CATALOG_ENV, "")).strip()
    if not configured:
        return None
    return AssetReferenceCatalog.load(configured)


def build_asset_reference_handler(
    catalog: AssetReferenceCatalog,
    *,
    output_root: str | Path = DEFAULT_ASSET_REFERENCE_OUTPUT_ROOT,
    downloader: AssetDownloader | None = None,
    timeout_s: float = DEFAULT_ASSET_REFERENCE_TIMEOUT_S,
    max_bytes: int = DEFAULT_ASSET_REFERENCE_MAX_BYTES,
) -> ToolHandler:
    """Build a read-only handler that materializes catalog-owned reference images."""

    resolved_root = Path(output_root)
    fetch = downloader or _download_asset_reference

    def handler(context: ToolExecutionContext):
        environment = str(context.parameters.get("environment") or "").strip()
        target_object = str(context.parameters.get("target_object") or "").strip()
        scene_image = str(context.parameters.get("scene_image") or "").strip()
        if not environment or not target_object or not scene_image:
            return make_tool_result(
                context,
                success=False,
                content=(
                    "Asset reference lookup requires environment, target_object, and scene_image."
                ),
                outputs={"reason": "missing_parameters"},
                diagnostics=[{"code": "missing_parameters"}],
            )
        if not Path(scene_image).is_file():
            return make_tool_result(
                context,
                success=False,
                content="Asset reference lookup failed: scene image not found.",
                outputs={"reason": "scene_image_not_found"},
                diagnostics=[{"code": "scene_image_not_found"}],
            )
        try:
            resolved = catalog.resolve(
                environment=environment,
                target_object=target_object,
            )
            session_root = artifact_session_root(
                resolved_root,
                artifact_session_id(context.metadata),
            )
            run_dir = session_root / f"reference-{uuid4().hex[:12]}"
            run_dir.mkdir(parents=True, exist_ok=False)
            paths = [
                catalog.materialize_reference(
                    reference,
                    output_path=run_dir / f"reference_{index:03d}.png",
                    downloader=fetch,
                    timeout_s=timeout_s,
                    max_bytes=max_bytes,
                )
                for index, reference in enumerate(resolved.references)
            ]
        except (LookupError, OSError, ValueError) as exc:
            return make_tool_result(
                context,
                success=False,
                content=f"Asset reference lookup failed: {exc}",
                outputs={"reason": "asset_reference_unavailable"},
                diagnostics=[
                    {
                        "code": "asset_reference_unavailable",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            )

        reference_images = [str(path) for path in paths]
        bundle = {
            "scene_image_ref": scene_image,
            "reference_image_refs": reference_images,
            "environment": resolved.environment,
            "target_object": resolved.target_object,
            "bbox_coordinate_space": "original_image_pixels_xyxy_right_bottom_exclusive",
        }
        artifacts = [
            {
                "type": "asset_reference_image",
                "kind": "rgb",
                "tool": context.name,
                "index": index,
                "path": str(path),
                "environment": resolved.environment,
                "target_object": resolved.target_object,
            }
            for index, path in enumerate(paths)
        ]
        return make_tool_result(
            context,
            success=True,
            content=f"Materialized {len(paths)} controlled asset reference image(s).",
            outputs={
                "environment": resolved.environment,
                "target_object": resolved.target_object,
                "scene_image": scene_image,
                "reference_images": reference_images,
                "localization_bundle": bundle,
            },
            artifacts=artifacts,
        )

    return handler


def build_object_memory_reference_handler(
    client: ObjectMemoryBankClient,
    localizer: ReferencePointLocalizer,
    *,
    output_root: str | Path = DEFAULT_ASSET_REFERENCE_OUTPUT_ROOT,
) -> ToolHandler:
    """Build a reference tool backed by object memory and an isolated VLM localizer."""

    resolved_root = Path(output_root)

    def handler(context: ToolExecutionContext):
        environment = str(context.parameters.get("environment") or "").strip()
        target_object = str(context.parameters.get("target_object") or "").strip()
        scene_image = str(context.parameters.get("scene_image") or "").strip()
        if not environment or not target_object or not scene_image:
            return make_tool_result(
                context,
                success=False,
                content=(
                    "Object memory localization requires environment, target_object, "
                    "and scene_image."
                ),
                outputs={"reason": "missing_parameters"},
                diagnostics=[{"code": "missing_parameters"}],
            )
        scene_path = Path(scene_image)
        if not scene_path.is_file():
            return make_tool_result(
                context,
                success=False,
                content="Object memory localization failed: scene image not found.",
                outputs={"reason": "scene_image_not_found"},
                diagnostics=[{"code": "scene_image_not_found"}],
            )
        try:
            resolver = getattr(client, "resolve", None)
            bundle = (
                resolver(
                    environment=environment,
                    target_object=target_object,
                )
                if callable(resolver)
                else client.retrieve(
                    environment=environment,
                    target_object=target_object,
                )
            )
            session_root = artifact_session_root(
                resolved_root,
                artifact_session_id(context.metadata),
            )
            run_dir = session_root / f"reference-{uuid4().hex[:12]}"
            run_dir.mkdir(parents=True, exist_ok=False)
            reference_paths = [
                _normalize_reference_image(
                    reference.image_bytes,
                    output_path=run_dir / f"reference_{reference.view}.png",
                )
                for reference in bundle.references
            ]
            scene_size = _image_size(scene_path)
            localized = localizer.localize(
                environment=environment,
                target_object=target_object,
                scene_image=scene_path,
                reference_images=reference_paths,
                image_size=scene_size,
            )
            marker_path = _draw_reference_point_marker(
                scene_path,
                output_path=run_dir / "scene_target_point.png",
                point=(localized.x, localized.y),
            )
        except ObjectMemoryResolutionError as exc:
            candidates = [candidate.to_dict() for candidate in exc.candidates]
            return make_tool_result(
                context,
                success=False,
                content=f"Object memory asset resolution failed: {exc}",
                outputs={
                    "reason": "object_memory_resolution_failed",
                    "resolution_code": exc.code,
                    "search_candidates": candidates,
                },
                diagnostics=[
                    {
                        "code": "object_memory_resolution_failed",
                        "resolution_code": exc.code,
                        "message": str(exc),
                        "search_candidates": candidates,
                    }
                ],
            )
        except Exception as exc:  # noqa: BLE001 - network/VLM failures stay structured.
            return make_tool_result(
                context,
                success=False,
                content=f"Object memory localization failed: {exc}",
                outputs={"reason": "object_memory_localization_failed"},
                diagnostics=[
                    {
                        "code": "object_memory_localization_failed",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            )

        references = [str(path) for path in reference_paths]
        positive_points = [localized.as_prompt_point()]
        resolved_asset_key = (
            bundle.resolved_key
            or str(bundle.manifest.get("key") or "")
            or f"{bundle.namespace}/{bundle.asset_id}"
        )
        resolution = bundle.resolution.to_dict() if bundle.resolution is not None else None
        bbox_xyxy = (
            [float(value) for value in localized.bbox_xyxy]
            if localized.bbox_xyxy is not None
            else None
        )
        localization_bundle = {
            "scene_image_ref": scene_image,
            "reference_image_refs": references,
            "marked_scene_image_ref": str(marker_path),
            "environment": bundle.namespace,
            "target_object": bundle.asset_id,
            "memory_query_key": bundle.query_key,
            "resolved_asset_key": resolved_asset_key,
            "memory_resolution": resolution,
            "positive_points": positive_points,
            "bbox_xyxy": bbox_xyxy,
            "point_coordinate_space": "original_image_pixels_top_left_xy",
            "required_sam3_parameter": "positive_points",
        }
        artifacts = [
            {
                "type": "asset_reference_image",
                "kind": "rgb",
                "tool": context.name,
                "index": reference.view,
                "path": str(path),
                "environment": bundle.namespace,
                "target_object": bundle.asset_id,
                "memory_query_key": bundle.query_key,
                "resolved_asset_key": resolved_asset_key,
            }
            for reference, path in zip(bundle.references, reference_paths, strict=True)
        ]
        artifacts.append(
            {
                "type": "reference_target_point",
                "kind": "rgb",
                "tool": context.name,
                "path": str(marker_path),
                "source_image": scene_image,
                "positive_points": positive_points,
                "bbox_xyxy": bbox_xyxy,
            }
        )
        return make_tool_result(
            context,
            success=True,
            content=(
                f"Resolved {bundle.query_key} to {resolved_asset_key} and localized it "
                f"from {len(reference_paths)} object-memory reference views."
            ),
            outputs={
                "environment": bundle.namespace,
                "target_object": bundle.asset_id,
                "memory_query_key": bundle.query_key,
                "resolved_asset_key": resolved_asset_key,
                "memory_resolution": resolution,
                "scene_image": scene_image,
                "reference_images": references,
                "marked_scene_image": str(marker_path),
                "positive_points": positive_points,
                "bbox_xyxy": bbox_xyxy,
                "localization_bundle": localization_bundle,
                "localizer": {
                    "provider": localized.provider,
                    "model": localized.model,
                    "confidence": localized.confidence,
                    "reason": localized.reason,
                    **dict(localized.details or {}),
                },
            },
            artifacts=artifacts,
        )

    return handler


def build_object_memory_configuration_warning_handler(
    *,
    configuration_error: str = "",
    setup_url: str = OBJECT_MEMORY_BANK_SETUP_URL,
) -> ToolHandler:
    """Return a call-time warning when the object-memory service is unavailable."""

    message = (
        "WARNING: Object Memory Bank URL is not configured. Setup: "
        f"{setup_url}. Set {OBJECT_MEMORY_BANK_URL_ENV} and "
        f"{OBJECT_MEMORY_BANK_API_KEY_ENV} together."
    )

    def handler(context: ToolExecutionContext):
        warning = {
            "code": "object_memory_bank_unconfigured",
            "severity": "warning",
            "message": message,
            "setup_url": setup_url,
            "required_environment_variables": [
                OBJECT_MEMORY_BANK_URL_ENV,
                OBJECT_MEMORY_BANK_API_KEY_ENV,
            ],
        }
        if configuration_error:
            warning["configuration_error"] = configuration_error
        return make_tool_result(
            context,
            success=False,
            content=message,
            outputs={
                "reason": "object_memory_bank_unconfigured",
                "warning": warning,
            },
            diagnostics=[warning],
        )

    return handler


def _download_asset_reference(url: str, timeout_s: float, max_bytes: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "OpenETA-AssetReference/1.0"},
        method="GET",
    )
    opener = urllib.request.build_opener(_RejectRedirects())
    with opener.open(request, timeout=timeout_s) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            raise ValueError("asset reference exceeds the configured byte limit")
        raw = response.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("asset reference exceeds the configured byte limit")
    return raw


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        del request, file_pointer, code, message, headers, new_url
        return None


def _normalize_reference_image(raw: bytes, *, output_path: Path) -> Path:
    from PIL import Image

    if not raw:
        raise ValueError("asset reference image is empty")
    with Image.open(BytesIO(raw)) as loaded:
        loaded.verify()
    with Image.open(BytesIO(raw)) as loaded:
        image = loaded.convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return output_path


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def _draw_reference_point_marker(
    source_path: Path,
    *,
    output_path: Path,
    point: tuple[float, float],
) -> Path:
    from PIL import Image, ImageDraw

    with Image.open(source_path) as loaded:
        image = loaded.convert("RGB")
    x = min(image.width - 1, max(0, int(round(point[0]))))
    y = min(image.height - 1, max(0, int(round(point[1]))))
    radius = max(5, min(image.size) // 64)
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=(255, 40, 40),
        outline=(255, 255, 255),
        width=max(2, radius // 3),
    )
    cross = radius + max(4, radius // 2)
    draw.line((x - cross, y, x + cross, y), fill=(255, 255, 255), width=2)
    draw.line((x, y - cross, x, y + cross), fill=(255, 255, 255), width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return output_path


def _normalize_alias(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())
