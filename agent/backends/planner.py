"""Backends for closed-loop LLM/VLM planner decisions."""

from __future__ import annotations

import base64
import json
import mimetypes
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from adapter.protocol import JsonDict
from agent.runtime.actions import PipelineStatus
from agent.backends.provider_config import PlannerProviderConfig, ProviderEndpointConfig
from agent.runtime.token_counting import estimate_json_tokens, estimate_text_tokens


@dataclass(slots=True)
class PlannerBackendRequest:
    """Request sent to a model or deterministic planner backend."""

    tool_context: JsonDict
    system_prompt: str = ""
    conversation_messages: list[JsonDict] = field(default_factory=list)
    conversation_summary: str = ""
    attempt: int = 1
    validation_errors: list[str] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class PlannerBackendResult:
    """Raw decision returned by a planner backend before schema validation."""

    payload: JsonDict | str
    status: PipelineStatus = PipelineStatus.PLANNED
    provider: str = "placeholder"
    model: str = "none"
    details: JsonDict = field(default_factory=dict)
    rollout_exchange: JsonDict = field(default_factory=dict)


class PlannerBackend(ABC):
    """Backend that chooses the next closed-loop action from tool context.

    Implementations may call a commercial LLM/VLM API, a local model, or a
    deterministic test fixture. They only return structured decision payloads;
    tool execution remains owned by `ActionPipeline` and `ToolRegistry`.
    """

    @abstractmethod
    def decide(self, request: PlannerBackendRequest) -> PlannerBackendResult:
        """Return one raw planner decision payload."""

    def descriptor(self) -> JsonDict:
        return {
            "name": type(self).__name__,
            "implemented": not isinstance(self, PlaceholderPlannerBackend),
        }


class ProviderQueueTimeoutError(RuntimeError):
    """Raised before a provider call when the shared batch queue times out."""

    code = "provider_queue_timeout"


class ProviderConcurrencyLimiter:
    """Thread-safe concurrency and queue metrics shared by planner backends."""

    def __init__(self, limit: int, *, queue_timeout_s: float) -> None:
        if limit < 1:
            raise ValueError("provider concurrency limit must be positive")
        if queue_timeout_s <= 0:
            raise ValueError("provider queue timeout must be positive")
        self.limit = limit
        self.queue_timeout_s = queue_timeout_s
        self._semaphore = threading.BoundedSemaphore(limit)
        self._lock = threading.Lock()
        self._request_count = 0
        self._queue_timeout_count = 0
        self._active = 0
        self._max_active = 0
        self._total_queue_wait_s = 0.0
        self._max_queue_wait_s = 0.0

    def wrap(self, backend: PlannerBackend) -> PlannerBackend:
        return ConcurrencyLimitedPlannerBackend(backend, limiter=self)

    def snapshot(self) -> JsonDict:
        with self._lock:
            return {
                "schema_version": "openeta.provider_concurrency.v1",
                "limit": self.limit,
                "queue_timeout_s": self.queue_timeout_s,
                "request_count": self._request_count,
                "queue_timeout_count": self._queue_timeout_count,
                "active": self._active,
                "max_active": self._max_active,
                "total_queue_wait_s": round(self._total_queue_wait_s, 6),
                "max_queue_wait_s": round(self._max_queue_wait_s, 6),
            }

    def _acquire(self) -> float:
        started = time.monotonic()
        acquired = self._semaphore.acquire(timeout=self.queue_timeout_s)
        waited_s = time.monotonic() - started
        with self._lock:
            self._request_count += 1
            self._total_queue_wait_s += waited_s
            self._max_queue_wait_s = max(self._max_queue_wait_s, waited_s)
            if not acquired:
                self._queue_timeout_count += 1
            else:
                self._active += 1
                self._max_active = max(self._max_active, self._active)
        if not acquired:
            raise ProviderQueueTimeoutError(
                f"provider queue wait exceeded {self.queue_timeout_s:g}s "
                f"with concurrency limit {self.limit}"
            )
        return waited_s

    def _release(self) -> None:
        with self._lock:
            self._active -= 1
        self._semaphore.release()


class ConcurrencyLimitedPlannerBackend(PlannerBackend):
    """Planner backend wrapper using one shared provider concurrency limiter."""

    def __init__(self, backend: PlannerBackend, *, limiter: ProviderConcurrencyLimiter) -> None:
        self.backend = backend
        self.limiter = limiter

    def decide(self, request: PlannerBackendRequest) -> PlannerBackendResult:
        waited_s = self.limiter._acquire()
        try:
            result = self.backend.decide(request)
        finally:
            self.limiter._release()
        result.details = {
            **result.details,
            "provider_queue_wait_s": round(waited_s, 6),
            "provider_concurrency": self.limiter.snapshot(),
        }
        return result

    def descriptor(self) -> JsonDict:
        return {
            "name": type(self).__name__,
            "implemented": True,
            "backend": self.backend.descriptor(),
            "provider_concurrency": self.limiter.snapshot(),
        }


class PlaceholderPlannerBackend(PlannerBackend):
    """Backend used until a real LLM/VLM provider is connected."""

    def decide(self, request: PlannerBackendRequest) -> PlannerBackendResult:
        del request
        return PlannerBackendResult(
            payload={
                "kind": "response",
                "name": "talk",
                "parameters": {},
                "reasoning": "No LLM/VLM planner backend is connected yet.",
            },
            status=PipelineStatus.PENDING,
            details={"backend_status": "pending"},
        )


class StaticPlannerBackend(PlannerBackend):
    """Deterministic backend useful for examples and tests."""

    def __init__(self, payloads: list[JsonDict | str] | JsonDict | str) -> None:
        if isinstance(payloads, list):
            self._payloads = list(payloads)
        else:
            self._payloads = [payloads]
        self._idx = 0

    def decide(self, request: PlannerBackendRequest) -> PlannerBackendResult:
        del request
        if not self._payloads:
            payload: JsonDict = {
                "kind": "response",
                "name": "talk",
                "parameters": {},
                "reasoning": "Static backend has no payloads.",
            }
        else:
            payload = self._payloads[min(self._idx, len(self._payloads) - 1)]
            self._idx += 1
        return PlannerBackendResult(
            payload=payload,
            provider="static",
            model="fixture",
        )

    def descriptor(self) -> JsonDict:
        payload = super().descriptor()
        payload.update({"provider": "static", "remaining_payloads": len(self._payloads)})
        return payload


PlannerBackendCallable = Callable[[PlannerBackendRequest], PlannerBackendResult | JsonDict | str]


class CallablePlannerBackend(PlannerBackend):
    """Adapter for SDK/API wrappers that already return JSON-like decisions."""

    def __init__(
        self,
        fn: PlannerBackendCallable,
        *,
        provider: str = "callable",
        model: str = "custom",
        metadata: JsonDict | None = None,
    ) -> None:
        self.fn = fn
        self.provider = provider
        self.model = model
        self.metadata = dict(metadata or {})

    def decide(self, request: PlannerBackendRequest) -> PlannerBackendResult:
        value = self.fn(request)
        if isinstance(value, PlannerBackendResult):
            return value
        return PlannerBackendResult(
            payload=value,
            provider=self.provider,
            model=self.model,
            details={"metadata": self.metadata},
        )

    def descriptor(self) -> JsonDict:
        payload = super().descriptor()
        payload.update(
            {
                "provider": self.provider,
                "model": self.model,
                "metadata": self.metadata,
            }
        )
        return payload


@dataclass(slots=True)
class CommercialApiPlannerBackendConfig:
    """Configuration placeholder for an API-backed LLM/VLM planner."""

    provider: str
    model: str
    api_base: str | None = None
    api_key_env: str | None = None
    timeout_s: float = 60.0
    metadata: JsonDict = field(default_factory=dict)


class CommercialApiPlannerBackend(PlannerBackend):
    """Shape of a future commercial-API backed tool-calling planner.

    The network implementation is intentionally absent. The runtime can depend
    on this boundary now while provider/model/auth details are filled in later.
    """

    def __init__(self, config: CommercialApiPlannerBackendConfig) -> None:
        self.config = config

    def decide(self, request: PlannerBackendRequest) -> PlannerBackendResult:
        del request
        raise NotImplementedError(
            "Commercial API planner backend is not wired yet. Implement "
            "provider/model/auth handling behind PlannerBackend."
        )

    def descriptor(self) -> JsonDict:
        payload = super().descriptor()
        payload.update(
            {
                "provider": self.config.provider,
                "model": self.config.model,
                "api_base": self.config.api_base,
                "api_key_env": self.config.api_key_env,
            }
        )
        return payload


OpenAICompatibleTransport = Callable[[str, JsonDict, dict[str, str], float], JsonDict]


@dataclass(slots=True)
class OpenAICompatiblePlannerBackendConfig:
    """Config for an OpenAI-compatible `/v1/chat/completions` planner backend."""

    provider: str = "openai-compatible"
    model: str = ""
    api_base: str = ""
    api_key: str = ""
    timeout_s: float = 60.0
    max_attempts: int = 3
    retry_backoff_s: float = 0.5
    temperature: float = 0.0
    max_tokens: int = 512
    context_window_tokens: int | None = None
    use_json_response_format: bool = True
    enable_vision: bool = True
    max_vision_images: int = 2
    max_vision_image_bytes: int = 8 * 1024 * 1024
    fallback: ProviderEndpointConfig | None = None
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_provider_config(
        cls, config: PlannerProviderConfig
    ) -> "OpenAICompatiblePlannerBackendConfig":
        metadata = dict(config.metadata)
        return cls(
            provider=config.provider,
            model=config.model,
            api_base=config.api_base,
            api_key=config.api_key,
            timeout_s=config.timeout_s,
            max_attempts=config.max_attempts,
            retry_backoff_s=config.retry_backoff_s,
            context_window_tokens=config.context_window_tokens,
            enable_vision=_metadata_bool(metadata, "enable_vision", default=True),
            max_vision_images=_metadata_positive_int(
                metadata,
                "max_vision_images",
                default=2,
            ),
            max_vision_image_bytes=_metadata_positive_int(
                metadata,
                "max_vision_image_bytes",
                default=8 * 1024 * 1024,
            ),
            fallback=config.fallback,
            metadata=metadata,
        )

    def missing_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.model:
            missing.append("model")
        if not self.api_base:
            missing.append("api_base")
        if not self.api_key:
            missing.append("api_key")
        if self.fallback is not None:
            missing.extend(f"fallback.{field}" for field in self.fallback.missing_fields())
        return missing

    def redacted(self) -> JsonDict:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_base": self.api_base,
            "api_key": _redact_secret(self.api_key),
            "timeout_s": self.timeout_s,
            "max_attempts": self.max_attempts,
            "retry_backoff_s": self.retry_backoff_s,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "context_window_tokens": self.context_window_tokens,
            "use_json_response_format": self.use_json_response_format,
            "enable_vision": self.enable_vision,
            "max_vision_images": self.max_vision_images,
            "max_vision_image_bytes": self.max_vision_image_bytes,
            "fallback": self.fallback.redacted() if self.fallback is not None else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class OpenAICompatibleModelInfo:
    """Best-effort model metadata from an OpenAI-compatible `/v1/models` endpoint."""

    id: str
    context_window_tokens: int | None = None
    metadata: JsonDict = field(default_factory=dict)


class OpenAICompatiblePlannerBackend(PlannerBackend):
    """LLM planner backend using an OpenAI-compatible chat-completions API."""

    def __init__(
        self,
        config: OpenAICompatiblePlannerBackendConfig,
        *,
        transport: OpenAICompatibleTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.transport = transport or _post_json
        self.sleep = sleep
        self._preferred_provider_role = "primary"

    def decide(self, request: PlannerBackendRequest) -> PlannerBackendResult:
        missing = self.config.missing_fields()
        if missing:
            return PlannerBackendResult(
                payload={
                    "kind": "response",
                    "name": "ask_human",
                    "parameters": {
                        "message": "Planner provider config is incomplete.",
                        "missing_fields": missing,
                    },
                    "reasoning": "Planner provider config is incomplete.",
                },
                status=PipelineStatus.FAILED,
                provider=self.config.provider,
                model=self.config.model or "unknown",
                details={"missing_fields": missing},
            )

        user_content, vision_attachments = _planner_user_content(request, self.config)
        messages: list[JsonDict] = [
            {"role": "system", "content": request.system_prompt},
        ]
        if request.conversation_summary.strip():
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Earlier OpenETA execution summary from the current session:\n"
                        + request.conversation_summary.strip()
                    ),
                }
            )
        messages.extend(_validated_conversation_messages(request.conversation_messages))
        messages.append({"role": "user", "content": user_content})
        body: JsonDict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.use_json_response_format:
            body["response_format"] = {"type": "json_object"}

        (
            response,
            provider_attempts,
            retry_errors,
            final_error,
            active_endpoint,
            provider_role,
            provider_exchanges,
        ) = self._request_with_retry(
            body,
        )
        provider_switch_count = sum(
            bool(error.get("switch_provider_next")) for error in retry_errors
        )
        if final_error is not None:
            return PlannerBackendResult(
                payload={
                    "kind": "response",
                    "name": "ask_human",
                    "parameters": {
                        "message": "Planner provider request failed.",
                        "error_type": type(final_error).__name__,
                        "provider_attempts": provider_attempts,
                    },
                    "reasoning": f"Planner provider request failed: {final_error}",
                },
                status=PipelineStatus.FAILED,
                provider=active_endpoint.provider,
                model=active_endpoint.model,
                details={
                    "error_type": type(final_error).__name__,
                    "error": str(final_error),
                    "provider_attempts": provider_attempts,
                    "retry_errors": retry_errors,
                    "provider_role": provider_role,
                    "provider_failover": provider_switch_count > 0,
                    "provider_switch_count": provider_switch_count,
                },
                rollout_exchange={"attempts": provider_exchanges},
            )
        assert response is not None

        usage_body = dict(body)
        usage_body["model"] = active_endpoint.model
        content = _extract_chat_content(response)
        usage, usage_source, usage_estimator = _provider_or_estimated_usage(
            response.get("usage"),
            request_body=_usage_estimation_body(usage_body),
            completion=content,
            model=active_endpoint.model,
        )
        return PlannerBackendResult(
            payload=content,
            status=PipelineStatus.PLANNED,
            provider=active_endpoint.provider,
            model=active_endpoint.model,
            details={
                "finish_reason": _extract_finish_reason(response),
                "usage": usage,
                "usage_source": usage_source,
                "usage_estimator": usage_estimator,
                "response_id": response.get("id"),
                "vision_attachments": vision_attachments,
                "provider_attempts": provider_attempts,
                "retry_errors": retry_errors,
                "provider_role": provider_role,
                "provider_failover": provider_switch_count > 0,
                "provider_switch_count": provider_switch_count,
            },
            rollout_exchange={"attempts": provider_exchanges},
        )

    def _request_with_retry(
        self,
        body: JsonDict,
    ) -> tuple[
        JsonDict | None,
        int,
        list[JsonDict],
        Exception | None,
        ProviderEndpointConfig,
        str,
        list[JsonDict],
    ]:
        max_attempts = max(1, self.config.max_attempts)
        retry_errors: list[JsonDict] = []
        provider_exchanges: list[JsonDict] = []
        primary_endpoint = ProviderEndpointConfig(
            provider=self.config.provider,
            model=self.config.model,
            api_base=self.config.api_base,
            api_key=self.config.api_key,
            timeout_s=self.config.timeout_s,
        )
        if self._preferred_provider_role == "fallback" and self.config.fallback is not None:
            active_endpoint = self.config.fallback
            provider_role = "fallback"
        else:
            active_endpoint = primary_endpoint
            provider_role = "primary"
        for attempt in range(1, max_attempts + 1):
            url = _chat_completions_url(active_endpoint.api_base)
            attempt_body = dict(body)
            attempt_body["model"] = active_endpoint.model
            headers = {
                "Authorization": f"Bearer {active_endpoint.api_key}",
                "Content-Type": "application/json",
            }
            started_at_s = time.time()
            try:
                response = self.transport(
                    url,
                    attempt_body,
                    headers,
                    active_endpoint.timeout_s,
                )
                completed_at_s = time.time()
                provider_exchanges.append(
                    {
                        "attempt": attempt,
                        "provider_role": provider_role,
                        "provider": active_endpoint.provider,
                        "model": active_endpoint.model,
                        "url": url,
                        "timeout_s": active_endpoint.timeout_s,
                        "started_at_s": started_at_s,
                        "completed_at_s": completed_at_s,
                        "duration_s": max(0.0, completed_at_s - started_at_s),
                        "request_body": attempt_body,
                        "response": response,
                    }
                )
                self._preferred_provider_role = provider_role
                return (
                    response,
                    attempt,
                    retry_errors,
                    None,
                    active_endpoint,
                    provider_role,
                    provider_exchanges,
                )
            except Exception as exc:  # noqa: BLE001 - provider failures stay structured.
                completed_at_s = time.time()
                switch_provider_next = (
                    self.config.fallback is not None and _is_provider_failover_error(exc)
                )
                next_provider_role = None
                if switch_provider_next:
                    next_provider_role = "fallback" if provider_role == "primary" else "primary"
                retryable = _is_transient_provider_error(exc) or switch_provider_next
                provider_exchanges.append(
                    {
                        "attempt": attempt,
                        "provider_role": provider_role,
                        "provider": active_endpoint.provider,
                        "model": active_endpoint.model,
                        "url": url,
                        "timeout_s": active_endpoint.timeout_s,
                        "started_at_s": started_at_s,
                        "completed_at_s": completed_at_s,
                        "duration_s": max(0.0, completed_at_s - started_at_s),
                        "request_body": attempt_body,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                )
                if attempt >= max_attempts or not retryable:
                    return (
                        None,
                        attempt,
                        retry_errors,
                        exc,
                        active_endpoint,
                        provider_role,
                        provider_exchanges,
                    )
                delay_s = max(0.0, self.config.retry_backoff_s) * (2 ** (attempt - 1))
                retry_errors.append(
                    {
                        "attempt": attempt,
                        "provider_role": provider_role,
                        "provider": active_endpoint.provider,
                        "model": active_endpoint.model,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "retry_delay_s": delay_s,
                        "failover_next": next_provider_role == "fallback",
                        "switch_provider_next": switch_provider_next,
                        "next_provider_role": next_provider_role,
                    }
                )
                if next_provider_role == "fallback":
                    active_endpoint = self.config.fallback
                    provider_role = next_provider_role
                elif next_provider_role == "primary":
                    active_endpoint = primary_endpoint
                    provider_role = next_provider_role
                if delay_s:
                    self.sleep(delay_s)
        raise AssertionError("provider retry loop exited unexpectedly")

    def descriptor(self) -> JsonDict:
        payload = super().descriptor()
        payload.update(self.config.redacted())
        return payload


def list_openai_compatible_models(
    config: OpenAICompatiblePlannerBackendConfig,
    *,
    timeout_s: float | None = None,
) -> list[str]:
    """Return model ids from an OpenAI-compatible `/v1/models` endpoint."""

    return [model.id for model in list_openai_compatible_model_info(config, timeout_s=timeout_s)]


def list_openai_compatible_model_info(
    config: OpenAICompatiblePlannerBackendConfig,
    *,
    timeout_s: float | None = None,
) -> list[OpenAICompatibleModelInfo]:
    """Return model metadata when the provider exposes non-standard fields.

    The OpenAI-compatible model-list shape is not strict about context-window
    metadata. Some providers expose fields such as `context_length`,
    `context_window`, or `max_context_tokens`; others expose only `id`.
    """

    missing = [field for field in ("api_base", "api_key") if not getattr(config, field)]
    if missing:
        raise ValueError(f"Missing OpenAI-compatible model-list fields: {missing}")
    url = f"{config.api_base.rstrip('/')}/v1/models"
    headers = {"Authorization": f"Bearer {config.api_key}"}
    response = _get_json(url, headers, timeout_s or config.timeout_s)
    data = response.get("data", [])
    if not isinstance(data, list):
        return []
    models: list[OpenAICompatibleModelInfo] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            models.append(
                OpenAICompatibleModelInfo(
                    id=item["id"],
                    context_window_tokens=extract_context_window_tokens(item),
                    metadata={
                        key: value
                        for key, value in item.items()
                        if key not in {"id", "object", "created", "owned_by"}
                    },
                )
            )
    return models


def resolve_openai_compatible_context_window_tokens(
    config: OpenAICompatiblePlannerBackendConfig,
    *,
    timeout_s: float | None = None,
) -> int | None:
    """Resolve the configured model's context window when metadata is available."""

    if config.context_window_tokens is not None:
        return config.context_window_tokens
    if not config.model:
        return None
    for model in list_openai_compatible_model_info(config, timeout_s=timeout_s):
        if model.id == config.model:
            return model.context_window_tokens
    return None


def extract_context_window_tokens(model_payload: JsonDict) -> int | None:
    """Best-effort extraction for common provider-specific context fields."""

    paths = (
        ("context_window_tokens",),
        ("context_window",),
        ("context_length",),
        ("max_context_tokens",),
        ("max_context_length",),
        ("max_model_len",),
        ("max_input_tokens",),
        ("input_token_limit",),
        ("metadata", "context_window_tokens"),
        ("metadata", "context_window"),
        ("metadata", "context_length"),
        ("limits", "context_window_tokens"),
        ("limits", "context_window"),
        ("limits", "max_context_tokens"),
    )
    for path in paths:
        value: object = model_payload
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        parsed = _coerce_positive_int(value)
        if parsed is not None:
            return parsed
    return None


def _planner_user_prompt(request: PlannerBackendRequest) -> str:
    instruction = (
        "Follow the system prompt for this isolated role. Return only the exact "
        "JSON object requested by that prompt, without markdown."
        if request.metadata.get("isolated_context") is True
        else (
            "Choose exactly one next OpenETA action. Return only JSON with "
            "fields: kind, name, parameters, reasoning. Do not include markdown."
        )
    )
    payload = {
        "instruction": instruction,
        "tool_context": request.tool_context,
        "attempt": request.attempt,
        "validation_errors": request.validation_errors,
    }
    return json.dumps(payload, ensure_ascii=False)


def _planner_user_content(
    request: PlannerBackendRequest,
    config: OpenAICompatiblePlannerBackendConfig,
) -> tuple[str | list[JsonDict], list[JsonDict]]:
    text = _planner_user_prompt(request)
    if not config.enable_vision:
        return text, []
    explicit_paths = request.tool_context.get("vision_image_paths")
    paths = (
        [value for value in explicit_paths if isinstance(value, str) and value]
        if isinstance(explicit_paths, list)
        else []
    )
    localization = request.tool_context.get("reference_localization_obligation")
    if isinstance(localization, dict):
        if localization.get("required_parameter") != "positive_points":
            scene_image = localization.get("scene_image")
            if isinstance(scene_image, str) and scene_image and scene_image not in paths:
                paths.append(scene_image)
            references = localization.get("reference_images")
            if isinstance(references, list):
                for value in references:
                    if isinstance(value, str) and value and value not in paths:
                        paths.append(value)
                    if len(paths) >= config.max_vision_images:
                        break
    else:
        obligation = request.tool_context.get("selection_obligation")
        if isinstance(obligation, dict):
            bundle = obligation.get("selection_bundle")
            if not isinstance(bundle, dict):
                bundle = {}
            for field in ("original_image_ref", "contact_sheet_ref"):
                value = bundle.get(field)
                if isinstance(value, str) and value and value not in paths:
                    paths.append(value)
            if len(paths) < config.max_vision_images:
                candidates = bundle.get("candidates")
                if not isinstance(candidates, list):
                    candidates = []
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    for field in ("overlay_ref", "crop_ref"):
                        value = candidate.get(field)
                        if isinstance(value, str) and value and value not in paths:
                            paths.append(value)
                        if len(paths) >= config.max_vision_images:
                            break
                    if len(paths) >= config.max_vision_images:
                        break
    if not paths:
        return text, []

    content: list[JsonDict] = [{"type": "text", "text": text}]
    attachments: list[JsonDict] = []
    evidence_roles: dict[str, str] = {}
    raw_evidence = request.tool_context.get("vision_evidence")
    if isinstance(raw_evidence, list):
        for item in raw_evidence:
            if not isinstance(item, dict):
                continue
            evidence_path = item.get("path")
            evidence_role = item.get("role")
            if isinstance(evidence_path, str) and isinstance(evidence_role, str):
                evidence_roles[evidence_path] = evidence_role
    for image_index, path_value in enumerate(paths[: config.max_vision_images], start=1):
        path = Path(path_value)
        try:
            size = path.stat().st_size
        except OSError as exc:
            attachments.append(
                {
                    "path": path_value,
                    "attached": False,
                    "reason": "unreadable",
                    "error_type": type(exc).__name__,
                }
            )
            continue
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if not mime_type.startswith("image/"):
            attachments.append(
                {
                    "path": path_value,
                    "attached": False,
                    "reason": "unsupported_media_type",
                    "mime_type": mime_type,
                }
            )
            continue
        if size > config.max_vision_image_bytes:
            attachments.append(
                {
                    "path": path_value,
                    "attached": False,
                    "reason": "image_too_large",
                    "bytes": size,
                }
            )
            continue
        try:
            raw_image = path.read_bytes()
        except OSError as exc:
            attachments.append(
                {
                    "path": path_value,
                    "attached": False,
                    "reason": "unreadable",
                    "error_type": type(exc).__name__,
                }
            )
            continue
        encoded = base64.b64encode(raw_image).decode("ascii")
        evidence_role = evidence_roles.get(path_value)
        if evidence_role:
            role_note = f"Image #{image_index} role: {evidence_role}."
            if evidence_role == "current_scene":
                role_note += " This is the current state used for action review."
            elif evidence_role == "target_source_before_grasp":
                role_note += " This is a historical baseline, not the current state."
            content.append({"type": "text", "text": role_note})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{encoded}",
                    "detail": "high",
                },
            }
        )
        attachments.append(
            {
                "path": path_value,
                "attached": True,
                "mime_type": mime_type,
                "bytes": size,
                **({"role": evidence_role} if evidence_role else {}),
            }
        )
    if len(content) == 1:
        return text, attachments
    return content, attachments


def _validated_conversation_messages(messages: list[JsonDict]) -> list[JsonDict]:
    validated: list[JsonDict] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            continue
        validated.append({"role": role, "content": content})
    return validated


def _usage_estimation_body(body: JsonDict) -> JsonDict:
    def scrub(value: object) -> object:
        if isinstance(value, str) and value.startswith("data:image/"):
            return "<inline_image_omitted_for_token_estimation>"
        if isinstance(value, dict):
            return {str(key): scrub(item) for key, item in value.items()}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    scrubbed = scrub(body)
    return scrubbed if isinstance(scrubbed, dict) else dict(body)


def _metadata_bool(metadata: JsonDict, key: str, *, default: bool) -> bool:
    value = metadata.get(key)
    return value if isinstance(value, bool) else default


def _metadata_positive_int(metadata: JsonDict, key: str, *, default: int) -> int:
    parsed = _coerce_positive_int(metadata.get(key))
    return parsed if parsed is not None else default


class ProviderHttpError(RuntimeError):
    """HTTP provider failure that preserves the status for retry classification."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code


def _is_transient_provider_error(exc: Exception) -> bool:
    if isinstance(exc, ProviderHttpError):
        return exc.status_code in {408, 429, 500, 502, 503, 504} or (520 <= exc.status_code <= 527)
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 429, 500, 502, 503, 504}
    return isinstance(exc, (TimeoutError, ConnectionError, urllib.error.URLError))


def _is_provider_failover_error(exc: Exception) -> bool:
    """Return whether a primary-provider failure should activate fallback."""

    if isinstance(exc, ProviderHttpError):
        if exc.status_code == 500 and _provider_error_reports_capacity(exc):
            return True
        return exc.status_code in {401, 403, 408, 429, 502, 503, 504} or (
            520 <= exc.status_code <= 527
        )
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {401, 403, 408, 429, 502, 503, 504} or 520 <= exc.code <= 527
    return isinstance(exc, (TimeoutError, ConnectionError, urllib.error.URLError))


def _provider_error_reports_capacity(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "overload",
            "overloaded",
            "capacity",
            "concurrency",
            "负载",
            "并发",
        )
    )


def _chat_completions_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _post_json(url: str, body: JsonDict, headers: dict[str, str], timeout_s: float) -> JsonDict:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise ProviderHttpError(exc.code, message) from exc


def _get_json(url: str, headers: dict[str, str], timeout_s: float) -> JsonDict:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {message}") from exc


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer():
        parsed = int(value)
        return parsed if parsed > 0 else None
    if isinstance(value, str):
        normalized = value.strip().replace(",", "").replace("_", "")
        if normalized.isdigit():
            parsed = int(normalized)
            return parsed if parsed > 0 else None
    return None


def _extract_chat_content(response: JsonDict) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Provider response did not include choices.")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("Provider response choice is not an object.")
    message = first.get("message", {})
    if not isinstance(message, dict):
        raise RuntimeError("Provider response choice did not include message.")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Provider response message content is empty.")
    return content


def _extract_finish_reason(response: JsonDict) -> str | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    finish_reason = first.get("finish_reason")
    return finish_reason if isinstance(finish_reason, str) else None


def _provider_or_estimated_usage(
    raw_usage: object,
    *,
    request_body: JsonDict,
    completion: str,
    model: str,
) -> tuple[JsonDict, str, JsonDict]:
    usage: JsonDict = {}
    if isinstance(raw_usage, dict):
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cached_tokens",
        ):
            parsed = _coerce_non_negative_int(raw_usage.get(key))
            if parsed is not None:
                usage[key] = parsed
        if "cached_tokens" not in usage:
            for details_key in ("prompt_tokens_details", "input_tokens_details"):
                details = raw_usage.get(details_key)
                if not isinstance(details, dict):
                    continue
                parsed = _coerce_non_negative_int(details.get("cached_tokens"))
                if parsed is not None:
                    usage["cached_tokens"] = parsed
                    break
    total = _coerce_positive_int(usage.get("total_tokens"))
    if total is not None:
        return usage, "provider", {}
    derived_total = int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
    if derived_total > 0:
        usage["total_tokens"] = derived_total
        return usage, "provider_derived", {}

    prompt_estimate = estimate_json_tokens(request_body, model=model)
    completion_estimate = estimate_text_tokens(
        completion,
        model=model,
        scope="planner_completion",
    )
    return (
        {
            "prompt_tokens": prompt_estimate.tokens,
            "completion_tokens": completion_estimate.tokens,
            "total_tokens": prompt_estimate.tokens + completion_estimate.tokens,
        },
        "estimated",
        {
            "prompt": prompt_estimate.estimator,
            "completion": completion_estimate.estimator,
        },
    )


def _coerce_non_negative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer():
        parsed = int(value)
        return parsed if parsed >= 0 else None
    if isinstance(value, str):
        normalized = value.strip().replace(",", "").replace("_", "")
        if normalized.isdigit():
            return int(normalized)
    return None


def _redact_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:3]}...{value[-4:]}"
