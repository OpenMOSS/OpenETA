"""Protocol boundary for the standalone manual VLM console.

The proxy imports only this module. Harness-specific request interpretation and
response encoding live in separately loaded adapters.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable


JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class EncodedResponse:
    """One OpenAI-compatible assistant result produced by an adapter."""

    message: JsonObject
    finish_reason: str = "stop"


@runtime_checkable
class ProtocolAdapter(Protocol):
    """Adapter contract between the generic proxy and a wire protocol."""

    adapter_id: str
    label: str

    def classify_request(self, body: JsonObject) -> JsonObject:
        """Return queue metadata for one provider request."""

    def session_identity(
        self,
        body: JsonObject,
        headers: Mapping[str, str],
    ) -> tuple[str, str]:
        """Return an explicit session id and its source, or empty strings."""

    def attempt(self, body: JsonObject) -> int:
        """Return the protocol retry attempt used for lineage inference."""

    def presentation(self, body: JsonObject, *, request_id: str) -> JsonObject:
        """Return the data-driven operator view and composer specification."""

    def audit_records(self, body: JsonObject) -> list[JsonObject]:
        """Return stable, deduplicatable protocol audit records visible in the body."""

    def encode_response(
        self,
        body: JsonObject,
        submission: JsonObject,
        *,
        request_id: str,
    ) -> EncodedResponse:
        """Encode a UI submission as an OpenAI-compatible assistant message."""


class GenericProtocolAdapter:
    """Protocol-neutral fallback: inspect the request and enter raw content."""

    adapter_id = "generic"
    label = "Generic OpenAI"

    def classify_request(self, body: JsonObject) -> JsonObject:
        return {
            "type": "chat_completion",
            "label": "Chat completion",
            "response_mode": "raw",
            "attempt": 1,
            "validation_error_count": 0,
            "task": "",
        }

    def session_identity(
        self,
        body: JsonObject,
        headers: Mapping[str, str],
    ) -> tuple[str, str]:
        header = headers.get("x-session-id", "").strip()
        if header:
            return header, "header:x-session-id"
        for key in ("session_id", "conversation_id"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), f"body:{key}"
        return "", ""

    def attempt(self, body: JsonObject) -> int:
        return 1

    def presentation(self, body: JsonObject, *, request_id: str) -> JsonObject:
        messages = body.get("messages")
        messages = messages if isinstance(messages, list) else []
        latest = ""
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                latest = content
            elif isinstance(content, list):
                latest = "\n".join(
                    str(part.get("text") or "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            break
        sections: list[JsonObject] = []
        if latest:
            sections.append(
                {
                    "type": "text",
                    "title": "Latest user message",
                    "value": latest,
                    "column": "main",
                }
            )
        return {
            "view": {
                "eyebrow": "OpenAI-compatible request",
                "title": str(body.get("model") or "Manual response requested"),
                "badges": [],
                "alerts": [],
                "sections": sections,
            },
            "composer": {
                "kind": "raw",
                "label": "Assistant content",
                "placeholder": "Enter the exact assistant message content.",
            },
        }

    def audit_records(self, body: JsonObject) -> list[JsonObject]:
        return []

    def encode_response(
        self,
        body: JsonObject,
        submission: JsonObject,
        *,
        request_id: str,
    ) -> EncodedResponse:
        content = submission.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        return EncodedResponse(message={"role": "assistant", "content": content})


def load_protocol_adapter(spec: str) -> ProtocolAdapter:
    """Load an adapter from ``module:attribute``; ``generic`` needs no plugin."""

    normalized = spec.strip()
    if not normalized or normalized == "generic":
        return GenericProtocolAdapter()
    module_name, separator, attribute = normalized.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("adapter must be 'generic' or use module:attribute syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    adapter = factory() if isinstance(factory, type) or callable(factory) else factory
    if not isinstance(adapter, ProtocolAdapter):
        raise TypeError(f"{normalized} does not implement ProtocolAdapter")
    return adapter
