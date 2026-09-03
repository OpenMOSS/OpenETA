"""Minimal OpenAI-compatible backend for read-only multimodal observations."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from agent.runtime.observation_interfaces import (
    ReadOnlyObservationContext,
    ReadOnlyObservationError,
    ReadOnlyObservationReport,
    find_absolute_posix_paths,
)

SYSTEM_PROMPT = (
    "This is a read-only embodied observation. Do not call tools, propose executable "
    "actions, or output robot control commands. Return only the structured observation report."
)
LOCAL_DUMMY_API_KEY = "local-capture-no-secret"


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


@dataclass(frozen=True)
class OpenAICompatibleReadOnlyObservationBackend:
    base_url: str
    model: str
    api_key: str = LOCAL_DUMMY_API_KEY
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        parsed = urllib.parse.urlparse(self.base_url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
            raise ReadOnlyObservationError("read-only backend must use loopback HTTP")
        if parsed.path.rstrip("/") != "/v1" or parsed.query or parsed.fragment:
            raise ReadOnlyObservationError("read-only backend URL must end in /v1")
        if self.api_key != LOCAL_DUMMY_API_KEY:
            raise ReadOnlyObservationError("read-only capture must use the explicit dummy key")

    def observe(self, context: ReadOnlyObservationContext) -> ReadOnlyObservationReport:
        text_payload = {
            "task_instruction": context.instruction,
            "step_identifiers": dict(context.step_identifiers),
            "proprio": dict(context.proprio),
            "visual_descriptors": [dict(item) for item in context.visual_descriptors],
        }
        text = json.dumps(text_payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        forbidden_text = ("outputs/", "snapshot_pre.json", "host_only")
        if find_absolute_posix_paths(text_payload) or any(
            token in text for token in forbidden_text
        ):
            raise ReadOnlyObservationError("local path or host-only data reached backend text")
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for image in context.images:
            encoded = base64.b64encode(image.path.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _RejectRedirectHandler(),
        )
        try:
            with opener.open(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ReadOnlyObservationError(f"read-only backend transport failed: {exc}") from exc
        try:
            content_text = response_payload["choices"][0]["message"]["content"]
            report_payload = json.loads(content_text)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ReadOnlyObservationError("backend returned an invalid chat completion") from exc
        if not isinstance(report_payload, dict):
            raise ReadOnlyObservationError("backend observation report must be a JSON object")
        return ReadOnlyObservationReport.from_mapping(report_payload)
