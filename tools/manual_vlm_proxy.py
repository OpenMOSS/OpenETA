"""Human-driven OpenAI-compatible VLM proxy with a local inspection GUI.

This module deliberately knows nothing about OpenETA's runtime classes. Point any
OpenAI-compatible client at it, inspect the exact wire request in a browser, and
manually supply the assistant response. It is useful for debugging prompts,
multimodal attachment selection, overlays, history, and harness validation.

Run with::

    python -m tools.manual_vlm_proxy --port 8099 --open

Then configure the client with API base ``http://127.0.0.1:8099/v1``.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import mimetypes
import re
import threading
import time
import webbrowser
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4


JsonObject = dict[str, Any]
DEFAULT_MAX_BODY_BYTES = 64 * 1024 * 1024
CONSOLE_HTML_PATH = Path(__file__).with_name("manual_vlm_console.html")


@dataclass(slots=True)
class PendingRequest:
    request_id: str
    body: JsonObject
    session_id: str = ""
    session_source: str = "inferred"
    session_turn: int = 1
    received_at: float = field(default_factory=time.time)
    response_text: str | None = None
    response_error: str | None = None
    completed_at: float | None = None
    event: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def status(self) -> str:
        if not self.event.is_set():
            return "pending"
        return "cancelled" if self.response_error is not None else "responded"

    def summary(self) -> JsonObject:
        messages = self.body.get("messages")
        if not isinstance(messages, list):
            messages = []
        classification = classify_request(self.body)
        return {
            "id": self.request_id,
            "status": self.status,
            "received_at": self.received_at,
            "completed_at": self.completed_at,
            "model": str(self.body.get("model") or ""),
            "session_id": self.session_id,
            "session_source": self.session_source,
            "session_turn": self.session_turn,
            "message_count": len(messages),
            "image_count": len(extract_images(self.body)),
            "request_type": classification["type"],
            "request_label": classification["label"],
            "response_mode": classification["response_mode"],
            "attempt": classification["attempt"],
            "validation_error_count": classification["validation_error_count"],
            "task": classification["task"],
            "response_preview": (self.response_text or self.response_error or "")[:240],
        }


class RequestStore:
    """Thread-safe queue shared by HTTP client requests and the browser GUI."""

    def __init__(self, *, history_limit: int = 100, record_dir: Path | None = None) -> None:
        self.history_limit = max(1, history_limit)
        self.record_dir = record_dir
        self._requests: dict[str, PendingRequest] = {}
        self._order: list[str] = []
        self._session_histories: dict[str, list[str]] = {}
        self._session_turns: dict[str, int] = {}
        self._inferred_session_counter = 0
        self._lock = threading.RLock()

    def add(self, body: JsonObject, *, session_hint: str = "") -> PendingRequest:
        with self._lock:
            session_id, session_source, history = self._resolve_session_locked(
                body, session_hint=session_hint
            )
            session_turn = self._session_turns.get(session_id, 0) + 1
            self._session_turns[session_id] = session_turn
            self._session_histories[session_id] = history
            request = PendingRequest(
                request_id=uuid4().hex,
                body=body,
                session_id=session_id,
                session_source=session_source,
                session_turn=session_turn,
            )
            self._requests[request.request_id] = request
            self._order.append(request.request_id)
            self._prune_locked()
        self._record(request, "request", body)
        return request

    def _resolve_session_locked(
        self, body: JsonObject, *, session_hint: str
    ) -> tuple[str, str, list[str]]:
        explicit, source = extract_session_identity(body, session_hint=session_hint)
        history = conversation_lineage(body)
        if explicit:
            return explicit, source, history

        # A later turn contains the earlier turn's conversation as a prefix. This
        # groups normal closed-loop growth while keeping simultaneous first turns
        # with identical task text separate.
        candidates = [
            (len(previous), session_id)
            for session_id, previous in self._session_histories.items()
            if len(previous) < len(history) and history[: len(previous)] == previous
        ]
        if candidates:
            return max(candidates)[1], "inferred-lineage", history

        attempt = extract_prompt_attempt(body)
        if attempt > 1:
            exact = [
                session_id
                for session_id, previous in self._session_histories.items()
                if previous == history
            ]
            if exact:
                return exact[-1], "inferred-validation-retry", history

        self._inferred_session_counter += 1
        root = history[0] if history else str(body.get("model") or "request")
        digest = hashlib.sha256(root.encode("utf-8")).hexdigest()[:6]
        return (
            f"inferred-{self._inferred_session_counter:03d}-{digest}",
            "inferred-new",
            history,
        )

    def get(self, request_id: str) -> PendingRequest | None:
        with self._lock:
            return self._requests.get(request_id)

    def summaries(self) -> list[JsonObject]:
        with self._lock:
            return [self._requests[item].summary() for item in reversed(self._order)]

    def respond(self, request_id: str, text: str) -> PendingRequest:
        if not text.strip():
            raise ValueError("assistant response must not be empty")
        with self._lock:
            request = self._require_pending_locked(request_id)
            request.response_text = text
            request.completed_at = time.time()
            request.event.set()
        self._record(request, "response", {"content": text})
        return request

    def cancel(self, request_id: str, reason: str) -> PendingRequest:
        with self._lock:
            request = self._require_pending_locked(request_id)
            request.response_error = reason.strip() or "Cancelled by the human VLM operator."
            request.completed_at = time.time()
            request.event.set()
        self._record(request, "cancel", {"error": request.response_error})
        return request

    def public_detail(self, request_id: str) -> JsonObject | None:
        request = self.get(request_id)
        if request is None:
            return None
        return {
            **request.summary(),
            "messages": public_messages(request.body, request_id=request_id),
            "request_options": {
                key: value for key, value in request.body.items() if key != "messages"
            },
            "tools": extract_tool_catalog(request.body),
            "tool_forms": build_tool_form_catalog(request.body),
            "response_mode": detect_response_mode(request.body),
            "operator": build_operator_summary(request.body, request_id=request_id),
            "wire_audit": build_wire_audit(request.body),
            "response_text": request.response_text,
            "response_error": request.response_error,
        }

    def _require_pending_locked(self, request_id: str) -> PendingRequest:
        request = self._requests.get(request_id)
        if request is None:
            raise KeyError(request_id)
        if request.event.is_set():
            raise ValueError(f"request is already {request.status}")
        return request

    def _prune_locked(self) -> None:
        while len(self._order) > self.history_limit:
            removable = next(
                (item for item in self._order if self._requests[item].event.is_set()), None
            )
            if removable is None:
                return
            self._order.remove(removable)
            self._requests.pop(removable, None)

    def _record(self, request: PendingRequest, suffix: str, value: JsonObject) -> None:
        if self.record_dir is None:
            return
        request_dir = self.record_dir / request.request_id
        request_dir.mkdir(parents=True, exist_ok=True)
        path = request_dir / f"{suffix}.json"
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(frozen=True, slots=True)
class ImagePart:
    index: int
    message_index: int
    part_index: int
    url: str
    detail: str


def extract_images(body: JsonObject) -> list[ImagePart]:
    images: list[ImagePart] = []
    messages = body.get("messages")
    if not isinstance(messages, list):
        return images
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if isinstance(image_url, str):
                url, detail = image_url, ""
            elif isinstance(image_url, dict):
                url = str(image_url.get("url") or "")
                detail = str(image_url.get("detail") or "")
            else:
                continue
            if url:
                images.append(
                    ImagePart(len(images), message_index, part_index, url, detail)
                )
    return images


def public_messages(body: JsonObject, *, request_id: str) -> list[JsonObject]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return []
    images = {(item.message_index, item.part_index): item for item in extract_images(body)}
    public: list[JsonObject] = []
    for message_index, raw in enumerate(messages):
        if not isinstance(raw, dict):
            public.append({"role": "invalid", "content": repr(raw)})
            continue
        message = {key: value for key, value in raw.items() if key != "content"}
        content = raw.get("content")
        if isinstance(content, str):
            message["content"] = content
        elif isinstance(content, list):
            parts: list[Any] = []
            for part_index, raw_part in enumerate(content):
                image = images.get((message_index, part_index))
                if image is None:
                    parts.append(raw_part)
                    continue
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"/api/requests/{request_id}/images/{image.index}",
                            "detail": image.detail,
                        },
                        "wire_url_kind": "data_url" if image.url.startswith("data:") else "url",
                    }
                )
            message["content"] = parts
        else:
            message["content"] = content
        public.append(message)
    return public


def extract_tool_catalog(body: JsonObject) -> list[JsonObject]:
    """Best-effort extraction of tool descriptors from serialized prompt text.

    The proxy intentionally has no OpenETA imports. It discovers the conventional
    ``available_tools`` field in any JSON payload sent over the wire and prefers
    the richest descriptor for each name.
    """

    tools: dict[str, JsonObject] = {}
    for value in _wire_json_values(body):
        for key, nested in _walk_json(value):
            if key != "available_tools" or not isinstance(nested, list):
                continue
            for item in nested:
                if isinstance(item, str) and item:
                    candidate: JsonObject = {"name": item}
                elif isinstance(item, dict) and isinstance(item.get("name"), str):
                    candidate = dict(item)
                else:
                    continue
                name = str(candidate["name"])
                previous = tools.get(name)
                if previous is None or len(json.dumps(candidate)) > len(json.dumps(previous)):
                    tools[name] = candidate
    return [tools[name] for name in sorted(tools)]


_DEFAULT_VALUE_PATTERN = re.compile(
    r"\bdefaults?\s+to\s+([^.;]+)", re.IGNORECASE
)


def _parameter_value_type(name: str, description: str, declared: str = "") -> str:
    normalized = declared.strip().lower()
    if normalized in {"string", "integer", "number", "boolean", "json"}:
        return normalized
    if normalized in {"array", "object"}:
        return "json"
    text = f"{name} {description}".lower()
    if any(marker in text for marker in ("boolean", "toggle", "true or false")):
        return "boolean"
    json_patterns = (
        r"\bjson\b",
        r"\blist(?:\s+of)?\b",
        r"\barray\b",
        r"\bdict\b",
        r"\bintrinsics\b",
        r"\bmatrix\b",
        r"\bbbox\b",
        r"\bpose\b",
        r"\btrajectory\b",
        r"\bconfig(?:uration)?\b",
        r"\{\s*x\s*,",
        r"\[\s*left\s*,",
    )
    if re.search(r"(?:^|_)points?$", name.lower()) or any(
        re.search(pattern, text) for pattern in json_patterns
    ):
        return "json"
    if any(
        marker in text
        for marker in (
            "integer",
            " count",
            "_count",
            "width",
            "height",
            "seed",
            "limit",
            "max_",
        )
    ):
        return "integer"
    if any(
        marker in text
        for marker in (
            "number",
            "float",
            "scale",
            "timestamp",
            "resolution",
            "timeout",
            "position",
        )
    ):
        return "number"
    return "string"


def _coerce_documented_default(value: Any, value_type: str) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip().strip("`\"")
    lowered = text.lower()
    if value_type == "boolean":
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    if value_type == "integer":
        try:
            return int(text)
        except ValueError:
            return text
    if value_type == "number":
        try:
            return float(text)
        except ValueError:
            return text
    if value_type == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    if lowered in {"none", "null"}:
        return None
    return text


def _parameter_choices(description: str) -> list[str]:
    first_clause = description.split(";", 1)[0].strip()
    if "|" not in first_clause:
        return []
    values = [value.strip().strip("`\"") for value in first_clause.split("|")]
    if not 2 <= len(values) <= 8:
        return []
    if any(not value or len(value) > 64 for value in values):
        return []
    return values


def _normalize_parameter_field(name: str, spec: Any) -> JsonObject:
    if isinstance(spec, dict):
        description = str(spec.get("description") or spec.get("help") or "")
        declared_type = str(spec.get("type") or "")
        explicit_required = spec.get("required")
        has_default = "default" in spec
        default = spec.get("default")
        raw_choices = spec.get("enum") or spec.get("choices") or []
        choices = [str(value) for value in raw_choices] if isinstance(raw_choices, list) else []
    else:
        description = str(spec or "")
        declared_type = ""
        explicit_required = None
        default_match = _DEFAULT_VALUE_PATTERN.search(description)
        has_default = default_match is not None
        default = default_match.group(1).strip() if default_match else None
        choices = _parameter_choices(description)

    lowered = description.lower()
    if isinstance(explicit_required, bool):
        required: bool | None = explicit_required
    elif re.search(r"\brequired\b", lowered):
        required = True
    elif re.search(r"\boptional\b|\bomitt?\b", lowered) or has_default:
        required = False
    else:
        required = None
    value_type = _parameter_value_type(name, description, declared_type)
    default = _coerce_documented_default(default, value_type) if has_default else None
    return {
        "name": name,
        "description": description,
        "value_type": value_type,
        "required": required,
        "has_default": has_default,
        "default": default,
        "choices": choices,
    }


def build_tool_form_catalog(body: JsonObject) -> list[JsonObject]:
    """Normalize loose tool descriptors into fields suitable for a human form."""

    output: list[JsonObject] = []
    for tool in extract_tool_catalog(body):
        raw_parameters = tool.get("parameters")
        raw_parameters = raw_parameters if isinstance(raw_parameters, dict) else {}
        output.append(
            {
                "name": tool.get("name"),
                "description": tool.get("description") or tool.get("category") or "",
                "fields": [
                    _normalize_parameter_field(str(name), spec)
                    for name, spec in raw_parameters.items()
                ],
            }
        )
    return output


def detect_response_mode(body: JsonObject) -> str:
    response_format = body.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") == "json_object":
        return "json"
    for text in _message_texts(body):
        lowered = text.lower()
        if "return only the exact json object" in lowered:
            return "json"
    return "xml"


def extract_session_identity(
    body: JsonObject, *, session_hint: str = ""
) -> tuple[str, str]:
    if session_hint.strip():
        return session_hint.strip(), "header"
    priority = ("active_agent_session_id", "agent_session_id", "session_id")
    for field in priority:
        for value in _wire_json_values(body):
            for key, nested in _walk_json(value):
                if key == field and isinstance(nested, str) and nested.strip():
                    return nested.strip(), f"wire:{field}"
    return "", "inferred"


def conversation_lineage(body: JsonObject) -> list[str]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return []
    normalized: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") == "system":
            continue
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            text = json.dumps(content, ensure_ascii=False, sort_keys=True)
        normalized.append(f"{message.get('role')}:{text}")
    # The final user message is the current dynamic tool context, not durable
    # conversation history. Removing it makes lineage prefix comparison stable.
    if normalized and normalized[-1].startswith("user:"):
        normalized.pop()
    return normalized


def extract_prompt_attempt(body: JsonObject) -> int:
    for value in _wire_json_values(body):
        if not isinstance(value, dict):
            continue
        attempt = value.get("attempt")
        if isinstance(attempt, int) and not isinstance(attempt, bool):
            return attempt
    return 1


def _wire_json_values(body: JsonObject) -> list[Any]:
    values: list[Any] = [body]
    decoder = json.JSONDecoder()
    for text in _message_texts(body):
        stripped = text.strip()
        try:
            values.append(json.loads(stripped))
            continue
        except json.JSONDecodeError:
            pass
        # OpenETA's stable context is explanatory text, a newline, then one JSON
        # object. Limiting candidates to line starts avoids quadratic rescanning of
        # every nested brace in a large prompt.
        positions = [0]
        positions.extend(index + 1 for index, char in enumerate(text) if char == "\n")
        for position in positions:
            while position < len(text) and text[position].isspace():
                position += 1
            if position >= len(text) or text[position] not in "{[":
                continue
            try:
                value, _end = decoder.raw_decode(text, position)
            except json.JSONDecodeError:
                continue
            values.append(value)
    return values


def _message_texts(body: JsonObject) -> list[str]:
    texts: list[str] = []
    messages = body.get("messages")
    if not isinstance(messages, list):
        return texts
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(
                str(part.get("text"))
                for part in content
                if isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            )
    return texts


def _walk_json(value: Any) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            found.append((str(key), nested))
            found.extend(_walk_json(nested))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_walk_json(nested))
    return found


def _planner_wire_payload(body: JsonObject) -> JsonObject:
    """Return the final serialized planner envelope carried by the request."""

    candidates = [
        value
        for value in _wire_json_values(body)
        if isinstance(value, dict)
        and isinstance(value.get("tool_context"), dict)
        and isinstance(value.get("instruction"), str)
    ]
    return candidates[-1] if candidates else {}


def _stable_wire_context(body: JsonObject) -> JsonObject:
    candidates = [
        value
        for value in _wire_json_values(body)
        if isinstance(value, dict)
        and value.get("schema_version") == "openeta.planner_static_context.v1"
    ]
    return candidates[-1] if candidates else {}


def _request_label_from_schema(schema: str, *, response_mode: str) -> tuple[str, str]:
    labels = {
        "openeta.agent_context.v2": ("main_planner", "Main planner"),
        "openeta.visual_delta_request.v1": (
            "visual_differencing",
            "Visual differencing",
        ),
        "openeta.reference_localization_request.v1": (
            "reference_localization",
            "Reference localization",
        ),
        "openeta.grasp_pose_advisor_request.v1": (
            "grasp_advisor",
            "Grasp advisor",
        ),
    }
    if schema in labels:
        return labels[schema]
    if schema:
        readable = schema.removeprefix("openeta.").removesuffix(".v1")
        return readable.replace(".", "_"), readable.replace("_", " ").title()
    if response_mode == "json":
        return "isolated_json", "Isolated JSON planner"
    return "planner", "Planner"


def classify_request(body: JsonObject) -> JsonObject:
    """Classify one provider request for the human-facing queue."""

    payload = _planner_wire_payload(body)
    context = payload.get("tool_context")
    context = context if isinstance(context, dict) else {}
    stable = _stable_wire_context(body)
    schema = str(
        context.get("schema_version")
        or stable.get("agent_context_schema_version")
        or ""
    )
    response_mode = detect_response_mode(body)
    request_type, label = _request_label_from_schema(
        schema, response_mode=response_mode
    )
    objective = context.get("objective")
    objective = objective if isinstance(objective, dict) else {}
    task = objective.get("task") or context.get("task") or ""
    errors = payload.get("validation_errors")
    errors = errors if isinstance(errors, list) else []
    attempt = payload.get("attempt")
    if not isinstance(attempt, int) or isinstance(attempt, bool):
        attempt = 1
    return {
        "type": request_type,
        "label": label,
        "schema_version": schema,
        "response_mode": response_mode,
        "attempt": attempt,
        "validation_error_count": len(errors),
        "task": str(task or ""),
    }


def _compact_latest_action(context: JsonObject, observation: JsonObject) -> JsonObject:
    decision_state = context.get("decision_state")
    decision_state = decision_state if isinstance(decision_state, dict) else {}
    latest = decision_state.get("last_action_effect")
    if isinstance(latest, dict):
        return {
            key: latest.get(key)
            for key in (
                "tool",
                "status",
                "content",
                "operational_success",
                "semantic_outcome",
                "effect",
            )
            if latest.get(key) is not None
        }
    metadata = observation.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    previous = metadata.get("previous_action")
    if not isinstance(previous, dict):
        return {}
    calls = previous.get("tool_calls")
    call = calls[-1] if isinstance(calls, list) and calls and isinstance(calls[-1], dict) else {}
    result = call.get("result")
    result = result if isinstance(result, dict) else {}
    return {
        "tool": call.get("name") or previous.get("request_name"),
        "status": call.get("status") or previous.get("status"),
        "content": result.get("content"),
        "operational_success": result.get("success"),
    }


def _operator_images(
    body: JsonObject, context: JsonObject, *, request_id: str
) -> list[JsonObject]:
    evidence = context.get("vision_evidence")
    evidence = evidence if isinstance(evidence, list) else []
    paths = context.get("vision_image_paths")
    paths = paths if isinstance(paths, list) else []
    evidence_by_path = {
        str(item.get("path")): item
        for item in evidence
        if isinstance(item, dict) and item.get("path")
    }
    output: list[JsonObject] = []
    for item in extract_images(body):
        path = str(paths[item.index]) if item.index < len(paths) else ""
        descriptor = evidence_by_path.get(path, {})
        label = (
            descriptor.get("camera_role")
            or descriptor.get("role")
            or descriptor.get("evidence_id")
            or f"Image {item.index + 1}"
        )
        output.append(
            {
                "index": item.index,
                "url": f"/api/requests/{request_id}/images/{item.index}",
                "label": str(label),
                "role": str(descriptor.get("role") or "visual evidence"),
                "freshness": str(descriptor.get("freshness") or ""),
                "detail": item.detail,
            }
        )
    return output


def build_operator_summary(body: JsonObject, *, request_id: str) -> JsonObject:
    """Build a bounded, human-oriented projection without changing the wire body."""

    payload = _planner_wire_payload(body)
    context = payload.get("tool_context")
    context = context if isinstance(context, dict) else {}
    stable = _stable_wire_context(body)
    classification = classify_request(body)

    objective = context.get("objective")
    objective = objective if isinstance(objective, dict) else {}
    active_task = objective.get("active_environment_task")
    active_task = active_task if isinstance(active_task, dict) else {}
    current = context.get("current_observation")
    current = current if isinstance(current, dict) else {}
    observation = current.get("summary")
    observation = observation if isinstance(observation, dict) else {}
    observation_metadata = observation.get("metadata")
    observation_metadata = (
        observation_metadata if isinstance(observation_metadata, dict) else {}
    )
    robot = observation.get("robot")
    robot = robot if isinstance(robot, dict) else {}
    decision_state = context.get("decision_state")
    decision_state = decision_state if isinstance(decision_state, dict) else {}
    skills = stable.get("relevant_skills")
    skills = skills if isinstance(skills, list) else []
    errors = payload.get("validation_errors")
    errors = [str(value) for value in errors] if isinstance(errors, list) else []

    return {
        "classification": classification,
        "instruction": str(payload.get("instruction") or ""),
        "task": str(
            objective.get("task")
            or active_task.get("task")
            or context.get("task")
            or ""
        ),
        "environment": {
            "env_id": active_task.get("env_id"),
            "handle": active_task.get("handle"),
        },
        "attempt": classification["attempt"],
        "validation_errors": errors,
        "observation": {
            "status": current.get("status"),
            "step": observation_metadata.get("step_idx"),
            "fresh": observation_metadata.get("observation_fresh"),
            "camera_ids": observation.get("camera_ids") or [],
            "object_count": observation.get("object_count"),
            "objects": observation.get("objects") or [],
            "end_effector_pose": robot.get("end_effector_pose"),
            "gripper_state": robot.get("gripper_state"),
        },
        "latest_action": _compact_latest_action(context, observation),
        "unresolved_obligations": decision_state.get("unresolved_obligations") or {},
        "open_questions": context.get("open_questions") or {},
        "skills": [
            {
                "name": item.get("name"),
                "description": item.get("description"),
            }
            for item in skills
            if isinstance(item, dict) and item.get("name")
        ],
        "required_output": context.get("required_output"),
        "images": _operator_images(body, context, request_id=request_id),
    }


def _content_part_audit(content: Any) -> list[JsonObject]:
    if isinstance(content, str):
        return [{"type": "text", "chars": len(content)}]
    if not isinstance(content, list):
        return [{"type": type(content).__name__, "chars": 0}]
    parts: list[JsonObject] = []
    for item in content:
        if not isinstance(item, dict):
            parts.append({"type": type(item).__name__, "chars": 0})
            continue
        part_type = str(item.get("type") or "object")
        if part_type == "text":
            parts.append({"type": "text", "chars": len(str(item.get("text") or ""))})
        elif part_type == "image_url":
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                url = str(image_url.get("url") or "")
                detail = str(image_url.get("detail") or "")
            else:
                url, detail = str(image_url or ""), ""
            parts.append(
                {
                    "type": "image_url",
                    "wire_url_kind": "data_url" if url.startswith("data:") else "url",
                    "encoded_chars": len(url),
                    "detail": detail,
                }
            )
        else:
            parts.append(
                {
                    "type": part_type,
                    "chars": len(json.dumps(item, ensure_ascii=False)),
                }
            )
    return parts


def build_wire_audit(body: JsonObject) -> JsonObject:
    """Describe the complete model input while leaving exact values in messages/raw."""

    messages = body.get("messages")
    messages = messages if isinstance(messages, list) else []
    rows: list[JsonObject] = []
    text_chars = 0
    image_count = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            rows.append({"index": index, "role": "invalid", "parts": []})
            continue
        parts = _content_part_audit(message.get("content"))
        text_chars += sum(
            int(part.get("chars") or 0) for part in parts if part.get("type") == "text"
        )
        image_count += sum(part.get("type") == "image_url" for part in parts)
        rows.append(
            {
                "index": index,
                "role": str(message.get("role") or "unknown"),
                "parts": parts,
            }
        )
    normalized = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    normalized_bytes = normalized.encode("utf-8")
    return {
        "message_count": len(messages),
        "text_chars": text_chars,
        "image_count": image_count,
        "normalized_body_bytes": len(normalized_bytes),
        "normalized_sha256": hashlib.sha256(normalized_bytes).hexdigest(),
        "messages": rows,
        "note": (
            "Message values are complete. The audit view replaces inline image data URLs "
            "with byte-preserving image endpoints; Raw JSON retains the original data URLs."
        ),
    }


def decode_data_url(url: str) -> tuple[str, bytes]:
    if not url.startswith("data:") or "," not in url:
        raise ValueError("only inline data URLs can be served by this endpoint")
    header, encoded = url.split(",", 1)
    metadata = header[5:]
    mime_type = metadata.split(";", 1)[0] or "application/octet-stream"
    if ";base64" not in metadata.lower():
        raise ValueError("image data URL is not base64 encoded")
    try:
        return mime_type, base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise ValueError("invalid image base64") from exc


def _safe_xml_tag(name: object) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name or "value"))
    if not value or not re.match(r"[A-Za-z_]", value):
        value = f"field_{value}"
    return value


def _append_xml_value(parent: ET.Element, name: object, value: Any) -> ET.Element:
    element = ET.SubElement(parent, _safe_xml_tag(name))
    if value is None:
        element.set("type", "null")
    elif isinstance(value, bool):
        element.set("type", "boolean")
        element.text = "true" if value else "false"
    elif isinstance(value, int):
        element.set("type", "integer")
        element.text = str(value)
    elif isinstance(value, float):
        element.set("type", "number")
        element.text = str(value)
    elif isinstance(value, list):
        element.set("type", "array")
        for item in value:
            _append_xml_value(element, "item", item)
    elif isinstance(value, dict):
        element.set("type", "object")
        for key, nested in value.items():
            _append_xml_value(element, key, nested)
    else:
        element.set("type", "string")
        element.text = str(value)
    return element


def serialize_decision_xml(decision: JsonObject) -> str:
    """Serialize a structured human decision into the planner's typed XML format."""

    kind = decision.get("kind")
    name = decision.get("name")
    parameters = decision.get("parameters", {})
    reasoning = decision.get("reasoning", "")
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("decision.kind must be a non-empty string")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("decision.name must be a non-empty string")
    if not isinstance(parameters, dict):
        raise ValueError("decision.parameters must be an object")

    root = ET.Element("decision")
    _append_xml_value(root, "kind", kind)
    _append_xml_value(root, "name", name)
    _append_xml_value(root, "parameters", parameters)
    _append_xml_value(root, "reasoning", reasoning)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", short_empty_elements=True)


class ManualVLMServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        store: RequestStore,
        decision_timeout_s: float | None,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    ) -> None:
        super().__init__(address, ManualVLMHandler)
        self.store = store
        self.decision_timeout_s = decision_timeout_s
        self.max_body_bytes = max_body_bytes


class ManualVLMHandler(BaseHTTPRequestHandler):
    server: ManualVLMServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        if path == "/":
            self._send_bytes(
                HTTPStatus.OK,
                load_console_html().encode(),
                "text/html; charset=utf-8",
            )
            return
        if path in {"/health", "/api/health"}:
            self._send_json(HTTPStatus.OK, {"ok": True, "pending": self._pending_count()})
            return
        if path in {"/v1/models", "/models"}:
            self._send_json(
                HTTPStatus.OK,
                {"object": "list", "data": [{"id": "human-vlm", "object": "model"}]},
            )
            return
        if path == "/api/requests":
            self._send_json(HTTPStatus.OK, {"requests": self.server.store.summaries()})
            return
        route = self._request_route(path)
        if route is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "not_found", "Route not found.")
            return
        request_id, tail = route
        request = self.server.store.get(request_id)
        if request is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "not_found", "Request not found.")
            return
        if tail == "":
            self._send_json(HTTPStatus.OK, self.server.store.public_detail(request_id) or {})
            return
        if tail == "raw":
            self._send_json(HTTPStatus.OK, request.body)
            return
        if tail.startswith("images/"):
            self._serve_image(request, tail.split("/", 1)[1])
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, "not_found", "Route not found.")

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path.rstrip("/")
        if path in {"/v1/chat/completions", "/chat/completions"}:
            self._handle_completion()
            return
        route = self._request_route(path)
        if route is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "not_found", "Route not found.")
            return
        request_id, tail = route
        try:
            body = self._read_json_body()
            if tail == "response":
                response = body.get("content")
                if not isinstance(response, str):
                    decision = body.get("decision")
                    if not isinstance(decision, dict):
                        raise ValueError("content must be a string or decision must be an object")
                    response = serialize_decision_xml(decision)
                request = self.server.store.respond(request_id, response)
            elif tail == "cancel":
                reason = body.get("reason")
                request = self.server.store.cancel(
                    request_id, reason if isinstance(reason, str) else ""
                )
            else:
                self._send_error_json(HTTPStatus.NOT_FOUND, "not_found", "Route not found.")
                return
        except KeyError:
            self._send_error_json(HTTPStatus.NOT_FOUND, "not_found", "Request not found.")
            return
        except ValueError as exc:
            self._send_error_json(HTTPStatus.CONFLICT, "invalid_state", str(exc))
            return
        self._send_json(HTTPStatus.OK, request.summary())

    def _handle_completion(self) -> None:
        try:
            body = self._read_json_body()
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "invalid_request", str(exc))
            return
        messages = body.get("messages")
        if not isinstance(messages, list):
            self._send_error_json(
                HTTPStatus.BAD_REQUEST, "invalid_request", "messages must be an array"
            )
            return
        session_hint = self.headers.get("X-OpenETA-Session-ID", "") or self.headers.get(
            "X-Session-ID", ""
        )
        request = self.server.store.add(body, session_hint=session_hint)
        completed = request.event.wait(timeout=self.server.decision_timeout_s)
        if not completed:
            try:
                self.server.store.cancel(request.request_id, "Human decision timed out.")
            except ValueError:
                pass
        if request.response_error is not None:
            self._send_error_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "human_cancelled",
                request.response_error,
                request_id=request.request_id,
            )
            return
        content = request.response_text or ""
        now = int(time.time())
        response = {
            "id": f"chatcmpl-human-{request.request_id}",
            "object": "chat.completion",
            "created": now,
            "model": str(body.get("model") or "human-vlm"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        self._send_json(HTTPStatus.OK, response)

    def _serve_image(self, request: PendingRequest, raw_index: str) -> None:
        try:
            index = int(raw_index)
            image = extract_images(request.body)[index]
            if image.url.startswith("data:"):
                mime_type, data = decode_data_url(image.url)
                self._send_bytes(HTTPStatus.OK, data, mime_type, cache="no-store")
                return
            self.send_response(HTTPStatus.TEMPORARY_REDIRECT)
            self.send_header("Location", image.url)
            self.send_header("Content-Length", "0")
            self.end_headers()
        except (ValueError, IndexError):
            self._send_error_json(HTTPStatus.NOT_FOUND, "image_not_found", "Image not found.")

    def _read_json_body(self) -> JsonObject:
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "0")
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length <= 0:
            raise ValueError("request body is empty")
        if length > self.server.max_body_bytes:
            raise ValueError(
                f"request body exceeds {self.server.max_body_bytes} byte limit"
            )
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body is not valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _request_route(self, path: str) -> tuple[str, str] | None:
        prefix = "/api/requests/"
        if not path.startswith(prefix):
            return None
        remainder = unquote(path[len(prefix) :]).strip("/")
        if not remainder:
            return None
        request_id, separator, tail = remainder.partition("/")
        return request_id, tail if separator else ""

    def _pending_count(self) -> int:
        return sum(item["status"] == "pending" for item in self.server.store.summaries())

    def _send_json(self, status: HTTPStatus, value: JsonObject) -> None:
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, data, "application/json; charset=utf-8", cache="no-store")

    def _send_error_json(
        self,
        status: HTTPStatus,
        code: str,
        message: str,
        **details: Any,
    ) -> None:
        self._send_json(
            status,
            {"error": {"message": message, "type": code, "code": code, **details}},
        )

    def _send_bytes(
        self,
        status: HTTPStatus,
        data: bytes,
        content_type: str,
        *,
        cache: str = "no-cache",
    ) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format: str, *args: Any) -> None:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"[{stamp}] {self.client_address[0]} {format % args}")


GUI_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Manual VLM Console</title>
<style>
:root{color-scheme:dark;--bg:#0b0e14;--panel:#121722;--line:#283143;--text:#e8edf6;--muted:#91a0b8;--accent:#62d3a5;--warn:#ffbd66;--bad:#ff6b7d;--system:#8db7ff;--user:#a9e5cd;--assistant:#d5abff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif;height:100vh;overflow:hidden}button,input,textarea{font:inherit}.app{display:grid;grid-template-columns:280px minmax(0,1fr);height:100vh}.sidebar{border-right:1px solid var(--line);background:#0e131c;display:flex;flex-direction:column;min-width:0;min-height:0}.brand{padding:18px;border-bottom:1px solid var(--line)}.brand h1{font-size:17px;margin:0}.brand p{color:var(--muted);font-size:12px;margin:5px 0 0}.queue{overflow:auto;padding:10px;flex:1;min-height:0}.request{width:100%;text-align:left;background:transparent;color:inherit;border:1px solid transparent;border-radius:8px;padding:10px;margin-bottom:6px;cursor:pointer}.request:hover,.request.selected{background:var(--panel);border-color:var(--line)}.request-top{display:flex;justify-content:space-between;gap:8px}.status{font-size:11px;text-transform:uppercase;letter-spacing:.06em}.pending{color:var(--warn)}.responded{color:var(--accent)}.cancelled{color:var(--bad)}.small{font-size:11px;color:var(--muted);margin-top:5px}.main{display:grid;grid-template-rows:auto minmax(0,1fr) auto;min-width:0;min-height:0;height:100vh;overflow:hidden}.topbar{padding:11px 18px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:12px}.topbar a{color:var(--system);text-decoration:none}.viewer{overflow:auto;padding:18px;min-height:0;overscroll-behavior:contain}.empty{height:100%;display:grid;place-items:center;color:var(--muted)}.meta{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);margin-bottom:14px}.message{border:1px solid var(--line);background:var(--panel);border-radius:10px;margin-bottom:14px;overflow:hidden}.message-head{padding:8px 12px;border-bottom:1px solid var(--line);font-size:12px;text-transform:uppercase;letter-spacing:.08em}.role-system{color:var(--system)}.role-user{color:var(--user)}.role-assistant{color:var(--assistant)}.text-part{white-space:pre-wrap;overflow-wrap:anywhere;padding:13px;font:13px/1.52 ui-monospace,SFMono-Regular,Menlo,monospace;max-height:50vh;overflow:auto}.part-label{padding:8px 13px 0;color:var(--muted);font-size:11px}.image-wrap{padding:10px 13px 15px}.image-wrap img{display:block;max-width:100%;max-height:62vh;border:1px solid var(--line);background:#05070a;cursor:zoom-in;image-rendering:auto}.object-part{margin:10px 13px;padding:10px;white-space:pre-wrap;overflow:auto;max-height:45vh;background:#080b10;border-radius:6px;color:#c7d1df;font:12px ui-monospace,monospace}.pager{display:flex;align-items:center;justify-content:center;gap:8px;padding:8px 12px;color:var(--muted);border-top:1px solid var(--line)}.main-pager{border:1px solid var(--line);border-radius:8px;background:#0e131c;margin:0 0 14px}.tools{border:1px solid var(--line);border-radius:10px;background:var(--panel);margin-bottom:14px}.tools summary{padding:11px 13px;cursor:pointer;color:var(--accent)}.tool-search{width:calc(100% - 24px);margin:0 12px 10px;background:#080b10;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:8px}.tool-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:8px;padding:0 12px 12px;max-height:44vh;overflow:auto}.tool-card{border:1px solid var(--line);border-radius:7px;padding:10px;background:#0c1119}.tool-card-head{display:flex;align-items:center;justify-content:space-between;gap:8px}.tool-name{font:13px ui-monospace,monospace;color:var(--system)}.tool-desc{color:var(--muted);font-size:12px;margin-top:7px}.tool-card details{margin-top:7px}.tool-card pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#b9c5d6;font:11px ui-monospace,monospace;max-height:180px;overflow:auto}.composer{border-top:1px solid var(--line);background:#0e131c;padding:10px 18px;z-index:5;box-shadow:0 -8px 24px #0008}.composer-top{display:flex;gap:8px;margin-bottom:8px;align-items:center;flex-wrap:wrap}.composer textarea{width:100%;height:116px;min-height:80px;max-height:24vh;resize:vertical;background:#080b10;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:11px;font:13px/1.5 ui-monospace,monospace}.mode{color:var(--warn);font-size:12px}.btn{border:1px solid var(--line);background:#17202e;color:var(--text);padding:7px 11px;border-radius:7px;cursor:pointer}.btn:disabled{opacity:.4;cursor:not-allowed}.btn:hover:not(:disabled){border-color:#53647f}.btn-primary{background:#146147;border-color:#258b68}.btn-danger{color:#ffc3cb}.spacer{flex:1}.hidden{display:none!important}.modal{position:fixed;inset:0;background:#000e;z-index:20;display:grid;grid-template-rows:auto 1fr}.modal-head{padding:10px 14px;display:flex;justify-content:flex-end}.modal-stage{overflow:auto;display:grid;place-items:center;padding:16px}.modal-stage img{max-width:none;cursor:zoom-out}.toast{position:fixed;right:18px;top:16px;background:#202a39;border:1px solid var(--line);padding:9px 13px;border-radius:7px;z-index:30}@media(max-width:800px){.app{grid-template-columns:1fr}.sidebar{display:none}.viewer{padding:10px}.composer{padding:8px 10px}.composer textarea{height:90px}.topbar{padding:9px 10px}}
.app{grid-template-columns:300px minmax(0,1fr)}.session-group{border:1px solid var(--line);border-radius:9px;margin-bottom:10px;background:#0a0f17;overflow:hidden}.session-group>summary{cursor:pointer;padding:9px 10px;color:var(--accent);font:12px ui-monospace,monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.session-source{display:block;color:var(--muted);font:10px ui-sans-serif,system-ui;margin-top:3px}.session-requests{padding:0 6px 6px}.session-requests .request{padding:8px;margin-bottom:4px}.composer textarea{max-height:65vh}.resize-hint{color:var(--muted);font-size:11px}
</style>
</head>
<body><div class="app">
<aside class="sidebar"><div class="brand"><h1>Manual VLM Console</h1><p>Exact OpenAI-compatible wire view</p></div><div id="queue" class="queue"></div></aside>
<main class="main"><div class="topbar"><div id="title">等待 harness 请求…</div><div><a id="rawLink" class="hidden" target="_blank">raw JSON</a></div></div><section id="viewer" class="viewer"><div class="empty">启动一次 agent turn 后，请求会出现在左侧。</div></section>
<section id="composer" class="composer hidden"><div class="composer-top"><span id="mode" class="mode"></span><span class="resize-hint">拖动输入框上边缘可放大，自动记住高度</span><button class="btn" data-template="tool">Tool XML</button><button class="btn" data-template="talk">Talk XML</button><button class="btn" data-template="complete">Complete XML</button><span class="spacer"></span><button id="cancel" class="btn btn-danger">Cancel request</button><button id="submit" class="btn btn-primary">Send to harness</button></div><textarea id="response" spellcheck="false" placeholder="输入模型应该返回的完整 XML 或 JSON。Ctrl/Cmd+Enter 发送。"></textarea></section></main>
</div><div id="modal" class="modal hidden"><div class="modal-head"><button id="closeModal" class="btn">Close</button></div><div class="modal-stage"><img id="modalImage"></div></div><div id="toast" class="toast hidden"></div>
<script>
const state={requests:[],selected:null,detail:null,messagePage:0,textPages:{},toolFilter:''};const PAGE_MESSAGES=4,PAGE_CHARS=12000,PAGE_LINES=100;const $=s=>document.querySelector(s);const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function toast(s){const e=$('#toast');e.textContent=s;e.classList.remove('hidden');setTimeout(()=>e.classList.add('hidden'),2200)}
async function api(url,opts){const r=await fetch(url,opts);const j=await r.json();if(!r.ok)throw new Error(j.error?.message||r.statusText);return j}
function renderQueue(){const q=$('#queue'),groups=[];for(const r of state.requests){let g=groups.find(x=>x.id===r.session_id);if(!g){g={id:r.session_id,source:r.session_source,requests:[]};groups.push(g)}g.requests.push(r)}q.innerHTML=groups.map(g=>`<details class="session-group" open><summary title="${esc(g.id)}">${esc(g.id)}<span class="session-source">${esc(g.source)} · ${g.requests.length} request${g.requests.length===1?'':'s'}</span></summary><div class="session-requests">${g.requests.map(r=>`<button class="request ${state.selected===r.id?'selected':''}" data-id="${r.id}"><div class="request-top"><span>Turn ${r.session_turn}</span><span class="status ${r.status}">${r.status}</span></div><div class="small">${new Date(r.received_at*1000).toLocaleTimeString()} · ${r.message_count} msg · ${r.image_count} img</div></button>`).join('')}</div></details>`).join('')||'<div class="small">No sessions yet</div>';q.querySelectorAll('.request').forEach(e=>e.onclick=()=>select(e.dataset.id))}
function readableText(raw){const s=String(raw??'');const candidates=[0,...Array.from(s.matchAll(/\n(?=\s*[\[{])/g),m=>m.index+1)].sort((a,b)=>b-a);for(const at of candidates){try{const prefix=s.slice(0,at),value=JSON.parse(s.slice(at).trim());return prefix+JSON.stringify(value,null,2)}catch{}}return s}
function splitText(raw){const lines=readableText(raw).split('\n'),pages=[];let page=[],chars=0;for(const line of lines){if(page.length&&(page.length>=PAGE_LINES||chars+line.length>PAGE_CHARS)){pages.push(page.join('\n'));page=[];chars=0}if(line.length>PAGE_CHARS){if(page.length){pages.push(page.join('\n'));page=[];chars=0}for(let i=0;i<line.length;i+=PAGE_CHARS)pages.push(line.slice(i,i+PAGE_CHARS));continue}page.push(line);chars+=line.length+1}if(page.length||!pages.length)pages.push(page.join('\n'));return pages}
function textHtml(raw,key){const pages=splitText(raw),page=Math.min(state.textPages[key]||0,pages.length-1);return `<div class="text-part">${esc(pages[page])}</div>`+(pages.length>1?`<div class="pager"><button class="btn text-page" data-key="${esc(key)}" data-delta="-1" ${page===0?'disabled':''}>← 上一页</button><span>正文 ${page+1} / ${pages.length}</span><button class="btn text-page" data-key="${esc(key)}" data-delta="1" ${page===pages.length-1?'disabled':''}>下一页 →</button></div>`:'')}
function partHtml(p,i,key){if(typeof p==='string')return textHtml(p,key);if(!p||typeof p!=='object')return `<pre class="object-part">${esc(JSON.stringify(p,null,2))}</pre>`;if(p.type==='text')return `<div class="part-label">TEXT PART ${i+1}</div>${textHtml(p.text||'',key)}`;if(p.type==='image_url'){const u=p.image_url?.url||p.image_url||'';return `<div class="part-label">IMAGE PART ${i+1} · detail=${esc(p.image_url?.detail||'default')} · ${esc(p.wire_url_kind||'')}</div><div class="image-wrap"><img src="${esc(u)}" data-full="${esc(u)}" alt="VLM image part ${i+1}"></div>`}return `<pre class="object-part">${esc(JSON.stringify(p,null,2))}</pre>`}
function mainPager(page,count){return count<=1?'':`<div class="pager main-pager"><button class="btn main-page" data-delta="-1" ${page===0?'disabled':''}>← 上一页</button><span>消息页 ${page+1} / ${count}</span><button class="btn main-page" data-delta="1" ${page===count-1?'disabled':''}>下一页 →</button></div>`}
function toolCatalog(d){if(!d.tools?.length)return '<details class="tools"><summary>Tool List · 当前 wire prompt 未发现工具描述</summary></details>';const q=state.toolFilter.toLowerCase(),shown=d.tools.filter(t=>!q||JSON.stringify(t).toLowerCase().includes(q));return `<details class="tools" open><summary>Tool List · ${d.tools.length} tools（点击 Use 自动生成 XML）</summary><input id="toolSearch" class="tool-search" value="${esc(state.toolFilter)}" placeholder="搜索工具名、说明或参数…"><div class="tool-grid">${shown.map((t,i)=>`<div class="tool-card"><div class="tool-card-head"><span class="tool-name">${esc(t.name)}</span><button class="btn use-tool" data-name="${esc(t.name)}">Use</button></div><div class="tool-desc">${esc(t.description||t.category||'')}</div><details><summary>参数 / schema</summary><pre>${esc(JSON.stringify(t.parameters||{},null,2))}</pre></details></div>`).join('')||'<div class="small">没有匹配工具</div>'}</div></details>`}
function xmlTag(name,value){const safe=String(name).replace(/[^A-Za-z0-9_.-]/g,'_');return `    <${safe}>${typeof value==='object'?'VALUE_JSON':'VALUE'}</${safe}>`}
function useTool(name){const t=state.detail.tools.find(x=>x.name===name),p=t?.parameters&&typeof t.parameters==='object'?Object.entries(t.parameters):[];const params=p.length?`<parameters>\n${p.map(([k,v])=>xmlTag(k,v)).join('\n')}\n  </parameters>`:'<parameters/>';$('#response').value=`<decision>\n  <kind>tool_call</kind>\n  <name>${name}</name>\n  ${params}\n  <reasoning>Why this is the next action.</reasoning>\n</decision>`;$('#response').focus()}
function renderDetail(){const d=state.detail,v=$('#viewer'),c=$('#composer');if(!d){v.innerHTML='<div class="empty">选择一个请求。</div>';c.classList.add('hidden');return}$('#title').textContent=`${d.model} · ${d.status} · ${d.id.slice(0,10)}`;$('#rawLink').href=`/api/requests/${d.id}/raw`;$('#rawLink').classList.remove('hidden');const pageCount=Math.max(1,Math.ceil(d.messages.length/PAGE_MESSAGES));state.messagePage=Math.min(state.messagePage,pageCount-1);const start=state.messagePage*PAGE_MESSAGES,visible=d.messages.slice(start,start+PAGE_MESSAGES),pager=mainPager(state.messagePage,pageCount);v.innerHTML=`<div class="meta"><span>${d.message_count} messages</span><span>${d.image_count} images</span><span>${d.tools?.length||0} tools</span><span>received ${new Date(d.received_at*1000).toLocaleString()}</span></div>`+toolCatalog(d)+pager+visible.map((m,j)=>{const i=start+j;return `<article class="message"><div class="message-head role-${esc(m.role)}">${i+1} · ${esc(m.role||'unknown')}</div>${Array.isArray(m.content)?m.content.map((p,k)=>partHtml(p,k,`m${i}p${k}`)).join(''):textHtml(m.content??'',`m${i}`)}</article>`}).join('')+pager+`<details class="message"><summary class="message-head">Request options</summary><pre class="object-part">${esc(JSON.stringify(d.request_options,null,2))}</pre></details>`+(d.response_text?`<article class="message"><div class="message-head role-assistant">MANUAL RESPONSE</div>${textHtml(d.response_text,'response')}</article>`:'')+(d.response_error?`<article class="message"><div class="message-head cancelled">CANCELLED</div>${textHtml(d.response_error,'error')}</article>`:'');v.querySelectorAll('img[data-full]').forEach(img=>img.onclick=()=>{$('#modalImage').src=img.dataset.full;$('#modal').classList.remove('hidden')});v.querySelectorAll('.text-page').forEach(b=>b.onclick=()=>{state.textPages[b.dataset.key]=(state.textPages[b.dataset.key]||0)+Number(b.dataset.delta);renderDetail()});v.querySelectorAll('.main-page').forEach(b=>b.onclick=()=>{state.messagePage+=Number(b.dataset.delta);v.scrollTop=0;renderDetail()});v.querySelectorAll('.use-tool').forEach(b=>b.onclick=()=>useTool(b.dataset.name));const search=$('#toolSearch');if(search)search.oninput=e=>{state.toolFilter=e.target.value;renderDetail();const next=$('#toolSearch');if(next){next.focus();next.setSelectionRange(next.value.length,next.value.length)}};if(d.status==='pending'){c.classList.remove('hidden');$('#mode').textContent=d.response_mode==='json'?'当前应回复 JSON':'当前应回复 XML <decision>'}else c.classList.add('hidden')}
async function select(id){state.selected=id;state.messagePage=0;state.textPages={};state.toolFilter='';renderQueue();state.detail=await api(`/api/requests/${id}`);renderDetail()}
async function poll(){try{const x=await api('/api/requests');const old=state.requests.map(r=>r.id+':'+r.status).join();state.requests=x.requests;renderQueue();if(!state.selected){const p=state.requests.find(r=>r.status==='pending')||state.requests[0];if(p)await select(p.id)}else{const now=state.requests.map(r=>r.id+':'+r.status).join();if(now!==old){state.detail=await api(`/api/requests/${state.selected}`);renderDetail();const p=state.requests.find(r=>r.status==='pending');if(state.detail.status!=='pending'&&p)await select(p.id)}}}catch(e){console.error(e)}finally{setTimeout(poll,750)}}
const templates={tool:'<decision>\n  <kind>tool_call</kind>\n  <name>TOOL_NAME</name>\n  <parameters>\n    <PARAMETER>VALUE</PARAMETER>\n  </parameters>\n  <reasoning>Why this is the next action.</reasoning>\n</decision>',talk:'<decision>\n  <kind>response</kind>\n  <name>talk</name>\n  <parameters><message>MESSAGE</message></parameters>\n  <reasoning>Why no tool call is needed.</reasoning>\n</decision>',complete:'<decision>\n  <kind>response</kind>\n  <name>task_complete</name>\n  <parameters/>\n  <reasoning>Trusted completion evidence.</reasoning>\n</decision>'};document.querySelectorAll('[data-template]').forEach(b=>b.onclick=()=>{$('#response').value=templates[b.dataset.template];$('#response').focus()});
$('#submit').onclick=async()=>{try{await api(`/api/requests/${state.selected}/response`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({content:$('#response').value})});$('#response').value='';toast('Response sent to harness')}catch(e){toast(e.message)}};$('#cancel').onclick=async()=>{if(!confirm('Cancel this waiting provider request?'))return;try{await api(`/api/requests/${state.selected}/cancel`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reason:'Cancelled by the human VLM operator.'})});toast('Request cancelled')}catch(e){toast(e.message)}};const responseBox=$('#response'),savedHeight=Number(localStorage.getItem('manual-vlm-response-height'));if(savedHeight>=80)responseBox.style.height=Math.min(savedHeight,window.innerHeight*.65)+'px';new ResizeObserver(entries=>{const height=Math.round(entries[0].contentRect.height);if(height>=80)localStorage.setItem('manual-vlm-response-height',String(height))}).observe(responseBox);responseBox.onkeydown=e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){$('#submit').click()}};$('#closeModal').onclick=()=>$('#modal').classList.add('hidden');$('#modal').onclick=e=>{if(e.target.id==='modal'||e.target.classList.contains('modal-stage'))$('#modal').classList.add('hidden')};poll();
</script></body></html>'''


def load_console_html() -> str:
    """Load the human-oriented console, retaining the legacy inline UI as fallback."""

    try:
        return CONSOLE_HTML_PATH.read_text(encoding="utf-8")
    except OSError:
        return GUI_HTML


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Human-driven OpenAI-compatible VLM proxy and prompt inspector."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: loopback).")
    parser.add_argument("--port", type=int, default=8099, help="Listen port.")
    parser.add_argument(
        "--decision-timeout",
        type=float,
        default=0,
        metavar="SECONDS",
        help="Cancel unanswered requests after this many seconds; 0 waits forever.",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        help="Optionally persist exact request/response JSON under this directory.",
    )
    parser.add_argument("--history-limit", type=int, default=100)
    parser.add_argument("--max-body-mib", type=int, default=64)
    parser.add_argument("--open", action="store_true", help="Open the GUI in a browser.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    timeout = args.decision_timeout if args.decision_timeout > 0 else None
    store = RequestStore(history_limit=args.history_limit, record_dir=args.record_dir)
    server = ManualVLMServer(
        (args.host, args.port),
        store=store,
        decision_timeout_s=timeout,
        max_body_bytes=max(1, args.max_body_mib) * 1024 * 1024,
    )
    actual_host, actual_port = server.server_address[:2]
    browser_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
    url = f"http://{browser_host}:{actual_port}/"
    print(f"Manual VLM GUI: {url}")
    print(f"OpenAI API base: http://{browser_host}:{actual_port}/v1")
    print("Use any non-empty API key. Press Ctrl-C to stop.")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print("WARNING: this server has no authentication; do not expose it to untrusted networks.")
    if args.open:
        threading.Timer(0.25, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping manual VLM proxy.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
