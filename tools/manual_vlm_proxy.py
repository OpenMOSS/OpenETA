"""Standalone human-driven OpenAI-compatible chat-completions proxy.

The server owns transport, storage, image proxying and exact wire audit only.
Request interpretation and response encoding are supplied by an optional
``ProtocolAdapter`` loaded at startup.

Run a protocol-neutral console with::

    python -m tools.manual_vlm_proxy --port 8099 --open

Load a project adapter without introducing a dependency into this module::

    python -m tools.manual_vlm_proxy \
      --adapter some_package.some_adapter:Adapter --port 8099 --open
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse
from uuid import uuid4

from tools.manual_vlm_protocol import (
    EncodedResponse,
    GenericProtocolAdapter,
    JsonObject,
    ProtocolAdapter,
    load_protocol_adapter,
)


DEFAULT_MAX_BODY_BYTES = 64 * 1024 * 1024
CONSOLE_HTML_PATH = Path(__file__).with_name("manual_vlm_console.html")
CONSOLE_JS_PATH = Path(__file__).with_name("manual_vlm_console.js")
DEFAULT_CONFIG_PATH = Path(__file__).with_name("manual_vlm_config.json")


@dataclass(slots=True)
class PendingRequest:
    request_id: str
    body: JsonObject
    session_id: str = ""
    session_source: str = "inferred"
    session_turn: int = 1
    received_at: float = field(default_factory=time.time)
    response: EncodedResponse | None = None
    response_error: str | None = None
    completed_at: float | None = None
    event: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def status(self) -> str:
        if not self.event.is_set():
            return "pending"
        return "cancelled" if self.response_error is not None else "responded"

    @property
    def response_text(self) -> str | None:
        if self.response is None:
            return None
        content = self.response.message.get("content")
        if isinstance(content, str):
            return content
        return json.dumps(self.response.message, ensure_ascii=False)


class RequestStore:
    """Thread-safe queue shared by provider requests and the browser UI."""

    def __init__(
        self,
        *,
        adapter: ProtocolAdapter | None = None,
        history_limit: int = 100,
        record_dir: Path | None = None,
    ) -> None:
        self.adapter = adapter or GenericProtocolAdapter()
        self.history_limit = max(1, history_limit)
        self.record_dir = record_dir
        self._requests: dict[str, PendingRequest] = {}
        self._order: list[str] = []
        self._session_histories: dict[str, list[str]] = {}
        self._session_turns: dict[str, int] = {}
        self._session_audit_records: dict[str, dict[str, JsonObject]] = {}
        self._inferred_session_counter = 0
        self._lock = threading.RLock()

    def add(
        self,
        body: JsonObject,
        *,
        headers: Mapping[str, str] | None = None,
        session_hint: str = "",
    ) -> PendingRequest:
        normalized_headers = {
            str(key).lower(): str(value) for key, value in (headers or {}).items()
        }
        if session_hint:
            normalized_headers.setdefault("x-session-id", session_hint)
        with self._lock:
            session_id, session_source, history = self._resolve_session_locked(
                body, headers=normalized_headers
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
            records = self._session_audit_records.setdefault(session_id, {})
            for raw_record in self.adapter.audit_records(body):
                if not isinstance(raw_record, dict):
                    continue
                record_id = raw_record.get("id")
                if not isinstance(record_id, str) or not record_id:
                    continue
                previous = records.get(record_id, {})
                records[record_id] = {
                    **previous,
                    **raw_record,
                    "first_seen_turn": previous.get("first_seen_turn", session_turn),
                    "last_seen_turn": session_turn,
                }
            while len(records) > 1_000:
                records.pop(next(iter(records)))
            self._prune_locked()
        self._record(request, "request", body)
        return request

    def _resolve_session_locked(
        self,
        body: JsonObject,
        *,
        headers: Mapping[str, str],
    ) -> tuple[str, str, list[str]]:
        explicit, source = self.adapter.session_identity(body, headers)
        history = conversation_lineage(body)
        if explicit:
            return explicit, source or "adapter", history
        candidates = [
            (len(previous), session_id)
            for session_id, previous in self._session_histories.items()
            if len(previous) < len(history) and history[: len(previous)] == previous
        ]
        if candidates:
            return max(candidates)[1], "inferred-lineage", history
        if self.adapter.attempt(body) > 1:
            exact = [
                session_id
                for session_id, previous in self._session_histories.items()
                if previous == history
            ]
            if exact:
                return exact[-1], "inferred-retry", history
        self._inferred_session_counter += 1
        root = history[0] if history else str(body.get("model") or "request")
        digest = hashlib.sha256(root.encode("utf-8")).hexdigest()[:6]
        return f"inferred-{self._inferred_session_counter:03d}-{digest}", "inferred-new", history

    def get(self, request_id: str) -> PendingRequest | None:
        with self._lock:
            return self._requests.get(request_id)

    def _summary(self, request: PendingRequest) -> JsonObject:
        messages = request.body.get("messages")
        messages = messages if isinstance(messages, list) else []
        classification = self.adapter.classify_request(request.body)
        return {
            "id": request.request_id,
            "status": request.status,
            "received_at": request.received_at,
            "completed_at": request.completed_at,
            "model": str(request.body.get("model") or ""),
            "session_id": request.session_id,
            "session_source": request.session_source,
            "session_turn": request.session_turn,
            "message_count": len(messages),
            "image_count": len(extract_images(request.body)),
            "request_type": str(classification.get("type") or "request"),
            "request_label": str(classification.get("label") or "Request"),
            "response_mode": str(classification.get("response_mode") or "raw"),
            "attempt": int(classification.get("attempt") or 1),
            "validation_error_count": int(classification.get("validation_error_count") or 0),
            "task": str(classification.get("task") or ""),
            "adapter_id": self.adapter.adapter_id,
            "response_preview": (request.response_text or request.response_error or "")[:240],
        }

    def summaries(self) -> list[JsonObject]:
        with self._lock:
            return [self._summary(self._requests[item]) for item in reversed(self._order)]

    def respond(self, request_id: str, submission: JsonObject | str) -> PendingRequest:
        if isinstance(submission, str):
            submission = {"content": submission}
        with self._lock:
            request = self._require_pending_locked(request_id)
            encoded = self.adapter.encode_response(
                request.body,
                submission,
                request_id=request.request_id,
            )
            if not isinstance(encoded.message, dict) or encoded.message.get("role") != "assistant":
                raise ValueError("adapter must return an assistant message")
            request.response = encoded
            request.completed_at = time.time()
            request.event.set()
        self._record(
            request,
            "response",
            {"message": encoded.message, "finish_reason": encoded.finish_reason},
        )
        return request

    def cancel(self, request_id: str, reason: str) -> PendingRequest:
        with self._lock:
            request = self._require_pending_locked(request_id)
            request.response_error = reason.strip() or "Cancelled by the human operator."
            request.completed_at = time.time()
            request.event.set()
        self._record(request, "cancel", {"error": request.response_error})
        return request

    def public_detail(self, request_id: str) -> JsonObject | None:
        request = self.get(request_id)
        if request is None:
            return None
        with self._lock:
            audit_records = [
                dict(record)
                for record in self._session_audit_records.get(
                    request.session_id, {}
                ).values()
            ]
        return {
            **self._summary(request),
            "adapter": {"id": self.adapter.adapter_id, "label": self.adapter.label},
            "messages": public_messages(request.body, request_id=request_id),
            "request_options": {
                key: value for key, value in request.body.items() if key != "messages"
            },
            "presentation": self.adapter.presentation(request.body, request_id=request_id),
            "audit_records": audit_records,
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
                images.append(ImagePart(len(images), message_index, part_index, url, detail))
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
        if isinstance(content, list):
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
    if normalized and normalized[-1].startswith("user:"):
        normalized.pop()
    return normalized


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
                {"type": part_type, "chars": len(json.dumps(item, ensure_ascii=False))}
            )
    return parts


def build_wire_audit(body: JsonObject) -> JsonObject:
    """Describe the complete model input without changing the original body."""

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
            {"index": index, "role": str(message.get("role") or "unknown"), "parts": parts}
        )
    normalized = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {
        "message_count": len(messages),
        "text_chars": text_chars,
        "image_count": image_count,
        "normalized_body_bytes": len(normalized),
        "normalized_sha256": hashlib.sha256(normalized).hexdigest(),
        "messages": rows,
        "note": (
            "Message values are complete. Inline image data is exposed through "
            "byte-preserving image endpoints; Raw JSON retains the original data URLs."
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

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._serve_static(CONSOLE_HTML_PATH, "text/html; charset=utf-8")
            return
        if path == "/manual-vlm-console.js":
            self._serve_static(CONSOLE_JS_PATH, "text/javascript; charset=utf-8")
            return
        if path in {"/health", "/api/health"}:
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "pending": self._pending_count(),
                    "adapter": {
                        "id": self.server.store.adapter.adapter_id,
                        "label": self.server.store.adapter.label,
                    },
                },
            )
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
            detail = self.server.store.public_detail(request_id)
            if detail is None:
                self._send_error_json(HTTPStatus.NOT_FOUND, "not_found", "Request not found.")
                return
            self._send_json(HTTPStatus.OK, detail)
        elif tail == "raw":
            self._send_json(HTTPStatus.OK, request.body)
        elif tail.startswith("images/"):
            self._serve_image(request, tail.split("/", 1)[1])
        else:
            self._send_error_json(HTTPStatus.NOT_FOUND, "not_found", "Route not found.")

    def do_POST(self) -> None:  # noqa: N802
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
                request = self.server.store.respond(request_id, body)
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
        self._send_json(HTTPStatus.OK, self.server.store._summary(request))

    def _handle_completion(self) -> None:
        try:
            body = self._read_json_body()
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "invalid_request", str(exc))
            return
        if not isinstance(body.get("messages"), list):
            self._send_error_json(
                HTTPStatus.BAD_REQUEST, "invalid_request", "messages must be an array"
            )
            return
        headers = {str(key).lower(): str(value) for key, value in self.headers.items()}
        request = self.server.store.add(body, headers=headers)
        completed = request.event.wait(timeout=self.server.decision_timeout_s)
        if not completed:
            try:
                self.server.store.cancel(request.request_id, "Human response timed out.")
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
        encoded = request.response or EncodedResponse(
            message={"role": "assistant", "content": ""}
        )
        response = {
            "id": f"chatcmpl-human-{request.request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(body.get("model") or "human-vlm"),
            "choices": [
                {
                    "index": 0,
                    "message": encoded.message,
                    "finish_reason": encoded.finish_reason,
                }
            ],
        }
        self._send_json(HTTPStatus.OK, response)

    def _serve_static(self, path: Path, content_type: str) -> None:
        try:
            data = path.read_bytes()
        except OSError:
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "asset_missing",
                f"Console asset is missing: {path.name}",
            )
            return
        self._send_bytes(HTTPStatus.OK, data, content_type)

    def _serve_image(self, request: PendingRequest, raw_index: str) -> None:
        try:
            image = extract_images(request.body)[int(raw_index)]
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
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length <= 0:
            raise ValueError("request body is empty")
        if length > self.server.max_body_bytes:
            raise ValueError(f"request body exceeds {self.server.max_body_bytes} byte limit")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body is not valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    @staticmethod
    def _request_route(path: str) -> tuple[str, str] | None:
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
        self._send_bytes(
            status,
            json.dumps(value, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            cache="no-store",
        )

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


def load_console_html() -> str:
    return CONSOLE_HTML_PATH.read_text(encoding="utf-8")


def load_default_adapter_spec(config_path: Path = DEFAULT_CONFIG_PATH) -> str:
    """Read the host project's adapter choice without coupling the core to it."""

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "generic"
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid manual VLM config {config_path}: {exc}") from exc
    adapter = config.get("default_adapter") if isinstance(config, dict) else None
    if not isinstance(adapter, str) or not adapter.strip():
        raise ValueError(
            f"manual VLM config {config_path} must define a non-empty default_adapter"
        )
    return adapter.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Human-driven OpenAI-compatible chat-completions proxy."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: loopback).")
    parser.add_argument("--port", type=int, default=8099, help="Listen port.")
    parser.add_argument(
        "--adapter",
        help=(
            "Protocol adapter as module:attribute. If omitted, use the project "
            "configuration; pass 'generic' for protocol-neutral mode."
        ),
    )
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
    try:
        adapter_spec = args.adapter or load_default_adapter_spec()
        adapter = load_protocol_adapter(adapter_spec)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        selected = args.adapter or str(DEFAULT_CONFIG_PATH)
        raise SystemExit(f"Unable to load protocol adapter from {selected!r}: {exc}") from exc
    timeout = args.decision_timeout if args.decision_timeout > 0 else None
    store = RequestStore(
        adapter=adapter,
        history_limit=args.history_limit,
        record_dir=args.record_dir,
    )
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
    print(f"Protocol adapter: {adapter.label} ({adapter.adapter_id})")
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
