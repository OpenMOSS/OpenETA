"""Small local client for the persistent Viser grasp-inspector sidecar."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


class GraspInspectorClient:
    """Call the viewer's JSON API without inheriting shell proxy settings."""

    def __init__(self, url: str, *, timeout_s: float = 15.0) -> None:
        self.url = url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self._opener = build_opener(ProxyHandler({}))

    def state(self) -> dict[str, Any]:
        return self._request("GET", "/state")

    def configure(self, **arguments: Any) -> dict[str, Any]:
        return self._request("POST", "/configure", arguments)

    def add_pose(self, **arguments: Any) -> dict[str, Any]:
        return self._request("POST", "/add_pose", arguments)

    def capture(self, **arguments: Any) -> dict[str, Any]:
        return self._request("POST", "/capture", arguments)

    def capture_image(self, **arguments: Any) -> tuple[dict[str, Any], Path | None]:
        result = self.capture(**arguments)
        if not result.get("success"):
            return result, None
        image_ref = result.get("image_ref")
        if not isinstance(image_ref, str) or not image_ref:
            return self._failure(
                "capture_missing_image", "Viewer reported success without image_ref."
            ), None
        path = Path(image_ref).expanduser().resolve()
        if not path.is_file():
            return self._failure(
                "capture_missing_image", f"Viewer image does not exist: {path}"
            ), None
        return result, path

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None
        headers: dict[str, str] = {}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.url}{path}", data=body, headers=headers, method=method
        )
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                raw = response.read()
        except HTTPError as exc:
            raw = exc.read()
            if not raw:
                return self._failure(
                    "inspector_http_error", f"Viewer returned HTTP {exc.code}."
                )
        except (URLError, TimeoutError, OSError) as exc:
            return self._failure("inspector_unavailable", str(exc))
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return self._failure("invalid_inspector_response", str(exc))
        if not isinstance(value, dict):
            return self._failure(
                "invalid_inspector_response", "Viewer response is not a JSON object."
            )
        return value

    @staticmethod
    def _failure(code: str, message: str) -> dict[str, Any]:
        return {
            "success": False,
            "code": code,
            "retryable": code
            in {
                "inspector_unavailable",
                "inspector_http_error",
                "capture_missing_image",
            },
            "message": message,
        }


__all__ = ["GraspInspectorClient"]
