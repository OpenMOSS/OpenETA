from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tools.grasp_inspector_client import GraspInspectorClient


class Handler(BaseHTTPRequestHandler):
    image_path: Path

    def do_GET(self) -> None:  # noqa: N802
        self._write({"success": True, "viewer_clients": [{"viewer_id": "0"}]})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/capture":
            self._write(
                {
                    "success": True,
                    "camera_source": request.get("camera"),
                    "image_ref": str(self.image_path),
                }
            )
        else:
            self._write({"success": True, "request": request})

    def _write(self, payload: object) -> None:
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        pass


def test_client_state_configure_and_capture_image(tmp_path: Path) -> None:
    image = tmp_path / "view.png"
    image.write_bytes(b"png")
    Handler.image_path = image
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = GraspInspectorClient(
            f"http://127.0.0.1:{server.server_address[1]}"
        )
        assert client.state()["viewer_clients"] == [{"viewer_id": "0"}]
        configured = client.configure(pose_scope="focus", focus_pose_id="GX0")
        assert configured["request"]["pose_scope"] == "focus"
        added = client.add_pose(
            pose_id="refined-1",
            transform_world_from_grip_site=[
                [1, 0, 0, 0.1],
                [0, 1, 0, 0.2],
                [0, 0, 1, 0.3],
                [0, 0, 0, 1],
            ],
        )
        assert added["request"]["pose_id"] == "refined-1"
        captured, path = client.capture_image(camera="top")
        assert captured["camera_source"] == "top"
        assert path == image.resolve()
    finally:
        server.shutdown()
        server.server_close()


def test_client_reports_unavailable_sidecar() -> None:
    client = GraspInspectorClient("http://127.0.0.1:1", timeout_s=0.05)
    result = client.state()
    assert result["success"] is False
    assert result["code"] == "inspector_unavailable"
    assert result["retryable"] is True
