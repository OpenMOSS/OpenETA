"""Read-only client for a Robotiq gripper via the UR URCap socket (port 63352).

The Robotiq URCap runs a small socket server on the UR controller that exposes
gripper registers as text commands: ``GET <VAR>\n`` -> ``<VAR> <value>\n``.
This client issues **only GET queries** — it never activates or moves the
gripper (activation physically re-calibrates the fingers, so it is out of
scope for an observation-first read).

Register reference (subset we read):
  ACT  activation flag        0 = not activated, 1 = activated
  GTO  go-to / motion request 0/1
  STA  gripper status         0 = reset/not-activated, 1/2 = activating,
                              3 = activation complete (ready)
  OBJ  object detection       0 = moving, 1 = stopped opening (obj on open side),
                              2 = stopped closing (obj on close side),
                              3 = at requested position (no object)
  POS  current position       0 = fully open .. 255 = fully closed
  PRE  requested position echo
  FLT  fault status           00 = no fault
  SPE  speed setting          0..255
  FOR  force setting          0..255
"""

from __future__ import annotations

import socket
import threading

_STA_TEXT = {0: "reset", 1: "activating", 2: "activating", 3: "ready"}
_OBJ_TEXT = {
    0: "moving",
    1: "object_detected_opening",
    2: "object_detected_closing",
    3: "at_position_no_object",
}


class RobotiqGripperReader:
    """Thread-safe, read-only reader for a URCap-hosted Robotiq gripper."""

    def __init__(self, host: str, port: int = 63352, *, timeout: float = 3.0) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def _connect(self) -> socket.socket:
        if self._sock is not None:
            return self._sock
        sock = socket.create_connection((self._host, self._port), timeout=self._timeout)
        sock.settimeout(self._timeout)
        self._sock = sock
        return sock

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None

    def _get(self, var: str) -> int | None:
        """Send ``GET <var>`` and parse the trailing integer, or None on error."""
        sock = self._connect()
        sock.sendall(f"GET {var}\n".encode())
        reply = sock.recv(1024).decode(errors="replace").strip()
        # reply looks like "POS 25"; take the last whitespace token.
        parts = reply.split()
        if len(parts) >= 2:
            try:
                return int(parts[-1])
            except ValueError:
                return None
        return None

    def _set(self, var: str, value: int) -> str:
        """Send ``SET <var> <value>`` and return the raw ack, e.g. ``'ack'``."""
        sock = self._connect()
        sock.sendall(f"SET {var} {value}\n".encode())
        return sock.recv(1024).decode(errors="replace").strip()

    # -- control (issues MOTION on the physical gripper) ------------------
    def activate(self, *, wait: bool = True, timeout: float = 8.0) -> dict:
        """Activate the gripper (``SET ACT 1``) — physically re-homes the fingers.

        Idempotent-ish: if already activated (STA==3) this returns immediately.
        Waits for STA==3 (activation complete) unless ``wait=False``.
        """
        import time
        with self._lock:
            if self._get("STA") == 3 and self._get("ACT") == 1:
                return {"ok": True, "already_active": True}
            self._set("ACT", 0)   # clear first, per Robotiq activation sequence
            self._set("ACT", 1)
            if not wait:
                return {"ok": True, "already_active": False, "waited": False}
            deadline = time.time() + timeout
            while time.time() < deadline:
                if self._get("STA") == 3:
                    return {"ok": True, "already_active": False, "waited": True}
                if self._get("FLT"):
                    return {"ok": False, "fault": self._get("FLT")}
                time.sleep(0.2)
            return {"ok": False, "timeout": True, "sta": self._get("STA")}

    def go_to(self, position: int, *, speed: int = 255, force: int = 255,
              wait: bool = True, timeout: float = 5.0) -> dict:
        """Move to ``position`` (0=open .. 255=closed). Requires prior activate().

        Sets speed/force, target position, then GTO=1 to start motion. Waits
        until OBJ != 0 (stopped: at target or object-blocked) unless wait=False.
        """
        import time
        position = max(0, min(255, int(position)))
        with self._lock:
            if self._get("STA") != 3:
                return {"ok": False, "error": "gripper not activated (STA!=3); call activate()"}
            self._set("SPE", max(0, min(255, speed)))
            self._set("FOR", max(0, min(255, force)))
            self._set("POS", position)
            self._set("GTO", 1)
            if not wait:
                return {"ok": True, "waited": False, "target": position}
            deadline = time.time() + timeout
            while time.time() < deadline:
                obj = self._get("OBJ")
                if obj in (1, 2, 3):  # motion finished (blocked or at target)
                    return {"ok": True, "waited": True, "target": position,
                            "position": self._get("POS"), "object_detection": obj}
                time.sleep(0.1)
            return {"ok": False, "timeout": True, "target": position,
                    "position": self._get("POS")}

    # NB: fully-open == go_to(0), fully-closed == go_to(255). No open()/close()
    # convenience methods — close() is reserved for socket teardown above.

    def read_state(self) -> dict:
        """Return a JSON-serialisable snapshot of the gripper's read-only state.

        On any socket error the connection is dropped (so the next call retries
        cleanly) and an ``error`` key is returned rather than raising, keeping
        observation reads robust against a transient gripper-socket hiccup.
        """
        with self._lock:
            try:
                raw = {v: self._get(v) for v in ("ACT", "GTO", "STA", "OBJ",
                                                 "POS", "PRE", "FLT", "SPE", "FOR")}
            except OSError as exc:
                self.close()
                return {"model": "robotiq", "connected": False, "error": str(exc)}

        pos = raw.get("POS")
        state = {
            "model": "robotiq",
            "connected": True,
            "activated": raw.get("ACT") == 1,
            "status": _STA_TEXT.get(raw.get("STA"), "unknown"),
            "object_detection": _OBJ_TEXT.get(raw.get("OBJ"), "unknown"),
            "position": pos,  # 0=open .. 255=closed
            "position_normalized": round(pos / 255.0, 4) if pos is not None else None,
            "requested_position": raw.get("PRE"),
            "fault": raw.get("FLT"),
            "speed": raw.get("SPE"),
            "force": raw.get("FOR"),
        }
        # Derive a human-friendly open/closed hint from position.
        if pos is not None:
            state["is_open"] = pos <= 10
            state["is_closed"] = pos >= 230
        return state
