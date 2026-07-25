"""Approved Python execution in a disposable host subprocess."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adapter.protocol import JsonDict


_CHILD_RUNNER = r"""
import json
import pathlib
import sys
import traceback

payload = json.loads(sys.stdin.read())
output_path = pathlib.Path(sys.argv[1])
scope = {
    "observation": payload.get("observation"),
    "parameters": payload.get("parameters", {}),
}

try:
    exec(compile(payload["code"], "<openeta-outside-python-exec>", "exec"), scope, scope)
    result = scope.get("result")
    try:
        json.dumps(result)
    except (TypeError, ValueError):
        result = repr(result)
    response = {"success": True, "result": result}
except BaseException as exc:
    response = {
        "success": False,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(limit=10),
    }

output_path.write_text(json.dumps(response), encoding="utf-8")
sys.exit(0 if response["success"] else 1)
"""


@dataclass(frozen=True, slots=True)
class OutsidePythonExecution:
    """Structured result from one approved host subprocess."""

    success: bool
    result: Any = None
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    error_type: str = ""
    message: str = ""
    traceback: str = ""
    timed_out: bool = False


@dataclass(slots=True)
class OutsidePythonExecutor:
    """Run approved code outside the restricted in-process globals."""

    executable: str = sys.executable
    cwd: str | Path | None = None
    max_output_chars: int = 100_000

    def execute(
        self,
        code: str,
        *,
        parameters: JsonDict,
        observation: JsonDict | None,
        timeout_s: float,
    ) -> OutsidePythonExecution:
        payload = json.dumps(
            {
                "code": code,
                "parameters": parameters,
                "observation": observation,
            }
        )
        with tempfile.TemporaryDirectory(prefix="openeta-outside-python-") as temp_dir:
            output_path = Path(temp_dir) / "result.json"
            process = subprocess.Popen(  # noqa: S603 - executable and argv are fixed.
                [self.executable, "-c", _CHILD_RUNNER, str(output_path)],
                cwd=str(self.cwd) if self.cwd is not None else None,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(payload, timeout=max(0.01, timeout_s))
            except subprocess.TimeoutExpired:
                self._terminate_process_group(process)
                stdout, stderr = process.communicate()
                return OutsidePythonExecution(
                    success=False,
                    stdout=self._bounded(stdout),
                    stderr=self._bounded(stderr),
                    returncode=process.returncode,
                    error_type="TimeoutExpired",
                    message=f"outside_sandbox execution exceeded {timeout_s:g}s",
                    timed_out=True,
                )

            response = self._read_response(output_path)
            return OutsidePythonExecution(
                success=bool(response.get("success")) and process.returncode == 0,
                result=response.get("result"),
                stdout=self._bounded(stdout),
                stderr=self._bounded(stderr),
                returncode=process.returncode,
                error_type=str(response.get("error_type") or ""),
                message=str(response.get("message") or ""),
                traceback=str(response.get("traceback") or ""),
            )

    def _read_response(self, path: Path) -> JsonDict:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "success": False,
                "error_type": type(exc).__name__,
                "message": f"outside_sandbox subprocess produced no valid result: {exc}",
            }
        if not isinstance(value, dict):
            return {
                "success": False,
                "error_type": "InvalidResult",
                "message": "outside_sandbox subprocess result must be a JSON object",
            }
        return value

    def _bounded(self, value: str) -> str:
        if len(value) <= self.max_output_chars:
            return value
        omitted = len(value) - self.max_output_chars
        return f"{value[: self.max_output_chars]}\n... <{omitted} chars omitted>"

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[str]) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
