"""Restricted coding tool runtime for OpenETA agent tools."""

from __future__ import annotations

import io
import importlib
import hashlib
import json
import math
import re
import statistics
import tempfile
import traceback
from types import SimpleNamespace
from collections import Counter, defaultdict, deque
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from adapter.protocol import JsonDict
from agent.runtime.artifact_paths import (
    artifact_session_id,
    artifact_session_root,
)
from agent.runtime.image_artifacts import (
    DEFAULT_MCP_IMAGE_OUTPUT_ROOT,
)
from agent.runtime.response_artifacts import DEFAULT_RESPONSE_ARTIFACT_OUTPUT_ROOT
from agent.runtime.structured_artifacts import materialize_structured_tool_output
from agent.tools.outside_python import OutsidePythonExecutor
from agent.runtime.text_artifacts import (
    DEFAULT_MAX_INLINE_TEXT_CHARS,
    DEFAULT_TEXT_ARTIFACT_OUTPUT_ROOT,
    grep_text_artifact,
    materialize_long_texts,
)
from agent.tools.registry import (
    ToolExecutionContext,
    ToolResult,
    make_tool_result_details,
)


ApprovalCallback = Callable[[ToolExecutionContext, str], bool]

_SAFE_IMPORT_ROOTS = frozenset(
    {
        "base64",
        "bisect",
        "collections",
        "csv",
        "datetime",
        "decimal",
        "fractions",
        "functools",
        "heapq",
        "io",
        "itertools",
        "json",
        "math",
        "matplotlib",
        "numpy",
        "pandas",
        "pathlib",
        "PIL",
        "random",
        "re",
        "statistics",
    }
)


@dataclass(slots=True)
class PythonExecConfig:
    """Configuration for the generic coding tool."""

    default_timeout_s: float = 120.0
    image_output_root: str = str(DEFAULT_MCP_IMAGE_OUTPUT_ROOT)
    text_output_root: str = str(DEFAULT_TEXT_ARTIFACT_OUTPUT_ROOT)
    response_output_root: str = str(DEFAULT_RESPONSE_ARTIFACT_OUTPUT_ROOT)
    structured_output_root: str = str(Path("tmp") / "tool_result")
    max_inline_text_chars: int = DEFAULT_MAX_INLINE_TEXT_CHARS
    max_inline_structured_chars: int = 8000
    allow_outside_sandbox: bool = False
    approve_outside_sandbox: ApprovalCallback | None = None
    outside_executor: OutsidePythonExecutor = field(default_factory=OutsidePythonExecutor)
    max_outside_timeout_s: float = 600.0
    extra_globals: JsonDict = field(default_factory=dict)
    session_root: str | None = None
    workspace_root: str | None = None


def _compile_agent_code(code: str) -> Any:
    """Compile agent code, tolerating over-escaped JSON string payloads.

    Some planner providers emit the ``code`` argument with literal ``\\n`` or
    ``\\t`` two-character sequences instead of real control characters. Only
    retry with a repaired variant when the verbatim snippet fails to compile,
    so valid Python string escapes keep their original meaning.
    """

    try:
        return compile(code, "<openeta-python-exec>", "exec")
    except SyntaxError:
        if "\\n" not in code and "\\t" not in code:
            raise
        repaired = (
            code.replace("\\r\\n", "\n")
            .replace("\\n", "\n")
            .replace("\\t", "\t")
        )
        repaired = repaired.rstrip()
        if repaired.endswith("\\") and not repaired.endswith("\\\\"):
            repaired = repaired[:-1].rstrip()
        if repaired == code:
            raise
        return compile(repaired, "<openeta-python-exec>", "exec")


class PythonExecRuntime:
    """Execute small agent-generated Python snippets with a narrow API surface."""

    def __init__(self, config: PythonExecConfig | None = None) -> None:
        self.config = config or PythonExecConfig()

    def handler(self, context: ToolExecutionContext) -> ToolResult:
        code = str(context.parameters.get("code", "") or "")
        if not code.strip():
            return ToolResult(False, content="python_exec requires non-empty code.")

        sandbox_mode = str(context.parameters.get("sandbox", "sandbox") or "sandbox").strip()
        if sandbox_mode not in {"sandbox", "outside_sandbox"}:
            return _python_exec_result(
                context,
                success=False,
                content=f"Unsupported sandbox mode: {sandbox_mode}",
                diagnostics=[{"code": "unsupported_sandbox_mode"}],
            )
        if sandbox_mode == "outside_sandbox" and not self._outside_sandbox_allowed(context):
            return _python_exec_result(
                context,
                success=False,
                content="outside_sandbox execution requires user approval.",
                diagnostics=[{"code": "outside_sandbox_requires_approval"}],
            )
        if sandbox_mode == "outside_sandbox":
            return self._handle_outside_sandbox(context, code)

        session_id = artifact_session_id(context.metadata)
        invocation_id = uuid4().hex[:10]
        artifacts = _ArtifactApi(
            image_output_root=self.config.image_output_root,
            text_output_root=self.config.text_output_root,
            response_output_root=self.config.response_output_root,
            max_inline_text_chars=self.config.max_inline_text_chars,
            session_id=session_id,
            session_root=self.config.session_root,
        )
        safe_globals = _safe_globals(
            workspace_root=self.config.workspace_root,
            read_roots=(self.config.session_root,) if self.config.session_root else (),
        )
        safe_globals.update(self.config.extra_globals)
        safe_globals.update(
            {
                "api": SimpleNamespace(artifacts=artifacts),
                "artifacts": artifacts,
                "observation": context.observation.to_dict() if context.observation else None,
                "parameters": dict(context.parameters),
                "workspace": _workspace_descriptor(
                    session_root=self.config.session_root,
                    sandbox_root=self.config.workspace_root,
                ),
            }
        )
        before_files = _file_snapshot(self.config.workspace_root)
        stdout = io.StringIO()
        try:
            with redirect_stdout(stdout):
                exec(_compile_agent_code(code), safe_globals, safe_globals)
        except Exception as exc:  # noqa: BLE001 - agent feedback must stay structured.
            outputs = {
                "stdout": stdout.getvalue(),
                "artifacts": list(artifacts.outputs),
            }
            text_bundle = materialize_long_texts(
                outputs,
                output_root=self.config.text_output_root,
                bundle_id=f"python_exec-failed-{invocation_id}",
                max_inline_chars=self.config.max_inline_text_chars,
                session_id=session_id,
            )
            all_artifacts = [
                *artifacts.outputs,
                *_changed_file_artifacts(
                    self.config.workspace_root,
                    before=before_files,
                ),
                *[artifact.to_dict() for artifact in text_bundle.artifacts],
            ]
            text_bundle.payload["artifacts"] = all_artifacts
            diagnostic = _python_exec_exception_diagnostic(exc)
            return _python_exec_result(
                context,
                success=False,
                content=f"python_exec failed: {type(exc).__name__}: {exc}",
                outputs=text_bundle.payload,
                artifacts=all_artifacts,
                diagnostics=[diagnostic],
            )

        result = _json_safe(safe_globals.get("result"))
        structured_artifacts: list[JsonDict] = []
        result_artifact: JsonDict | None = None
        try:
            result_chars = len(json.dumps(result, ensure_ascii=False))
        except (TypeError, ValueError):
            result_chars = 0
        if result_chars > self.config.max_inline_structured_chars:
            result_artifact = materialize_structured_tool_output(
                output_root=self.config.structured_output_root,
                tool="python_exec",
                outputs={"result": result},
                result_id=f"python-exec-{invocation_id}",
            )
            structured_artifacts.append(result_artifact)
        outputs = {
            "result": (
                _structured_result_preview(result, complete_artifact=result_artifact)
                if result_artifact is not None
                else result
            ),
            "result_inline_complete": result_artifact is None,
            **(
                {"result_artifact": dict(result_artifact)}
                if result_artifact is not None
                else {}
            ),
            "stdout": stdout.getvalue(),
            "artifacts": list(artifacts.outputs),
            "sandbox": sandbox_mode,
            "workspace": _workspace_descriptor(
                session_root=self.config.session_root,
                sandbox_root=self.config.workspace_root,
            ),
        }
        text_bundle = materialize_long_texts(
            outputs,
            output_root=self.config.text_output_root,
            bundle_id=(
                f"python_exec-{context.metadata.get('session_id', '') or 'local'}-"
                f"{invocation_id}"
            ),
            max_inline_chars=self.config.max_inline_text_chars,
            session_id=session_id,
        )
        all_artifacts = [
            *artifacts.outputs,
            *structured_artifacts,
            *_changed_file_artifacts(
                self.config.workspace_root,
                before=before_files,
            ),
            *[artifact.to_dict() for artifact in text_bundle.artifacts],
        ]
        text_bundle.payload["artifacts"] = all_artifacts
        return _python_exec_result(
            context,
            success=True,
            content=(
                "python_exec completed; complete structured result saved to "
                f"{result_artifact['path']}"
                if result_artifact is not None
                else "python_exec completed"
            ),
            outputs=text_bundle.payload,
            artifacts=all_artifacts,
            diagnostics=[],
        )

    def _outside_sandbox_allowed(self, context: ToolExecutionContext) -> bool:
        if not self.config.allow_outside_sandbox:
            return False
        if self.config.approve_outside_sandbox is None:
            return False
        return bool(self.config.approve_outside_sandbox(context, "outside_sandbox"))

    def _handle_outside_sandbox(
        self,
        context: ToolExecutionContext,
        code: str,
    ) -> ToolResult:
        try:
            requested_timeout = float(
                context.parameters.get("timeout_s", self.config.default_timeout_s)
            )
        except (TypeError, ValueError):
            requested_timeout = self.config.default_timeout_s
        timeout_s = min(
            max(0.01, requested_timeout),
            max(0.01, self.config.max_outside_timeout_s),
        )
        execution = self.config.outside_executor.execute(
            code,
            parameters=dict(context.parameters),
            observation=context.observation.to_dict() if context.observation else None,
            timeout_s=timeout_s,
        )
        outputs: JsonDict = {
            "result": _json_safe(execution.result),
            "stdout": execution.stdout,
            "stderr": execution.stderr,
            "sandbox": "outside_sandbox",
            "executor": "host_subprocess",
            "returncode": execution.returncode,
            "timeout_s": timeout_s,
        }
        diagnostics: list[JsonDict] = []
        if not execution.success:
            diagnostics.append(
                {
                    "code": (
                        "outside_sandbox_timeout"
                        if execution.timed_out
                        else "outside_sandbox_execution_failed"
                    ),
                    "error_type": execution.error_type,
                    "message": execution.message,
                    **(
                        {"traceback": execution.traceback}
                        if execution.traceback
                        else {}
                    ),
                }
            )
        return _python_exec_result(
            context,
            success=execution.success,
            content=(
                "python_exec completed in approved host subprocess"
                if execution.success
                else f"outside_sandbox python_exec failed: {execution.message}"
            ),
            outputs=outputs,
            diagnostics=diagnostics,
        )


class _ArtifactApi:
    def __init__(
        self,
        *,
        image_output_root: str,
        text_output_root: str,
        response_output_root: str,
        max_inline_text_chars: int,
        session_id: str = "",
        session_root: str | None = None,
    ) -> None:
        self.image_output_root = image_output_root
        self.text_output_root = text_output_root
        self.response_output_root = response_output_root
        self.max_inline_text_chars = max_inline_text_chars
        self.session_id = session_id
        self.session_root = Path(session_root).resolve() if session_root else None
        self.outputs: list[JsonDict] = []

    def describe(self) -> JsonDict:
        roots = [str(root) for root in self._owned_roots()]
        return {
            "session_root": str(self.session_root) if self.session_root else None,
            "read_only_roots": roots,
            "capabilities": ["list_files", "list_images", "read_json", "read_text", "grep_text"],
        }

    def list_files(self, *, pattern: str = "*", limit: int = 100) -> JsonDict:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise PermissionError("Artifact glob must stay inside the current Agent session.")
        files: list[JsonDict] = []
        for root in self._owned_roots():
            if not root.exists():
                continue
            for path in root.rglob(pattern):
                if not path.is_file():
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                files.append(
                    {
                        "path": str(path),
                        "relative_path": str(path.relative_to(root)),
                        "root": str(root),
                        "byte_size": stat.st_size,
                        "mtime_s": stat.st_mtime,
                    }
                )
        files.sort(key=lambda item: float(item.get("mtime_s", 0.0)), reverse=True)
        bounded_limit = max(0, min(int(limit), 1000))
        return {
            "files": files[:bounded_limit],
            "file_count": len(files),
            "truncated": len(files) > bounded_limit,
        }

    def list_images(self, *, limit: int = 20) -> JsonDict:
        root = (
            self.session_root
            if self.session_root is not None
            else artifact_session_root(self.image_output_root, self.session_id).resolve()
        )
        suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bin"}
        images: list[JsonDict] = []
        if root.exists():
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in suffixes:
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                kind = path.parent.parent.name if path.parent.parent != root else path.parent.name
                images.append(
                    {
                        "path": str(path),
                        "kind": kind,
                        "format": path.suffix.lower().lstrip("."),
                        "byte_size": stat.st_size,
                        "mtime_s": stat.st_mtime,
                    }
                )
        images.sort(key=lambda item: float(item.get("mtime_s", 0.0)), reverse=True)
        bounded_limit = max(0, int(limit))
        selected = images[:bounded_limit]
        paths = [str(image["path"]) for image in selected if image.get("path")]
        return {
            "image_root": str(root),
            "images": selected,
            "image_count": len(images),
            "paths": paths,
            "latest_image_path": paths[0] if paths else None,
        }

    def grep_text(
        self,
        path: str,
        pattern: str,
        *,
        max_matches: int = 20,
        ignore_case: bool = True,
    ) -> JsonDict:
        self._require_owned_path(path)
        return grep_text_artifact(
            path,
            pattern,
            max_matches=max_matches,
            ignore_case=ignore_case,
        )

    def read_json(self, path: str) -> JsonDict:
        owned_path = self._require_owned_path(path)
        value = json.loads(owned_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON artifact root must be an object")
        return value

    def read_text(self, path: str, *, max_chars: int = 200_000) -> str:
        owned_path = self._require_owned_path(path)
        limit = max(1, min(int(max_chars), 1_000_000))
        with owned_path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(limit)

    def _owned_roots(self) -> tuple[Path, ...]:
        if self.session_root is not None:
            return (self.session_root,)
        return (
            artifact_session_root(self.image_output_root, self.session_id).resolve(),
            artifact_session_root(self.text_output_root, self.session_id).resolve(),
            artifact_session_root(self.response_output_root, self.session_id).resolve(),
        )

    def _require_owned_path(self, path: str) -> Path:
        resolved = Path(path).expanduser().resolve()
        roots = self._owned_roots()
        if not any(_is_relative_to(resolved, root) for root in roots):
            raise PermissionError("Artifact path is outside the current Agent session.")
        return resolved


def _safe_globals(
    *,
    workspace_root: str | None = None,
    read_roots: tuple[str, ...] = (),
) -> JsonDict:
    workspace = Path(workspace_root).resolve() if workspace_root else None
    readable = tuple(Path(root).resolve() for root in read_roots if str(root).strip())

    def sandbox_open(
        file: str | Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        closefd: bool = True,
        opener: Callable[..., Any] | None = None,
    ) -> Any:
        return _safe_open(
            file,
            mode,
            buffering,
            encoding,
            errors,
            newline,
            closefd,
            opener,
            workspace_root=workspace,
            read_roots=readable,
        )

    def sandbox_import(
        name: str,
        globals: JsonDict | None = None,  # noqa: A002
        locals: JsonDict | None = None,  # noqa: A002
        fromlist: tuple[str, ...] | list[str] = (),
        level: int = 0,
    ) -> Any:
        if workspace is not None and str(name).split(".", 1)[0] == "pathlib":
            del globals, locals, fromlist
            if level != 0:
                raise ImportError("Relative imports are not available in python_exec sandbox.")
            return SimpleNamespace(
                Path=lambda value=".": _WorkspacePath(
                    value,
                    workspace,
                    read_roots=readable,
                )
            )
        return _safe_import(name, globals, locals, fromlist, level)

    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "Exception": Exception,
        "float": float,
        "getattr": getattr,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "next": next,
        "open": sandbox_open,
        "print": print,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "__import__": sandbox_import,
    }
    return {
        "__builtins__": safe_builtins,
        "Counter": Counter,
        "Date": date,
        "Decimal": Decimal,
        "Fraction": Fraction,
        "Path": (
            lambda value=".": _WorkspacePath(value, workspace, read_roots=readable)
        )
        if workspace
        else Path,
        "defaultdict": defaultdict,
        "deque": deque,
        "datetime": datetime,
        "json": json,
        "math": math,
        "re": re,
        "statistics": statistics,
        "timezone": timezone,
    }


def _safe_import(
    name: str,
    globals: JsonDict | None = None,  # noqa: A002 - mirrors __import__ signature.
    locals: JsonDict | None = None,  # noqa: A002 - mirrors __import__ signature.
    fromlist: tuple[str, ...] | list[str] = (),
    level: int = 0,
) -> Any:
    del globals, locals
    if level != 0:
        raise ImportError("Relative imports are not available in python_exec sandbox.")
    root = str(name).split(".", 1)[0]
    if root not in _SAFE_IMPORT_ROOTS:
        raise ImportError(
            f"Import of module '{root}' is not available in python_exec sandbox."
        )
    try:
        module = importlib.import_module(str(name))
    except ImportError as exc:
        raise ImportError(
            f"Module '{name}' is allowed by python_exec sandbox but is not installed "
            "in the current OpenETA runtime environment."
        ) from exc
    if fromlist:
        return module
    return importlib.import_module(root)


def _python_exec_exception_diagnostic(exc: Exception) -> JsonDict:
    diagnostic: JsonDict = {
        "code": "python_exec_exception",
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(limit=5),
    }
    if isinstance(exc, ImportError):
        message = str(exc)
        diagnostic["code"] = "python_exec_import_error"
        if "not available in python_exec sandbox" in message:
            diagnostic["remediation"] = (
                "This module is outside the python_exec sandbox import allowlist. "
                "Do not retry with pip install from the agent loop; either rewrite the snippet "
                "using available helpers, use a dedicated tool/server, or ask the user to approve "
                "outside_sandbox when host-level access is required."
            )
        elif "not installed in the current OpenETA runtime environment" in message:
            diagnostic["remediation"] = (
                "This module is allowed but missing from the configured OpenETA runtime. "
                "Report the missing dependency to the user in natural language so it can be "
                "added to the project environment; do not install packages from inside sandbox."
            )
        else:
            diagnostic["remediation"] = (
                "Import failed in python_exec sandbox. Prefer built-in artifact helpers such as "
                "artifacts.read_json/grep_text; otherwise report the missing dependency to the user."
            )
    return diagnostic


def _safe_open(
    file: str | Path,
    mode: str = "r",
    buffering: int = -1,
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
    closefd: bool = True,
    opener: Callable[..., Any] | None = None,
    *,
    workspace_root: Path | None = None,
    read_roots: tuple[Path, ...] = (),
) -> Any:
    writes = any(flag in mode for flag in ("w", "a", "x", "+"))
    if writes and workspace_root is None:
        raise PermissionError("python_exec sandbox open() is read-only.")
    path = _resolve_workspace_path(file, workspace_root)
    allowed_roots = (
        tuple(root for root in (workspace_root, *read_roots) if root is not None)
        if workspace_root is not None
        else _safe_open_roots()
    )
    if not any(_is_relative_to(path, root) for root in allowed_roots):
        roots = ", ".join(str(root) for root in allowed_roots)
        raise PermissionError(f"python_exec sandbox can only read files under: {roots}")
    if writes and workspace_root is not None and not _is_relative_to(path, workspace_root):
        raise PermissionError(
            "python_exec sandbox can only write below the current session sandbox root."
        )
    return open(  # noqa: PTH123 - this wrapper intentionally delegates to builtins open.
        path,
        mode,
        buffering=buffering,
        encoding=encoding,
        errors=errors,
        newline=newline,
        closefd=closefd,
        opener=opener,
    )


def _resolve_workspace_path(file: str | Path, workspace_root: Path | None) -> Path:
    path = Path(file).expanduser()
    if workspace_root is not None and not path.is_absolute():
        path = workspace_root / path
    return path.resolve()


class _WorkspacePath:
    """Small pathlib-compatible facade with session reads and sandbox-only writes."""

    def __init__(
        self,
        value: str | Path,
        root: Path,
        *,
        read_roots: tuple[Path, ...] = (),
    ) -> None:
        self._root = root.resolve()
        self._read_roots = tuple(path.resolve() for path in read_roots)
        self._path = _resolve_workspace_path(value, self._root)
        self._assert_readable()

    def _assert_readable(self) -> None:
        if not any(
            _is_relative_to(self._path, root)
            for root in (self._root, *self._read_roots)
        ):
            raise PermissionError("Path is outside the current Agent session workspace.")

    def _assert_writable(self) -> None:
        if not _is_relative_to(self._path, self._root):
            raise PermissionError(
                "Path is read-only; python_exec writes are limited to the session sandbox."
            )

    def __fspath__(self) -> str:
        return str(self._path)

    def __str__(self) -> str:
        return str(self._path)

    def __truediv__(self, value: object) -> "_WorkspacePath":
        return _WorkspacePath(
            self._path / str(value),
            self._root,
            read_roots=self._read_roots,
        )

    @property
    def name(self) -> str:
        return self._path.name

    @property
    def suffix(self) -> str:
        return self._path.suffix

    @property
    def parent(self) -> "_WorkspacePath":
        return _WorkspacePath(
            self._path.parent,
            self._root,
            read_roots=self._read_roots,
        )

    def exists(self) -> bool:
        return self._path.exists()

    def is_file(self) -> bool:
        return self._path.is_file()

    def is_dir(self) -> bool:
        return self._path.is_dir()

    def mkdir(self, *, parents: bool = False, exist_ok: bool = False) -> None:
        self._assert_writable()
        self._path.mkdir(parents=parents, exist_ok=exist_ok)

    def read_text(self, *, encoding: str = "utf-8") -> str:
        return self._path.read_text(encoding=encoding)

    def write_text(self, data: str, *, encoding: str = "utf-8") -> int:
        self._assert_writable()
        return self._path.write_text(data, encoding=encoding)

    def read_bytes(self) -> bytes:
        return self._path.read_bytes()

    def write_bytes(self, data: bytes) -> int:
        self._assert_writable()
        return self._path.write_bytes(data)

    def glob(self, pattern: str) -> list["_WorkspacePath"]:
        return [
            _WorkspacePath(path, self._root, read_roots=self._read_roots)
            for path in self._path.glob(pattern)
        ]

    def rglob(self, pattern: str) -> list["_WorkspacePath"]:
        return [
            _WorkspacePath(path, self._root, read_roots=self._read_roots)
            for path in self._path.rglob(pattern)
        ]


def _workspace_descriptor(
    *,
    session_root: str | None,
    sandbox_root: str | None,
) -> JsonDict:
    session = Path(session_root).resolve() if session_root else None
    sandbox = Path(sandbox_root).resolve() if sandbox_root else None
    return {
        "session_root": str(session) if session else None,
        "artifacts_root": str(session / "artifacts") if session else None,
        "memory_root": str(session / "working") if session else None,
        "rollout_root": str(session / "rollout") if session else None,
        "sandbox_root": str(sandbox) if sandbox else None,
        "read_scope": "current_agent_session",
        "write_scope": "sandbox_only" if sandbox else "none",
        "simulator_mcp_available": False,
        "network_available": False,
    }


def _file_snapshot(root: str | None) -> dict[str, tuple[int, int]]:
    if not root:
        return {}
    base = Path(root).resolve()
    if not base.exists():
        return {}
    snapshot: dict[str, tuple[int, int]] = {}
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        snapshot[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def _changed_file_artifacts(
    root: str | None,
    *,
    before: dict[str, tuple[int, int]],
) -> list[JsonDict]:
    after = _file_snapshot(root)
    artifacts: list[JsonDict] = []
    for path_text, identity in sorted(after.items()):
        if before.get(path_text) == identity:
            continue
        path = Path(path_text)
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        artifacts.append(
            {
                "type": "sandbox_file",
                "kind": "derived_artifact",
                "index": path.name,
                "path": path_text,
                "byte_size": identity[0],
                "sha256": digest,
                "mutable": True,
            }
        )
    return artifacts


def _safe_open_roots() -> tuple[Path, ...]:
    roots = [Path.cwd().resolve(), Path(tempfile.gettempdir()).resolve()]
    private_tmp = Path("/private/tmp")
    if private_tmp.exists():
        roots.append(private_tmp.resolve())
    return tuple(roots)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _python_exec_result(
    context: ToolExecutionContext,
    *,
    success: bool,
    content: str,
    outputs: JsonDict | None = None,
    artifacts: list[JsonDict] | None = None,
    diagnostics: list[JsonDict] | None = None,
) -> ToolResult:
    redacted_parameters = {
        key: ("<code omitted>" if key == "code" else value)
        for key, value in context.parameters.items()
    }
    return ToolResult(
        success,
        content=content,
        details=make_tool_result_details(
            context.spec,
            redacted_parameters,
            success=success,
            outputs=outputs,
            artifacts=artifacts,
            diagnostics=diagnostics,
        ),
    )


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _structured_result_preview(
    value: Any,
    *,
    complete_artifact: JsonDict,
) -> JsonDict:
    """Return a small index while the complete Python value remains queryable."""

    preview: JsonDict = {
        "inline_complete": False,
        "complete_result_path": complete_artifact.get("path"),
        "byte_size": complete_artifact.get("byte_size"),
        "sha256": complete_artifact.get("sha256"),
        "grep_hint": complete_artifact.get("grep_hint"),
    }
    if isinstance(value, dict):
        preview["type"] = "object"
        preview["top_level_keys"] = [str(key) for key in list(value)[:50]]
        preview["collection_sizes"] = {
            str(key): len(item)
            for key, item in value.items()
            if isinstance(item, (dict, list, tuple))
        }
    elif isinstance(value, list):
        preview.update({"type": "array", "item_count": len(value)})
        ids = [
            item.get("id")
            for item in value[:20]
            if isinstance(item, dict) and item.get("id") is not None
        ]
        if ids:
            preview["first_ids"] = ids
    else:
        preview["type"] = type(value).__name__
    return preview
