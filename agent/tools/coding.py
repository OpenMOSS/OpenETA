"""Restricted coding tool runtime for OpenETA agent tools."""

from __future__ import annotations

import io
import importlib
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
    materialize_mcp_images,
)
from agent.runtime.response_artifacts import (
    DEFAULT_RESPONSE_ARTIFACT_OUTPUT_ROOT,
    build_response_reference,
    materialize_json_response,
)
from agent.tools.sim_mcp import (
    SimulatorMcpTransport,
    _with_anygrasp_camera_intrinsics,
    mcp_dashboard_url,
    mcp_server_url_from_transport,
)
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
McpResponseCallback = Callable[[str, JsonDict, JsonDict], None]

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
        "pathlib",
        "random",
        "re",
        "statistics",
    }
)


@dataclass(slots=True)
class PythonExecConfig:
    """Configuration for the generic coding tool."""

    mcp_transport: SimulatorMcpTransport | None = None
    default_timeout_s: float = 120.0
    image_output_root: str = str(DEFAULT_MCP_IMAGE_OUTPUT_ROOT)
    text_output_root: str = str(DEFAULT_TEXT_ARTIFACT_OUTPUT_ROOT)
    response_output_root: str = str(DEFAULT_RESPONSE_ARTIFACT_OUTPUT_ROOT)
    max_inline_text_chars: int = DEFAULT_MAX_INLINE_TEXT_CHARS
    allow_outside_sandbox: bool = False
    approve_outside_sandbox: ApprovalCallback | None = None
    outside_executor: OutsidePythonExecutor = field(default_factory=OutsidePythonExecutor)
    max_outside_timeout_s: float = 600.0
    mcp_response_callback: McpResponseCallback | None = None
    extra_globals: JsonDict = field(default_factory=dict)
    workspace_root: str | None = None


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
        mcp = _McpApi(
            transport=self.config.mcp_transport,
            default_timeout_s=self.config.default_timeout_s,
            image_output_root=self.config.image_output_root,
            text_output_root=self.config.text_output_root,
            response_output_root=self.config.response_output_root,
            max_inline_text_chars=self.config.max_inline_text_chars,
            response_callback=self.config.mcp_response_callback,
            session_id=session_id,
            invocation_id=invocation_id,
        )
        artifacts = _ArtifactApi(
            image_output_root=self.config.image_output_root,
            text_output_root=self.config.text_output_root,
            response_output_root=self.config.response_output_root,
            max_inline_text_chars=self.config.max_inline_text_chars,
            session_id=session_id,
        )
        safe_globals = _safe_globals(workspace_root=self.config.workspace_root)
        safe_globals.update(self.config.extra_globals)
        safe_globals.update(
            {
                "api": _OpenEtaCodeApi(mcp=mcp, artifacts=artifacts),
                "mcp": mcp,
                "artifacts": artifacts,
                "observation": context.observation.to_dict() if context.observation else None,
                "parameters": dict(context.parameters),
            }
        )
        stdout = io.StringIO()
        try:
            with redirect_stdout(stdout):
                exec(compile(code, "<openeta-python-exec>", "exec"), safe_globals, safe_globals)
        except Exception as exc:  # noqa: BLE001 - agent feedback must stay structured.
            outputs = {
                "stdout": stdout.getvalue(),
                "mcp_calls": mcp.calls,
                "artifacts": [
                    *artifacts.outputs,
                    *mcp.image_artifacts,
                    *mcp.response_artifacts,
                    *mcp.text_artifacts,
                ],
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
                *mcp.image_artifacts,
                *mcp.response_artifacts,
                *mcp.text_artifacts,
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

        result = safe_globals.get("result")
        outputs = {
            "result": _json_safe(result),
            "stdout": stdout.getvalue(),
            "mcp_calls": mcp.calls,
            "artifacts": [
                *artifacts.outputs,
                *mcp.image_artifacts,
                *mcp.response_artifacts,
                *mcp.text_artifacts,
            ],
            "sandbox": sandbox_mode,
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
            *mcp.image_artifacts,
            *mcp.response_artifacts,
            *mcp.text_artifacts,
            *[artifact.to_dict() for artifact in text_bundle.artifacts],
        ]
        text_bundle.payload["artifacts"] = all_artifacts
        failed_mcp_calls = [
            call for call in mcp.calls if isinstance(call, dict) and call.get("success") is False
        ]
        success = not failed_mcp_calls
        diagnostics = []
        if failed_mcp_calls:
            diagnostics.append(
                {
                    "code": "python_exec_mcp_call_failed",
                    "message": "One or more MCP calls returned success=false.",
                    "failed_tools": [
                        str(call.get("tool") or "") for call in failed_mcp_calls
                    ],
                }
            )
        return _python_exec_result(
            context,
            success=success,
            content=(
                "python_exec completed"
                if success
                else "python_exec completed with failed MCP call(s)"
            ),
            outputs=text_bundle.payload,
            artifacts=all_artifacts,
            diagnostics=diagnostics,
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


class _OpenEtaCodeApi:
    def __init__(self, *, mcp: "_McpApi", artifacts: "_ArtifactApi") -> None:
        self.mcp = mcp
        self.artifacts = artifacts


class _McpApi:
    def __init__(
        self,
        *,
        transport: SimulatorMcpTransport | None,
        default_timeout_s: float,
        image_output_root: str,
        text_output_root: str,
        response_output_root: str,
        max_inline_text_chars: int,
        response_callback: McpResponseCallback | None = None,
        session_id: str = "",
        invocation_id: str = "",
    ) -> None:
        self.transport = transport
        self.default_timeout_s = default_timeout_s
        self.image_output_root = image_output_root
        self.text_output_root = text_output_root
        self.response_output_root = response_output_root
        self.max_inline_text_chars = max_inline_text_chars
        self.response_callback = response_callback
        self.session_id = session_id
        self.invocation_id = invocation_id or uuid4().hex[:10]
        self.calls: list[JsonDict] = []
        self.image_artifacts: list[JsonDict] = []
        self.response_artifacts: list[JsonDict] = []
        self.text_artifacts: list[JsonDict] = []

    def call_tool(
        self,
        name: str,
        arguments: JsonDict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> JsonDict:
        if self.transport is None:
            raise RuntimeError("No MCP transport is configured for python_exec.")
        args = dict(arguments or {})
        if str(name) == "create_env":
            raise RuntimeError(
                "Simulator environment creation is only available through the "
                "create_simulator_env agent tool."
            )
        response_ref, materialized_response = self._call_and_record_mcp_tool(
            str(name),
            args,
            timeout_s=timeout_s,
        )
        return response_ref

    def _call_and_record_mcp_tool(
        self,
        name: str,
        args: JsonDict,
        *,
        timeout_s: float | None,
    ) -> tuple[JsonDict, JsonDict]:
        response = self.transport.call_tool(
            name,
            args,
            timeout_s=timeout_s if timeout_s is not None else self.default_timeout_s,
        )
        bundle_id = (
            f"mcp-{_safe_artifact_token(name)}-{self.invocation_id}-"
            f"{len(self.calls):03d}"
        )
        image_bundle = materialize_mcp_images(
            response,
            output_root=self.image_output_root,
            bundle_id=bundle_id,
            session_id=self.session_id,
        )
        materialized_response = image_bundle.payload
        image_artifacts = [image.to_dict() for image in image_bundle.images]
        if image_bundle.images:
            self.image_artifacts.extend(image_artifacts)
        materialized_response = _with_anygrasp_camera_intrinsics(materialized_response)
        response_artifact = materialize_json_response(
            materialized_response,
            output_root=self.response_output_root,
            bundle_id=bundle_id,
            name="response",
            session_id=self.session_id,
        )
        response_artifact_dict = response_artifact.to_dict()
        self.response_artifacts.append(response_artifact_dict)
        response_ref = build_response_reference(
            materialized_response,
            response_artifact,
            image_artifacts=image_artifacts,
        )
        _attach_mcp_session_urls(
            response_ref,
            transport=self.transport,
            payload=materialized_response,
            arguments=args,
        )
        if self.response_callback is not None:
            self.response_callback(str(name), args, response_ref)
        mcp_server_url = response_ref.get("mcp_server_url")
        dashboard_url = response_ref.get("dashboard_url")
        self.calls.append(
            {
                "tool": name,
                "arguments": args,
                "success": _mcp_response_success(materialized_response),
                "response_path": response_artifact.path,
                "response_chars": response_artifact.chars,
                "grep_hint": response_artifact.grep_hint,
                "image_artifacts": image_artifacts,
                "response_artifact": response_artifact_dict,
                **({"mcp_server_url": mcp_server_url} if mcp_server_url else {}),
                **({"dashboard_url": dashboard_url} if dashboard_url else {}),
            }
        )
        return response_ref, materialized_response

    def list_tools(self, *, timeout_s: float | None = None) -> JsonDict:
        if self.transport is None:
            raise RuntimeError("No MCP transport is configured for python_exec.")
        if not hasattr(self.transport, "list_tools"):
            raise RuntimeError("Configured MCP transport does not support list_tools.")
        response = self.transport.list_tools(
            timeout_s=timeout_s if timeout_s is not None else self.default_timeout_s,
        )
        bundle_id = f"mcp-list_tools-{self.invocation_id}-{len(self.calls):03d}"
        response_artifact = materialize_json_response(
            response,
            output_root=self.response_output_root,
            bundle_id=bundle_id,
            name="response",
            session_id=self.session_id,
        )
        response_artifact_dict = response_artifact.to_dict()
        self.response_artifacts.append(response_artifact_dict)
        response_ref = build_response_reference(response, response_artifact)
        server_url = mcp_server_url_from_transport(self.transport)
        if server_url:
            response_ref["mcp_server_url"] = server_url
        if "tool_count" in response:
            response_ref["tool_count"] = response.get("tool_count")
        self.calls.append(
            {
                "tool": "list_tools",
                "arguments": {},
                "success": True,
                "response_path": response_artifact.path,
                "response_chars": response_artifact.chars,
                "grep_hint": response_artifact.grep_hint,
                "image_artifacts": [],
                "response_artifact": response_artifact_dict,
                "tool_count": response.get("tool_count"),
            }
        )
        return response_ref


def _mcp_response_success(payload: JsonDict) -> bool:
    return not (
        payload.get("success") is False
        or payload.get("ok") is False
        or "error" in payload
    )


def _attach_mcp_session_urls(
    response_ref: JsonDict,
    *,
    transport: SimulatorMcpTransport | None,
    payload: JsonDict,
    arguments: JsonDict,
) -> None:
    server_url = mcp_server_url_from_transport(transport) if transport is not None else ""
    if server_url:
        response_ref["mcp_server_url"] = server_url
    session_id = payload.get("session_id") or arguments.get("session_id")
    dashboard_url = mcp_dashboard_url(server_url, session_id)
    if dashboard_url:
        response_ref["dashboard_url"] = dashboard_url


class _ArtifactApi:
    def __init__(
        self,
        *,
        image_output_root: str,
        text_output_root: str,
        response_output_root: str,
        max_inline_text_chars: int,
        session_id: str = "",
    ) -> None:
        self.image_output_root = image_output_root
        self.text_output_root = text_output_root
        self.response_output_root = response_output_root
        self.max_inline_text_chars = max_inline_text_chars
        self.session_id = session_id
        self.outputs: list[JsonDict] = []

    def materialize_images(self, payload: JsonDict, *, bundle_id: str | None = None) -> JsonDict:
        existing_images = payload.get("image_artifacts")
        if isinstance(existing_images, list) and existing_images:
            images = [dict(image) for image in existing_images if isinstance(image, dict)]
            self.outputs.extend(images)
            return {
                "bundle_id": str(bundle_id or payload.get("session_id") or "already-materialized"),
                "artifact_root": str(
                    artifact_session_root(self.image_output_root, self.session_id).resolve()
                ),
                "payload": dict(payload),
                "images": images,
            }
        bundle = materialize_mcp_images(
            dict(payload),
            output_root=self.image_output_root,
            bundle_id=bundle_id,
            session_id=self.session_id,
        )
        bundle_dict = bundle.to_dict()
        self.outputs.extend(bundle_dict["images"])
        return bundle_dict

    def materialize_texts(self, payload: JsonDict, *, bundle_id: str | None = None) -> JsonDict:
        bundle = materialize_long_texts(
            dict(payload),
            output_root=self.text_output_root,
            bundle_id=bundle_id,
            max_inline_chars=self.max_inline_text_chars,
            session_id=self.session_id,
        )
        bundle_dict = bundle.to_dict()
        self.outputs.extend(bundle_dict["artifacts"])
        return bundle_dict

    def list_images(self, *, limit: int = 20) -> JsonDict:
        root = artifact_session_root(self.image_output_root, self.session_id).resolve()
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

    def _require_owned_path(self, path: str) -> Path:
        resolved = Path(path).expanduser().resolve()
        roots = (
            artifact_session_root(self.image_output_root, self.session_id).resolve(),
            artifact_session_root(self.text_output_root, self.session_id).resolve(),
            artifact_session_root(self.response_output_root, self.session_id).resolve(),
        )
        if not any(_is_relative_to(resolved, root) for root in roots):
            raise PermissionError("Artifact path is outside the current Agent session.")
        return resolved


def _safe_globals(*, workspace_root: str | None = None) -> JsonDict:
    workspace = Path(workspace_root).resolve() if workspace_root else None

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
            return SimpleNamespace(Path=lambda value=".": _WorkspacePath(value, workspace))
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
        "Path": (lambda value=".": _WorkspacePath(value, workspace)) if workspace else Path,
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
) -> Any:
    writes = any(flag in mode for flag in ("w", "a", "x", "+"))
    if writes and workspace_root is None:
        raise PermissionError("python_exec sandbox open() is read-only.")
    path = _resolve_workspace_path(file, workspace_root)
    allowed_roots = (workspace_root,) if workspace_root is not None else _safe_open_roots()
    if not any(_is_relative_to(path, root) for root in allowed_roots):
        roots = ", ".join(str(root) for root in allowed_roots)
        raise PermissionError(f"python_exec sandbox can only read files under: {roots}")
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
    """Small pathlib-compatible facade confined to one session sandbox root."""

    def __init__(self, value: str | Path, root: Path) -> None:
        self._root = root.resolve()
        self._path = _resolve_workspace_path(value, self._root)
        self._assert_owned()

    def _assert_owned(self) -> None:
        if not _is_relative_to(self._path, self._root):
            raise PermissionError("Path is outside the current session sandbox workspace.")

    def __fspath__(self) -> str:
        return str(self._path)

    def __str__(self) -> str:
        return str(self._path)

    def __truediv__(self, value: object) -> "_WorkspacePath":
        return _WorkspacePath(self._path / str(value), self._root)

    @property
    def name(self) -> str:
        return self._path.name

    @property
    def suffix(self) -> str:
        return self._path.suffix

    @property
    def parent(self) -> "_WorkspacePath":
        return _WorkspacePath(self._path.parent, self._root)

    def exists(self) -> bool:
        return self._path.exists()

    def is_file(self) -> bool:
        return self._path.is_file()

    def is_dir(self) -> bool:
        return self._path.is_dir()

    def mkdir(self, *, parents: bool = False, exist_ok: bool = False) -> None:
        self._path.mkdir(parents=parents, exist_ok=exist_ok)

    def read_text(self, *, encoding: str = "utf-8") -> str:
        return self._path.read_text(encoding=encoding)

    def write_text(self, data: str, *, encoding: str = "utf-8") -> int:
        self._assert_owned()
        return self._path.write_text(data, encoding=encoding)


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


def _safe_artifact_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return token.strip("_") or "tool"
