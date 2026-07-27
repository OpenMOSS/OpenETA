from __future__ import annotations

import base64
import os
from pathlib import Path

from adapter.protocol import JsonDict
import agent.tools.coding as coding_module
from agent.tools.coding import PythonExecConfig, PythonExecRuntime
from agent.tools.registry import ToolExecutionContext, build_default_tool_registry


PNG_1X1 = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c6360000002000100ffff03000006000557bfab0d000000"
        "0049454e44ae426082"
    )
).decode("ascii")


class FakeMcpTransport:
    def __init__(
        self,
        response: JsonDict | list[JsonDict],
        *,
        tools: JsonDict | None = None,
        url: str = "",
    ) -> None:
        self.responses = list(response) if isinstance(response, list) else [response]
        self.tools = tools or {"tools": [], "tool_count": 0}
        self.url = url
        self.calls: list[JsonDict] = []

    def call_tool(
        self,
        name: str,
        arguments: JsonDict,
        *,
        timeout_s: float | None = None,
    ) -> JsonDict:
        self.calls.append({"name": name, "arguments": dict(arguments), "timeout_s": timeout_s})
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]

    def list_tools(self, *, timeout_s: float | None = None) -> JsonDict:
        self.calls.append({"name": "list_tools", "arguments": {}, "timeout_s": timeout_s})
        return self.tools


def _context(
    code: str,
    *,
    sandbox: str = "sandbox",
    extra_parameters: JsonDict | None = None,
    session_id: str = "",
) -> ToolExecutionContext:
    spec = build_default_tool_registry().get("python_exec")
    parameters = {"code": code, "sandbox": sandbox, **dict(extra_parameters or {})}
    return ToolExecutionContext(
        name="python_exec",
        spec=spec,
        parameters=parameters,
        metadata={"session_id": session_id} if session_id else {},
    )


def test_python_exec_runs_restricted_code_and_returns_result() -> None:
    runtime = PythonExecRuntime()

    result = runtime.handler(_context("print('hello')\nresult = {'value': sum([1, 2, 3])}"))

    assert result.success is True
    assert result.details["outputs"]["result"] == {"value": 6}
    assert result.details["outputs"]["stdout"] == "hello\n"
    assert result.details["parameters"]["code"] == "<code omitted>"


def test_python_exec_allows_safe_imports_and_readonly_artifact_open(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "response.json"
    artifact_path.write_text('{"value": 7}', encoding="utf-8")
    runtime = PythonExecRuntime()

    result = runtime.handler(
        _context(
            "import json\n"
            "import math\n"
            "from pathlib import Path\n"
            "with open(parameters['path'], 'r', encoding='utf-8') as f:\n"
            "    data = json.load(f)\n"
            "result = {\n"
            "    'value': data['value'],\n"
            "    'sqrt': math.sqrt(9),\n"
            "    'name': Path(parameters['path']).name,\n"
            "}\n",
            extra_parameters={"path": str(artifact_path)},
        )
    )

    assert result.success is True
    assert result.details["outputs"]["result"] == {
        "value": 7,
        "sqrt": 3.0,
        "name": "response.json",
    }


def test_python_exec_blocks_unapproved_imports() -> None:
    runtime = PythonExecRuntime()

    result = runtime.handler(_context("import os\nresult = {'ok': True}"))

    assert result.success is False
    assert result.details["diagnostics"][0]["code"] == "python_exec_import_error"
    assert result.details["diagnostics"][0]["error_type"] == "ImportError"
    assert "not available" in result.details["diagnostics"][0]["message"]
    assert "outside the python_exec sandbox import allowlist" in (
        result.details["diagnostics"][0]["remediation"]
    )


def test_python_exec_reports_allowed_but_missing_import(monkeypatch) -> None:
    real_import_module = coding_module.importlib.import_module

    def fake_import_module(name: str):
        if name == "numpy":
            raise ImportError("No module named numpy")
        return real_import_module(name)

    monkeypatch.setattr(coding_module.importlib, "import_module", fake_import_module)
    runtime = PythonExecRuntime()

    result = runtime.handler(_context("import numpy\nresult = {'ok': True}"))

    assert result.success is False
    diagnostic = result.details["diagnostics"][0]
    assert diagnostic["code"] == "python_exec_import_error"
    assert "allowed by python_exec sandbox but is not installed" in diagnostic["message"]
    assert "missing from the configured OpenETA runtime" in diagnostic["remediation"]


def test_python_exec_open_is_readonly(tmp_path: Path) -> None:
    artifact_path = tmp_path / "response.json"
    runtime = PythonExecRuntime()

    result = runtime.handler(
        _context(
            "with open(parameters['path'], 'w', encoding='utf-8') as f:\n"
            "    f.write('bad')\n"
            "result = {'ok': True}\n",
            extra_parameters={"path": str(artifact_path)},
        )
    )

    assert result.success is False
    assert result.details["diagnostics"][0]["error_type"] == "PermissionError"


def test_python_exec_workspace_allows_owned_writes_and_blocks_escape(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "session" / "sandbox"
    workspace.mkdir(parents=True)
    runtime = PythonExecRuntime(PythonExecConfig(workspace_root=str(workspace)))

    written = runtime.handler(
        _context(
            "from pathlib import Path\n"
            "Path('notes').mkdir(exist_ok=True)\n"
            "Path('notes/result.txt').write_text('ok', encoding='utf-8')\n"
            "result = Path('notes/result.txt').read_text(encoding='utf-8')"
        )
    )
    escaped = runtime.handler(
        _context("from pathlib import Path\nresult = Path('/tmp/outside.txt').write_text('bad')")
    )

    assert written.success is True
    assert written.details["outputs"]["result"] == "ok"
    assert (workspace / "notes" / "result.txt").read_text(encoding="utf-8") == "ok"
    assert escaped.success is False
    assert escaped.details["diagnostics"][0]["error_type"] == "PermissionError"


def test_python_exec_exposes_mcp_helper() -> None:
    transport = FakeMcpTransport({"ok": True, "envs": [{"id": "openeta/demo-v0"}]})
    runtime = PythonExecRuntime(PythonExecConfig(mcp_transport=transport))

    result = runtime.handler(
        _context(
            "envs = mcp.call_tool('search_envs', {'query': 'libero panda'})\n"
            "full_response = artifacts.read_json(envs['response_path'])\n"
            "result = {\n"
            "    'envs_count': envs['envs_count'],\n"
            "    'has_inline_envs': 'envs' in envs,\n"
            "    'ids': [item['id'] for item in full_response['envs']],\n"
            "}"
        )
    )

    assert result.success is True
    assert result.details["outputs"]["result"] == {
        "envs_count": 1,
        "has_inline_envs": False,
        "ids": ["openeta/demo-v0"],
    }
    assert transport.calls == [
        {
            "name": "search_envs",
            "arguments": {"query": "libero panda"},
            "timeout_s": 120.0,
        }
    ]
    assert result.details["outputs"]["mcp_calls"][0]["tool"] == "search_envs"


def test_python_exec_marks_failed_mcp_call_as_failed() -> None:
    transport = FakeMcpTransport({"success": False, "content": "render_env failed"})
    runtime = PythonExecRuntime(PythonExecConfig(mcp_transport=transport))

    result = runtime.handler(
        _context(
            "rendered = mcp.call_tool('render_env', {'handle': 'env-1'})\n"
            "result = {'ok': True, 'response': rendered}"
        )
    )

    assert result.success is False
    assert result.content == "python_exec completed with failed MCP call(s)"
    assert result.details["diagnostics"][0]["code"] == "python_exec_mcp_call_failed"
    assert result.details["diagnostics"][0]["failed_tools"] == ["render_env"]
    assert result.details["outputs"]["mcp_calls"][0]["success"] is False


def test_python_exec_rejects_simulator_environment_creation() -> None:
    transport = FakeMcpTransport({"success": True})
    runtime = PythonExecRuntime(PythonExecConfig(mcp_transport=transport))

    result = runtime.handler(
        _context("result = mcp.call_tool('create_env', {'env_id': 'openeta/demo-v0'})")
    )

    assert result.success is False
    assert "create_simulator_env agent tool" in result.content
    assert transport.calls == []


def test_python_exec_exposes_mcp_list_tools(tmp_path: Path) -> None:
    transport = FakeMcpTransport(
        {"ok": True},
        tools={
            "tools": [
                {
                    "name": "create_env",
                    "description": "Create an environment.",
                    "input_schema": {
                        "type": "object",
                        "required": ["env_id"],
                        "properties": {"env_id": {"type": "string"}},
                    },
                }
            ],
            "tool_count": 1,
        },
    )
    runtime = PythonExecRuntime(
        PythonExecConfig(
            mcp_transport=transport,
            response_output_root=str(tmp_path),
        )
    )

    result = runtime.handler(
        _context(
            "catalog = mcp.list_tools()\n"
            "full = artifacts.read_json(catalog['response_path'])\n"
            "result = {'tool_count': catalog['tool_count'], 'first': full['tools'][0]['name']}"
        )
    )

    assert result.success is True
    assert result.details["outputs"]["result"] == {"tool_count": 1, "first": "create_env"}
    assert transport.calls[0] == {"name": "list_tools", "arguments": {}, "timeout_s": 120.0}
    assert result.details["outputs"]["mcp_calls"][0]["tool"] == "list_tools"


def test_python_exec_materializes_long_mcp_text_and_exposes_grep(tmp_path: Path) -> None:
    long_text = "alpha\n" + ("needle beta\n" * 300)
    transport = FakeMcpTransport({"ok": True, "content": long_text})
    runtime = PythonExecRuntime(
        PythonExecConfig(
            mcp_transport=transport,
            response_output_root=str(tmp_path),
            max_inline_text_chars=100,
        )
    )

    result = runtime.handler(
        _context(
            "response = mcp.call_tool('describe_env', {})\n"
            "matches = artifacts.grep_text(\n"
            "    response['response_path'], 'needle', max_matches=3\n"
            ")\n"
            "result = {'path': response['response_path'], 'matches': matches['matches']}"
        )
    )

    outputs = result.details["outputs"]
    path = outputs["result"]["path"]
    assert result.success is True
    assert Path(path).exists()
    assert "needle beta" in Path(path).read_text(encoding="utf-8")
    assert "needle beta" in outputs["result"]["matches"][0]["text"]
    assert "needle beta" * 20 not in str(outputs)
    assert outputs["mcp_calls"][0]["response_path"] == path
    assert outputs["mcp_calls"][0]["response_artifact"]["type"] == "json"


def test_python_exec_mcp_call_materializes_images_before_long_text(tmp_path: Path) -> None:
    transport = FakeMcpTransport(
        {
            "ok": True,
            "cameras": [
                {
                    "frame_id": "front",
                    "rgb_base64": PNG_1X1,
                    "content": "camera log\n" + ("needle\n" * 300),
                }
            ],
        }
    )
    runtime = PythonExecRuntime(
        PythonExecConfig(
            mcp_transport=transport,
            image_output_root=str(tmp_path / "images"),
            response_output_root=str(tmp_path / "responses"),
            max_inline_text_chars=100,
        )
    )

    result = runtime.handler(
        _context(
            "response = mcp.call_tool('reset_env', {})\n"
            "images = artifacts.materialize_images(response, bundle_id='again')\n"
            "camera = images['payload']['cameras'][0]\n"
            "full_response = artifacts.read_json(response['response_path'])\n"
            "result = {\n"
            "    'rgb_path': camera['rgb_path'],\n"
            "    'response_image_count': len(response.get('image_artifacts', [])),\n"
            "    'rematerialized_image_count': len(images['images']),\n"
            "    'response_path': response['response_path'],\n"
            "    'content': full_response['cameras'][0]['content'],\n"
            "}\n"
        )
    )

    camera = result.details["outputs"]["result"]
    assert result.success is True
    assert Path(camera["rgb_path"]).exists()
    assert camera["response_image_count"] == 1
    assert camera["rematerialized_image_count"] == 1
    assert Path(camera["response_path"]).exists()
    assert "needle" in camera["content"]
    assert result.details["outputs"]["mcp_calls"][0]["image_artifacts"]
    assert result.details["outputs"]["mcp_calls"][0]["response_artifact"]["type"] == "json"
    assert PNG_1X1 not in str(result.details["outputs"])
    assert "needle" * 20 not in str(result.details["outputs"])


def test_python_exec_mcp_call_exposes_anygrasp_intrinsics(tmp_path: Path) -> None:
    transport = FakeMcpTransport(
        {
            "ok": True,
            "cameras": [
                {
                    "frame_id": "agentview",
                    "rgb_base64": PNG_1X1,
                    "depth_base64": PNG_1X1,
                    "width": 512,
                    "height": 512,
                    "intrinsics": {
                        "fx": 618.0386719675123,
                        "fy": 618.0386719675123,
                        "cx": 256,
                        "cy": 256,
                    },
                }
            ],
        }
    )
    runtime = PythonExecRuntime(
        PythonExecConfig(
            mcp_transport=transport,
            image_output_root=str(tmp_path / "images"),
            response_output_root=str(tmp_path / "responses"),
        )
    )

    result = runtime.handler(
        _context(
            "response = mcp.call_tool('reset_env', {})\n"
            "full_response = artifacts.read_json(response['response_path'])\n"
            "result = {\n"
            "    'inline_intrinsics': response['cameras'][0]['anygrasp_intrinsics'],\n"
            "    'stored_intrinsics': full_response['cameras'][0]['anygrasp_intrinsics'],\n"
            "}\n"
        )
    )

    assert result.success is True
    assert result.details["outputs"]["result"]["inline_intrinsics"]["scale"] == 1000.0
    assert result.details["outputs"]["result"]["stored_intrinsics"]["scale"] == 1000.0


def test_python_exec_lists_materialized_image_paths_without_os_import(tmp_path: Path) -> None:
    transport = FakeMcpTransport(
        {
            "ok": True,
            "cameras": [
                {
                    "frame_id": "front",
                    "rgb_base64": PNG_1X1,
                    "width": 1,
                    "height": 1,
                }
            ],
        }
    )
    runtime = PythonExecRuntime(
        PythonExecConfig(
            mcp_transport=transport,
            image_output_root=str(tmp_path / "images"),
        )
    )

    result = runtime.handler(
        _context(
            "mcp.call_tool('render_env', {})\n"
            "result = artifacts.list_images(limit=5)\n"
        )
    )

    output = result.details["outputs"]["result"]
    assert result.success is True
    assert output["image_count"] == 1
    assert len(output["paths"]) == 1
    assert output["latest_image_path"] == output["paths"][0]
    assert output["images"][0]["kind"] == "rgb"
    assert Path(output["images"][0]["path"]).exists()
    assert PNG_1X1 not in str(result.details["outputs"])


def test_python_exec_mcp_artifacts_are_isolated_by_agent_session(tmp_path: Path) -> None:
    config = PythonExecConfig(
        mcp_transport=FakeMcpTransport(
            {"ok": True, "frame_id": "front", "rgb_base64": PNG_1X1}
        ),
        image_output_root=str(tmp_path / "images"),
        text_output_root=str(tmp_path / "text"),
        response_output_root=str(tmp_path / "responses"),
    )
    runtime = PythonExecRuntime(config)
    code = (
        "response = mcp.call_tool('render_env', {})\n"
        "result = {'response_path': response['response_path'], "
        "'image_path': response['image_artifacts'][0]['path'], "
        "'visible_images': artifacts.list_images()['paths']}"
    )

    first = runtime.handler(_context(code, session_id="session-a"))
    second = runtime.handler(_context(code, session_id="session-b"))
    first_again = runtime.handler(_context(code, session_id="session-a"))
    first_output = first.details["outputs"]["result"]
    second_output = second.details["outputs"]["result"]
    first_again_output = first_again.details["outputs"]["result"]

    assert first_output["response_path"] != second_output["response_path"]
    assert first_output["image_path"] != second_output["image_path"]
    assert first_output["image_path"] != first_again_output["image_path"]
    assert first_output["response_path"] != first_again_output["response_path"]
    assert "/session-a/" in first_output["response_path"]
    assert "/session-a/" in first_output["image_path"]
    assert "/session-b/" in second_output["response_path"]
    assert "/session-b/" in second_output["image_path"]
    assert first_output["visible_images"] == [first_output["image_path"]]
    assert second_output["visible_images"] == [second_output["image_path"]]
    assert set(first_again_output["visible_images"]) == {
        first_output["image_path"],
        first_again_output["image_path"],
    }
    cross_session_read = runtime.handler(
        _context(
            "result = artifacts.read_json(parameters['path'])",
            session_id="session-b",
            extra_parameters={"path": first_output["response_path"]},
        )
    )
    assert cross_session_read.success is False
    assert cross_session_read.details["diagnostics"][0]["error_type"] == (
        "PermissionError"
    )


def test_python_exec_blocks_outside_sandbox_without_approval() -> None:
    runtime = PythonExecRuntime(PythonExecConfig(allow_outside_sandbox=True))

    result = runtime.handler(_context("result = 1", sandbox="outside_sandbox"))

    assert result.success is False
    assert result.details["diagnostics"][0]["code"] == "outside_sandbox_requires_approval"


def test_python_exec_runs_outside_sandbox_after_approval() -> None:
    runtime = PythonExecRuntime(
        PythonExecConfig(
            allow_outside_sandbox=True,
            approve_outside_sandbox=lambda _context, _mode: True,
        )
    )

    result = runtime.handler(
        _context(
            "import asyncio, os, time\n"
            "time.sleep(0.01)\n"
            "result = {\n"
            "    'approved': True,\n"
            "    'asyncio_available': hasattr(asyncio, 'run'),\n"
            "    'separate_process': os.getpid() != parameters['parent_pid'],\n"
            "}",
            sandbox="outside_sandbox",
            extra_parameters={"parent_pid": os.getpid()},
        )
    )

    assert result.success is True
    assert result.details["outputs"]["result"] == {
        "approved": True,
        "asyncio_available": True,
        "separate_process": True,
    }
    assert result.details["outputs"]["executor"] == "host_subprocess"


def test_python_exec_outside_sandbox_enforces_timeout() -> None:
    runtime = PythonExecRuntime(
        PythonExecConfig(
            allow_outside_sandbox=True,
            approve_outside_sandbox=lambda _context, _mode: True,
        )
    )

    result = runtime.handler(
        _context(
            "import time\ntime.sleep(2)\nresult = True",
            sandbox="outside_sandbox",
            extra_parameters={"timeout_s": 0.05},
        )
    )

    assert result.success is False
    assert result.details["diagnostics"][0]["code"] == "outside_sandbox_timeout"
