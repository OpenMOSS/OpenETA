from __future__ import annotations

import json
import os
from pathlib import Path

import agent.tools.coding as coding_module
from agent.tools.coding import PythonExecConfig, PythonExecRuntime
from agent.tools.registry import ToolExecutionContext, build_default_tool_registry


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


def test_python_exec_exposes_set_builtin() -> None:
    runtime = PythonExecRuntime()

    result = runtime.handler(
        _context(
            "values = set([3, 1, 3, 2])\n"
            "values.add(4)\n"
            "result = {\n"
            "    'unique': sorted(values),\n"
            "    'intersection': sorted(values & set([2, 4, 5])),\n"
            "}\n"
        )
    )

    assert result.success is True
    assert result.details["outputs"]["result"] == {
        "unique": [1, 2, 3, 4],
        "intersection": [2, 4],
    }


def test_python_exec_recovers_over_escaped_newlines() -> None:
    runtime = PythonExecRuntime()

    code = 'a = 1\\nresult = {"value": a + 2}\\'
    result = runtime.handler(_context(code))

    assert result.success is True
    assert result.details["outputs"]["result"] == {"value": 3}


def test_python_exec_preserves_intended_string_escapes() -> None:
    runtime = PythonExecRuntime()

    result = runtime.handler(_context('result = {"text": "l1\\nl2"}'))

    assert result.success is True
    assert result.details["outputs"]["result"] == {"text": "l1\nl2"}


def test_python_exec_materializes_large_structured_result_with_clear_path(
    tmp_path: Path,
) -> None:
    runtime = PythonExecRuntime(
        PythonExecConfig(
            structured_output_root=str(tmp_path / "artifacts"),
            max_inline_structured_chars=100,
        )
    )

    result = runtime.handler(
        _context(
            "result = {'placements': "
            "[{'id': f'p{i}', 'matrix': list(range(16))} for i in range(5)]}"
        )
    )

    outputs = result.details["outputs"]
    artifact = outputs["result_artifact"]
    assert result.success is True
    assert outputs["result_inline_complete"] is False
    assert outputs["result"]["collection_sizes"] == {"placements": 5}
    assert outputs["result"]["complete_result_path"] == artifact["path"]
    assert artifact in result.details["artifacts"]
    assert artifact["path"] in result.content
    persisted = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))
    assert persisted["outputs"]["result"]["placements"][4]["id"] == "p4"


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


def test_python_exec_has_no_simulator_mcp_helper() -> None:
    runtime = PythonExecRuntime()

    result = runtime.handler(_context("result = mcp.list_tools()"))

    assert result.success is False
    assert result.details["diagnostics"][0]["error_type"] == "NameError"


def test_python_exec_reads_full_session_and_writes_only_sandbox(tmp_path: Path) -> None:
    session = tmp_path / "sessions" / "session-a"
    artifacts_root = session / "artifacts" / "structured" / "anygrasp" / "result-1"
    sandbox = session / "sandbox"
    artifacts_root.mkdir(parents=True)
    sandbox.mkdir(parents=True)
    payload_path = artifacts_root / "outputs.json"
    payload_path.write_text(
        '{"outputs":{"grasp_candidates":[{"id":"g0"},{"id":"g1"}]}}',
        encoding="utf-8",
    )
    runtime = PythonExecRuntime(
        PythonExecConfig(session_root=str(session), workspace_root=str(sandbox))
    )

    result = runtime.handler(
        _context(
            "from pathlib import Path\n"
            "payload = artifacts.read_json(parameters['path'])\n"
            "visible = artifacts.list_files(pattern='*.json')\n"
            "Path('ranked.json').write_text('selected=' + payload['outputs']['grasp_candidates'][1]['id'])\n"
            "result = {\n"
            "  'selected': payload['outputs']['grasp_candidates'][1]['id'],\n"
            "  'visible': [row['path'] for row in visible['files']],\n"
            "  'derived': Path('ranked.json').read_text(),\n"
            "  'workspace': workspace,\n"
            "}"
            ,
            extra_parameters={"path": str(payload_path)},
        )
    )

    output = result.details["outputs"]["result"]
    assert result.success is True
    assert output["selected"] == "g1"
    assert str(payload_path) in output["visible"]
    assert output["derived"] == "selected=g1"
    assert output["workspace"]["simulator_mcp_available"] is False
    assert (sandbox / "ranked.json").read_text(encoding="utf-8") == "selected=g1"
    assert any(
        artifact.get("kind") == "derived_artifact"
        for artifact in result.details["artifacts"]
    )

    forbidden_write = runtime.handler(
        _context(
            "from pathlib import Path\n"
            "result = Path(parameters['path']).write_text('overwrite')",
            extra_parameters={"path": str(payload_path)},
        )
    )
    assert forbidden_write.success is False
    assert forbidden_write.details["diagnostics"][0]["error_type"] == "PermissionError"


def test_python_exec_cannot_read_another_session(tmp_path: Path) -> None:
    own_session = tmp_path / "sessions" / "session-a"
    other_session = tmp_path / "sessions" / "session-b"
    own_sandbox = own_session / "sandbox"
    own_sandbox.mkdir(parents=True)
    other_session.mkdir(parents=True)
    other_path = other_session / "secret.json"
    other_path.write_text('{"secret":true}', encoding="utf-8")
    runtime = PythonExecRuntime(
        PythonExecConfig(session_root=str(own_session), workspace_root=str(own_sandbox))
    )

    result = runtime.handler(
        _context(
            "result = artifacts.read_json(parameters['path'])",
            extra_parameters={"path": str(other_path)},
        )
    )

    assert result.success is False
    assert result.details["diagnostics"][0]["error_type"] == "PermissionError"

    escaped_glob = runtime.handler(
        _context("result = artifacts.list_files(pattern='../session-b/*')")
    )
    assert escaped_glob.success is False
    assert escaped_glob.details["diagnostics"][0]["error_type"] == "PermissionError"


def test_python_exec_is_planning_and_does_not_require_observation_refresh() -> None:
    runtime = PythonExecRuntime()

    result = runtime.handler(_context("result = 1"))

    assert result.success is True
    assert result.details["effect"] == "planning"
    assert result.details["requires_observation_after_call"] is False


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
