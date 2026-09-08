"""Trusted bootstrap for python_exec; communicate only through bounded JSON.

Started with isolated Python (-I), closed inherited descriptors, and a minimal
environment. No agent code is compiled/executed until kernel policy is active.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace


def main() -> None:
    result_fd = int(sys.argv[1])
    # The repository is used only to bootstrap trusted host modules. It is not
    # granted as a readable root to generated code.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from agent.tools.coding import (
        _ArtifactApi, _compile_agent_code, _json_safe, _safe_globals,
        _python_exec_exception_diagnostic, _workspace_descriptor,
    )
    from agent.tools.python_sandbox_policy import install_policy, runtime_read_roots

    payload = json.loads(sys.stdin.read())
    options = payload["options"]
    policy = None
    try:
        policy = install_policy(
            read_roots=[*runtime_read_roots(), *payload["read_roots"]],
            write_root=options.get("workspace_root"),
            cpu_seconds=max(1, math.ceil(payload["timeout_s"])),
            memory_bytes=payload["memory_bytes"], file_bytes=payload["file_bytes"],
        )
        artifacts = _ArtifactApi(**options["artifact_api"])
        scope = _safe_globals(workspace_root=options.get("workspace_root"),
                              read_roots=tuple(payload["read_roots"]))
        scope.update(payload.get("extra_globals", {}))
        scope.update({
            "api": SimpleNamespace(artifacts=artifacts), "artifacts": artifacts,
            "observation": payload.get("observation"), "parameters": payload["parameters"],
            "workspace": _workspace_descriptor(session_root=options.get("session_root"),
                                                sandbox_root=options.get("workspace_root")),
        })
        exec(_compile_agent_code(payload["code"]), scope, scope)
        response = {"success": True, "result": _json_safe(scope.get("result")), "policy": policy}
    except BaseException as exc:
        diagnostic = (_python_exec_exception_diagnostic(exc) if policy is not None else {
            "code": "python_sandbox_unavailable", "error_type": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(limit=5),
        })
        response = {"success": False, "diagnostic": diagnostic, "policy": policy}
    # Agent output can be arbitrary but never carries authority or raw handles.
    encoded = json.dumps(response, ensure_ascii=True).encode()
    while encoded:
        written = os.write(result_fd, encoded)
        encoded = encoded[written:]
    os.close(result_fd)


if __name__ == "__main__":
    main()
