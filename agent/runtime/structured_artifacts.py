"""Immutable session-local JSON artifacts for complete structured tool outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from adapter.protocol import JsonDict
from agent.runtime.artifact_paths import safe_artifact_component


STRUCTURED_TOOL_ARTIFACT_SCHEMA_VERSION = "openeta.structured_tool_artifact.v1"


def materialize_structured_tool_output(
    *,
    output_root: str | Path,
    tool: str,
    outputs: JsonDict,
    result_id: str = "",
) -> JsonDict:
    """Persist one complete tool output and return a compact immutable reference."""

    if not isinstance(outputs, dict):
        raise TypeError("structured tool outputs must be a JSON object")
    safe_tool = safe_artifact_component(tool, fallback="tool")
    identity = safe_artifact_component(result_id, fallback="result")
    bundle_id = f"{identity}-{uuid4().hex[:10]}"
    root = Path(output_root).resolve() / "structured" / safe_tool / bundle_id
    root.mkdir(parents=True, exist_ok=False)
    path = root / "outputs.json"
    envelope: JsonDict = {
        "schema_version": STRUCTURED_TOOL_ARTIFACT_SCHEMA_VERSION,
        "tool": str(tool),
        "result_id": str(result_id),
        "outputs": outputs,
    }
    text = json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    path.chmod(0o400)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "type": "json",
        "kind": "structured_tool_output",
        "index": result_id or bundle_id,
        "path": str(path),
        "byte_size": len(text.encode("utf-8")),
        "chars": len(text),
        "sha256": digest,
        "schema_version": STRUCTURED_TOOL_ARTIFACT_SCHEMA_VERSION,
        "tool": str(tool),
        "result_id": str(result_id),
        "immutable": True,
        "grep_hint": f"grep -n '<pattern>' {path}",
    }
