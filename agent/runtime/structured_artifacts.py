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


def load_structured_tool_output(
    reference: JsonDict,
    *,
    expected_tool: str = "",
    expected_result_id: str = "",
    allowed_root: str | Path | None = None,
) -> JsonDict:
    """Load and verify one host-materialized complete structured output.

    Working-memory projections are intentionally bounded, but IDs exposed by a
    tool may refer to entries outside that projection.  Host-side resolvers use
    this function to recover the complete immutable evidence without trusting a
    planner-supplied path or an unchecked JSON file.
    """

    if not isinstance(reference, dict):
        raise ValueError("structured tool artifact reference must be an object")
    if reference.get("schema_version") != STRUCTURED_TOOL_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported structured tool artifact schema")
    if reference.get("kind") != "structured_tool_output" or reference.get(
        "immutable"
    ) is not True:
        raise ValueError("structured tool artifact must be immutable host evidence")
    path_value = reference.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("structured tool artifact has no path")
    path = Path(path_value).expanduser().resolve(strict=True)
    if allowed_root is not None:
        root = Path(allowed_root).expanduser().resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("structured tool artifact is outside the session artifact root")
    raw = path.read_bytes()
    expected_sha256 = str(reference.get("sha256") or "")
    if not expected_sha256 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("structured tool artifact integrity check failed")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("structured tool artifact payload must be an object")
    if payload.get("schema_version") != STRUCTURED_TOOL_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("structured tool artifact envelope schema mismatch")
    reference_tool = str(reference.get("tool") or "")
    payload_tool = str(payload.get("tool") or "")
    if reference_tool != payload_tool or (expected_tool and payload_tool != expected_tool):
        raise ValueError("structured tool artifact tool identity mismatch")
    reference_result_id = str(reference.get("result_id") or "")
    payload_result_id = str(payload.get("result_id") or "")
    if reference_result_id != payload_result_id or (
        expected_result_id and payload_result_id != expected_result_id
    ):
        raise ValueError("structured tool artifact result identity mismatch")
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("structured tool artifact outputs must be an object")
    return outputs
