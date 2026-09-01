"""Create a no-action rollout manifest for ToolContract authority auditing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapter.protocol import JsonDict
from agent.backends.planner import StaticPlannerBackend
from agent.runtime.memory import AgentMemory
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime


SCHEMA_VERSION = "openeta.tool_contract_authority_baseline.v1"


def build_tool_contract_authority_baseline(
    root: str | Path,
    *,
    session_id: str,
) -> JsonDict:
    """Persist current catalog/policy provenance without model or tool execution."""

    store_root = Path(root)
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "response",
                "name": "talk",
                "parameters": {"message": "unused authority baseline payload"},
            }
        )
    )
    runtime = OpenEtaAgentRuntime(
        planner=planner,
        memory=AgentMemory(store=JsonMemoryStore(store_root)),
    )
    runtime.start_session(
        task="ToolContract no-action authority baseline",
        session_id=session_id,
    )
    manifest = store_root / "sessions" / session_id / "rollout" / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    provenance = payload.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    contract_runtime = provenance.get("tool_contract_runtime")
    contract_runtime = contract_runtime if isinstance(contract_runtime, dict) else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "manifest": str(manifest),
        "catalog_sha256": contract_runtime.get("catalog_sha256"),
        "policy": contract_runtime.get("policy"),
        "planner_pipeline_alignment": contract_runtime.get(
            "planner_pipeline_alignment"
        ),
        "model_call_count": 0,
        "tool_call_count": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    report = build_tool_contract_authority_baseline(
        args.root,
        session_id=args.session_id,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    alignment = report.get("planner_pipeline_alignment")
    return 0 if isinstance(alignment, dict) and alignment.get("catalog_match") else 1


if __name__ == "__main__":
    raise SystemExit(main())
