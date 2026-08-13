from __future__ import annotations

import json
from pathlib import Path

from adapter.protocol import EnvAction
from agent.runtime.memory import AgentMemory


def _candidate(index: int) -> dict:
    return {
        "id": f"grasp_{index:03d}",
        "rank": index,
        "score": 1.0 - index * 0.01,
        "translation_xyz": [0.1, 0.2, 0.3 + index * 0.001],
    }


def test_memory_persists_complete_candidates_and_keeps_bounded_preview(
    tmp_path: Path,
) -> None:
    candidates = [_candidate(index) for index in range(8)]
    memory = AgentMemory(artifact_root=tmp_path / "artifacts")
    memory.start_session(task="pick object", session_id="session-a")
    action = EnvAction(
        action_type="tool_call",
        command={
            "request_name": "anygrasp",
            "tool_calls": [
                {
                    "name": "anygrasp",
                    "status": "executed",
                    "result": {
                        "success": True,
                        "content": "complete",
                        "details": {
                            "tool": "anygrasp",
                            "result_id": "run-001",
                            "candidate_count": len(candidates),
                            "grasp_candidates": candidates,
                            "artifacts": [],
                        },
                    },
                }
            ],
        },
    )

    memory.add_action(action)

    details = action.command["tool_calls"][0]["result"]["details"]
    reference = details["structured_artifact"]
    artifact_path = Path(reference["path"])
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact_path.is_file()
    assert artifact_path.stat().st_mode & 0o777 == 0o400
    assert payload["outputs"]["grasp_candidates"] == candidates
    assert reference in details["artifacts"]

    preview = memory.artifacts["anygrasp_grasp_candidates_latest"]["value"]
    assert preview["candidate_count"] == 8
    assert preview["preview_count"] == 5
    assert preview["truncated"] is True
    assert len(preview["grasp_candidates"]) == 5
    assert preview["complete_outputs_artifact"]["path"] == str(artifact_path)
    assert "python_exec" in preview["query_hint"]


def test_memory_does_not_duplicate_an_existing_structured_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "existing.json"
    artifact_path.write_text("{}", encoding="utf-8")
    existing = {"type": "json", "kind": "structured_tool_output", "path": str(artifact_path)}
    memory = AgentMemory(artifact_root=tmp_path / "artifacts")
    memory.start_session(task="place object", session_id="session-a")
    action = EnvAction(
        action_type="tool_call",
        command={
            "tool_calls": [
                {
                    "name": "anyplace",
                    "result": {
                        "success": True,
                        "details": {
                            "placement_candidates": [{"id": "placement_000"}],
                            "structured_artifact": existing,
                            "artifacts": [existing],
                        },
                    },
                }
            ]
        },
    )

    memory.add_action(action)

    assert action.command["tool_calls"][0]["result"]["details"]["artifacts"] == [existing]
    assert not (tmp_path / "artifacts").exists()
