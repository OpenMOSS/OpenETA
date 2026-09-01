from __future__ import annotations

import json
from pathlib import Path

from adapter.protocol import EnvAction
from agent.runtime.memory import AgentMemory, GRASP_PROVENANCE_KEY
from agent.runtime.structured_artifacts import load_structured_tool_output


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

    loaded = load_structured_tool_output(
        reference,
        expected_tool="anygrasp",
        expected_result_id="run-001",
        allowed_root=tmp_path / "artifacts",
    )
    assert loaded["grasp_candidates"] == candidates


def test_compile_can_bind_candidate_outside_bounded_preview(tmp_path: Path) -> None:
    candidates = [_candidate(index) for index in range(8)]
    source = {
        "mode": "targeted",
        "rgb": str(tmp_path / "rgb.png"),
        "depth": str(tmp_path / "depth.png"),
        "object_mask": str(tmp_path / "mask.png"),
        "source_tool": "anygrasp",
        "source_backend": "anygrasp",
    }
    memory = AgentMemory(artifact_root=tmp_path / "artifacts")
    memory.start_session(task="pick object", session_id="session-a")
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {"name": "anygrasp", "parameters": {}},
                "tool_calls": [
                    {
                        "name": "anygrasp",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "result_id": "run-001",
                                "scene_epoch": 0,
                                "grasp_candidates": candidates,
                                "source": source,
                                "artifacts": [],
                            },
                        },
                    }
                ],
            },
        )
    )

    selected = candidates[7]
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "name": "compile_grasp_seed",
                    "parameters": {"camera_pose": selected},
                },
                "tool_calls": [
                    {
                        "name": "compile_grasp_seed",
                        "status": "executed",
                        "parameters": {"camera_pose": selected},
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "schema_version": "openeta.compiled_grasp_seed.v1",
                                    "compiled_grasp_id": "compiled-007",
                                    "candidate_id": selected["id"],
                                }
                            },
                        },
                    }
                ],
            },
        )
    )

    provenance = memory.facts[GRASP_PROVENANCE_KEY]["value"]
    assert provenance["candidate_id"] == "grasp_007"
    assert provenance["compiled_grasp_id"] == "compiled-007"


def test_complete_candidate_artifact_rejects_tampering(tmp_path: Path) -> None:
    candidates = [_candidate(index) for index in range(6)]
    memory = AgentMemory(artifact_root=tmp_path / "artifacts")
    memory.start_session(task="pick object", session_id="session-a")
    action = EnvAction(
        action_type="tool_call",
        command={
            "tool_calls": [
                {
                    "name": "anygrasp",
                    "status": "executed",
                    "result": {
                        "success": True,
                        "details": {
                            "result_id": "run-001",
                            "grasp_candidates": candidates,
                            "artifacts": [],
                        },
                    },
                }
            ]
        },
    )
    memory.add_action(action)
    reference = action.command["tool_calls"][0]["result"]["details"][
        "structured_artifact"
    ]
    path = Path(reference["path"])
    path.chmod(0o600)
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    try:
        load_structured_tool_output(reference, allowed_root=tmp_path / "artifacts")
    except ValueError as exc:
        assert "integrity check failed" in str(exc)
    else:
        raise AssertionError("tampered structured evidence must not be accepted")


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


def test_memory_indexes_grasp_selection_evidence_from_tool_result_outputs(
    tmp_path: Path,
) -> None:
    memory = AgentMemory(artifact_root=tmp_path / "artifacts")
    memory.start_session(task="pick object", session_id="session-a")
    advice = {
        "schema_version": "openeta.grasp_selection_advice.v1",
        "recommended_candidate_id": "grasp_001",
        "confidence": 0.8,
    }
    bundle = {
        "schema_version": "openeta.grasp_selection_bundle.v1",
        "bundle_id": "grasp-selection:example",
        "bundle_ref": str(tmp_path / "selection_bundle.json"),
    }
    action = EnvAction(
        action_type="tool_call",
        command={
            "tool_calls": [
                {
                    "name": "grasp_pose_estimate",
                    "result": {
                        "success": True,
                        "details": {
                            "outputs": {
                                "result_id": "run-001",
                                "grasp_candidates": [_candidate(0), _candidate(1)],
                                "grasp_selection_advice": advice,
                                "grasp_selection_bundle": bundle,
                            },
                            "artifacts": [],
                        },
                    },
                }
            ]
        },
    )

    memory.add_action(action)

    preview = memory.artifacts["grasp_pose_estimate_grasp_candidates_latest"]["value"]
    assert preview["grasp_selection_advice"] == advice
    assert preview["grasp_selection_bundle"] == bundle
