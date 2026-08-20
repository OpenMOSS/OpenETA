from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.evals.toolchain_coverage import (
    SPEC_SCHEMA_VERSION,
    extract_toolchain_coverage,
    load_toolchain_spec,
)


def _write_calls(path: Path, calls: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for seq, call in enumerate(calls, start=1):
        details = {
            "operational_success": call.get("operational_success", True),
            "semantic_outcome": call.get("semantic_outcome", "completed"),
            "outputs": call.get("outputs", {}),
        }
        rows.append(
            {
                "seq": seq,
                "event": {
                    "phase": "end",
                    "name": call["name"],
                    "effect": call.get("effect", "read_only"),
                    "success": call.get("success", True),
                    "parameters": call.get("parameters", {}),
                    "details": details,
                },
            }
        )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_toolchain_coverage_scans_run_and_reports_completion_frontier(
    tmp_path: Path,
) -> None:
    complete = tmp_path / "sessions" / "session-complete" / "rollout" / "tool_calls.jsonl"
    partial = tmp_path / "sessions" / "session-partial" / "rollout" / "tool_calls.jsonl"
    viewpoint_pose = {
        "target_pose": {"waypoint_role": "wrist_observation_viewpoint"}
    }
    _write_calls(
        complete,
        [
            {"name": "propose_wrist_viewpoints"},
            {"name": "ik_preview_check", "parameters": viewpoint_pose},
            {
                "name": "move_to",
                "parameters": viewpoint_pose,
                "outputs": {
                    "post_motion_evidence_handoff": {
                        "status": "fresh_wrist_packet_expected"
                    }
                },
            },
            {
                "name": "sam3",
                "parameters": {"camera_frame_id": "wrist"},
                "semantic_outcome": "detections_available",
            },
            {"name": "select_sam3_detection"},
            {"name": "compute_wrist_alignment"},
        ],
    )
    _write_calls(
        partial,
        [
            {"name": "propose_wrist_viewpoints"},
            {"name": "ik_preview_check", "parameters": viewpoint_pose},
            {"name": "move_to", "parameters": viewpoint_pose},
            {
                "name": "sam3",
                "parameters": {"camera_frame_id": "agentview"},
            },
        ],
    )
    sequence = {
        "id": "wrist_refinement",
        "milestones": [
            "propose_wrist_viewpoints",
            {
                "tool": "ik_preview_check",
                "where": {
                    "parameters.target_pose.waypoint_role": "wrist_observation_viewpoint"
                },
            },
            {
                "tool": "move_to",
                "where": {
                    "outputs.post_motion_evidence_handoff.status": (
                        "fresh_wrist_packet_expected"
                    )
                },
            },
            {
                "tool": "sam3",
                "where": {"parameters.camera_frame_id": "wrist"},
                "any_of": [
                    {"semantic_outcome": "detections_available"},
                    {"semantic_outcome": "no_detection"},
                ],
            },
            "select_sam3_detection",
            {
                "id": "consume_wrist_geometry",
                "tools": ["compute_wrist_alignment", "grasp_pose_estimate"],
            },
        ],
    }

    report = extract_toolchain_coverage(tmp_path, [sequence])

    assert report["rollout_count"] == 2
    summary = report["sequence_summaries"][0]
    assert summary["complete_rollout_count"] == 1
    assert summary["completion_rate"] == 0.5
    assert summary["frontier_counts"] == {"move_to": 1}
    complete_report = next(
        rollout for rollout in report["rollouts"] if rollout["session_id"] == "session-complete"
    )["sequences"][0]
    assert complete_report["complete"] is True
    assert complete_report["matched_calls"][-1]["milestone_id"] == "consume_wrist_geometry"


def test_toolchain_coverage_respects_max_gap_and_chooses_later_complete_start(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tool_calls.jsonl"
    _write_calls(
        source,
        [
            {"name": "sam3"},
            {"name": "unrelated_a"},
            {"name": "unrelated_b"},
            {"name": "select_sam3_detection"},
            {"name": "sam3"},
            {"name": "unrelated_c"},
            {"name": "select_sam3_detection"},
        ],
    )

    report = extract_toolchain_coverage(
        source,
        [
            {
                "id": "bounded_selection",
                "max_gap": 1,
                "milestones": ["sam3", "select_sam3_detection"],
            }
        ],
    )

    sequence = report["rollouts"][0]["sequences"][0]
    assert sequence["complete"] is True
    assert [call["seq"] for call in sequence["matched_calls"]] == [5, 7]


def test_toolchain_spec_loads_operators_and_rejects_unknown_schema(
    tmp_path: Path,
) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": SPEC_SCHEMA_VERSION,
                "sequences": [
                    {
                        "id": "successful_motion",
                        "milestones": [
                            {
                                "tool": "move_to",
                                "where": {
                                    "success": True,
                                    "outputs.motion_summary.reached_target": {
                                        "$in": [True]
                                    },
                                },
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    loaded = load_toolchain_spec(spec)
    assert loaded[0]["id"] == "successful_motion"

    spec.write_text(
        json.dumps({"schema_version": "wrong", "sequences": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported toolchain spec"):
        load_toolchain_spec(spec)


def test_toolchain_coverage_binds_producer_id_to_consumers(tmp_path: Path) -> None:
    source = tmp_path / "tool_calls.jsonl"
    _write_calls(
        source,
        [
            {
                "name": "sam3",
                "parameters": {"source_packet_id": "obs-10"},
                "semantic_outcome": "no_detection",
            },
            {
                "name": "molmopoint",
                "parameters": {"source_packet_id": "obs-11"},
            },
            {
                "name": "molmopoint",
                "parameters": {"source_packet_id": "obs-10"},
            },
        ],
    )
    sequence = {
        "id": "same_packet",
        "milestones": [
            {
                "tool": "sam3",
                "capture": {"packet": "parameters.source_packet_id"},
            },
            {
                "tool": "molmopoint",
                "where": {"parameters.source_packet_id": {"$ref": "packet"}},
            },
        ],
    }

    report = extract_toolchain_coverage(source, [sequence])

    matched = report["rollouts"][0]["sequences"][0]
    assert matched["complete"] is True
    assert matched["bindings"] == {"packet": "obs-10"}
    assert [call["seq"] for call in matched["matched_calls"]] == [1, 3]
