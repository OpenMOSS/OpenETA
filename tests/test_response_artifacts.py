from __future__ import annotations

import json
from pathlib import Path

from agent.runtime.response_artifacts import (
    DEFAULT_RESPONSE_ARTIFACT_OUTPUT_ROOT,
    build_observation_snapshot,
    build_motion_summary,
    build_reachability_summary,
    build_response_reference,
    materialize_json_response,
)


def test_materialize_json_response_strips_preview_values(tmp_path: Path) -> None:
    payload = {
        "ok": True,
        "handle": "env-1",
        "preview": "drop me",
        "cameras": [
            {
                "frame_id": "front",
                "rgb_path": "/tmp/front.png",
                "preview": "drop nested",
            }
        ],
        "content": "full response text",
        "results": [
            {"id": "openeta/demo-0", "description": "first"},
            {"id": "openeta/demo-1", "description": "second"},
        ],
    }

    artifact = materialize_json_response(
        payload,
        output_root=tmp_path,
        bundle_id="bundle",
    )
    saved = json.loads(Path(artifact.path).read_text(encoding="utf-8"))
    reference = build_response_reference(payload, artifact)

    assert saved["content"] == "full response text"
    assert "preview" not in saved
    assert "preview" not in saved["cameras"][0]
    assert reference["handle"] == "env-1"
    assert reference["response_path"] == artifact.path
    assert reference["cameras"][0]["rgb_path"] == "/tmp/front.png"
    assert reference["results_count"] == 2
    assert "results" not in reference


def test_control_spec_and_controller_receipt_survive_compact_reference(
    tmp_path: Path,
) -> None:
    control_spec = {
        "schema_version": "openeta.sim_control.v1",
        "controller": {"controller_id": "robosuite.osc_pose"},
    }
    receipt = {
        "schema_version": "openeta.controller_execution_receipt.v1",
        "controller_id": "robosuite.osc_pose",
        "steps_executed": 12,
        "reached_target": True,
    }
    failure = {
        "schema_version": "openeta.controller_failure.v1",
        "code": "constraint_escape_preview_rejected",
        "current_minimum_distance_m": 0.012,
        "predicted_minimum_distance_m": 0.011,
    }
    create_payload = {"ok": True, "control_spec": control_spec}
    artifact = materialize_json_response(create_payload, output_root=tmp_path)

    reference = build_response_reference(create_payload, artifact)
    motion = build_motion_summary(
        {
            "target": {"x": 0.1, "y": 0.2, "z": 0.3},
            "end": {"xyz": [0.1, 0.2, 0.3]},
            "controller_receipt": receipt,
            "controller_failure": failure,
        }
    )

    assert reference["control_spec"] == control_spec
    assert motion["controller_receipt"] == receipt
    assert motion["controller_failure"] == failure


def test_convergence_and_joint_margin_diagnostics_survive_compact_projection() -> None:
    convergence = {
        "schema_version": "openeta.controller_convergence_diagnostics.v1",
        "code": "full_pose_local_convergence_stalled",
        "near_joint_limits": [{"joint_index": 5, "boundary": "upper"}],
        "recovery": "choose a higher-margin orientation",
    }
    proximity = {
        "near_limit": True,
        "margin_rad": 0.0329,
        "warning_threshold_rad": 0.05,
    }
    seed_quality = {
        "risk_level": "elevated",
        "selected_joint_margin_rad": 0.081,
        "robust_margin_threshold_rad": 0.1,
    }

    motion = build_motion_summary({"convergence_diagnostics": convergence})
    reachability = build_reachability_summary(
        {
            "status": "reachable",
            "reason_code": "ik_solution_found",
            "joint_limit_proximity": proximity,
            "execution_seed_quality": seed_quality,
        }
    )

    assert motion["convergence_diagnostics"] == convergence
    assert reachability["joint_limit_proximity"] == proximity
    assert reachability["execution_seed_quality"] == seed_quality


def test_motion_summary_preserves_per_waypoint_and_sequential_preview_evidence() -> None:
    motion = build_motion_summary(
        {
            "reached_target": True,
            "waypoints_requested": 2,
            "waypoints_completed": 2,
            "waypoint_results": [
                {
                    "reached_target": True,
                    "stop_reason": "target_reached",
                    "steps_executed": 12,
                    "end": {"xyz": [0.0, 0.0, 0.3]},
                    "controller_receipt": {
                        "stable_steps_completed": 3,
                        "ik_execution_seed_receipt_id": "seed-1",
                    },
                },
                {
                    "reached_target": True,
                    "stop_reason": "target_reached",
                    "steps_executed": 18,
                    "end": {"xyz": [0.1, 0.0, 0.3]},
                },
            ],
            "sequential_route_preview": {
                "waypoints_previewed": 2,
                "waypoints_authorized": 2,
                "receipts": [
                    {
                        "index": 0,
                        "preview_id": "preview-1",
                        "status": "reachable",
                        "feasible": True,
                        "joint_positions": [0.1] * 7,
                    }
                ],
            },
        }
    )

    assert motion["waypoints_requested"] == 2
    assert motion["waypoints_completed"] == 2
    assert [item["reached_target"] for item in motion["waypoint_results"]] == [
        True,
        True,
    ]
    assert motion["waypoint_results"][0]["controller_receipt"] == {
        "stable_steps_completed": 3,
        "ik_execution_seed_receipt_id": "seed-1",
    }
    preview = motion["sequential_route_preview"]
    assert preview["waypoints_authorized"] == 2
    assert preview["receipts"][0] == {
        "index": 0,
        "preview_id": "preview-1",
        "status": "reachable",
        "feasible": True,
    }


def test_default_response_output_root_uses_repo_tmp_tool_result_tree() -> None:
    assert DEFAULT_RESPONSE_ARTIFACT_OUTPUT_ROOT == Path("tmp") / "tool_result"


def test_camera_roles_survive_compact_refs_and_observation_snapshots(
    tmp_path: Path,
) -> None:
    payload = {
        "observation": {
            "task": "pick the cup",
            "cameras": [
                {
                    "frame_id": "zed_head",
                    "role": "scene_primary",
                    "rgb_path": str(tmp_path / "zed.png"),
                    "intrinsics": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 50.0,
                    },
                }
            ],
        }
    }
    image_artifacts = [
        {
            "kind": "rgb",
            "frame_id": "zed_head",
            "role": "scene_primary",
            "path": str(tmp_path / "zed.png"),
        }
    ]
    artifact = materialize_json_response(payload, output_root=tmp_path)

    reference = build_response_reference(
        payload,
        artifact,
        image_artifacts=image_artifacts,
    )
    snapshot = build_observation_snapshot(
        payload,
        image_artifacts=image_artifacts,
    )

    assert reference["cameras"][0]["role"] == "scene_primary"
    assert snapshot["observation"]["cameras"][0]["role"] == "scene_primary"
    assert snapshot["observation"]["metadata"]["image_artifacts"][0]["role"] == (
        "scene_primary"
    )


def test_json_responses_with_same_bundle_are_isolated_by_session(tmp_path: Path) -> None:
    first = materialize_json_response(
        {"session": "a"},
        output_root=tmp_path,
        session_id="session-a",
        bundle_id="same",
    )
    second = materialize_json_response(
        {"session": "b"},
        output_root=tmp_path,
        session_id="session-b",
        bundle_id="same",
    )

    assert first.path != second.path
    assert Path(first.path).relative_to(tmp_path).parts[0] == "session-a"
    assert Path(second.path).relative_to(tmp_path).parts[0] == "session-b"
    assert json.loads(Path(first.path).read_text())["session"] == "a"
    assert json.loads(Path(second.path).read_text())["session"] == "b"
