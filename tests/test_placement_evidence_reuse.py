from __future__ import annotations

import pytest

from adapter.protocol import CameraFrame, EnvAction, EnvObservation, RobotState
from agent.runtime.memory import (
    ANYPLACE_INPUT_BUNDLES_KEY,
    GRASP_PROVENANCE_KEY,
    AgentMemory,
    _memory_fact_entry,
    _same_grasp_candidate,
)
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import PlannerDecision
from agent.runtime.skills import build_default_skill_registry
from agent.tools.handlers import bind_dummy_tool_handlers
from agent.tools.registry import build_default_tool_registry


INTRINSICS = {"fx": 600.0, "fy": 600.0, "cx": 256.0, "cy": 256.0, "scale": 1000.0}


def _candidate():
    return {
        "id": "grasp_001",
        "frame": "camera",
        "camera_frame": "opencv",
        "score": 0.8,
        "translation_xyz": [0.1, 0.2, 0.3],
        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "gripper_tip_position_xyz": [0.1, 0.2, 0.33],
        "depth": 0.03,
        "width": 0.06,
        "height": 0.03,
    }


def _source(observation_index: int, *, frame: str = "agentview"):
    base = f"artifacts/obs-{observation_index:04d}/cameras.0.{frame}"
    return {
        "mode": "targeted",
        "source_packet_id": f"packet-{observation_index}",
        "camera_frame_id": frame,
        "rgb": base + ".rgb.png",
        "depth": base + ".depth.png",
        "object_mask": f"artifacts/obs-{observation_index:04d}/object-mask.png",
        "intrinsics": dict(INTRINSICS),
    }


def _memory_with_grasp(source, *, placement_source=None):
    memory = AgentMemory()
    memory.start_session(task="pick and place")
    placement_source = placement_source or _source(1)
    indexed_sources = [placement_source, source]
    for index, indexed in enumerate(indexed_sources):
        memory.add_observation(
            EnvObservation(
                task="pick and place",
                cameras=[
                    CameraFrame(
                        frame_id=indexed["camera_frame_id"],
                        rgb=[],
                        intrinsics=dict(INTRINSICS),
                        extrinsics={
                            "camera_frame": "opengl",
                            "frame_transform": "camera_to_world",
                            "pos": [0.8, 0.0, 0.6],
                            "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        },
                    )
                ],
                robot=RobotState(),
                metadata={
                    "image_artifacts": [
                        {
                            "kind": "rgb",
                            "frame_id": indexed["camera_frame_id"],
                            "path": indexed["rgb"],
                            "packet_id": indexed["source_packet_id"],
                        },
                        {
                            "kind": "depth",
                            "frame_id": indexed["camera_frame_id"],
                            "path": indexed["depth"],
                            "packet_id": indexed["source_packet_id"],
                        },
                    ],
                    "step_idx": index,
                },
            )
        )
    memory.facts[GRASP_PROVENANCE_KEY] = _memory_fact_entry(
        {
            "schema_version": "openeta.grasp_provenance.v1",
            "evidence_id": "grasp:new",
            "candidate": _candidate(),
            "source": source,
        },
        source="test",
    )
    memory._store_selected_sam3_detection(
        {
            "id": "detection_target",
            "result_id": "sam3-target-1",
            "mask_ref": source["object_mask"],
            "source_image": source["rgb"],
            "target_prompt": "milk carton",
            "evidence_role": "target_object",
        },
        evidence_role="target_object",
        source="test",
    )
    memory._store_selected_sam3_detection(
        {
            "id": "detection_000",
            "result_id": "sam3-placement-1",
            "mask_ref": "artifacts/placement-mask.png",
            "source_image": placement_source["rgb"],
            "target_prompt": "basket",
            "evidence_role": "placement_region",
        },
        evidence_role="placement_region",
        source="test",
    )
    return memory


def test_anyplace_accepts_byte_identical_fixed_camera_frame_without_resegmenting(
    tmp_path,
) -> None:
    placement_rgb = tmp_path / "placement-frame.png"
    grasp_rgb = tmp_path / "grasp-frame.png"
    placement_rgb.write_bytes(b"same deterministic fixed-camera frame")
    grasp_rgb.write_bytes(b"same deterministic fixed-camera frame")
    placement_source = {
        **_source(1),
        "rgb": str(placement_rgb),
    }
    grasp_source = {
        **_source(2),
        "rgb": str(grasp_rgb),
    }
    memory = _memory_with_grasp(
        grasp_source,
        placement_source=placement_source,
    )

    assert memory._refresh_anyplace_input_bundle() is True
    public = memory.anyplace_input_bundle()
    assert public["status"] == "ready"
    assert public["placement_evidence_reuse"]["mode"] == "identical_frame_content"
    resolved = memory.resolve_anyplace_input_bundle(public["bundle_id"])
    assert resolved["parameters"]["placement_region_mask"]["source_image"] == str(
        grasp_rgb
    )


def test_anyplace_reuses_placement_mask_on_same_fixed_camera() -> None:
    memory = _memory_with_grasp(_source(2))
    placement_id = "placement:sam3-placement-1:detection_000"
    prior_source = _source(1)
    prior_bundle = {
        "schema_version": "openeta.anyplace_input_bundle.v1",
        "bundle_id": "anyplace:prior",
        "placement_evidence_id": placement_id,
        "parameters": {
            "rgb": prior_source["rgb"],
            "intrinsics": dict(INTRINSICS),
            "selected_grasp": {"candidate": _candidate(), "source": prior_source},
        },
    }
    memory.facts[ANYPLACE_INPUT_BUNDLES_KEY] = _memory_fact_entry(
        {
            "schema_version": "openeta.anyplace_input_bundle_store.v1",
            "active_bundle_id": "anyplace:prior",
            "public": {"status": "ready", "bundle_id": "anyplace:prior"},
            "bundles": {"anyplace:prior": prior_bundle},
        },
        source="test",
    )

    assert memory._refresh_anyplace_input_bundle() is True
    public = memory.anyplace_input_bundle()
    assert public["status"] == "ready"
    assert public["bundle_id"] != "anyplace:prior"
    assert public["placement_evidence_reuse"]["mode"] == "fixed_camera_identity"
    resolved = memory.resolve_anyplace_input_bundle(public["bundle_id"])
    parameters = resolved["parameters"]
    assert parameters["placement_region_mask"]["mask_ref"] == "artifacts/placement-mask.png"
    assert parameters["placement_region_mask"]["source_image"] == _source(2)["rgb"]


def test_anyplace_wrist_mismatch_requires_exact_repair_call() -> None:
    memory = _memory_with_grasp(_source(2, frame="robot0_eye_in_hand"))

    assert memory._refresh_anyplace_input_bundle() is True
    public = memory.anyplace_input_bundle()
    assert public["status"] == "placement_source_mismatch"
    assert public["repair_call"] == {
        "tool": "sam3",
        "parameters": {
            "source_packet_id": "packet-1",
            "camera_frame_id": "agentview",
            "prompt": "milk carton",
            "evidence_role": "target_object",
        },
    }
    assert public["required_source_image"] == _source(1)["rgb"]
    assert "fixed placement camera" in public["recovery"]


def test_anyplace_rebases_same_target_wrist_grasp_into_prior_fixed_camera() -> None:
    fixed_source = _source(1)
    wrist_source = _source(2, frame="robot0_eye_in_hand")
    memory = _memory_with_grasp(wrist_source, placement_source=fixed_source)
    anchor_id = "target:milk"

    memory.facts[GRASP_PROVENANCE_KEY] = _memory_fact_entry(
        {
            "schema_version": "openeta.grasp_provenance.v1",
            "evidence_id": "grasp:fixed",
            "candidate": _candidate(),
            "source": fixed_source,
            "target_identity_anchor_id": anchor_id,
        },
        source="test",
    )
    memory._store_selected_sam3_detection(
        {
            "id": "fixed-target",
            "result_id": "sam3-fixed-target",
            "mask_ref": fixed_source["object_mask"],
            "source_image": fixed_source["rgb"],
            "target_prompt": "milk carton",
            "evidence_role": "target_object",
        },
        evidence_role="target_object",
        source="test",
    )
    assert memory._refresh_anyplace_input_bundle() is True
    assert memory.anyplace_input_bundle()["status"] == "ready"

    memory.facts[GRASP_PROVENANCE_KEY] = _memory_fact_entry(
        {
            "schema_version": "openeta.grasp_provenance.v1",
            "evidence_id": "grasp:wrist",
            "candidate": _candidate(),
            "source": wrist_source,
            "target_identity_anchor_id": anchor_id,
        },
        source="test",
    )
    memory._store_selected_sam3_detection(
        {
            "id": "wrist-target",
            "result_id": "sam3-wrist-target",
            "mask_ref": wrist_source["object_mask"],
            "source_image": wrist_source["rgb"],
            "target_prompt": "milk carton",
            "evidence_role": "target_object",
        },
        evidence_role="target_object",
        source="test",
    )

    assert memory._refresh_anyplace_input_bundle() is True
    public = memory.anyplace_input_bundle()
    assert public["status"] == "ready"
    assert public["grasp_rebase"]["mode"] == "calibrated_world_invariant"
    assert public["grasp_rebase"]["source_camera_frame_id"] == "robot0_eye_in_hand"
    assert public["grasp_rebase"]["target_camera_frame_id"] == "agentview"
    resolved = memory.resolve_anyplace_input_bundle(public["bundle_id"])
    selected_grasp = resolved["parameters"]["selected_grasp"]
    assert selected_grasp["source"]["camera_frame_id"] == "agentview"
    assert selected_grasp["source"]["object_mask"] == fixed_source["object_mask"]
    assert selected_grasp["candidate"]["id"] == "grasp_001"
    assert (
        selected_grasp["candidate"]["cross_camera_rebase"][
            "target_identity_anchor_id"
        ]
        == anchor_id
    )


def test_anyplace_rebases_from_durable_fixed_grasp_before_first_anyplace_call() -> None:
    fixed_source = _source(1)
    wrist_source = _source(2, frame="wrist")
    memory = _memory_with_grasp(wrist_source, placement_source=fixed_source)
    anchor_id = "target:milk"
    fixed_provenance = {
        "schema_version": "openeta.grasp_provenance.v1",
        "evidence_id": "grasp:fixed-history",
        "candidate": _candidate(),
        "source": fixed_source,
        "object_scene_epoch": 0,
        "target_identity_anchor_id": anchor_id,
    }
    memory.record(
        "grasp_provenance_bound",
        {"provenance": fixed_provenance},
    )
    memory.facts[GRASP_PROVENANCE_KEY] = _memory_fact_entry(
        {
            "schema_version": "openeta.grasp_provenance.v1",
            "evidence_id": "grasp:wrist-current",
            "candidate": _candidate(),
            "source": wrist_source,
            "object_scene_epoch": 0,
            "target_identity_anchor_id": anchor_id,
        },
        source="test",
    )
    memory._store_selected_sam3_detection(
        {
            "id": "wrist-target",
            "result_id": "sam3-wrist-target",
            "mask_ref": wrist_source["object_mask"],
            "source_image": wrist_source["rgb"],
            "target_prompt": "milk carton",
            "evidence_role": "target_object",
        },
        evidence_role="target_object",
        source="test",
    )

    assert memory._refresh_anyplace_input_bundle() is True
    public = memory.anyplace_input_bundle()
    assert public["status"] == "ready"
    assert public["grasp_rebase"]["source_bundle_id"] is None
    assert (
        public["grasp_rebase"]["source_grasp_evidence_id"]
        == "grasp:fixed-history"
    )

    # Unrelated actions must not recursively rebase this bundle onto itself and
    # change its content-addressed identity.
    bundle_id = public["bundle_id"]
    memory.add_action(EnvAction(action_type="response", command={}))
    assert memory.anyplace_input_bundle()["bundle_id"] == bundle_id

    # A successful result is frozen against the exact requested bundle and
    # remains resolvable after the normal post-action bundle refresh.
    result_id = "anyplace-result:wrist-rebase-stable"
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "anyplace",
                    "parameters": {"bundle_id": bundle_id},
                },
                "tool_calls": [
                    {
                        "name": "anyplace",
                        "parameters": {"bundle_id": bundle_id},
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "result_id": result_id,
                                    "placement_candidates": [
                                        {
                                            "id": "placement_000",
                                            "place_grasp_pose": _candidate(),
                                        }
                                    ],
                                }
                            },
                        },
                    }
                ],
            },
        )
    )
    assert memory.anyplace_input_bundle()["materialized_result_id"] == result_id
    resolved = memory.resolve_placement_candidate_input(
        placement_result_id=result_id,
        candidate_id="placement_000",
    )
    assert resolved["placement_result_id"] == result_id


def test_grasp_candidate_identity_tolerates_json_float_noise_only() -> None:
    original = _candidate()
    round_tripped = _candidate()
    round_tripped["rotation_matrix"] = [row[:] for row in original["rotation_matrix"]]
    round_tripped["rotation_matrix"][0][0] += 8e-14

    assert _same_grasp_candidate(original, round_tripped) is True

    edited = _candidate()
    edited["rotation_matrix"] = [row[:] for row in original["rotation_matrix"]]
    edited["rotation_matrix"][0][0] += 1e-6
    assert _same_grasp_candidate(original, edited) is False


def test_materialized_anyplace_candidate_resolves_original_packet_calibration() -> None:
    source = _source(1)
    memory = _memory_with_grasp(source)
    assert memory._refresh_anyplace_input_bundle() is True
    bundle_id = memory.anyplace_input_bundle()["bundle_id"]
    place_pose = {
        **_candidate(),
        "id": "place_grasp_002",
        "source_grasp_id": "grasp_001",
    }
    result_id = "anyplace-result:resolved-1"
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "anyplace",
                    "parameters": {"bundle_id": bundle_id},
                },
                "tool_calls": [
                    {
                        "name": "anyplace",
                        "parameters": {"bundle_id": bundle_id},
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "result_id": result_id,
                                    "placement_candidates": [
                                        {
                                            "id": "placement_002",
                                            "source_grasp_id": "grasp_001",
                                            "place_grasp_pose": place_pose,
                                        }
                                    ],
                                }
                            },
                        },
                    }
                ],
            },
        )
    )

    resolved = memory.resolve_placement_candidate_input(
        placement_result_id=result_id,
        candidate_id="placement_002",
    )

    assert resolved["source_packet_id"] == "packet-1"
    assert resolved["camera_frame_id"] == "agentview"
    assert resolved["parameters"]["camera_pose"] == place_pose
    assert resolved["parameters"]["camera_extrinsics"]["pos"] == [0.8, 0.0, 0.6]


def test_placement_candidate_resolver_reports_valid_ids() -> None:
    source = _source(1)
    memory = _memory_with_grasp(source)
    assert memory._refresh_anyplace_input_bundle() is True
    bundle_id = memory.anyplace_input_bundle()["bundle_id"]
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {"parameters": {"bundle_id": bundle_id}},
                "tool_calls": [
                    {
                        "name": "anyplace",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "result_id": "anyplace-result:known",
                                    "placement_candidates": [
                                        {
                                            "id": "placement_000",
                                            "place_grasp_pose": _candidate(),
                                        }
                                    ],
                                }
                            },
                        },
                    }
                ],
            },
        )
    )

    with pytest.raises(ValueError, match="valid ids: \\['placement_000'\\]"):
        memory.resolve_placement_candidate_input(
            placement_result_id="anyplace-result:known",
            candidate_id="placement_missing",
        )


def test_pipeline_atomically_resolves_placement_reference_before_transform() -> None:
    source = _source(1)
    memory = _memory_with_grasp(source)
    assert memory._refresh_anyplace_input_bundle() is True
    bundle_id = memory.anyplace_input_bundle()["bundle_id"]
    place_pose = {
        **_candidate(),
        "id": "place_grasp_004",
        "translation_xyz": [0.2, 0.1, 0.4],
    }
    result_id = "anyplace-result:pipeline"
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {"parameters": {"bundle_id": bundle_id}},
                "tool_calls": [
                    {
                        "name": "anyplace",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "result_id": result_id,
                                    "placement_candidates": [
                                        {
                                            "id": "placement_004",
                                            "place_grasp_pose": place_pose,
                                        }
                                    ],
                                }
                            },
                        },
                    }
                ],
            },
        )
    )
    tools = bind_dummy_tool_handlers(build_default_tool_registry())

    plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="camera_pose_to_world",
            parameters={
                "placement_result_id": result_id,
                "candidate_id": "placement_004",
            },
        ),
        observation=EnvObservation(task="place object", cameras=[], robot=RobotState()),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert plan.status.value == "executed"
    call = plan.tool_calls[0]
    # Agent-facing action history keeps the compact public reference. The host
    # resolved pose/extrinsics were consumed during execution and must not be
    # copied back into the conversation ledger.
    assert "camera_pose" not in call.parameters
    assert "camera_extrinsics" not in call.parameters
    assert call.parameters["placement_result_id"] == result_id
    assert call.parameters["candidate_id"] == "placement_004"
    assert call.result["details"]["outputs"]["world_pose"]["frame"] == "world"
    assert call.result["details"]["outputs"]["candidate_id"] == "placement_004"
