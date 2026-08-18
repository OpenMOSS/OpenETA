from __future__ import annotations

from adapter.protocol import CameraFrame, EnvObservation, RobotState
from agent.runtime.memory import (
    ANYPLACE_INPUT_BUNDLES_KEY,
    GRASP_PROVENANCE_KEY,
    AgentMemory,
    _memory_fact_entry,
    _same_grasp_candidate,
)


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


def _memory_with_grasp(source):
    memory = AgentMemory()
    memory.start_session(task="pick and place")
    indexed_sources = [_source(1), source]
    for index, indexed in enumerate(indexed_sources):
        memory.add_observation(
            EnvObservation(
                task="pick and place",
                cameras=[
                    CameraFrame(
                        frame_id=indexed["camera_frame_id"],
                        rgb=[],
                        intrinsics=dict(INTRINSICS),
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
            "source_image": _source(1)["rgb"],
            "target_prompt": "basket",
            "evidence_role": "placement_region",
        },
        evidence_role="placement_region",
        source="test",
    )
    return memory


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
