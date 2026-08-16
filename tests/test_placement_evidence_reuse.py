from __future__ import annotations

from agent.runtime.memory import (
    ANYPLACE_INPUT_BUNDLES_KEY,
    GRASP_PROVENANCE_KEY,
    AgentMemory,
    _memory_fact_entry,
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
        "rgb": base + ".rgb.png",
        "depth": base + ".depth.png",
        "object_mask": f"artifacts/obs-{observation_index:04d}/object-mask.png",
        "intrinsics": dict(INTRINSICS),
    }


def _memory_with_grasp(source):
    memory = AgentMemory()
    memory.start_session(task="pick and place")
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
            "image": _source(2, frame="robot0_eye_in_hand")["rgb"],
            "prompt": "basket",
            "evidence_role": "placement_region",
        },
    }
