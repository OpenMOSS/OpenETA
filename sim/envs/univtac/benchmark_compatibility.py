"""Static UniVTAC Isaac 4.5 versus 5.1 compatibility judgments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from sim.envs.univtac.source_parity import require_text


FTP1_TASKS = (
    "lift_bottle",
    "lift_can",
    "put_bottle_in_shelf",
    "pull_out_key",
    "insert_hole",
    "insert_tube",
)
CONTACT_RICH_TASKS = ("pull_out_key", "insert_hole", "insert_tube")


def evidence(side: str, path: str, lines: str) -> dict[str, str]:
    return {"source": side, "path": path, "lines": lines}


def classify_rtx5090_recipe(
    *, support_claim: bool, explicit_5090: bool, explicit_sm120: bool,
    explicit_ptx_forward_compatibility: bool
) -> str:
    if explicit_5090 and (explicit_sm120 or explicit_ptx_forward_compatibility):
        return "explicitly_documented"
    if support_claim and explicit_ptx_forward_compatibility:
        return "implied_by_forward_compatibility"
    if support_claim:
        return "claimed_but_not_fully_documented"
    return "unresolved"


def cross_version_comparability(layers: Mapping[str, str]) -> str:
    if any(value in {"changed", "incompatible"} for value in layers.values()):
        return "not_preserved"
    if any(value == "unresolved" for value in layers.values()):
        return "unresolved"
    return "preserved"


def dependency_matrix() -> dict[str, Any]:
    return {
        "schema_version": "openeta.univtac.dependency_matrix.v1",
        "components": [
            {"component": "Python", "legacy": "3.10", "isaac51": "3.11", "status": "changed", "evidence": [evidence("legacy", "docs/Installation.md", "5-8"), evidence("isaac51", "docs/Installation.md", "7-14")]},
            {"component": "PyTorch", "legacy": "2.5.1 (installation guide)", "isaac51": "2.7.0", "status": "changed", "evidence": [evidence("legacy", "docs/Installation.md", "45-50"), evidence("isaac51", "scripts/install.sh", "218-226")]},
            {"component": "torchvision", "legacy": "0.20.1 (installation guide)", "isaac51": "0.22.0", "status": "changed", "evidence": [evidence("legacy", "docs/Installation.md", "45-50"), evidence("isaac51", "scripts/install.sh", "218-226")]},
            {"component": "PyTorch CUDA runtime", "legacy": "cu118 in guide; repository requirements also mention cu128", "isaac51": "cu126", "status": "legacy_source_inconsistent_and_new_changed", "evidence": [evidence("legacy", "docs/Installation.md", "45-50"), evidence("legacy", "requirements.txt", "82-83"), evidence("isaac51", "docs/Installation.md", "20-26")]},
            {"component": "CUDA toolkit", "legacy": "12.4", "isaac51": "12.6", "status": "changed", "evidence": [evidence("legacy", "third_party/TacEx/source/tacex_uipc/libuipc/conda/env.yaml", "6-10"), evidence("isaac51", "third_party/TacEx/source/tacex_uipc/libuipc/conda/env.yaml", "7-13")]},
            {"component": "Isaac Sim", "legacy": "4.5.0", "isaac51": "5.1.0", "status": "changed", "evidence": [evidence("legacy", "docs/Installation.md", "5-8"), evidence("isaac51", "docs/Installation.md", "7-12")]},
            {"component": "Isaac Lab", "legacy": "2.1.1", "isaac51": "2.3.0", "status": "changed", "evidence": [evidence("legacy", "docs/Installation.md", "5-8"), evidence("isaac51", "docs/Installation.md", "7-12")]},
            {"component": "TacEx", "legacy": "vendored modified source; revision not declared", "isaac51": "f2051944e469a961241271fbf6fb60e272fc336b", "status": "newly_pinned", "evidence": [evidence("legacy", "docs/Installation.md", "8-9"), evidence("isaac51", "third_party/TacEx/UNIVTAC_BASELINE.md", "3-6")]},
            {"component": "libuipc", "legacy": "expanded vendored source; revision not declared", "isaac51": "1a7e93ef68765e4d3c15d5583f5c387e89af5183", "status": "newly_pinned", "evidence": [evidence("isaac51", "third_party/TacEx/UNIVTAC_BASELINE.md", "5-9")]},
            {"component": "muda", "legacy": "revision not declared", "isaac51": "8f9e17d8e76a658df3b6ffeeffbbc9ac47ac54bf", "status": "newly_pinned", "evidence": [evidence("isaac51", "third_party/TacEx/UNIVTAC_BASELINE.md", "6-9")]},
            {"component": "SymEigen", "legacy": "revision not declared", "isaac51": "c72a0082e44b3b8062727b25c33ce7450f9fa933", "status": "newly_pinned", "evidence": [evidence("isaac51", "third_party/TacEx/UNIVTAC_BASELINE.md", "7-9")]},
            {"component": "pyuipc", "legacy": "0.9.0 source family; build not pinned", "isaac51": "0.9.0", "status": "version_nominally_preserved_build_changed", "evidence": [evidence("isaac51", "scripts/install.sh", "169-185")]},
            {"component": "cuRobo", "legacy": "official guide; revision not pinned", "isaac51": "ebb71702f3f70e767f40fd8e050674af0288abe8", "status": "newly_pinned", "evidence": [evidence("legacy", "docs/Installation.md", "150-152"), evidence("isaac51", "scripts/install.sh", "10-11,265-279")]},
            {"component": "vcpkg", "legacy": "unversioned clone", "isaac51": "dd3097e305afa53f7b4312371f62058d2e665320", "status": "newly_pinned", "evidence": [evidence("legacy", "docs/Installation.md", "109-125"), evidence("isaac51", "scripts/install.sh", "8-9,238-251")]},
            {"component": "host compiler", "legacy": "GCC/G++ 11.4", "isaac51": "Conda GCC/G++ 12 with system-linker wrappers", "status": "changed", "evidence": [evidence("legacy", "docs/Installation.md", "127-134"), evidence("isaac51", "scripts/install.sh", "146-158")]},
            {"component": "CMake", "legacy": "3.26", "isaac51": "3.26", "status": "preserved", "evidence": [evidence("legacy", "third_party/TacEx/source/tacex_uipc/libuipc/conda/env.yaml", "7-10"), evidence("isaac51", "third_party/TacEx/source/tacex_uipc/libuipc/conda/env.yaml", "9-13")]},
            {"component": "Ninja", "legacy": "not source-declared in environment", "isaac51": "declared", "status": "changed", "evidence": [evidence("isaac51", "third_party/TacEx/source/tacex_uipc/libuipc/conda/env.yaml", "9-13")]},
            {"component": "torch_scatter", "legacy": "source references py310/pt28cu128 despite guide using pt2.5/cu118", "isaac51": "py311/pt27cu126 wheel", "status": "changed_and_legacy_inconsistent", "evidence": [evidence("legacy", "third_party/TacEx/source/tacex/setup.py", "23-30"), evidence("isaac51", "third_party/TacEx/source/tacex/setup.py", "23-26")]},
            {"component": "sensor optical backend", "legacy": "Taxim path", "isaac51": "Taxim default plus Pix2Pix startup selection", "status": "expanded", "evidence": [evidence("isaac51", "docs/isaacsim_5_1_migration.md", "25-43")]},
            {"component": "supported sensor types", "legacy": "gsmini/gf225/xensews code; collection/eval gsmini caveat", "isaac51": "same factories; public phase-one validation only gsmini", "status": "validation_scope_not_equivalent_to_code_support", "evidence": [evidence("isaac51", "README.md", "34-42,76-78"), evidence("isaac51", "docs/Installation.md", "108-116")]},
            {"component": "num_envs", "legacy": "multi-env APIs present", "isaac51": "phase one intentionally num_envs=1", "status": "restricted", "evidence": [evidence("isaac51", "docs/Installation.md", "131-132"), evidence("isaac51", "docs/isaacsim_5_1_migration.md", "72-80")]},
            {"component": "data format", "legacy": "legacy HDF5", "isaac51": "HDF5 extended with press_depth; cross-version data declared incompatible", "status": "incompatible", "evidence": [evidence("isaac51", "README.md", "8-14"), evidence("isaac51", "docs/Collection.md", "63-120")]},
            {"component": "render synchronization", "legacy": "TacEx callback/extra render behavior", "isaac51": "one explicit UIPC surface-to-Fabric copy per render", "status": "changed", "evidence": [evidence("isaac51", "docs/isaacsim_5_1_migration.md", "83-89")]},
        ],
    }


def cuda_build_matrix() -> dict[str, Any]:
    status = classify_rtx5090_recipe(
        support_claim=True,
        explicit_5090=False,
        explicit_sm120=False,
        explicit_ptx_forward_compatibility=False,
    )
    return {
        "schema_version": "openeta.univtac.cuda_build_matrix.v1",
        "rtx5090_build_recipe_status": status,
        "facts": {
            "public_40_50_series_support_claim": True,
            "default_univtac_cuda_arch": "89",
            "documented_allowed_values": ["89", "8.9"],
            "cmake_cuda_architectures_propagation": "normalized integer string",
            "torch_cuda_arch_list_propagation": "decimal inserted before final digit",
            "sm_120_documented": False,
            "compute_120_documented": False,
            "blackwell_documented": False,
            "rtx5090_named": False,
            "ptx_forward_compatibility_documented": False,
            "cuda_12_6_declared": True,
        },
        "judgment": "The branch claims RTX 50-series support, but the pinned public source does not fully document an RTX 5090/sm_120 build recipe or PTX fallback.",
        "evidence": [
            evidence("isaac51", "README.md", "8-14"),
            evidence("isaac51", "docs/Installation.md", "20-26,71-82"),
            evidence("isaac51", "scripts/install.sh", "6-7,28,67-72,132-136"),
            evidence("isaac51", "third_party/TacEx/source/tacex_uipc/setup.py", "68-70,86-87"),
        ],
    }


def _task_entry(task: str, compatibility: str, diffs: list[dict[str, Any]]) -> dict[str, Any]:
    audited_fields = {
        "create_actors": "changed" if task in {"lift_bottle", "put_bottle_in_shelf", "pull_out_key", "insert_hole", "insert_tube"} else "preserved_with_config_extension",
        "actor_initial_pose": "changed" if task in {"lift_bottle", "put_bottle_in_shelf", "pull_out_key", "insert_hole", "insert_tube"} else "preserved",
        "actor_body_motion_type": "changed" if task in {"lift_bottle", "pull_out_key", "insert_hole", "insert_tube"} else "runtime_default_requires_validation",
        "actor_density": "changed_to_explicit_kinematic" if task in {"lift_bottle", "pull_out_key", "insert_hole", "insert_tube"} else "preserved_or_runtime_default",
        "reset_noise": "changed_rotation_composition" if task == "pull_out_key" else "preserved",
        "reset_seed_handling": "preserved_at_task_source; shared_reset_pipeline_changed",
        "pre_move": "source_logic_equivalent",
        "expert_play_once": "source_logic_equivalent",
        "check_success": "source_logic_equivalent",
        "check_mid_success": "changed" if task == "lift_bottle" else "source_logic_equivalent_or_not_overridden",
        "check_early_stop": "press_depth_semantics_changed" if task in {"lift_can", "put_bottle_in_shelf"} else "source_logic_equivalent",
        "task_instruction": "no_task_specific_literal; optional BaseTask instruction sampling",
        "task_metadata": "expanded",
        "step_action_budget": "preserved_at_task_source",
        "cleanup": "shared_base_runtime_changed",
        "save_eval_behavior": "shared_base_runtime_changed",
    }
    return {
        "task": task,
        "task_semantic_compatibility": compatibility,
        "step_action_budget": 500 if task == "lift_bottle" else 300,
        "audited_fields": audited_fields,
        "semantic_differences": diffs,
    }


def task_semantic_matrix() -> dict[str, Any]:
    entries = [
        _task_entry("lift_bottle", "physics_semantics_changed_but_goal_preserved", [
            {"field": "grasp_threshold", "legacy": "absolute camera-depth threshold 27.8 mm", "isaac51": "sensor-specific positive press-depth threshold", "classification": "control_semantics_changed", "evidence": [evidence("legacy", "envs/lift_bottle.py", "5-7"), evidence("isaac51", "envs/lift_bottle.py", "5-8")]},
            {"field": "wall.body_behavior", "legacy": "density=1e5; z=0.005", "isaac51": "motion_type=kinematic; z=0.001", "classification": "physics_semantics_changed", "evidence": [evidence("legacy", "envs/lift_bottle.py", "13-21"), evidence("isaac51", "envs/lift_bottle.py", "14-22")]},
            {"field": "check_mid_success", "legacy": "height only", "isaac51": "height plus bottle-axis alignment", "classification": "success_path_changed", "evidence": [evidence("legacy", "envs/lift_bottle.py", "71-73"), evidence("isaac51", "envs/lift_bottle.py", "74-78")]},
        ]),
        _task_entry("lift_can", "physics_semantics_changed_but_goal_preserved", [
            {"field": "actor_set", "legacy": "fixed can sizes 4/5/6", "isaac51": "configurable can_sizes, default 4/5/6", "classification": "config_surface_expanded", "evidence": [evidence("legacy", "envs/lift_can.py", "12-35"), evidence("isaac51", "envs/lift_can.py", "16-41")]},
            {"field": "check_early_stop.tactile", "legacy": "minimum camera depth <20", "isaac51": "maximum positive press depth exceeds sensor safety limit", "classification": "early_stop_semantics_changed", "evidence": [evidence("legacy", "envs/lift_can.py", "74-88"), evidence("isaac51", "envs/lift_can.py", "80-94")]},
        ]),
        _task_entry("put_bottle_in_shelf", "physics_semantics_changed_but_goal_preserved", [
            {"field": "shelf_base.z", "legacy": 0.01, "isaac51": 0.001, "classification": "initial_geometry_changed", "evidence": [evidence("legacy", "envs/put_bottle_in_shelf.py", "15-22"), evidence("isaac51", "envs/put_bottle_in_shelf.py", "15-22")]},
            {"field": "check_early_stop.tactile", "legacy": "minimum camera depth", "isaac51": "maximum positive press depth", "classification": "early_stop_semantics_changed", "evidence": [evidence("legacy", "envs/put_bottle_in_shelf.py", "82-89"), evidence("isaac51", "envs/put_bottle_in_shelf.py", "82-89")]},
        ]),
        _task_entry("pull_out_key", "physics_semantics_changed_but_goal_preserved", [
            {"field": "slot.body_behavior", "legacy": "density=1e5", "isaac51": "motion_type=kinematic", "classification": "physics_semantics_changed", "evidence": [evidence("legacy", "envs/pull_out_key.py", "19-30"), evidence("isaac51", "envs/pull_out_key.py", "19-30")]},
            {"field": "reset.rotation_composition", "legacy": "key bias before shared rotation; random+key rotation applied to key", "isaac51": "base random rotation before key bias; key gets only relative key rotation", "classification": "initial_distribution_changed", "evidence": [evidence("legacy", "envs/pull_out_key.py", "32-43"), evidence("isaac51", "envs/pull_out_key.py", "32-43")]},
            {"field": "pre_move_and_success", "legacy": "same targets, constraints and success thresholds", "isaac51": "source logic equivalent", "classification": "preserved", "evidence": [evidence("legacy", "envs/pull_out_key.py", "45-108"), evidence("isaac51", "envs/pull_out_key.py", "45-108")]},
        ]),
        _task_entry("insert_hole", "physics_semantics_changed_but_goal_preserved", [
            {"field": "initial_z", "legacy": {"slot": 0.002, "base": 0.002, "prism": 0.005}, "isaac51": {"slot": 0.001, "base": 0.001, "prism": 0.003}, "classification": "initial_geometry_changed", "evidence": [evidence("legacy", "envs/insert_hole.py", "12-15"), evidence("isaac51", "envs/insert_hole.py", "12-15")]},
            {"field": "slot_and_base.body_behavior", "legacy": "density=1e5", "isaac51": "motion_type=kinematic", "classification": "physics_semantics_changed", "evidence": [evidence("legacy", "envs/insert_hole.py", "17-34"), evidence("isaac51", "envs/insert_hole.py", "17-34")]},
            {"field": "pre_move.place_sequence", "legacy": "two place_actor calls: .05/.01 then .01/.002; final constraint [1,1,1,1,1,0]", "isaac51": "identical", "classification": "preserved", "evidence": [evidence("legacy", "envs/insert_hole.py", "42-88"), evidence("isaac51", "envs/insert_hole.py", "42-88")]},
            {"field": "constraint_target_success", "legacy": "target +/-pi/6; success xy .01, z .04, axis .99", "isaac51": "identical", "classification": "preserved", "evidence": [evidence("legacy", "envs/insert_hole.py", "62-85,113-124"), evidence("isaac51", "envs/insert_hole.py", "62-85,113-124")]},
        ]),
        _task_entry("insert_tube", "physics_semantics_changed_but_goal_preserved", [
            {"field": "initial_z", "legacy": {"slot": 0.002, "base": 0.002, "prism": 0.005}, "isaac51": {"slot": 0.001, "base": 0.001, "prism": 0.003}, "classification": "initial_geometry_changed", "evidence": [evidence("legacy", "envs/insert_tube.py", "12-15"), evidence("isaac51", "envs/insert_tube.py", "12-15")]},
            {"field": "slot_and_base.body_behavior", "legacy": "density=1e5", "isaac51": "motion_type=kinematic", "classification": "physics_semantics_changed", "evidence": [evidence("legacy", "envs/insert_tube.py", "17-34"), evidence("isaac51", "envs/insert_tube.py", "17-34")]},
            {"field": "pre_move.place_sequence", "legacy": "two place_actor calls .1/.05 then .05/.002 with final constraint", "isaac51": "identical", "classification": "preserved", "evidence": [evidence("legacy", "envs/insert_tube.py", "41-81"), evidence("isaac51", "envs/insert_tube.py", "41-81")]},
            {"field": "success_contract", "legacy": "same signed xy comparison, z/upright/in-hand thresholds", "isaac51": "identical, including existing signed-xy behavior", "classification": "preserved", "evidence": [evidence("legacy", "envs/insert_tube.py", "100-122"), evidence("isaac51", "envs/insert_tube.py", "100-122")]},
        ]),
    ]
    return {
        "schema_version": "openeta.univtac.task_semantic_matrix.v1",
        "tasks": entries,
        "shared_runtime_changes": [
            {"field": "camera_resolution", "legacy": "640x360", "isaac51": "480x270", "classification": "observation_changed", "evidence": [evidence("isaac51", "envs/_base_task.py", "162-200")]},
            {"field": "reset_pipeline", "legacy": "direct actor reset/pre_move", "isaac51": "initial replay, 20-step actor stabilization and marker calibration", "classification": "reset_semantics_changed", "evidence": [evidence("legacy", "envs/_base_task.py", "430-449"), evidence("isaac51", "envs/_base_task.py", "437-500")]},
            {"field": "render_pipeline", "legacy": "callback/extra render path", "isaac51": "explicit UIPC-to-Fabric copy before one render", "classification": "sensor_timing_changed", "evidence": [evidence("isaac51", "envs/_base_task.py", "487-496,581-607")]},
            {"field": "adaptive_grasp", "legacy": "absolute camera depth", "isaac51": "positive press depth", "classification": "control_semantics_changed", "evidence": [evidence("legacy", "envs/_base_task.py", "760-769"), evidence("isaac51", "envs/_base_task.py", "921-933,1084+")]},
        ],
    }


def action_contract_matrix() -> dict[str, Any]:
    return {
        "schema_version": "openeta.univtac.action_contract_matrix.v1",
        "openeta_adapter_migration_scope": "observation_and_runtime",
        "fields": [
            {"field": "Action/Atom move schemas", "legacy": "move/open/close/gripper/all", "isaac51": "preserved", "status": "preserved"},
            {"field": "move_by_displacement", "legacy": "world translation, local RPY default", "isaac51": "same public semantics", "status": "preserved", "evidence": [evidence("isaac51", "envs/utils/atom.py", "288-303")]},
            {"field": "place_actor/grasp_actor", "legacy": "contact/functional points with pre_dis/dis", "isaac51": "same task-facing API", "status": "preserved", "evidence": [evidence("isaac51", "envs/utils/atom.py", "174-198,261-286")]},
            {"field": "adaptive gripper threshold", "legacy": "gripper_depth_threshold / camera distance", "isaac51": "gripper_press_depth_threshold / indentation", "status": "changed"},
            {"field": "Cartesian frame convention", "legacy": "world translation and local RPY defaults", "isaac51": "preserved by Atom API", "status": "preserved"},
            {"field": "Pose quaternion convention", "legacy": "Pose quaternion accepted by shared pose utilities", "isaac51": "task-facing convention preserved; Isaac transforms explicitly convert to wxyz", "status": "preserved_with_runtime_conversion"},
            {"field": "qpos", "legacy": "8 values (7 arm + gripper)", "isaac51": "8 values", "status": "preserved"},
            {"field": "qpos absolute/relative semantics", "legacy": "qpos is absolute; delta_ee is relative", "isaac51": "preserved", "status": "preserved"},
            {"field": "gripper action convention", "legacy": "percent in collect/pre_move and qpos in eval", "isaac51": "same mode split; threshold meaning changes to press depth", "status": "partially_changed"},
            {"field": "ee", "legacy": "position+quaternion+gripper implementation", "isaac51": "same implementation; docstring shape remains inconsistent", "status": "preserved_with_documentation_risk", "evidence": [evidence("isaac51", "envs/_base_task.py", "1038-1073")]},
            {"field": "delta_ee", "legacy": "3 translation + 3 rotation + gripper implementation", "isaac51": "same 7-value implementation while docstring says 6", "status": "preserved_with_documentation_risk", "evidence": [evidence("isaac51", "envs/_base_task.py", "1046-1073")]},
            {"field": "control decimation", "legacy": "no configured evaluation wrapper", "isaac51": "take_action executes under configured decimation", "status": "changed"},
            {"field": "action frequency", "legacy": "physical/control frequency coupling in legacy config", "isaac51": "explicit physical/collect/save/eval ratios with integral validation", "status": "changed"},
            {"field": "planner execution", "legacy": "plan_arm then dense action", "isaac51": "same high-level path plus force propagation/render changes", "status": "changed_at_runtime_boundary"},
            {"field": "world-changing action count", "legacy": "take_action_cnt increments per public action", "isaac51": "preserved; atom_id also advances on move/delay", "status": "preserved"},
            {"field": "settle/render", "legacy": "delay and render path", "isaac51": "delay remains; explicit render synchronization", "status": "changed"},
        ],
        "migration_judgment": "The public action vocabulary remains usable, but runtime launch/render timing and observation adaptation must change. A full worker rewrite is not justified by static source alone.",
    }


def tactile_contract_matrix() -> dict[str, Any]:
    return {
        "schema_version": "openeta.univtac.tactile_contract_matrix.v1",
        "task_facing_keys": ["rgb", "rgb_marker", "marker", "depth", "press_depth", "pose"],
        "sensor_ordering": "discovered from tactile manager; no static left/right order guarantee",
        "rgb": {"shape_gsmini": [240, 320, 3], "dtype": "uint8 at documented boundary", "range": [0, 255], "channel_order": "HWC"},
        "rgb_marker": {"shape_gsmini": [240, 320, 3], "dtype": "uint8", "range": [0, 255], "channel_order": "HWC", "public_smoke_evidence": True},
        "marker": {"gsmini": [2, 63, 2], "gf225_count": 81, "xensews_count": 220, "meaning": "marker motion/reference-current pair"},
        "depth": {"legacy": "raw sensor-camera distance in mm", "isaac51": "legacy raw distance retained"},
        "press_depth": {"legacy": "absent as distinct dataset field", "isaac51": "positive indentation in mm; zero means no press"},
        "pose": {"shape": [7], "convention": "task Pose vector", "visibility": "privileged host-only in OpenETA contract"},
        "baseline_reference_semantics": "marker stores reference/current motion pair; Isaac51 recalibrates marker reference during reset",
        "marker_correspondence": "migration notes declare 82 gel-to-case attachment ids and preserved marker correspondence; no source assertion was found",
        "optical_backend": {"legacy": "Taxim", "isaac51": "Taxim or Pix2Pix; normalized boundary is documented as HWC uint8 0-255", "normalization_implementation_verified": False},
        "render_synchronization": {"legacy": "old callback/extra render", "isaac51": "one explicit UIPC surface-to-Fabric copy before camera read"},
        "observation_frequency": "derived from physical/collect/save/eval frequencies; evaluation decimation is explicit in Isaac51",
        "hdf5_encoding": "datasets whose names contain rgb use JPEG streams; joint/ee state-action arrays follow repository HDF5 handler conventions",
        "data_format": {"legacy": "HDF5 legacy depth", "isaac51": "HDF5 plus press_depth", "cross_version_compatible": False},
        "e1_supported": True,
        "e2_supported": True,
        "e3_supported": True,
        "blocking_changes": [
            "Isaac51 runtime has not yet passed an isolated RTX5090 smoke.",
            "Old FTP-1 checkpoints and demonstrations are not proven compatible with changed sensor/control timing.",
        ],
        "non_blocking_changes": [
            "Raw rgb_marker remains available for within-version E1.",
            "Structured tactile can be computed from rgb_marker for E2.",
            "Pre/action/post binding can be recorded by the existing OpenETA trace contract for E3.",
        ],
        "compatibility": {
            "new_agent_experiments": "supported_after_runtime_smoke",
            "legacy_ftp1_checkpoint": "unresolved",
            "legacy_demonstrations": "incompatible_as_direct_isaac51_data",
            "legacy_preprocessing": "requires_depth_and_timing_audit",
            "legacy_paper_success_rate": "not_directly_comparable",
        },
        "evidence": [
            evidence("isaac51", "envs/sensors/tactile.py", "355-423"),
            evidence("isaac51", "scripts/smoke_isaac51.py", "44-126"),
            evidence("isaac51", "docs/isaacsim_5_1_migration.md", "25-58,83-89"),
            evidence("isaac51", "docs/Collection.md", "63-120"),
        ],
    }


def official_validation_scope() -> dict[str, Any]:
    tasks = {
        task: {
            "code_contains_task": True,
            "smoke_tested": "unresolved_from_public_source",
            "data_collection_validated": "unresolved_from_public_source",
            "evaluation_validated": "unresolved_from_public_source",
            "dataset_currently_released_for_isaac51": "not_released_per_isaac51_notice",
            "checkpoint_currently_released_for_isaac51": "unresolved_from_public_source",
        }
        for task in FTP1_TASKS
    }
    return {
        "schema_version": "openeta.univtac.official_validation_scope.v1",
        "tasks": tasks,
        "phase_one_explicit_evidence": {
            "smoke_isaac51": "grasp_classify-based camera, gsmini RGB, Actor pose and render-copy checks",
            "collection": "one grasp_classify/demo/Taxim episode",
            "six_ftp1_tasks": "unresolved_from_public_source",
            "evaluation": "unresolved_from_public_source",
        },
        "dataset": {
            "readme_existing_download_links": True,
            "readme_says_isaac51_dataset_soon": True,
            "internal_consistency": "ambiguous_between_legacy_downloads_and_future_isaac51_dataset",
            "isaac51_dataset_status": "not_released_per_isaac51_notice",
        },
        "documentation_inconsistencies": [
            "README says isaac51 targets 5.1 and later says main targets 5.1, while the same note says main preserves 4.5.",
            "README task gallery lists three tactile sensor families, but TODO limits collection/evaluation to GelSight Mini.",
            "Migration notes reference docs/Install.md, while the pinned tree contains docs/Installation.md.",
        ],
        "evidence": [
            evidence("isaac51", "README.md", "8-18,34-50,76-78"),
            evidence("isaac51", "docs/Installation.md", "98-116"),
            evidence("isaac51", "docs/isaacsim_5_1_migration.md", "16-19,91-108"),
            evidence("isaac51", "scripts/smoke_isaac51.py", "44-126"),
        ],
    }


def benchmark_compatibility(
    *, legacy_commit: str, isaac51_commit: str
) -> dict[str, Any]:
    layers = {
        "task_identity_parity": "preserved",
        "success_contract_parity": "changed",
        "control_parity": "changed",
        "physics_sensor_parity": "changed",
        "dataset_checkpoint_parity": "incompatible",
    }
    return {
        "schema_version": "openeta.univtac.benchmark_compatibility.v1",
        "isaac51_commit": isaac51_commit,
        "legacy_commit": legacy_commit,
        "rtx5090_build_recipe_status": "claimed_but_not_fully_documented",
        "layers": layers,
        "within_version_experimental_validity": "conditionally_valid_pending_isolated_runtime_smoke",
        "cross_version_numeric_comparability": cross_version_comparability(layers),
        "openeta_adapter_migration_scope": "observation_and_runtime",
        "six_task_validation_status": "unresolved_from_public_source",
        "recommended_next_step": "clarify_rtx5090_build_recipe_then_isolated_smoke",
        "interpretation": {
            "within_version": "E0/E1/E2/E3 can be compared fairly inside one pinned Isaac51 environment after runtime validation.",
            "cross_version": "Isaac51 results must be labeled UniVTAC-Isaac51 and cannot be presented as direct reproduction of FTP-1 Isaac4.5 numbers.",
        },
    }


def validate_expected_source(legacy: Path, isaac51: Path) -> None:
    require_text(legacy, "envs/insert_hole.py", ["density=1e5", "pre_dis=0.01, dis=0.002", "constraint_pose=[1, 1, 1, 1, 1, 0]"])
    require_text(isaac51, "envs/insert_hole.py", ["motion_type=\"kinematic\"", "pre_dis=0.01, dis=0.002", "constraint_pose=[1, 1, 1, 1, 1, 0]"])
    require_text(legacy, "envs/pull_out_key.py", ["random_rotate+self.key_rotation", "density=1e5"])
    require_text(isaac51, "envs/pull_out_key.py", ["motion_type=\"kinematic\"", "key_pose = base_pose.add_bias"])
    require_text(isaac51, "README.md", ["supports NVIDIA RTX 40- and 50-series GPUs", "Data collected with the two", "not cross-compatible"])
    require_text(isaac51, "scripts/install.sh", ["UNIVTAC_CUDA_ARCH:-89", "CMAKE_CUDA_ARCHITECTURES", "TORCH_CUDA_ARCH_LIST"])
