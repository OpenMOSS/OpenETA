from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.univtac.run_pull_out_key_gate import _recover_child_result
from sim.envs.univtac.contract import UniVTACContractError, validate_operator_visible
from sim.envs.univtac.observation import REQUIRED_TACTILE_FIELDS, capture_snapshot
from sim.envs.univtac.pull_out_key_gate import (
    EXPECTED_SEED,
    summarize_pull_out_key_observation,
    validate_gate_config,
)
from sim.envs.univtac.trace import verify_artifacts

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/univtac/pull_out_key_seed1000000_gate.yaml"
PROBE_PATH = REPO_ROOT / "scripts/univtac/probe_pull_out_key_seed.py"
RUNNER_PATH = REPO_ROOT / "scripts/univtac/run_pull_out_key_gate.py"


def _packet(press_depth: np.ndarray | None = None) -> dict[str, np.ndarray]:
    packet = {
        "rgb": np.zeros((240, 320, 3), dtype=np.uint8),
        "rgb_marker": np.zeros((240, 320, 3), dtype=np.uint8),
        "marker": np.zeros((2, 63, 2), dtype=np.float32),
        "depth": np.full((240, 320), 30.0, dtype=np.float32),
        "pose": np.zeros((7,), dtype=np.float32),
    }
    if press_depth is not None:
        packet["press_depth"] = press_depth
    return packet


def _observation(*, positive: bool = True, include_press_depth: bool = True) -> dict:
    press = np.zeros((240, 320), dtype=np.float32)
    if positive:
        press[10, 20] = 0.25
    packets = {
        name: _packet(press.copy() if include_press_depth else None)
        for name in ("left_tactile", "right_tactile")
    }
    return {
        "step": 42,
        "atom": {"id": 0, "tag": ""},
        "observation": {
            "head": {"rgb": np.zeros((270, 480, 3), dtype=np.uint8)},
            "wrist": {"rgb": np.zeros((270, 480, 3), dtype=np.uint8)},
        },
        "embodiment": {
            "joint": np.zeros((9,), dtype=np.float32),
            "ee": np.zeros((7,), dtype=np.float32),
        },
        "tactile": packets,
        "actor": {
            "slot": np.zeros((7,), dtype=np.float32),
            "key": np.zeros((7,), dtype=np.float32),
        },
    }


def test_gate_config_fixes_one_seed_and_forbids_actions() -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    validated = validate_gate_config(payload)
    assert validated["seed"] == EXPECTED_SEED
    assert validated["run_play_once"] is False
    assert validated["call_success_checks"] is False
    assert validated["capture_post_action"] is False
    changed = dict(payload, seed=0)
    with pytest.raises(UniVTACContractError, match="seed=1000000"):
        validate_gate_config(changed)


def test_observation_contract_requires_press_depth_and_positive_contact() -> None:
    summary, contact = summarize_pull_out_key_observation(_observation())
    assert summary["step"] == 42
    assert contact["any_contact_candidate"] is True
    assert contact["bilateral_contact_candidate"] is True
    assert all(sensor["positive_pixel_count"] == 1 for sensor in contact["sensors"].values())
    _, no_contact = summarize_pull_out_key_observation(_observation(positive=False))
    assert no_contact["any_contact_candidate"] is False
    with pytest.raises(UniVTACContractError, match="press_depth"):
        summarize_pull_out_key_observation(_observation(include_press_depth=False))


def test_press_depth_is_optional_for_legacy_but_saved_host_only(tmp_path: Path) -> None:
    assert "press_depth" not in REQUIRED_TACTILE_FIELDS
    observation = _observation()
    capture = capture_snapshot(
        observation,
        output_root=tmp_path,
        seed_dir=tmp_path / "seed_1000000",
        task_name="pull_out_key",
        seed=EXPECTED_SEED,
        phase="pre_action",
        action_id="pull-out-key-seed-1000000-ready",
        simulator_step=42,
        take_action_count=0,
        task_instruction="Pull the key out of the slot.",
        task_metadata={"seed_label": "legacy_ftp1_eval_seed_index_aligned"},
        native_check_success=None,
        save_host_only=True,
        strict_two_tactile_sensors=True,
        fail_on_missing_rgb_marker=True,
    )
    validate_operator_visible(capture.snapshot.operator_visible)
    for sensor_name in ("left_tactile", "right_tactile"):
        visible = capture.snapshot.operator_visible["tactile"][sensor_name]
        assert set(visible) == {"rgb_marker"}
        host = capture.snapshot.host_only["tactile"][sensor_name]
        assert host["press_depth"]["encoding"] == "npy"
        artifact = tmp_path / host["press_depth"]["path"]
        assert artifact.is_file()
        assert np.load(artifact, allow_pickle=False)[10, 20] == pytest.approx(0.25)
        assert capture.snapshot.tactile_sensors[sensor_name]["press_depth"]["shape"] == [240, 320]
    serialized = json.dumps(capture.snapshot.operator_visible, sort_keys=True)
    assert "press_depth" not in serialized
    assert "actor" not in capture.snapshot.operator_visible
    verify_artifacts(tmp_path, capture.snapshot.to_dict())


def test_legacy_capture_without_press_depth_still_succeeds(tmp_path: Path) -> None:
    capture = capture_snapshot(
        _observation(include_press_depth=False),
        output_root=tmp_path,
        seed_dir=tmp_path / "seed",
        task_name="legacy",
        seed=0,
        phase="pre_action",
        action_id="ready",
        simulator_step=0,
        take_action_count=0,
        task_instruction="legacy",
        task_metadata={},
        native_check_success=None,
        save_host_only=True,
        strict_two_tactile_sensors=True,
        fail_on_missing_rgb_marker=True,
    )
    assert all(
        "press_depth" not in payload for payload in capture.snapshot.host_only["tactile"].values()
    )


def test_probe_static_contract_has_one_reset_one_observation_and_no_action() -> None:
    source = PROBE_PATH.read_text(encoding="utf-8")
    assert source.count("task.reset(") == 1
    assert source.count("task._get_observations()") == 1
    assert "task.play_once()" not in source
    assert "task.check_success()" not in source
    assert "task.check_early_stop()" not in source
    assert "build_transition" not in source
    assert 'write_snapshot(seed_dir / "snapshot_post.json"' not in source
    assert source.index("recorder.install()") < source.index("task.reset(")
    assert 'phase="pre_action"' in source
    assert "capture_snapshot(" in source
    assert "PlannerDiagnosticRecorder(task)" in source
    assert "no_parallel_runtime" not in source
    assert "hashlib" not in source


def test_stage_contract_and_single_simulator_invocation_are_explicit() -> None:
    probe = PROBE_PATH.read_text(encoding="utf-8")
    for stage in (
        "app_launcher_construct",
        "app_launcher_started",
        "post_launcher_imports",
        "task_config_load",
        "task_constructor",
        "planner_recorder_install",
        "reset",
        "reset_actors",
        "marker_stabilization",
        "marker_calibration",
        "pre_move",
        "pre_move_delay",
        "pre_move_grasp_actor",
        "reset_returned",
        "get_observations",
        "snapshot_projection",
        "task_close",
        "simulation_app_close",
    ):
        assert f'"{stage}"' in probe
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    assert '"simulator_invocation_limit": 1' in runner
    assert 'run_manifest["simulator_invocation_count"] = 1' in runner
    assert "run_scoped_isaac51_command(spec)" in runner
    assert "retry" not in runner.lower()


def test_child_result_is_persisted_before_simulation_app_close() -> None:
    source = PROBE_PATH.read_text(encoding="utf-8")
    close_call = source.index("simulation_app.close()")
    prefix = source[:close_call]
    assert prefix.rfind('seed_dir / "child_result.json"') > prefix.rfind(
        'stages.record("simulation_app_close", "enter")'
    )


def test_clean_close_process_exit_can_recover_persisted_gate_evidence(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    for relative in (
        "envs/pull_out_key.py",
        "envs/_base_task.py",
        "third_party/TacEx/source/tacex/tacex/__init__.py",
        "third_party/TacEx/source/tacex_uipc/tacex_uipc/__init__.py",
        "third_party/TacEx/source/tacex_assets/tacex_assets/__init__.py",
    ):
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    seed_dir = tmp_path / "pull_out_key_seed1000000"
    seed_dir.mkdir()
    required_exits = (
        "task_constructor",
        "planner_recorder_install",
        "reset_actors",
        "marker_stabilization",
        "marker_calibration",
        "pre_move",
        "reset",
        "reset_returned",
        "get_observations",
        "snapshot_projection",
        "task_close",
    )
    records = [{"stage": stage, "event": "exit"} for stage in required_exits]
    records.append({"stage": "simulation_app_close", "event": "enter"})
    (seed_dir / "stages.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    payloads = {
        "planner_diagnostics.json": {
            "move_calls": [{}],
            "planning_calls": [{"motion_gen_success": True}],
        },
        "contact_summary.json": {"any_contact_candidate": True},
        "observation_summary.json": {"step": 238},
        "snapshot_pre.json": {
            "operator_visible": {
                "cameras": {},
                "tactile": {},
                "proprio": {},
                "task_instruction": "Pull the key out of the slot.",
                "step_identifiers": {},
            },
            "host_only": {
                "task_metadata": {
                    "plan_success": True,
                    "cid_present": True,
                    "target_pose_present": True,
                    "slot_init_pose_present": True,
                }
            },
        },
    }
    for name, payload in payloads.items():
        (seed_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    recovered = _recover_child_result(
        tmp_path,
        source_root,
        {
            "returncode": 0,
            "timed_out": False,
            "cleanup_complete": True,
            "sigterm_sent": False,
            "sigkill_sent": False,
        },
    )
    assert recovered is not None
    assert recovered["classification"] == (
        "scoped_launcher_and_pull_out_key_seed1000000_gate_passed"
    )
    assert recovered["cleanup"]["simulation_app_close_ended_interpreter"] is True
    assert recovered["counters"]["play_once_call_count"] == 0
