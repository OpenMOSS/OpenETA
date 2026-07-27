"""Lightweight BEHAVIOR integration tests (no Isaac Sim import)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_behavior_registry_is_static_and_complete_for_openeta_manifest():
    from sim.env_registry import list_envs

    specs = list_envs(env_type="behavior")
    assert len(specs) == 1016
    assert any(spec.id == "openeta/behavior_turning_on_radio-v0" for spec in specs)
    assert all(spec.default_robot == "R1Pro" for spec in specs)
    assert all(spec.requires_gpu and spec.requires_sim_install for spec in specs)


def test_behavior_version_manifest_is_pinned():
    path = Path("sim/envs/behavior/benchmark_versions.json")
    manifest = json.loads(path.read_text())
    assert manifest["behavior_1k_tag"] == "v3.9.0"
    assert manifest["behavior_1k_commit"] == "6559858f7c814143f08be27830d24fac16a12058"
    assert manifest["isaac_sim"] == "5.1.0"
    assert manifest["python"] == "3.11"
    assert manifest["bddl_activity_count"] == 1016
    assert manifest["challenge_2026_task_count"] == 100
    assert manifest["rlinf_evaluation_manifest_count"] == 50


def test_behavior_challenge_metadata_selects_canonical_variant(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from sim.envs.behavior.direct_env import _challenge_task_config

    metadata = tmp_path / "2026-challenge-task-instances" / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "available_tasks.yaml").write_text("fixture")
    monkeypatch.setitem(
        sys.modules,
        "yaml",
        SimpleNamespace(
            safe_load=lambda _: {
                "task_a": {0: {"scene_model": "house_a", "robot_start_position": [1, 2, 3]}}
            }
        ),
    )
    assert _challenge_task_config(tmp_path, "task_a") == {
        "scene_model": "house_a",
        "robot_start_position": [1, 2, 3],
    }
    assert _challenge_task_config(tmp_path, "missing") is None


def test_behavior_sensor_flattening_without_omnigibson():
    from sim.envs.behavior.direct_env import BehaviorDirectEnv

    raw = {
        "robot0": {
            "zed_link:Camera:0": {
                "rgb": np.ones((2, 3, 4), dtype=np.float32),
                "depth": np.full((2, 3, 1), 1.25, dtype=np.float32),
            },
            "left_realsense_link:Camera:0": {
                "rgb": np.zeros((2, 3, 4), dtype=np.uint8),
                "depth_linear": np.full((2, 3), 0.5, dtype=np.float32),
            },
            "right_realsense_link:Camera:0": {
                "rgb": np.full((2, 3, 4), 2, dtype=np.uint8),
                "depth": np.full((2, 3), 0.75, dtype=np.float32),
            },
            "proprio": np.arange(5, dtype=np.float32),
        }
    }
    obs = BehaviorDirectEnv._flatten_sensor_obs(raw)
    assert obs["main_images"].shape == (2, 3, 3)
    assert obs["main_images"].dtype == np.uint8
    assert obs["wrist_images"].shape == (2, 2, 3, 3)
    np.testing.assert_allclose(obs["main_depth"], 1.25)
    assert obs["main_depth"].dtype == np.float32
    assert obs["wrist_depths"].shape == (2, 2, 3)
    np.testing.assert_allclose(obs["wrist_depths"][0], 0.5)
    np.testing.assert_allclose(obs["wrist_depths"][1], 0.75)
    np.testing.assert_array_equal(obs["states"], np.arange(5, dtype=np.float32))


def test_behavior_unified_observation_preserves_rgbd_calibration_and_robot_state():
    from adapter.protocol import EnvObservation
    from sim.unified_env import UnifiedEnv

    wrapper = object.__new__(UnifiedEnv)
    raw = {
        "main_images": np.zeros((2, 3, 3), dtype=np.uint8),
        "main_depth": np.full((2, 3), 1.25, dtype=np.float32),
        "wrist_images": np.zeros((2, 2, 3, 3), dtype=np.uint8),
        "wrist_depths": np.full((2, 2, 3), 0.5, dtype=np.float32),
        "_openeta_camera_params": {
            "zed_head": {
                "intrinsics": {"fx": 100.0, "fy": 101.0, "cx": 1.0, "cy": 0.5},
                "extrinsics": {
                    "pos": [0.0, 0.0, 1.0],
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "frame_transform": "camera_to_world",
                },
            }
        },
        "_openeta_proprio": {
            "joint_positions": np.arange(6, dtype=np.float32),
            "joint_velocities": np.ones(6, dtype=np.float32),
            "ee_pose": np.array([1, 2, 3, 0, 0, 0, 1], dtype=np.float32),
            "gripper_open": 0.75,
            "gripper_state": {"open": True, "open_fraction": 0.75},
            "metadata": {"primary_arm": "right"},
        },
        "_openeta_metadata": {"benchmark": "behavior-1k", "step_index": 4},
    }

    unified = wrapper._normalise_behavior(raw)
    observation = EnvObservation.from_dict(unified, task="place the object")

    assert [camera.frame_id for camera in observation.cameras] == [
        "zed_head",
        "wrist_left",
        "wrist_right",
    ]
    np.testing.assert_allclose(observation.cameras[0].depth, 1.25)
    assert observation.cameras[0].intrinsics["fx"] == 100.0
    assert observation.cameras[0].extrinsics["frame_transform"] == "camera_to_world"
    assert observation.robot.end_effector_pose["quat_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert observation.robot.gripper_state["open"] is True
    assert observation.robot.gripper_state["open_fraction"] == 0.75
    assert observation.metadata["step_index"] == 4


def test_behavior_unified_observation_sanitizes_invalid_depth():
    from sim.unified_env import UnifiedEnv

    wrapper = object.__new__(UnifiedEnv)
    unified = wrapper._normalise_behavior(
        {
            "main_images": np.zeros((1, 4, 3), dtype=np.uint8),
            "main_depth": np.array([[np.nan, np.inf, -1.0, 1.5]], dtype=np.float32),
        }
    )
    np.testing.assert_array_equal(
        unified["cameras"]["zed_head"]["depth"],
        np.array([[0.0, 0.0, 0.0, 1.5]], dtype=np.float32),
    )


def test_behavior_direct_env_extracts_public_robot_and_camera_state():
    from sim.envs.behavior.direct_env import BehaviorDirectEnv

    class FakeRobot:
        arm_names = ("left", "right")
        default_arm = "left"
        gripper_control_idx = {
            "left": np.array([2, 3]),
            "right": np.array([4, 5]),
        }
        joint_lower_limits = np.zeros(6, dtype=np.float32)
        joint_upper_limits = np.ones(6, dtype=np.float32)

        def get_joint_positions(self):
            return np.array([0.1, 0.2, 0.3, 0.4, 0.8, 1.0], dtype=np.float32)

        def get_joint_velocities(self):
            return np.arange(6, dtype=np.float32)

        def get_position_orientation(self):
            return np.array([1, 2, 3]), np.array([0, 0, 0, 1])

        def get_eef_pose(self, arm):
            x = 1.0 if arm == "right" else -1.0
            return np.array([x, 0, 1]), np.array([0, 0, 0, 1])

    class FakeSensor:
        intrinsic_matrix = np.array([[10, 0, 2], [0, 11, 3], [0, 0, 1]])
        image_width = 4
        image_height = 6

        def get_position_orientation(self):
            return np.array([0, 0, 1]), np.array([0, 0, 0, 1])

    robot = FakeRobot()
    robot.sensors = {"right_realsense_link:Camera:0": FakeSensor()}
    direct = object.__new__(BehaviorDirectEnv)
    direct._env = SimpleNamespace(
        robots=[robot], external_sensors={"external_sensor0": FakeSensor()}
    )

    proprio = direct._structured_proprio()
    assert proprio["metadata"]["primary_arm"] == "right"
    np.testing.assert_allclose(proprio["ee_pose"], [1, 0, 1, 0, 0, 0, 1])
    assert proprio["gripper_state"]["open_fraction"] == pytest.approx(0.9)

    cameras = direct._camera_parameters()
    assert set(cameras) == {"zed_head", "wrist_right"}
    assert cameras["zed_head"]["intrinsics"]["fx"] == 10.0
    assert cameras["wrist_right"]["extrinsics"]["frame_transform"] == "camera_to_world"


def test_behavior_worker_resolves_nested_conda_runtime(tmp_path, monkeypatch):
    from sim.mcp_server import worker_mgr

    runtime = tmp_path / "venvs" / "behavior" / "runtime" / "bin"
    runtime.mkdir(parents=True)
    python = runtime / "python3.11"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    monkeypatch.setattr(worker_mgr, "_SIM_DIR", tmp_path)
    assert worker_mgr._venv_python("behavior") == str(python)


def test_behavior_main_thread_executor_runs_on_owner_thread():
    import threading

    from sim.bench_worker import _MainThreadExecutor

    owner = threading.get_ident()
    executor = _MainThreadExecutor()
    future = executor.submit(threading.get_ident)
    executor.run_once()
    assert future.result() == owner
