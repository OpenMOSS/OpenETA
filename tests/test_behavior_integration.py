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


def test_behavior_sensor_config_requests_optical_z_depth():
    from sim.envs.behavior.direct_env import _require_rgbd_modalities

    sensor_config = {
        "modalities": ["rgb", "seg_instance"],
        "sensor_kwargs": {"image_height": 128, "image_width": 128},
    }
    _require_rgbd_modalities(sensor_config)

    assert sensor_config["modalities"] == ["rgb", "seg_instance", "depth_linear"]
    assert sensor_config["sensor_kwargs"] == {
        "image_height": 128,
        "image_width": 128,
    }


def test_behavior_sensor_flattening_without_omnigibson():
    from sim.envs.behavior.direct_env import BehaviorDirectEnv

    raw = {
        "robot0": {
            "zed_link:Camera:0": {
                "rgb": np.ones((2, 3, 4), dtype=np.float32),
                "depth_linear": np.full((2, 3, 1), 1.25, dtype=np.float32),
            },
            "left_realsense_link:Camera:0": {
                "rgb": np.zeros((2, 3, 4), dtype=np.uint8),
                "depth_linear": np.full((2, 3), 0.5, dtype=np.float32),
            },
            "right_realsense_link:Camera:0": {
                "rgb": np.full((2, 3, 4), 2, dtype=np.uint8),
                "depth_linear": np.full((2, 3), 0.75, dtype=np.float32),
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
    assert obs["_openeta_camera_sources"] == {
        "zed_head": {
            "observation_path": "robot0/zed_link:Camera:0",
            "sensor_name": "zed_link:Camera:0",
        },
        "wrist_left": {
            "observation_path": "robot0/left_realsense_link:Camera:0",
            "sensor_name": "left_realsense_link:Camera:0",
        },
        "wrist_right": {
            "observation_path": "robot0/right_realsense_link:Camera:0",
            "sensor_name": "right_realsense_link:Camera:0",
        },
    }


def test_behavior_calibration_binds_robot_zed_frame_not_external_sensor():
    from sim.envs.behavior.direct_env import BehaviorDirectEnv

    class FakeSensor:
        def __init__(self, name, prim_path, fx, position):
            self.name = name
            self.prim_path = prim_path
            self._intrinsic_matrix = np.array(
                [[fx, 0.0, 1.0], [0.0, fx + 1.0, 0.5], [0.0, 0.0, 1.0]]
            )
            self.image_width = 3
            self.image_height = 2
            self._position = np.asarray(position, dtype=np.float64)
            self.intrinsic_reads = 0

        @property
        def intrinsic_matrix(self):
            self.intrinsic_reads += 1
            return self._intrinsic_matrix

        def get_position_orientation(self):
            return self._position, np.array([0.0, 0.0, 0.0, 1.0])

    robot_sensor_names = (
        "robot_r1:zed_link:Camera:0",
        "robot_r1:left_realsense_link:Camera:0",
        "robot_r1:right_realsense_link:Camera:0",
    )
    raw = {
        "external_sensor0": {
            "rgb": np.full((2, 3, 4), 99, dtype=np.uint8),
            "depth_linear": np.full((2, 3), 9.0, dtype=np.float32),
        },
        "robot_r1": {
            name: {
                "rgb": np.full((2, 3, 4), index + 1, dtype=np.uint8),
                "depth_linear": np.full(
                    (2, 3),
                    0.5 + index,
                    dtype=np.float32,
                ),
            }
            for index, name in enumerate(robot_sensor_names)
        },
    }
    flattened = BehaviorDirectEnv._flatten_sensor_obs(raw)

    external = FakeSensor(
        "external_sensor0",
        "/World/external_sensor0",
        900.0,
        [9.0, 9.0, 9.0],
    )
    robot_sensors = {
        robot_sensor_names[0]: FakeSensor(
            robot_sensor_names[0],
            "/World/robot_r1/zed_link/Camera",
            100.0,
            [1.0, 2.0, 3.0],
        ),
        robot_sensor_names[1]: FakeSensor(
            robot_sensor_names[1],
            "/World/robot_r1/left_realsense_link/Camera",
            200.0,
            [4.0, 5.0, 6.0],
        ),
        robot_sensor_names[2]: FakeSensor(
            robot_sensor_names[2],
            "/World/robot_r1/right_realsense_link/Camera",
            300.0,
            [7.0, 8.0, 9.0],
        ),
    }
    direct = object.__new__(BehaviorDirectEnv)
    direct._env = SimpleNamespace(
        external_sensors={"external_sensor0": external},
        robots=[SimpleNamespace(sensors=robot_sensors)],
    )

    parameters = direct._camera_parameters(
        flattened["_openeta_camera_sources"]
    )

    assert set(parameters) == {"zed_head", "wrist_left", "wrist_right"}
    assert parameters["zed_head"]["intrinsics"]["fx"] == 100.0
    assert parameters["zed_head"]["extrinsics"]["pos"] == [1.0, 2.0, 3.0]
    assert parameters["zed_head"]["calibration_source"] == {
        "observation_path": (
            "robot_r1/robot_r1:zed_link:Camera:0"
        ),
        "registry_key": "robot_r1:zed_link:Camera:0",
        "sensor_name": "robot_r1:zed_link:Camera:0",
        "prim_path": "/World/robot_r1/zed_link/Camera",
    }
    assert external.intrinsic_reads == 0
    assert all(sensor.intrinsic_reads == 1 for sensor in robot_sensors.values())
    np.testing.assert_array_equal(
        flattened["main_images"],
        np.full((2, 3, 3), 1, dtype=np.uint8),
    )
    np.testing.assert_allclose(flattened["main_depth"], 0.5)


def test_behavior_sensor_flattening_rejects_radial_depth_fallback():
    from sim.envs.behavior.direct_env import BehaviorDirectEnv

    raw = {
        "robot0": {
            name: {
                "rgb": np.zeros((2, 3, 4), dtype=np.uint8),
                "depth": np.full((2, 3), 0.75, dtype=np.float32),
            }
            for name in (
                "zed_link:Camera:0",
                "left_realsense_link:Camera:0",
                "right_realsense_link:Camera:0",
            )
        }
    }

    with pytest.raises(RuntimeError, match="requires depth_linear"):
        BehaviorDirectEnv._flatten_sensor_obs(raw)


def test_behavior_sensor_flattening_rejects_misaligned_linear_depth():
    from sim.envs.behavior.direct_env import BehaviorDirectEnv

    raw = {
        "robot0": {
            name: {
                "rgb": np.zeros((2, 3, 4), dtype=np.uint8),
                "depth_linear": np.zeros(
                    (1, 3) if "right_realsense" in name else (2, 3),
                    dtype=np.float32,
                ),
            }
            for name in (
                "zed_link:Camera:0",
                "left_realsense_link:Camera:0",
                "right_realsense_link:Camera:0",
            )
        }
    }

    with pytest.raises(RuntimeError, match="dimensions differ"):
        BehaviorDirectEnv._flatten_sensor_obs(raw)


def test_behavior_sensor_flattening_fails_closed_without_right_wrist():
    from sim.envs.behavior.direct_env import BehaviorDirectEnv

    raw = {
        "robot0": {
            "zed_link:Camera:0": {
                "rgb": np.ones((2, 3, 4), dtype=np.float32),
            },
            "left_realsense_link:Camera:0": {
                "rgb": np.zeros((2, 3, 4), dtype=np.uint8),
            },
        }
    }

    with pytest.raises(RuntimeError, match="right-wrist"):
        BehaviorDirectEnv._flatten_sensor_obs(raw)


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
    from adapter.protocol import EnvObservation
    from sim.envs.behavior.direct_env import BehaviorDirectEnv
    from sim.unified_env import UnifiedEnv

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
        def __init__(self, name, prim_path):
            self.name = name
            self.prim_path = prim_path
            self.intrinsic_matrix = np.array(
                [[10, 0, 2], [0, 11, 3], [0, 0, 1]]
            )
            self.image_width = 4
            self.image_height = 6

        def get_position_orientation(self):
            return np.array([0, 0, 1]), np.array([0, 0, 0, 1])

    robot = FakeRobot()
    robot.sensors = {
        "right_realsense_link:Camera:0": FakeSensor(
            "right_realsense_link:Camera:0",
            "/World/robot/right_realsense_link/Camera",
        )
    }
    direct = object.__new__(BehaviorDirectEnv)
    direct._env = SimpleNamespace(
        robots=[robot],
        external_sensors={
            "external_sensor0": FakeSensor(
                "external_sensor0",
                "/World/external_sensor0",
            )
        },
    )

    proprio = direct._structured_proprio()
    assert proprio["metadata"]["primary_arm"] == "right"
    np.testing.assert_allclose(proprio["ee_pose"], [1, 0, 1, 0, 0, 0, 1])
    assert proprio["gripper_state"]["open_fraction"] == pytest.approx(0.9)

    camera_sources = {
        "zed_head": {
            "observation_path": "external_sensor0",
            "sensor_name": "external_sensor0",
        },
        "wrist_right": {
            "observation_path": "robot0/right_realsense_link:Camera:0",
            "sensor_name": "right_realsense_link:Camera:0",
        },
    }
    cameras = direct._camera_parameters(camera_sources)
    assert set(cameras) == {"zed_head", "wrist_right"}
    assert cameras["zed_head"]["intrinsics"]["fx"] == 10.0
    assert cameras["zed_head"]["intrinsics"]["depth_unit"] == "meter"
    assert cameras["zed_head"]["calibration_source"]["prim_path"] == (
        "/World/external_sensor0"
    )
    wrist_extrinsics = cameras["wrist_right"]["extrinsics"]
    assert wrist_extrinsics["frame_transform"] == "camera_to_world"
    assert wrist_extrinsics["camera_frame"] == "opencv"
    assert wrist_extrinsics["matrix_layout"] == "row_major"
    assert wrist_extrinsics["image_origin"] == "top_left"
    assert wrist_extrinsics["raw_camera_convention"] == "omnigibson_usd"
    assert wrist_extrinsics["normalized_from"] == "omnigibson_usd"
    np.testing.assert_allclose(
        np.asarray(wrist_extrinsics["mat"]).reshape(3, 3),
        np.diag([1.0, -1.0, -1.0]),
    )
    np.testing.assert_allclose(
        wrist_extrinsics["camera_to_world"],
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )

    direct.activity_name = "test_activity"
    direct._step_index = 0
    annotated = direct._annotate_observation(
        {
            "main_images": np.zeros((6, 4, 3), dtype=np.uint8),
            "main_depth": np.ones((6, 4), dtype=np.float32),
            "_openeta_camera_sources": {
                "zed_head": camera_sources["zed_head"],
            },
        }
    )
    assert set(annotated["_openeta_camera_params"]) == {"zed_head"}
    assert annotated["_openeta_metadata"]["depth_unit"] == "meter"
    unified = object.__new__(UnifiedEnv)._normalise_behavior(annotated)
    mcp_camera = EnvObservation.from_dict(unified).to_mcp_dict()["cameras"][0]
    assert mcp_camera["frame_id"] == "zed_head"
    assert mcp_camera["intrinsics"]["depth_unit"] == "meter"
    assert mcp_camera["extrinsics"]["camera_frame"] == "opencv"
    assert mcp_camera["extrinsics"]["image_origin"] == "top_left"
    assert mcp_camera["depth_encoding"] == "uint16_png"
    assert mcp_camera["depth_scale"] == 1000.0
    assert len(mcp_camera["extrinsics"]["camera_to_world"]) == 4

    with pytest.raises(RuntimeError, match="dimensions do not match"):
        direct._annotate_observation(
            {
                "main_images": np.zeros((5, 4, 3), dtype=np.uint8),
                "main_depth": np.ones((5, 4), dtype=np.float32),
                "_openeta_camera_sources": {
                    "zed_head": camera_sources["zed_head"],
                },
            }
        )


def test_behavior_direct_reset_uses_shared_deterministic_seeding(monkeypatch):
    from sim.envs.behavior.direct_env import BehaviorDirectEnv

    seeded: list[int] = []
    monkeypatch.setattr(
        "sim.envs.behavior.direct_env.seed_behavior_reset_rngs",
        seeded.append,
    )
    direct = object.__new__(BehaviorDirectEnv)
    direct._env = SimpleNamespace(reset=lambda: ({"raw": True}, {"reset": True}))
    direct._step_index = 8
    direct._flatten_sensor_obs = lambda _raw: {
        "main_images": np.zeros((1, 1, 3), dtype=np.uint8)
    }
    direct._annotate_observation = lambda obs: obs

    obs, info = direct.reset(seed=23)

    assert seeded == [23]
    assert direct._step_index == 0
    assert obs["main_images"] is direct._last_render
    assert info == {"reset": True}


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
