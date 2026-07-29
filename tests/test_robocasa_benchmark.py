from __future__ import annotations

import json

import numpy as np
import pytest

from adapter.protocol import EnvAction, EnvObservation, RobotState, StepResult
from agent.runtime.episode import EpisodeResult, EpisodeStep
from sim.robocasa_benchmark import (
    RoboCasaBenchmarkManifest,
    RoboCasaRolloutResult,
    aggregate_parallel_batch_results,
    aggregate_results,
    build_manifest,
    build_parallel_episode_manifest,
    evaluate_manifest,
)


REGISTRY = {
    "tiny": ["CoffeeSetupMug", "PnPCounterToCab"],
    "reordered": ["PnPCounterToCab", "CoffeeSetupMug"],
}


def _manifest(task_set: str = "tiny", scenarios_per_task: int = 2):
    return build_manifest(
        task_set,
        "target",
        scenarios_per_task=scenarios_per_task,
        master_seed=7,
        task_registry=REGISTRY,
        horizon_lookup=lambda task: 500 if task.startswith("Coffee") else 300,
        robocasa_version="test",
        robocasa_commit="deadbeef",
    )


def test_manifest_is_deterministic_and_task_order_independent(tmp_path) -> None:
    first = _manifest()
    second = _manifest("reordered")
    first_seeds = {s.task: [] for s in first.scenarios}
    second_seeds = {s.task: [] for s in second.scenarios}
    for scenario in first.scenarios:
        first_seeds[scenario.task].append(scenario.seed)
    for scenario in second.scenarios:
        second_seeds[scenario.task].append(scenario.seed)
    assert first_seeds == second_seeds
    assert first.task_count == 2
    assert first.rollout_count == 4
    assert first.scenarios[0].env_id.startswith("openeta/robocasa_target_")

    path = tmp_path / "manifest.json"
    first.write_json(path)
    assert RoboCasaBenchmarkManifest.read_json(path).to_dict() == first.to_dict()


def test_manifest_checksum_detects_mutation() -> None:
    payload = _manifest().to_dict()
    payload["scenarios"][0]["seed"] += 1
    with pytest.raises(ValueError, match="checksum"):
        RoboCasaBenchmarkManifest.from_dict(payload)


def test_manifest_adapts_to_shared_parallel_episode_contract(tmp_path) -> None:
    from agent.cli.batch_eval import load_parallel_episode_manifest

    manifest = _manifest(scenarios_per_task=1)
    native_payload = manifest.to_dict()
    assert "episodes" not in native_payload

    output = tmp_path / "parallel.json"
    manifest.write_parallel_json(
        output,
        task_resolver=lambda scenario: f"perform {scenario.task}",
        episode_limits={"max_turns": 12, "timeout_s": 45.0},
        metadata={"campaign": "contract-test", "require_official_reward": False},
    )
    specs = load_parallel_episode_manifest(output)

    assert [spec.episode_id for spec in specs] == [
        scenario.scenario_id for scenario in manifest.scenarios
    ]
    assert specs[0].task == f"perform {manifest.scenarios[0].task}"
    assert specs[0].env_id == manifest.scenarios[0].env_id
    assert specs[0].seed == manifest.scenarios[0].seed
    assert specs[0].max_turns == 12
    assert specs[0].timeout_s == 45.0
    assert specs[0].metadata == {
        "campaign": "contract-test",
        "benchmark": "robocasa365",
        "task_class": manifest.scenarios[0].task,
        "task_set": manifest.task_set,
        "split": manifest.scenarios[0].split,
        "scenario_index": manifest.scenarios[0].scenario_index,
        "horizon": manifest.scenarios[0].horizon,
        "manifest_sha256": native_payload["manifest_sha256"],
        "require_official_reward": True,
    }


def test_parallel_manifest_rejects_unknown_limits_and_empty_tasks() -> None:
    manifest = _manifest(scenarios_per_task=1)
    with pytest.raises(ValueError, match="Unsupported ParallelEpisodeSpec"):
        build_parallel_episode_manifest(
            manifest,
            episode_limits={"max_parallelism": 2},
        )
    with pytest.raises(ValueError, match="empty instruction"):
        build_parallel_episode_manifest(
            manifest,
            task_resolver=lambda _scenario: "",
        )


def test_evaluate_manifest_checkpoints_and_resumes(tmp_path) -> None:
    manifest = _manifest(scenarios_per_task=1)
    output = tmp_path / "results.json"
    calls: list[str] = []

    def runner(scenario):
        calls.append(scenario.scenario_id)
        return RoboCasaRolloutResult(
            scenario_id=scenario.scenario_id,
            task=scenario.task,
            split=scenario.split,
            seed=scenario.seed,
            success=scenario.task.startswith("Coffee"),
            steps=12,
        )

    partial = evaluate_manifest(
        manifest, runner, output_path=output, max_rollouts=1
    )
    assert partial["completed_rollouts"] == 1
    assert partial["complete"] is False

    complete = evaluate_manifest(manifest, runner, output_path=output)
    assert complete["completed_rollouts"] == 2
    assert complete["success_rate"] == 0.5
    assert complete["complete"] is True
    assert len(calls) == 2
    assert json.loads(output.read_text())["manifest_sha256"] == manifest.to_dict()[
        "manifest_sha256"
    ]


def test_aggregate_rejects_duplicate_results() -> None:
    manifest = _manifest(scenarios_per_task=1)
    scenario = manifest.scenarios[0]
    result = RoboCasaRolloutResult(
        scenario_id=scenario.scenario_id,
        task=scenario.task,
        split=scenario.split,
        seed=scenario.seed,
        success=False,
        steps=1,
    )
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_results(manifest, [result, result])


def _parallel_outcome(scenario, *, status: str, trusted_success: bool = False):
    execution_id = f"execution-{scenario.scenario_index}"
    reward = 1.0 if trusted_success else 0.0
    info = {}
    if trusted_success:
        info = {
            "environment_receipt_trusted": True,
            "official_reward": True,
            "environment_receipt": {
                "schema_version": "openeta.environment_receipt.v1",
                "execution_id": execution_id,
            },
        }
    return {
        "episode_id": scenario.scenario_id,
        "status": status,
        "episode": {
            "task": scenario.task,
            "num_steps": 3,
            "metadata": {"execution_id": execution_id},
            "steps": [
                {
                    "step_result": {
                        "reward": reward,
                        "terminated": trusted_success,
                        "truncated": False,
                        "info": info,
                        "observation": {
                            "metadata": {
                                "benchmark": {"elapsed_steps": 17}
                            }
                        },
                    }
                }
            ],
        },
        "cleanup": {"ok": True},
        "assistance": {"assisted": False},
        "error": {},
    }


def test_parallel_batch_results_convert_to_native_summary() -> None:
    manifest = _manifest(scenarios_per_task=1)
    success_scenario, failed_scenario = manifest.scenarios
    payload = {
        "schema_version": "openeta.parallel_episode_batch.v2",
        "batch_id": "batch-robocasa",
        "outcomes": [
            _parallel_outcome(
                success_scenario,
                status="success",
                trusted_success=True,
            ),
            _parallel_outcome(failed_scenario, status="fail"),
        ],
    }

    summary = aggregate_parallel_batch_results(manifest, payload)

    assert summary["schema_version"] == "openeta.robocasa_benchmark_result.v1"
    assert summary["complete"] is True
    assert summary["successes"] == 1
    assert summary["failures"] == 1
    assert summary["rollouts"][0]["steps"] == 17
    assert summary["rollouts"][0]["metadata"]["parallel_status"] == "success"


def test_parallel_success_without_trusted_official_receipt_is_rejected() -> None:
    manifest = _manifest(scenarios_per_task=1)
    payload = {
        "schema_version": "openeta.parallel_episode_batch.v2",
        "outcomes": [
            _parallel_outcome(manifest.scenarios[0], status="success"),
        ],
    }

    with pytest.raises(ValueError, match="no trusted official reward receipt"):
        aggregate_parallel_batch_results(manifest, payload)


def test_parallel_summary_accepts_live_validated_compacted_episode_receipt() -> None:
    manifest = _manifest(scenarios_per_task=1)
    scenario = manifest.scenarios[0]
    execution_id = "execution-from-real-episode"
    observation = EnvObservation(
        task=scenario.task,
        cameras=[],
        robot=RobotState(),
        metadata={"benchmark": {"elapsed_steps": 4}},
    )
    receipt = {
        "schema_version": "openeta.environment_receipt.v1",
        "receipt_id": "receipt-1",
        "backend": "simulator_mcp_episode_environment",
        "agent_tool": "environment_step",
        "remote_tool": "render_env",
        "execution_id": execution_id,
        "agent_session_id": "agent-1",
        "handle": "env-1",
        "observation_fresh": True,
        "reward_present": True,
        "reward": 1.0,
        "simulator_session_id": "sim-1",
        "terminated": True,
        "truncated": False,
    }
    episode = EpisodeResult(
        task=scenario.task,
        session_id="agent-1",
        steps=[
            EpisodeStep(
                turn_index=0,
                observation=observation,
                action=EnvAction(action_type="tool_call"),
                step_result=StepResult(
                    observation=observation,
                    reward=1.0,
                    terminated=True,
                    info={
                        "environment_receipt_trusted": True,
                        "official_reward": True,
                        "environment_receipt": receipt,
                    },
                ),
            )
        ],
        terminated=True,
        metadata={"execution_id": execution_id},
    )
    serialized_episode = episode.to_dict()
    serialized_receipt = serialized_episode["steps"][0]["step_result"]["info"][
        "environment_receipt"
    ]
    assert "schema_version" not in serialized_receipt

    summary = aggregate_parallel_batch_results(
        manifest,
        {
            "schema_version": "openeta.parallel_episode_batch.v2",
            "batch_id": "batch-compacted-receipt",
            "outcomes": [
                {
                    "episode_id": scenario.scenario_id,
                    "status": "success",
                    "episode": serialized_episode,
                    "cleanup": {"ok": True},
                    "assistance": {"assisted": False},
                    "error": {},
                }
            ],
        },
    )

    assert summary["successes"] == 1


def test_aggregate_rejects_result_identity_mismatch() -> None:
    manifest = _manifest(scenarios_per_task=1)
    scenario = manifest.scenarios[0]
    result = RoboCasaRolloutResult(
        scenario_id=scenario.scenario_id,
        task="WrongTask",
        split=scenario.split,
        seed=scenario.seed,
        success=False,
        steps=1,
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        aggregate_results(manifest, [result])


def test_robocasa_unified_observation_is_calibrated_metric_rgbd(monkeypatch) -> None:
    from adapter.protocol import EnvObservation
    from sim.unified_env import UnifiedEnv

    wrapper = object.__new__(UnifiedEnv)
    wrapper._backend = "robocasa"
    wrapper._include_objects = False
    monkeypatch.setattr(wrapper, "_depth_to_metres", lambda value: np.asarray(value))
    monkeypatch.setattr(
        wrapper,
        "_depth_to_metres",
        lambda value: np.asarray(value, dtype=np.float32) * 2.0,
    )
    monkeypatch.setattr(
        wrapper,
        "_robocasa_camera_params",
        lambda name, image_width, image_height: {
            "fx": 120.0,
            "fy": 121.0,
            "cx": 1.0,
            "cy": 0.5,
            "width": image_width,
            "height": image_height,
            "extrinsics": {
                "pos": [0.0, 0.0, 1.0],
                "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "matrix_layout": "row_major",
                "frame_transform": "camera_to_world",
                "camera_frame": "opengl",
            },
        },
    )

    source_rgb = np.asarray(
        [
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            [[20, 21, 22], [23, 24, 25], [26, 27, 28]],
        ],
        dtype=np.uint8,
    )
    source_depth = np.array(
        [[[0.25], [np.nan], [0.75]], [[np.inf], [-1.0], [0.5]]],
        dtype=np.float32,
    )
    unified = wrapper._normalise_robocasa(
        {
            "robot0_robotview_image": source_rgb,
            "robot0_robotview_depth": source_depth,
            "robot0_joint_pos": np.arange(7, dtype=np.float32),
            "robot0_joint_vel": np.ones(7, dtype=np.float32),
            "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }
    )
    observation = EnvObservation.from_dict(unified, task="move the mug")

    camera = observation.cameras[0]
    assert camera.frame_id == "agentview"
    assert camera.intrinsics["fx"] == 120.0
    assert camera.intrinsics["depth_unit"] == "meter"
    assert camera.extrinsics["camera_frame"] == "opencv"
    assert camera.extrinsics["matrix_layout"] == "row_major"
    assert camera.extrinsics["image_origin"] == "top_left"
    assert camera.extrinsics["raw_camera_convention"] == "opengl"
    assert camera.extrinsics["normalized_from"] == "mujoco_opengl"
    np.testing.assert_allclose(
        np.asarray(camera.extrinsics["mat"]).reshape(3, 3),
        np.diag([1.0, -1.0, -1.0]),
    )
    np.testing.assert_allclose(
        camera.extrinsics["camera_to_world"],
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    np.testing.assert_array_equal(np.asarray(camera.rgb), np.flipud(source_rgb))
    np.testing.assert_array_equal(
        np.asarray(camera.depth),
        np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 1.5]], dtype=np.float32),
    )
    assert observation.robot.end_effector_pose["xyz"] == pytest.approx([0.1, 0.2, 0.3])

    import base64
    import io

    from PIL import Image

    mcp_camera = observation.to_mcp_dict()["cameras"][0]
    assert mcp_camera["intrinsics"]["depth_unit"] == "meter"
    assert mcp_camera["extrinsics"]["camera_frame"] == "opencv"
    assert mcp_camera["extrinsics"]["image_origin"] == "top_left"
    assert mcp_camera["depth_encoding"] == "uint16_png"
    assert mcp_camera["depth_scale"] == 1000.0
    with Image.open(io.BytesIO(base64.b64decode(mcp_camera["rgb_base64"]))) as image:
        np.testing.assert_array_equal(np.asarray(image), np.flipud(source_rgb))
    with Image.open(io.BytesIO(base64.b64decode(mcp_camera["depth_base64"]))) as image:
        np.testing.assert_array_equal(
            np.asarray(image),
            np.array([[0, 0, 1000], [500, 0, 1500]], dtype=np.uint16),
        )

    opencv_unified = wrapper._normalise_robocasa(
        {
            "_openeta_image_convention": "opencv",
            "robot0_robotview_image": source_rgb,
            "robot0_robotview_depth": source_depth,
        }
    )
    opencv_camera = EnvObservation.from_dict(opencv_unified).cameras[0]
    np.testing.assert_array_equal(np.asarray(opencv_camera.rgb), source_rgb)
    np.testing.assert_array_equal(
        np.asarray(opencv_camera.depth),
        np.array([[0.5, 0.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float32),
    )


def test_robocasa_direct_freezes_image_convention_for_annotation_and_render() -> None:
    from types import SimpleNamespace

    from sim.envs.robocasa.direct_env import RoboCasaDirectEnv

    source = np.asarray(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ],
        dtype=np.uint8,
    )
    direct = object.__new__(RoboCasaDirectEnv)
    direct._env = SimpleNamespace(get_ep_meta=lambda: {"lang": "test task"})
    direct._camera_names = ["robot0_robotview"]
    direct._last_raw_obs = {"robot0_robotview_image": source}
    direct._image_convention = "opencv"
    direct.task_name = "PnPCounterToCab"
    direct.split = "target"
    direct.scenario_seed = 3
    direct.horizon = 10
    direct.elapsed_steps = 0
    direct.robot = "Panda"
    direct.action_space = SimpleNamespace(shape=(7,))

    annotated = direct._annotate(
        {
            "robot0_robotview_image": source,
            "robot0_robotview_depth": np.ones(source.shape[:2], dtype=np.float32),
        }
    )

    assert annotated["_openeta_image_convention"] == "opencv"
    np.testing.assert_array_equal(direct.render(), source)

    direct._image_convention = "opengl"
    np.testing.assert_array_equal(direct.render(), np.flipud(source))

    direct._robosuite_macros = SimpleNamespace(IMAGE_CONVENTION="opencv")
    with pytest.raises(RuntimeError, match="changed after"):
        direct.render()


def test_robocasa_official_camera_packet_fails_closed_without_rgbd_calibration(
    monkeypatch,
) -> None:
    from sim.envs.robocasa.direct_env import RoboCasaDirectEnv
    from sim.unified_env import UnifiedEnv

    with pytest.raises(ValueError, match="camera_depths=True"):
        RoboCasaDirectEnv("PnPCounterToCab", camera_depths=False)

    direct = object.__new__(RoboCasaDirectEnv)
    direct._camera_names = ["robot0_robotview", "robot0_eye_in_hand"]
    with pytest.raises(RuntimeError, match="missing configured RGB-D"):
        direct._annotate({})

    wrapper = object.__new__(UnifiedEnv)
    wrapper._backend = "robocasa"
    wrapper._include_objects = False
    monkeypatch.setattr(wrapper, "_depth_to_metres", lambda value: np.asarray(value))
    monkeypatch.setattr(
        wrapper,
        "_robocasa_camera_params",
        lambda *_args, **_kwargs: {},
    )

    with pytest.raises(RuntimeError, match="missing metric depth"):
        wrapper._normalise_robocasa(
            {
                "robot0_robotview_image": np.zeros((2, 3, 3), dtype=np.uint8),
            }
        )

    with pytest.raises(RuntimeError, match="missing aligned intrinsics"):
        wrapper._normalise_robocasa(
            {
                "robot0_robotview_image": np.zeros((2, 3, 3), dtype=np.uint8),
                "robot0_robotview_depth": np.ones((2, 3), dtype=np.float32),
            }
        )


def test_robocasa_camera_calibration_requires_exact_name_and_live_pose(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from sim.unified_env import UnifiedEnv

    class FakeModel:
        ncam = 1
        cam_fovy = np.array([60.0])
        stat = SimpleNamespace(extent=2.0)
        vis = SimpleNamespace(map=SimpleNamespace(znear=0.01, zfar=10.0))
        # Deliberately different static values: the strict path must not use
        # these in place of the live MjData pose.
        cam_pos = np.array([[90.0, 91.0, 92.0]])
        cam_mat = np.eye(3).reshape(1, 9)

        @staticmethod
        def camera_name2id(name):
            if name == "invalid_id":
                return 7
            if name != "robot0_robotview":
                raise ValueError(name)
            return 0

    live_position = np.array([[1.0, 2.0, 3.0]])
    live_rotation = np.array(
        [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    wrapper = object.__new__(UnifiedEnv)
    monkeypatch.setattr(wrapper, "_find_mj_model", lambda: FakeModel())
    monkeypatch.setattr(
        wrapper,
        "_find_mj_data",
        lambda: SimpleNamespace(
            cam_xpos=live_position,
            cam_xmat=live_rotation,
        ),
    )

    params = wrapper._robocasa_camera_params(
        "robot0_robotview",
        image_width=640,
        image_height=480,
    )

    assert params["extrinsics"]["pos"] == [1.0, 2.0, 3.0]
    assert 0.0 < params["znear"] < params["zfar"]
    assert params["fy"] > 0.0
    np.testing.assert_allclose(
        np.asarray(params["extrinsics"]["mat"]).reshape(3, 3),
        live_rotation[0],
    )
    with pytest.raises(RuntimeError, match="not in the MuJoCo model"):
        wrapper._robocasa_camera_params(
            "unknown_camera",
            image_width=640,
            image_height=480,
        )
    with pytest.raises(RuntimeError, match="invalid id"):
        wrapper._robocasa_camera_params(
            "invalid_id",
            image_width=640,
            image_height=480,
        )

    monkeypatch.setattr(wrapper, "_find_mj_data", lambda: None)
    with pytest.raises(RuntimeError, match="no live MuJoCo data"):
        wrapper._robocasa_camera_params(
            "robot0_robotview",
            image_width=640,
            image_height=480,
        )
