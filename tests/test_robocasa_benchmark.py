from __future__ import annotations

import json

import numpy as np
import pytest

from sim.robocasa_benchmark import (
    RoboCasaBenchmarkManifest,
    RoboCasaRolloutResult,
    aggregate_results,
    build_manifest,
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


def test_robocasa_unified_observation_is_calibrated_metric_rgbd(monkeypatch) -> None:
    from adapter.protocol import EnvObservation
    from sim.unified_env import UnifiedEnv

    wrapper = object.__new__(UnifiedEnv)
    wrapper._include_objects = False
    monkeypatch.setattr(
        wrapper,
        "_depth_to_metres",
        lambda value: np.asarray(value, dtype=np.float32) * 2.0,
    )
    monkeypatch.setattr(
        wrapper,
        "_extract_camera_params",
        lambda name, image_width, image_height: {
            "intrinsics": {
                "fx": 120.0,
                "fy": 121.0,
                "cx": 1.0,
                "cy": 0.5,
                "width": image_width,
                "height": image_height,
            },
            "extrinsics": {
                "pos": [0.0, 0.0, 1.0],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                "frame_transform": "camera_to_world",
            },
        },
    )

    unified = wrapper._normalise_robocasa(
        {
            "robot0_robotview_image": np.zeros((2, 3, 3), dtype=np.uint8),
            "robot0_robotview_depth": np.array(
                [[[0.25], [np.nan], [0.75]], [[np.inf], [-1.0], [0.5]]],
                dtype=np.float32,
            ),
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
    assert camera.extrinsics["quat_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    np.testing.assert_array_equal(
        np.asarray(camera.depth),
        np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 1.5]], dtype=np.float32),
    )
    assert observation.robot.end_effector_pose["xyz"] == pytest.approx([0.1, 0.2, 0.3])
