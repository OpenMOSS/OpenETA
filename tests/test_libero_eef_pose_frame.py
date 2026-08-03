from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from adapter.protocol import EnvObservation
from sim.unified_env import UnifiedEnv, _rotation_matrix_to_quat_xyzw


def _unified_with_site_matrix(matrix: np.ndarray | None) -> UnifiedEnv:
    unified = UnifiedEnv.__new__(UnifiedEnv)
    unified._include_objects = False
    if matrix is None:
        unified._env = SimpleNamespace()
        return unified

    sim = SimpleNamespace(
        data=SimpleNamespace(site_xmat=np.asarray([matrix], dtype=np.float64))
    )
    robot = SimpleNamespace(eef_site_id=0)
    offscreen = SimpleNamespace(sim=sim, env=SimpleNamespace(robots=[robot]))
    unified._env = SimpleNamespace(_env=offscreen)
    return unified


def _raw_libero_pose(*, body_quat: list[float]) -> dict:
    return {
        "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
        "robot0_eef_quat": np.asarray(body_quat, dtype=np.float64),
    }


def test_rotation_matrix_to_quat_xyzw_uses_xyzw_order() -> None:
    rotation_z_90 = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    quat = _rotation_matrix_to_quat_xyzw(rotation_z_90)

    assert quat is not None
    np.testing.assert_allclose(
        quat,
        [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
        atol=1e-7,
    )


def test_libero_ee_pose_uses_grip_site_position_and_orientation() -> None:
    unified = _unified_with_site_matrix(np.eye(3))
    raw = _raw_libero_pose(
        body_quat=[0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]
    )

    observation = unified._normalise_libero(raw)

    np.testing.assert_allclose(
        observation["proprio"]["ee_pose"],
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        atol=1e-7,
    )
    assert observation["proprio"]["metadata"] == {
        "ee_position_frame": "grip_site",
        "ee_orientation_frame": "grip_site",
        "ee_pose_frame_consistent": True,
        "right_hand_quat_xyzw": raw["robot0_eef_quat"].tolist(),
    }


def test_libero_ee_pose_fails_closed_without_grip_site_orientation() -> None:
    unified = _unified_with_site_matrix(None)
    body_quat = [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]

    observation = unified._normalise_libero(_raw_libero_pose(body_quat=body_quat))

    np.testing.assert_allclose(
        observation["proprio"]["ee_pose"],
        [0.1, 0.2, 0.3],
        atol=1e-7,
    )
    assert observation["proprio"]["metadata"] == {
        "ee_position_frame": "grip_site",
        "ee_orientation_frame": None,
        "ee_pose_frame_consistent": False,
        "right_hand_quat_xyzw": body_quat,
    }


def test_grip_site_pose_survives_mcp_serialization() -> None:
    unified = _unified_with_site_matrix(np.eye(3))
    observation = unified._normalise_libero(
        _raw_libero_pose(body_quat=[0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    )

    payload = EnvObservation.from_dict(
        {
            "task_description": "test",
            "cameras": {},
            "proprio": observation["proprio"],
        }
    ).to_mcp_dict()

    assert payload["robot"]["end_effector_pose"] == {
        "xyz": [0.1, 0.2, 0.3],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    assert payload["robot"]["metadata"]["ee_pose_frame_consistent"] is True


def test_libero_native_finger_state_survives_mcp_serialization() -> None:
    unified = _unified_with_site_matrix(np.eye(3))
    raw = _raw_libero_pose(body_quat=[0.0, 0.0, 0.0, 1.0])
    raw["robot0_gripper_qpos"] = np.asarray([0.012, -0.011])
    raw["robot0_gripper_qvel"] = np.asarray([-0.001, 0.002])

    observation = unified._normalise_libero(raw)
    payload = EnvObservation.from_dict(
        {
            "task_description": "test",
            "cameras": {},
            "proprio": observation["proprio"],
        }
    ).to_mcp_dict()

    assert payload["robot"]["gripper_state"] == {
        "open": False,
        "finger_qpos": [0.012, -0.011],
        "aperture_m": 0.023,
        "finger_qvel": [-0.001, 0.002],
    }


def test_libero_step_publishes_only_current_near_gripper_mujoco_contacts() -> None:
    geom_names = {
        0: "gripper0_finger1_pad_collision",
        1: "cream_cheese_collision",
        2: "basket_collision",
        3: "floor",
    }
    model = SimpleNamespace(
        cam_fovy=np.asarray([]),
        ncam=0,
        site_name2id=lambda _name: 0,
        geom_id2name=lambda geom_id: geom_names[geom_id],
    )
    data = SimpleNamespace(
        cam_xpos=np.empty((0, 3)),
        cam_xmat=np.empty((0, 9)),
        site_xpos=np.asarray([[0.0, 0.0, 0.5]]),
        ncon=4,
        contact=[
            SimpleNamespace(
                pos=np.asarray([0.0, 0.0, 0.5]),
                geom1=0,
                geom2=1,
            ),
            SimpleNamespace(
                pos=np.asarray([0.001, 0.0, 0.5]),
                geom1=0,
                geom2=1,
            ),
            SimpleNamespace(
                pos=np.asarray([0.05, 0.0, 0.5]),
                geom1=1,
                geom2=2,
            ),
            SimpleNamespace(
                pos=np.asarray([1.0, 0.0, 0.0]),
                geom1=2,
                geom2=3,
            ),
        ],
    )

    class FakeLibero:
        _task_description = "test"

        def __init__(self) -> None:
            self.model = model
            self.data = data

        def step(self, _action):
            return {}, 0.0, False, False, {}

    unified = UnifiedEnv.__new__(UnifiedEnv)
    unified._env = FakeLibero()
    unified._backend = "libero"
    unified._include_objects = False

    observation, *_ = unified.step([0.0] * 7)

    contacts = observation["metadata"]["mujoco_contacts"]
    assert contacts["source"] == "mujoco_data.contact"
    assert contacts["points"] == [
        {
            "position_xyz_m": [0.0, 0.0, 0.5],
            "geom_names": [
                "gripper0_finger1_pad_collision",
                "cream_cheese_collision",
            ],
        },
        {
            "position_xyz_m": [0.05, 0.0, 0.5],
            "geom_names": [
                "cream_cheese_collision",
                "basket_collision",
            ],
        },
    ]
