from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("anyio")

from sim.envs.behavior.direct_env import (
    BehaviorDirectEnv,
    _configure_agent_cartesian_control,
)
from sim.mcp_server import collision, server, session
from sim.mcp_server.action_codecs import (
    ControlCodecError,
    cartesian_scales,
    make_cartesian_action,
    make_gripper_action,
    require_controller_capability,
)
from sim.env_registry import _LibEnvWrapper
from sim.controllers import mink_goal
from sim.controllers.collision_recovery import (
    project_velocity_to_joint_limits,
    verified_collision_boundary_escape,
    verified_joint_limit_escape,
)
from sim.controllers.dependency_overlay import validate_mink_dependency_overlay


BEHAVIOR_META = {
    "action_dim": 18,
    "control_spec": {
        "schema_version": "openeta.sim_control.v1",
        "cartesian_delta": {
            "supported": True,
            "position_indices": [7, 8, 9],
            "rotation_indices": [10, 11, 12],
            "command_frame": "robot_base",
            "position_scale_m": 0.05,
            "rotation_scale_rad": 0.25,
        },
        "gripper": {
            "supported": True,
            "indices": [13],
            "open_value": 1.0,
            "close_value": -1.0,
        },
    },
}

LIBERO_CONTROL_SPEC = {
    "schema_version": "openeta.sim_control.v1",
    "controller": {
        "controller_id": "robosuite.osc_pose",
        "configured_name": "OSC_POSE",
        "command_interface": "normalized_cartesian_delta_pose",
        "goal_executor": "openeta.outer_closed_loop_cartesian.v1",
        "execution_location": "mcp_server",
        "supports_position": True,
        "supports_orientation": True,
    },
    "cartesian_delta": {
        "supported": True,
        "position_indices": [0, 1, 2],
        "rotation_indices": [3, 4, 5],
        "command_frame": "world",
        "position_scale_m": 0.05,
        "rotation_scale_rad": 0.5,
    },
    "gripper": {
        "supported": True,
        "indices": [6],
        "open_value": -1.0,
        "close_value": 1.0,
    },
}


def _libero_meta() -> dict:
    return {
        "backend": "libero",
        "action_dim": 7,
        "remote_handle": "remote",
        "control_spec": LIBERO_CONTROL_SPEC,
    }


def test_libero_cartesian_scales_match_robosuite_osc_pose_contract() -> None:
    assert cartesian_scales({}, "libero") == (0.05, 0.5)


def test_mink_penetration_escape_requires_monotonic_progress_without_new_collision() -> None:
    current = {(1, 2): -0.012, (3, 4): 0.01}

    assert verified_collision_boundary_escape(
        current,
        {(1, 2): -0.010, (3, 4): 0.009},
        hard_stop_distance_m=-0.001,
    )
    assert not verified_collision_boundary_escape(
        current,
        {(1, 2): -0.013, (3, 4): 0.009},
        hard_stop_distance_m=-0.001,
    )
    assert not verified_collision_boundary_escape(
        current,
        {(1, 2): -0.010, (3, 4): -0.002},
        hard_stop_distance_m=-0.001,
    )


def test_mink_escape_can_recover_from_active_margin_without_crossing_hard_stop() -> None:
    current = {(1, 2): 0.0026, (3, 4): 0.01}

    assert verified_collision_boundary_escape(
        current,
        {(1, 2): 0.0028, (3, 4): 0.009},
        hard_stop_distance_m=-0.001,
        recovery_boundary_distance_m=0.003,
    )


def test_mink_emergency_escape_must_preserve_or_repair_joint_limits() -> None:
    assert verified_joint_limit_escape(
        [1.01, 0.0],
        [1.00, 0.1],
        [-1.0, -1.0],
        [1.0, 1.0],
    )
    assert not verified_joint_limit_escape(
        [1.01, 0.0],
        [1.02, 0.1],
        [-1.0, -1.0],
        [1.0, 1.0],
    )


def test_mink_emergency_velocity_projects_only_outward_joint_components() -> None:
    projected, clipped = project_velocity_to_joint_limits(
        [0.2, 0.5, 0.3],
        [0.0, 0.99, -1.01],
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        dt=0.05,
    )

    assert clipped == [1]
    assert projected[0] == pytest.approx(0.2)
    assert projected[1] == pytest.approx((1.0 - 1e-6 - 0.99) / 0.05)
    assert projected[2] == pytest.approx(0.3)
    assert verified_joint_limit_escape(
        [0.0, 0.99, -1.01],
        [
            0.0 + projected[0] * 0.05,
            0.99 + projected[1] * 0.05,
            -1.01 + projected[2] * 0.05,
        ],
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
    )


def test_mink_dependency_overlay_rejects_runtime_package_shadowing(tmp_path) -> None:
    minimal = tmp_path / "minimal"
    minimal.mkdir()
    (minimal / "mink").mkdir()
    (minimal / "qpsolvers").mkdir()
    assert validate_mink_dependency_overlay(minimal) == minimal.resolve()

    broad = tmp_path / "broad"
    broad.mkdir()
    (broad / "mink").mkdir()
    (broad / "mujoco").mkdir()
    (broad / "numpy").mkdir()
    with pytest.raises(RuntimeError) as rejected:
        validate_mink_dependency_overlay(broad)

    assert "minimal overlay" in str(rejected.value)
    assert "mujoco, numpy" in str(rejected.value)
    assert "sim/venvs/libero" in str(rejected.value)


def test_libero_wrapper_declares_actual_osc_controller_contract() -> None:
    wrapper = object.__new__(_LibEnvWrapper)
    wrapper._controller = "OSC_POSE"

    assert wrapper.openeta_control_spec == LIBERO_CONTROL_SPEC


def test_libero_wrapper_declares_worker_local_mink_contract() -> None:
    wrapper = object.__new__(_LibEnvWrapper)
    wrapper._controller = "JOINT_VELOCITY"
    wrapper._controller_profile = "mink_joint_velocity"

    spec = wrapper.openeta_control_spec

    assert spec["controller"] == {
        "controller_id": "mink.robosuite_joint_velocity",
        "configured_name": "JOINT_VELOCITY",
        "command_interface": "joint_velocity",
        "goal_executor": "openeta.worker_mink_goal.v1",
        "execution_location": "bench_worker",
        "supports_position": True,
        "supports_orientation": True,
        "collision_callback": True,
        "collision_scope": "worker_per_step_pre_actuation_and_post_step_configuration",
        "intentional_contact_policy": "host_compiled_grasp_target_gripper_subtree_only",
        "contact_validated": True,
        "contact_validation_scope": (
            "libero_task2_seed2_approach_contact_close_lift_canary"
        ),
        "attached_object_trajectory_coverage": (
            "worker_per_step_predicted_and_actual_live_geometry"
        ),
    }
    assert spec["cartesian_delta"] == {"supported": False}
    assert spec["gripper"]["indices"] == [7]


def test_libero_mink_wrapper_declares_actual_eight_dimensional_action_space() -> None:
    raw = SimpleNamespace(observation_space=SimpleNamespace())
    wrapper = _LibEnvWrapper(
        raw,
        controller="JOINT_VELOCITY",
        controller_profile="mink_joint_velocity",
    )

    assert wrapper.action_space.shape == (8,)


def test_libero_controller_contract_fails_closed_without_silent_osc_fallback() -> None:
    with pytest.raises(ControlCodecError) as missing:
        require_controller_capability({}, "libero", orientation_requested=False)
    assert missing.value.code == "controller_capability_missing"

    mink_spec = {
        **LIBERO_CONTROL_SPEC,
        "controller": {
            **LIBERO_CONTROL_SPEC["controller"],
            "controller_id": "mink.robosuite_joint_velocity",
            "configured_name": "JOINT_VELOCITY",
            "command_interface": "joint_velocity",
            "goal_executor": "openeta.worker_mink_goal.v1",
            "execution_location": "bench_worker",
        },
        "cartesian_delta": {"supported": False},
    }
    declared = require_controller_capability(
        {"control_spec": mink_spec},
        "libero",
        orientation_requested=True,
    )
    assert declared["goal_executor"] == "openeta.worker_mink_goal.v1"

    mink_spec["controller"]["goal_executor"] = "unknown_executor"
    with pytest.raises(ControlCodecError) as mismatch:
        require_controller_capability(
            {"control_spec": mink_spec}, "libero", orientation_requested=True
        )
    assert mismatch.value.code == "controller_capability_mismatch"
    assert "no OSC fallback was attempted" in str(mismatch.value)


def test_move_to_dispatches_mink_goal_to_worker_without_outer_osc_steps(
    monkeypatch,
) -> None:
    control_spec = {
        "schema_version": "openeta.sim_control.v1",
        "controller": {
            "controller_id": "mink.robosuite_joint_velocity",
            "configured_name": "JOINT_VELOCITY",
            "command_interface": "joint_velocity",
            "goal_executor": "openeta.worker_mink_goal.v1",
            "execution_location": "bench_worker",
            "supports_position": True,
            "supports_orientation": True,
        },
        "cartesian_delta": {"supported": False},
    }
    meta = {
        "backend": "libero",
        "action_dim": 7,
        "remote_handle": "remote",
        "control_spec": control_spec,
    }
    calls: list[dict] = []
    monkeypatch.setattr(server, "_session_envs", {"sid": {"handle": meta}})
    monkeypatch.setattr(server, "_touch_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_proxy_controller_goal",
        lambda _meta, body: calls.append(body) or {"reached_target": True},
    )
    monkeypatch.setattr(
        server,
        "_proxy_step",
        lambda *_args, **_kwargs: pytest.fail("outer OSC step must not run"),
    )

    result = server.move_to.__wrapped__(
        "handle",
        0.1,
        0.2,
        0.3,
        num_steps=40,
        tolerance=0.003,
        enable_collision_check=False,
        session_id="sid",
    )

    assert result == {"reached_target": True}
    assert calls == [
        {
            "target_xyz": [0.1, 0.2, 0.3],
            "preserve_current_orientation": True,
            "max_steps": 40,
            "position_tolerance_m": 0.003,
            "orientation_tolerance_rad": 0.05,
                "gripper_command": 0.0,
                "enable_collision_check": False,
                "motion_execution_condition": "A",
            }
        ]


def test_move_to_forwards_private_ik_execution_seed_to_worker(monkeypatch) -> None:
    control_spec = {
        "schema_version": "openeta.sim_control.v1",
        "controller": {
            "controller_id": "mink.robosuite_joint_velocity",
            "configured_name": "JOINT_VELOCITY",
            "command_interface": "joint_velocity",
            "goal_executor": "openeta.worker_mink_goal.v1",
            "execution_location": "bench_worker",
            "supports_position": True,
            "supports_orientation": True,
        },
        "cartesian_delta": {"supported": False},
    }
    meta = {
        "backend": "libero",
        "action_dim": 7,
        "remote_handle": "remote",
        "control_spec": control_spec,
    }
    calls: list[dict] = []
    monkeypatch.setattr(server, "_session_envs", {"sid": {"handle": meta}})
    monkeypatch.setattr(server, "_touch_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_proxy_controller_goal",
        lambda _meta, body: calls.append(body) or {"reached_target": True},
    )
    seed = {
        "schema_version": "openeta.ik_execution_seed.v1",
        "receipt_id": "ik-1",
        "joint_positions": [0.1] * 7,
    }

    result = server.move_to.__wrapped__(
        "handle",
        0.1,
        0.2,
        0.3,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        ik_execution_seed=seed,
        session_id="sid",
    )

    assert result == {"reached_target": True}
    assert calls[0]["ik_execution_seed"] == seed


def test_mink_joint_seed_validation_binds_candidate_to_execution_tolerance(
    monkeypatch,
) -> None:
    robot = SimpleNamespace(_ref_joint_pos_indexes=np.arange(7))
    seed = {
        "schema_version": "openeta.ik_execution_seed.v1",
        "joint_positions": [0.1] * 7,
    }
    monkeypatch.setattr(
        mink_goal,
        "_configuration_eef_pose",
        lambda *_args, **_kwargs: (
            np.asarray([0.1005, 0.1995, 0.3005]),
            np.asarray([0.0, 0.0, 0.0, 1.0]),
        ),
    )
    monkeypatch.setattr(mink_goal, "_angular_error_rad", lambda *_args: 0.0)

    accepted = mink_goal._validated_explicit_pose_seed(
        SimpleNamespace(),
        np.zeros(7),
        robot,
        seed=seed,
        target_xyz=np.asarray([0.1, 0.2, 0.3]),
        target_quat_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
        position_tolerance_m=0.002,
        orientation_tolerance_rad=0.05,
    )
    assert np.allclose(accepted, [0.1] * 7)

    rejected = mink_goal._validated_explicit_pose_seed(
        SimpleNamespace(),
        np.zeros(7),
        robot,
        seed=seed,
        target_xyz=np.asarray([0.1, 0.2, 0.3]),
        target_quat_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
        position_tolerance_m=0.0001,
        orientation_tolerance_rad=0.05,
    )
    assert isinstance(rejected, str)
    assert rejected.startswith("ik_execution_seed_target_mismatch:")


def test_attached_object_prediction_preserves_rigid_transform_during_translation() -> None:
    q = np.arange(14, dtype=np.float64)
    q[4:7] = [0.2, -0.1, 0.3]
    q[7:11] = [1.0, 0.0, 0.0, 0.0]

    transformed = mink_goal._transform_attached_object_with_eef(
        q,
        {"attached_object_qpos_adr": 4},
        current_eef_xyz=np.asarray([0.1, -0.2, 0.25]),
        current_eef_quat_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
        predicted_eef_xyz=np.asarray([0.11, -0.22, 0.28]),
        predicted_eef_quat_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
    )

    assert np.allclose(transformed[4:7], q[4:7] + [0.01, -0.02, 0.03])
    assert np.allclose(transformed[7:11], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(transformed[:4], q[:4])
    assert np.allclose(transformed[11:], q[11:])
    assert np.allclose(q[4:11], [0.2, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0])


def test_attached_object_prediction_rotates_offset_and_orientation() -> None:
    q = np.zeros(9, dtype=np.float64)
    q[:3] = [0.1, 0.0, 0.0]
    q[3:7] = [1.0, 0.0, 0.0, 0.0]
    quarter_turn = np.sqrt(0.5)

    transformed = mink_goal._transform_attached_object_with_eef(
        q,
        {"attached_object_qpos_adr": 0},
        current_eef_xyz=np.zeros(3),
        current_eef_quat_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
        predicted_eef_xyz=np.asarray([0.0, 0.2, 0.0]),
        predicted_eef_quat_xyzw=np.asarray(
            [0.0, 0.0, quarter_turn, quarter_turn]
        ),
    )

    assert np.allclose(transformed[:3], [0.0, 0.3, 0.0], atol=1e-9)
    assert np.allclose(
        transformed[3:7],
        [quarter_turn, 0.0, 0.0, quarter_turn],
        atol=1e-9,
    )


def test_attached_object_rotation_prediction_exposes_obstacle_collision() -> None:
    q = np.zeros(7, dtype=np.float64)
    q[3] = 1.0
    quarter_turn = np.sqrt(0.5)
    transformed = mink_goal._transform_attached_object_with_eef(
        q,
        {"attached_object_qpos_adr": 0},
        current_eef_xyz=np.zeros(3),
        current_eef_quat_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
        predicted_eef_xyz=np.zeros(3),
        predicted_eef_quat_xyzw=np.asarray(
            [0.0, 0.0, quarter_turn, quarter_turn]
        ),
    )

    held_half_extent = np.asarray([0.10, 0.01, 0.01])
    obstacle_min = np.asarray([-0.02, 0.07, -0.02])
    obstacle_max = np.asarray([0.02, 0.11, 0.02])

    def overlaps_obstacle(configuration: np.ndarray) -> bool:
        center = configuration[:3]
        quat_wxyz = configuration[3:7]
        quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
        corners = np.asarray(
            [
                [sx * held_half_extent[0], sy * held_half_extent[1], sz * held_half_extent[2]]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ]
        )
        world_corners = np.asarray(
            [
                center
                + mink_goal._rotate_vector_by_quaternion_xyzw(quat_xyzw, corner)
                for corner in corners
            ]
        )
        held_min = world_corners.min(axis=0)
        held_max = world_corners.max(axis=0)
        return bool(np.all(held_max >= obstacle_min) and np.all(held_min <= obstacle_max))

    assert overlaps_obstacle(q) is False
    assert overlaps_obstacle(transformed) is True
    assert np.allclose(
        transformed[3:7],
        [quarter_turn, 0.0, 0.0, quarter_turn],
    )


def test_production_collision_distance_uses_contacts_for_orthogonal_boxes() -> None:
    mujoco = pytest.importorskip("mujoco")
    mink = pytest.importorskip("mink")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option gravity="0 0 0"/>
          <worldbody>
            <body name="held" pos="0 0 0">
              <freejoint/>
              <geom name="held_geom" type="box" size="0.10 0.01 0.01"/>
            </body>
            <body name="obstacle" pos="0 0.08 0">
              <geom name="obstacle_geom" type="box" size="0.02 0.02 0.02"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    half_angle = math.pi / 4.0
    qpos = model.qpos0.copy()
    qpos[:7] = [
        0.0,
        0.0,
        0.0,
        math.cos(half_angle),
        0.0,
        0.0,
        math.sin(half_angle),
    ]
    configuration = mink.Configuration(model, q=qpos)
    data = configuration.data
    assert data.ncon == 0
    held = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "held_geom",
    )
    obstacle = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "obstacle_geom",
    )

    fromto = np.empty(6, dtype=np.float64)
    raw_distance = float(
        mujoco.mj_geomDistance(model, data, held, obstacle, 0.02, fromto)
    )
    if mujoco.__version__ == "3.3.0":
        # This is the upstream degeneracy that originally let the collision
        # through OpenETA's -1 mm hard stop.
        assert raw_distance == pytest.approx(0.0)

    report = mink_goal._collision_distance_report(
        configuration,
        [(held, obstacle)],
        distance_limit_m=-0.001,
    )
    pair_distances = mink_goal._collision_pair_distances(
        configuration,
        [(held, obstacle)],
    )
    matching_contacts = [
        float(data.contact[index].dist)
        for index in range(data.ncon)
        if {
            int(data.contact[index].geom1),
            int(data.contact[index].geom2),
        }
        == {held, obstacle}
    ]

    assert matching_contacts
    assert min(matching_contacts) < -0.01
    assert report["detected"] is True
    assert report["minimum_distance_m"] == pytest.approx(
        min(raw_distance, *matching_contacts)
    )
    assert pair_distances[(held, obstacle)] == pytest.approx(
        report["minimum_distance_m"]
    )


def test_mink_goal_rejects_orthogonal_attached_collision_before_actuation(
    monkeypatch,
) -> None:
    mujoco = pytest.importorskip("mujoco")
    mink = pytest.importorskip("mink")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <compiler angle="radian"/>
          <option gravity="0 0 0"/>
          <worldbody>
            <body name="robot">
              <joint name="robot_joint" type="hinge" axis="0 0 1" range="-2 2"/>
              <geom type="sphere" size="0.01" contype="0" conaffinity="0"/>
              <site name="grip_site"/>
            </body>
            <body name="held">
              <freejoint name="held_free"/>
              <geom name="held_geom" type="box" size="0.10 0.01 0.01"/>
            </body>
            <body name="obstacle" pos="0 0.08 0">
              <geom name="obstacle_geom" type="box" size="0.02 0.02 0.02"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    live_data = mujoco.MjData(model)
    mujoco.mj_forward(model, live_data)
    held = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "held_geom",
    )
    obstacle = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "obstacle_geom",
    )
    held_free = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "held_free",
    )
    attached_qpos_adr = int(model.jnt_qposadr[held_free])
    raw = SimpleNamespace(
        sim=SimpleNamespace(
            model=SimpleNamespace(_model=model),
            data=live_data,
        ),
        env=SimpleNamespace(control_freq=20),
    )
    robot = SimpleNamespace(
        robot_joints=["robot_joint"],
        _ref_joint_vel_indexes=[0],
        _ref_joint_pos_indexes=[0],
        controller=SimpleNamespace(output_max=np.ones(1)),
        gripper=SimpleNamespace(important_sites={"grip_site": "grip_site"}),
        robot_model=SimpleNamespace(eef_name="robot"),
    )
    environment = SimpleNamespace(
        _env=SimpleNamespace(_controller="JOINT_VELOCITY")
    )
    collision_policy = {
        "limit": object(),
        "protected_pairs": [],
        "hard_stop_distance_m": -0.001,
        "minimum_distance_from_collisions_m": 0.003,
        "minimum_distance_m": math.inf,
        "minimum_attached_object_distance_m": math.inf,
        "world_geom_count": 2,
        "world_object_count": 2,
        "robot_geom_count": 0,
        "protected_pair_count": 0,
        "authorized_target_object": "held",
        "authorized_target_geom_count": 1,
        "contact_authorization": {},
        "attachment_proxy": {"status": "confirmed", "object_name": "held"},
        "attached_object_pairs": [(held, obstacle)],
        "attached_object_qpos_adr": attached_qpos_adr,
        "attached_object_geom_count": 1,
        "attached_object_world_geom_count": 1,
    }
    identity_quat = np.asarray([0.0, 0.0, 0.0, 1.0])
    target_quat = np.asarray(
        [0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)]
    )
    callback_actions: list[np.ndarray] = []

    monkeypatch.setattr(mink_goal, "_libero_runtime", lambda _env: (raw, robot))
    monkeypatch.setattr(
        mink_goal,
        "_eef_pose",
        lambda _raw, _robot: (np.zeros(3), identity_quat.copy()),
    )
    monkeypatch.setattr(mink_goal, "_site_rotation", lambda *_args: np.eye(3))
    monkeypatch.setattr(mink_goal, "_body_rotation", lambda *_args: np.eye(3))
    monkeypatch.setattr(
        mink_goal,
        "_configuration_eef_pose",
        lambda *_args: (np.zeros(3), target_quat.copy()),
    )
    monkeypatch.setattr(
        mink_goal,
        "_libero_collision_policy",
        lambda *_args, **_kwargs: collision_policy,
    )
    monkeypatch.setattr(
        mink_goal,
        "_fixed_nonrobot_velocity_limit",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        mink,
        "solve_ik",
        lambda *_args, **_kwargs: np.zeros(model.nv),
    )

    result = mink_goal.execute_libero_mink_goal(
        environment,
        target_xyz=[0.0, 0.0, 0.0],
        target_quat_xyzw=target_quat.tolist(),
        preserve_current_orientation=False,
        max_steps=1,
        position_tolerance_m=0.002,
        orientation_tolerance_rad=0.05,
        gripper_command=0.0,
        enable_collision_check=True,
        contact_authorization=None,
        attachment_proxy={"status": "confirmed", "object_name": "held"},
        ik_execution_seed=None,
        step_callback=lambda action, _render: callback_actions.append(action) or {},
    )

    assert callback_actions == []
    assert result["steps_executed"] == 0
    assert result["stop_reason"] == "collision_detected"
    assert result["collision"]["detected"] is True
    assert result["collision"]["collision_type"] == "attached_object_world"
    assert result["collision"]["minimum_distance_m"] == pytest.approx(-0.015)


def test_attached_object_collision_receipt_reports_per_step_geometry_coverage() -> None:
    receipt = mink_goal._collision_receipt(
        {
            "minimum_distance_m": 0.01,
            "minimum_attached_object_distance_m": 0.004,
            "hard_stop_distance_m": -0.001,
            "minimum_distance_from_collisions_m": 0.003,
            "world_geom_count": 12,
            "world_object_count": 3,
            "protected_pair_count": 20,
            "authorized_target_object": "bottle_1",
            "authorized_target_geom_count": 2,
            "contact_authorization": {},
            "attachment_proxy": {"object_name": "bottle_1"},
            "attached_object_pairs": [(1, 8), (2, 8)],
            "attached_object_geom_count": 2,
            "attached_object_world_geom_count": 1,
            "attached_object_boundary_recovery_steps": 1,
        }
    )

    assert receipt["trajectory_checked"] is True
    coverage = receipt["attached_object_coverage"]
    assert coverage["trajectory_checked"] is True
    assert coverage["predicted_step_checked"] is True
    assert coverage["actual_step_checked"] is True
    assert coverage["protected_pair_count"] == 2
    assert coverage["prediction_policy"] == (
        "rigid_object_to_eef_transform_per_predicted_step"
    )
    assert coverage["boundary_recovery"]["verified_escape_steps"] == 1


def test_mink_contact_move_resolves_host_anchor_before_worker_dispatch(monkeypatch) -> None:
    control_spec = {
        "schema_version": "openeta.sim_control.v1",
        "controller": {
            "controller_id": "mink.robosuite_joint_velocity",
            "configured_name": "JOINT_VELOCITY",
            "command_interface": "joint_velocity",
            "goal_executor": "openeta.worker_mink_goal.v1",
            "execution_location": "bench_worker",
            "supports_position": True,
            "supports_orientation": True,
        },
        "cartesian_delta": {"supported": False},
    }
    meta = {
        "backend": "libero",
        "action_dim": 7,
        "remote_handle": "remote",
        "control_spec": control_spec,
        "_collision_objects": [
            {
                "name": "salad_dressing_1",
                "category": "salad_dressing",
                "position": [0.1, 0.2, 0.12],
                "dims": [0.05, 0.05, 0.12],
            },
            {
                "name": "basket_1",
                "category": "basket",
                "position": [0.3, 0.2, 0.1],
                "dims": [0.2, 0.2, 0.2],
            },
        ],
    }
    calls: list[dict] = []
    monkeypatch.setattr(server, "_session_envs", {"sid": {"handle": meta}})
    monkeypatch.setattr(server, "_touch_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_proxy_controller_goal",
        lambda _meta, body: calls.append(body) or {"reached_target": True},
    )

    result = server.move_to.__wrapped__(
        "handle",
        0.1,
        0.2,
        0.16,
        enable_collision_check=True,
        contact_authorization={
            "schema_version": "openeta.contact_authorization.v1",
            "compiled_grasp_id": "compiled-1",
            "waypoint_role": "grasp_contact",
            "target_anchor_world_xyz": [0.1, 0.2, 0.12],
            "object_scene_epoch": 2,
        },
        session_id="sid",
    )

    assert result == {"reached_target": True}
    assert calls[0]["contact_authorization"]["target_object_name"] == (
        "salad_dressing_1"
    )
    assert calls[0]["enable_collision_check"] is True


def test_behavior_ik_config_and_runtime_layout_are_explicit() -> None:
    config = {"controller_config": {"arm_left": {}, "arm_right": {}}}
    _configure_agent_cartesian_control(config)
    assert config["controller_config"]["arm_left"]["name"] == "InverseKinematicsController"
    assert config["controller_config"]["arm_right"]["mode"] == "pose_delta_ori"
    assert config["controller_config"]["arm_right"]["command_output_limits"][1] == [
        0.05,
        0.05,
        0.05,
        0.25,
        0.25,
        0.25,
    ]

    robot = SimpleNamespace(
        arm_names=("left", "right"),
        default_arm="left",
        arm_action_idx={"left": np.arange(1, 7), "right": np.arange(7, 13)},
        gripper_action_idx={"left": np.array([6]), "right": np.array([13])},
    )
    direct = object.__new__(BehaviorDirectEnv)
    direct._env = SimpleNamespace(robots=[robot])
    spec = direct.openeta_control_spec
    assert spec["cartesian_delta"]["arm"] == "right"
    assert spec["cartesian_delta"]["position_indices"] == [7, 8, 9]
    assert spec["gripper"]["indices"] == [13]


def test_behavior_codec_writes_only_declared_arm_and_gripper_slots() -> None:
    action = make_cartesian_action(
        BEHAVIOR_META,
        [0.1, -0.2, 0.3],
        "behavior",
        delta_rot=[0.4, 0.5, -0.6],
    )
    assert action[7:13] == [0.1, -0.2, 0.3, 0.4, 0.5, -0.6]
    assert sum(abs(value) for value in action[:7] + action[13:]) == 0.0

    opened = make_gripper_action(BEHAVIOR_META, open_gripper=True, backend="behavior")
    closed = make_gripper_action(BEHAVIOR_META, open_gripper=False, backend="behavior")
    assert opened[13] == 1.0
    assert closed[13] == -1.0
    assert sum(abs(value) for value in opened) == 1.0


def test_unknown_and_undeclared_backends_fail_closed() -> None:
    with pytest.raises(ControlCodecError) as behavior_error:
        make_cartesian_action({}, [1, 2, 3], "behavior")
    assert behavior_error.value.code == "unsupported_cartesian_control"
    with pytest.raises(ControlCodecError) as unknown_error:
        make_cartesian_action({}, [1, 2, 3], "mystery_sim")
    assert unknown_error.value.code == "unsupported_cartesian_control"

    monkey_meta = {"backend": "behavior", "action_dim": 18, "remote_handle": "remote"}
    server._session_envs["sid"] = {"handle": monkey_meta}
    try:
        result = server.move_to.__wrapped__(
            "handle", 0.1, 0.2, 0.3, session_id="sid"
        )
    finally:
        server._session_envs.pop("sid", None)
    assert result["ok"] is False
    assert result["code"] == "unsupported_cartesian_control"


def test_move_to_stops_on_worker_error_and_preserves_last_pose(monkeypatch) -> None:
    meta = _libero_meta()
    start = [0.1, 0.2, 0.3]
    calls = 0

    monkeypatch.setattr(server, "_session_envs", {"sid": {"handle": meta}})
    monkeypatch.setattr(server, "_touch_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_proxy_observe",
        lambda *_args, **_kwargs: {
            "observation": {"robot": {"end_effector_pose": {"xyz": start}}}
        },
    )
    monkeypatch.setattr(server, "_proxy_render", lambda *_args, **_kwargs: {})

    def fail_step(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {"error": "Step failed: executing action in terminated episode"}

    monkeypatch.setattr(server, "_proxy_step", fail_step)

    result = server.move_to.__wrapped__(
        "handle", 0.2, 0.2, 0.3, num_steps=100, session_id="sid"
    )

    assert calls == 1
    assert result["ok"] is False
    assert result["code"] == "control_step_failed"
    assert result["steps_executed"] == 1
    assert result["end"]["xyz"] == start
    assert result["reached_target"] is False
    assert result["stop_reason"] == "control_step_failed"
    assert "terminated episode" in result["error"]


def test_move_to_receipt_reports_controller_residual_and_iteration_limit(
    monkeypatch,
) -> None:
    meta = _libero_meta()
    start = [0.0, 0.0, 0.0]

    monkeypatch.setattr(server, "_session_envs", {"sid": {"handle": meta}})
    monkeypatch.setattr(server, "_touch_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_proxy_observe",
        lambda *_args, **_kwargs: {
            "observation": {"robot": {"end_effector_pose": {"xyz": start}}}
        },
    )
    monkeypatch.setattr(server, "_proxy_render", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        server,
        "_proxy_step",
        lambda *_args, **_kwargs: {
            "observation": {"robot": {"end_effector_pose": {"xyz": start}}}
        },
    )

    result = server.move_to.__wrapped__(
        "handle", 0.1, 0.0, 0.0, num_steps=2, session_id="sid"
    )

    assert result["steps_executed"] == 2
    assert result["reached_target"] is False
    assert result["position_error_m"] == pytest.approx(0.1)
    assert result["max_axis_position_error_m"] == pytest.approx(0.1)
    assert result["stop_reason"] == "iteration_limit"
    assert result["controller_receipt"] == {
        "schema_version": "openeta.controller_execution_receipt.v1",
        "controller_id": "robosuite.osc_pose",
        "configured_name": "OSC_POSE",
        "command_interface": "normalized_cartesian_delta_pose",
        "goal_executor": "openeta.outer_closed_loop_cartesian.v1",
        "execution_location": "mcp_server",
        "orientation_policy": "preserve_current",
        "iteration_budget": 2,
        "steps_executed": 2,
        "stop_reason": "iteration_limit",
        "reached_target": False,
    }


def test_ik_preview_check_returns_structured_unreachable_without_moving(monkeypatch) -> None:
    meta = {"backend": "libero", "remote_handle": "remote"}
    monkeypatch.setattr(server, "_session_envs", {"sid": {"handle": meta}})
    monkeypatch.setattr(server, "_touch_session", lambda *_args, **_kwargs: None)
    calls: list[dict] = []

    def fake_preview(_meta, body):
        calls.append(body)
        return {
            "status": "unreachable",
            "kinematic_status": "unreachable",
            "feasible": False,
            "reason_code": "full_pose_infeasible",
            "message": "Position and orientation cannot be satisfied together.",
            "position_only_reachable": True,
            "orientation_only_reachable": True,
            "best_candidate": {
                "joint_positions": [0.0] * 7,
                "max_axis_position_error_m": 0.011,
                "orientation_error_rad": 0.05,
            },
        }

    monkeypatch.setattr(server, "_proxy_reachability", fake_preview)
    result = server.ik_preview_check.__wrapped__(
        "handle",
        0.1,
        0.2,
        0.3,
        roll=180.0,
        pitch=0.0,
        yaw=0.0,
        session_id="sid",
    )

    assert result["success"] is False
    assert result["status"] == "unreachable"
    assert result["reason_code"] == "full_pose_infeasible"
    assert result["collision"] == {"checked": False}
    assert result["path"]["checked"] is False
    assert calls[0]["target_xyz"] == [0.1, 0.2, 0.3]
    assert calls[0]["target_euler_xyz_deg"] == [180.0, 0.0, 0.0]


def test_ik_preview_unknown_does_not_become_a_false_rejection(monkeypatch) -> None:
    meta = {"backend": "libero", "remote_handle": "remote"}
    monkeypatch.setattr(server, "_session_envs", {"sid": {"handle": meta}})
    monkeypatch.setattr(server, "_touch_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_proxy_reachability",
        lambda *_args, **_kwargs: {
            "status": "unknown",
            "kinematic_status": "unknown",
            "feasible": None,
            "reason_code": "ik_search_timeout",
            "message": "Search budget expired.",
        },
    )

    result = server.ik_preview_check.__wrapped__(
        "handle", 0.1, 0.2, 0.3, session_id="sid"
    )

    assert result["ok"] is True
    assert result["success"] is True
    assert result["status"] == "unknown"
    assert result["feasible"] is None


def test_ik_preview_reports_fragile_joint_limit_margin_without_rejecting(
    monkeypatch,
) -> None:
    meta = {"backend": "libero", "remote_handle": "remote"}
    monkeypatch.setattr(server, "_session_envs", {"sid": {"handle": meta}})
    monkeypatch.setattr(server, "_touch_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_proxy_reachability",
        lambda *_args, **_kwargs: {
            "status": "reachable",
            "kinematic_status": "reachable",
            "feasible": True,
            "reason_code": "ik_solution_found",
            "message": "A joint-limit-respecting IK solution was found.",
            "best_candidate": {
                "joint_positions": [0.0] * 7,
                "joint_margin_min_rad": 0.0329,
                "nearest_joint_limit": {
                    "joint_index": 5,
                    "boundary": "upper",
                },
            },
            "suggestions": [],
        },
    )

    result = server.ik_preview_check.__wrapped__(
        "handle", 0.1, 0.2, 0.3, session_id="sid"
    )

    assert result["success"] is True
    assert result["joint_limit_proximity"]["near_limit"] is True
    assert result["joint_limit_proximity"]["nearest_joint_limit"] == {
        "joint_index": 5,
        "boundary": "upper",
    }
    assert result["joint_limit_proximity"]["warning_threshold_rad"] == 0.05
    assert "0.032900 rad" in result["content"]
    assert result["execution_seed_quality"]["risk_level"] == "critical"
    assert result["execution_seed_quality"]["robust_margin_threshold_rad"] == 0.1
    assert "select_higher_joint_margin_target_or_orientation" in result["suggestions"]
    assert "compare_alternative_grasp_candidate_before_motion" in result["suggestions"]


def test_ik_preview_marks_elevated_execution_seed_without_rejecting(
    monkeypatch,
) -> None:
    meta = {"backend": "libero", "remote_handle": "remote"}
    monkeypatch.setattr(server, "_session_envs", {"sid": {"handle": meta}})
    monkeypatch.setattr(server, "_touch_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_proxy_reachability",
        lambda *_args, **_kwargs: {
            "status": "reachable",
            "kinematic_status": "reachable",
            "feasible": True,
            "reason_code": "ik_solution_found",
            "message": "A joint-limit-respecting IK solution was found.",
            "best_candidate": {
                "joint_positions": [0.0] * 7,
                "joint_margin_min_rad": 0.081,
            },
            "solver": {
                "execution_seed_search": {
                    "feasible_solution_count": 3,
                    "robust_solution_selected": False,
                }
            },
            "suggestions": [],
        },
    )

    result = server.ik_preview_check.__wrapped__(
        "handle", 0.1, 0.2, 0.3, session_id="sid"
    )

    assert result["success"] is True
    assert "joint_limit_proximity" not in result
    assert result["execution_seed_quality"] == {
        "risk_level": "elevated",
        "selected_joint_margin_rad": 0.081,
        "robust_margin_threshold_rad": 0.1,
        "robust_alternative_found": False,
        "feasible_solution_count": 3,
        "distant_robust_solution_count": None,
        "interpretation": result["execution_seed_quality"]["interpretation"],
    }
    assert "not a positive execution recommendation" in result[
        "execution_seed_quality"
    ]["interpretation"]
    assert "execution-fragile" in result["content"]


def test_trajectory_pose_arguments_accept_quaternion_and_validate_endpoint() -> None:
    arguments = server._trajectory_pose_arguments(
        {"frame": "world", "xyz": [0.1, 0.2, 0.3], "quat_xyzw": [0, 0, 0, 1]},
        index=0,
    )
    assert arguments == {
        "x": 0.1,
        "y": 0.2,
        "z": 0.3,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    }
    assert server._trajectory_waypoint_reached(
        {"end": {"xyz": [0.101, 0.2, 0.3]}},
        arguments,
        tolerance=0.002,
    ) is True
    assert server._trajectory_waypoint_reached(
        {"end": {"xyz": [0.104, 0.2, 0.3]}},
        arguments,
        tolerance=0.002,
    ) is False
    assert server._trajectory_waypoint_reached(
        {
            "reached_target": False,
            "stop_reason": "local_convergence_stalled",
            "end": {"xyz": [0.1, 0.2, 0.3]},
        },
        arguments,
        tolerance=0.002,
    ) is False


def test_condition_c_route_bundle_must_match_host_resolved_trajectory() -> None:
    trajectory = [{"frame": "world", "xyz": [0.1, 0.2, 0.3]}]
    bundle = {
        "schema_version": "openeta.experimental_route_execution_bundle.v1",
        "condition": "C",
        "authority": "host_memory_exact_receipt_resolution",
        "entries": [
            {
                "source_ik_receipt_id": "ik-route-1",
                "target_pose": {"frame": "world", "xyz": [0.1, 0.2, 0.31]},
            }
        ],
    }

    with pytest.raises(ValueError, match="does not match"):
        server._condition_c_route_entries(bundle, trajectory)


def test_condition_c_sequential_preview_issues_private_seed(monkeypatch) -> None:
    monkeypatch.setattr(
        server,
        "_proxy_reachability",
        lambda meta, body: {
            "status": "reachable",
            "feasible": True,
            "target": {
                "xyz": body["target_xyz"],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "best_candidate": {
                "joint_positions": [0.1] * 7,
                "joint_margin_min_rad": 0.2,
                "joint_travel_l2_rad": 0.4,
            },
        },
    )
    entry = {
        "source_ik_receipt_id": "ik-route-1",
        "execution_arguments": {"x": 0.1, "y": 0.2, "z": 0.3},
    }

    receipt, seed = server._sequential_route_preview(
        {"backend": "libero", "remote_handle": "remote"},
        entry,
        index=0,
        tolerance=0.002,
        ori_tolerance=0.05,
    )

    assert receipt["feasible"] is True
    assert receipt["preview_state"] == "actual_preceding_segment_end"
    assert receipt["path_collision_checked"] is False
    assert seed is not None
    assert seed["schema_version"] == "openeta.ik_execution_seed.v1"
    assert seed["joint_positions"] == [0.1] * 7


def test_ttl_cleanup_closes_releases_and_removes_every_handle(monkeypatch) -> None:
    calls: list[tuple] = []

    class Manager:
        def proxy_handle_op(self, meta, path, method="GET"):
            calls.append(("close", meta["remote_handle"], path, method))
            return {"ok": True}

        def release_worker(self, worker_url):
            calls.append(("release", worker_url))

    monkeypatch.setattr(session, "_get_mgr", lambda: Manager())
    monkeypatch.setattr(collision, "remove_checker", lambda handle: calls.append(("checker", handle)))
    monkeypatch.setattr(
        session,
        "_session_envs",
        {"sid": {"local": {"remote_handle": "remote", "worker_url": "worker"}}},
    )
    monkeypatch.setattr(session, "_session_last_obs", {"sid": {"remote": {}}})
    monkeypatch.setattr(session, "_session_last_activity", {"sid": 1.0})
    monkeypatch.setattr(session, "_session_stream_interval", {"sid": 0.1})
    monkeypatch.setattr(session, "_sse_sessions", {"sid"})

    session._cleanup_session("sid")

    assert ("release", "worker") in calls
    assert ("checker", "local") in calls
    assert "sid" not in session._session_envs
    assert "sid" not in session._session_last_obs


def test_close_env_is_idempotent_and_releases_after_remote_error(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class Manager:
        def proxy_handle_op(self, meta, path, method="GET"):
            raise RuntimeError("transport down")

        def release_worker(self, worker_url):
            calls.append(("release", worker_url))

    monkeypatch.setattr(server, "_get_mgr", lambda: Manager())
    monkeypatch.setattr(server, "_touch_session", lambda sid: None)
    monkeypatch.setattr(server, "remove_checker", lambda handle: calls.append(("checker", handle)))
    monkeypatch.setattr(
        server,
        "_session_envs",
        {"sid": {"local": {"remote_handle": "remote", "worker_url": "worker"}}},
    )
    monkeypatch.setattr(server, "_session_last_obs", {"sid": {"local": {}}})

    first = server.close_env.__wrapped__("local", session_id="sid")
    second = server.close_env.__wrapped__("local", session_id="sid")

    assert first["ok"] is False
    assert first["cleanup_errors"][0].startswith("remote_close:")
    assert ("release", "worker") in calls
    assert second == {"ok": True, "already_closed": True, "cleanup_errors": []}
