# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import pytorch_kinematics as pk

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    sample_uniform,
)

# from tactile_sim import GsMiniSensorCfg, GsMiniSensor
from tacex_assets import TACEX_ASSETS_DATA_DIR

from .base_env import BallRollingEnv, BallRollingEnvCfg

#  from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
# from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg


@configclass
class BallRollingIKResetEnvCfg(BallRollingEnvCfg):
    # use an proper ik solver for computing desired ee pose after resets
    ik_solver_cfg = {
        "urdf_path": f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Single_Adapter/physx_rigid_gelpad.urdf",
        "ee_link_name": "panda_hand",  # gelsight_mini_gelpad
        "max_iterations": 100,
        "num_retries": 1,
        "learning_rate": 0.2,
    }
    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")


class BallRollingIKResetEnv(BallRollingEnv):
    """RL env in which the robot has to push/roll a ball to a goal position.

    This base env uses (absolute) joint positions.
    Absolute joint pos and vel are used for the observations.
    """

    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: BallRollingIKResetEnvCfg

    def __init__(self, cfg: BallRollingIKResetEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # --- IK Solver ---
        with open(self.cfg.ik_solver_cfg["urdf_path"], mode="rb") as urdf_file:
            ik_chain = pk.build_chain_from_urdf(urdf_file.read())
        # ik_chain.print_tree()
        # extract a specific serial chain such for inverse kinematics
        ik_chain = pk.SerialChain(ik_chain, self.cfg.ik_solver_cfg["ee_link_name"])
        # ik_chain.print_tree()
        ik_chain = ik_chain.to(dtype=torch.float32, device=self.device)

        # get robot joint limits
        ik_chain_lim = torch.tensor(ik_chain.get_joint_limits(), device=self.device)

        # create the IK object
        # see the constructor for more options and their explanations, such as convergence tolerances
        self.ik_solver = pk.PseudoInverseIK(
            ik_chain,
            max_iterations=self.cfg.ik_solver_cfg["max_iterations"],
            num_retries=self.cfg.ik_solver_cfg["num_retries"],
            joint_limits=ik_chain_lim.T,
            early_stopping_any_converged=True,
            early_stopping_no_improvement="all",  # "all", None
            debug=False,
            lr=self.cfg.ik_solver_cfg["learning_rate"],
        )
        self.des_reset_ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # self.des_reset_ee_rot = lab_math.matrix_from_quat(torch.tensor([0,1,0,0],device=self.device).repeat(self.num_envs, 1))
        self.des_reset_ee_rot = (
            torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1, 1)
        )

    # MARK: pre-physics step calls
    # same as base_env
    # uncomment if you only want to check behavior of IK solver for reset
    # def _apply_action(self):
    #     pass

    # MARK: dones
    # same as base_env

    # MARK: rewards
    # same as base_env

    # MARK: reset
    def _reset_idx(self, env_ids: torch.Tensor | None):
        # ---------------- From the DirectRLEnv class ----------
        self.scene.reset(env_ids)
        # apply events such as randomization for environments that need a reset
        if self.cfg.events:
            if "reset" in self.event_manager.available_modes:
                env_step_count = self._sim_step_counter // self.cfg.decimation
                self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # reset noise models
        if self.cfg.action_noise_model:
            self._action_noise_model.reset(env_ids)
        if self.cfg.observation_noise_model:
            self._observation_noise_model.reset(env_ids)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0
        # ------------------------------------------------------

        # spawn obj at random position
        obj_pos = self.object.data.default_root_state[env_ids]
        obj_pos[:, :3] += self.scene.env_origins[env_ids]
        obj_pos[:, :2] += sample_uniform(
            self.cfg.obj_pos_randomization_range[self.curriculum_phase_id][0],
            self.cfg.obj_pos_randomization_range[self.curriculum_phase_id][1],
            (len(env_ids), 2),
            self.device,
        )
        self.object.write_root_state_to_sim(obj_pos, env_ids=env_ids)

        # compute desired ee pose so that its at the ball
        # make sure that ee pose is in robot frame
        self.des_reset_ee_pos[env_ids, :] = obj_pos[:, :3].clone() - self.scene.env_origins[env_ids]
        # add offset between gelsight mini case frame (which is at the bottom of the sensor) to the gelpad
        self.des_reset_ee_pos[
            env_ids, 2
        ] += 0.131  # cannot set it too close to the ball, otherwise "teleporting" robot there is gonna kick ball away

        # convert desired pos into transformation matrix
        goal_poses = pk.Transform3d(
            pos=self.des_reset_ee_pos[env_ids], rot=self.des_reset_ee_rot[env_ids], device=self.device
        )
        # solve via IK for desired joint pos
        sol = self.ik_solver.solve(goal_poses)
        # # num goals x num retries x DOF tensor of joint angles; if not converged, best solution found so far
        # print(sol.solutions.shape)
        # # num goals x num retries can check for the convergence of each run
        # print(sol.converged)
        # # num goals x num retries can look at errors directly
        # print(sol.err_pos)
        # print(sol.err_rot)
        # indices = torch.argmin(sol.err_pos, dim=1)
        # best_sol_currently = sol.solutions[torch.arange(indices.size(0)), indices]

        # write the computed IK values into the joint state of the robot
        joint_pos = torch.clamp(sol.solutions[:, 0], self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # set commands: random position
        self._desired_pos_w[env_ids, :2] = (
            self.object.data.default_root_state[env_ids][:, :2] + self.scene.env_origins[env_ids][:, :2]
        )
        self._desired_pos_w[env_ids, :2] += sample_uniform(
            self.cfg.obj_pos_randomization_range[self.curriculum_phase_id][0],
            self.cfg.obj_pos_randomization_range[self.curriculum_phase_id][1],
            (len(env_ids), 2),
            self.device,
        )

        # reset actions
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._ik_controller.reset(env_ids)
