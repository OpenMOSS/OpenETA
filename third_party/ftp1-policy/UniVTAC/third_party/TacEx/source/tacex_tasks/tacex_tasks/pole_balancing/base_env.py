# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    sample_uniform,
    subtract_frame_transforms,
    wrap_to_pi,
)
from isaaclab.utils.noise import NoiseModelCfg, UniformNoiseCfg

from tacex import GelSightSensor

# from tactile_sim import GsMiniSensorCfg, GsMiniSensor
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_single_rigid import (
    FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_RIGID_CFG,
)
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg

from tacex_tasks.utils import DirectLiveVisualizer

#  from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
# from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg


class CustomEnvWindow(BaseEnvWindow):
    """Window manager for the RL environment."""

    def __init__(self, env: DirectRLEnvCfg, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class PoleBalancingEnvCfg(DirectRLEnvCfg):
    # viewer settings
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1, -0.5, 0.1)
    viewer.lookat = (-19.4, 18.2, -1.1)

    viewer.origin_type = "env"
    viewer.env_idx = 0
    viewer.resolution = (1280, 720)

    debug_vis = True

    ui_window_class_type = CustomEnvWindow

    decimation = 1
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,  # 0.001
        render_interval=decimation,
        # device="cpu",
        physx=PhysxCfg(
            enable_ccd=True,  # needed for more stable ball_rolling
            # bounce_threshold_velocity=10000,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=5.0,
            dynamic_friction=5.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=1, replicate_physics=True)

    # use robot with stiff PD control for better IK tracking
    robot: ArticulationCfg = FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            # joint_pos={
            #     "panda_joint1": -1.5,
            #     "panda_joint2": -1.5,
            #     "panda_joint3": 2.11,
            #     "panda_joint4": -2.46,
            #     "panda_joint5": -1.18,
            #     "panda_joint6": 1.17,
            #     "panda_joint7": -1.5,
            # },
            joint_pos={
                "panda_joint1": 1.5,
                "panda_joint2": -1.76,
                "panda_joint3": -1.84,
                "panda_joint4": -2.52,
                "panda_joint5": 1.25,
                "panda_joint6": 1.58,
                "panda_joint7": -1.72,
            },
        ),
    )

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    ee_frame_cfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.131),  # 0ffset from panda hand frame origin to gelpad top
                    rot=(1.0, 0.0, 0.0, 0.0),
                    # rot=(0, 0.92388, -0.38268, 0) # our panda hand asset has rotation from (180,0,-45) -> we subtract 180 for defining the rotation limits
                ),
            ),
        ],
    )

    # rigid body pole
    pole: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/pole",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/pole.usd",
            # scale=(2, 1, 0.6),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=120,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.41336, 0.01123, 0.4637)),  # (0.3821, 0.04255, 0.37877)
    )

    # sensors
    gsmini = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case",
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=0,
            resolution=(32, 32),
            data_types=["depth"],
            clipping_range=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=True,  # for being able to see sensor output in the gui
        # update Taxim cfg
        optical_sim_cfg=None,
        # update FOTS cfg
        marker_motion_sim_cfg=None,
        # marker_motion_sim_cfg=FOTSMarkerSimulatorCfg(
        #     lamb = [0.00125,0.00021,0.00038],
        #     pyramid_kernel_size = [51, 21, 11, 5], #[11, 11, 11, 11, 11, 5],
        #     kernel_size = 5,
        #     marker_params = FOTSMarkerSimulatorCfg.MarkerParams(
        #         num_markers_col=25, #11,
        #         num_markers_row=20, #9,
        #         x0=26,
        #         y0=15,
        #         dx=29,
        #         dy=26
        #     ),
        #     tactile_img_res = (480, 640),
        #     device = "cuda",
        #     frame_transformer_cfg = FrameTransformerCfg(
        #         prim_path="/World/envs/env_.*/Robot/gelsight_mini_gelpad", #"/World/envs/env_.*/Robot/gelsight_mini_case",
        #         source_frame_offset=OffsetCfg(),
        #         target_frames=[
        #             FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/rigid_ball")
        #         ],
        #         debug_vis=False,
        #     )
        # ),
        data_types=["camera_depth"],  # marker_motion
    )

    # noise models
    action_noise_model = NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.001, n_max=0.001, operation="add"))
    # observation_noise_model =

    # MARK: reward configuration
    reward_terms = {
        "at_obj_reward": {"weight": 0.75, "minimal_distance": 0.005},
        "height_reward": {"weight": 0.25, "w": 10.0, "v": 0.3, "alpha": 0.00067, "target_height_cm": 50},
        "orient_reward": {"weight": 0.25},
        "staying_alive_rew": {"weight": 0.5},
        "termination_penalty": {"weight": -10.0},
        "ee_goal_tracking_penalty": {"weight": -0.001},
        "ee_goal_fine_tracking_reward": {"weight": 0.75, "std": 0.0380},
        "action_rate_penalty": {"weight": -1e-4},
        "joint_vel_penalty": {"weight": -1e-4},
    }

    # curriculum settings
    num_levels = 10

    obj_pos_randomization_range = [-0.05, 0.05]

    # env
    episode_length_s = 8.3333 / 2  # 1000/2 timesteps (dt = 1/120 -> 8.3333/(1/120) = 1000)
    action_space = 6  # we use relative task_space actions: (dx, dy, dz, droll, dpitch) -> dyaw is omitted
    observation_space = {
        "proprio_obs": 14,  # 16, # 3 for ee pos, 2 for orient (roll, pitch), 2 for init goal-pos (x,y), 5 for actions
        "vision_obs": [32, 32, 1],  # from tactile sensor
    }
    state_space = 0
    action_scale = 0.05  # [cm]

    x_bounds = (0.15, 0.75)
    y_bounds = (-0.75, 0.75)
    too_far_away_threshold = 0.05  # 0.01 #0.2 #0.125 #0.2 #0.15
    min_height_threshold = 0.3  # 0.37877


class PoleBalancingEnv(DirectRLEnv):
    """RL env in which the robot has to balance a pole towards a goal position."""

    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: PoleBalancingEnvCfg

    def __init__(self, cfg: PoleBalancingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # for training curriculum
        self.current_curriculum_level = 0
        self.curriculum_weights = torch.linspace(0, 1, self.cfg.num_levels, device=self.device)

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        # --- For IK actions ---

        # create the differential IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        # Obtain the frame index of the end-effector
        body_ids, body_names = self._robot.find_bodies("panda_hand")
        # save only the first body index
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]

        # For a fixed base robot, the frame index is one less than the body index.
        # This is because the root body is not included in the returned Jacobians.
        self._jacobi_body_idx = self._body_idx - 1
        # self._jacobi_joint_ids = self._joint_ids # we take every joint

        # ee offset w.r.t panda hand -> based on the asset
        self._offset_pos = torch.tensor([0.0, 0.0, 0.131], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        # self._offset_rot = torch.tensor([0, 0.92388, -0.38268, 0], device=self.device).repeat(self.num_envs, 1)

        # ---

        # create auxiliary variables for computing applied action, observations and rewards
        self.processed_actions = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)

        self._goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._goal_pos_w[:, 2] = self.cfg.reward_terms["height_reward"]["target_height_cm"] * 0.01

        self.reward_terms = {}
        for rew_terms in self.cfg.reward_terms:
            self.reward_terms[rew_terms] = torch.zeros((self.num_envs), device=self.device)

        if self.cfg.debug_vis:
            # add plots
            self.visualizers = {
                "Actions": DirectLiveVisualizer(
                    self.cfg.debug_vis, self.num_envs, self._window, visualizer_name="Actions"
                ),
                "Observations": DirectLiveVisualizer(
                    self.cfg.debug_vis, self.num_envs, self._window, visualizer_name="Observations"
                ),
                "Rewards": DirectLiveVisualizer(
                    self.cfg.debug_vis, self.num_envs, self._window, visualizer_name="Rewards"
                ),
                "Metrics": DirectLiveVisualizer(
                    self.cfg.debug_vis, self.num_envs, self._window, visualizer_name="Metrics"
                ),
            }
            self.visualizers["Actions"].terms["actions"] = self.actions

            self.visualizers["Observations"].terms["ee_pos"] = torch.zeros((self.num_envs, 3))
            self.visualizers["Observations"].terms["ee_rot"] = torch.zeros((self.num_envs, 3))
            self.visualizers["Observations"].terms["goal"] = torch.zeros((self.num_envs, 2))
            self.visualizers["Observations"].terms["sensor_output"] = self._get_observations()["policy"]["vision_obs"]

            self.visualizers["Rewards"].terms["rewards"] = torch.zeros((self.num_envs, 10))
            self.visualizers["Rewards"].terms_names["rewards"] = [
                "at_obj_reward",
                "height_reward",
                "orient_reward",
                "staying_alive_rew",
                "termination_penalty",
                "ee_goal_tracking",
                "ee_goal_fine_tracking_reward",
                "action_rate_penalty",
                "joint_vel_penalty",
                "full",
            ]

            self.visualizers["Metrics"].terms["ee_height"] = torch.zeros((self.num_envs, 1))
            self.visualizers["Metrics"].terms["pole_orient_x"] = torch.zeros((self.num_envs, 1))
            self.visualizers["Metrics"].terms["pole_orient_y"] = torch.zeros((self.num_envs, 1))
            self.visualizers["Metrics"].terms["obj_ee_distance"] = torch.zeros((self.num_envs, 1))

            for vis in self.visualizers.values():
                vis.create_visualizer()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.object = RigidObject(self.cfg.pole)
        self.scene.rigid_objects["object"] = self.object

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)

        # sensors
        self._ee_frame = FrameTransformer(self.cfg.ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        self.gsmini = GelSightSensor(self.cfg.gsmini)
        self.scene.sensors["gsmini"] = self.gsmini

        # Ground-plane
        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
            spawn=sim_utils.GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        ground.spawn.func(
            ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        )

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # MARK: pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions.clone()
        self.actions[:] = actions  # .clamp(-1.0, 1.0)

        self.processed_actions[:, :] = self.actions * self.cfg.action_scale

        # obtain ee positions and orientation w.r.t root (=base) frame
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        # set command into controller
        self._ik_controller.set_command(self.processed_actions, ee_pos_curr_b, ee_quat_curr_b)

    def _apply_action(self):
        # obtain quantities from simulation
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]
        # compute the delta in joint-space
        if ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        self._robot.set_joint_position_target(joint_pos_des)

        # pass

    # post-physics step calls

    # MARK: dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:  # which environment is done
        obj_pos = self.object.data.root_link_pos_w - self.scene.env_origins
        out_of_bounds_x = (obj_pos[:, 0] < self.cfg.x_bounds[0]) | (obj_pos[:, 0] > self.cfg.x_bounds[1])
        out_of_bounds_y = (obj_pos[:, 1] < self.cfg.y_bounds[0]) | (obj_pos[:, 1] > self.cfg.y_bounds[1])

        obj_goal_distance = torch.norm(self._goal_pos_w[:, :2] - self.scene.env_origins[:, :2] - obj_pos[:, :2], dim=1)
        obj_too_far_away = obj_goal_distance > 1.0

        ee_frame_pos = (
            self._ee_frame.data.target_pos_w[..., 0, :] - self.scene.env_origins
        )  # end-effector positions in world frame: (num_envs, 3)
        ee_too_far_away = torch.norm(obj_pos - ee_frame_pos, dim=1) > self.cfg.too_far_away_threshold

        # reset when pole orient is too large
        pole_orient = euler_xyz_from_quat(self.object.data.root_link_quat_w)
        x = wrap_to_pi(pole_orient[0])
        y = wrap_to_pi(pole_orient[1])
        orient_cond = (torch.abs(x) > math.pi / 4) | (torch.abs(y) > math.pi / 4)

        ee_min_height = ee_frame_pos[:, 2] < self.cfg.min_height_threshold
        obj_min_height = obj_pos[:, 2] < self.cfg.min_height_threshold

        reset_cond = (
            out_of_bounds_x
            | out_of_bounds_y
            | obj_too_far_away
            | ee_too_far_away
            | orient_cond
            | ee_min_height
            | obj_min_height
        )

        time_out = self.episode_length_buf >= self.max_episode_length - 1  # episode length limit

        return reset_cond, time_out

    # MARK: rewards
    def _get_rewards(self) -> torch.Tensor:
        # - Reward the agent for reaching the object using tanh-kernel.
        obj_pos = self.object.data.root_link_pos_w
        ee_frame_pos = self._ee_frame.data.target_pos_w[
            ..., 0, :
        ]  # end-effector positions in world frame: (num_envs, 3)

        # Distance of the end-effector to the object: (num_envs,)
        object_ee_distance = torch.norm(obj_pos - ee_frame_pos, dim=1)
        # for giving agent incentive to touch the obj
        self.reward_terms["at_obj_reward"][:] = torch.where(
            object_ee_distance <= self.cfg.reward_terms["at_obj_reward"]["minimal_distance"],
            self.cfg.reward_terms["at_obj_reward"]["weight"],
            0.0,
        )

        height_diff = (
            self.cfg.reward_terms["height_reward"]["target_height_cm"] - ee_frame_pos[:, 2] * 100
        ) * 0.1  # [dm]
        height_reward = -(
            self.cfg.reward_terms["height_reward"]["w"] * height_diff**2
            + self.cfg.reward_terms["height_reward"]["v"]
            * torch.log(height_diff**2 + self.cfg.reward_terms["height_reward"]["alpha"])
        ).clamp(-1, 1)
        # penalize ee being too close to ground
        height_reward = torch.where(
            (ee_frame_pos[:, 2] <= self.cfg.min_height_threshold), height_reward - 10, height_reward
        )
        self.reward_terms["height_reward"][:] = height_reward * self.cfg.reward_terms["height_reward"]["weight"]

        # reward for upright pole
        pole_orient = euler_xyz_from_quat(self.object.data.root_link_quat_w)
        x = wrap_to_pi(pole_orient[0])
        y = wrap_to_pi(pole_orient[1])
        orient_reward = torch.where(
            (torch.abs(x) < math.pi / 8) | (torch.abs(y) < math.pi / 8),
            1.0 * self.cfg.reward_terms["orient_reward"]["weight"],
            0.0,
        )
        self.reward_terms["orient_reward"][:] = orient_reward

        ee_goal_distance = torch.norm(ee_frame_pos - self._goal_pos_w, dim=1)
        self.reward_terms["ee_goal_tracking_penalty"][:] = (
            torch.square(ee_goal_distance * 100) * self.cfg.reward_terms["ee_goal_tracking_penalty"]["weight"]
        )
        self.reward_terms["ee_goal_fine_tracking_reward"][:] = (
            1 - torch.tanh(ee_goal_distance / self.cfg.reward_terms["ee_goal_fine_tracking_reward"]["std"]) ** 2
        )

        self.reward_terms["staying_alive_rew"][:] = (
            self.cfg.reward_terms["staying_alive_rew"]["weight"] * (1.0 - self.reset_terminated.float())
        )[:]

        self.reward_terms["termination_penalty"][:] = (
            self.cfg.reward_terms["termination_penalty"]["weight"] * self.reset_terminated.float()
        )

        # Penalize the rate of change of the actions using L2 squared kernel.
        # action_rate_penalty = torch.sum(torch.square(self.actions), dim=1)
        self.reward_terms["action_rate_penalty"][:] = self.cfg.reward_terms["action_rate_penalty"][
            "weight"
        ] * torch.sum(torch.square(self.actions - self.prev_actions), dim=1)
        # Penalize joint velocities on the articulation using L2 squared kernel.
        self.reward_terms["joint_vel_penalty"][:] = self.cfg.reward_terms["joint_vel_penalty"]["weight"] * torch.sum(
            torch.square(self._robot.data.joint_vel[:, :]), dim=1
        )

        rewards = (
            +self.reward_terms["at_obj_reward"]
            + self.reward_terms["height_reward"]
            + self.reward_terms["orient_reward"]
            # + self.reward_terms["ee_goal_tracking_penalty"]
            + self.reward_terms["ee_goal_fine_tracking_reward"]
            + self.reward_terms["staying_alive_rew"]
            + self.reward_terms["termination_penalty"]
            + self.reward_terms["action_rate_penalty"]
            + self.reward_terms["joint_vel_penalty"]
        )

        self.extras["log"] = {}
        for rew_name, rew in self.reward_terms.items():
            self.extras["log"][f"rew_{rew_name}"] = rew.mean()

        if self.cfg.debug_vis:
            for i, name in enumerate(self.reward_terms.keys()):
                self.visualizers["Rewards"].terms["rewards"][:, i] = self.reward_terms[name]
            self.visualizers["Rewards"].terms["rewards"][:, -1] = rewards

            self.visualizers["Metrics"].terms["ee_height"] = ee_frame_pos[:, 2].reshape(-1, 1)
            self.visualizers["Metrics"].terms["pole_orient_x"] = torch.rad2deg(x).reshape(-1, 1)
            self.visualizers["Metrics"].terms["pole_orient_y"] = torch.rad2deg(y).reshape(-1, 1)
            self.visualizers["Metrics"].terms["obj_ee_distance"] = object_ee_distance.reshape(-1, 1)

        return rewards

    # MARK: reset
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # spawn obj at random position
        obj_pos = self.object.data.default_root_state[env_ids]
        obj_pos[:, :3] += self.scene.env_origins[env_ids]
        self.object.write_root_state_to_sim(obj_pos, env_ids=env_ids)

        # reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        # randomize joints 3 and 4 a little bit
        # joint_pos[:, 2:4] += sample_uniform(-0.0015, 0.0015, (len(env_ids), 2), self.device)

        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # set commands: random target position
        self._goal_pos_w[env_ids, :2] = (
            self.object.data.default_root_state[env_ids, :2]
            + self.scene.env_origins[env_ids, :2]
            + sample_uniform(
                self.cfg.obj_pos_randomization_range[0],
                self.cfg.obj_pos_randomization_range[1],
                (len(env_ids), 2),
                self.device,
            )
        )

        self.prev_actions[env_ids] = 0.0

        # reset sensors
        self.gsmini.reset(env_ids=env_ids)

    # MARK: observations
    def _get_observations(self) -> dict:
        """The position of the object in the robot's root frame."""

        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        ee_frame_orient = euler_xyz_from_quat(ee_quat_curr_b)
        x = wrap_to_pi(ee_frame_orient[0]).unsqueeze(1)  # add dimension for concatenating with other observations
        y = wrap_to_pi(ee_frame_orient[1]).unsqueeze(1)
        z = wrap_to_pi(ee_frame_orient[2]).unsqueeze(1)

        goal_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self._goal_pos_w
        )
        proprio_obs = torch.cat(
            (ee_pos_curr_b, x, y, z, goal_pos_b[:, :2], self.actions),
            dim=-1,
        )
        vision_obs = self.gsmini._data.output["camera_depth"]

        obs = {"proprio_obs": proprio_obs, "vision_obs": vision_obs}

        # self.visualizers["Actions"].terms["actions"][:] = self.actions[:]
        if self.cfg.debug_vis:
            self.visualizers["Observations"].terms["ee_pos"] = ee_pos_curr_b[:, :3]
            self.visualizers["Observations"].terms["ee_rot"][:, :1] = x
            self.visualizers["Observations"].terms["ee_rot"][:, 1:2] = y
            self.visualizers["Observations"].terms["ee_rot"][:, 2:3] = z
            self.visualizers["Observations"].terms["sensor_output"] = vision_obs.clone()
        return {"policy": obs}

    """
    Helper Functions for IK control (from task_space_actions.py of IsaacLab).
    """

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        base_rot = self._robot.data.root_link_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        # obtain quantities from simulation
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        # account for the offset
        # if self.cfg.body_offset is not None:
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
        )

        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        """Computes the geometric Jacobian of the target frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        # read the parent jacobian
        jacobian = self.jacobian_b

        # account for the offset
        # if self.cfg.body_offset is not None:
        # Modify the jacobian to account for the offset
        # -- translational part
        # v_link = v_ee + w_ee x r_link_ee = v_J_ee * q + w_J_ee * q x r_link_ee
        #        = (v_J_ee + w_J_ee x r_link_ee ) * q
        #        = (v_J_ee - r_link_ee_[x] @ w_J_ee) * q
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
        # -- rotational part
        # w_link = R_link_ee @ w_ee
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])

        return jacobian

    ###################################

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = VisualizationMarkersCfg(
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.005,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.5),
                        ),
                    }
                )
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)

        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            # if hasattr(self, "ik_des_pose_visualizer"):
            #     self.ik_des_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        translations = self._goal_pos_w.clone()
        self.goal_pos_visualizer.visualize(translations=translations)
