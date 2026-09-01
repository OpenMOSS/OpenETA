# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch

# for Domain Randomization
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
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
    quat_error_magnitude,
    sample_uniform,
    wrap_to_pi,
)
from isaaclab.utils.noise import (
    GaussianNoiseCfg,
    NoiseModelCfg,
    UniformNoiseCfg,
    gaussian_noise,
)

from tacex import GelSightSensor
from tacex.simulation_approaches.gpu_taxim import TaximSimulatorCfg

# from tactile_sim import GsMiniSensorCfg, GsMiniSensor
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_single_uipc import (
    FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_UIPC_CFG,
)
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg

from tacex_tasks.utils import DirectLiveVisualizer
from tacex_uipc import (
    UipcIsaacAttachments,
    UipcIsaacAttachmentsCfg,
    UipcObject,
    UipcObjectCfg,
    UipcRLEnv,
    UipcSimCfg,
)
from tacex_uipc.utils import TetMeshCfg

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
class EventCfg:
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        # workaround for having compliant properties on gelpad:
        # - we set compliant contact as default material
        # - but overwrite all the other materials (ball, sensor case, plate) by using the
        # randomization functions here -> they create new materials without compliant contact
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gelsight_mini_case"),
            "static_friction_range": (0.1, 1.3),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.65, 1.0),
            "num_buckets": 1,
        },
    )
    robot_add_gelpad_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gelsight_mini_gelpad"),
            "mass_distribution_params": (-0.005, 0.005),
            "operation": "add",
        },
    )
    # robot_joint_stiffness_and_damping = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "stiffness_distribution_params": (0.75, 1.5),
    #         "damping_distribution_params": (0.3, 3.0),
    #         "operation": "scale",
    #         "distribution": "log_uniform",
    #     },
    # )
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

    # ball properties
    ball_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.1, 1.0),
            "dynamic_friction_range": (0.25, 1.0),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 250,
        },
    )
    ball_add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    # friction of plate where ball rolls
    plate_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("plate"),
            "static_friction_range": (0.1, 1.0),
            "dynamic_friction_range": (0.1, 1.0),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 250,
        },
    )


@configclass
class BallRollingTactileRGBUipcCfg(DirectRLEnvCfg):
    # viewer settings
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1, -0.5, 0.1)
    viewer.lookat = (-19.4, 18.2, -1.1)

    viewer.origin_type = "env"
    viewer.env_idx = 0
    viewer.resolution = (1280, 720)

    debug_vis = True

    ui_window_class_type = CustomEnvWindow

    # events: EventCfg = EventCfg()

    decimation = 1
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        # device="cpu",
        physx=PhysxCfg(
            enable_ccd=True,  # needed for more stable ball_rolling
            # bounce_threshold_velocity=10000,
            gpu_max_rigid_contact_count=2**23,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=5.0,
            dynamic_friction=5.0,
            restitution=0.0,
        ),
        # physics_material=sim_utils.RigidBodyMaterialCfg(
        #     friction_combine_mode="multiply",
        #     restitution_combine_mode="multiply",
        #     static_friction=1.5,
        #     dynamic_friction=1.5,
        #     restitution=0.1,
        #     compliant_contact_damping=4.25,
        #     compliant_contact_stiffness=1.75
        # ),
    )

    uipc_sim = UipcSimCfg(
        # logger_level="Info"
        ground_height=0.0025,
        contact=UipcSimCfg.Contact(default_friction_ratio=0.5, d_hat=0.0005),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=1, replicate_physics=True)

    # use robot with stiff PD control for better IK tracking
    robot: ArticulationCfg = FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            # joint_pos={
            #     "panda_joint1": 0.0,
            #     "panda_joint2": 0.43,
            #     "panda_joint3": 0.0,
            #     "panda_joint4": -2.37,
            #     "panda_joint5": 0.0,
            #     "panda_joint6": 2.79,
            #     "panda_joint7": 0.741,
            # },
            joint_pos={
                "panda_joint1": -1.02,
                "panda_joint2": 0.3175,
                "panda_joint3": 0.06,
                "panda_joint4": -2.60,
                "panda_joint5": 0.0,
                "panda_joint6": 2.91,
                "panda_joint7": -0.12,
            },
        ),
    )

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")

    # plate
    plate: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ground_plate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/plate.usd",
            scale=(0.6, 0.8, 1),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                kinematic_enabled=True,
            ),
        ),
    )

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
                    rot=(0.0, 0.0, 1.0, 0.0),
                    # rot=(0, 0.92388, -0.38268, 0) # our panda hand asset has rotation from (180,0,-45) -> we subtract 180 for defining the rotation limits
                ),
            ),
        ],
    )

    # rigid body ball
    object: UipcObjectCfg = UipcObjectCfg(
        prim_path="/World/envs/env_.*/uipc_ball",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.25, -0.35, 0.0051 + 0.0025)),  # rot=(0.72,-0.3,0.42,-0.45)
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/ball_wood.usd",
        ),
        mesh_cfg=TetMeshCfg(
            stop_quality=8,
            max_its=100,
            edge_length_r=1 / 5,
            # epsilon_r=0.01
        ),
        constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
            youngs_modulus=100.0,
        ),  # UipcObjectCfg.AffineBodyConstitutionCfg() #
    )

    # simulate the gelpad as uipc mesh
    gelpad_cfg = UipcObjectCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_gelpad",
        mesh_cfg=TetMeshCfg(
            stop_quality=8,
            max_its=100,
            edge_length_r=1 / 5,
            # epsilon_r=0.01
        ),
        constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(youngs_modulus=10),
    )
    gelpad_attachment_cfg = UipcIsaacAttachmentsCfg(
        constraint_strength_ratio=200000.0,
        body_name="gelsight_mini_case",
        debug_vis=False,
    )

    # sensors
    gsmini = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case",
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=0,
            resolution=(32, 32),  # (48, 64),
            data_types=["depth"],
            clipping_range=(
                0.015,
                0.029,
            ),  # (0.024, 0.029), make min value little bit smaller, due to penetration issues
        ),
        device="cuda",
        debug_vis=True,  # for being able to see sensor output in the gui
        # update Taxim cfg
        optical_sim_cfg=TaximSimulatorCfg(
            calib_folder_path=f"{TACEX_ASSETS_DATA_DIR}/Sensors/GelSight_Mini/calibs/640x480",
            gelpad_height=GelSightMiniCfg().gelpad_dimensions.height,
            gelpad_to_camera_min_distance=0.024,
            with_shadow=False,
            tactile_img_res=(32, 32),
            device="cuda",
        ),
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
        data_types=["tactile_rgb"],  # marker_motion
    )

    # noise models
    action_noise_model: NoiseModelCfg = NoiseModelCfg(
        noise_cfg=UniformNoiseCfg(n_min=-0.001, n_max=0.001, operation="add")
    )
    # observation_noise_model =
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    # observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
    #   noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
    #   bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    # )
    # -- currently doesn't work with dictionary observation
    gaussian_noise_cfg: GaussianNoiseCfg = GaussianNoiseCfg(mean=0.0, std=0.002, operation="add")

    # MARK: reward cfg
    reward_cfg = {
        "at_obj_reward": {"weight": 0.25, "min_depth": 0.5, "max_depth": 4.0},
        "centering_error": {"weight": -0.05},
        "off_the_ground_penalty": {"weight": -15, "max_height": 0.025},
        "height_reward": {
            "weight": 0.15,
            "std": 0.4901,
            "alpha": 0.00067,
            "target_height_cm": 1.225,
            "min_height": 0.002,
        },  # target height: 1cm + 0.25cm - 0.125cm
        "orient_reward": {"weight": -1.25},
        # for solving the task
        "ee_goal_tracking": {"weight": 0.75, "std": 0.2, "std_fine": 0.36},
        "obj_goal_tracking": {"weight": 0.75, "std": 0.6},  # 0.2
        # "obj_goal_tracking": {"weight": -0.0108, "w": 0.0482, "v": 0.7870, "alpha": 0.0083},
        "obj_goal_fine_tracking": {"weight": 1.25, "std": 0.2},
        "obj_goal_super_fine_tracking": {"weight": 1.75, "std": 0.08},
        "success_reward": {
            "weight": 5.0,
            "threshold": 0.005,
        },  # 0.0025 we count it as a success when dist obj <-> goal is less than the threshold
        # penalties for nice behavior
        "action_rate_penalty": {"weight": -1e-4},
        "joint_vel_penalty": {"weight": -1e-4},
    }

    # goal_randomization_range_x = [-0.25, 0.25] #[-0.25, 0.25]
    # goal_randomization_range_y = [-0.35, 0.35] #[-0.35, 0.35]
    goal_randomization_range_x = [0.0, 0.5]  # [-0.25, 0.25]
    goal_randomization_range_y = [0.0, 0.7]  # [-0.35, 0.35]
    # env
    episode_length_s = 8.3333 * 2  # 1000 timesteps per episode (dt = 1/60 -> 1500*(1/60)=8.3333*3)
    action_space = 6  # we use relative task_space actions: (dx, dy, dz, droll, dpitch) -> dyaw is omitted
    observation_space = {
        "proprio_obs": 14,  # 16, # 3 for ee pos, 2 for orient (roll, pitch), 2 for init goal-pos (x,y), 5 for actions
        "vision_obs": [32, 32, 3],  # from tactile sensor
    }
    state_space = 0
    action_scale = 0.05  # [cm]

    ball_radius = 0.005  # don't change, because rewards (e.g. height reward) are tuned for this ball size

    x_bounds = (0.2, 0.8)
    y_bounds = (-0.4, 0.4)
    too_far_away_threshold = 0.015
    min_height_threshold = 0.002

    curriculum_cfg = {
        "goal_randomization_range": {
            "min": 0.00,
            "max": 0.00,
            "num_levels": 10,
            "threshold": 550.0,  # if episode reward is this high, then go to next level
        },
        "action_rate_penalty": {
            "min": 0,
            "max": 1e-5,
            "num_levels": 30,
            "threshold": 5500.0,  #
        },
        "joint_vel_penalty": {
            "min": 0,
            "max": 1e-5,
            "num_levels": 30,
            "threshold": 5500.0,  #
        },
    }


class BallRollingTactileRGBUipcEnv(UipcRLEnv):
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

    cfg: BallRollingTactileRGBUipcCfg

    def __init__(self, cfg: BallRollingTactileRGBUipcCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

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

        # ee offset w.r.t panda hand -> based on the asset
        self._offset_pos = torch.tensor([0.0, 0.0, 0.131], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        # self._offset_rot = torch.tensor([0, 0.92388, -0.38268, 0], device=self.device).repeat(self.num_envs, 1)
        # ---

        # create auxiliary variables for computing applied action, observations and rewards
        self.processed_actions = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)

        self._goal_pos_b = torch.zeros((self.num_envs, 3), device=self.device)
        # make height of goal pos fixed
        self._goal_pos_b[:, 2] = self.cfg.ball_radius * 2 + 0.0025  # plate height above ground = 0.0025
        self._goal_orient = torch.zeros(
            (self.num_envs, 4), device=self.device
        )  # goal orient is [0.70711,0,0,0.70711], i.e. upright ee (x=0, y=0, z=90°)
        self._goal_orient[:, 0] = 0.70711
        self._goal_orient[:, 3] = 0.70711

        self._time_out = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.object_ee_distance = torch.zeros((self.num_envs, 1), device=self.device)
        self.ee_goal_distance = torch.zeros((self.num_envs, 1), device=self.device)
        self.obj_goal_distance = torch.zeros((self.num_envs, 1), device=self.device)

        # for training curriculum
        self.curriculum_levels = {
            "goal_randomization_range": 0,
            "action_rate_penalty": 0,
            "joint_vel_penalty": 0,
        }
        self._goal_random_curr = torch.linspace(
            self.cfg.curriculum_cfg["goal_randomization_range"]["min"],
            self.cfg.curriculum_cfg["goal_randomization_range"]["max"],
            self.cfg.curriculum_cfg["goal_randomization_range"]["num_levels"],
            device=self.device,
        )
        self._action_rate_penalty_curr = torch.linspace(
            self.cfg.curriculum_cfg["action_rate_penalty"]["min"],
            self.cfg.curriculum_cfg["action_rate_penalty"]["max"],
            self.cfg.curriculum_cfg["action_rate_penalty"]["num_levels"],
            device=self.device,
        )
        self._joint_vel_penalty_curr = torch.linspace(
            self.cfg.curriculum_cfg["joint_vel_penalty"]["min"],
            self.cfg.curriculum_cfg["joint_vel_penalty"]["max"],
            self.cfg.curriculum_cfg["joint_vel_penalty"]["num_levels"],
            device=self.device,
        )
        # for the curr. level conditions
        self._total_episode_rew = torch.zeros(self.num_envs, device=self.device)

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
            self.visualizers["Observations"].terms["sensor_output"] = torch.zeros((
                self.num_envs,
                self.cfg.observation_space["vision_obs"][0],
                self.cfg.observation_space["vision_obs"][1],
                self.cfg.observation_space["vision_obs"][2],
            ))

            self.visualizers["Rewards"].terms["rewards"] = torch.zeros((self.num_envs, 11))
            self.visualizers["Rewards"].terms_names["rewards"] = [
                "at_obj_reward",
                "off_the_ground_penalty",
                "height_reward",
                "orient_reward",
                # for solving the task
                "ee_goal_tracking",
                "obj_goal_tracking",
                "obj_goal_fine_tracking",
                "obj_goal_super_fine_tracking",
                "success_reward",
                "too_far_penalty",
                "total",
            ]

            self.visualizers["Metrics"].terms["ee_height"] = torch.zeros(self.num_envs)
            self.visualizers["Metrics"].terms["ee_goal_distance"] = torch.zeros(self.num_envs)
            self.visualizers["Metrics"].terms["ee_orient_error"] = torch.zeros(self.num_envs)
            self.visualizers["Metrics"].terms["obj_ee_distance"] = torch.zeros(self.num_envs)
            self.visualizers["Metrics"].terms["obj_goal_distance"] = torch.zeros(self.num_envs)
            self.visualizers["Metrics"].terms["indentation_depth"] = torch.zeros(self.num_envs)

            for vis in self.visualizers.values():
                vis.create_visualizer()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    # MARK: setup
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # self.object = RigidObject(self.cfg.object)
        # self.scene.rigid_objects["object"] = self.object

        plate = RigidObject(self.cfg.plate)
        self.scene.rigid_objects["plate"] = plate

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

        # # Physics Materials
        # gelpad_mat = sim_utils.spawn_rigid_body_material(
        #     prim_path="/World/Materials/gelpad",
        #     cfg=sim_utils.RigidBodyMaterialCfg(
        #         static_friction=1.5,
        #         dynamic_friction=1.5,
        #         restitution=0.75,
        #         compliant_contact_damping=0.5,
        #         compliant_contact_stiffness=0.25
        #     )
        # )
        # sim_utils.bind_physics_material("/World/envs/env_.*/Robot/gelsight_mini_gelpad", "/World/Materials/gelpad")

        # --- UIPC simulation setup ---

        # gelpad simulated via uipc
        self._uipc_gelpad: UipcObject = UipcObject(self.cfg.gelpad_cfg, self.uipc_sim)

        self.object: UipcObject = UipcObject(self.cfg.object, self.uipc_sim)
        self.scene.uipc_objects["object"] = self.object

        # create attachment
        self.attachment = UipcIsaacAttachments(
            self.cfg.gelpad_attachment_cfg, self._uipc_gelpad, self.scene.articulations["robot"]
        )

    # MARK: pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        self.actions[:] = actions.clamp(-1.0, 1.0)
        # preprocess the action and turn it into IK action
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
        obj_pos = self.object.data.root_pos_w - self.scene.env_origins

        out_of_bounds_x = (obj_pos[:, 0] < self.cfg.x_bounds[0]) | (obj_pos[:, 0] > self.cfg.x_bounds[1])
        out_of_bounds_y = (obj_pos[:, 1] < self.cfg.y_bounds[0]) | (obj_pos[:, 1] > self.cfg.y_bounds[1])

        obj_goal_distance = torch.norm(self._goal_pos_b[:, :2] - obj_pos[:, :2], dim=1)
        obj_too_far_away = obj_goal_distance > 0.75

        ee_frame_pos = (
            self._ee_frame.data.target_pos_w[..., 0, :] - self.scene.env_origins
        )  # end-effector positions in world frame: (num_envs, 3)
        ee_too_far_away = torch.norm(obj_pos - ee_frame_pos, dim=1) > self.cfg.too_far_away_threshold

        # reset when ee orient is too large
        ee_frame_orient = euler_xyz_from_quat(self._ee_frame.data.target_quat_source[..., 0, :])
        x = wrap_to_pi(ee_frame_orient[0])
        y = wrap_to_pi(ee_frame_orient[1])
        orient_cond = (torch.abs(x) > math.pi / 4) | (torch.abs(y) > math.pi / 4)

        min_height = ee_frame_pos[:, 2] < self.cfg.min_height_threshold

        reset_cond = out_of_bounds_x | out_of_bounds_y | obj_too_far_away | ee_too_far_away | orient_cond | min_height

        # new goal position, if successful
        # success_env = (obj_goal_distance < self.cfg.success_reward["threshold"]).nonzero(as_tuple=False).squeeze(-1)
        # if len(success_env) > 0:
        #     self._goal_pos_w[success_env, :2] = self.object.data.default_root_state[success_env, :2] + self.scene.env_origins[success_env, :2] + sample_uniform(
        #         self.cfg.obj_pos_randomization_range[0],
        #         self.cfg.obj_pos_randomization_range[1],
        #         (len(success_env), 2),
        #         self.device
        #     )
        #     # reset episode length
        #     self.episode_length_buf[success_env] = 0

        self._time_out = self.episode_length_buf >= self.max_episode_length - 1  # episode length limit

        return reset_cond, self._time_out

    # MARK: reset
    def _reset_idx(self, env_ids: torch.Tensor | None):
        # # log task metrics
        # self.extras["log"] = {
        #     "Metric/ee_obj_error": self.object_ee_distance[env_ids].mean(),
        #     "Metric/ee_goal_error": self.ee_goal_distance[env_ids].mean(),
        #     "Metric/obj_goal_error": self.obj_goal_distance[env_ids].mean(),
        # }

        # don't do full reset for env where time_out, but robot still in contact with ball
        in_contact = self.gsmini.indentation_depth > 0
        partial_reset_env_ids = in_contact & self._time_out

        full_reset_env_ids = (
            (self.reset_buf & torch.logical_not(partial_reset_env_ids)).nonzero(as_tuple=False).squeeze(-1)
        )

        # reset buffers
        super()._reset_idx(env_ids)

        # # spawn obj at initial position
        # obj_pos = self.object.data.default_root_state[full_reset_env_ids]
        # obj_pos[:, :2] += sample_uniform(
        #     -0.00025,
        #     0.00025,
        #     (len(full_reset_env_ids), 2),
        #     self.device
        # )
        # obj_pos[:, :3] += self.scene.env_origins[full_reset_env_ids]
        # self.object.write_root_state_to_sim(obj_pos, env_ids=full_reset_env_ids)

        # reset robot state
        joint_pos = self._robot.data.default_joint_pos[full_reset_env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=full_reset_env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=full_reset_env_ids)

        # set commands: random target position
        # self._goal_pos_b[env_ids, :2] = self.object.data.default_root_state[env_ids, :2] #todo code that properly
        self._goal_pos_b[env_ids, :2] = self.object.data.root_pos_w[env_ids, :2]

        self._goal_pos_b[env_ids, 0] += sample_uniform(
            self.cfg.goal_randomization_range_x[0]
            - self._goal_random_curr[self.curriculum_levels["goal_randomization_range"]],
            self.cfg.goal_randomization_range_x[1]
            + self._goal_random_curr[self.curriculum_levels["goal_randomization_range"]],
            (len(env_ids)),
            self.device,
        )
        self._goal_pos_b[env_ids, 1] += sample_uniform(
            self.cfg.goal_randomization_range_y[0]
            - self._goal_random_curr[self.curriculum_levels["goal_randomization_range"]],
            self.cfg.goal_randomization_range_y[1]
            + self._goal_random_curr[self.curriculum_levels["goal_randomization_range"]],
            (len(env_ids)),
            self.device,
        )

        self.prev_actions[env_ids] = 0.0
        # reset sensors
        self.gsmini.reset(env_ids=env_ids)

        # reset buffers used for curriculum learning
        self._total_episode_rew[env_ids] = 0.0

        # reset uipc objects #todo implement that properly
        self._uipc_gelpad.write_vertex_positions_to_sim(vertex_positions=self._uipc_gelpad.init_vertex_pos)
        self.object.write_vertex_positions_to_sim(vertex_positions=self.object.init_vertex_pos)

    # MARK: rewards
    def _get_rewards(self) -> torch.Tensor:
        # - Reward the agent for reaching the object using tanh-kernel.
        obj_pos_b = self.object.data.root_pos_w - self.scene.env_origins
        ee_frame_pos_b, ee_frame_orient_b = self._compute_frame_pose()

        (self.object_ee_distance, self.ee_goal_distance, self.ee_orient_error, self.obj_goal_distance) = (
            _compute_intermediate_values(
                obj_pos_b, ee_frame_pos_b, ee_frame_orient_b, self._goal_pos_b, self._goal_orient
            )
        )

        (
            at_obj_reward,
            off_the_ground_penalty,
            height_reward,
            orient_reward,
            ee_goal_tracking_reward,
            obj_goal_tracking_reward,
            obj_goal_fine_tracking_reward,
            obj_goal_super_fine_tracking_reward,
            success_reward,
            action_rate_penalty,
            joint_vel_penalty,
            full_reward,
        ) = _compute_rewards(
            self.cfg.reward_cfg["at_obj_reward"],
            self.cfg.reward_cfg["centering_error"],
            self.cfg.reward_cfg["off_the_ground_penalty"],
            self.cfg.reward_cfg["height_reward"],
            self.cfg.reward_cfg["orient_reward"],
            self.cfg.reward_cfg["ee_goal_tracking"],
            self.cfg.reward_cfg["obj_goal_tracking"],
            self.cfg.reward_cfg["obj_goal_fine_tracking"],
            self.cfg.reward_cfg["obj_goal_super_fine_tracking"],
            self.cfg.reward_cfg["success_reward"],
            self.cfg.reward_cfg["action_rate_penalty"],
            self.cfg.reward_cfg["joint_vel_penalty"],
            obj_pos_b,
            ee_frame_pos_b,
            ee_frame_orient_b,
            self.actions,
            self.prev_actions,
            self._robot.data.joint_vel[:, :],
            self.gsmini.indentation_depth,
            self.ee_goal_distance,
            self.ee_orient_error,
            self.obj_goal_distance,
        )

        self._total_episode_rew[:] += full_reward

        self.extras["log"] = {
            "at_obj_reward": at_obj_reward.mean(),
            "off_the_ground_penalty": off_the_ground_penalty.mean(),
            "height_reward": height_reward.mean(),
            "orient_reward": orient_reward.mean(),
            "ee_goal_tracking_reward": ee_goal_tracking_reward.mean(),
            "obj_goal_tracking_reward": obj_goal_tracking_reward.mean(),
            "obj_goal_fine_tracking_reward": obj_goal_fine_tracking_reward.mean(),
            "obj_goal_super_fine_tracking_reward": obj_goal_super_fine_tracking_reward.mean(),
            "success_reward": success_reward.mean(),
            "full_reward": full_reward.mean(),
            # penalties for nice looking behavior
            "action_rate_penalty": action_rate_penalty.mean(),
            "joint_vel_penalty": joint_vel_penalty.mean(),
            "total_episode_rew": self._total_episode_rew.mean(),
            # task metrics
            "Metric/ee_obj_error": self.object_ee_distance.mean(),
            "Metric/ee_goal_error": self.ee_goal_distance.mean(),
            "Metric/ee_orient_error": self.ee_orient_error.mean(),
            "Metric/obj_goal_error": self.obj_goal_distance.mean(),
            "Metric/indentation_depth": self.gsmini.indentation_depth.mean(),
            # curriculum data
            "Curriculum/goal_rand_range": self._goal_random_curr[self.curriculum_levels["goal_randomization_range"]],
        }

        # training curriculum
        if (self._total_episode_rew.mean() > self.cfg.curriculum_cfg["goal_randomization_range"]["threshold"]) and (
            self.curriculum_levels["goal_randomization_range"]
            < self.cfg.curriculum_cfg["goal_randomization_range"]["num_levels"] - 1
        ):
            self.curriculum_levels["goal_randomization_range"] += 1
        elif (
            self._total_episode_rew.mean() < self.cfg.curriculum_cfg["goal_randomization_range"]["threshold"] * 0.90
        ) and (self.curriculum_levels["goal_randomization_range"] > 0):
            self.curriculum_levels["goal_randomization_range"] -= 1

        if (self._total_episode_rew.mean() > self.cfg.curriculum_cfg["action_rate_penalty"]["threshold"]) and (
            self.curriculum_levels["action_rate_penalty"]
            < self.cfg.curriculum_cfg["action_rate_penalty"]["num_levels"] - 1
        ):
            self.curriculum_levels["action_rate_penalty"] += 1
        elif (self._total_episode_rew.mean() < self.cfg.curriculum_cfg["action_rate_penalty"]["threshold"] * 0.90) and (
            self.curriculum_levels["action_rate_penalty"] > 0
        ):
            self.curriculum_levels["action_rate_penalty"] -= 1

        if (self._total_episode_rew.mean() > self.cfg.curriculum_cfg["joint_vel_penalty"]["threshold"]) and (
            self.curriculum_levels["joint_vel_penalty"] < self.cfg.curriculum_cfg["joint_vel_penalty"]["num_levels"] - 1
        ):
            self.curriculum_levels["joint_vel_penalty"] += 1
        elif (self._total_episode_rew.mean() < self.cfg.curriculum_cfg["joint_vel_penalty"]["threshold"] * 0.90) and (
            self.curriculum_levels["joint_vel_penalty"] > 0
        ):
            self.curriculum_levels["joint_vel_penalty"] -= 1

        self.cfg.reward_cfg["action_rate_penalty"]["weight"] -= self._action_rate_penalty_curr[
            self.curriculum_levels["action_rate_penalty"]
        ]
        self.cfg.reward_cfg["joint_vel_penalty"]["weight"] -= self._joint_vel_penalty_curr[
            self.curriculum_levels["joint_vel_penalty"]
        ]

        if self.cfg.debug_vis:
            self.visualizers["Rewards"].terms["rewards"][:, 0] = at_obj_reward
            self.visualizers["Rewards"].terms["rewards"][:, 1] = off_the_ground_penalty
            self.visualizers["Rewards"].terms["rewards"][:, 2] = height_reward
            self.visualizers["Rewards"].terms["rewards"][:, 3] = orient_reward
            self.visualizers["Rewards"].terms["rewards"][:, 4] = ee_goal_tracking_reward
            self.visualizers["Rewards"].terms["rewards"][:, 5] = obj_goal_tracking_reward
            self.visualizers["Rewards"].terms["rewards"][:, 6] = obj_goal_fine_tracking_reward
            self.visualizers["Rewards"].terms["rewards"][:, 7] = obj_goal_super_fine_tracking_reward
            self.visualizers["Rewards"].terms["rewards"][:, 8] = success_reward
            self.visualizers["Rewards"].terms["rewards"][:, 10] = full_reward

            self.visualizers["Metrics"].terms["ee_height"][:] = ee_frame_pos_b[:, 2]
            self.visualizers["Metrics"].terms["ee_goal_distance"][:] = self.ee_goal_distance
            self.visualizers["Metrics"].terms["ee_orient_error"][:] = self.ee_orient_error
            self.visualizers["Metrics"].terms["obj_ee_distance"][:] = self.object_ee_distance
            self.visualizers["Metrics"].terms["obj_goal_distance"][:] = self.obj_goal_distance
            self.visualizers["Metrics"].terms["indentation_depth"][:] = self.gsmini.indentation_depth
        return full_reward

    # MARK: observations
    def _get_observations(self) -> dict:
        """The position and rotation of the robot's endeffector, the (x,y) goal position, actions and camera_depth from the sensor."""

        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        ee_frame_orient = euler_xyz_from_quat(ee_quat_curr_b)
        x = wrap_to_pi(ee_frame_orient[0]).unsqueeze(1)  # add dimension for concatenating with other observations
        y = wrap_to_pi(ee_frame_orient[1]).unsqueeze(1)
        z = wrap_to_pi(ee_frame_orient[2]).unsqueeze(1)

        # # obj position in the robots root frame
        # object_pos_b, _ = subtract_frame_transforms(
        #     self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self.object.data.root_link_pos_w[:, :3]
        # )

        proprio_obs = torch.cat(
            (ee_pos_curr_b, x, y, z, self._goal_pos_b[:, :2], self.actions),
            dim=-1,
        )
        # add noise to proprio_obs:
        proprio_obs = gaussian_noise(proprio_obs, cfg=self.cfg.gaussian_noise_cfg)

        vision_obs = self.gsmini.data.output["tactile_rgb"]
        # print("indent depth ", self.gsmini._indentation_depth[0])
        obs = {"proprio_obs": proprio_obs, "vision_obs": vision_obs}

        # # obs = proprio_obs
        # # change goal_pos for env with long enough episodes
        # env_ids = ((self.episode_length_buf + 1) % int(self.max_episode_length/self.cfg.num_goals) == 0).nonzero(as_tuple=False).squeeze(-1)
        # if len(env_ids) > 0:
        #     # set commands: random target position
        #     self._goal_pos_w[env_ids, :2] = self.object.data.default_root_state[env_ids, :2] + self.scene.env_origins[env_ids, :2] + sample_uniform(
        #         self.cfg.obj_pos_randomization_range[0],
        #         self.cfg.obj_pos_randomization_range[1],
        #         (len(env_ids), 2),
        #         self.device
        #     )

        # self.visualizers["Actions"].terms["actions"][:] = self.actions[:]
        if self.cfg.debug_vis:
            self.visualizers["Observations"].terms["ee_pos"] = ee_pos_curr_b
            self.visualizers["Observations"].terms["ee_rot"][:, :1] = x
            self.visualizers["Observations"].terms["ee_rot"][:, 1:2] = y
            self.visualizers["Observations"].terms["ee_rot"][:, 2:3] = z
            self.visualizers["Observations"].terms["goal"] = self._goal_pos_b[:, :2]
            self.visualizers["Observations"].terms["sensor_output"] = vision_obs

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
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = VisualizationMarkersCfg(
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=self.cfg.reward_cfg["success_reward"]["threshold"],
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.5),
                        ),
                    }
                )
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)

            # if not hasattr(self, "ik_des_pose_visualizer"):
            #     marker_cfg = FRAME_MARKER_CFG.copy()
            #     marker_cfg.markers["frame"].scale = (0.025, 0.025, 0.025)
            #     marker_cfg.prim_path = "/Visuals/Command/ik_des_pose"
            #     self.ik_des_pose_visualizer = VisualizationMarkers(marker_cfg)
            # # set their visibility to true
            # self.ik_des_pose_visualizer.set_visibility(True)

        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            # if hasattr(self, "ik_des_pose_visualizer"):
            #     self.ik_des_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        translations = self._goal_pos_b.clone() + self.scene.env_origins
        translations[:, 2] = self.cfg.ball_radius + 0.0025
        self.goal_pos_visualizer.visualize(translations=translations)

        # ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        # self.ik_des_pose_visualizer.visualize(
        #     translations=ee_pos_curr + self.scene.env_origins,#self._ik_controller.ee_pos_des[:, :3] - self.scene.env_origins,
        #     orientations=ee_quat_curr
        #     )


# compute these values separately for logging
@torch.jit.script
def _compute_intermediate_values(
    obj_pos_b: torch.Tensor,
    ee_frame_pos_b: torch.Tensor,
    ee_frame_orient_b: torch.Tensor,
    goal_pos_b: torch.Tensor,
    goal_orient_b: torch.Tensor,
):
    # for compensating that obj_pos_b is at the center of the ball, but we want the be at the top of the ball
    obj_pos_b[:, 2] += 0.005  # ball has diameter of 1cm -> r=0.005m, plate height (above ground)=0.0025
    object_ee_distance = torch.norm(obj_pos_b - ee_frame_pos_b, dim=1)

    ee_goal_distance = torch.norm(ee_frame_pos_b[:, :2] - goal_pos_b[:, :2], dim=1)
    ee_orient_error = quat_error_magnitude(ee_frame_orient_b, goal_orient_b)

    obj_goal_distance = torch.norm(obj_pos_b[:, :2] - goal_pos_b[:, :2], dim=1)
    return (object_ee_distance, ee_goal_distance, ee_orient_error, obj_goal_distance)


@torch.jit.script
def _compute_rewards(
    at_obj_reward_cfg: dict[str, float],
    centering_error_cfg: dict[str, float],
    off_the_ground_penalty_cfg: dict[str, float],
    height_reward_cfg: dict[str, float],
    orient_reward_cfg: dict[str, float],
    ee_goal_tracking_cfg: dict[str, float],
    obj_goal_tracking_cfg: dict[str, float],
    obj_goal_fine_tracking_cfg: dict[str, float],
    obj_goal_super_fine_tracking_cfg: dict[str, float],
    success_reward_cfg: dict[str, float],
    action_rate_penalty_cfg: dict[str, float],
    joint_vel_penalty_cfg: dict[str, float],
    obj_pos_b: torch.Tensor,
    ee_frame_pos_b: torch.Tensor,
    ee_frame_orient_b: torch.Tensor,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    joint_vel: torch.Tensor,
    # intermediate values
    indentation_depth: torch.Tensor,
    ee_goal_distance: torch.Tensor,
    ee_orient_error: torch.Tensor,
    obj_goal_distance: torch.Tensor,
):
    # for giving agent incentive to touch the obj
    at_obj_reward = torch.where(
        (indentation_depth > at_obj_reward_cfg["min_depth"]) & (indentation_depth < at_obj_reward_cfg["max_depth"]),
        at_obj_reward_cfg["weight"],
        0.0,
    )
    # trying to keep ball in center of gelpad
    center_error = torch.square(torch.norm(obj_pos_b[:, :2] * 100 - ee_frame_pos_b[:, :2] * 100, dim=1))
    center_error *= centering_error_cfg["weight"]

    # penalty for preventing the ball from jumping
    off_the_ground_penalty = torch.where(
        obj_pos_b[:, 2] > off_the_ground_penalty_cfg["max_height"], off_the_ground_penalty_cfg["weight"], 0.0
    )

    height_diff = height_reward_cfg["target_height_cm"] - ee_frame_pos_b[:, 2] * 100.0
    # height_reward = -(
    #     height_reward_cfg["w"]*height_diff**2
    #     + height_reward_cfg["v"]*torch.log(height_diff**2 + height_reward_cfg["alpha"])
    # )#.clamp(-1,1)
    # # penalize ee being too close to ground
    # height_reward = torch.where(
    #     (ee_frame_pos_b[:, 2] <= height_reward_cfg["min_height"]),
    #     height_reward-10,
    #     height_reward
    # )
    height_reward = 1 - torch.tanh((height_diff) / height_reward_cfg["std"])
    height_reward *= height_reward_cfg["weight"]

    # ee should be upright
    # orient_reward = torch.where(
    #     (obj_goal_distance < success_reward_cfg["threshold"])
    #     & (object_ee_distance < at_obj_reward_cfg["minimal_distance"]),
    #     ee_orient_error * orient_reward_cfg["weight"]*1.5, # especially when goal position is reached
    #     ee_orient_error * orient_reward_cfg["weight"]
    # )
    ee_frame_orient = euler_xyz_from_quat(ee_frame_orient_b)
    x = wrap_to_pi(ee_frame_orient[0])
    y = wrap_to_pi(ee_frame_orient[1])
    orient_reward = torch.where(
        (torch.abs(x) < math.pi / 10) & (torch.abs(y) < math.pi / 10),
        0.0,
        1.0 * orient_reward_cfg["weight"],  # 8  # 8
    )
    # orient_reward = (
    #     (torch.abs(x))**2
    #     + (torch.abs(y))**2
    # )
    # orient_reward = torch.where(
    #     (torch.abs(x) < math.pi/8)
    #     & (torch.abs(y) < math.pi/8),
    #     orient_reward,
    #     orient_reward*1.5
    # )
    # orient_reward *= orient_reward_cfg["weight"]

    ee_goal_tracking_reward = (
        1
        - torch.tanh((ee_goal_distance) / ee_goal_tracking_cfg["std"])
        # - torch.tanh(ee_goal_distance / ee_goal_tracking_cfg["std_fine"])
    )
    ee_goal_tracking_reward *= ee_goal_tracking_cfg["weight"]
    # ee_goal_tracking_reward -= ee_goal_tracking_cfg["weight"]

    # obj_goal_tracking_reward = (obj_goal_distance)**2
    obj_goal_tracking_reward = 1 - torch.tanh((obj_goal_distance) / obj_goal_tracking_cfg["std"])
    obj_goal_tracking_reward *= obj_goal_tracking_cfg["weight"]

    obj_goal_fine_tracking_reward = 1 - torch.tanh((obj_goal_distance) / obj_goal_fine_tracking_cfg["std"])
    obj_goal_fine_tracking_reward *= obj_goal_fine_tracking_cfg["weight"]

    obj_goal_super_fine_tracking_reward = (
        1 - torch.tanh((obj_goal_distance) / obj_goal_super_fine_tracking_cfg["std"]) ** 2
    )
    obj_goal_super_fine_tracking_reward *= obj_goal_super_fine_tracking_cfg["weight"]

    success_reward = torch.where(
        (obj_goal_distance < success_reward_cfg["threshold"])
        & (indentation_depth > at_obj_reward_cfg["min_depth"])
        & (indentation_depth < at_obj_reward_cfg["max_depth"]),
        1.0 * success_reward_cfg["weight"],
        0.0,
    )

    # Penalize the rate of change of the actions using L2 squared kernel.
    # action_rate_penalty = torch.sum(torch.square(self.actions), dim=1)
    action_rate_penalty = torch.sum(torch.square(actions - prev_actions), dim=1) * action_rate_penalty_cfg["weight"]
    # Penalize joint velocities on the articulation using L2 squared kernel.
    joint_vel_penalty = torch.sum(torch.square(joint_vel), dim=1) * joint_vel_penalty_cfg["weight"]

    full_reward = (
        at_obj_reward
        + off_the_ground_penalty
        + center_error
        + orient_reward
        # + ee_goal_tracking_reward
        + obj_goal_tracking_reward
        + obj_goal_fine_tracking_reward
        + obj_goal_super_fine_tracking_reward
        + success_reward
        + action_rate_penalty
        + joint_vel_penalty
    )

    return (
        at_obj_reward,
        off_the_ground_penalty,
        height_reward,
        orient_reward,
        ee_goal_tracking_reward,
        obj_goal_tracking_reward,
        obj_goal_fine_tracking_reward,
        obj_goal_super_fine_tracking_reward,
        success_reward,
        action_rate_penalty,
        joint_vel_penalty,
        full_reward,
    )
