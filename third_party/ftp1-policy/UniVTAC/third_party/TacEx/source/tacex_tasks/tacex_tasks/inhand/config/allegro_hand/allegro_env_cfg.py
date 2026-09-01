# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from tacex_assets import GELSIGHT_MINI_TAXIM_FOTS_CFG

##
# Pre-defined configs
##
# from isaaclab_assets import ALLEGRO_HAND_CFG  # isort: skip
from tacex_assets.robots.allegro_gsmini import ALLEGRO_HAND_GSMINI_CFG

# import isaaclab_tasks.manager_based.manipulation.inhand.inhand_env_cfg as inhand_env_cfg
from ...inhand_env_cfg import InHandObjectEnvCfg, ObservationsCfg


@configclass
class AllegroCubeEnvCfg(InHandObjectEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene due to tactile simulation (FOTS does not scale well)
        self.scene.num_envs = 50

        # switch robot to allegro hand
        self.scene.robot = ALLEGRO_HAND_GSMINI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # GelSight Minis
        self.scene.gsmini_ring = GELSIGHT_MINI_TAXIM_FOTS_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot/Case_ring_3",
            debug_vis=True,
            # add FrameTransformer for FOTS simulation
            marker_motion_sim_cfg=GELSIGHT_MINI_TAXIM_FOTS_CFG.marker_motion_sim_cfg.replace(
                frame_transformer_cfg=FrameTransformerCfg(
                    prim_path="/World/envs/env_.*/Robot/Gelpad_ring_3",  # cannot use ENV_REGEX_NS here, cause we spawn the FrameTransformer manually during sensor init
                    target_frames=[FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/object")],
                    debug_vis=False,
                ),
            ),
        )
        self.scene.gsmini_middle = GELSIGHT_MINI_TAXIM_FOTS_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot/Case_middle_3",
            debug_vis=True,
            # add FrameTransformer for FOTS simulation
            marker_motion_sim_cfg=GELSIGHT_MINI_TAXIM_FOTS_CFG.marker_motion_sim_cfg.replace(
                frame_transformer_cfg=FrameTransformerCfg(
                    prim_path="/World/envs/env_.*/Robot/Gelpad_middle_3",  # cannot use ENV_REGEX_NS here, cause we spawn the FrameTransformer manually during sensor init
                    target_frames=[FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/object")],
                    debug_vis=False,
                ),
            ),
        )
        self.scene.gsmini_index = GELSIGHT_MINI_TAXIM_FOTS_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot/Case_index_3",
            debug_vis=True,
            # add FrameTransformer for FOTS simulation
            marker_motion_sim_cfg=GELSIGHT_MINI_TAXIM_FOTS_CFG.marker_motion_sim_cfg.replace(
                frame_transformer_cfg=FrameTransformerCfg(
                    prim_path="/World/envs/env_.*/Robot/Gelpad_index_3",  # cannot use ENV_REGEX_NS here, cause we spawn the FrameTransformer manually during sensor init
                    target_frames=[FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/object")],
                    debug_vis=False,
                ),
            ),
        )
        self.scene.gsmini_thumb = GELSIGHT_MINI_TAXIM_FOTS_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot/Case_thumb_3",
            debug_vis=True,
            # add FrameTransformer for FOTS simulation
            marker_motion_sim_cfg=GELSIGHT_MINI_TAXIM_FOTS_CFG.marker_motion_sim_cfg.replace(
                frame_transformer_cfg=FrameTransformerCfg(
                    prim_path="/World/envs/env_.*/Robot/Gelpad_thumb_3",  # cannot use ENV_REGEX_NS here, cause we spawn the FrameTransformer manually during sensor init
                    target_frames=[FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/object")],
                    debug_vis=False,
                ),
            ),
        )

        # switch observation group to no velocity group
        self.observations.policy = ObservationsCfg.TactileObsGroupCfg()


@configclass
class AllegroCubeEnvCfg_PLAY(AllegroCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove termination due to timeouts
        self.terminations.time_out = None


##
# Environment configuration with no velocity observations.
##


@configclass
class AllegroCubeNoVelObsEnvCfg(AllegroCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch observation group to no velocity group
        self.observations.policy = ObservationsCfg.NoVelocityKinematicObsGroupCfg()


@configclass
class AllegroCubeNoVelObsEnvCfg_PLAY(AllegroCubeNoVelObsEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove termination due to timeouts
        self.terminations.time_out = None
