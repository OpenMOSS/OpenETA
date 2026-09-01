# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Modified version of the original FRANKA_PANDA_CFG of orbit
#
"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG`: Franka Emika Panda robot with Panda hand

Reference: https://github.com/frankaemika/franka_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from tacex_assets import TACEX_ASSETS_DATA_DIR

##
# Configuration
##

FRANKA_PANDA_ARM_SINGLE_GSMINI_TEXTURED_UIPC_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Single_Adapter/uipc_textured_gelpad.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            # "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        # "panda_hand": ImplicitActuatorCfg(
        #     joint_names_expr=["panda_finger_joint.*"],
        #     effort_limit=200.0,
        #     velocity_limit=0.2,
        #     stiffness=2e3,
        #     damping=1e2,
        # ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot with a single GelSight Mini sensor.

Sensor case prim name: `gelsight_mini_case`
Gelpad prim name: `gelsight_mini_gelpad`
"""

FRANKA_PANDA_ARM_SINGLE_GSMINI_TEXTURED_HIGH_PD_UIPC_CFG = FRANKA_PANDA_ARM_SINGLE_GSMINI_TEXTURED_UIPC_CFG.copy()
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.

Sensor case prim name: `gelsight_mini_case`
Gelpad prim name: `gelsight_mini_gelpad`
"""
FRANKA_PANDA_ARM_SINGLE_GSMINI_TEXTURED_HIGH_PD_UIPC_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_ARM_SINGLE_GSMINI_TEXTURED_HIGH_PD_UIPC_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_PANDA_ARM_SINGLE_GSMINI_TEXTURED_HIGH_PD_UIPC_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_PANDA_ARM_SINGLE_GSMINI_TEXTURED_HIGH_PD_UIPC_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_PANDA_ARM_SINGLE_GSMINI_TEXTURED_HIGH_PD_UIPC_CFG.actuators["panda_forearm"].damping = 80.0
