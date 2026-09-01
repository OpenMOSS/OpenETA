# Copyright (c) 2022-2023, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Modified version of the original UR5E of Isaac Lab
#
"""Configuration for the UR5E with gf225 gripper finger.

Reference: https://vit.ai
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from assets import VITAI_ASSETS_DATA_DIR

# todo find a good way to save the prim path of the sensor for the user?
# -> currently, we need to look into the asset to figure out the prim name (in this case its /gelsight_mini_case)
UR5E_ARM_GF225_GRIPPER_RIGID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{VITAI_ASSETS_DATA_DIR}/robots/UR/ur5e_gripper_assemble/assemble_1.usd",
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
            "shoulder_pan_joint": 3.4136,         
            "shoulder_lift_joint": -1.6289,   
            "elbow_joint": -1.9204,           
            "wrist_1_joint": -1.1709,      
            "wrist_2_joint": 1.5645,            
            "wrist_3_joint": 0.2763,          
            "finger_0_joint": 0.025,         
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            stiffness=80.0,
            damping=4.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_.*"],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.2,
            stiffness=2e3,
            damping=1e2,
        )
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of ur5e robot with a Gripper and two Vitai gf225 sensors.

The gelpads are simulated via PhysX and rigid.

Sensor case prim names:
- `finger_case_0`
- `finger_case_1`

Gelpad prim names:
- `gelpad_0`
- `gelpad_1`
"""


UR5E_ARM_GF225_GRIPPER_HIGH_PD_RIGID_CFG = UR5E_ARM_GF225_GRIPPER_RIGID_CFG.copy()
"""Configuration of ur5e robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.

Sensor case prim names:
- `finger_case_0`
- `finger_case_1`

Gelpad prim names:
- `gelpad_0`
- `gelpad_1`
"""
UR5E_ARM_GF225_GRIPPER_HIGH_PD_RIGID_CFG.spawn.rigid_props.disable_gravity = True
UR5E_ARM_GF225_GRIPPER_HIGH_PD_RIGID_CFG.actuators["arm"].stiffness = 10000000000.0
UR5E_ARM_GF225_GRIPPER_HIGH_PD_RIGID_CFG.actuators["arm"].damping = 80.0
UR5E_ARM_GF225_GRIPPER_HIGH_PD_RIGID_CFG.actuators["gripper"].stiffness = 400.0
UR5E_ARM_GF225_GRIPPER_HIGH_PD_RIGID_CFG.actuators["gripper"].damping = 80.0
