"""
TacEx Simple Franka Pose Controller
简洁的机械臂位姿控制器

功能:
    - 给定目标位姿(位置+四元数), Franka机械臂末端执行器运动到目标位置
    - 使用TacEx内置的DifferentialIK控制器

使用示例:
    python drive_franka_pose_simple.py --target_pos 0.5 0.2 0.3 --target_quat 0 0 0 1
    python drive_franka_pose_simple.py --target_pos 0.4 0.1 0.4 --target_quat 0.707 0 0 0.707
"""

from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

# 解析命令行参数
parser = argparse.ArgumentParser(
    description="Control Franka arm to reach target pose using minimal implementation"
)
parser.add_argument("--target_pos", nargs=3, type=float, default=[0.5, 0.2, 0.3], 
                   help="Target position [x, y, z]")
parser.add_argument("--target_quat", nargs=4, type=float, default=[0, 0, 0, 1], 
                   help="Target quaternion [qx, qy, qz, qw]")
parser.add_argument("--hold_time", type=float, default=3.0, 
                   help="Time to hold position after reaching target (seconds)")
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
# 启用渲染相关选项
args_cli.enable_cameras = True
args_cli.headless = False

# 启动Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import traceback
import carb

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass

from tacex_assets.robots.franka.franka_gsmini_single_rigid import (
    FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_RIGID_CFG,
)


@configclass
class PoseControllerCfg:
    """最小化位姿控制器配置"""
    
    # 仿真设置
    sim_dt: float = 1.0 / 60.0  # 60Hz
    decimation: int = 1         # 控制频率倍数
    
    # 场景设置
    num_envs: int = 1
    env_spacing: float = 2.0
    
    # IK控制器设置
    ik_controller_cfg = DifferentialIKControllerCfg(
        command_type="pose", 
        use_relative_mode=False, 
        ik_method="svd"
    )
    
    # 容忍度设置
    position_tolerance: float = 0.002   # 2mm
    rotation_tolerance: float = 0.02    # ~1.1度


class MinimalPoseController:
    """简洁的Franka位姿控制器"""
    
    def __init__(self, cfg: PoseControllerCfg):
        self.cfg = cfg
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # 初始化仿真
        self._setup_simulation()
        
        # 创建场景
        self._setup_scene() 
        
        # 初始化机器人（基础部分，不访问PhysX属性）
        self._setup_robot_basic()
        
        # 状态变量
        self.target_pose = torch.zeros((self.cfg.num_envs, 7), device=self.device)
        self.step_count = 0
        self._robot_initialized = False
        
        print("[INFO] 位姿控制器初始化完成")
    
    def _setup_simulation(self):
        """设置仿真环境"""
        sim_cfg = SimulationCfg(
            dt=self.cfg.sim_dt,
            render_interval=self.cfg.decimation
        )
        
        # 创建仿真上下文
        self.sim = SimulationContext(sim_cfg)
        
        # 设置设备
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)
            
        print(f"[INFO] 仿真初始化完成")
    
    def _setup_scene(self):
        """设置场景"""
        scene_cfg = InteractiveSceneCfg(
            num_envs=self.cfg.num_envs,
            env_spacing=self.cfg.env_spacing,
            replicate_physics=True
        )
        
        self.scene = InteractiveScene(scene_cfg)
        
        # 添加地面
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)
        
        # 添加光照
        light_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)
        light_cfg.func("/World/light", light_cfg)
        
        print("[INFO] 场景创建完成")
    
    def _setup_robot_basic(self):
        """设置机器人基础部分"""
        robot_cfg = FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_RIGID_CFG.replace(
            prim_path="/World/envs/env_.*/Robot"
        )
        
        self._robot = Articulation(robot_cfg)
        self.scene.articulations["robot"] = self._robot
        self.scene.clone_environments(copy_from_source=False)
        
        # 末端执行器偏移
        self._offset_pos = torch.tensor([0.0, 0.0, 0.131], device=self.device).repeat(self.cfg.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.cfg.num_envs, 1)
        
        print("[INFO] 机器人创建完成")
    
    def _setup_robot_properties(self):
        """设置机器人属性"""
        body_ids, body_names = self._robot.find_bodies("panda_hand")
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        self._jacobi_body_idx = self._body_idx - 1
    
    def _setup_ik_controller(self):
        """设置IK控制器"""
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, 
            num_envs=self.cfg.num_envs, 
            device=self.device
        )
        
        self.ik_commands = torch.zeros((self.cfg.num_envs, 7), device=self.device)
        print("[INFO] IK控制器初始化完成")
    
    def start_simulation(self):
        """启动仿真"""
        print("[INFO] 启动仿真...")
        self.sim.reset()
        self.scene.update(dt=self.cfg.sim_dt)
        
        self._setup_robot_properties()
        self._setup_ik_controller()
        
        self._robot_initialized = True
        print("[INFO] 仿真就绪")
    
    def reset(self):
        """重置环境"""
        joint_pos = self._robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)
        
        self._robot.set_joint_position_target(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel)
        
        self.step_count = 0
    
    def set_target_pose(self, target_pos: list, target_quat: list, env_idx: int = 0):
        """设置目标位姿"""
        self.target_pose[env_idx, :3] = torch.tensor(target_pos, device=self.device)
        self.target_pose[env_idx, 3:] = torch.tensor(target_quat, device=self.device)
        
        self.ik_commands[env_idx, :3] = torch.tensor(target_pos, device=self.device)
        self.ik_commands[env_idx, 3:] = torch.tensor(target_quat, device=self.device)
        
        print(f"[INFO] 设置目标位姿: pos={target_pos}, quat={target_quat}")
    
    def step(self):
        """执行一步仿真"""
        if not self._robot_initialized:
            raise RuntimeError("机器人未初始化")
            
        self._ik_controller.set_command(self.ik_commands)
        self._apply_ik_control()
        
        self.scene.write_data_to_sim()
        self.sim.step(render=True)
        simulation_app.update()
        self.scene.update(dt=self.cfg.sim_dt)
        
        self.step_count += 1
    
    def _apply_ik_control(self):
        """应用IK控制"""
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]
        
        if ee_pos_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
            
        self._robot.set_joint_position_target(joint_pos_des)

    def check_pose_reached(self, env_idx: int = 0) -> tuple:
        """检查是否到达目标位姿"""
        if not self._robot_initialized:
            return False, float('inf'), float('inf')
            
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        
        pos_error = torch.norm(ee_pos_curr[env_idx] - self.target_pose[env_idx, :3]).item()
        quat_dot = torch.sum(ee_quat_curr[env_idx] * self.target_pose[env_idx, 3:]).item()
        rot_error = 1.0 - abs(quat_dot)
        
        pos_ok = pos_error < self.cfg.position_tolerance
        rot_ok = rot_error < self.cfg.rotation_tolerance
        
        return pos_ok and rot_ok, pos_error, rot_error
    
    def get_current_pose(self, env_idx: int = 0) -> tuple:
        """获取当前末端执行器位姿"""
        if not self._robot_initialized:
            return np.zeros(3), np.array([0, 0, 0, 1])
            
        ee_pos, ee_quat = self._compute_frame_pose()
        return ee_pos[env_idx].cpu().numpy(), ee_quat[env_idx].cpu().numpy()
    
    def close(self):
        """关闭控制器"""
        pass
    
    
    # IK计算辅助函数
    @property
    def jacobian_w(self) -> torch.Tensor:
        """世界坐标系下的雅可比矩阵"""
        return self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]

    @property
    def jacobian_b(self) -> torch.Tensor:
        """基座坐标系下的雅可比矩阵"""
        jacobian = self.jacobian_w
        base_rot = self._robot.data.root_link_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """计算目标frame在根坐标系中的位姿

        Returns:
            (position, quaternion) 元组
        """
        # 获取仿真数据
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        
        # 计算相对位姿
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        
        # 考虑偏移
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
        )

        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self) -> torch.Tensor:
        """计算目标frame在根坐标系中的几何雅可比矩阵"""
        # 读取父body的雅可比矩阵
        jacobian = self.jacobian_b

        # 考虑偏移的影响
        # 平移部分: v_link = v_ee + w_ee x r_link_ee
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
        
        # 旋转部分: w_link = R_link_ee @ w_ee  
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])

        return jacobian


def main():
    """主函数"""
    print(f"[INFO] 启动Franka位姿控制器")
    print(f"[INFO] 目标位置: {args_cli.target_pos}")
    print(f"[INFO] 目标四元数: {args_cli.target_quat}")
    
    target_pos = np.array(args_cli.target_pos)
    target_quat = np.array(args_cli.target_quat)
    
    # 验证四元数
    quat_norm = np.linalg.norm(target_quat)
    if quat_norm < 1e-6:
        print(f"[ERROR] 无效的四元数: {target_quat}")
        return
    
    # 标准化四元数
    target_quat = target_quat / quat_norm
    args_cli.target_pos = target_pos.tolist()
    args_cli.target_quat = target_quat.tolist()
    
    try:
        cfg = PoseControllerCfg()
        controller = MinimalPoseController(cfg)
        
        controller.start_simulation()
        controller.reset()
        controller.set_target_pose(args_cli.target_pos, args_cli.target_quat)
        
        max_steps = 1200  # 20秒
        print("[INFO] 开始运动到目标位姿...")
        
        reached = False
        for step in range(max_steps):
            controller.step()
            
            # 每秒检查一次进度
            if step % 60 == 0:
                reached, pos_error, rot_error = controller.check_pose_reached()
                if reached:
                    print(f"[SUCCESS] 步数 {step}: 成功到达目标位姿!")
                    break
                else:
                    print(f"[INFO] 步数 {step}: 位置误差 {pos_error:.4f}m, 旋转误差 {rot_error:.4f}")
        
        if not reached:
            print("[INFO] 未能完全到达目标位姿")
        
        # 保持位置
        hold_steps = int(args_cli.hold_time * 60)
        print(f"[INFO] 保持位置 {args_cli.hold_time} 秒...")
        for _ in range(hold_steps):
            controller.step()
        
        # 显示最终位姿
        final_pos, final_quat = controller.get_current_pose()
        print(f"[INFO] 最终位置: {final_pos}")
        print(f"[INFO] 最终四元数: {final_quat}")
        
        print("[INFO] 任务完成")
        controller.close()
        
    except Exception as err:
        carb.log_error(f"执行错误: {err}")
        carb.log_error(traceback.format_exc())
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # 关闭仿真应用
        simulation_app.close()