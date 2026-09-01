from .actor import *
from .transforms import *

from copy import deepcopy

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tacex_uipc import UipcInteractiveScene
    from .._base_task import BaseTask

# 世界坐标euler角
# t3d.euler.quat2euler(quat) = theta_x, theta_y, theta_z
# theta_y 控制俯仰角，theta_z控制垂直桌面平面上的旋转
GRASP_DIRECTION_DIC = {
    "left": [0, 0, 0, -1],
    "front_left": [-0.383, 0, 0, -0.924],
    "front": [-0.707, 0, 0, -0.707],
    "front_right": [-0.924, 0, 0, -0.383],
    "right": [-1, 0, 0, 0],
    "top_down": [-0.5, 0.5, -0.5, -0.5],
    "down_right": [-0.707, 0, -0.707, 0],
    "down_left": [0, 0.707, 0, -0.707],
    "top_down_little_left": [-0.353523, 0.61239, -0.353524, -0.61239],
    "top_down_little_right": [-0.61239, 0.353523, -0.61239, -0.353524],
    "left_arm_perf": [-0.853532, 0.146484, -0.353542, -0.3536],
    "right_arm_perf": [-0.353518, 0.353564, -0.14642, -0.853568],
}

class Action:
    action: Literal["move", "gripper", "all"]
    target_pose: list = None
    target_gripper_pos: float = None

    def __init__(
        self,
        action: Literal["move", "open", "close", "gripper", "all"],
        target_pose: Pose = None,
        target_gripper_pos: float = None,
        **args,
    ):
        if action == "move":
            self.action = "move"
            assert (target_pose is not None), "target_pose cannot be None for move action."
            self.target_pose = target_pose
        elif action == "open":
            self.action = "gripper"
            self.target_gripper_pos = (target_gripper_pos if target_gripper_pos is not None else 1.0)
        elif action == "close":
            self.action = "gripper"
            self.target_gripper_pos = (target_gripper_pos if target_gripper_pos is not None else 0.0)
        elif action == "gripper":
            self.action = "gripper"
            self.target_gripper_pos = target_gripper_pos
        elif action == "all":
            self.action = "all"
            assert (target_pose is not None), "target_pose cannot be None for all action."
            assert (target_gripper_pos is not None), "target_gripper_pos cannot be None for all action."
            self.target_pose = target_pose
            self.target_gripper_pos = target_gripper_pos
        else:
            raise ValueError(f"Invalid action: {action}. Must be 'move', 'open', 'close', 'gripper', or 'all'.")
        
        if self.action == "gripper":
            assert (self.target_gripper_pos is not None), "target_gripper_pos cannot be None for gripper action."
        
        self.args = args

    def __str__(self):
        result = f"{self.action}"
        if self.action == "move":
            result += f"({self.target_pose})"
        elif self.action == "gripper":
            result += f"({self.target_gripper_pos})"
        elif self.action == "all":
            result += f"({self.target_pose}, {self.target_gripper_pos})"
        if self.args:
            result += f"    {self.args}"
        return result


class Atom:
    def __init__(self, task: 'BaseTask'):
        self.task = task
        self.robot = self.task._robot_manager

    def get_grasp_pose(
        self,
        actor: Actor,
        contact_point_id: int = 0,
        pre_dis: float = 0.0,
    ) -> Pose:
        """
        Obtain the grasp pose through the marked grasp point.
        - actor: The instance of the object to be grasped.
        - arm_tag: The arm to be used, either "left" or "right".
        - pre_dis: The distance in front of the grasp point.
        - contact_point_id: The index of the grasp point.
        """
        if not self.task.plan_success:
            return None
        
        res_pose = actor.get_point('contact', contact_point_id, 'pose').add_bias([0, 0, -pre_dis])
        return res_pose
    
    def choose_grasp_pose(
        self,
        actor: Actor,
        pre_dis=0.1,
        target_dis=0,
        contact_point_id: list | float = None,
    ) -> tuple[Pose, Pose]:
        """
        Test the grasp pose function.
        - actor: The actor to be grasped.
        - arm_tag: The arm to be used for grasping, either "left" or "right".
        - pre_dis: The distance in front of the grasp point, default is 0.1.
        """
        if not self.task.plan_success:
            return None, None
        
        res_pre_top_down_pose = None
        res_top_down_pose = None
        dis_top_down = 1e9
        res_pre_side_pose = None
        res_side_pose = None
        dis_side = 1e9
        res_pre_pose = None
        res_pose = None
        dis = 1e9

        pref_direction = self.robot.get_grasp_perfect_direction()

        if contact_point_id is not None:
            if type(contact_point_id) != list:
                contact_point_id = [contact_point_id]
            contact_point_id = [(i, None) for i in contact_point_id]
        else:
            contact_point_id = actor.iter_point('contact')

        for i, _ in contact_point_id:
            pre_pose = self.get_grasp_pose(actor, contact_point_id=i, pre_dis=pre_dis)

            if pre_pose is None:
                continue
            pose = pre_pose.add_bias([0, 0, pre_dis-target_dis])
            now_dis_top_down = cal_quat_dis(
                pose.q,
                GRASP_DIRECTION_DIC['top_down'],
            )
            now_dis_side = cal_quat_dis(pose.q, GRASP_DIRECTION_DIC[pref_direction])

            if res_pre_top_down_pose is None or now_dis_top_down < dis_top_down:
                res_pre_top_down_pose = pre_pose
                res_top_down_pose = pose
                dis_top_down = now_dis_top_down

            if res_pre_side_pose is None or now_dis_side < dis_side:
                res_pre_side_pose = pre_pose
                res_side_pose = pose
                dis_side = now_dis_side

            now_dis = 0.7 * now_dis_top_down + 0.3 * now_dis_side
            if res_pre_pose is None or now_dis < dis:
                res_pre_pose = pre_pose
                res_pose = pose
                dis = now_dis

        if dis_top_down < 0.15:
            return res_pre_top_down_pose, res_top_down_pose
        if dis_side < 0.15:
            return res_pre_side_pose, res_side_pose
        return res_pre_pose, res_pose
    
    def grasp_actor(
        self,
        actor: Actor,
        pre_dis=0.1,
        dis=0,
        gripper_pos=0.0,
        contact_point_id: list | float = None,
        is_close: bool = True
    ):
        if not self.task.plan_success:
            return None
        
        pre_grasp_pose, grasp_pose = self.choose_grasp_pose(
            actor,
            pre_dis=pre_dis,
            target_dis=dis,
            contact_point_id=contact_point_id,
        )
        pre_grasp_pose = self.robot.gripper_center_to_ee(pre_grasp_pose)
        grasp_pose = self.robot.gripper_center_to_ee(grasp_pose)

        actions = [Action("move", target_pose=grasp_pose, pre_dis=pre_dis-dis)]
        if is_close:
            actions.extend(self.close_gripper(gripper_pos))
        return actions
    
    def get_place_pose(
        self,
        actor: Actor,
        target_pose: Pose,
        constrain: Literal["free", "align"] = "align",
        align_axis: list[np.ndarray] | np.ndarray | list = None,
        actor_axis: np.ndarray | list = [1, 0, 0],
        actor_axis_type: Literal["actor", "world"] = "actor",
        functional_point_id: int = None,
        pre_dis: float = 0.1,
        pre_dis_axis: Literal["grasp", "fp"] | np.ndarray | list = "grasp",
    ):
        if not self.task.plan_success:
            return None

        actor_matrix = actor.get_pose().to_transformation_matrix()
        if functional_point_id is not None:
            place_start_pose = actor.get_point('functional', functional_point_id, "pose")
        else:
            place_start_pose = actor.get_pose()
        end_effector_pose:Pose = self.robot.get_ee_pose()

        place_pose = get_place_pose(
            place_start_pose,
            target_pose,
            constrain=constrain,
            actor_axis=actor_axis,
            actor_axis_type=actor_axis_type,
            align_axis=align_axis,
        )
        start2target = (place_pose.to_transformation_matrix()[:3, :3]
                        @ place_start_pose.to_transformation_matrix()[:3, :3].T)
        target_point = (start2target @ (actor_matrix[:3, 3] - place_start_pose.p).reshape(3, 1)).reshape(3) + np.array(
            place_pose[:3])

        ee_pose_matrix = t3d.quaternions.quat2mat(end_effector_pose[-4:])
        target_grasp_matrix = start2target @ ee_pose_matrix

        res_matrix = np.eye(4)
        res_matrix[:3, 3] = actor_matrix[:3, 3] - end_effector_pose[:3]
        res_matrix[:3, 3] = np.linalg.inv(ee_pose_matrix) @ res_matrix[:3, 3]
        target_grasp_qpose = t3d.quaternions.mat2quat(target_grasp_matrix)

        grasp_bias = target_grasp_matrix @ res_matrix[:3, 3]
        if pre_dis_axis == "grasp":
            target_dis_vec = target_grasp_matrix @ res_matrix[:3, 3]
            target_dis_vec /= np.linalg.norm(target_dis_vec)
        else:
            target_pose_mat = target_pose.to_transformation_matrix()
            if pre_dis_axis == "fp":
                pre_dis_axis = [0.0, 0.0, 1.0]
            pre_dis_axis = np.array(pre_dis_axis)
            pre_dis_axis /= np.linalg.norm(pre_dis_axis)
            target_dis_vec = (target_pose_mat[:3, :3] @ np.array(pre_dis_axis).reshape(3, 1)).reshape(3)
            target_dis_vec /= np.linalg.norm(target_dis_vec)
        return Pose(
            target_point - grasp_bias - pre_dis * target_dis_vec,
            target_grasp_qpose
        )


    def place_actor(
        self,
        actor: Actor,
        target_pose: Pose,
        functional_point_id: int = None,
        pre_dis: float = 0.1,
        dis: float = 0.02,
        gripper_pos=1.0,
        is_open: bool = True,
        **args,
    ):
        if not self.task.plan_success:
            return None
        
        place_pose = self.get_place_pose(
            actor,
            target_pose,
            functional_point_id=functional_point_id,
            pre_dis=dis,
            **args,
        )
        actions = [Action("move", target_pose=place_pose, pre_dis=pre_dis-dis)]
            
        if is_open:
            actions.extend(self.open_gripper(gripper_pos))
        return actions

    def move_by_displacement(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        xyz_coord: Literal["world", "local"]|Pose = "world",
        rpy: list[float] = None,
        rpy_coord: Literal["world", "local", "gripper"]|Pose = "local",
    ):
        origin_pose = self.robot.get_ee_pose()
        origin_pose = origin_pose.add_bias([x, y, z], coord=xyz_coord)
        if rpy is not None:
            if rpy_coord == 'gripper':
                rpy_coord = self.robot.get_gripper_center_pose()
            origin_pose = origin_pose.add_rotation(rpy, coord=rpy_coord)
        return [Action("move", target_pose=origin_pose)]

    def move_to_pose(
        self,
        target_pose: Pose,
    ):
        return [Action("move", target_pose=target_pose)]
    
    def close_gripper(self, pos: float = 0.0, depth_threshold:Literal['auto']|float='auto'):
        if depth_threshold == 'auto':
            if self.task.cfg.use_adaptive_grasp:
                depth_threshold = self.task.cfg.adaptive_grasp_depth_threshold
            else:
                depth_threshold = None
        return [Action("close", target_gripper_pos=pos, gripper_depth_threshold=depth_threshold)]

    def open_gripper(self, pos: float = 1.0, depth_threshold: float = None):
        return [Action("open", target_gripper_pos=pos, gripper_depth_threshold=depth_threshold)]

    def back_to_origin(self):
        return [Action("move", target_pose=self.robot.origin_pose)]

    def get_arm_pose(self):
        return self.robot.get_ee_pose()