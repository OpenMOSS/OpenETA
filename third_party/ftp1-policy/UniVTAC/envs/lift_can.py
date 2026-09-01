from ._base_task import *
import numpy as np

@configclass
class TaskCfg(BaseTaskCfg):
    adaptive_grasp_depth_threshold = 27.75

class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        super().__init__(cfg, mode, render_mode, **kwargs)
 
    def create_actors(self):
        self.cans:dict[int, Actor] = {}
        pose_dict = {
            4: Pose([-1.0, 0.0, 0.022], [1, 0, 0, 0]),
            5: Pose([-1.0, 1.0, 0.027], [1, 0, 0, 0]),
            6: Pose([-1.0, -1.0, 0.032], [1, 0, 0, 0])
        }
        for d in [4, 5, 6]:
            self.cans[d] = self._actor_manager.add_from_usd_file(
                name=f'can_d{d}',
                asset_path=f"Can_d{d}cm.usd",
                pose=pose_dict[d]
            )

    def _reset_actors(self):
        can_offset = self.create_noise([0.02, 0.05, 0.0])
        can_size = self.rng.choice([4, 5, 6])
        can_pose = Pose(
            [0.7, 0.0, 0.005*can_size+0.001], [1, 0, 0, 0]
        ).add_offset(can_offset)
 
        self.can = self.cans[can_size]
        self.metadata['can_size'] = int(can_size)
        self.can.set_pose(can_pose)
 
    def pre_move(self):
        self.delay(10)

        self.move(self.atom.open_gripper(1.0))
        can_pose = self.can.get_pose()
        target_pose = can_pose.add_bias([-0.065, 0, -0.008])
        target_mat = target_pose.to_transformation_matrix()
        x = target_mat[:3, 0].reshape(-1)
        target_mat = np.vstack([
            x, np.cross(x, [0, 0, 1]), [0, 0, 1],
        ])
        self.grasp_noise = self.create_noise(euler=[0, [-np.pi/6, -np.pi/18], 0])
        self.metadata['grasp_noise'] = self.grasp_noise.tolist()
        target_pose = construct_grasp_pose(
            target_pose.p,
            target_mat[:3, 2],
            target_mat[:3, 0],
        ).add_offset(self.grasp_noise)
        grasp_idx = self.can.register_point(
            pose=target_pose,
            type='contact'
        )
        self.move(self.atom.grasp_actor(self.can, contact_point_id=grasp_idx, is_close=False))
        self.origin_inhand_pose = self._robot_manager.get_inhand_pose(self.can)
        
    def _play_once(self):
        self.move(self.atom.close_gripper())
        self.gripper_rotate(self.can, 70/180*np.pi, steps=4)
        if not self.check_mid_success():
            self.gravity_rotate(self.can, [0, 0, 1], [-1, 0, 0])
        self.move(self.atom.open_gripper())
        self.delay(30, is_save=False)

    def check_mid_success(self):
        can_pose = self.can.get_pose()
        return np.abs(np.dot(can_pose.to_transformation_matrix()[:3, 0], np.array([0, 0, 1]))) > 0.95

    def check_early_stop(self):
        can_pose = self.can.get_pose()
        inhand_pose = self._robot_manager.get_inhand_pose(self.can)
        min_depth = torch.min(self._tactile_manager.get_min_depth()).item()
        
        if min_depth < 20:
            self.metadata['early_stop'] = True
            self.metadata['min_depth'] = float(min_depth)
            return True
        if np.abs(inhand_pose.p[2] - self.origin_inhand_pose.p[2]) > 0.05 and \
            np.abs(np.dot(can_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.99:
            self.metadata['early_stop'] = True
            self.metadata['inhand_dis'] = float(np.abs(inhand_pose.p[2] - self.origin_inhand_pose.p[2]))
            return True
        return False
 
    def check_success(self):
        can_pose = self.can.get_pose()
        return can_pose[2] < 0.01 and \
            np.abs(np.dot(can_pose.to_transformation_matrix()[:3, 0], np.array([0, 0, 1]))) > 0.99
