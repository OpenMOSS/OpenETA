from ._base_task import *
import numpy as np

@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 500
    adaptive_grasp_depth_threshold = 27.8

class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        super().__init__(cfg, mode, render_mode, **kwargs)
    
    def create_actors(self):
        wall_pose = Pose([0.75, 0.0, 0.005], [1, 0, 0, 0])
        bottle_pose = wall_pose.add_bias([-0.08, 0.0, 0.03])

        self.wall = self._actor_manager.add_from_usd_file(
            name='wall',
            asset_path="Wall.usd", 
            pose=wall_pose,
            density=1e5
        )
        self.bottle = self._actor_manager.add_from_usd_file(
            name='bottle',
            asset_path="Bottle.usd", 
            pose=bottle_pose
        )
    
    def _reset_actors(self):
        bottle_offset = self.create_noise([0.01, 0.05, 0.0], [0, 0, np.pi/18])
        bottle_pose = self.wall.get_pose().add_bias([-0.08, 0.0, 0.03]).add_offset(bottle_offset)
        self.bottle.set_pose(bottle_pose)

    def pre_move(self):
        self.delay(10)

        bottle_pose = self.bottle.get_pose()
        target_pose = bottle_pose.add_bias([-0.13, 0, -0.015])
        target_mat = target_pose.to_transformation_matrix()
        self.grasp_noise = self.create_noise(euler=[0, [-np.pi/12, 0.0], 0])
        target_pose = construct_grasp_pose(
            target_pose.p,
            target_mat[:3, 2],
            target_mat[:3, 0]
        ).add_offset(self.grasp_noise)
        grasp_idx = self.bottle.register_point(
            pose=target_pose,
            type='contact'
        )
        self.move(self.atom.grasp_actor(
            self.bottle,
            contact_point_id=grasp_idx,
            is_close=False,
            pre_dis=0.5
        ))
        self.target_pose = self.wall.get_pose().add_bias([-0.08, 0, 0])
        
    def _play_once(self):
        self.move(self.atom.close_gripper())
        self.gripper_rotate(self.bottle, 70/180*np.pi, steps=4)
        if not self.check_mid_success():
            self.move(self.atom.move_by_displacement(
                rpy=[0, np.pi/6, 0], rpy_coord='gripper'
            ), time_dilation_factor=0.5)
            self.move(self.atom.move_by_displacement(
                x = self.target_pose[0] - self.bottle.get_pose()[0] + 0.02
            ), time_dilation_factor=0.5)
        self.move(self.atom.open_gripper(0.5))
        self.delay(30, is_save=False)

    def check_mid_success(self):
        rel_pose = self.bottle.get_pose().rebase(self.target_pose)
        return rel_pose[0] > -0.01
    
    def check_early_stop(self):
        rel_pose = self.bottle.get_pose().rebase(self.target_pose)
        if self.take_action_cnt > 300 and np.abs(np.dot(rel_pose.to_transformation_matrix()[:3, 0], np.array([-1, 0, 0]))) > 0.99:
            return True
        return False

    def check_success(self):
        rel_pose = self.bottle.get_pose().rebase(self.target_pose)
        return rel_pose[0] > -0.02 and np.all(np.abs(rel_pose[1:3]) < np.array([0.1, 0.001])) \
            and np.abs(np.dot(rel_pose.to_transformation_matrix()[:3, 0], np.array([0, 0, 1]))) > 0.99