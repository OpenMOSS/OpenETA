from ._base_task import *
import numpy as np

@configclass
class TaskCfg(BaseTaskCfg):
    adaptive_grasp_depth_threshold = 27.5

class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 1
        cfg.sim.physics_material.static_friction = 1
        cfg.uipc_sim.contact.default_friction_ratio = 1
        super().__init__(cfg, mode, render_mode, **kwargs)
    
    def create_actors(self):
        base_pose = Pose([0.9, 0.0, 0.01], [1, 0, 0, 0])
        bottle_pose = Pose([0.5, 0.0, 0.01], [1, 0, 0, 0])

        self.shelf = self._actor_manager.add_from_usd_file(
            name='shelf',
            asset_path="Shelf.usd", 
            pose=base_pose,
        )
        self.bottle = self._actor_manager.add_from_usd_file(
            name='prism',
            asset_path="BottleLift.usd",
            pose=bottle_pose
        )
        
    def _reset_actors(self):
        base_offset = self.create_noise([0.05, 0.0, 0.0])
        base_pose = Pose([0.9, 0.0, 0.01], [1, 0, 0, 0]).add_offset(base_offset)
        bottle_offset = self.create_noise([0.0, 0.03, 0.0])
        bottle_pose = Pose([0.5, 0.0, 0.01], [1, 0, 0, 0]).add_offset(bottle_offset)

        self.shelf.set_pose(base_pose)
        self.bottle.set_pose(bottle_pose)
 
    def pre_move(self):
        self.delay(10)

        bottle_pose = self.bottle.get_pose()
        target_pose = bottle_pose.add_bias([0, 0, 0.11+0.01*self.rng.random()])
        self.grasp_noise = self.create_noise(euler=[0, np.pi/18, 0])
        target_pose = construct_grasp_pose(
            target_pose.p,
            [0, 0, 1],
            [1, 0, 0]
        ).add_offset(self.grasp_noise)
        grasp_idx = self.bottle.register_point(
            pose=target_pose,
            type='contact'
        )
        self.move(self.atom.grasp_actor(
            self.bottle, contact_point_id=grasp_idx, pre_dis=0.0, is_close=False
        ))
        
        base_pose = self.shelf.get_pose()
        self.place_target = base_pose.add_bias([-0.2, 0, 0.21])
        self.move(self.atom.close_gripper())

    def _play_once(self):
        lift_height = 0.15 + self.rng.uniform(0.0, 0.05)
        self.move(self.atom.move_by_displacement(
            z=lift_height
        ), constraint_pose=[1, 1, 1, 0, 1, 0])
        self.move(self.atom.move_by_displacement(
            rpy=[0, -np.pi/3-0.2*self.rng.random(), 0]
        ), constraint_pose=[0, 0, 0, 0, 1, 0])

        self.gravity_rotate(self.bottle, target_vec=np.array([0, 0, 1]))
        self.move(self.atom.place_actor(
            self.bottle,
            pre_dis=0.01,
            dis=0.005,
            target_pose=self.place_target,
            is_open=False
        ), time_dilation_factor=0.3)
        self.move(self.atom.open_gripper(0.5))
        self.delay(20, is_save=False)

    def check_early_stop(self):
        min_depth = torch.min(self._tactile_manager.get_min_depth()).item()
        
        if min_depth < 20:
            self.metadata['early_stop'] = True
            self.metadata['min_depth'] = float(min_depth)
            return True
        return False

    def check_success(self):
        bottle_pose = self.bottle.get_pose().rebase(self.place_target)
        return np.all(np.abs(bottle_pose.p) < np.array([0.02, 0.1, 0.02])) \
            and np.dot(bottle_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1])) > 0.965 # 15°