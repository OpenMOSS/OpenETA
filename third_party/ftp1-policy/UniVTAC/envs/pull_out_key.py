from ._base_task import *
import numpy as np

@configclass
class TaskCfg(BaseTaskCfg):
    pass

class Task(BaseTask):
    def __init__(self, cfg: TaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 2.0
        cfg.sim.physics_material.static_friction = 2.0
        cfg.uipc_sim.contact.default_friction_ratio = 2.0
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self) -> None:
        base_pose = Pose([0.5, 0.0, 0.002], [1, 0, 0, 0])
        key_pose = base_pose.add_bias([-0.0025, 0, 0.0785])

        self.slot = self._actor_manager.add_from_usd_file(
            name='slot',
            asset_path="KeySlot.usd",
            pose=base_pose,
            density=1e5
        )
        self.key = self._actor_manager.add_from_usd_file(
            name='key',
            asset_path="Key.usd",
            pose=key_pose,
            density=10
        )
    
    def _reset_actors(self):
        random_pose = self.create_noise([0.005, 0.005, 0.0])
        random_rotate = self.rng.uniform(-np.pi/4, np.pi/4)
        base_pose = Pose([0.5, 0.0, 0.002], [1, 0, 0, 0]).add_offset(random_pose)
        key_pose = base_pose.add_bias([-0.0025, 0, 0.0785])

        base_pose = base_pose.add_rotation([0, 0, random_rotate])
        self.key_rotation = self.rng.uniform(-np.pi/2, -np.pi/4)
        key_pose = key_pose.add_rotation([0, 0, random_rotate+self.key_rotation])

        self.slot.set_pose(base_pose)
        self.key.set_pose(key_pose)

    def pre_move(self):
        self.delay(10)
        target_pose = self.key.get_pose().add_bias([0, 0, self.rng.uniform(-0.015, -0.01)])
        target_mat = target_pose.to_transformation_matrix()
        cpose = construct_grasp_pose(
            target_pose.p,
            target_mat[:3, 2],
            target_mat[:3, 0]
        )
        self.cid = self.key.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(
            self.key,
            contact_point_id=self.cid,
            pre_dis=0.08, dis=0.0
        ))
        
        self.target_pose = self.key.get_pose()
        self.target_pose[3:] = self.slot.get_pose().add_rotation([0, 0, 0.1])[3:]
        self.slot_init_pose = self.slot.get_pose()

    def _play_once(self):
        over_rotate = self.rng.uniform(0.09, 0.16) # 5° ~ 9°
        self.metadata['over_rotate'] = over_rotate

        self.move(
            self.atom.move_by_displacement(rpy=[0, 0, -self.key_rotation+over_rotate], xyz_coord='local'),
            time_dilation_factor=0.5, constraint_pose=[0, 0, 0, 1, 1, 1], delay=False
        )
        self.move(
            self.atom.move_by_displacement(rpy=[0, 0, -over_rotate+0.05], xyz_coord='local'),
            constraint_pose=[0, 0, 0, 1, 1, 1], delay=False
        )
        self.move(
            self.atom.move_by_displacement(z=-0.03, xyz_coord='local'),
            time_dilation_factor=0.2
        )
        self.delay(20)

    def check_early_stop(self):
        z_dis = np.abs(self._robot_manager.get_ee_pose()[2] - self.key.get_pose()[2])
        slot_rel_pose = self.slot.get_pose().rebase(self.slot_init_pose)
        slot_x_rotate = np.dot(
            slot_rel_pose.to_transformation_matrix()[:3, 0], np.array([1, 0, 0]))
        if z_dis >= 0.14:
            self.metadata['early_stop'] = True
            self.metadata['z_dis'] = z_dis
            return True
        if slot_x_rotate < 0.99:
            self.metadata['early_stop'] = True
            self.metadata['slot_x_rotate'] = slot_x_rotate
            return True
        return False

    def check_success(self):
        target_height = 0.09
        key_pose = self.key.get_pose()
        slot_rel_pose = self.slot.get_pose().rebase(self.slot_init_pose)
        slot_x_rotate = np.dot(
            slot_rel_pose.to_transformation_matrix()[:3, 0], np.array([1, 0, 0]))
        z_dis = np.abs(self._robot_manager.get_ee_pose()[2] - self.key.get_pose()[2])
        return key_pose.p[2] > target_height \
            and np.dot(key_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1])) > 0.965 \
            and slot_x_rotate > 0.99 \
            and z_dis < 0.14
