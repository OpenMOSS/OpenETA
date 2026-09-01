from ._base_task import *
import numpy as np

@configclass
class TaskCfg(BaseTaskCfg):
    pass

class Task(BaseTask):
    def __init__(self, cfg: TaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        slot_pose = Pose([0.6, 0.0, 0.002], [1, 0, 0, 0])
        base_pose = Pose([0.4, 0.0, 0.002], [1, 0, 0, 0])
        prism_pose = Pose([0.4, 0.0, 0.005], [1, 0, 0, 0])

        self.slot = self._actor_manager.add_from_usd_file(
            name='slot',
            asset_path="TestTubeHoleSlot.usd",
            pose=slot_pose,
            density=1e5
        )
        self.prism_base = self._actor_manager.add_from_usd_file(
            name='prism_base',
            asset_path="TestTubeBase.usd",
            pose=base_pose,
            density=1e5
        )
        self.prism = self._actor_manager.add_from_usd_file(
            name='prism',
            asset_path="TestTube.usd",
            pose=prism_pose,
            density=10
        )
    
    def _reset_actors(self):
        self.rotate = self.rng.choice([0, np.pi])
        base_offset = self.create_noise([0.02, 0.03, 0.0]).add_rotation([0, 0, self.rotate])
        base_pose = Pose([0.6, 0.0, 0.002], [1, 0, 0, 0]).add_offset(base_offset)
        self.slot.set_pose(base_pose)
    
    def pre_move(self):
        self.delay(10)
        
        grasp_bias = self.rng.uniform(0.095, 0.10)
        target_pose = self.prism.get_pose().add_bias([0, 0, grasp_bias])

        cpose = construct_grasp_pose(
            target_pose.p,
            [0, 0, 1],
            [1, 0, 0]
        )
        self.cid = self.prism.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(
            self.prism,
            contact_point_id=self.cid,
            pre_dis=0.0, dis=0.0
        ))
        self.origin_inhand_pose = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose())

        base_pose = self.slot.get_pose()
        base_pose[3:] = (1, 0, 0, 0)
        self.metadata['rotate'] = self.rotate

        self.hole_pose = base_pose.add_bias([0.0, 0, 0.1])
        if self.rotate == 0:
            self.target_pose = self.hole_pose.add_rotation([0, -np.pi/6, 0])
        else:
            self.target_pose = self.hole_pose.add_rotation([0, np.pi/6, 0])
        try_pose = self.hole_pose

        self.move(self.atom.move_by_displacement(z=0.15), constraint_pose=[1, 1, 1, 1, 1, 0])
        self.move(self.atom.place_actor(
            self.prism,
            target_pose=try_pose,
            pre_dis=0.05, dis=0.01,
            is_open=False
        ))
        self.move(self.atom.place_actor(
            self.prism,
            target_pose=try_pose,
            pre_dis=0.01, dis=0.002,
            is_open=False
        ), constraint_pose=[1, 1, 1, 1, 1, 0])

        self.origin_inhand_pose = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose())

    def _play_once(self):
        self.move(self.atom.move_by_displacement(z=-0.03), time_dilation_factor=0.2)
        rel_pose = self.prism.get_pose().rebase(self.target_pose)
        gripper_dis = self._robot_manager.get_gripper_center_pose().rebase(self.prism.get_pose())[2]
        x_move = - gripper_dis * np.sin(rel_pose.euler[1])
        z_move = gripper_dis * (np.cos(rel_pose.euler[1]) - 1)
        self.move(self.atom.move_by_displacement(
            x = x_move, z = z_move
        ), time_dilation_factor=0.5)
        self.move(self.atom.move_by_displacement(
            z=-0.04, xyz_coord=self.prism.get_pose()
        ), time_dilation_factor=0.5)
        self.delay(20, is_save=False)
    
    def check_early_stop(self):
        prism_inhand_pose = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose())
        inhand_bias = np.abs(self.origin_inhand_pose[2] - prism_inhand_pose[2])
        if inhand_bias > 0.04:
            self.metadata['early_stop'] = True
            self.metadata['inhand_bias'] = float(inhand_bias)
            return True

    def check_success(self, z_threshold=0.04):
        prism_pose = self.prism.get_pose().rebase(self.target_pose)
        prism_inhand_pose = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose())
        self.metadata['rel_pose'] = prism_pose.tolist()
        return np.all(np.abs(prism_pose[:2]) < np.array([0.01, 0.01])) \
            and prism_pose[2] < -z_threshold \
            and np.dot(
                prism_pose.to_transformation_matrix()[:3, 2],
                [0, 0, 1]
            ) > 0.99 \
            and np.abs(self.origin_inhand_pose[2] - prism_inhand_pose[2]) < 0.04