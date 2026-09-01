from ._base_task import *
import numpy as np

@configclass
class TaskCfg(BaseTaskCfg):
    pass

class Task(BaseTask):
    def __init__(self, cfg: TaskCfg, mode:Literal['collect', 'eval'] = 'collect', **kwargs):
        super().__init__(cfg=cfg, mode=mode, **kwargs)

    def create_actors(self):
        slot_pose = Pose([0.6, 0.0, 0.002], [1, 0, 0, 0])
        base_pose = Pose([0.4, 0.0, 0.002], [1, 0, 0, 0])
        prism_pose = Pose([0.4, 0.0, 0.005], [1, 0, 0, 0])

        self.slot = self._actor_manager.add_from_usd_file(
            name='slot',
            asset_path="TestTubeSlot.usd",
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
        slot_offset = self.create_noise([0.005, 0.01, 0.0])
        slot_pose = Pose([0.6, 0.0, self.slot.get_pose()[2]], [1, 0, 0, 0]).add_offset(slot_offset)
        self.slot.set_pose(slot_pose)
    
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
        self.random_noise = self.create_noise(
            [[0.001, 0.004], [0.001, 0.004], 0])
        self.random_noise[:2] *= np.sign(self.rng.uniform(-1, 1, size=2))
        self.metadata['random_noise'] = self.random_noise.tolist()
        self.hole_pose = base_pose.add_bias([-0.008, 0, 0.077]).add_rotation([0, -np.pi/6, 0])
        try_pose = self.hole_pose.add_offset(self.random_noise)

        self.move(self.atom.move_by_displacement(z=0.15), constraint_pose=[1, 1, 1, 1, 1, 0])
        self.move(self.atom.place_actor(
            self.prism,
            target_pose=try_pose,
            pre_dis=0.1, dis=0.05,
            is_open=False
        ))
        self.move(self.atom.place_actor(
            self.prism,
            target_pose=try_pose,
            pre_dis=0.05, dis=0.002,
            is_open=False
        ), constraint_pose=[1, 1, 1, 1, 1, 0])

    def _play_once(self):
        self.try_forward(self.prism, dis=0.02, delta_d=0.01)
        if not self.check_mid_success():
            dis = self.prism.get_pose().rebase(self.hole_pose)[2]
            self.move(self.atom.place_actor(
                self.prism,
                target_pose=self.hole_pose,
                pre_dis=dis, dis=dis,
                is_open=False
            ), time_dilation_factor=0.5, delay=False)
            self.try_forward(self.prism, dis=0.02, delta_d=0.01)

        self.move(self.atom.move_by_displacement(
            z=-0.04, xyz_coord=self.prism.get_pose()
        ), time_dilation_factor=0.2)
        self.delay(20, is_save=False)
 
    def check_mid_success(self):
        prism_pose = self.prism.get_pose().rebase(self.hole_pose)
        return np.all(prism_pose.p[:2] < np.array([0.005, 0.005])) and prism_pose.p[2] < -0.02 and\
            np.dot(prism_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1])) > 0.99 # 8°

    def check_early_stop(self):
        prism_inhand_pose = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose())
        inhand_bias = np.abs(self.origin_inhand_pose[2] - prism_inhand_pose[2])
        if inhand_bias > 0.03:
            self.metadata['early_stop'] = True
            self.metadata['inhand_bias'] = float(inhand_bias)
            return True

    def check_success(self, z_threshold=0.03):
        prism_pose = self.prism.get_pose().rebase(self.hole_pose)
        prism_inhand_pose = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose())
        self.metadata['rel_pose'] = prism_pose.tolist()
        self.metadata['inhand_bias'] = np.abs(self.origin_inhand_pose[2] - prism_inhand_pose[2])
        return np.all(prism_pose.p[:2] < np.array([0.005, 0.005])) and prism_pose.p[2] < -z_threshold \
            and np.dot(prism_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1])) > 0.965 \
            and np.abs(self.origin_inhand_pose[2] - prism_inhand_pose[2]) < 0.03