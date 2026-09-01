from ._base_task import *
import numpy as np

@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(1, 0.0, 0.15), rot=(0.5, 0.5, 0.5, 0.5), convention="opengl"),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.5, focus_distance=1.0, horizontal_aperture=3.6, clipping_range=(0.1, 100.0)  
            ),
            width=480,
            height=270,
            update_period=1/120
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None, # use existing camera
            width=480,
            height=270,
            update_period=1/120,
        )
    ]
    use_adaptive_grasp = False

class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        green_pose = Pose([0.4, 0.08, 0.01], [1, 0, 0, 0])
        orange_pose = Pose([0.4, -0.08, 0.01], [1, 0, 0, 0])

        self.green_pad = self._actor_manager.add_from_usd_file(
            name='green_pad',
            asset_path="GreenPad.usd",
            pose=green_pose,
        )
        self.orange_pad = self._actor_manager.add_from_usd_file(
            name='orange_pad',
            asset_path="OrangePad.usd",
            pose=orange_pose,
        )
        
        # rough -> orange; plain -> green
        self.rough_prism = self._actor_manager.add_from_usd_file(
            name='rough_prism',
            asset_path="RoughPrism.usd",
            pose=Pose([0.35, 1.0, 0.01], [1, 0, 0, 0])
        )
        self.plain_prism = self._actor_manager.add_from_usd_file(
            name='plain_prism',
            asset_path="PlainPrism.usd",
            pose=Pose([0.35, -1.0, 0.01], [1, 0, 0, 0])
        )
    
    def _reset_actors(self):
        self.choice = self.rng.choice(['rough', 'plain'])
        start_pose = Pose([0.35, 0.0, 0.01], [1, 0, 0, 0])
        if self.choice == 'rough':
            self.prism = self.rough_prism
            self.target = self.orange_pad
            self.other_target = self.green_pad
        else:
            self.prism = self.plain_prism
            self.target = self.green_pad
            self.other_target = self.orange_pad
        self.prism.set_pose(start_pose)

    def pre_move(self):
        self.delay(10)

        self.move(self.atom.open_gripper(0.5))

        target_pose = self.prism.get_pose().add_bias([0.0, 0.0, 0.04+0.01*self.rng.random()])
        cpose = construct_grasp_pose(
            target_pose.p,
            [0, 0, 1],
            [1, 0, 0]
        )
        cid = self.prism.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(
            self.prism, contact_point_id=cid, pre_dis=0.04, dis=0.0, is_close=False
        ))
        gripper_qpos = self.rng.uniform(0.0065, 0.0075) / 0.039
        self.move(self.atom.close_gripper(gripper_qpos))
        self.move(self.atom.move_by_displacement(z=0.05))
        
        self.target_pose = self.target.get_pose().add_bias([0.0, 0.0, 0.015])

    def _play_once(self):
        self.move(self.atom.place_actor(
            self.prism,
            target_pose=self.target_pose,
            pre_dis=0.0, dis=0.0,
            is_open=False
        ), time_dilation_factor=0.5)
        self.delay(20, is_save=False)

    def check_success(self):
        card_pose = self.prism.get_pose().rebase(self.target_pose)
        return np.all(np.abs(card_pose.p) < np.array([0.02, 0.02, 0.01])) and \
            np.dot(card_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1])) > 0.965  # 15°