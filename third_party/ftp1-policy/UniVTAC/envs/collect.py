from ._base_task import *
import numpy as np

@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(1, 0.15, 0.15), rot=(-0.354, -0.354, -0.612, -0.612), convention="opengl"),
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
    max_save_frames = 400

class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        stand_pose = Pose([0.7, 0.0, 0.005], [1, 0, 0, 0])
        prism_pose = stand_pose.add_bias([0, 0, 0.06])
        self.stand = self._actor_manager.add_from_usd_file(
            name='stand',
            asset_path="Stand.usd",
            pose=stand_pose,
            density=1e5
        )
        self.prism_name = os.environ.get('PRISM_NAME', 'Hemisphere')
        self.prism = self._actor_manager.add_from_usd_file(
            name='prism',
            asset_path=f"Bar_{self.prism_name}.usd",
            pose=prism_pose,
            density=1e5
        )
    
    def pre_move(self):
        self.delay(10)

        self.move(self.atom.open_gripper(0.8))

        grasp_height = 0.081+0.005*self.rng.uniform(-1, 1)
        target_pose = self.prism.get_pose().add_bias([0.0, 0.0, grasp_height])
        self.metadata['grasp_height'] = grasp_height
        cpose = construct_grasp_pose(
            target_pose.p,
            [0, 0, 1],
            [1, 0, 0]
        )
        cid = self.prism.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(
            self.prism, contact_point_id=cid, pre_dis=0.04, dis=0.0, is_close=False
        ))

        if self.cfg.tactile_sensor_type == 'gsmini':
            self.cfg.use_adaptive_grasp = True
            self.cfg.adaptive_grasp_depth_threshold = self.rng.uniform(27.7, 28.1)
            self.move(self.atom.close_gripper())
            self.metadata['grasp_threshold'] = self._tactile_manager.get_min_depth().tolist()
        elif self.cfg.tactile_sensor_type == 'gf225':
            self.cfg.use_adaptive_grasp = False
            grasp_qpos = np.random.uniform(0.0118, 0.013) / self._robot_manager.gripper_max_qpos
            self.move(self.atom.close_gripper(pos=grasp_qpos))
            self.metadata['grasp_qpos'] = grasp_qpos
        elif self.cfg.tactile_sensor_type == 'xensews':
            grasp_qpos = np.random.uniform(0.0118, 0.013) / self._robot_manager.gripper_max_qpos
            self.move(self.atom.close_gripper(pos=grasp_qpos))
        self.cfg.keep_contact = True

    def _play_once(self):
        rotate = self.rng.choice([-1, 1]) * self.rng.uniform(np.pi/3, np.pi*7/18)
        sub_rotate_num = int(np.ceil(np.abs(rotate / 0.2)))
        sub_rotate = rotate / sub_rotate_num
        for r in range(sub_rotate_num):
            self.move(self.atom.move_by_displacement(
                rpy=[0, sub_rotate, 0],
                rpy_coord='gripper'
            ), time_dilation_factor=0.2, constraint_pose=[0, 0, 0, 0, 1, 0])
        self.metadata['rotate'] = rotate
        self.metadata['prism'] = self.prism_name

    def check_success(self):
        return True