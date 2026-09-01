import os
import sys
import json
import time
import torch
import pickle
import torchvision

from envs.utils.data import HDF5Handler, VideoHandler
from warp import Function
import isaaclab
import numpy as np
from pathlib import Path
from typing import Generator, Literal
import decorator

import carb
import omni.ui
import logging
from contextlib import suppress
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.prims import XFormPrim
with suppress(ImportError):
    # isaacsim.gui is not available when running in headless mode.
    import isaacsim.gui.components.ui_utils as ui_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, RenderCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_error_magnitude,
    sample_uniform,
    wrap_to_pi,
)
from isaaclab.utils.noise import (
    GaussianNoiseCfg,
    NoiseModelCfg,
    UniformNoiseCfg,
    gaussian_noise,
)

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg
from tacex_uipc import (
    UipcRLEnv,
    UipcIsaacAttachments,
    UipcIsaacAttachmentsCfg,
    UipcObject,
    UipcObjectCfg,
    UipcSimCfg,
)
from tacex_uipc.utils import TetMeshCfg

from typing import Any

from ._global import *
from .utils import *
from .robot.robot import RobotManager
from .robot.robot_cfg import *
from .sensors.camera import CameraManager, CameraCfg
from .sensors.tactile import TactileManager, TactileCfg, create_tactile_cfg


@configclass
class BaseTaskCfg(DirectRLEnvCfg):
    logger_level = "error"
    debug_vis = False

    # viewer settings
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (0.6, 0.15, 0.05)
    viewer.lookat = (-3.0, -4.5, -0.6)

    step_lim = 300

    save_dir = "auto"
    worker_name: str | None = None
    obs_data_type = {}

    save_frequency = 1
    video_frequency = 1
    render_frequency = 0
    video_size = (960, 320)  # (width, height); keep 320 to avoid upscaling low-res camera (would look blocky)

    ui_window_class_type = BaseEnvWindow

    decimation = 1
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,
        render_interval=decimation,
        # device="cpu",
        physx=PhysxCfg(
            enable_ccd=True,  # needed for more stable ball_rolling
            # bounce_threshold_velocity=10000,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            restitution=0.0,
        ),
        # 避免 A800 headless 下 "Render resolution below minimal input resolution of 300" 导致的模糊：
        # 用 FXAA 完全不走 DLSS/DLAA 管线，不会触发低分辨率放大；若仍模糊可再检查各任务是否覆盖相机为 480x270
        render=RenderCfg(
            rendering_mode="quality",
            enable_dl_denoiser=True,
            antialiasing_mode="FXAA",
        ),
    )

    uipc_sim = UipcSimCfg(
        # logger_level="Info"
        dt=sim.dt,
        ground_height=0.001,
        contact=UipcSimCfg.Contact(
            d_hat=0.0005,
            enable_friction=True,
            eps_velocity=0.1
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=1.5,
        replicate_physics=True,
        lazy_sensor_update=True,
    )

    # light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75), intensity=3000.0,
            texture_file=str(SCENE_ASSETS_ROOT / 'base0.exr')
        ),
    )

    # plate
    plate = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ground_plate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(SCENE_ASSETS_ROOT / "plate.usda"),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                kinematic_enabled=True,
            ),
        ),
    )

    use_adaptive_grasp: bool = True
    adaptive_grasp_depth_threshold = None # in mm
    reset_time_limit: float = 120.0  # in seconds

    # 相机分辨率需满足 RTX/DLSS 最小输入（短边 >= 300），否则会报 "below minimal input resolution of 300" 且画面模糊
    cameras: list[CameraCfg] = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.554, 1.0, 0.150), rot=(0, 0, 0.707, 0.707), convention="opengl"),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.94, focus_distance=1.0, horizontal_aperture=2.688, clipping_range=(0.01, 100.0)
            ),
            width=640,
            height=360,
            update_period=1/120
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None, # use existing camera
            width=640,
            height=360,
            update_period=1/120,
        )
    ]

    robot: RobotCfg = None
    tactile_sensor_type:Literal['gsmini', 'xensews', 'gf225'] = 'gsmini'

    planner_time_dilation_factor: float = 1.0

    gaussian_noise_cfg: GaussianNoiseCfg = GaussianNoiseCfg(mean=0.0, std=0.002, operation="add")
    
    random_texture: bool = False
    keep_contact: bool = False
    max_save_frames: int = 1000

    # some filler values, needed for DirectRLEnv
    episode_length_s = 0
    action_space = 0
    observation_space = 0
    state_space = 0

class BaseTask(UipcRLEnv):
    cfg: BaseTaskCfg

    def __init__(self, cfg: BaseTaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode=None, **kwargs):
        cfg = self.load_robot_and_sensors(cfg)
        
        self.cfg = cfg
        self.mode = mode
        self.render_outdated = True

        self._setup_save()
        self.rng = np.random.default_rng()
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs) # Full Render

        self.logger = logging.getLogger(name=self.__class__.__name__)
        self.logger.setLevel(getattr(logging, self.cfg.logger_level.upper(), logging.ERROR))

        self.first_frame = None
        
        self.start_time = 0.0
        self.step_count = 0
        self.save_count = 0
        self.last_render = -1
        self.step_cost = np.zeros(20)
        self.last_step = time.perf_counter()
        self.mean_steps = 0
        self.take_action_cnt = 0
        self.plan_success = True
        self.eval_success = False
        self.in_pre_move = False
        self.last_qpos = None
        self.keep_still_times = 0
        self.atom_tag = ''
        self.atom_id = 0
        self.log = ''
        self.metadata = {}
 
        self.instruction = ""
        self.video_handler = VideoHandler()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._robot_manager.setup()
        self._camera_manager.setup()
        self._tactile_manager.setup()
        self._tactile_manager.set_debug_vis(self.cfg.debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)
    
    def load_robot_and_sensors(self, cfg:BaseTaskCfg):
        data_type = ["camera_depth", "tactile_rgb", "marker_rgb", "marker_motion"]
        if cfg.tactile_sensor_type == 'gsmini':
            cfg.robot = create_franka_gsmini_gripper(data_type=data_type)
        elif cfg.tactile_sensor_type == 'gf225':
            cfg.robot = create_franka_gf225_gripper(data_type=data_type)
        elif cfg.tactile_sensor_type == 'xensews':
            cfg.robot = create_franka_xensews_gripper(data_type=data_type)
        else:
            raise ValueError(f'Unknown tactile sensor type: {cfg.tactile_sensor_type}')
        
        if cfg.adaptive_grasp_depth_threshold is None:
            cfg.adaptive_grasp_depth_threshold = cfg.robot.adaptive_grasp_depth_threshold
        return cfg
 
    def _setup_save(self):
        if self.cfg.save_dir == "auto":
            module_name = self.__class__.__module__
            module = sys.modules[module_name]
            file_name = Path(module.__file__).stem
            if self.mode == 'collect':
                save_dir = Path('./data') / file_name
            else:
                save_dir = Path('./eval_result') / file_name
        else:
            save_dir = self.cfg.save_dir

        self.save_root = Path(save_dir)
        self.save_root.mkdir(parents=True, exist_ok=True)

        if self.mode == 'eval':
            self.worker_name = self.cfg.worker_name or 'worker_0'
            self.tmp_save_dir = self.save_root / '.cache' / self.worker_name / str(self.cfg.seed)
            self.save_path = self.save_root / 'hdf5' / self.worker_name / f'{self.cfg.seed}.hdf5'
            self.save_video_path = self.save_root / 'video' / self.worker_name / f'{self.cfg.seed}.mp4'
            self.metadata_path = self.save_root / 'metadata' / f'{self.worker_name}.json'
            self.cfg.uipc_sim.workspace = str(self.save_root / 'scene' / self.worker_name)
        else:
            self.worker_name = None
            self.tmp_save_dir = self.save_root / '.cache' / str(self.cfg.seed)
            self.save_path = self.save_root / 'hdf5' / f'{self.cfg.seed}.hdf5'
            self.save_video_path = self.save_root / 'video' / f'{self.cfg.seed}.mp4'
            self.metadata_path = self.save_root / 'metadata.json'
            self.cfg.uipc_sim.workspace = str(self.save_root / 'scene')

    def _setup_scene(self):
        '''
            call once when initializing the environment
        '''
        self._setup_base_scene()
        self.scene.clone_environments(copy_from_source=False)
        
        self._actor_manager = ActorManager(self)
        self.create_actors()

        # add sensors
        self._camera_manager = CameraManager(self.cfg.cameras, self)
        self._tactile_manager = TactileManager(self.cfg.robot.tactiles, self)

    def _setup_base_scene(self):
        # add robot
        self._robot_manager:RobotManager = RobotManager(
            robot_cfg=self.cfg.robot,
            task=self,
            planner_time_dilation_factor=self.cfg.planner_time_dilation_factor
        )
        self.atom:Atom = Atom(self)

        self.plate = RigidObject(self.cfg.plate)

        # add lights
        self.cfg.light.spawn.func(self.cfg.light.prim_path, self.cfg.light.spawn)
    
    def create_noise(self, vec=[0.0, 0.0, 0.0], euler=[0.0, 0.0, 0.0]) -> Pose:
        '''Create random noise pose'''
        return Pose.create_noise(vec=vec, euler=euler, rng=self.rng)

    def timer(self, name):
        def log(*msg):
            # with open('log.log', 'a') as f:
            #     f.write(' '.join(str(m) for m in msg) + '\n')
            pass
        if not hasattr(self, '_timers'):
            self._timers = {}
        if name not in self._timers:
            self._timers[name] = time.perf_counter()
        else:
            log(f'[{self.step_count:>3d}][{name:^20}] cost: {(time.perf_counter() - self._timers[name])*1000:.2f} ms')
            self._timers.pop(name)

    def pre_move(self):
        pass

    def create_actors(self):
        pass

    def seed(self, seed:int = -1):
        seed = super().seed(seed)
        self.cfg.seed = seed
        self.rng = np.random.default_rng(seed)
        self._setup_save()
    
    def show_scene(self, actor_names:list[str]=None, show_next:bool=True):
        import trimesh
        geos = []
        
        if actor_names is None:
            actor_names = list(self._actor_manager.actors.keys())

        for actor_name in actor_names:
            if actor_name not in self._actor_manager.actors:
                continue
            actor = self._actor_manager.actors[actor_name]
            p1, p2 = actor.vertices, actor.next_pts
            geos.append(trimesh.PointCloud(
                p1, colors=[0, 0, 0]
            ))
            if show_next and p2 is not None:
                geos.append(trimesh.PointCloud(
                    p2, colors=[255, 0, 0]
                ))
        trimesh.Scene(geos).show()

    def reset(self, seed:int=-1, instructions:list[str]|None=None, options:dict[str, Any]|None=None):
        self.seed(seed)
        ret = super().reset()
        
        if self.first_frame is not None:
            self.uipc_sim.replay_frame(self.first_frame)

        total_cost = time.perf_counter() - self.start_time
        if total_cost > self.cfg.reset_time_limit:
            raise RuntimeError(
                f'Timeout: reset exceed time limit of {self.cfg.reset_time_limit} s, cost {total_cost} s.'
            )

        if self.cfg.video_frequency > 0:
            self.video_handler.reset(self.save_video_path, self.cfg.video_size)
        if instructions is not None:
            self.instruction = self.rng.choice(instructions)
        
        self.in_pre_move = True
        if self.first_frame is None:
            reset_test_start = time.perf_counter()
            for _ in range(5):
                self._step(is_save=False)
                reset_test_cost = time.perf_counter() - reset_test_start
                if reset_test_cost > 8 * self.cfg.reset_time_limit:
                    raise RuntimeError(
                        f'Timeout: reset exceed time limit of {8 * self.cfg.reset_time_limit} s, cost {reset_test_cost} s.'
                    )
            self._update_render()

            self.first_frame = self.uipc_sim.world.frame()
            self.uipc_sim.save_frame()

        if hasattr(self, '_reset_actors'):
            self._reset_actors()

            reset_test_start = time.perf_counter()
            for _ in range(20):
                self._step(is_save=False)
                reset_test_cost = time.perf_counter() - reset_test_start
                if reset_test_cost > self.cfg.reset_time_limit * 8:
                    raise RuntimeError(
                        f'Timeout: reset exceed time limit of {self.cfg.reset_time_limit * 8} s, cost {reset_test_cost} s.'
                    )
            self._update_render()
            self._actor_manager.remove_animate()
        
        reset_test_start = time.perf_counter()
        for _ in range(5):
            self._step(is_save=False)
            reset_test_cost = time.perf_counter() - reset_test_start
            if reset_test_cost > self.cfg.reset_time_limit:
                raise RuntimeError(
                    f'Timeout: reset exceed time limit of {self.cfg.reset_time_limit} s, cost {reset_test_cost} s.'
                )
        self._update_render()

        self.pre_move()
        self.in_pre_move = False

        # update render to avoid artifacts
        for _ in range(5):
            self._update_render()

        if self.mode == 'eval':
            self.delay()

        self.atom_id = 0
        self.atom_tag = ''

        return ret

    # def _reset_actors(self):
    #     pass

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        if self.cfg.random_texture:
            Actor._set_texture('/World/envs/env_0/ground_plate', 'random', self.rng)
        self._tactile_manager._reset_idx()
        self._actor_manager._reset_idx(self.rng)
        self._robot_manager._reset_idx()

        self.plan_success = True
        self.eval_success = False
        self.step_count = 0
        self.save_count = 0
        self.step_cost = np.zeros(20)
        self.last_step = time.perf_counter()
        self.last_render = -1
        self.take_action_cnt = 0
        self.current_goal_idx = 0
        self.render_outdated = True
        self.start_time = time.perf_counter()
        self.last_qpos = None
        self.keep_still_times = 0
        self.metadata = {}
        self.log = ''

    def pause(self):
        self.sim.pause()

    def _update_render(self):
        self.uipc_sim.update_render_meshes()
        self.sim.render()
        
        dt = self.physics_dt * self.cfg.decimation * max(1, self.step_count - self.last_render)
        self.scene.update(dt=dt)
        self._actor_manager.update(dt=dt)
        self._tactile_manager.update(dt=dt, force_recompute=True)
 
        self.last_render = self.step_count
    
    def get_frame_shot(self, obs):
        head_obs = obs['observation']['head']['rgb'].clone()
        wrist_obs = obs['observation']['wrist']['rgb'].clone()
        # Match video_size (960, 320): do not upscale (sim camera may be 224/320; upscale → blocky video)
        tac_w, tac_h = 160, 320  # tactile on sides
        img_h, cam_w = 320, 320  # center: head and wrist each 320
        left_tac = torchvision.transforms.Resize((tac_h, tac_w))(
            obs['tactile']['left_tactile']['rgb_marker'].clone().permute(2, 0, 1)).permute(1, 2, 0)
        right_tac = torchvision.transforms.Resize((tac_h, tac_w))(
            obs['tactile']['right_tactile']['rgb_marker'].clone().permute(2, 0, 1)).permute(1, 2, 0)
        head_img = torchvision.transforms.Resize((img_h, cam_w))(head_obs.permute(2, 0, 1)).permute(1, 2, 0)
        wrist_img = torchvision.transforms.Resize((img_h, cam_w))(wrist_obs.permute(2, 0, 1)).permute(1, 2, 0)

        # Layout: [left_tac | head | wrist | right_tac], tactile on sides so camera view is unobstructed
        img = torch.zeros((img_h, tac_w + cam_w * 2 + tac_w, 3), dtype=head_obs.dtype)
        img[:, :tac_w] = left_tac
        img[:, tac_w : tac_w + cam_w] = head_img
        img[:, tac_w + cam_w : tac_w + cam_w * 2] = wrist_img
        img[:, -tac_w:] = right_tac
        return img

    @staticmethod
    def _step_callback(status:dict):
        mode = status['mode']
        is_save = status['is_save']
        atom_id = status['atom_id']
        atom_tag = status['atom_tag']
        step_count = status['step_count']
        total_cost = status['total_cost']
        mean_steps = status['mean_steps']
        step_mean_cost = status['step_mean_cost']
        take_action_cnt = status['take_action_cnt']
        step_status = f'FPS {1/step_mean_cost:6.2f}, Running {total_cost:7.2f}s'

        if mean_steps > 0.0:
            if mode == 'eval':
                step_percent = f'({take_action_cnt / mean_steps * 100:6.2f}%)'
            else:
                step_percent = f'({step_count / mean_steps * 100:6.2f}%)'
        else:
            step_percent = '(   N/A%)'

        log = ''
        if mode == 'collect':
            atom_status = f'Atom ID: {atom_id:>2d}, Tag: {atom_tag:<15}'
            log = (f'Step {step_count:>5d}{step_percent}'
                    f', save {is_save}, {step_status}, {atom_status}')
        elif mode == 'eval_test':
            log = (f'Step {step_count:>5d}{step_percent}, testing     '
                  f', {step_status}')
        else:
            if not is_save:
                log = (f'Step {step_count:>5d}{step_percent}, pre moving  '
                      f', {step_status}')
            else:
                log = (f'Step {step_count:>5d}{step_percent}, action {take_action_cnt:>5d}'
                      f', {step_status}')
        return log

    def _step(self, is_save:bool=True):
        if self.plan_success is False:
            return 
        
        self.step_count += 1

        is_save = is_save and (not self.in_pre_move) and (not self.mode == 'eval_test')
        save_freq = (self.cfg.video_frequency > 0 and self.step_count % self.cfg.save_frequency == 0)
        video_freq = (self.cfg.video_frequency > 0 and self.step_count % self.cfg.video_frequency == 0)
        render_freq = (self.cfg.render_frequency > 0 and self.step_count % self.cfg.render_frequency == 0)

        self.scene.write_data_to_sim()
        for _ in range(self.cfg.decimation):
            self.sim.step(render=False)

        if render_freq or (self.mode == 'collect' and is_save and save_freq) or (is_save and video_freq) \
            or (self.mode == 'eval' and not self.in_pre_move):
            self._update_render()

        obs = None
        if self.mode == 'collect' and is_save and save_freq:
            obs = self._get_observations()
            self.save_observations(obs)

            def check(d):
                depth = d[50:-50, 50:-50]
                if depth.min() == depth.max():
                    return False
                return True
            if self.cfg.keep_contact:
                if not check(obs['tactile']['left_tactile']['depth']) \
                    or not check(obs['tactile']['right_tactile']['depth']):
                        self.plan_success = False
            if self.save_count > self.cfg.max_save_frames-1:
                self.plan_success = False
 
        if is_save and video_freq:
            if obs is None:
                obs = self._get_observations()
            self.video_handler.write(self.get_frame_shot(obs))

        step_mean_cost = 0.0
        step_cost = time.perf_counter() - self.last_step
        self.last_step = time.perf_counter()
        self.step_cost[(self.step_count-1) % 20] = step_cost
        if self.step_count <= len(self.step_cost):
            step_mean_cost = np.mean(self.step_cost[:self.step_count])
        else:
            step_mean_cost = np.mean(self.step_cost)
        total_cost = time.perf_counter() - self.start_time

        status_dict = {
            'mode': self.mode,
            'is_save': is_save,
            'mean_steps': self.mean_steps,
            'step_count': self.step_count,
            'take_action_cnt': self.take_action_cnt,
            'atom_id': self.atom_id,
            'atom_tag': self.atom_tag,
            'step_mean_cost': step_mean_cost,
            'step_cost': step_cost,
            'total_cost': total_cost
        }
        self.log = self._step_callback(status_dict)
        print(self.log, flush=True)
    
    def _play_once(self):
        pass

    def play_once(self):
        ret = self._play_once()
        if ret is not None:
            self.metadata.update()
        self._save_metadata()

    def _get_observations(self):
        obs = {
            'observation': {},
            'embodiment': {},
            'tactile': {},
            'actor': {},
            'step': self.step_count,
            'atom': {
                'id': self.atom_id,
                'tag': self.atom_tag
            }
        }

        if 'embodiment' in self.cfg.obs_data_type:
            obs['embodiment'] = self._robot_manager.get_observations(self.cfg.obs_data_type['embodiment'])
        if 'camera' in self.cfg.obs_data_type:
            obs['observation'] = self._camera_manager.get_observations(self.cfg.obs_data_type['camera'])
        if 'tactile' in self.cfg.obs_data_type:
            obs['tactile'] = self._tactile_manager.get_observations(self.cfg.obs_data_type['tactile'])
        if 'actor' in self.cfg.obs_data_type:
            obs['actor'] = self._actor_manager.get_observations()
        return obs
    
    def clean_cache(self, mean_steps:float=0.0, result:str=None):
        self.mean_steps = mean_steps
        if self.tmp_save_dir.exists():
            for f in self.tmp_save_dir.iterdir():
                f.unlink()
            self.tmp_save_dir.rmdir()
        if self.cfg.video_frequency > 0:
            self.video_handler.close(result)
        if result is not None:
            self.metadata['cost_step'] = self.step_count
            self.metadata['cost_time'] = time.perf_counter() - self.start_time
            self.metadata['result'] = result
            self._save_metadata()
 
    def save_to_hdf5(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        HDF5Handler().pkls_to_hdf5(self.tmp_save_dir, self.save_path)
    
    def _save_metadata(self):
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    all_metadata = json.load(f)
            except Exception as _:
                all_metadata = {}
        else:
            all_metadata = {}
        all_metadata[str(self.cfg.seed)] = self.metadata
        tmp_path = self.metadata_path.with_name(
            f"{self.metadata_path.name}.tmp.{os.getpid()}.{time.time_ns()}"
        )
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(all_metadata, f, ensure_ascii=False, indent=4)
            os.replace(tmp_path, self.metadata_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def save_observations(self, obs: dict):
        def to_cpu(data):
            if isinstance(data, dict):
                return {k: to_cpu(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [to_cpu(v) for v in data]
            elif isinstance(data, torch.Tensor):
                return data.cpu().numpy()
            else:
                return data

        self.tmp_save_dir.mkdir(parents=True, exist_ok=True)
        with open(self.tmp_save_dir / f'{self.save_count}.pkl', 'wb') as f:
            pickle.dump(to_cpu(obs), f)
        self.save_count += 1
 
    def check_success(self):
        return False
    
    def move(
        self,
        actions: list[Action],
        tag:str = 'move',
        is_save: bool = True,
        delay: bool = True,
        constraint_pose = None,
        time_dilation_factor = None,
        gripper_depth_threshold = None
    ):
        """
        Take action for the robot.
        """
        if self.plan_success is False:
            return False
        
        self.atom_id += 1
        self.atom_tag = tag

        for idx, action in enumerate(actions):
            control_seq = {
                "arm": None,
                "gripper": None,
            }
            if action.action == 'move' or action.action == 'all':
                action.args['constraint_pose'] = action.args.get(
                    'constraint_pose', constraint_pose)
                action.args['time_dilation_factor'] = action.args.get(
                    'time_dilation_factor', time_dilation_factor)
                control_seq['arm'] = self._robot_manager.plan_arm(
                    action.target_pose,
                    pre_dis=action.args.get('pre_dis'),
                    constraint_pose=action.args['constraint_pose'],
                    time_dilation_factor=action.args['time_dilation_factor'],
                )
                if control_seq['arm']['status'] == 'Fail':
                    self.logger.error(f'Arm motion planning failed on action {idx}: {action.__str__()}')
                    if self.cfg.debug_vis:
                        add_visual_box(action.target_pose, 'failed_target')
                        self.delay(100)
                    self.plan_success = False
                    return False

                if self.cfg.debug_vis:
                    add_visual_box(action.target_pose, 'target')

            if action.action == 'gripper' or action.action == 'all':
                if self.mode in ['collect', 'eval_test'] or (self.mode == 'eval' and self.in_pre_move):
                    if self.cfg.use_adaptive_grasp:
                        target_pos = self._robot_manager.gripper_percent2qpos(action.target_gripper_pos)
                        control_seq['gripper'] = {
                            'status': 'success',
                            'num_steps': -1,
                            'target': target_pos,
                            'threshold': action.args.get('gripper_depth_threshold', gripper_depth_threshold)
                        }
                    else:
                        control_seq['gripper'] = self._robot_manager.plan_gripper(
                            action.target_gripper_pos, type='percent'
                        )
                else:
                    control_seq['gripper'] = self._robot_manager.plan_gripper(
                        action.target_gripper_pos, type='qpos'
                    )
                if control_seq['gripper']['status'] == 'Fail':
                    self.logger.error(f'Gripper motion planning failed on action {idx}: {action.__str__()}')
                    self.plan_success = False
                    return False
            
            self.take_dense_action(control_seq, is_save)
            if delay:
                self.delay(10, is_save)
        self._update_render()
        return True
 
    def delay(self, steps=20, is_save:bool=False, force:bool=False):
        if not force and not self.plan_success:
            return False
        self.logger.info(f"Delaying for {steps} steps")
        self.atom_tag = 'delay'
        self.atom_id += 1
        for _ in range(steps):
            self._step(is_save)
        self._update_render()
        return True
 
    def take_dense_action(self, control_seq, is_save:bool=True):
        """
        control_seq:
            arm, gripper
        """
        arm_seq, gripper_seq = (
            control_seq['arm'],
            control_seq['gripper'],
        )

        arm_steps = arm_seq['num_steps'] if arm_seq is not None else 0
        gripper_steps = gripper_seq['num_steps'] if gripper_seq is not None else 0

        if gripper_steps == -1: # adaptive grasp
            idx, gripper_active = 0, True
            gripper_planner = self.adaptive_set_gripper(
                gripper_seq['target'], gripper_seq['threshold'])
            while True:
                if idx >= arm_steps and not gripper_active:
                    break
                if arm_seq is not None and idx < arm_steps:
                    self._robot_manager.set_arm(
                        arm_seq['position'][idx],
                        arm_seq['velocity'][idx]
                    )
                if gripper_active:
                    pos, vel, gripper_active = next(gripper_planner)
                    self._robot_manager.set_gripper(pos, vel)
                self._step(is_save)
                idx += 1
        else:
            max_control_len = max(arm_steps, gripper_steps)
            for idx in range(max_control_len):
                if arm_seq is not None and idx < arm_steps:
                    self._robot_manager.set_arm(
                        arm_seq['position'][idx],
                        arm_seq['velocity'][idx]
                    )
                if gripper_steps is not None and idx < gripper_steps:
                    self._robot_manager.set_gripper(
                        gripper_seq['position'][idx],
                        gripper_seq['velocity'][idx]
                    )
                self._step(is_save)
        return True

    def check_early_stop(self):
        return False

    def take_action(self, action:torch.Tensor, action_type:Literal['qpos', 'ee', 'delta_ee']='qpos', force:bool=True):
        '''
            qpos     : actions is Tensor([8]), qpos (7 DOFS + gripper)
            ee       : actions is Tensor([7]), position (3), orientation (4)
            delta_ee : actions is Tensor([6]), delta_position (3), delta_orientation (3)
        '''
        if self.take_action_cnt >= self.cfg.step_lim or self.eval_success:
            return True, self.eval_success

        exec_success = True
        self.take_action_cnt += 1
        self.logger.info(f"step: {self.take_action_cnt} / {self.cfg.step_lim}")

        if action_type == 'ee':
            target_pose = Pose(p=action[:3], q=action[3:7])
            target_gripper_pos = action[7:]
            exec_success = self.move([
                Action(action='all', target_pose=target_pose, target_gripper_pos=target_gripper_pos)
            ], delay=False)
        elif action_type == 'delta_ee':
            ee_pose = self._robot_manager.get_ee_pose()
            ee_next_pose = ee_pose.add_bias(action[:3], coord='world')\
                .add_rotation(euler=action[3:6].tolist(), coord='world')
            gripper_pos = self._robot_manager.get_gripper_qpos()
            gripper_next_pos = gripper_pos + action[6]
            exec_success = self.move([
                Action(action='all', target_pose=ee_next_pose, target_gripper_pos=gripper_next_pos)
            ], delay=False)
        else:
            self._robot_manager.set_arm(action[:-1], force=force)
            self._robot_manager.set_gripper(action[-1], force=force)
            self._step()
        
        if self.check_success():
            self.eval_success = True

        return exec_success, self.eval_success

    def adaptive_set_gripper(self, qpos, depth_threshold:float=None):
        max_steps = 1000
        default_step, contact_step = 0.0005, 0.00005
        last_qpos = self._robot_manager.get_gripper_qpos()
        max_depth = self.cfg.robot.tactile_far_plane \
            * torch.ones_like(self._tactile_manager.get_min_depth()) # mm
        if depth_threshold is not None:
            depth_threshold = depth_threshold * torch.ones_like(max_depth)
        direct = 'open' if self._robot_manager.get_gripper_qpos() < qpos else 'close'

        step_size = contact_step if direct == 'open' else -default_step
        for i in range(max_steps):
            current_qpos = self._robot_manager.get_gripper_qpos()
            tactile_depth = self._tactile_manager.get_min_depth()

            if direct == 'close':
                if torch.allclose(max_depth, tactile_depth, atol=1e-5):
                    step_size = -default_step
                elif depth_threshold is not None:
                    if torch.all(tactile_depth < depth_threshold):
                        break
                    else:
                        step_size = - min(
                            torch.min(torch.abs(tactile_depth - depth_threshold)).item()/1000,
                            contact_step
                        )
                else:
                    step_size = -default_step
            else:
                if torch.allclose(max_depth, tactile_depth, atol=1e-5):
                    step_size = default_step
                if depth_threshold is not None:
                    if torch.all(tactile_depth > depth_threshold):
                        break
                    else:
                        step_size = min(
                            torch.min(torch.abs(depth_threshold - tactile_depth)).item()/1000,
                            contact_step
                        )
                else:
                    step_size = default_step

            if np.allclose(current_qpos, qpos, atol=1e-5):
                break
            elif np.abs(current_qpos - qpos) < np.abs(step_size):
                target_qpos = qpos
            else:
                target_qpos = current_qpos + step_size
            position = torch.tensor([target_qpos, target_qpos], device=self._robot_manager.device)
            velocity = (position - current_qpos)/self.cfg.sim.dt
            last_qpos = current_qpos
            yield position, velocity, True

        final_position = torch.tensor([last_qpos, last_qpos], device=self._robot_manager.device)
        yield final_position, torch.zeros_like(final_position), False

    def gravity_rotate(self, actor:Actor, target_vec, target_axis=[0, 0, 1], is_save=True):
        if self.plan_success is False:
            return False
        
        max_steps = 200
        omega_threshold = 0.05
        contact_threshold = self.cfg.robot.contact_threshold # [min, max]
        target_axis = np.array(target_axis).reshape(3, 1)

        def get_axis():
            nonlocal actor, target_axis
            axis = (actor.get_pose().to_transformation_matrix()[:3, :3] @ target_axis).reshape(-1)
            axis /= np.linalg.norm(axis)
            return axis

        target_vec = np.array(target_vec) / np.linalg.norm(target_vec) 
        last_z = get_axis()
        last_theta = np.arccos(np.dot(last_z, target_vec))
        for _ in range(max_steps):
            curr_z = get_axis()
            curr_qpos = self._robot_manager.get_gripper_qpos()
            curr_depth = torch.min(self._tactile_manager.get_min_depth()).item()

            theta = np.arccos(np.dot(curr_z, target_vec))
            if theta < 0.05 or theta > last_theta:
                break
            omega = theta - last_theta
            last_theta = theta

            if np.abs(omega) < omega_threshold:
                if curr_depth < contact_threshold[1]:
                    curr_qpos += 0.0001
            elif curr_depth > contact_threshold[0]:
                curr_qpos -= 0.0001

            position = torch.tensor([curr_qpos, curr_qpos],
                                    dtype=torch.float32, device=self._robot_manager.device)
            velocity = torch.clip((position - curr_qpos)/self.cfg.sim.dt, -0.0001, 0.0001)
            self._robot_manager.set_gripper(position, velocity)

            for _ in range(5):
                self._step(is_save)
            last_z = curr_z
        self.move(self.atom.close_gripper())
    
    def gripper_rotate(self, actor:Actor, theta, steps:int=6, is_save=True):
        if self.plan_success is False:
            return False

        for i in range(steps):
            rpy = [0, theta/steps, 0]
            actor_pose = actor.get_pose()
            gripper_center_pose = self._robot_manager.get_gripper_center_pose()
            new_gripper_center = gripper_center_pose.add_rotation(rpy, coord=actor_pose)
            new_gripper_center.q = gripper_center_pose.q.copy()
            new_target_pose = self._robot_manager.gripper_center_to_ee(new_gripper_center)
            self.move([Action(
                action='move', target_pose=new_target_pose
            )], tag='rotate', is_save=is_save, delay=False, time_dilation_factor=0.5)
    
    def try_forward(self, actor:Actor, dis=0.01, delta_d=0.004, is_save=True):
        if self.plan_success is False:
            return False

        actor_last_pose = actor.get_pose()
        max_trials = int(np.ceil(np.abs(dis/delta_d)))
        delta = np.sign(dis) * delta_d
        for i in range(max_trials):
            success = self.move(self.atom.move_by_displacement(
                z=delta, xyz_coord='local'
            ), tag='try_forward', is_save=is_save, delay=False, cosntraint_pose=[1, 1, 1, 1, 1, 0])
            actor_pose = actor.get_pose()
            if np.linalg.norm(actor_pose.p - actor_last_pose.p) < np.abs(delta):
                return False
            actor_last_pose = actor_pose
        return True

    def try_forward(self, actor:Actor, dis=0.01, delta_d=0.004, is_save=True):
        if self.plan_success is False:
            return False

        actor_last_pose = actor.get_pose()
        max_trials = int(np.ceil(np.abs(dis/delta_d)))
        delta = np.sign(dis) * delta_d
        for i in range(max_trials):
            success = self.move(self.atom.move_by_displacement(
                z=delta, xyz_coord='local'
            ), tag='try_forward', is_save=is_save, delay=False)
            actor_pose = actor.get_pose()
            if np.linalg.norm(actor_pose.p - actor_last_pose.p) < np.abs(delta):
                return False
            actor_last_pose = actor_pose
        return True
