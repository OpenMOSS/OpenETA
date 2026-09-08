"""Opt-in goal-pad intervention and read-only initialization evidence."""
from __future__ import annotations

import copy
import functools
import inspect
import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np

from sim.envs.univtac.autonomous_operation import (
    NativeController,
    append_row,
    pose_dict,
    tensor_array,
)
from sim.envs.univtac.tactile_history import TactileRecorder

PADS = ('green_pad', 'orange_pad')


def pad_arguments(condition, name, pose, motion_type):
    if name not in PADS:
        return pose, motion_type
    if motion_type != 'dynamic' or not np.isclose(pose.p[2], .002, atol=1e-10):
        raise ValueError('Goal-pad original configuration differs from dynamic z=0.002')
    if condition == 'K':
        return pose, 'kinematic'
    if condition == 'Z':
        pose = copy.deepcopy(pose)
        pose.p[2] = .010
    return pose, motion_type


def transformed_surface(points, matrix, ground_height, ground_normal):
    world = np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]
    normal = np.asarray(ground_normal, dtype=float)
    normal /= np.linalg.norm(normal)
    # Native ground(height, normal) uses the point (0,0,height).
    distances = (world-np.array([0., 0., ground_height])) @ normal
    return {'world_vertices_m':world.tolist(), 'world_min_m':world.min(0).tolist(),
            'world_max_m':world.max(0).tolist(), 'min_signed_ground_gap_m':float(distances.min()),
            'gap_method':'minimum over actual piecewise-linear boundary vertices against implicit plane; not AABB overlap'}


class InitializationState:
    """Read robot buffers for the existing recorder; no controller or checker."""
    def __init__(self, task):
        self.task, self.robot = task, task._robot_manager

    def counts(self):
        n = int(self.task._physics_step_count)
        return {'physics_steps':n, 'control_steps':int(self.task.step_count),
                'simulation_time_seconds':n*float(self.task.cfg.sim.dt)}

    def state(self):
        robot = self.robot
        return {**pose_dict(NativeController.tcp(self)),
                'joint_positions_rad':tensor_array(robot.get_qpos()).reshape(-1).tolist(),
                'gripper_finger_positions_m':tensor_array(robot.get_gripper_qpos_all()).reshape(-1).tolist(),
                'gripper_target_positions_m':tensor_array(robot.robot._joint_pos_target_sim[0,robot._gripper_ids]).tolist(),
                'gripper_command':'official_initialization'}

    def terminal(self):
        return None


class PadSceneDiagnostic:
    def __init__(self, root, condition):
        self.root, self.condition = root, condition
        self.created = {};self.local_surfaces = {};self.restores = []
        self.task = self.recorder = None

    @contextmanager
    def creation(self, manager_cls):
        original = manager_cls.add_from_usd_file
        signature = inspect.signature(original)
        @functools.wraps(original)
        def add(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            name = bound.arguments['name']
            if name in PADS:
                pose, motion = pad_arguments(self.condition, name, bound.arguments['pose'], bound.arguments.get('motion_type','dynamic'))
                if self.condition != 'O':
                    bound.arguments.update(pose=pose,motion_type=motion)
            actor = original(*bound.args, **bound.kwargs)
            if name in PADS:
                from uipc import builtin, view
                from uipc.geometry import extract_surface
                mesh = actor.body.uipc_meshes[0]
                fixed = int(np.asarray(view(mesh.instances().find(builtin.is_fixed))).reshape(-1)[0])
                contact = mesh.meta().find(builtin.contact_element_id)
                if fixed != int(self.condition=='K') or contact is None:
                    raise ValueError('Goal-pad UIPC fixed/contact configuration did not take effect')
                self.local_surfaces[name] = np.asarray(extract_surface(mesh).positions().view()).reshape(-1,3).copy()
                self.created[name] = {'motion_type':actor.motion_type,'is_fixed':fixed,
                    'contact_element_id':np.asarray(view(contact)).tolist(), 'asset':bound.arguments['asset_path'],
                    'prim_path':actor.cfg.prim_path,'mesh_prim_path':str(actor.body._usd_mesh_prim.GetPath()),
                    'requested_initial_pose_m':np.asarray(bound.arguments['pose'].p).tolist(),
                    'initial_uipc_transform':np.asarray(mesh.transforms().view()).reshape(-1,4,4)[0].tolist(),
                    'collision_surface_local_vertices_m':self.local_surfaces[name].tolist()}
                (self.root/'pad_creation.json').write_text(json.dumps(self.created,indent=2))
            return actor
        manager_cls.add_from_usd_file = add
        try:
            yield
        finally:
            manager_cls.add_from_usd_file = original

    def snapshot(self, phase):
        task = self.task
        actors = {}
        for name, actor in task._actor_manager.actors.items():
            matrix = np.asarray(actor.get_pose('matrix')).copy()
            row = {'pose_matrix':matrix.tolist()}
            if name in PADS:
                row.update(transformed_surface(self.local_surfaces[name],matrix,task.uipc_sim.cfg.ground_height,task.uipc_sim.cfg.ground_normal))
                row['geometry_id'] = actor.body.geo_slot_list[0].id()
            actors[name] = row
        target = getattr(task,'target_pose',None)
        append_row(self.root/'pad_geometry.jsonl', {'phase':phase,'native_step':task.step_count,
            'physics_step':task._physics_step_count,'actors':actors,
            'robot':InitializationState(task).state(),
            'target_pose':target.to_transformation_matrix().tolist() if target is not None else None,
            'ground_height_m':task.uipc_sim.cfg.ground_height,'ground_normal':task.uipc_sim.cfg.ground_normal,
            'native_min_toi_pair':'unavailable: installed Python binding exposes no direct minimum-pair query'})

    def install(self, task, config):
        if set(self.created) != set(PADS):
            raise ValueError('Both native goal pads must be recorded before reset')
        self.task = task
        self.snapshot('after_construct_before_reset')
        # Read the already-updated buffers, bypassing lazy .data getters.
        cameras = task._camera_manager.cameras;tactiles = task._tactile_manager.tactiles
        camera_reader = SimpleNamespace(cameras=cameras,get_observations=lambda kinds:
            {n:{'rgb':s._data.output['rgb'].squeeze(0)} for n,s in cameras.items()})
        touch_reader = SimpleNamespace(tactiles=tactiles,get_observations=lambda kinds:
            {n:{'rgb_marker':t.sensor._data.output['marker_rgb'].squeeze(0)} for n,t in tactiles.items()})
        proxy = SimpleNamespace(_camera_manager=camera_reader,_tactile_manager=touch_reader)
        self.recorder = TactileRecorder(proxy,InitializationState(task),self.root,config['tactile_history'])
        self.recorder.action_id = 'official_initialization_no_agent'
        original_render = task._update_render
        def render(*args, **kwargs):
            result = original_render(*args, **kwargs)
            proxy.last_render = task.last_render
            proxy._last_render_physics_step = task._last_render_physics_step
            self.recorder.sample()
            return result
        task._update_render = render;self.restores.append((task,'_update_render',original_render))
        original_step = task._step
        def step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            self.snapshot('after_native_step')
            return result
        task._step = step;self.restores.append((task,'_step',original_step))

    def close(self):
        if self.task is not None:
            self.snapshot('final_before_close')
        if self.recorder:
            self.recorder.close()
        for obj,name,original in reversed(self.restores):
            setattr(obj,name,original)
