from collections import defaultdict
from pathlib import Path
import transforms3d as t3d
from functools import partial
from typing import Literal, Generator

from pxr import Usd, Vt, Gf, Sdf, UsdShade, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.utils.math import *
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg

from tacex_uipc import (
    UipcRLEnv,
    UipcIsaacAttachments,
    UipcIsaacAttachmentsCfg,
    UipcObject,
    UipcObjectCfg,
    UipcSimCfg,
    UipcSim
)
from uipc import Animation, builtin, view
from uipc.constitution import SoftTransformConstraint, SoftPositionConstraint
from uipc.geometry import GeometrySlot, SimplicialComplex
from tacex_uipc.utils import TetMeshCfg
from .transforms import *
from .._global import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .._base_task import BaseTask

@configclass
class ActorCfg(UipcObjectCfg):
    name: str = 'actor'
    asset: str = None
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    extents: tuple[float, float, float] = (0.1, 0.1, 0.1)

    # Target point matrix (multiple), target points are special points that can be accessed during planning (e.g., cup handle)
    target_points: list = []
    # Grasping point matrix (multiple), grasping points are the positions where the robotic arm grasps the object (e.g., cup mouth)
    contact_points: list = []
    # Functional point matrix (multiple), functional points are the positions where the object interacts with other objects (e.g., hammer head)
    functional_points: list = []
    # Orientation point matrix (single), orientation points specify the orientation of the object (e.g., shoe head facing left)
    orientation_points: list = []

    constraint_strength_ratio: float = 100

class Actor(UipcObject):
    cfg: ActorCfg

    def __init__(self, task: 'BaseTask', cfg: ActorCfg):
        self.task = task
        self.init_pose = Pose(cfg.init_state.pos, cfg.init_state.rot)
        super().__init__(cfg, task.uipc_sim)
        task.scene.uipc_objects[cfg.name] = self

        self.next_status, self.next_pts, self.next_mat = None, None, None
        self.next_status, self.next_mat, self.next_pts = None, None, None
        if isinstance(self.cfg.constitution_cfg, UipcObjectCfg.AffineBodyConstitutionCfg):
            self.actor_type = 'affine_body'
            soft_transform_constraint = SoftTransformConstraint()
            soft_transform_constraint.apply_to(self.uipc_meshes[0], np.array([
                self.cfg.constraint_strength_ratio, self.cfg.constraint_strength_ratio
            ]))
        else:
            self.actor_type = 'soft_body'
            soft_position_constraint = SoftPositionConstraint()
            soft_position_constraint.apply_to(self.uipc_meshes[0], self.cfg.constraint_strength_ratio)
        
    def _initialize_impl(self):
        ret = super()._initialize_impl()
        self.origin_surf_pts = self.vertices - self.init_pose.p
        self.origin_surf_pts = self.origin_surf_pts @ self.init_pose.R

        self.animator = self._uipc_sim.scene.animator()
        self.animator.insert(self.uipc_scene_objects[0], self._set_pose_animate)
        return ret

    @classmethod
    def from_usd_file(cls, task: 'BaseTask', name:str, asset_path, pose:Pose, constitution_cfg=None, density=1e3):
        asset_path = Path(asset_path)
        if not asset_path.is_absolute():
            asset_path = OBJECTS_ROOT / asset_path
        asset_path = str(asset_path.absolute())
        
        cfg = ActorCfg(
            name=name,
            asset=asset_path,
            prim_path=f"/World/envs/env_.*/{name}",
            init_state=AssetBaseCfg.InitialStateCfg(pos=pose.p, rot=pose.q),
            spawn=sim_utils.UsdFileCfg(
                usd_path=asset_path,
                mass_props=sim_utils.MassPropertiesCfg(density=density),
            ),
            constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg() \
                if constitution_cfg is None else constitution_cfg,
            mass_density=density,
        )
        return cls(task, cfg)

    def get_pose(self, type:Literal['pose', 'matrix']='pose'):
        mat = estimate_rigid_transform(self.origin_surf_pts, self.vertices)
        if type == 'matrix':
            return mat
        else:
            return Pose.from_matrix(mat)
    
    def _set_pose_animate(self, info: Animation.UpdateInfo):
        if self.next_status is None:
            return 

        geo_slots: list[GeometrySlot] = info.geo_slots()
        if len(geo_slots) == 0:
            return
        geo: SimplicialComplex = geo_slots[0].geometry()
 
        if self.actor_type == 'affine_body':
            is_constrained = geo.instances().find(builtin.is_constrained)
            if self.next_status == 'unset':
                view(is_constrained)[:] = 0
            else:
                view(is_constrained)[:] = 1
                aim_transform_view = view(geo.instances().find(builtin.aim_transform))
                aim_transform_view[:] = self.next_mat.reshape(1, 4, 4)
        else:
            is_constrained = geo.vertices().find(builtin.is_constrained)
            if self.next_status == 'unset':
                view(is_constrained)[:] = 0
            else:
                view(is_constrained)[:] = 1
                aim_position_view = view(geo.vertices().find(builtin.aim_position))
                aim_position_view[:] = self.next_pts.reshape(1, -1, 3)
 
        if self.next_status == 'unset':
            self.next_status = None

    def set_pose(self, pose:Pose):
        mat = torch.tensor(
            pose.to_transformation_matrix() @ np.linalg.inv(self.init_pose.to_transformation_matrix()), dtype=torch.float64, device=self._device)
 
        self.next_pts = (self.init_vertex_pos @ mat[:3, :3].T + mat[:3, 3]).cpu().numpy()
        self.next_mat = mat.cpu().numpy()
        self.next_status = 'set'
    
    def remove_animate(self):
        self.next_status = 'unset'

    def set_texture(self, mdl_path:str, rng=None):
        prim = self._prim_view.prims[0]
        prim_path = str(prim.GetPath())
        self._set_texture(prim_path, mdl_path, rng=rng)
    
    @staticmethod
    def _set_texture(prim_path:str, mdl_path:str, rng=None):
        if rng is None:
            rng = np.random
        def find_mesh(prim):
            if prim.GetTypeName() == "Mesh":
                return prim
            for child in prim.GetChildren():
                mesh_prim = find_mesh(child)
                if mesh_prim is not None:
                    return mesh_prim
            return None

        if mdl_path == 'random':
            mdl_files = list(TEXTURES_ROOT.glob('*.mdl'))
            mdl_path:Path = rng.choice(mdl_files)

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        mesh = UsdGeom.Mesh(find_mesh(prim))
        
        if not stage.GetPrimAtPath(f'/World/envs/env_0/ground_plate/Looks/{mdl_path.stem}').IsValid():
            success, result = omni.kit.commands.execute(
                'CreateMdlMaterialPrimCommand',
                mtl_url=str(mdl_path),
                mtl_name=mdl_path.stem,
                mtl_path=f'/World/envs/env_0/ground_plate/Looks/{mdl_path.stem}'
            )
            if not success:
                return 

        mtl = UsdShade.Material(stage.GetPrimAtPath(f'/World/envs/env_0/ground_plate/Looks/{mdl_path.stem}'))
        shader = UsdShade.Shader(mtl.GetPrim().GetChild("Shader"))
        shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mtl)

    @property
    def points(self):
        return {
            'contact': self.cfg.contact_points,
            'target': self.cfg.target_points,
            'functional': self.cfg.functional_points,
            'orientation': self.cfg.orientation_points
        }

    @property
    def vertices(self):
        all_trimesh_points = self._uipc_sim.sio.simplicial_surface(2).positions().view().reshape(-1, 3)
        surf_points = all_trimesh_points[
            self._uipc_sim._surf_vertex_offsets[self.obj_id - 1] : self._uipc_sim._surf_vertex_offsets[
                self.obj_id
            ]
        ]
        return surf_points
    
    def get_point(
        self,
        type:Literal['contact', 'target', 'functional', 'orientation'],
        idx:int,
        ret:Literal['pose', 'matrix']='pose'
    ):
        points = self.points[type]
        if idx >= len(points):
            raise IndexError(f"Index {idx} out of range for {type} points.")
        
        local_matrix = points[idx]
        actor_matrix = self.get_pose('matrix')
        world_matrix = actor_matrix @ local_matrix

        if ret == 'matrix':
            return world_matrix
        else:
            return Pose.from_matrix(world_matrix)
    
    def iter_point(
        self,
        type:Literal['contact', 'target', 'functional', 'orientation'],
        ret:Literal['pose', 'matrix']='pose'
    ) -> Generator:
        points = self.points[type]
        for idx in range(len(points)):
            yield self.get_point(type, idx, ret)

    def register_point(
        self,
        pose: Pose,
        type:Literal['contact', 'target', 'functional', 'orientation'],
    ):
        actor_matrix = self.get_pose('matrix')
        world_matrix = pose.to_transformation_matrix()
        local_matrix = np.linalg.inv(actor_matrix) @ world_matrix
        self.points[type].append(local_matrix)
        return len(self.points[type]) - 1

class ActorManager:
    def __init__(self, task: 'BaseTask'):
        self.task = task
        self.actors: dict[str, Actor] = {}

    def add_from_usd_file(self, name:str, asset_path:str, pose:Pose, constitution_cfg=None, density=1e3):
        actor = Actor.from_usd_file(
            self.task, name, asset_path, pose,
            constitution_cfg=constitution_cfg, density=density
        )
        self.actors[actor.cfg.name] = actor
        return actor
    
    def _reset_idx(self, rng=None):
        for actor in self.actors.values():
            # actor.write_vertex_positions_to_sim(vertex_positions=actor.init_vertex_pos)
            if self.task.cfg.random_texture:
                actor.set_texture('random', rng=rng)
 
    def update(self, dt):
        for actor in self.actors.values():
            actor.update(dt=dt)
    
    def remove_animate(self):
        for actor in self.actors.values():
            actor.remove_animate()
    
    def get_observations(self):
        obs = {}
        for name, actor in self.actors.items():
            obs[name] = actor.get_pose().totensor()
        return obs