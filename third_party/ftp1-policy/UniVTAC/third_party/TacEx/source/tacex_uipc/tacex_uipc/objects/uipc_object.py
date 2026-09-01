# Copyright (c) 2022-2025, The TacEx Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import omni.physics.tensors.impl.api as physx
import omni.usd
import usdrt
import usdrt.UsdGeom
from isaacsim.core.prims import XFormPrim
from pxr import UsdGeom

try:
    from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
except Exception:
    import warnings

    warnings.warn("_debug_draw failed to import", ImportWarning)
    draw = None

import numpy as np

import warp as wp
from uipc import builtin, view
from uipc.constitution import AffineBodyConstitution, ElasticModuli, StableNeoHookean
from uipc.geometry import extract_surface, flip_inward_triangles, label_surface, label_triangle_orient, tetmesh
from uipc.unit import MPa

import isaaclab.utils.string as string_utils
from isaaclab.assets import AssetBase, AssetBaseCfg
from isaaclab.utils import configclass

wp.init()


from tacex_uipc.utils import MeshGenerator, TetMeshCfg

from .uipc_object_deformable_data import UipcObjectDeformableData
from .uipc_object_rigid_data import UipcObjectRigidData

if TYPE_CHECKING:
    from tacex_uipc.sim import UipcIsaacAttachmentsCfg, UipcSim


@configclass
class UipcObjectCfg(AssetBaseCfg):
    mesh_cfg: TetMeshCfg = None
    # contact_model:

    mass_density: float = 1e3

    @configclass
    class AffineBodyConstitutionCfg:
        # class_type = AffineBodyConstitution # doesn't work, cause no builtin signature found for AffineBodyConstitution class
        m_kappa: float = 100.0
        """Stiffness (hardness) of the object
        in [MPa]

        E.g. 100.0 MPa = hard-rubber-like material
        """

        kinematic: bool = False
        """Makes the DoF of the ABD body fixed.

        """

    @configclass
    class StableNeoHookeanCfg:
        # class_type = StableNeoHookean
        youngs_modulus: float = 0.01
        """
        in [MPa]
        """

        poisson_rate: float = 0.49
        """ Poission Rate

        Has to be < 0.5.
        """

    constitution_cfg: AffineBodyConstitutionCfg | StableNeoHookeanCfg = None

    attachment_cfg: UipcIsaacAttachmentsCfg = None


class UipcObject(AssetBase):
    """A rigid object asset class.

    Rigid objects are assets comprising of rigid bodies. They can be used to represent dynamic objects
    such as boxes, spheres, etc. A rigid body is described by its pose, velocity and mass distribution.

    For an asset to be considered a rigid object, the root prim of the asset must have the `USD RigidBodyAPI`_
    applied to it. This API is used to define the simulation properties of the rigid body. On playing the
    simulation, the physics engine will automatically register the rigid body and create a corresponding
    rigid body handle. This handle can be accessed using the :attr:`root_physx_view` attribute.

    .. note::

        For users familiar with Isaac Sim, the PhysX view class API is not the exactly same as Isaac Sim view
        class API. Similar to Isaac Lab, Isaac Sim wraps around the PhysX view API. However, as of now (2023.1 release),
        we see a large difference in initializing the view classes in Isaac Sim. This is because the view classes
        in Isaac Sim perform additional USD-related operations which are slow and also not required.

    .. _`USD RigidBodyAPI`: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html
    """

    cfg: UipcObjectCfg
    """Configuration instance for the rigid object."""

    def __init__(self, cfg: UipcObjectCfg, uipc_sim: UipcSim):
        """Initialize the uipc object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)
        self._uipc_sim: UipcSim = uipc_sim

        prim_paths_expr = self.cfg.prim_path  # + "/mesh"
        omni.log.info(f"Initializing uipc objects {prim_paths_expr}...")
        self._prim_view = XFormPrim(prim_paths_expr=prim_paths_expr, name=f"{prim_paths_expr}", usd=False)
        self._prim_view.initialize()

        self.stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())

        self.uipc_scene_objects = []
        self.geo_slot_list = []

        def find_mesh(prim):
            if prim.GetTypeName() == "Mesh":
                return prim
            for child in prim.GetChildren():
                mesh_prim = find_mesh(child)
                if mesh_prim is not None:
                    return mesh_prim
            return None

        self.uipc_meshes = []
        # setup tet meshes for uipc
        for (
            prim
        ) in (
            self._prim_view.prims
        ):  # todo dont loop over all prims of the view -> just take one base prim. Rather loop over the prim children?
            # need to access the mesh data of the usd prim
            prim_children = [find_mesh(prim)]
            usd_mesh = UsdGeom.Mesh(prim_children[0])
            usd_mesh_path = str(usd_mesh.GetPath())
            omni.log.info("usd_mesh_path ", usd_mesh_path)

            # Load precomputed mesh data from USD prim.
            tet_points = np.array(prim_children[0].GetAttribute("tet_points").Get())
            tet_indices = prim_children[0].GetAttribute("tet_indices").Get()
            surf_points = np.array(prim_children[0].GetAttribute("tet_surf_points").Get())
            tet_surf_indices = prim_children[0].GetAttribute("tet_surf_indices").Get()
            
            replace_color = False
            if tet_indices is None:
                mesh_gen = MeshGenerator(config=TetMeshCfg(
                    stop_quality=8,
                    max_its=100,
                    edge_length_r=1 / 5,
                    epsilon_r=0.001
                ))
                tet_points, tet_indices, surf_points, tet_surf_indices = mesh_gen.generate_tet_mesh_for_prim(
                    usd_mesh
                )
                replace_color = True

            # transform local tet points to world coor
            tf_world = omni.usd.get_world_transform_matrix(usd_mesh)

            tet_points_world = np.array(tf_world).T @ np.vstack((tet_points.T, np.ones(tet_points.shape[0])))
            tet_points_world = tet_points_world[:-1].T

            self.init_world_transform = torch.tensor(np.array(tf_world).T.copy(), device=self.uipc_sim.cfg.device)

            # uipc wants 2D array
            tet_indices = np.array(tet_indices).reshape(-1, 4)
            tet_surf_indices = np.array(tet_surf_indices).reshape(-1, 3)

            # create uipc mesh
            mesh = tetmesh(tet_points_world.copy(), tet_indices.copy())
            # enable the contact by labeling the surface
            label_surface(mesh)
            label_triangle_orient(mesh)
            # flip the triangles inward for better rendering
            mesh = flip_inward_triangles(mesh)  # todo idk if this makes a difference for us
            self.uipc_meshes.append(mesh)

            # libuipc uses different indexing for the surface topology
            surf = extract_surface(mesh)
            tet_surf_points_world = surf.positions().view().reshape(-1, 3)
            tet_surf_tri = surf.triangles().topo().view().reshape(-1).tolist()

            # Set Vertex and Triangle data into USD mesh for rendering, skip
            MeshGenerator.update_usd_mesh(
                prim=usd_mesh, surf_points=tet_surf_points_world, triangles=tet_surf_tri,
                replace_color=replace_color
            )

            # enable contact for uipc meshes etc.
            # mesh = self.uipc_meshes[0] #todo code properly cloned envs (i.e. for instanced objects?)
            self._create_constitutions(mesh)

            # setup mesh updates via Fabric
            fabric_prim = self.stage.GetPrimAtPath(usdrt.Sdf.Path(usd_mesh_path))
            if not fabric_prim:
                omni.log.warning(f"Prim at path {usd_mesh_path} is not in Fabric")
            if not fabric_prim.HasAttribute("points"):
                omni.log.warning(f"Prim at path {usd_mesh_path} does not have points attribute")

            # Tell OmniHydra to render points from Fabric
            if not fabric_prim.HasAttribute("Deformable"):
                fabric_prim.CreateAttribute("Deformable", usdrt.Sdf.ValueTypeNames.PrimTypeTag, True)

            # extract world transform
            rtxformable = usdrt.Rt.Xformable(fabric_prim)
            rtxformable.CreateFabricHierarchyWorldMatrixAttr()
            # set world matrix to identity matrix -> uipc already gives us vertices in world frame
            rtxformable.GetFabricHierarchyWorldMatrixAttr().Set(usdrt.Gf.Matrix4d())

            # update fabric mesh with world coor. points
            fabric_mesh_points_attr = fabric_prim.GetAttribute("points")
            fabric_mesh_points_attr.Set(usdrt.Vt.Vec3fArray(tet_surf_points_world))

            self.fabric_prim = fabric_prim

            # add fabric meshes to uipc sim class for updating the render meshes
            self._uipc_sim._fabric_meshes.append(fabric_prim)

            # save surface offsets for finding corresponding surface points of the meshes for rendering
            num_surf_points = tet_surf_points_world.shape[0]  # np.unique(tet_surf_indices)
            self._uipc_sim._surf_vertex_offsets.append(self._uipc_sim._surf_vertex_offsets[-1] + num_surf_points)

            # required for writing vertex positions to sim
            num_vertex_points = mesh.positions().view().shape[0]
            self._vertex_count = num_vertex_points

            # update local vertex offset of the subsystem
            self._uipc_sim._system_vertex_offsets[self._system_name].append(
                self._uipc_sim._system_vertex_offsets[self._system_name][-1] + self._vertex_count
            )
            self.local_system_id = len(self._uipc_sim._system_vertex_offsets[self._system_name]) - 1

            # will be updated once _uipc_sim.setup_sim() is called
            self.global_system_id = 0

            self._data = None

    """
    Properties
    """

    @property
    def data(self) -> UipcObjectDeformableData | UipcObjectRigidData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self._prim_view.count

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 since each object is a single rigid body.
        """
        return 1

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in the rigid object."""
        prim_paths = self.root_physx_view.prim_paths[: self.num_bodies]
        return [path.split("/")[-1] for path in prim_paths]

    @property
    def uipc_sim(self) -> physx.RigidBodyView:
        """uipc simulation instance of this uipc object."""
        return self._uipc_sim

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # TODO implement this
        pass
        # # resolve all indices
        # if env_ids is None:
        #     env_ids = slice(None)

    def write_data_to_sim(self):
        pass

    def update(self, dt: float):
        self._data.update(dt)

    """
    Operations - Finders.
    """

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the rigid body based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    """
    Operations - Write to simulation.
    """

    def write_vertex_positions_to_sim(self, vertex_positions: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        # physx_env_ids = env_ids
        # if env_ids is None:
        #     env_ids = slice(None)
        #     physx_env_ids = self._ALL_INDICES

        # # note: we need to do this here since tensors are not set into simulation until step.
        # # set into internal buffers
        # self._data.root_state_w[env_ids, :7] = root_pose.clone()
        # # convert root quaternion from wxyz to xyzw
        # root_poses_xyzw = self._data.root_state_w[:, :7].clone()
        # root_poses_xyzw[:, 3:] = math_utils.convert_quat(root_poses_xyzw[:, 3:], to="xyzw")
        # # set into simulation
        # self.root_physx_view.set_transforms(root_poses_xyzw, indices=physx_env_ids)
        # omni.log.info("")
        # omni.log.info(f"Write vertex pos for {self.cfg.prim_path} with obj id [{self.obj_id}]")

        # omni.log.info(f"num geo_slots: {len(self.geo_slot_list)}")
        # omni.log.info(f"global sys id: {self.global_system_id}")
        geo_slot = self.geo_slot_list[0]
        geo = geo_slot.geometry()
        gvo = geo.meta().find(builtin.global_vertex_offset)
        # omni.log.info(f"global Vertex Offset: {gvo.view()}")
        global_vertex_offset = int(gvo.view()[0])
        local_vertex_offset = self._uipc_sim._system_vertex_offsets[self._system_name][self.local_system_id - 1]
        # omni.log.info(f"system: {self._system_name}")
        # omni.log.info(f"local sys id: {self.local_system_id}")
        # omni.log.info(f"local vertex offset: {local_vertex_offset}")
        # omni.log.info(f"vertex count: {self._vertex_count}")
        # omni.log.info("")
        if self._system_name == "uipc::backend::cuda::AffineBodyDynamics":
            self.uipc_sim.world.write_vertex_pos_to_sim(
                vertex_positions.cpu().numpy(),
                global_vertex_offset,
                self.local_system_id - 1,
                self._vertex_count,
                self._system_name,
            )
        else:
            self.uipc_sim.world.write_vertex_pos_to_sim(
                vertex_positions.cpu().numpy(),
                global_vertex_offset,
                local_vertex_offset,
                self._vertex_count,
                self._system_name,
            )

    """
    Internal helper.
    """

    def _initialize_impl(self):
        # create objects in the uipc scene for the meshes
        mesh = self.uipc_meshes[0]

        obj = self._uipc_sim.scene.objects().create(self.cfg.prim_path)
        self.uipc_scene_objects.append(obj)

        obj_geo_slot, _ = obj.geometries().create(mesh)
        self.obj_id = obj_geo_slot.id()
        omni.log.info(f"obj id of {self.cfg.prim_path}: {self.obj_id} ")
        self.geo_slot_list.append(obj_geo_slot)

        # save initial world vertex positions
        geom = self._uipc_sim.scene.geometries()
        geo_slot, geo_slot_rest = geom.find(self.obj_id)
        self.init_vertex_pos = torch.tensor(
            geo_slot.geometry().positions().view().copy().reshape(-1, 3), device=self.device
        )

        # log information the uipc body
        omni.log.info(f"UIPC body initialized at: {self.cfg.prim_path}.")
        omni.log.info(f"Number of instances: {self.num_instances}")

        # create buffers

        # container for data access
        if type(self.constitution) is StableNeoHookean:
            self._data = UipcObjectDeformableData(self._uipc_sim, self, self.device)
        elif type(self.constitution) is AffineBodyConstitution:
            self._data = UipcObjectRigidData(self._uipc_sim, self, self.device)

        self._create_buffers()
        # process configuration
        self._process_cfg()
        # update the uipc_object data
        self.update(0.0)

        # add this object to the list of all uipc objects in the world
        self._uipc_sim.uipc_objects.append(self)

    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants
        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)

        # self._data._nodal_pos_w = torch.zeros(self.num_instances, self._vertex_count)

    #     # set information about rigid body into data
    #     self._data.body_names = self.body_names
    #     self._data.default_mass = self.root_physx_view.get_masses().clone()
    #     self._data.default_inertia = self.root_physx_view.get_inertias().clone()

    def _process_cfg(self):
        """Post processing of configuration parameters."""
        # default state
        # -- root state
        # note: we cast to tuple to avoid torch/numpy type mismatch.
        default_root_state = (
            tuple(self.cfg.init_state.pos)
            + tuple(self.cfg.init_state.rot)
            # + tuple(self.cfg.init_state.lin_vel)
            # + tuple(self.cfg.init_state.ang_vel)
        )
        default_root_state = torch.tensor(default_root_state, dtype=torch.float, device=self.device)
        # self._data.default_root_state = default_root_state.repeat(self.num_instances, 1)

    def _create_constitutions(self, mesh):
        # create constitutions
        constitution_types = {
            UipcObjectCfg.AffineBodyConstitutionCfg: AffineBodyConstitution,
            UipcObjectCfg.StableNeoHookeanCfg: StableNeoHookean,
        }
        self.constitution = constitution_types[type(self.cfg.constitution_cfg)]()

        if type(self.constitution) is StableNeoHookean:
            youngs = self.cfg.constitution_cfg.youngs_modulus
            poisson = self.cfg.constitution_cfg.poisson_rate
            moduli = ElasticModuli.youngs_poisson(youngs * MPa, poisson)
            # apply the constitution and contact model to the base mesh
            self.constitution.apply_to(mesh, moduli, mass_density=self.cfg.mass_density)
            # needed for writing vertex position to sim
            self._system_name = "uipc::backend::cuda::FiniteElementMethod"
        elif type(self.constitution) is AffineBodyConstitution:
            stiffness = self.cfg.constitution_cfg.m_kappa
            self.constitution.apply_to(mesh, stiffness * MPa, mass_density=self.cfg.mass_density)
            self._system_name = "uipc::backend::cuda::AffineBodyDynamics"

            # make ABD body kinematic
            if self.cfg.constitution_cfg.kinematic:
                is_fixed_attr = mesh.instances().find(builtin.is_fixed)
                view(is_fixed_attr)[0] = 1

        # apply the default contact model to the base mesh
        default_element = self._uipc_sim.scene.contact_tabular().default_element()
        default_element.apply_to(mesh)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._physics_sim_view = None
        self._root_physx_view = None
