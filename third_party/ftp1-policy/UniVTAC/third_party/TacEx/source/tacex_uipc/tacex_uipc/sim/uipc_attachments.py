from __future__ import annotations

import inspect
import numpy as np
import torch
import weakref

import omni
from omni.physx import get_physx_interface, get_physx_scene_query_interface
from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils

try:
    from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
except Exception:
    import warnings

    warnings.warn("_debug_draw failed to import", ImportWarning)
    draw = None

from uipc import Animation, builtin, view
from uipc.constitution import SoftPositionConstraint
from uipc.geometry import GeometrySlot, SimplicialComplex

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils import configclass
from isaaclab.utils.math import transform_points

from tacex_uipc.objects import UipcObject

@configclass
class UipcIsaacAttachmentsCfg:
    constraint_strength_ratio: float = 100.0
    """
    E.g., 100.0 means the stiffness of the constraint is 100 times of the mass of the uipc object.
    """

    debug_vis: bool = False
    """Draw attachment offsets and aim_position via IsaacSim's _debug_draw api.

    """

    body_name: str = None
    """Name of the body in the rigid object that should be used for the attachment.

    Useful, e.g. when attaching to a part of an articulation.
    """
    
    isaac_rigid_prim_path: str = None
    """Prim path of the isaac rigid object to attach to. If None, the prim path is set to robot_base_prim_path + "/.*" + body_name
    """

    compute_attachment_data: bool = True
    """False to use precomputed attachment data.

    Note: Precomputed attachment data is only valid if the corresponding precomputed tet mesh data exists.
    This means, that the uipc_object of the attachment class should also use precomputed data.
    """

    attachment_points_radius: float = 5e-4
    """Distance between tet points and isaac collider, which is used to determine the attachment points.

    If the collision mesh of the isaaclab_rigid_object is in the radius of a point, then the
    point is considered "attached" to the isaaclab_rigid_object.
    """


class UipcIsaacAttachments:
    cfg: UipcIsaacAttachmentsCfg

    # todo code init properly
    def __init__(
        self, cfg: UipcIsaacAttachmentsCfg, uipc_object: UipcObject, isaaclab_rigid_object: RigidObject | Articulation
    ) -> None:
        # check that the config is valid
        cfg.validate()
        self.cfg = cfg.copy()

        self.uipc_object: UipcObject = uipc_object
        # check if prim of uipc_object has rigid body API applied to it, if yes -> throw error
        if UsdPhysics.RigidBodyAPI(self.uipc_object._prim_view.prims[0]):
            raise RuntimeWarning(
                f"Prim {self.uipc_object.cfg.prim_path} of UIPC object is a Isaac Rigid Body. This (usually) leads to"
                " unwanted behavior (e.g. when part of articulation)."
            )

        self.isaaclab_rigid_object: RigidObject | Articulation = isaaclab_rigid_object

        self.rigid_body_id = None  # used to query the position of the rigid body

        self.uipc_object_vertex_indices = []
        self.attachment_points_init_positions = []

        # self.attachments_offsets_idx_range = [0]
        self.aim_positions = np.zeros(0)

        # create the attachment

        if not self.cfg.compute_attachment_data:
            # todo hmm, its kinda tricky. Can only use precomputed attachment data, if its the same tet mesh.
            # -- how can we make sure that this case?

            # Load precomputed attachmend data from USD prim.
            prim_children = self.uipc_object._prim_view.prims[0].GetChildren()  # todo properly deal with multiple meshs
            prim = prim_children[0]
            usd_mesh = UsdGeom.Mesh(prim)
            attachment_offsets = np.array(prim.GetAttribute("attachment_offsets").Get())
            idx = prim.GetAttribute("attachment_indices").Get()

            if idx is None:
                raise Exception(
                    f"No precomputed attachment data found for prim at {str(usd_mesh.GetPath())}. Use set a cfg for the"
                    " attachment to compute attachment data."
                )
        else:
            prim_children = self.uipc_object._prim_view.prims[0].GetChildren()
            prim = prim_children[0]
            att_ids = prim.GetAttribute("tet_att_indices").Get()
            if att_ids is None:
                # compute attachment
                attachment_points_radius = self.cfg.attachment_points_radius

                if self.cfg.isaac_rigid_prim_path is not None:
                    isaac_rigid_prim_path = self.cfg.isaac_rigid_prim_path
                else:
                    isaac_rigid_prim_path = self.isaaclab_rigid_object.cfg.prim_path
                    if self.cfg.body_name is not None:
                        isaac_rigid_prim_path += "/.*" + self.cfg.body_name
                print("isaac_rigid_prim ", isaac_rigid_prim_path)

                mesh = self.uipc_object.uipc_meshes[0]
                tet_points_world = mesh.positions().view()[:, :, 0]
                tet_indices = mesh.tetrahedra().topo().view()[:, :, 0]
                attachment_offsets, idx, rigid_prims, attachment_points_pos, obj_pos = self.compute_attachment_data(
                    isaac_rigid_prim_path, tet_points_world, tet_indices, attachment_points_radius
                )
            else:
                # compute attachment
                attachment_points_radius = self.cfg.attachment_points_radius

                if self.cfg.isaac_rigid_prim_path is not None:
                    isaac_rigid_prim_path = self.cfg.isaac_rigid_prim_path
                else:
                    isaac_rigid_prim_path = self.isaaclab_rigid_object.cfg.prim_path
                    if self.cfg.body_name is not None:
                        isaac_rigid_prim_path += "/.*" + self.cfg.body_name
                    print("isaac_rigid_prim ", isaac_rigid_prim_path)

                mesh = self.uipc_object.uipc_meshes[0]
                tet_points_world = mesh.positions().view()[:, :, 0]
                tet_indices = mesh.tetrahedra().topo().view()[:, :, 0]
                attachment_offsets = self.compute_attachment_data_from_given_ids(
                    isaac_rigid_prim_path, tet_points_world, tet_indices, att_ids
                )
                idx = list(att_ids)

        # set attachment data
        self.attachment_offsets = attachment_offsets
        self.attachment_points_idx = idx
        self.num_attachment_points_per_obj = len(idx)

        self._num_instances = 1

        # set uipc constraint
        soft_position_constraint = SoftPositionConstraint()
        # todo handle multiple meshes properly (currently just single mesh)
        soft_position_constraint.apply_to(self.uipc_object.uipc_meshes[0], self.cfg.constraint_strength_ratio)

        # flag for whether the asset is initialized
        self._is_initialized = False

        # note: Use weakref on all callbacks to ensure that this object can be deleted when its destructor is called.
        # add callbacks for stage play/stop
        # The order is set to 10 which is arbitrary but should be lower priority than the default order of 0
        timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        self._initialize_handle = timeline_event_stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.PLAY),
            lambda event, obj=weakref.proxy(self): obj._initialize_callback(event),
            order=10,
        )
        self._invalidate_initialize_handle = timeline_event_stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP),
            lambda event, obj=weakref.proxy(self): obj._invalidate_initialize_callback(event),
            order=10,
        )
        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def __del__(self):
        """Unsubscribe from the callbacks."""
        # clear physics events handles
        if self._initialize_handle:
            self._initialize_handle.unsubscribe()
            self._initialize_handle = None
        if self._invalidate_initialize_handle:
            self._invalidate_initialize_handle.unsubscribe()
            self._invalidate_initialize_handle = None
        # clear debug visualization
        if self._debug_vis_handle:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    """
    Properties
    """

    @property
    def is_initialized(self) -> bool:
        """Whether the asset is initialized.

        Returns True if the asset is initialized, False otherwise.
        """
        return self._is_initialized

    @property
    def num_instances(self) -> int:
        """Number of instances of the asset.

        This is equal to the number of asset instances per environment multiplied by the number of environments.
        """
        return self._num_instances

    @property
    def device(self) -> str:
        """Memory device for computation."""
        return self._device

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the asset has a debug visualization implemented."""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self._set_debug_vis_impl)
        return "NotImplementedError" not in source_code

    """
    Operations.
    """

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Sets whether to visualize the asset data.

        Args:
            debug_vis: Whether to visualize the asset data.

        Returns:
            Whether the debug visualization was successfully set. False if the asset
            does not support debug visualization.
        """
        # check if debug visualization is supported
        if not self.has_debug_vis_implementation:
            return False
        # toggle debug visualization objects
        self._set_debug_vis_impl(debug_vis)
        # toggle debug visualization handles
        if debug_vis:
            # create a subscriber for the post update event if it doesn't exist
            if self._debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_pre_update_event_stream().create_subscription_to_pop(  # get_post_update_event_stream get_pre_update_event_stream
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:
            # remove the subscriber if it exists
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None
        # return success
        return True

    @staticmethod
    def compute_attachment_data(
        isaac_mesh_path, tet_points, tet_indices, sphere_radius=5e-4, max_dist=1e-5
    ):  # really small distances to prevent intersection with unwanted geometries
        """

        Returns: attachment_offsets, attachment_indices and the found rigid prims to the isaac_mesh_path
        """
        print(f"Creating Uipc x Isaac attachments for {isaac_mesh_path}")
        # force Physx to cook everything in the scene so it get cached
        get_physx_interface().force_load_physics_from_usd()

        # check if base asset path is valid
        # note: currently the spawner does not work if there is a regex pattern in the leaf
        #   For example, if the prim path is "/World/Robot_[1,2]" since the spawner will not
        #   know which prim to spawn. This is a limitation of the spawner and not the asset.
        # asset_path = isaac_mesh_path.split("/")[-1]
        # asset_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", asset_path) is None

        # check that spawn was successful
        matching_prims = sim_utils.find_matching_prims(isaac_mesh_path)
        if len(matching_prims) == 0:
            raise RuntimeError(
                f"Could not find prim with path {isaac_mesh_path}. The body_name in the cfg might not exist."
            )
        init_prim = matching_prims[0]

        pose = omni.usd.get_world_transform_matrix(init_prim)
        obj_position = pose.ExtractTranslation()
        obj_position = np.array([obj_position])

        q = pose.ExtractRotation().GetQuaternion()
        obj_orientation = [q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]]
        obj_orientation = torch.tensor(np.array([obj_orientation]), device="cuda:0").float()

        idx = []
        attachment_points_positions = []
        attachment_offsets = []

        # get position of current object for offset computation
        obj_pos = obj_position[0, :]
        # print("obj_pos ", obj_pos)
        # print("orient ", obj_orientation)

        # uipc_mesh = self.uipc_object.uipc_meshes[0]
        # vertex_positions = torch.tensor(uipc_mesh.positions().view().copy().reshape(-1,3), device=self.device)
        # print("vertex_positions ", vertex_positions)

        # topology = uipc_mesh.topo().view().reshape(-1).tolist()
        # print("topology ", topology)

        vertex_positions = tet_points
        # indices = tet_indices

        for i, v in enumerate(vertex_positions):
            # print("raycast ", i)
            ray_dir = [
                0,
                0,
                1,
            ]  # unit direction of the sphere ray cast -> doesn't really matter here, cause we only use a super short ray.
            
            hitInfo = get_physx_scene_query_interface().sweep_sphere_closest(
                radius=sphere_radius, origin=v, dir=ray_dir, distance=max_dist, bothSides=True
            )
            if hitInfo["hit"]:
                # print("hiiiit, ", hitInfo["collision"])
                if str(init_prim.GetPath()) in hitInfo["collision"]:  # prevent attaching to unrelated geometry
                    attachment_points_positions.append(v)
                    # idx.append(i+min_vertex_idx) unlike the gipc simulation, we use the object specific idx here
                    idx.append(i)
                    # TODO do this at the end and compute in a vectorized fashion?
                    # compute offsets from object position to attachment points
                    offset = v - obj_pos
                    offset = torch.tensor(offset, device="cuda:0").float()
                    offset = math_utils.quat_apply_inverse(obj_orientation[0].reshape((1, 4)), offset.reshape((1, 3)))[
                        0
                    ]
                    offset = offset.cpu().numpy()
                    attachment_offsets.append(offset)

        attachment_points_positions = np.array(attachment_points_positions).reshape(-1, 3)

        # offset to later compute the `should-be` positions of the attachment point
        attachment_offsets = np.array(attachment_offsets).reshape(-1, 3)
        assert len(idx) == attachment_offsets.shape[0]

        # print("offsets, ", attachment_offsets)
        # print("attachment local idx, ", idx)
        # print("Init pos, ", attachment_points_positions)

        # self._create_attachment_data_attributes(isaac_mesh_path, self.objects_gipc, idx, attachment_points_positions, attachment_offsets)

        # # draw attachment data
        # draw.draw_points(attachment_points_positions, [(255,0,0,0.5)]*attachment_points_positions.shape[0], [30]*attachment_points_positions.shape[0]) # the new positions
        # obj_center = obj_position[0]

        # for j in range(0, attachment_points_positions.shape[0]):
        #     draw.draw_lines([obj_center], [attachment_points_positions[j,:]], [(255,255,0,0.5)], [10])

        return attachment_offsets, idx, matching_prims, attachment_points_positions, obj_pos
    
    @staticmethod
    def compute_attachment_data_from_given_ids(
        isaac_mesh_path, tet_points, tet_indices, att_ids
    ):  # really small distances to prevent intersection with unwanted geometries
        """

        Returns: attachment_offsets, attachment_indices and the found rigid prims to the isaac_mesh_path
        """
        print(f"Creating Uipc x Isaac attachments for {isaac_mesh_path}")
        # force Physx to cook everything in the scene so it get cached
        get_physx_interface().force_load_physics_from_usd()

        # check if base asset path is valid
        # note: currently the spawner does not work if there is a regex pattern in the leaf
        #   For example, if the prim path is "/World/Robot_[1,2]" since the spawner will not
        #   know which prim to spawn. This is a limitation of the spawner and not the asset.
        # asset_path = isaac_mesh_path.split("/")[-1]
        # asset_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", asset_path) is None

        # check that spawn was successful
        matching_prims = sim_utils.find_matching_prims(isaac_mesh_path)
        if len(matching_prims) == 0:
            raise RuntimeError(
                f"Could not find prim with path {isaac_mesh_path}. The body_name in the cfg might not exist."
            )
        init_prim = matching_prims[0]

        pose = omni.usd.get_world_transform_matrix(init_prim)
        obj_position = pose.ExtractTranslation()
        obj_position = np.array([obj_position])

        q = pose.ExtractRotation().GetQuaternion()
        obj_orientation = [q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]]
        obj_orientation = torch.tensor(np.array([obj_orientation]), device="cuda:0").float()

        attachment_points_positions = []
        attachment_offsets = []

        # get position of current object for offset computation
        obj_pos = obj_position[0, :]
        vertex_positions = tet_points
        for i in att_ids:
            v = vertex_positions[i]
            attachment_points_positions.append(v)
            offset = v - obj_pos
            offset = torch.tensor(offset, device="cuda:0").float()
            offset = math_utils.quat_apply_inverse(
                obj_orientation[0].reshape((1, 4)), offset.reshape((1, 3)))[0]
            offset = offset.cpu().numpy()
            attachment_offsets.append(offset)

        attachment_points_positions = np.array(attachment_points_positions).reshape(-1, 3)
        attachment_offsets = np.array(attachment_offsets).reshape(-1, 3)
        return attachment_offsets

    @staticmethod
    def compute_attachment_data_from_given_ids(
        isaac_mesh_path, tet_points, tet_indices, att_ids
    ):  # really small distances to prevent intersection with unwanted geometries
        """

        Returns: attachment_offsets, attachment_indices and the found rigid prims to the isaac_mesh_path
        """
        print(f"Creating Uipc x Isaac attachments for {isaac_mesh_path}")
        # force Physx to cook everything in the scene so it get cached
        get_physx_interface().force_load_physics_from_usd()

        # check if base asset path is valid
        # note: currently the spawner does not work if there is a regex pattern in the leaf
        #   For example, if the prim path is "/World/Robot_[1,2]" since the spawner will not
        #   know which prim to spawn. This is a limitation of the spawner and not the asset.
        # asset_path = isaac_mesh_path.split("/")[-1]
        # asset_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", asset_path) is None

        # check that spawn was successful
        matching_prims = sim_utils.find_matching_prims(isaac_mesh_path)
        if len(matching_prims) == 0:
            raise RuntimeError(
                f"Could not find prim with path {isaac_mesh_path}. The body_name in the cfg might not exist."
            )
        init_prim = matching_prims[0]

        pose = omni.usd.get_world_transform_matrix(init_prim)
        obj_position = pose.ExtractTranslation()
        obj_position = np.array([obj_position])

        q = pose.ExtractRotation().GetQuaternion()
        obj_orientation = [q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]]
        obj_orientation = torch.tensor(np.array([obj_orientation]), device="cuda:0").float()

        attachment_points_positions = []
        attachment_offsets = []

        # get position of current object for offset computation
        obj_pos = obj_position[0, :]
        vertex_positions = tet_points
        for i in att_ids:
            v = vertex_positions[i]
            attachment_points_positions.append(v)
            offset = v - obj_pos
            offset = torch.tensor(offset, device="cuda:0").float()
            offset = math_utils.quat_apply_inverse(
                obj_orientation[0].reshape((1, 4)), offset.reshape((1, 3)))[0]
            offset = offset.cpu().numpy()
            attachment_offsets.append(offset)

        attachment_points_positions = np.array(attachment_points_positions).reshape(-1, 3)
        attachment_offsets = np.array(attachment_offsets).reshape(-1, 3)
        return attachment_offsets

    """
    Internal helper.
    """

    def _initialize_impl(self):
        if self.cfg.body_name is not None:
            self.rigid_body_id, found_body_name = self.isaaclab_rigid_object.find_bodies(self.cfg.body_name)

        self._create_animation()

        sim: sim_utils.SimulationContext = sim_utils.SimulationContext.instance()
        sim.add_physics_callback(
            f"{self.uipc_object.cfg.prim_path}_X_{self.isaaclab_rigid_object.cfg.prim_path}_attachment_update",
            self._compute_aim_positions,
        )

    def _create_animation(self):
        animator = self.uipc_object._uipc_sim.scene.animator()

        def animate_tet(info: Animation.UpdateInfo):  # animation function
            # print("test, ", self.aim_positions)
            if len(self.aim_positions) == 0:
                return
            
            geo_slots: list[GeometrySlot] = info.geo_slots()
            geo: SimplicialComplex = geo_slots[0].geometry()
            # rest_geo_slots: list[GeometrySlot] = info.rest_geo_slots()
            # rest_geo: SimplicialComplex = rest_geo_slots[0].geometry()

            is_constrained = geo.vertices().find(builtin.is_constrained)
            is_constrained_view = view(is_constrained)
            aim_position = geo.vertices().find(builtin.aim_position)
            aim_position_view = view(aim_position)
            # rest_position_view = rest_geo.positions().view()

            is_constrained_view[self.attachment_points_idx] = 1

            aim_position_view[self.attachment_points_idx] = self.aim_positions.reshape(-1, 3, 1)

        animator.insert(self.uipc_object.uipc_scene_objects[0], animate_tet)

    def _compute_aim_positions(self, dt=0):
        # make sure we have the newest data

        if type(self.isaaclab_rigid_object) is Articulation:
            # this only works when rigid body is an articulation
            # self.isaaclab_rigid_object._physics_sim_view.update_articulations_kinematic()
            # read data from simulation
            poses = self.isaaclab_rigid_object._root_physx_view.get_link_transforms().clone()
            poses[..., 3:7] = math_utils.convert_quat(poses[..., 3:7], to="wxyz")
            pose = poses[:, self.rigid_body_id, 0:7].clone()
        elif type(self.isaaclab_rigid_object) is RigidObject:
            # only works with rigid body
            pose = self.isaaclab_rigid_object._root_physx_view.root_state_w.view(-1, 1, 13)
            pose = pose[:, self.rigid_body_id, 0:7].clone()
        else:
            raise RuntimeError("Need an Articulation or a RigidBody object for the Isaac X UIPC attachment.")
        # - doing this is undesirable -> need to update the scene to get newest data
        # pose = self.isaaclab_rigid_object.data.body_state_w[:, self.rigid_body_id, 0:7].clone()

        self.obj_pose = pose

        # obj_pos = pose[:, 0, :3].cpu().numpy().flatten().reshape(-1,3)
        # self._draw.clear_points()
        # draw.draw_points(obj_pos, [(0,255,0,0.5)]*obj_pos.shape[0], [50]*obj_pos.shape[0]) # the new positions

        attachment_offsets = torch.tensor(
            self.attachment_offsets.reshape((self._num_instances, self.num_attachment_points_per_obj, 3)),
            device=self.device,
        ).float()
        # only access pose of a single body (that's why idx=0 in second dim)
        aim_pos = transform_points(
            attachment_offsets, pos=pose[:, 0, 0:3], quat=pose[:, 0, 3:]
        )  # todo give over batch of pos and quat for each instance (i.e. pos has shape N,3 and quat N,4)
        aim_pos = aim_pos.cpu().numpy()
        self.aim_positions = aim_pos.flatten().reshape(-1, 3)

        #      # extract velocity
        #      lin_vel = scene["robot"].data.body_state_w[:, robot_entity_cfg.body_ids[1], 7:10]
        #      lin_vel = lin_vel.cpu().numpy()
        #      lin_vel = np.tile(lin_vel, len(attachment_points)).reshape(len(attachment_points),3)

        return self.aim_positions

    """
    Internal simulation callbacks.

    Same as AssetBase class from asset_base.py
    """

    def _initialize_callback(self, event):
        """Initializes the scene elements.

        Note:
            PhysX handles are only enabled once the simulator starts playing. Hence, this function needs to be
            called whenever the simulator "plays" from a "stop" state.
        """
        if not self._is_initialized:
            # obtain simulation related information
            sim = sim_utils.SimulationContext.instance()
            if sim is None:
                raise RuntimeError("SimulationContext is not initialized! Please initialize SimulationContext first.")
            self._backend = sim.backend
            self._device = sim.device
            # initialize attachments
            self._initialize_impl()
            # set flag
            self._is_initialized = True

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        self._is_initialized = False

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            try:
                from isaacsim.util.debug_draw import _debug_draw

                self._draw = _debug_draw.acquire_debug_draw_interface()
            except ImportError:
                import warnings

                warnings.warn("_debug_draw failed to import", ImportWarning)
                self._draw = None
                print("No debug_vis for attachment. Reason: Cannot import _debug_draw")

    def _debug_vis_callback(self, event):
        if self.aim_positions.shape[0] == 0:
            return

        # self._compute_aim_positions()

        # # draw attachment data
        self._draw.clear_points()
        self._draw.clear_lines()

        # drawing with the debug method leads to render delay
        self._draw.draw_points(
            self.aim_positions, [(255, 0, 0, 0.5)] * self.aim_positions.shape[0], [60] * self.aim_positions.shape[0]
        )  # the new positions
        # pose = self.isaaclab_rigid_object.data.body_state_w[:, self.rigid_body_id, 0:7].clone()
        pose = self.obj_pose.clone()

        for i in range(self._num_instances):
            obj_center = pose[i, 0, 0:3].cpu().numpy()
            # self._draw.clear_points()
            # print("Obj_center_debug ", obj_center)
            # draw.draw_points([obj_center], [(255,255,0,0.5)]*obj_center.shape[0], [50]*obj_center.shape[0]) # the new positions
            # print("")
            for j in range(i, self._num_instances * self.num_attachment_points_per_obj):
                self._draw.draw_lines([obj_center], [self.aim_positions[j]], [(255, 255, 0, 0.5)], [10])
