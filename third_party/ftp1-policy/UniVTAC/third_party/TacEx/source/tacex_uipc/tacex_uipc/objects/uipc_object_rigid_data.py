from __future__ import annotations

import torch
import weakref
from typing import TYPE_CHECKING

from isaaclab.utils.buffers import TimestampedBuffer

try:
    from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
except Exception:
    import warnings

    warnings.warn("_debug_draw failed to import", ImportWarning)
    draw = None

if TYPE_CHECKING:
    from tacex_uipc.sim import UipcSim

    from .uipc_object import UipcObject


class UipcObjectRigidData:
    """Data container for a uipc object.

    This class contains the data for a rigid object in the simulation. The data includes the state of
    the root rigid body and the state of all the bodies in the object. The data is stored in the simulation
    world frame unless otherwise specified.

    For a rigid body, there are two frames of reference that are used:

    - Actor frame: The frame of reference of the rigid body prim. This typically corresponds to the Xform prim
      with the rigid body schema.
    - Center of mass frame: The frame of reference of the center of mass of the rigid body.

    Depending on the settings of the simulation, the actor frame and the center of mass frame may be the same.
    This needs to be taken into account when interpreting the data.

    The data is lazily updated, meaning that the data is only updated when it is accessed. This is useful
    when the data is expensive to compute or retrieve. The data is updated when the timestamp of the buffer
    is older than the current simulation timestamp. The timestamp is updated whenever the data is updated.
    """

    def __init__(self, uipc_sim: UipcSim, uipc_object: UipcObject, device: str):
        """Initializes the rigid object data.

        Args:
            root_physx_view: The root rigid body view.
            device: The device used for processing.
        """
        # Set the parameters
        self.device = device

        self._uipc_sim: UipcSim = weakref.proxy(uipc_sim)

        # Set the root rigid body view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._uipc_object: UipcObject = weakref.proxy(uipc_object)

        # Set initial time stamp
        self._sim_timestamp = 0.0

        # Initialize the lazy buffers.
        self._nodal_pos_w = TimestampedBuffer()
        self._root_state_w = TimestampedBuffer()
        self._root_link_state_w = TimestampedBuffer()
        self._root_com_state_w = TimestampedBuffer()
        self._body_acc_w = TimestampedBuffer()

    def update(self, dt: float):
        """Updates the data for the rigid object.

        Args:
            dt: The time step for the update. This must be a positive value.
        """
        # update the simulation timestamp
        self._sim_timestamp += dt

    ##
    # Names.
    ##

    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""

    ##
    # Defaults.
    ##

    default_nodal_state_w: torch.Tensor = None
    """Default nodal state ``[nodal_pos, nodal_vel]`` in simulation world frame.
    Shape is (num_instances, max_sim_vertices_per_body, 6).
    """

    ##
    # Kinematic commands
    ##

    nodal_kinematic_target: torch.Tensor = None
    """Simulation mesh kinematic targets for the deformable bodies.
    Shape is (num_instances, max_sim_vertices_per_body, 4).

    The kinematic targets are used to drive the simulation mesh vertices to the target positions.
    The targets are stored as (x, y, z, is_not_kinematic) where "is_not_kinematic" is a binary
    flag indicating whether the vertex is kinematic or not. The flag is set to 0 for kinematic vertices
    and 1 for non-kinematic vertices.
    """

    ##
    # Properties.
    ##

    @property
    def nodal_pos_w(self):
        """Nodal positions in simulation world frame. Shape is (num_instances, max_sim_vertices_per_body, 3)."""
        if self._nodal_pos_w.timestamp < self._sim_timestamp:
            # get current world vertex positions
            geom = self._uipc_sim.scene.geometries()
            geo_slot, geo_slot_rest = geom.find(
                self._uipc_object.obj_id
            )  # todo instead of finding obj, lets just save ref to geo_slot

            vertex_positions_world = torch.tensor(
                geo_slot.geometry().positions().view().reshape(-1, 3), device=self.device
            )
            self._nodal_pos_w.data = vertex_positions_world

            self._nodal_pos_w.timestamp = self._sim_timestamp
        return self._nodal_pos_w.data

    @property
    def surf_nodal_pos_w(self):
        """Nodal positions in simulation world frame. Shape is (num_instances, max_sim_vertices_per_body, 3)."""
        if self._nodal_pos_w.timestamp < self._sim_timestamp:
            all_trimesh_points = self._uipc_sim.sio.simplicial_surface(2).positions().view().reshape(-1, 3)
            surf_points = all_trimesh_points[
                self._uipc_sim._surf_vertex_offsets[self._uipc_object.obj_id - 1] : self._uipc_sim._surf_vertex_offsets[
                    self._uipc_object.obj_id
                ]
            ]
            self._nodal_pos_w.data = torch.tensor(surf_points, device=self.device, dtype=torch.float)

            self._nodal_pos_w.timestamp = self._sim_timestamp
        return self._nodal_pos_w.data

    ##
    # Derived properties.
    ##

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position from nodal positions of the simulation mesh for the deformable bodies in simulation world frame.
        Shape is (num_instances, 3).

        This quantity is computed as the mean of the nodal positions.
        """
        # return self.nodal_pos_w.mean(dim=1)
        geom = self._uipc_sim.scene.geometries()
        geo_slot, geo_slot_rest = geom.find(
            self._uipc_object.obj_id
        )  # todo instead of finding obj, lets just save ref to geo_slot in uipc_object

        # transformation is w.r.t. to initial pose
        # trans = geo_slot.geometry().instances().find(builtin.transform).view()
        # trans = torch.tensor(trans, device=self.device).reshape(1, 4, 4)
        # root_pos_w, root_orient_w = math_utils.unmake_pose(trans)
        # print("trans ", trans)

        # mean_pos = torch.tensor(geo_slot.geometry().positions().view().copy().reshape(-1, 3), device=self.device)

        root_pos_w = self.surf_nodal_pos_w.mean(dim=0)
        return root_pos_w.reshape(1, 3)  # todo need to adjust once we go multi env

    # @property
    # def root_vel_w(self) -> torch.Tensor:
    #     """Root velocity from vertex velocities for the deformable bodies in simulation world frame.
    #     Shape is (num_instances, 3).

    #     This quantity is computed as the mean of the nodal velocities.
    #     """
    #     return self.nodal_vel_w.mean(dim=1)
