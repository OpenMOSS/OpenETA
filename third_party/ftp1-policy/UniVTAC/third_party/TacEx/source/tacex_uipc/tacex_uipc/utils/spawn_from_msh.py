from __future__ import annotations

import omni.usd
import usdrt
from pxr import UsdGeom
from uipc.geometry import extract_surface, tetmesh

from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg, RigidObjectSpawnerCfg
from isaaclab.utils import configclass

from tacex_uipc.utils import MeshGenerator


@configclass
class FileCfg(RigidObjectSpawnerCfg, DeformableObjectSpawnerCfg):
    """Configuration parameters for spawning an asset from a file.

    This class is a base class for spawning assets from files. It includes the common parameters
    for spawning assets from files, such as the path to the file and the function to use for spawning
    the asset.

    Note:
        By default, all properties are set to None. This means that no properties will be added or modified
        to the prim outside of the properties available by default when spawning the prim.

        If they are set to a value, then the properties are modified on the spawned prim in a nested manner.
        This is done by calling the respective function with the specified properties.
    """

    scale: tuple[float, float, float] | None = None
    """Scale of the asset. Defaults to None, in which case the scale is not modified."""

    # articulation_props: schemas.ArticulationRootPropertiesCfg | None = None
    # """Properties to apply to the articulation root."""

    # fixed_tendons_props: schemas.FixedTendonsPropertiesCfg | None = None
    # """Properties to apply to the fixed tendons (if any)."""

    # joint_drive_props: schemas.JointDrivePropertiesCfg | None = None
    # """Properties to apply to a joint."""

    visual_material_path: str = "material"
    """Path to the visual material to use for the prim. Defaults to "material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `visual_material` is not None.
    """

    visual_material: materials.VisualMaterialCfg | None = None
    """Visual material properties to override the visual material properties in the URDF file.

    Note:
        If None, then no visual material will be added.
    """


# @configclass
# class MeshFileCfg(FileCfg):
#     """USD file to spawn asset from.

#     USD files are imported directly into the scene. However, given their complexity, there are various different
#     operations that can be performed on them. For example, selecting variants, applying materials, or modifying
#     existing properties.

#     To prevent the explosion of configuration parameters, the available operations are limited to the most common
#     ones. These include:

#     - **Selecting variants**: This is done by specifying the :attr:`variants` parameter.
#     - **Creating and applying materials**: This is done by specifying the :attr:`visual_material` parameter.
#     - **Modifying existing properties**: This is done by specifying the respective properties in the configuration
#       class. For instance, to modify the scale of the imported prim, set the :attr:`scale` parameter.

#     See :meth:`spawn_from_usd` for more information.

#     .. note::
#         The configuration parameters include various properties. If not `None`, these properties
#         are modified on the spawned prim in a nested manner.

#         If they are set to a value, then the properties are modified on the spawned prim in a nested manner.
#         This is done by calling the respective function with the specified properties.
#     """

#     func: Callable = from_files.spawn_from_usd

#     msh_path: str = MISSING
#     """Path to the USD file to spawn asset from."""

# @clone
# def spawn_from_mesh_file(
#     prim_path: str,
#     cfg: MeshFileCfg,
#     translation: tuple[float, float, float] | None = None,
#     orientation: tuple[float, float, float, float] | None = None,
# ) -> Usd.Prim:
#     """Spawn an asset from a msh file and override the settings with the given config.

#     In the case of a USD file, the asset is spawned at the default prim specified in the USD file.
#     If a default prim is not specified, then the asset is spawned at the root prim.

#     In case a prim already exists at the given prim path, then the function does not create a new prim
#     or throw an error that the prim already exists. Instead, it just takes the existing prim and overrides
#     the settings with the given config.

#     .. note::
#         This function is decorated with :func:`clone` that resolves prim path into list of paths
#         if the input prim path is a regex pattern. This is done to support spawning multiple assets
#         from a single and cloning the USD prim at the given path expression.

#     Args:
#         prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
#             then the asset is spawned at all the matching prim paths.
#         cfg: The configuration instance.
#         translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
#             case the translation specified in the USD file is used.
#         orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
#             in which case the orientation specified in the USD file is used.

#     Returns:
#         The prim of the spawned asset.

#     Raises:
#         FileNotFoundError: If the USD file does not exist at the given path.
#     """
#     # spawn asset from the given msh file if it doesn't exist in the stage.
#     if not prim_utils.is_prim_path_valid(prim_path):
#         # add prim as reference to stage
#         prim_utils.create_prim(
#             prim_path,
#             usd_path=usd_path,
#             translation=translation,
#             orientation=orientation,
#             scale=cfg.scale,
#         )
#     else:
#         omni.log.warn(f"A prim already exists at prim path: '{prim_path}'.")

#     return _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)

"""

Helper functions

"""


def create_prim_for_tet_data(prim_path, tet_points_world, tet_indices):
    # spawn a usd mesh in Isaac
    stage = omni.usd.get_context().get_stage()
    prim = UsdGeom.Mesh.Define(stage, prim_path)

    # extract surface from uipc computed tet mesh
    uipc_tet_mesh = tetmesh(tet_points_world.copy(), tet_indices.copy())
    surf = extract_surface(uipc_tet_mesh)
    tet_surf_tri = surf.triangles().topo().view().reshape(-1).tolist()
    tet_surf_points_world = surf.positions().view().reshape(-1, 3)

    MeshGenerator.update_usd_mesh(prim=prim, surf_points=tet_surf_points_world, triangles=tet_surf_tri)


def create_prim_for_uipc_scene_object(uipc_sim, prim_path, uipc_scene_object):
    # spawn a usd mesh in Isaac
    stage = omni.usd.get_context().get_stage()
    prim = UsdGeom.Mesh.Define(stage, prim_path)

    # get corresponding simplical complex from uipc_scene
    obj_id = uipc_scene_object.geometries().ids()[0]
    simplicial_complex_slot, _ = uipc_sim.scene.geometries().find(obj_id)

    # extract_surface
    surf = extract_surface(simplicial_complex_slot.geometry())
    tet_surf_tri = surf.triangles().topo().view().reshape(-1).tolist()
    tet_surf_points_world = surf.positions().view().reshape(-1, 3)

    MeshGenerator.update_usd_mesh(prim=prim, surf_points=tet_surf_points_world, triangles=tet_surf_tri)

    # setup mesh updates via Fabric
    fabric_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
    fabric_prim = fabric_stage.GetPrimAtPath(usdrt.Sdf.Path(prim_path))

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

    # add fabric meshes to uipc sim class for updating the render meshes
    uipc_sim._fabric_meshes.append(fabric_prim)

    # save indices to later find corresponding points of the meshes for rendering
    num_surf_points = tet_surf_points_world.shape[0]
    uipc_sim._surf_vertex_offsets.append(uipc_sim._surf_vertex_offsets[-1] + num_surf_points)
