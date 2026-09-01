# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to convert a OBJ/STL/FBX into USD format.

The OBJ file format is a simple data-format that represents 3D geometry alone — namely, the position
of each vertex, the UV position of each texture coordinate vertex, vertex normals, and the faces that
make each polygon defined as a list of vertices, and texture vertices.

An STL file describes a raw, unstructured triangulated surface by the unit normal and vertices (ordered
by the right-hand rule) of the triangles using a three-dimensional Cartesian coordinate system.

FBX files are a type of 3D model file created using the Autodesk FBX software. They can be designed and
modified in various modeling applications, such as Maya, 3ds Max, and Blender. Moreover, FBX files typically
contain mesh, material, texture, and skeletal animation data.
Link: https://www.autodesk.com/products/fbx/overview


This script uses the asset converter extension from Isaac Sim (``omni.kit.asset_converter``) to convert a
OBJ/STL/FBX asset into USD format. It is designed as a convenience script for command-line use.


positional arguments:
  input               The path to the input mesh (.OBJ/.STL/.FBX) file.
  output              The path to store the USD file.

optional arguments:
  -h, --help                    Show this help message and exit
  --make-instanceable,          Make the asset instanceable for efficient cloning. (default: False)
  --collision-approximation     The method used for approximating collision mesh. Defaults to convexDecomposition.
                                Set to \"none\" to not add a collision mesh to the converted mesh. (default: convexDecomposition)
  --mass                        The mass (in kg) to assign to the converted asset. (default: None)

"""

"""Launch Isaac Sim Simulator first."""


import os
import asyncio
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert a mesh file into USD format.")
parser.add_argument("--input", "-i", type=str, help="The path to the input mesh file.", default='assets/objects/ipt')
parser.add_argument("--output", "-o", type=str, help="The path to store the USD file.", default='assets/objects/opt')
parser.add_argument(
    "--make-instanceable",
    action="store_true",
    default=False,
    help="Make the asset instanceable for efficient cloning.",
)
parser.add_argument(
    "--collision-approximation",
    type=str,
    default="convexDecomposition",
    choices=["convexDecomposition", "convexHull", "boundingCube", "boundingSphere", "meshSimplification", "none"],
    help=(
        'The method used for approximating collision mesh. Set to "none" '
        "to not add a collision mesh to the converted mesh."
    ),
)
parser.add_argument(
    "--mass",
    type=float,
    default=None,
    help="The mass (in kg) to assign to the converted asset. If not provided, then no mass is added.",
)
parser.add_argument(
    '--show', action='store_true',
    help='Show trimesh visualization of the generated tetrahedral mesh.'
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True  # enforce headless mode

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os
import numpy as np

import carb
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app

from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict

import omni
import omni.kit.commands
import omni.usd
from omni.physx.scripts import deformableUtils
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Tf, Usd, UsdGeom, UsdPhysics, UsdUtils

from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
from isaaclab.sim.converters.mesh_converter_cfg import MeshConverterCfg
from isaaclab.sim.schemas import schemas
from isaaclab.sim.utils import export_prim_to_file

from pathlib import Path
from pxr import Usd, Sdf, Gf
from utils.mesh_gen import MeshGenerator, TetMeshCfg

def visualize_tet(tet_points, tet_indices, is_save=False):
    import trimesh
    pts = tet_points
    faces = []
    for i in range(0, len(tet_indices), 4):
        v0, v1, v2, v3 = tet_indices[i:i+4]
        faces.extend([
            [v0, v2, v1],
            [v1, v2, v3],
            [v0, v1, v3],
            [v0, v3, v2]
        ])
    msh = trimesh.Trimesh(vertices=pts, faces=faces)
    trimesh.Scene([msh]).show()
    
    if is_save:
        msh.export('tet_mesh_visualize.glb')

class MeshConverter(AssetConverterBase):
    """Converter for a mesh file in OBJ / STL / FBX format to a USD file.

    This class wraps around the `omni.kit.asset_converter`_ extension to provide a lazy implementation
    for mesh to USD conversion. It stores the output USD file in an instanceable format since that is
    what is typically used in all learning related applications.

    To make the asset instanceable, we must follow a certain structure dictated by how USD scene-graph
    instancing and physics work. The rigid body component must be added to each instance and not the
    referenced asset (i.e. the prototype prim itself). This is because the rigid body component defines
    properties that are specific to each instance and cannot be shared under the referenced asset. For
    more information, please check the `documentation <https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/rigid-bodies.html#instancing-rigid-bodies>`_.

    Due to the above, we follow the following structure:

    * ``{prim_path}`` - The root prim that is an Xform with the rigid body and mass APIs if configured.
    * ``{prim_path}/geometry`` - The prim that contains the mesh and optionally the materials if configured.
      If instancing is enabled, this prim will be an instanceable reference to the prototype prim.

    .. _omni.kit.asset_converter: https://docs.omniverse.nvidia.com/extensions/latest/ext_asset-converter.html

    .. caution::
        When converting STL files, Z-up convention is assumed, even though this is not the default for many CAD
        export programs. Asset orientation convention can either be modified directly in the CAD program's export
        process or an offset can be added within the config in Isaac Lab.

    """

    cfg: MeshConverterCfg
    """The configuration instance for mesh to USD conversion."""

    def __init__(self, cfg: MeshConverterCfg):
        """Initializes the class.

        Args:
            cfg: The configuration instance for mesh to USD conversion.
        """
        super().__init__(cfg=cfg)
    
    @staticmethod
    def set_attr(prim:Usd.Prim, attr_name:str, attr_type, attr_value):
        prim.CreateAttribute(attr_name, attr_type).Set(attr_value)

    """
    Implementation specific methods.
    """

    def _convert_asset(self, cfg: MeshConverterCfg):
        """Generate USD from OBJ, STL or FBX.

        The USD file has Y-up axis and is scaled to meters.
        The asset hierarchy is arranged as follows:

        .. code-block:: none
            mesh_file_basename (default prim)
                |- /geometry/Looks
                |- /geometry/mesh

        Args:
            cfg: The configuration for conversion of mesh to USD.

        Raises:
            RuntimeError: If the conversion using the Omniverse asset converter fails.
        """
        # resolve mesh name and format
        mesh_file_basename, mesh_file_format = os.path.basename(cfg.asset_path).split(".")
        mesh_file_format = mesh_file_format.lower()

        # Check if mesh_file_basename is a valid USD identifier
        if not Tf.IsValidIdentifier(mesh_file_basename):
            # Correct the name to a valid identifier and update the basename
            mesh_file_basename_original = mesh_file_basename
            mesh_file_basename = Tf.MakeValidIdentifier(mesh_file_basename)
            omni.log.warn(
                f"Input file name '{mesh_file_basename_original}' is an invalid identifier for the mesh prim path."
                f" Renaming it to '{mesh_file_basename}' for the conversion."
            )

        # Convert USD
        asyncio.get_event_loop().run_until_complete(
            self._convert_mesh_to_usd(in_file=cfg.asset_path, out_file=self.usd_path)
        )
        # Create a new stage, set Z up and meters per unit
        temp_stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(temp_stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(temp_stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(temp_stage, 1.0)
        # Add mesh to stage
        base_prim = temp_stage.DefinePrim(f"/{mesh_file_basename}", "Xform")
        # prim = temp_stage.DefinePrim(f"/{mesh_file_basename}/geometry", "Xform")
        # prim.GetReferences().AddReference(self.usd_path)
        base_prim.GetReferences().AddReference(self.usd_path)
        temp_stage.SetDefaultPrim(base_prim)
        temp_stage.Export(self.usd_path)

        # Open converted USD stage
        stage = Usd.Stage.Open(self.usd_path)
        # Need to reload the stage to get the new prim structure, otherwise it can be taken from the cache
        stage.Reload()
        # Add USD to stage cache
        stage_id = UsdUtils.StageCache.Get().Insert(stage)
        # Get the default prim (which is the root prim) -- "/{mesh_file_basename}"
        xform_prim = stage.GetDefaultPrim()
        geom_prim = stage.GetPrimAtPath(f"/{mesh_file_basename}")
        # Move all meshes to underneath new Xform
        for child_mesh_prim in geom_prim.GetChildren():
            if child_mesh_prim.GetTypeName() == "Mesh":
                # Apply collider properties to mesh
                if cfg.collision_props is not None:
                    # -- Collision approximation to mesh
                    # TODO: Move this to a new Schema: https://github.com/isaac-orbit/IsaacLab/issues/163
                    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(child_mesh_prim)
                    mesh_collision_api.GetApproximationAttr().Set(cfg.collision_approximation)
                    # -- Collider properties such as offset, scale, etc.
                    schemas.define_collision_properties(
                        prim_path=child_mesh_prim.GetPath(), cfg=cfg.collision_props, stage=stage
                    )
        # Delete the old Xform and make the new Xform the default prim
        stage.SetDefaultPrim(xform_prim)
        # Apply default Xform rotation to mesh -> enable to set rotation and scale
        omni.kit.commands.execute(
            "CreateDefaultXformOnPrimCommand",
            prim_path=xform_prim.GetPath(),
            **{"stage": stage},
        )

        # Apply translation, rotation, and scale to the Xform
        geom_xform = UsdGeom.Xform(geom_prim)
        geom_xform.ClearXformOpOrder()

        # Remove any existing rotation attributes
        rotate_attr = geom_prim.GetAttribute("xformOp:rotateXYZ")
        if rotate_attr:
            geom_prim.RemoveProperty(rotate_attr.GetName())

        # translation
        translate_op = geom_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(*cfg.translation))
        # rotation
        orient_op = geom_xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        orient_op.Set(Gf.Quatd(*cfg.rotation))
        # scale
        scale_op = geom_xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
        scale_op.Set(Gf.Vec3d(*cfg.scale))

        for child_mesh_prim in geom_prim.GetChildren():
            if child_mesh_prim.GetTypeName() != "Mesh":
                continue
            # add tet information
            usd_mesh = UsdGeom.Mesh(child_mesh_prim)
            tet_points, tet_indices, surf_points, tet_surf_indices = self.gen_tet(usd_mesh, backend='tetgen')
            print('total tet points:', len(tet_points), ' total tets:', len(tet_indices) // 4)
            self.set_attr(
                child_mesh_prim, 'tet_points',
                Sdf.ValueTypeNames.Float3Array, tet_points
            )
            self.set_attr(
                child_mesh_prim, 'tet_indices',
                Sdf.ValueTypeNames.UIntArray, tet_indices
            )
            self.set_attr(
                child_mesh_prim, 'tet_surf_points',
                Sdf.ValueTypeNames.Float3Array, surf_points
            )
            self.set_attr(
                child_mesh_prim, 'tet_surf_indices',
                Sdf.ValueTypeNames.UIntArray, tet_surf_indices
            )
            
            global args_cli
            if args_cli.show:
                visualize_tet(tet_points, tet_indices)

        # Handle instanceable
        # Create a new Xform prim that will be the prototype prim
        if cfg.make_instanceable:
            # Export Xform to a file so we can reference it from all instances
            export_prim_to_file(
                path=os.path.join(self.usd_dir, self.usd_instanceable_meshes_path),
                source_prim_path=geom_prim.GetPath(),
                stage=stage,
            )
            # Delete the original prim that will now be a reference
            geom_prim_path = geom_prim.GetPath().pathString
            omni.kit.commands.execute("DeletePrims", paths=[geom_prim_path], stage=stage)
            # Update references to exported Xform and make it instanceable
            geom_undef_prim = stage.DefinePrim(geom_prim_path)
            geom_undef_prim.GetReferences().AddReference(self.usd_instanceable_meshes_path, primPath=geom_prim_path)
            geom_undef_prim.SetInstanceable(True)

        # Apply mass and rigid body properties after everything else
        # Properties are applied to the top level prim to avoid the case where all instances of this
        #   asset unintentionally share the same rigid body properties
        # apply mass properties
        if cfg.mass_props is not None:
            schemas.define_mass_properties(prim_path=xform_prim.GetPath(), cfg=cfg.mass_props, stage=stage)
        # apply rigid body properties
        if cfg.rigid_props is not None:
            schemas.define_rigid_body_properties(prim_path=xform_prim.GetPath(), cfg=cfg.rigid_props, stage=stage)

        # Save changes to USD stage
        stage.Save()
        if stage_id is not None:
            UsdUtils.StageCache.Get().Erase(stage_id)
    
    def gen_tet(self, prim:UsdGeom.Mesh, backend='tetgen'):
        if backend == 'tetgen':
            import tetgen
            import pymeshfix
            import pyvista as pv

            points = np.array(prim.GetPointsAttr().Get())
            triangles = np.array(deformableUtils.triangulate_mesh(prim))
            
            import trimesh
            msh = trimesh.Trimesh(vertices=points, faces=triangles.reshape(-1, 3))
            msh.merge_vertices(digits_vertex=8)
            msh.update_faces(msh.unique_faces())
            msh.update_faces(msh.nondegenerate_faces())
            
            v_clean, f_clean = pymeshfix.clean_from_arrays(
                msh.vertices,
                msh.faces.astype(np.int32),
                joincomp=False,
                remove_smallest_components=False
            )
            
            # points, triangles = msh.vertices, msh.faces
            points, triangles = v_clean, f_clean
            trimesh.Scene([trimesh.Trimesh(vertices=points, faces=triangles.reshape(-1, 3))]).show()

            tg = tetgen.TetGen(points, triangles)
            tg.tetrahedralize()

            grid = tg.grid
            tet_points = grid.points          # (N, 3) float array
            cells = grid.cells_dict[10]       # 所有 type=10 (tetrahedron) 的单元
            tet_indices = cells.flatten().tolist()

            surface_polydata: pv.PolyData = grid.extract_surface(
                pass_pointid=False,
                pass_cellid=False,
                nonlinear_subdivision=1,
                progress_bar=False
            )
            faces = surface_polydata.faces
            surf_faces = faces.reshape(-1, 4)[:, 1:4]   # 取每行的后三个顶点索引
            surf_indices = surf_faces.flatten().tolist()
            surf_points = np.array(surface_polydata.points).tolist()
            return tet_points, tet_indices, surf_points, surf_indices
        else:
            mesh_gen = MeshGenerator(config=TetMeshCfg(
                stop_quality=6,
                max_its=100,
                # edge_length_r=0.01,
                edge_length_r=0.02,
                epsilon_r=0.01
            ))
            return mesh_gen.generate_tet_mesh_for_prim(
                prim
            )

    """
    Helper methods.
    """

    @staticmethod
    async def _convert_mesh_to_usd(in_file: str, out_file: str, load_materials: bool = True) -> bool:
        """Convert mesh from supported file types to USD.

        This function uses the Omniverse Asset Converter extension to convert a mesh file to USD.
        It is an asynchronous function and should be called using `asyncio.get_event_loop().run_until_complete()`.

        The converted asset is stored in the USD format in the specified output file.
        The USD file has Y-up axis and is scaled to cm.

        Args:
            in_file: The file to convert.
            out_file: The path to store the output file.
            load_materials: Set to True to enable attaching materials defined in the input file
                to the generated USD mesh. Defaults to True.

        Returns:
            True if the conversion succeeds.
        """
        enable_extension("omni.kit.asset_converter")

        import omni.kit.asset_converter
        import omni.usd

        # Create converter context
        converter_context = omni.kit.asset_converter.AssetConverterContext()
        # Set up converter settings
        # Don't import/export materials
        converter_context.ignore_materials = not load_materials
        converter_context.ignore_animations = True
        converter_context.ignore_camera = True
        converter_context.ignore_light = True
        # Merge all meshes into one
        converter_context.merge_all_meshes = True
        # Sets world units to meters, this will also scale asset if it's centimeters model.
        # This does not work right now :(, so we need to scale the mesh manually
        converter_context.use_meter_as_world_unit = True
        converter_context.baking_scales = True
        # Uses double precision for all transform ops.
        converter_context.use_double_precision_to_usd_transform_op = True

        # Create converter task
        instance = omni.kit.asset_converter.get_instance()
        task = instance.create_converter_task(in_file, out_file, None, converter_context)
        # Start conversion task and wait for it to finish
        success = await task.wait_until_finished()
        if not success:
            raise RuntimeError(f"Failed to convert {in_file} to USD. Error: {task.get_error_message()}")
        return success


def convert_mesh(input_path:Path, output_path:Path,
                 make_instanceable:bool=False,
                 collision_approximation:str="convexDecomposition",
                 mass:float=None):
    input_path = input_path.absolute()
    output_path = output_path.absolute()
    
    if not check_file_path(str(input_path)):
        raise ValueError(f"Invalid mesh file path: {input_path}")
    
    if mass is not None:
        mass_props = schemas_cfg.MassPropertiesCfg(mass=mass)
        rigid_props = schemas_cfg.RigidBodyPropertiesCfg()
    else:
        mass_props = None
        rigid_props = None
    
    collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)
    mesh_converter_cfg = MeshConverterCfg(
        mass_props=mass_props,
        rigid_props=rigid_props,
        collision_props=collision_props,
        asset_path=str(input_path),
        force_usd_conversion=True,
        usd_dir=str(output_path.parent),
        usd_file_name=output_path.name,
        make_instanceable=make_instanceable,
        collision_approximation=collision_approximation,
    )
    mesh_converter = MeshConverter(mesh_converter_cfg)
    return Path(mesh_converter.usd_path)

def main():
    global args_cli    
    input_path = Path(args_cli.input)
    output_path = Path(args_cli.output)

    process_list = []
    if input_path.is_dir():
        if output_path.exists() and not output_path.is_dir():
            output_path = output_path.parent
        output_path.mkdir(parents=True, exist_ok=True)
        for file in input_path.iterdir():
            if file.suffix.lower() in [".obj", ".stl", ".fbx", ".glb"]:
                out_file = output_path / (file.stem + ".usd")
                process_list.append((file, out_file))
    else:
        if output_path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
            out_file = output_path / (input_path.stem + ".usd")
        else:
            out_file = output_path
        process_list.append((input_path, out_file))
    
    total_files = len(process_list)
    print(f'{total_files} files to process:')
    for idx, (i, o) in enumerate(process_list):
        print(f"[{idx + 1}/{total_files}] Converting mesh {i} to USD file {o}")
        usd_path = convert_mesh(i, o,
                                make_instanceable=args_cli.make_instanceable,
                                collision_approximation=args_cli.collision_approximation,
                                mass=args_cli.mass)
        print(f"[{idx + 1}/{total_files}] Converted USD file saved at: {usd_path}")

def visualize(name):
    usd_path = Path(f'assets/objects/{name}.usd')
    stage = Usd.Stage.Open(str(usd_path))
    prim = stage.GetPrimAtPath(f'/{name}/mesh')
    tet_points = prim.GetAttribute('tet_points').Get()
    tet_indices = prim.GetAttribute('tet_indices').Get()
    visualize_tet(tet_points, tet_indices, is_save=True)

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
