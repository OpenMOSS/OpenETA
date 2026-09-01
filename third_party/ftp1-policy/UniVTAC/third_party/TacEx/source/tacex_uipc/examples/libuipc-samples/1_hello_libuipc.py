"""Showcase on how to use libuipc with Isaac Sim/Lab.

This example corresponds to
https://github.com/spiriMirror/libuipc-samples/blob/main/python/1_hello_libuipc/main.py


"""

"""Launch Isaac Sim Simulator first."""
import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Showcase on how to use libuipc with Isaac Sim/Lab.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np

import omni.usd
import uipc
from pxr import UsdGeom
from uipc.constitution import AffineBodyConstitution
from uipc.geometry import flip_inward_triangles, label_surface, label_triangle_orient, tetmesh
from uipc.unit import GPa, MPa

import isaaclab.sim as sim_utils
from isaaclab.utils.timer import Timer

from tacex_uipc import UipcSim, UipcSimCfg


def setup_base_scene(sim: sim_utils.SimulationContext):
    """To make the scene pretty."""
    # set upAxis to Y to match libuipc-samples
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    # Design scene by spawning assets
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func(
        prim_path="/World/defaultGroundPlane",
        cfg=cfg_ground,
        translation=[0, -1, 0],
        orientation=[0.7071068, -0.7071068, 0, 0],
    )

    # spawn distant light
    cfg_light_dome = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_dome.func("/World/lightDome", cfg_light_dome, translation=(1, 10, 0))


def setup_libuipc_scene(scene):
    # create constitution and contact model
    abd = AffineBodyConstitution()

    # friction ratio and contact resistance
    scene.contact_tabular().default_model(0.5, 1.0 * GPa)
    default_element = scene.contact_tabular().default_element()

    # create a regular tetrahedron
    Vs = np.array([[0, 1, 0], [0, 0, 1], [-np.sqrt(3) / 2, 0, -0.5], [np.sqrt(3) / 2, 0, -0.5]])
    Ts = np.array([[0, 1, 2, 3]])

    # setup a base mesh to reduce the later work
    base_mesh = tetmesh(Vs, Ts)
    # apply the constitution and contact model to the base mesh
    abd.apply_to(base_mesh, 100 * MPa)
    # apply the default contact model to the base mesh
    default_element.apply_to(base_mesh)

    # label the surface, enable the contact
    label_surface(base_mesh)
    # label the triangle orientation to export the correct surface mesh
    label_triangle_orient(base_mesh)
    # flip the triangles inward for better rendering
    base_mesh = flip_inward_triangles(base_mesh)

    mesh1 = base_mesh.copy()
    pos_view = uipc.view(mesh1.positions())
    # move the mesh up for 1 unit
    pos_view += uipc.Vector3.UnitY() * 1.5

    mesh2 = base_mesh.copy()
    is_fixed = mesh2.instances().find(uipc.builtin.is_fixed)
    is_fixed_view = uipc.view(is_fixed)
    is_fixed_view[:] = 1  # make the second mesh static

    # create objects
    object1 = scene.objects().create("upper_tet")
    object1.geometries().create(mesh1)

    object2 = scene.objects().create("lower_tet")
    object2.geometries().create(mesh2)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(
        dt=1 / 60,
        gravity=[0.0, -9.8, 0.0],
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    setup_base_scene(sim)

    # Initialize uipc sim
    uipc_cfg = UipcSimCfg(
        dt=0.02,
        gravity=[0.0, -9.8, 0.0],
        ground_normal=[0, 1, 0],
        ground_height=-1.0,
        # logger_level="Info",
        contact=UipcSimCfg.Contact(
            default_friction_ratio=0.5,
            default_contact_resistance=1.0,
        ),
    )
    uipc_sim = UipcSim(uipc_cfg)

    setup_libuipc_scene(uipc_sim.scene)

    # init liubipc world etc.
    uipc_sim.setup_sim()
    uipc_sim.init_libuipc_scene_rendering()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    step = 0

    total_uipc_sim_time = 0.0
    total_uipc_render_time = 0.0

    # Simulate physics
    while simulation_app.is_running():
        # perform Isaac rendering
        sim.render()

        if sim.is_playing():
            print("")
            print("====================================================================================")
            print("====================================================================================")
            print("Step number ", step)
            with Timer("[INFO]: Time taken for uipc sim step", name="uipc_step"):
                uipc_sim.step()
                # uipc_sim.save_current_world_state()
            with Timer("[INFO]: Time taken for rendering", name="render_update"):
                uipc_sim.update_render_meshes()
                sim.render()

            # get time reports
            uipc_sim.get_sim_time_report()
            total_uipc_sim_time += Timer.get_timer_info("uipc_step")
            total_uipc_render_time += Timer.get_timer_info("render_update")

            step += 1


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
