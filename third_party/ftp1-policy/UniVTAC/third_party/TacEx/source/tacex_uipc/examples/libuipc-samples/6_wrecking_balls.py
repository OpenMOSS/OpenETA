"""Showcase on how to use libuipc with Isaac Sim/Lab.

This example corresponds to
https://github.com/spiriMirror/libuipc-samples/blob/main/python/6_wrecking_balls/main.py

Notes:
The order of the object creation needs to be equal to the geometry creation order.
Otherwise the Isaac rendering is wrong.

Reason:
We map uipc vertex offsets to USD prims for the rendering.
The "mapping" happens through a list: element 1 of the list = offset for prim with ID 1, element 2 = offset for prim with ID 2, etc.
This is why the order has to equal.

e.g. if the original example does this:

cube_obj = scene.objects().create('cubes')
ball_obj = scene.objects().create('balls')
link_obj = scene.objects().create('links')

In this case we first need to create cube geometries,
then the ball geometries and in the end the link geometries.

Since the scene json has the order link.msh, ball.msh, cube.msh (which results in us creating geometries in this order)
we need to first create the scene objects for the links, then the balls and in the end the cube.
So, we do:

link_obj = scene.objects().create('links')
ball_obj = scene.objects().create('balls')
cube_obj = scene.objects().create('cubes')

instead.

(Key difference to the libuipc sample)

In the IsaacLab workflow this shouldn't be an issue, since we always create the geometries in the fitting order,
i.e. corresponding to the object creation.

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

import json
import numpy as np
import pathlib

import omni.usd
import uipc
from pxr import UsdGeom
from uipc import AngleAxis, Quaternion, Transform, Vector3, view
from uipc.constitution import AffineBodyConstitution
from uipc.geometry import (
    SimplicialComplex,
    SimplicialComplexIO,
    flip_inward_triangles,
    label_surface,
    label_triangle_orient,
)
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
    tetmesh_dir = str(pathlib.Path(__file__).parent.resolve() / "tet_meshes")

    def process_surface(sc: SimplicialComplex):
        label_surface(sc)
        label_triangle_orient(sc)
        sc = flip_inward_triangles(sc)
        return sc

    abd = AffineBodyConstitution()
    scene.contact_tabular().default_model(0.02, 10 * GPa)
    default_contact = scene.contact_tabular().default_element()

    io = SimplicialComplexIO()

    with open(f"{str(pathlib.Path(__file__).parent.resolve())}/6_wrecking_ball.json") as json_file:
        wrecking_ball_scene = json.load(json_file)

    cube = io.read(f"{tetmesh_dir}/cube.msh")
    cube = process_surface(cube)
    ball = io.read(f"{tetmesh_dir}/ball.msh")
    ball = process_surface(ball)
    link = io.read(f"{tetmesh_dir}/link.msh")
    link = process_surface(link)

    link_obj = scene.objects().create("links")
    ball_obj = scene.objects().create("balls")
    cube_obj = scene.objects().create("cubes")

    abd.apply_to(cube, 10 * MPa)
    default_contact.apply_to(cube)

    abd.apply_to(ball, 10 * MPa)
    default_contact.apply_to(ball)

    abd.apply_to(link, 10 * MPa)
    default_contact.apply_to(link)

    def build_mesh(json, obj: uipc.core.Object, mesh: SimplicialComplex):
        t = Transform.Identity()
        position = Vector3.Zero()
        if "position" in json:
            position[0] = json["position"][0]
            position[1] = json["position"][1]
            position[2] = json["position"][2]
            t.translate(position)

        Q = Quaternion.Identity()
        if "rotation" in json:
            rotation = Vector3.Zero()
            rotation[0] = json["rotation"][0]
            rotation[1] = json["rotation"][1]
            rotation[2] = json["rotation"][2]
            rotation *= np.pi / 180
            Q = (
                AngleAxis(rotation[2][0], Vector3.UnitZ())
                * AngleAxis(rotation[1][0], Vector3.UnitY())
                * AngleAxis(rotation[0][0], Vector3.UnitX())
            )
            t.rotate(Q)

        is_fixed = json.get("is_dof_fixed")

        this_mesh = mesh.copy()
        view(this_mesh.transforms())[0] = t.matrix()

        is_fixed_attr = this_mesh.instances().find("is_fixed")
        view(is_fixed_attr)[0] = is_fixed

        obj.geometries().create(this_mesh)

    for obj in wrecking_ball_scene:
        if obj["mesh"] == "link.msh":
            build_mesh(obj, link_obj, link)
        elif obj["mesh"] == "ball.msh":
            build_mesh(obj, ball_obj, ball)
        elif obj["mesh"] == "cube.msh":
            build_mesh(obj, cube_obj, cube)


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
        workspace=str(pathlib.Path(__file__).parent.resolve() / "dumps" / "6_wrecking_balls"),
        dt=0.033,
        gravity=[0.0, -9.8, 0.0],
        ground_normal=[0, 1, 0],
        ground_height=-1.0,
        # logger_level="Info",
        contact=UipcSimCfg.Contact(default_friction_ratio=0.5, default_contact_resistance=1.0, d_hat=0.01),
        line_search=UipcSimCfg.LineSearch(max_iter=8),
        newton=UipcSimCfg.Newton(velocity_tol=0.2),
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

    # to save/ load the simulation frames
    recover_sim = False

    # Simulate physics
    while simulation_app.is_running():
        # perform Isaac rendering
        sim.render()

        if sim.is_playing():
            if step == 500:
                break
            print("")
            print("====================================================================================")
            print("====================================================================================")
            print("Step number ", step)
            if recover_sim:
                if uipc_sim.world.recover(uipc_sim.world.frame() + 1):
                    uipc_sim.world.retrieve()
                    print("Replaying frame ", uipc_sim.world.frame() + 1)
                else:
                    with Timer("[INFO]: Time taken for uipc sim step", name="uipc_step"):
                        uipc_sim.step()
                    total_uipc_sim_time += Timer.get_timer_info("uipc_step")
                    uipc_sim.world.dump()
            else:
                with Timer("[INFO]: Time taken for uipc sim step", name="uipc_step"):
                    uipc_sim.step()
                total_uipc_sim_time += Timer.get_timer_info("uipc_step")

            with Timer("[INFO]: Time taken for rendering", name="render_update"):
                uipc_sim.update_render_meshes()
                sim.render()
            total_uipc_render_time += Timer.get_timer_info("render_update")
            # get time reports
            uipc_sim.get_sim_time_report()

            step += 1


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
