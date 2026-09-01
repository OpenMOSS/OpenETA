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
import pathlib

import omni.usd
from pxr import UsdGeom
from uipc import AngleAxis, Transform, Vector3, builtin, view
from uipc.constitution import AffineBodyConstitution
from uipc.core import ContactElement
from uipc.geometry import (
    SimplicialComplexIO,
    label_surface,
)

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
        translation=[0, -2, 0],
        orientation=[0.7071068, -0.7071068, 0, 0],
    )

    # spawn distant light
    cfg_light_dome = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_dome.func("/World/lightDome", cfg_light_dome, translation=(1, 10, 0))


def setup_libuipc_scene(scene):
    trimesh_path = str(pathlib.Path(__file__).parent.resolve() / "trimesh")

    abd = AffineBodyConstitution()
    scene.constitution_tabular().insert(abd)
    contact_tabular = scene.contact_tabular()
    contact_tabular.default_model(0.5, 1e9)
    default_element = scene.contact_tabular().default_element()

    io = SimplicialComplexIO()
    N = 8
    friction_rate_step = 1.0 / (N - 1)
    contact_elements: list[ContactElement] = []

    for i in range(N):
        friction_rate = i * friction_rate_step
        e = contact_tabular.create(f"element_{i}")
        contact_tabular.insert(e, default_element, friction_rate=friction_rate, resistance=1e9)
        contact_elements.append(e)

    pre_transform = Transform.Identity()
    pre_transform.scale(0.3)
    io = SimplicialComplexIO(pre_transform)
    cube_mesh = io.read(f"{trimesh_path}/cube.obj")
    label_surface(cube_mesh)

    abd.apply_to(cube_mesh, 1e8)
    step = 0.5
    start_x = -step * (N - 1) / 2

    # create cubes
    cube_object = scene.objects().create("cubes")
    for i in range(N):
        cube = cube_mesh.copy()
        contact_elements[i].apply_to(cube)
        t = Transform.Identity()
        t.translate(Vector3.Values([start_x + i * step, 1, -0.7]))
        t.rotate(AngleAxis(30 * np.pi / 180, Vector3.UnitX()))
        view(cube.transforms())[0] = t.matrix()
        cube_object.geometries().create(cube)

    # create ramp
    ramp_object = scene.objects().create("ramp")
    pre_transform = Transform.Identity()
    pre_transform.scale(Vector3.Values([0.5 * N, 0.1, 5]))
    io = SimplicialComplexIO(pre_transform)
    ramp_mesh = io.read(f"{trimesh_path}/cube.obj")
    label_surface(ramp_mesh)
    default_element.apply_to(ramp_mesh)
    abd.apply_to(ramp_mesh, 1e8)

    # rotate by 30 degrees
    t = Transform.Identity()
    t.rotate(AngleAxis(30 * np.pi / 180, Vector3.UnitX()))
    view(ramp_mesh.transforms())[0] = t.matrix()

    is_fixed = ramp_mesh.instances().find(builtin.is_fixed)
    view(is_fixed).fill(1)
    ramp_object.geometries().create(ramp_mesh)


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
        dt=0.01,
        gravity=[0.0, -9.8, 0.0],
        ground_normal=[0, 1, 0],
        ground_height=-2.0,
        # logger_level="Info",
        contact=UipcSimCfg.Contact(default_friction_ratio=0.5, default_contact_resistance=1e9, d_hat=0.01),
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
