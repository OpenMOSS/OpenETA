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
from uipc import Animation, Transform, Vector3, builtin, view
from uipc.constitution import AffineBodyConstitution, RotatingMotor
from uipc.geometry import (
    SimplicialComplexIO,
    SimplicialComplexSlot,
    label_surface,
)
from uipc.unit import MPa

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
    trimesh_path = str(pathlib.Path(__file__).parent.resolve() / "trimesh")

    t = Transform.Identity()
    t.scale(0.05)
    io = SimplicialComplexIO()
    abd = AffineBodyConstitution()
    rm = RotatingMotor()
    scene.contact_tabular().default_model(0, 1e9)

    screw_obj = scene.objects().create("screw")
    screw_mesh = io.read(f"{trimesh_path}/screw-and-nut/screw-big-2.obj")
    label_surface(screw_mesh)
    abd.apply_to(screw_mesh, 100 * MPa)
    rm.apply_to(screw_mesh, 100, motor_axis=Vector3.UnitY(), motor_rot_vel=-np.pi)
    screw_obj.geometries().create(screw_mesh)

    def screw_animation(info: Animation.UpdateInfo):
        geo_slots = info.geo_slots()
        geo_slot: SimplicialComplexSlot = geo_slots[0]
        geo = geo_slot.geometry()
        is_constrained = geo.instances().find(builtin.is_constrained)
        view(is_constrained)[0] = 1
        RotatingMotor.animate(geo, info.dt())

    scene.animator().insert(screw_obj, screw_animation)

    nut_obj = scene.objects().create("nut")
    nut_mesh = io.read(f"{trimesh_path}/screw-and-nut/nut-big-2.obj")
    label_surface(nut_mesh)
    abd.apply_to(nut_mesh, 100 * MPa)
    is_fixed = nut_mesh.instances().find(builtin.is_fixed)
    view(is_fixed)[:] = 1
    nut_obj.geometries().create(nut_mesh)


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
        workspace=str(pathlib.Path(__file__).parent.resolve() / "dumps" / "8_screw_and_nut"),
        dt=0.005,
        gravity=[0.0, 0.0, 0.0],
        ground_normal=[0, 1, 0],
        ground_height=-1.0,
        # logger_level="Info",
        contact=UipcSimCfg.Contact(
            enable_friction=False,
            d_hat=0.02,
        ),
        newton=UipcSimCfg.Newton(
            velocity_tol=0.05,
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
                if step == 500:
                    break

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
