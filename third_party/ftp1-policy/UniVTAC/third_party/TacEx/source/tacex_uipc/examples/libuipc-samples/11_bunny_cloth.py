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

import pathlib

import omni.usd
from pxr import UsdGeom
from uipc import Transform, Vector3, builtin, view
from uipc.constitution import AffineBodyConstitution, DiscreteShellBending, ElasticModuli, NeoHookeanShell
from uipc.geometry import SimplicialComplexIO, flip_inward_triangles, label_surface, label_triangle_orient
from uipc.unit import MPa, kPa

import isaaclab.sim as sim_utils
from isaaclab.utils.timer import Timer

from tacex_uipc.sim import UipcSim, UipcSimCfg


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
    tetmesh_path = str(pathlib.Path(__file__).parent.resolve() / "tet_meshes")

    # setup the scene
    cloth = scene.objects().create("cloth")
    t = Transform.Identity()
    t.scale(2.0)
    io = SimplicialComplexIO(t)
    cloth_mesh = io.read(f"{trimesh_path}/grid20x20.obj")
    label_surface(cloth_mesh)
    nks = NeoHookeanShell()
    dsb = DiscreteShellBending()
    moduli = ElasticModuli.youngs_poisson(10 * kPa, 0.499)
    nks.apply_to(cloth_mesh, moduli=moduli, mass_density=200, thickness=0.001)
    dsb.apply_to(cloth_mesh, E=10.0)
    view(cloth_mesh.positions())[:] += 1.0
    cloth.geometries().create(cloth_mesh)

    bunny = scene.objects().create("bunny")
    t = Transform.Identity()
    t.translate(Vector3.UnitX() + Vector3.UnitZ())
    io = SimplicialComplexIO(t)
    bunny_mesh = io.read(f"{tetmesh_path}/bunny0.msh")
    label_surface(bunny_mesh)
    label_triangle_orient(bunny_mesh)
    bunny_mesh = flip_inward_triangles(bunny_mesh)
    abd = AffineBodyConstitution()
    abd.apply_to(bunny_mesh, 100 * MPa)
    is_fixed = bunny_mesh.instances().find(builtin.is_fixed)
    view(is_fixed)[:] = 1

    bunny.geometries().create(bunny_mesh)


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
        ground_height=-1.0,
        # logger_level="Info",
        contact=UipcSimCfg.Contact(default_friction_ratio=0.5, default_contact_resistance=1.0, d_hat=0.01),
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
