"""Showcase on how to use libuipc with Isaac Sim/Lab.

This example corresponds to
https://github.com/spiriMirror/libuipc-samples/blob/main/python/5_walking_cube_moving_board/main.py

Note:
Before the simulation is started, the scene looks weird/wrong.
This is because libuipc sets transformations, but we create USD prims solely
with the mesh data (i.e. mesh vertex positions and topology).

The simulation is gonna look correct, once you run the simulation.

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
from uipc import Matrix4x4, Transform, Vector2, Vector3, builtin, view
from uipc.constitution import AffineBodyConstitution, RotatingMotor, SoftTransformConstraint
from uipc.core import Animation
from uipc.geometry import SimplicialComplex, SimplicialComplexIO, SimplicialComplexSlot, label_surface

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
    tetmesh_path = str(pathlib.Path(__file__).parent.resolve() / "tet_meshes")

    # friction ratio and contact resistance
    scene.contact_tabular().default_model(0.2, 1e9)
    default_element = scene.contact_tabular().default_element()

    # create constituiton
    abd = AffineBodyConstitution()
    # create constraint
    rm = RotatingMotor()
    stc = SoftTransformConstraint()

    def process_surface(sc: SimplicialComplex):
        label_surface(sc)
        return sc

    io = SimplicialComplexIO()
    cube_mesh = io.read(f"{trimesh_path}/cube.obj")
    cube_mesh = process_surface(cube_mesh)

    # move the cube up for 2.5 meters
    trans_view = view(cube_mesh.transforms())
    t = Transform.Identity()
    t.translate(Vector3.UnitY() * 2.5)
    trans_view[0] = t.matrix()

    abd.apply_to(cube_mesh, 1e8)  # 100 MPa
    default_element.apply_to(cube_mesh)
    # constraint the rotation
    rm.apply_to(cube_mesh, 100, motor_rot_vel=np.pi)
    cube_object = scene.objects().create("cube")
    cube_object.geometries().create(cube_mesh)

    pre_transform = Transform.Identity()
    pre_transform.scale(Vector3.Values([3, 0.1, 6]))

    io = SimplicialComplexIO(pre_transform)
    ground_mesh = io.read(f"{tetmesh_path}/cube.msh")
    ground_mesh = process_surface(ground_mesh)
    ground_mesh.instances().resize(2)

    abd.apply_to(ground_mesh, 1e7)  # 10 MPa
    default_element.apply_to(ground_mesh)
    stc.apply_to(ground_mesh, Vector2.Values([100.0, 100.0]))
    is_fixed = ground_mesh.instances().find(builtin.is_fixed)
    is_fixed_view = view(is_fixed)
    is_fixed_view[0] = 1  # fix the lower board
    is_fixed_view[1] = 0

    trans_view = view(ground_mesh.transforms())
    t = Transform.Identity()
    t.translate(Vector3.UnitZ() * 2)
    trans_view[0] = t.matrix()
    t.translate(Vector3.UnitZ() * -2.5 + Vector3.UnitY() * 1)
    trans_view[1] = t.matrix()

    ground_object = scene.objects().create("ground")
    ground_object.geometries().create(ground_mesh)

    animator = scene.animator()

    def cube_animation(info: Animation.UpdateInfo):
        geo_slots = info.geo_slots()
        geo_slot: SimplicialComplexSlot = geo_slots[0]
        geo = geo_slot.geometry()
        is_constrained = geo.instances().find(builtin.is_constrained)
        view(is_constrained)[0] = 1
        RotatingMotor.animate(geo, info.dt())

    def ground_animation(info: Animation.UpdateInfo):
        geo_slot: SimplicialComplexSlot = info.geo_slots()[0]
        rest_geo_slot: SimplicialComplexSlot = info.rest_geo_slots()[0]
        geo = geo_slot.geometry()
        rest_geo = rest_geo_slot.geometry()

        is_constrained = geo.instances().find(builtin.is_constrained)
        view(is_constrained)[1] = 1

        current_t = info.dt() * info.frame()
        angular_velocity = np.pi  # 180 degree per second
        theta = angular_velocity * current_t

        T: Matrix4x4 = rest_geo.transforms().view()[1]
        Y = np.sin(theta) * 0.4
        T: Transform = Transform(T)
        p = T.translation()
        p[1] += Y
        T = Transform.Identity()
        T.translate(p)

        aim_trans = geo.instances().find(builtin.aim_transform)
        view(aim_trans)[1] = T.matrix()

    animator.insert(cube_object, cube_animation)
    animator.insert(ground_object, ground_animation)


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
