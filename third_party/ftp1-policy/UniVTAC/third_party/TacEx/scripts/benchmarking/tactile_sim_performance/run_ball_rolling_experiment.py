import sys
print(sys.path)
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Ball rolling experiment with a Franka, which is equipped with one GelSight Mini Sensor."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn. Default 1.")
parser.add_argument(
    "--env",
    type=str,
    default="physx_rigid",
    help="What type of env cfg should be used. Options: [physx_rigid, uipc, uipc_textured]. Defaults to physx_rigid",
)
# parser.add_argument("--track_sys", type=bool, default=True, help="Whether to track system utilization.")
parser.add_argument(
    "--debug_vis",
    default=False,
    action="store_true",
    help="Whether to render tactile images in the# append AppLauncher cli args. Default True.",
)
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import datetime
import json
import numpy as np
import platform
import psutil
import time
import torch
import traceback
from pathlib import Path

import carb
import pynvml
from envs.ball_rolling_physx_rigid import PhysXRigidEnv, PhysXRigidEnvCfg
from envs.ball_rolling_uipc import UipcEnv, UipcEnvCfg
from envs.ball_rolling_uipc_texture import UipcTexturedEnv, UipcTexturedEnvCfg

""" System diagnosis

-> adapted from benchmark_cameras.py script of IsaacLab
"""


def _get_utilization_percentages(reset: bool = False, max_values: list[float] = [0.0, 0.0, 0.0, 0.0]) -> list[float]:
    """Get the maximum CPU, RAM, GPU utilization (processing), and
    GPU memory usage percentages since the last time reset was true.

    """
    if reset:
        max_values[:] = [0, 0, 0, 0]  # Reset the max values

    # # CPU utilization
    # cpu_usage = psutil.cpu_percent(interval=0.1) # blocking slows down Isaac Sim a lot
    cpu_usage = psutil.cpu_percent(interval=None)
    max_values[0] = max(max_values[0], cpu_usage)

    # # RAM utilization
    memory_info = psutil.virtual_memory()
    ram_usage = memory_info.percent
    max_values[1] = max(max_values[1], ram_usage)

    # GPU utilization using pynvml
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # GPU Utilization
            gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_processing_utilization_percent = gpu_utilization.gpu  # GPU core utilization
            max_values[2] = max(max_values[2], gpu_processing_utilization_percent)

            # GPU Memory Usage
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_total = memory_info.total
            gpu_memory_used = memory_info.used
            gpu_memory_utilization_percent = (gpu_memory_used / gpu_memory_total) * 100
            max_values[3] = max(max_values[3], gpu_memory_utilization_percent)
    else:
        gpu_processing_utilization_percent = None
        gpu_memory_utilization_percent = None

    return max_values


def run_simulator(env):
    """Runs the simulation loop."""

    # for convenience, we directly turn on debug_vis
    if env.cfg.gsmini.debug_vis:
        for data_type in env.cfg.gsmini.data_types:
            env.gsmini._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)

    # For time measurements
    timestamp = f"{datetime.datetime.now():%Y-%m-%d-%H_%M_%S}"
    output_dir = Path(__file__).parent.resolve() / "logs" / type(env).__name__
    output_dir = str(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    file_name = output_dir + f"/envs_{env.num_envs}_{timestamp}.txt"

    frame_times_physics = []
    frame_times_tactile = []
    total_num_in_contact_frames = 0

    save_after_num_resets = 2  # = do pattern twice
    current_num_resets = 0

    # track GPU utilization using pynvml
    if torch.cuda.is_available():
        pynvml.nvmlInit()
    system_utilization_analytics = _get_utilization_percentages(reset=False)

    print(f"Starting simulation with {env.num_envs} envs")
    print("Number of steps till reset: ", len(env.pattern_offsets) * env.num_step_goal_change)

    total_sim_time = time.time()
    # Simulation loop
    while simulation_app.is_running():
        # reset at the beginning of the simulation and after doing pattern 2 times
        if (
            env.step_count % (len(env.pattern_offsets) * env.num_step_goal_change) == 0
        ):  # reset after 750 steps, cause every 50 steps we change action and pattern consists of 15 actions
            print("*" * 15)
            print(f"[INFO]: Env reset num {current_num_resets} out of {save_after_num_resets}.")
            env.reset()
            system_utilization_analytics = _get_utilization_percentages(reset=True)
            print(
                f"Total amount of 'in-contact' frames per env (ran pattern {current_num_resets} times):"
                f" {total_num_in_contact_frames}"
            )
            print(f"Total time: {time.time() - total_sim_time:8.4f}s")
            print(
                "Avg physics_sim time for one frame:   "
                f" {np.sum(np.array(frame_times_physics)) / total_num_in_contact_frames:8.4f}ms"
            )
            print(
                "Avg tactile_sim time for one frame:   "
                f" {np.sum(np.array(frame_times_tactile)) / total_num_in_contact_frames:8.4f}ms"
            )
            print(
                f"| CPU:{system_utilization_analytics[0]}% | "
                f"RAM:{system_utilization_analytics[1]}% | "
                f"GPU Compute:{system_utilization_analytics[2]}% | "
                f"GPU Memory: {system_utilization_analytics[3]:.2f}% |"
            )
            if current_num_resets == save_after_num_resets:
                print("Writing performance data into ", file_name)
                with open(file_name, "a+") as f:
                    f.write("-" * 20)
                    f.write("System Info")
                    f.write("-" * 20)
                    f.write("\n")
                    f.write("[CPU Info] \n")
                    f.write(f"Name: {platform.processor()} \n")
                    f.write(f"Physical cores: {psutil.cpu_count(logical=False)} \n")
                    f.write(f"Total cores: {psutil.cpu_count(logical=True)} \n")
                    f.write("\n")

                    # currently only works with one gpu -> #todo add for loop, for multi gpu?
                    f.write("[GPU Info] \n")
                    f.write(f"Name: {pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))} \n")
                    f.write(f"Driver: {pynvml.nvmlSystemGetDriverVersion()} \n")
                    f.write("\n")
                    f.write("-" * 20)
                    f.write("Performance data")
                    f.write("-" * 20)
                    f.write("\n")
                    f.write(f"Number envs: {env.num_envs} \n")
                    f.write(
                        f"Total amount of 'in-contact' frames per env (ran pattern {current_num_resets} times):"
                        f" {total_num_in_contact_frames}\n"
                    )
                    f.write(f"Total time: {time.time() - total_sim_time:8.4f}s \n")
                    f.write(
                        "Avg physics_sim time for one frame:   "
                        f" {np.sum(np.array(frame_times_physics)) / total_num_in_contact_frames:8.4f}ms \n"
                    )
                    f.write(
                        "Avg tactile_sim time for one frame:   "
                        f" {np.sum(np.array(frame_times_tactile)) / total_num_in_contact_frames:8.4f}ms \n"
                    )
                    f.write("\n")
                    f.write(
                        f"| CPU:{system_utilization_analytics[0]}% | "
                        f"RAM:{system_utilization_analytics[1]}% | "
                        f"GPU Compute:{system_utilization_analytics[2]}% | "
                        f"GPU Memory: {system_utilization_analytics[3]:.2f}% |"
                    )
                    f.write("\n\n")
                    f.write("-" * 20)
                    f.write("Sensor Config")
                    f.write("-" * 20)
                    f.write("\n")
                    f.write(json.dumps(env.cfg.gsmini.to_dict(), indent=2))
                    f.write("\n")
                print("*" * 15)
                break
            current_num_resets += 1
            print("*" * 15)

        # apply action to robot
        env._pre_physics_step(None)
        env._apply_action()
        env.scene.write_data_to_sim()

        physics_start = time.time()
        # perform physics step
        env.sim.step(render=False)
        physics_end = time.time()

        # render scene for cameras
        if env.uipc_sim is not None:
            env.uipc_sim.update_render_meshes()
        env.sim.render()

        # update scene buffers (i.e. data from rigid bodies, uipc bodies, sensors...)
        env.scene.update(dt=env.physics_dt)

        # update sensors again to measure tactile sim time separately
        tactile_sim_start = time.time()
        env.gsmini.update(dt=env.physics_dt, force_recompute=True)
        tactile_sim_end = time.time()

        print(f"Total time: {time.time() - total_sim_time:8.4f}s")
        print(f"Current env episode step: {env.step_count}/{(len(env.pattern_offsets) * env.num_step_goal_change)}")

        (contact_idx,) = torch.where(env.gsmini._indentation_depth > 0)
        print(f"Current number of 'in-contact' frames across all env (num={env.num_envs}): {contact_idx.shape[0]}")
        # - measure sim times, if sensor was in contact
        if contact_idx.shape[0] != 0:
            frame_times_physics.append(1000 * (physics_end - physics_start))
            frame_times_tactile.append(1000 * (tactile_sim_end - tactile_sim_start))
            total_num_in_contact_frames += contact_idx.shape[0]
            print(
                "Avg physics_sim time for current step per env:   "
                f" {frame_times_physics[-1] / contact_idx.shape[0]:8.4f}ms"
            )
            print(
                "Avg tactile_sim time for current step per env:   "
                f" {frame_times_tactile[-1] / contact_idx.shape[0]:8.4f}ms"
            )
        else:
            # no sensor in contact
            print("Avg physics_sim time for current step per env:    ---------")
            print("Avg tactile_sim time for current step per env:    ---------")

        # print system utilization
        system_utilization_analytics = _get_utilization_percentages(reset=False)
        print(
            f"| CPU:{system_utilization_analytics[0]}% | "
            f"RAM:{system_utilization_analytics[1]}% | "
            f"GPU Compute:{system_utilization_analytics[2]}% | "
            f"GPU Memory: {system_utilization_analytics[3]:.2f}% |"
        )
        print("")

    env.close()

    pynvml.nvmlShutdown()


def main():
    """Main function."""
    # Define simulation env
    if args_cli.env == "physx_rigid":
        env_cfg = PhysXRigidEnvCfg()
    elif args_cli.env == "uipc":
        env_cfg = UipcEnvCfg()
    elif args_cli.env == "uipc_textured":
        env_cfg = UipcTexturedEnvCfg()
    else:
        raise RuntimeError(
            "Env not found. Try `--env_cfg physx_rigid` or `--env_cfg uipc` or `--env_cfg uipc_textured`."
        )
    print(env_cfg)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.gsmini.debug_vis = args_cli.debug_vis

    if args_cli.env == "physx_rigid":
        experiment = PhysXRigidEnv(env_cfg)
    elif args_cli.env == "uipc":
        experiment = UipcEnv(env_cfg)
    elif args_cli.env == "uipc_textured":
        experiment = UipcTexturedEnv(env_cfg)

    # # experiment = PhysXRigidEnv(PhysXRigidEnvCfg())
    # # experiment = UipcEnv(UipcEnvCfg())
    # experiment = UipcTexturedEnv(UipcTexturedEnvCfg())

    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(env=experiment)


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
