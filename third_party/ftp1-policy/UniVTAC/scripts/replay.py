import sys

sys.path.append(".")
sys.path.append('../')

import os
import time
import json
import yaml
import torch
import argparse
import traceback
import numpy as np
from pathlib import Path
from typing import Literal

from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(
    description="Replay Data"
)
parser.add_argument(
    "task_name",
    type=str,
    help="Task name",
)
parser.add_argument(
    "task_config",
    type=str,
    help="Task name",
)
parser.add_argument(
    "--gpu",
    type=str,
    default=None,
)
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.livestream = 2
args_cli.num_envs = 1

if args_cli.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args_cli.gpu

# launch omniverse app, must done before importing anything from omni.isaac
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib
from typing import TYPE_CHECKING
from envs.utils.data import HDF5Handler
if TYPE_CHECKING:
    from envs._base_task import BaseTask, BaseTaskCfg

log_path = Path('./log')
def log(msg):
    global log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    msg = f"[{time.strftime(r'%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(log_path, 'a') as f:
        f.write(msg + '\n')
    print(msg)

def replay(task: 'BaseTask', seed, data_root:Path):
    eval_start = time.perf_counter()
    task.reset(seed=seed)

    succ = False
    traj_data = HDF5Handler().load_hdf5(data_root / f'{seed}.hdf5')
    qpos_list = torch.from_numpy(traj_data['embodiment']['joint'][:, :8]).to(device=task.device)
    # vel_list = torch.from_numpy(traj_data['embodiment']['vel'][:, :8]).to(device=task.device)
    ee_list = torch.from_numpy(traj_data['embodiment']['ee'][:, :3]).to(device=task.device)
    
    # select = np.arange(1, qpos_list.shape[0], 2)
    # qpos_list = qpos_list[select]
    # vel_list = vel_list[select]
    # ee_list = ee_list[select]
    
    traj_list = []
    for idx in np.arange(0, qpos_list.shape[0], 2):
        action = qpos_list[idx]

        exec_succ, eval_succ = task.take_action(action, action_type='qpos', force=False)
        observation = task._get_observations()
        arm_dis = torch.abs(action[:7] - observation['embodiment']['joint'][:7])
        gripper_dis = torch.abs(action[7] - observation['embodiment']['joint'][7:])
        ee_dis = torch.abs(ee_list[idx] - observation['embodiment']['ee'][:3])

        if torch.any(gripper_dis > 1e-3) or torch.any(ee_dis > 1e-3):
            log(f"[{idx:3d}] arm_dis: {np.max(arm_dis.cpu().numpy())}, gripper_dis: {np.max(gripper_dis.cpu().numpy())}, ee_dis: {ee_dis.cpu().numpy()}, eval_succ: {eval_succ}, exec_succ: {exec_succ}")
        
        traj_list.append({
            'target_ee': ee_list[idx].cpu().tolist(),
            'target_action': action.cpu().tolist(),
            'result_qpos': observation['embodiment']['joint'][:8].cpu().tolist(),
            'result_ee': observation['embodiment']['ee'][:3].cpu().tolist(),
        })
    
    seed_root = task.save_root / 'replay_traj'
    seed_root.mkdir(parents=True, exist_ok=True)
    with open(seed_root / f'{seed}.json', 'w') as f:
        json.dump(traj_list, f, indent=4)

    if task.eval_success:
        succ = True

    eval_cost = time.perf_counter() - eval_start
    succ_status = 'success' if succ else 'failed'
    task.clean_cache(result=succ_status)
    return succ_status, eval_cost

def replay_seeds(task: 'BaseTask', seeds:list, data_root:Path):
    test_num, succ_num = 0, 0
    for seed in seeds:
        test_num += 1
        result, eval_cost = replay(task, seed, data_root)
        succ_num += 1 if result == 'success' else 0
        log(f"[{test_num:<3d}] Seed {seed} {result} after {eval_cost:.2f} s.\n"
        f"steps: {task.step_count:<5d}, actions: {task.take_action_cnt:<5d}.\n"
        f"Instruction: {task.instruction}\n"
        f"Total {succ_num}/{test_num}({succ_num/test_num*100:.2f}%) success.")
    return {
        'test_num': test_num,
        'succ_num': succ_num
    }


def get_config(file, default_root:Path, type:Literal['yaml', 'json']):
    if type == 'yaml':
        if file.endswith('.yml') or file.endswith('.yaml'):
            file = Path(file)
        else:
            file = default_root / f'{file}.yml'
        with open(file, 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config, file
    else:
        if file.endswith('.json'):
            file = Path(file)
        else:
            file = default_root / f'{file}.json'
        with open(file, 'r') as f:
            config = json.load(f)
        return config, file

task_module, policy_module = None, None
def main():
    global args_cli, task_module, policy_module, log_path

    task_file_name = args_cli.task_name
    task_config_name = args_cli.task_config
    
    task_config, task_config_file = get_config(
        task_config_name, default_root=Path(__file__).parent.parent / 'task_config', type='yaml'
    )

    task_module = importlib.import_module(f"envs.{task_file_name}")

    curr_time = time.strftime(r'%Y-%m-%d_%H:%M:%S')

    env_cfg:BaseTaskCfg = task_module.TaskCfg()
    env_cfg.save_dir = Path('eval_result') / 'replay' / task_file_name / task_config_file.stem / curr_time
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    env_cfg.obs_data_type = task_config.get("observations", {})
    env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
    env_cfg.video_frequency = task_config.get("video_frequency", env_cfg.video_frequency)
    env_cfg.random_texture = task_config.get("random_texture", False)

    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None \
        else env_cfg.sim.device

    init_start = time.perf_counter()
    task:BaseTask = task_module.Task(env_cfg, mode='eval')
    task_init_cost = time.perf_counter() - init_start
    
    log_path = task.save_root / f"log.log"
    log(f"Task Name: {task_file_name}")
    log(f"Task Config: {task_config_file.absolute()}") 
    log(f"Task init finish in {task_init_cost:.2f} seconds.")
    
    data_root = Path(__file__).parent.parent / 'data' / task_config_name / task_file_name / 'hdf5'
    seeds = sorted([int(p.stem) for p in data_root.glob('*.hdf5')])
    log(f"Start replaying {len(seeds)} seeds from {data_root}.")

    results = replay_seeds(task, seeds=seeds, data_root=data_root)
    log(f"Final Result: {results['succ_num']}/{results['test_num']}({results['succ_num']/results['test_num']*100:.2f}%) success.")
    
    task.close()
    simulation_app.close()

if __name__ == "__main__":
    main()