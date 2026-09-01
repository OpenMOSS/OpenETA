#!/usr/bin/env python3
"""
ACT Data Preprocessing for TacArena (Optimized)
Efficiently processes multiple episodes by leveraging batch_gather_hdf5
- Uses episode_ends for efficient per-episode data splitting
- Computes normalization statistics on-the-fly during processing
- Per-episode HDF5 output format compatible with TactileACT
"""

import os
import h5py
import shutil
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from envs.utils.data import HDF5Handler


def compute_norm_stats(qpos_data, action_data, left_tac_data, right_tac_data):
    """计算所有数据的归一化统计"""
    # 直接使用原始数据计算统计
    qpos_mean = qpos_data.mean(axis=0).astype(np.float32)
    qpos_std = qpos_data.std(axis=0).astype(np.float32)
    qpos_min = qpos_data.min(axis=0).astype(np.float32)
    qpos_max = qpos_data.max(axis=0).astype(np.float32)
    
    # 计算 action 统计
    action_mean = action_data.mean(axis=0).astype(np.float32)
    action_std = action_data.std(axis=0).astype(np.float32)
    action_min = action_data.min(axis=0).astype(np.float32)
    action_max = action_data.max(axis=0).astype(np.float32)
    
    # 计算触觉图像统计（归一化到 [0, 1]）
    left_tac_normalized_flatten = (left_tac_data.astype(np.float32) / 255.0).reshape(-1, 3)
    right_tac_normalized_flatten = (right_tac_data.astype(np.float32) / 255.0).reshape(-1, 3)
    
    left_tac_mean = left_tac_normalized_flatten.mean(axis=0).astype(np.float32)
    left_tac_std = left_tac_normalized_flatten.std(axis=0).astype(np.float32)
    
    right_tac_mean = right_tac_normalized_flatten.mean(axis=0).astype(np.float32)
    right_tac_std = right_tac_normalized_flatten.std(axis=0).astype(np.float32)
    
    stats = {
        "qpos_mean": qpos_mean.tolist(),
        "qpos_std": qpos_std.tolist(),
        "qpos_min": qpos_min.tolist(),
        "qpos_max": qpos_max.tolist(),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "action_min": action_min.tolist(),
        "action_max": action_max.tolist(),
        "left_tac_mean": left_tac_mean.tolist(),
        "left_tac_std": left_tac_std.tolist(),
        "right_tac_mean": right_tac_mean.tolist(),
        "right_tac_std": right_tac_std.tolist(),
    }
    return stats


def _resolve_tactile_keys(first_hdf5_path: Path) -> tuple[str, str]:
    """Resolve tactile key names from UniVTAC raw HDF5 (supports old/new naming)."""
    with h5py.File(first_hdf5_path, "r") as f:
        tactile_group = f.get("tactile")
        if tactile_group is None:
            raise KeyError(f"Missing 'tactile' group in {first_hdf5_path}")

        key_candidates = [
            ("left_gsmini", "right_gsmini"),
            ("left_tactile", "right_tactile"),
        ]
        for left_key, right_key in key_candidates:
            if left_key in tactile_group and right_key in tactile_group:
                return left_key, right_key

        available = sorted(list(tactile_group.keys()))
        raise KeyError(
            f"Cannot resolve tactile keys from {first_hdf5_path}. "
            f"Expected one of {key_candidates}, got {available}"
        )


def process_episodes_batch(hdf5_paths, save_dir):
    """
    使用 batch_gather_hdf5 高效处理多个 episode
    利用 episode_ends 进行有效的数据分割
    
    Args:
        hdf5_paths: HDF5 文件路径列表
        save_dir: 输出目录
        
    Returns:
        tuple: (successful_count, qpos_data, action_data, left_tac_data, right_tac_data)
               返回原始数据用于统计计算
    """
    if not hdf5_paths:
        return 0, None, None, None, None
    
    # 批量加载所有文件
    global camera_type, downsample_factor
    handler = HDF5Handler()
    left_tac_key, right_tac_key = _resolve_tactile_keys(Path(hdf5_paths[0]))
    data_paths = [
        'embodiment/joint',
        'embodiment/ee',
        f'tactile/{left_tac_key}/rgb_marker',
        f'tactile/{right_tac_key}/rgb_marker',
    ]
    if camera_type == 'all':
        data_paths.append(f'observation/head/rgb')
        data_paths.append(f'observation/wrist/rgb')
    else:
        data_paths.append(f'observation/{camera_type}/rgb')

    batch_data = handler.batch_gather_hdf5(
        [str(p) for p in hdf5_paths],
        data_paths=data_paths,
        resize=True,
        convert_channels=False,  # 保持 HWC 格式
        downsample_factor=downsample_factor
    )
    
    # 获取 episode_ends 以分割数据
    episode_ends = batch_data.get('episode_ends', None)
    
    if episode_ends is None:
        # 如果没有 episode_ends，假设整个批次是一个 episode
        episode_ends = [len(batch_data['embodiment/joint_state'])]
    
    # 提取数据引用
    joint_state = batch_data['embodiment/joint_state']  # (N, 8)
    joint_action = batch_data['embodiment/joint_action']  # (N, 8)
    ee_state = batch_data['embodiment/ee_state']  # (N, 7)
    if camera_type == 'all':
        wrist_cam = batch_data[f'observation/wrist/rgb']  # (N, H, W, 3)
        head_cam = batch_data[f'observation/head/rgb']  # (N, H, W, 3)
    else:
        head_cam = batch_data[f'observation/{camera_type}/rgb']  # (N, H, W, 3)
    left_tac = batch_data[f'tactile/{left_tac_key}/rgb_marker']  # (N, H, W, 3)
    right_tac = batch_data[f'tactile/{right_tac_key}/rgb_marker']  # (N, H, W, 3)
    
    successful_count = 0
    start_idx = 0
    
    # 按 episode_ends 分割数据并保存
    for ep_idx, end_idx in tqdm(
        enumerate(episode_ends), desc="Writing episodes", total=len(episode_ends)
    ):
        try:
            # 提取当前 episode 数据
            qpos = joint_state[start_idx:end_idx, :8]
            action = joint_action[start_idx:end_idx, :8]
            ee = ee_state[start_idx:end_idx, :7]
            head = head_cam[start_idx:end_idx]
            if camera_type == 'all':
                wrist = wrist_cam[start_idx:end_idx]
            left = left_tac[start_idx:end_idx]
            right = right_tac[start_idx:end_idx]
            
            # 保存 episode
            output_path = os.path.join(save_dir, f'episode_{ep_idx}.hdf5')
            _save_episode_hdf5(
                output_path, qpos, action, ee, head, left, right, wrist if camera_type == 'all' else None
            )
            
            successful_count += 1
            start_idx = end_idx
            
        except Exception as e:
            print(f"  Error processing episode {ep_idx}: {e}")
            start_idx = end_idx
            continue
    
    # 直接返回原始数据用于统计（无需重新收集）
    return successful_count, joint_state[:, :8], joint_action[:, :8], left_tac, right_tac


def _save_episode_hdf5(output_path, qpos, action, ee, head_cam, left_tac, right_tac, wrist=None):
    """保存单个 episode 到 HDF5 文件（TactileACT 格式）"""
    with h5py.File(output_path, 'w') as f:
        # 保存动作 (T, 8)
        f.create_dataset(
            'action',
            data=action.astype(np.float32),
            chunks=(1, action.shape[1]),
            dtype='float32',
            compression='gzip',
            compression_opts=4
        )
        
        # 创建观察 group
        obs = f.create_group('observations')
        
        # 保存关节位置 (T, 8)
        obs.create_dataset(
            'qpos',
            data=qpos.astype(np.float32),
            chunks=(1, qpos.shape[1]),
            dtype='float32',
            compression='gzip',
            compression_opts=4
        )
        obs.create_dataset(
            'ee',
            data=ee.astype(np.float32),
            chunks=(1, ee.shape[1]),
            dtype='float32',
            compression='gzip',
            compression_opts=4
        )
        
        # 创建图像 group
        images = obs.create_group('images')
        
        # 保存相机图像 (T, H, W, 3)
        images.create_dataset(
            'cam_high',
            data=head_cam.astype(np.uint8),
            chunks=(1, *head_cam.shape[1:]),
            dtype='uint8',
            compression='gzip',
            compression_opts=4
        )
        if wrist is not None:
            images.create_dataset(
                'cam_wrist',
                data=wrist.astype(np.uint8),
                chunks=(1, *wrist.shape[1:]),
                dtype='uint8',
                compression='gzip',
                compression_opts=4
            )
        
        images.create_dataset(
            'cam_left_tactile',
            data=left_tac.astype(np.uint8),
            chunks=(1, *left_tac.shape[1:]),
            dtype='uint8',
            compression='gzip',
            compression_opts=4
        )
        
        images.create_dataset(
            'cam_right_tactile',
            data=right_tac.astype(np.uint8),
            chunks=(1, *right_tac.shape[1:]),
            dtype='uint8',
            compression='gzip',
            compression_opts=4
        )
        
        # 添加 metadata
        f.attrs['sim'] = True
        f.attrs['image_height'] = head_cam.shape[1]
        f.attrs['image_width'] = head_cam.shape[2]
        f.attrs['gelsight_height'] = left_tac.shape[1]
        f.attrs['gelsight_width'] = left_tac.shape[2]
        f.attrs['num_timesteps'] = qpos.shape[0]

camera_type = 'head'
downsample_factor = 1
def main():
    parser = argparse.ArgumentParser(
        description="Process TacArena data for ACT training (optimized batch processing)"
    )
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., insert_hole)",
    )
    parser.add_argument(
        "task_config",
        type=str,
        help="Task configuration name (e.g., default)",
    )
    parser.add_argument(
        "expert_data_num",
        type=int,
        help="Number of episodes to process (e.g., 50)",
    )
    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    num = args.expert_data_num

    # 输入目录：TacArena 原始数据
    load_dir = f"../../data/{task_name}/{task_config}/"
    
    # 输出目录：per-episode HDF5 文件
    save_dir = f"./data/{task_name}-{task_config}-{num}"
    
    global camera_type, downsample_factor
    with open('../task_settings.json', 'r') as f:
        task_settings = json.load(f)
    assert task_name in task_settings, f"Task '{task_name}' not found in task_settings.json"
    camera_type = task_settings[task_name].get('camera_type', 'head')
    downsample_factor = task_settings[task_name].get('downsample', 1)
    print(f"Loading {num} episodes with camera type '{camera_type}', downsample factor {downsample_factor}.")

    # 清空已存在的输出目录
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # 获取所有 HDF5 文件路径
    hdf5_dir = Path(load_dir) / 'hdf5'
    if not hdf5_dir.exists():
        raise FileNotFoundError(f"HDF5 directory not found: {hdf5_dir}")
    
    hdf5_data_path = sorted(
        [i for i in hdf5_dir.glob('*.hdf5')],
        key=lambda x: int(x.stem)
    )
    
    if len(hdf5_data_path) < num:
        print(f"Warning: Only {len(hdf5_data_path)} episodes found, but {num} requested.")
        num = len(hdf5_data_path)
    
    hdf5_data_path = hdf5_data_path[:num]

    successful, qpos_data, action_data, left_tac_data, right_tac_data = \
        process_episodes_batch(hdf5_data_path, save_dir)
        
    stats = compute_norm_stats(qpos_data, action_data, left_tac_data, right_tac_data)
    stats['task_name'] = task_name
    stats['task_config'] = task_config
    stats['num_episodes'] = num
    stats['camera'] = ['cam_high', 'cam_wrist'] if camera_type == 'all' else ['cam_high']
    stats['tactile'] = ['cam_left_tactile', 'cam_right_tactile']
    
    # 保存为 JSON
    output_path = os.path.join(save_dir, 'norm_stats.json')
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    main()
