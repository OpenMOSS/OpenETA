import numpy as np
import torch
import os
import h5py
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

import IPython

e = IPython.embed


class EpisodicDataset(torch.utils.data.Dataset):
    tac_image_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])
    cam_image_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, episode_ids, dataset_dir, camera_names, tactile_names, norm_stats, max_action_len):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.tactile_names = tactile_names
        self.norm_stats = norm_stats
        self.max_action_len = max_action_len
        self.is_sim = None
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            is_sim = None
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root["/observations/qpos"][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][start_ts]
            tactile_dict = dict()
            for tactile_name in self.tactile_names:
                tactile_dict[tactile_name] = root[f"/observations/images/{tactile_name}"][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root["/action"][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root["/action"][max(0, start_ts - 1):]  # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros((self.max_action_len, action.shape[1]), dtype=np.float32)  # 根据max_action_len初始化
        padded_action[:action_len] = action
        is_pad = np.ones(self.max_action_len, dtype=bool)  # 初始化为全1（True）
        is_pad[:action_len] = 0  # 前action_len个位置设置为0（False），表示非填充部分

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(self.cam_image_trans(image_dict[cam_name]))
        all_cam_images = torch.stack(all_cam_images, dim=0)
        
        all_tac_images = []
        if len(self.tactile_names) > 0:
            for tac_name in self.tactile_names:
                all_tac_images.append(self.tac_image_trans(tactile_dict[tac_name]))
            all_tac_images = torch.stack(all_tac_images, dim=0)
        else:
            all_tac_images = torch.Tensor([])

        # construct observations
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # normalize image and change dtype to float
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return all_cam_images, all_tac_images, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    max_action_len = 0
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist."
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]  # Assuming this is a numpy array
            action = root["/action"][()]
        max_action_len = max(max_action_len, action.shape[0])
        all_qpos_data.append(qpos)
        all_action_data.append(action)
    
    all_qpos_data = np.concatenate(all_qpos_data, axis=0)
    all_action_data = np.concatenate(all_action_data, axis=0)
    
    # normalize action data
    action_mean = np.mean(all_action_data, axis=0)
    action_std = np.std(all_action_data, axis=0)
    action_std = np.clip(action_std, 1e-2, np.inf)  # clipping
    
    # normalize qpos data
    qpos_mean = np.mean(all_qpos_data, axis=0)
    qpos_std = np.std(all_qpos_data, axis=0)
    qpos_std = np.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean,
        "action_std": action_std,
        "qpos_mean": qpos_mean,
        "qpos_std": qpos_std,
        "example_qpos": qpos,
    }

    return stats, max_action_len

import threading
_worker_hdf5_cache = {}

def worker_init_fn(worker_id):
    _worker_hdf5_cache[threading.get_ident()] = {}

def worker_clear_fn():
    """清理当前线程的 HDF5 文件"""
    ident = threading.get_ident()
    if ident in _worker_hdf5_cache:
        for f in _worker_hdf5_cache[ident].values():
            f.close()
        del _worker_hdf5_cache[ident]

class TacArenaDataset(torch.utils.data.Dataset):
    tac_image_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])
    cam_image_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, episode_ids, dataset_dir, camera_names, tactile_names, norm_stats, chunk_size):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.tactile_names = tactile_names
        self.chunk_size = chunk_size
        self.norm_stats = norm_stats
        self.create_dataset()
        
    def create_dataset(self):
        self.hdf5_files = np.array(list(
            Path(self.dataset_dir).glob("*.hdf5")
        ))[self.episode_ids]
        
        self._dataset = []
        for episode_idx in self.episode_ids:
            dataset_path = str(Path(self.dataset_dir) / f"episode_{episode_idx}.hdf5")
            with h5py.File(dataset_path, "r") as root:
                qpos = root["/observations/qpos"][()]
            episode_len = qpos.shape[0]
            
            for start_idx in range(episode_len):
                self._dataset.append((episode_idx, start_idx, min(start_idx + self.chunk_size, episode_len)))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        thread_id = threading.get_ident()
        if thread_id not in _worker_hdf5_cache:
            _worker_hdf5_cache[thread_id] = {}
        cache = _worker_hdf5_cache[thread_id]

        episode_idx, start_ts, end_ts = self._dataset[index]

        hdf5_path = str(Path(self.dataset_dir) / f"episode_{episode_idx}.hdf5")
        if hdf5_path not in cache:
            cache[hdf5_path] = h5py.File(hdf5_path, 'r')

        root = cache[hdf5_path]

        # get observation at start_ts only
        qpos = root["/observations/qpos"][start_ts]
        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = root[f"/observations/images/{cam_name}"][start_ts]
        tactile_dict = dict()
        for tactile_name in self.tactile_names:
            tactile_dict[tactile_name] = root[f"/observations/images/{tactile_name}"][start_ts]

        action = root["/action"][start_ts:end_ts]
        padded_action = np.zeros((self.chunk_size, action.shape[1]), dtype=np.float32)
        padded_action[0:action.shape[0]] = action
        is_pad = np.ones(self.chunk_size, dtype=bool)
        is_pad[0:action.shape[0]] = 0

        if len(self.camera_names) > 0:
            all_cam_images = []
            for cam_name in self.camera_names:
                all_cam_images.append(self.cam_image_trans(image_dict[cam_name]))
            all_cam_images = torch.stack(all_cam_images, dim=0)
        else:
            all_cam_images = torch.Tensor([])
        
        if len(self.tactile_names) > 0:
            all_tac_images = []
            for tac_name in self.tactile_names:
                all_tac_images.append(self.tac_image_trans(tactile_dict[tac_name]))
            all_tac_images = torch.stack(all_tac_images, dim=0)
        else:
            all_tac_images = torch.Tensor([])

        # construct observations
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # normalize image and change dtype to float
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return all_cam_images, all_tac_images, qpos_data, action_data, is_pad


def load_data(dataset_dir, num_episodes, camera_names, tactile_names, batch_size_train, batch_size_val, chunk_size):
    print(f"\nData from: {dataset_dir}\n")
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats, max_action_len = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    # train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, tactile_names, norm_stats, max_action_len)
    # val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, tactile_names, norm_stats, max_action_len)
    train_dataset = TacArenaDataset(train_indices, dataset_dir, camera_names, tactile_names, norm_stats, chunk_size)
    val_dataset = TacArenaDataset(val_indices, dataset_dir, camera_names, tactile_names, norm_stats, chunk_size)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=6,
        persistent_workers=True, worker_init_fn=worker_init_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=6,
        persistent_workers=True, worker_init_fn=worker_init_fn
    )

    return train_dataloader, val_dataloader, norm_stats, True


### env utils


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
