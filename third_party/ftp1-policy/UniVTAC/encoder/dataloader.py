import torch
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('..')
from envs.utils.transforms import *
from envs.utils.data import HDF5Handler

import time
import h5py
import numpy as np
from tqdm import tqdm
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

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_paths:list):
        self.handler = HDF5Handler()

        self.name = {
            'left_marked_rgb': 'tactile/left_tactile/rgb_marker',
            'right_marked_rgb': 'tactile/right_tactile/rgb_marker',
            'left_marker': 'tactile/left_tactile/marker',
            'right_marker': 'tactile/right_tactile/marker',
            'left_depth': 'tactile/left_tactile/depth',
            'right_depth': 'tactile/right_tactile/depth',
            'right_rgb': 'tactile/right_tactile/rgb',
            'left_rgb': 'tactile/left_tactile/rgb',
            'prism_pose': 'actor/prism',
            'left_pose': 'tactile/left_tactile/pose',
            'right_pose': 'tactile/right_tactile/pose',
        }
        
        self._data_metadata = []
        for hdf5_file in tqdm(hdf5_paths, desc='Loading HDF5 metadata'):
            hdf5_path = str(hdf5_file)
            hdf5_metadata = self.handler.load_hdf5_metadata(hdf5_path)
            for length in range(hdf5_metadata['length']):
                self._data_metadata.extend([{
                    'hdf5_path': hdf5_path,
                    'index': length,
                    'lr_tag': 'left'
                }, {
                    'hdf5_path': hdf5_path,
                    'index': length,
                    'lr_tag': 'right'
                }])

        self.trans = transforms.Compose([
            transforms.ToTensor(), # auto convert HWC to CHW and scale to [0, 1]
            transforms.Resize((320, 240)),
        ])

    def __len__(self): 
        return len(self._data_metadata)

    def __getitem__(self, idx):
        hdf5_path = self._data_metadata[idx]['hdf5_path']
        real_idx  = self._data_metadata[idx]['index']
        lr_tag    = self._data_metadata[idx]['lr_tag']

        thread_id = threading.get_ident()
        if thread_id not in _worker_hdf5_cache:
            _worker_hdf5_cache[thread_id] = {}
        cache = _worker_hdf5_cache[thread_id]

        if hdf5_path not in cache:
            cache[hdf5_path] = h5py.File(hdf5_path, 'r')

        f = cache[hdf5_path]

        data = {}
        for key, path in self.name.items():
            if 'rgb' in path:
                data[key] = self.handler.stream_to_img(
                    f[path][real_idx],
                    resize=False, convert_channels=False, path=path
                ).squeeze(0)
            else:
                data[key] = f[path][real_idx]
        
        rgb = self.trans(data[f'{lr_tag}_rgb'])
        marked_rgb = self.trans(data[f'{lr_tag}_marked_rgb'])
        marker = data[f'{lr_tag}_marker'][0, :63] / np.array([320, 240], dtype=np.float32)
        depth = torch.from_numpy((data[f'{lr_tag}_depth'].astype(np.float32) - 24.0) / (34.0 - 24.0)).unsqueeze(0)
        pose = Pose.from_list(data[f'{lr_tag}_pose'])
        prism_pose = Pose.from_list(data['prism_pose'])
        vec = torch.tensor(prism_pose.rebase(pose).tolist())

        sample = {
            'pose': vec,
            'rgb': rgb,
            'marked_rgb': marked_rgb,
            'marker': torch.from_numpy(marker),
            'depth': depth,
        }
        return sample

import os
from tqdm import tqdm
if __name__ == '__main__':
    prism_names = ['CircleShell', 'Cross', 'Cubehole', 'Cuboid', 'Cylinder', 'Doubleslope', 'Hemisphere', 'Line', 'Pacman', 'S', 'Sphere', 'Star', 'Tetrahedron', 'Torus']
    hdf5_paths = []
    for name in tqdm(prism_names, desc='Loading'):
        hdf5_paths.extend(list(Path(f'../data/demo2/{name}/hdf5').glob('*.hdf5')))
    print(f'Found {len(hdf5_paths)} hdf5 files.')
    
    os.system('mkdir -p /dev/shm/hdf5_cache')
    for name in prism_names:
        os.system(f'mkdir -p /dev/shm/hdf5_cache/{name}')
    for p in tqdm(hdf5_paths, desc='Copying to shm'):
        name = p.parent.parent.stem
        os.system(f'cp {str(p)} /dev/shm/hdf5_cache/{name}/{p.stem}.hdf5')

    data = HDF5Dataset(hdf5_paths)
    
    # depth = data[0]['marker']
    # print(depth.shape, depth.min(), depth.max())

    import time
    it = iter(data)
    for i in range(10):
        t0 = time.time()
        batch = next(it)
        print(f"Batch {i}: {time.time() - t0:.3f}s")
