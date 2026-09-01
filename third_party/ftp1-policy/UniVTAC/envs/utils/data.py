import os
import cv2
import h5py
import pickle
import subprocess

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

class HDF5Handler:
    @staticmethod
    def stream_to_img(data, resize=False, convert_channels=False, path=None) -> np.ndarray:
        """
        将一个字节流数组解码为图像数组。

        Args:
            data: np.ndarray of shape (N,), 每个元素要么是 Python bytes，要么是 np.ndarray(dtype=uint8)
        Returns:
            imgs: np.ndarray of shape (N, H, W, C), dtype=uint8
        """
        # 确保 data 是可迭代的一维数组
        flat = data.ravel()

        imgs, img_len = None, len(flat)
        for idx, buf in enumerate(flat):
            # buf 可能是 bytes，也可能是 np.ndarray(dtype=uint8)
            if isinstance(buf, (bytes, bytearray)):
                arr = np.frombuffer(buf, dtype=np.uint8)
            elif isinstance(buf, np.ndarray) and buf.dtype == np.uint8:
                arr = buf
            else:
                raise TypeError(f"Unsupported buffer type: {type(buf)}")

            # 解码成 BGR 图像
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to decode image from buffer!")
            if resize and path is not None:
                if 'observation' in path:
                    # img = cv2.resize(img, (480, 270))  # Resize to 480x270 for observation images
                    # img = cv2.resize(img, (240, 240))  # Resize to 320x240 for observation images
                    img = cv2.resize(img, (256, 256))
                elif 'tactile' in path:
                    # img = cv2.resize(img, (320, 240))  # Resize to 320x240 for tactile images
                    img = cv2.resize(img, (256, 256))
            if convert_channels:
                img = np.transpose(img, (2, 0, 1)) # HWC -> CHW
            if imgs is None:
                imgs = np.empty((img_len, *img.shape), dtype=np.uint8)
            imgs[idx] = img

        return imgs

    @staticmethod
    def img_to_stream(imgs):
        max_len = 0
        encode_data = []
        for i in range(len(imgs)):
            success, encoded_image = cv2.imencode(".jpg", imgs[i])
            jpeg_data = encoded_image.tobytes()
            encode_data.append(jpeg_data)
            max_len = max(max_len, len(jpeg_data))
        return encode_data, max_len

    def gather(self, pkl_dir):
        assert Path(pkl_dir).exists(), f"{pkl_dir} does not exist"

        pkl_list = sorted([
            (int(p.stem), p) for p in Path(pkl_dir).glob("*.pkl")
        ], key=lambda x: x[0])
        
        results = {}
        for _, pkl_path in pkl_list:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            self.append(results, data)
        return results
 
    def append(self, target:dict, data:dict):
        for k, v in data.items():
            if isinstance(v, dict):
                if k not in target:
                    target[k] = {}
                self.append(target[k], v)
            else:
                if k not in target:
                    target[k] = []
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                if isinstance(v, np.ndarray):
                    while len(v.shape) > 1 and v.shape[0] == 1:
                        v = v[0]
                target[k].append(v)

    def hdf5_to_dict(self, node:h5py.Group, resize=False, convert_channels=False):
        result = {}
        for name, item in node.items():
            if isinstance(item, h5py.Dataset):
                data = item[()]
                if "rgb" in name:
                    result[name] = self.stream_to_img(data, resize, convert_channels)
                else:
                    result[name] = data
            elif isinstance(item, h5py.Group):
                result[name] = self.hdf5_to_dict(item, resize, convert_channels)

        if hasattr(node, "attrs") and len(node.attrs) > 0:
            result["_attrs"] = dict(node.attrs)
        return result

    def gather_hdf5(self, node:h5py.Group, data_paths:list[str], resize=False, convert_channels=False):
        result = {}
        for data_path in data_paths:
            endpoint = data_path.rsplit('/', 1)[-1]
            if 'rgb' in endpoint:
                result[data_path] = self.stream_to_img(node[data_path][()], resize, convert_channels, path=data_path)
            else:
                result[data_path] = node[data_path][()]
        return result
    
    def load_hdf5_metadata(self, hdf5_path, column:str='step'):
        with h5py.File(hdf5_path, "r") as f:
            column_data = f[column][()]
        return {
            'path': hdf5_path,
            'length': len(column_data)
        }

    def batch_gather_hdf5(
        self, hdf5_paths, data_paths:list[str], workers=4,
        resize=False, convert_channels=False, downsample_factor=2
    ):
        episode_start, episode_ends = 0, []
        for pid, path in enumerate(hdf5_paths):
            assert Path(path).exists(), f"{path} does not exist"
            with h5py.File(path, "r") as f:
                episode_start += len(np.arange(0, len(f['embodiment/joint']) - 1, downsample_factor))
                episode_ends.append(episode_start)
        total_length = episode_start
        print(f"Total data pairs: {total_length}")

        result = {}
        # process the first file to initialize the arrays
        with h5py.File(hdf5_paths[0], "r") as f:
            for data_path in data_paths:
                endpoint = data_path.rsplit('/', 1)[-1]
                if 'rgb' in endpoint:
                    data = self.stream_to_img(f[data_path][()], resize, convert_channels, path=data_path)
                else:
                    data = f[data_path][()]
                if data_path not in result:
                    if data_path == 'embodiment/joint':
                        result['embodiment/joint_action'] = np.empty(
                            (total_length, *data.shape[1:]), dtype=data.dtype)
                        result['embodiment/joint_state'] = np.empty(
                            (total_length, *data.shape[1:]), dtype=data.dtype)
                    elif data_path == 'embodiment/ee':
                        result['embodiment/ee_action'] = np.empty(
                            (total_length, *data.shape[1:]), dtype=data.dtype)
                        result['embodiment/ee_state'] = np.empty(
                            (total_length, *data.shape[1:]), dtype=data.dtype)
                    else:
                        result[data_path] = np.empty(
                            (total_length, *data.shape[1:]), dtype=data.dtype)
                
                downsample_arange = np.arange(0, len(data)-1, downsample_factor)
                if data_path == 'embodiment/joint':
                    result['embodiment/joint_action'][0:episode_ends[0]] = data[1:][downsample_arange]
                    result['embodiment/joint_state'][0:episode_ends[0]] = data[:-1][downsample_arange]
                elif data_path == 'embodiment/ee':
                    result['embodiment/ee_action'][0:episode_ends[0]] = data[1:][downsample_arange]
                    result['embodiment/ee_state'][0:episode_ends[0]] = data[:-1][downsample_arange]
                else:
                    result[data_path][0:episode_ends[0]] = data[:-1][downsample_arange]
        
        def process(eid):
            nonlocal result
            with h5py.File(hdf5_paths[eid], "r") as f:
                for data_path in data_paths:
                    endpoint = data_path.rsplit('/', 1)[-1]
                    if 'rgb' in endpoint:
                        data = self.stream_to_img(f[data_path][()], resize, convert_channels, path=data_path)
                    else:
                        data = f[data_path][()]
                
                    downsample_arange = np.arange(0, len(data)-1, downsample_factor)
                    if data_path == 'embodiment/joint':
                        result['embodiment/joint_action'][episode_ends[eid-1]:episode_ends[eid]] = data[1:][downsample_arange]
                        result['embodiment/joint_state'][episode_ends[eid-1]:episode_ends[eid]] = data[:-1][downsample_arange]
                    elif data_path == 'embodiment/ee':
                        result['embodiment/ee_action'][episode_ends[eid-1]:episode_ends[eid]] = data[1:][downsample_arange]
                        result['embodiment/ee_state'][episode_ends[eid-1]:episode_ends[eid]] = data[:-1][downsample_arange]
                    else:
                        result[data_path][episode_ends[eid-1]:episode_ends[eid]] = data[:-1][downsample_arange]
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            ids = list(range(len(hdf5_paths)))[1:]
            res = list(tqdm(executor.map(process, ids),
                            total=len(ids)+1, initial=1, desc="Loading dataset", unit="episode"))
        result['episode_ends'] = np.array(episode_ends, dtype=np.int64)
        return result

    def load_hdf5(self, hdf5_path, data_paths:list[str]|None=None, resize=False, convert_channels=False, downsample_factor=2):
        assert Path(hdf5_path).exists(), f"{hdf5_path} does not exist"
        with h5py.File(hdf5_path, "r") as f:
            if data_paths is None:
                data_dict = self.hdf5_to_dict(f, resize=resize, convert_channels=convert_channels)
            else:
                data_dict = self.gather_hdf5(f, data_paths, resize=resize, convert_channels=convert_channels, downsample_factor=downsample_factor)
        return data_dict
    
    def dict_to_hdf5(self, node:h5py.Group, data:dict, encode_images=True):
        for k, v in data.items():
            if isinstance(v, dict):
                subgroup = node.create_group(k)
                self.dict_to_hdf5(subgroup, v)
            elif isinstance(v, (list, np.ndarray)):
                if "rgb" in k and encode_images:
                    v = np.array(v)
                    encode_data, max_len = self.img_to_stream(v)
                    node.create_dataset(k, data=encode_data, dtype=f"S{max_len}")
                elif len(v) > 0 and isinstance(v[0], str):
                    max_len = np.max([len(s) for s in v])
                    node.create_dataset(k, data=v, dtype=f'S{max_len}')
                else:
                    v = np.array(v)
                    node.create_dataset(k, data=v)
            else:
                raise ValueError(f"Unsupported data type for key '{k}': {type(v)}")
        
    def pkls_to_hdf5(self, pkl_dir, hdf5_path):
        data = self.gather(pkl_dir)
        with h5py.File(hdf5_path, "w") as f:
            self.dict_to_hdf5(f, data)

class VideoHandler:
    def __init__(self):
        self.ffmpeg = None
        
    def reset(self, video_path, video_size):
        if self.ffmpeg is not None:
            self.close()

        self.video_path = Path(video_path)
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        self.video_size = video_size
        w, h = video_size
        self.ffmpeg = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{w}x{h}", "-framerate", "10",
            "-i", "-", "-pix_fmt", "yuv420p",
            "-vcodec", "libx264", "-crf", "18",  # 18 = higher quality (default 23)
            "-movflags", "+faststart",
            str(self.video_path)
        ], stdin=subprocess.PIPE)
    
    def __del__(self):
        if self.ffmpeg is not None:
            self.close()
 
    def write(self, frame:torch.Tensor):
        frame = frame.cpu().numpy()
        # video_size is (width, height); frame is (height, width, 3)
        target_w, target_h = self.video_size
        if frame.shape[0] != target_h or frame.shape[1] != target_w:
            # Prefer INTER_AREA when downscaling for sharper video
            interp = cv2.INTER_AREA if (frame.shape[0] > target_h or frame.shape[1] > target_w) else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (target_w, target_h), interpolation=interp)
        self.ffmpeg.stdin.write(frame.tobytes())
        # cv2.putText(frame, f'Streaming [{self.video_path.stem}]', (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        # self.stream.stdin.write(frame.tobytes())
    
    def forgive(self):
        if self.ffmpeg is None: return
        self.close()
        self.video_path.unlink(missing_ok=True)
 
    def close(self, result:str=None):
        self.ffmpeg.stdin.close()
        self.ffmpeg.wait()
        del self.ffmpeg

        # self.stream.stdin.close()
        # self.stream.wait()
        # del self.stream

        if result is not None:
            new_name = self.video_path.parent / f"{self.video_path.stem}_{result}.mp4"
            self.video_path.rename(new_name)
        self.ffmpeg = None
