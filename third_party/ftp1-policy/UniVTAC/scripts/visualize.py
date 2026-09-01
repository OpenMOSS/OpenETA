import cv2
import torch
import pickle
import imageio
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pprint import pprint
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.append('.')
from envs.utils.data import HDF5Handler

def to_cpu(data):
    if isinstance(data, dict):
        return {k: to_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_cpu(v) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.cpu()
    else:
        return data

data_length = 0
def read_from_cache(data_root, seed):
    global data_length

    cache = Path(data_root) / '.cache' / str(seed)
    data_path = sorted([i for i in cache.glob('*.pkl')], key=lambda x: int(x.stem))
    data_length = len(data_path)

    print('loading data from cache:', cache)
    yield None
    for p in data_path:
        try:
            data = pickle.load(open(p, 'rb'))
            yield data
        except Exception as e:
            print(f'[Warning] Failed to load {p}: {e}')

def get_dict_data(d:dict, idx):
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = get_dict_data(v, idx)
        else:
            result[k] = v[idx]
    return result

def read_from_hdf5(data_root, seed):
    global data_length

    data_path = Path(data_root) / 'hdf5' / f'{seed}.hdf5'
    data = HDF5Handler().load_hdf5(data_path)
    
    data_length = len(data['observation']['head']['rgb'])
    yield None

    for i in range(data_length):
        frame = get_dict_data(data, i)
        yield frame

def get_structure(data):
    structure = {}
    if isinstance(data, dict):
        for k, v in data.items():
            structure[k] = get_structure(v)
    elif isinstance(data, list):
        structure = f'list[{len(data)}]'
    elif isinstance(data, tuple):
        structure = f'tuple[{len(data)}]'
    elif isinstance(data, torch.Tensor):
        structure = f'torch.Tensor{tuple(data.shape)}'
    elif isinstance(data, np.ndarray):
        structure = f'np.ndarray{data.shape}'
    else:
        structure = 'type: ' + str(type(data))
    return structure

def gen_video(data_list, keys, save_path, FPS=30, COLS=2, UNI_W=None, UNI_H=None):
    global data_length
    def get_val(frame, key):
        if key:
            return get_val(frame[key[0]], key[1:])
        else:
            return frame

    fonts = '/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc'
    if not Path(fonts).exists():
        fonts = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(fonts, 30)

    def to_Image(array, anno=None):
        nonlocal font

        if isinstance(array, torch.Tensor):
            while len(array.shape) > 1 and array.shape[0] == 1:
                array = array[0].cpu().numpy()
        elif isinstance(array, np.ndarray):
            while len(array.shape) > 1 and array.shape[0] == 1:
                array = array[0]
        else:
            raise NotImplementedError(f'Not supported type {type(array)} for visualization!')
        
        # if np.max(array) <= 1.0:
        #     array = (array * 255).astype(np.uint8)
        #     if '[0]' in anno:
        #         print(f'[Warning] {anno}: Auto convert range [0, 1] to [0, 255] for visualization!')
        # else:
 
        if len(array.shape) == 3 and array.shape[2] == 1:
            # depth map
            depth = array[:, :, 0]
            depth = np.where(depth == np.inf, 100.0, depth)
            depth = np.where(depth > 5.0, 5.0, depth)
            depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_vis = (depth_vis * 255).astype(np.uint8)
            
            array = np.stack([depth_vis, depth_vis, depth_vis], axis=2)

        array = array.astype(np.uint8)
        
        img = Image.fromarray(array)
        if anno is not None:
            draw_obj = ImageDraw.Draw(img)
            draw_obj.text((20, 5), anno, fill='#FF0000', font=font)
        return img

    with imageio.get_writer(save_path, fps=FPS) as writer:
        for idx, frame in tqdm(enumerate(data_list), total=data_length):
            if idx == 0:
                pprint(get_structure(to_cpu(frame)))
            
            imgs:list[Image.Image] = []
            # print(get_val(frame, ['joint_action']))
            for key in keys:
                imgs.append(to_Image(
                    get_val(frame, key),
                    f'{"/".join(key)} [{idx}]'
                ))

            if UNI_W is None or UNI_H is None:
                UNI_W, UNI_H = imgs[0].width, imgs[0].height

            ROWS = np.ceil(len(imgs) / COLS).astype(int)
            opt_array = Image.new(
                'RGB', (
                    UNI_W*COLS+20*(COLS-1),
                    UNI_H*ROWS+20*(ROWS-1)
                )
            )
            for i, img in enumerate(imgs):
                if img.width > UNI_W or img.height > UNI_H:
                    img = img.resize((UNI_W, UNI_H))
                if img.width < UNI_W or img.height < UNI_H:
                    tmp = Image.new('RGB', (UNI_W, UNI_H), (255, 255, 255))
                    tmp.paste(img, ((UNI_W-img.width)//2, (UNI_H-img.height)//2))
                    img = tmp
                opt_array.paste(img, (
                    (i%COLS)*(UNI_W+20), (i//COLS)*(UNI_H+20)))
            opt_array = opt_array.resize(((opt_array.width+15)//16*16, (opt_array.height+15)//16*16))
            writer.append_data(np.array(opt_array))
    print(f'Video saved to {Path(save_path).absolute()}')

def print_frame(data_root, seed, frame=-1):
    joint_pos={
        "panda_joint1": 0.0,
        "panda_joint2": 0.0,
        "panda_joint3": 0.0,
        "panda_joint4": -2.46,
        "panda_joint5": 0.0,
        "panda_joint6": 2.5,
        "panda_joint7": 0.741,
        "panda_finger_joint1": 0.012,
        "panda_finger_joint2": 0.012
    }
    cache = Path(data_root) / '.cache' / str(seed)
    if cache.exists():
        data_path = sorted([i for i in cache.glob('*.pkl')], key=lambda x: int(x.stem))
        if frame < 0:
            frame = len(data_path) + frame
        data = pickle.load(open(data_path[frame], 'rb'))
    else:
        data_path = Path(data_root) / 'hdf5' / f'{seed}.hdf5'
        data = HDF5Handler().load_hdf5(data_path)
        data = get_dict_data(data, frame)
 
    action = np.round(data['joint_action'].reshape(-1), 4)
    for i, k in enumerate(joint_pos.keys()):
        joint_pos[k] = action[i]
    joint_pos['panda_finger_joint1'] = max(joint_pos['panda_finger_joint1'], joint_pos['panda_finger_joint2'])
    joint_pos['panda_finger_joint2'] = joint_pos['panda_finger_joint1']
    print('joint_pos =', end = ' ')
    pprint(joint_pos)

    for name, actor_pose in data['actor'].items():
        print(f'{name}: {actor_pose.tolist()}')

def main(task, name, config, seed, is_cache):
    global data_length
    data_root = f'./data/{config}/{name}'
    if task == 'video':
        if is_cache:
            data_list = read_from_cache(data_root, seed)
        else:
            if not (Path(data_root) / 'hdf5' / f'{seed}.hdf5').exists():
                print(f'[Warning] HDF5 file not found, try to load from cache instead.')
                data_list = read_from_cache(data_root, seed)
            else:
                data_list = read_from_hdf5(data_root, seed)
        next(data_list) # init
        print(f'Loaded {data_length} samples from {data_root}, structure:')
        gen_video(data_list, [
            ['tactile', 'left_tactile', 'rgb_marker'],
            ['tactile', 'right_tactile', 'rgb_marker'],
            ['observation', 'wrist', 'rgb'],
            ['observation', 'head', 'rgb'],
        ], f'{name}_{seed}.mp4', FPS=30, COLS=2)
    elif task == 'frame':
        print_frame(data_root, seed, frame=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        type=str,
    )
    parser.add_argument(
        "config",
        type=str,
    )
    parser.add_argument(
        "seed",
        type=int,
    )
    parser.add_argument(
        "--cache",
        action="store_true",
    )
    parser.add_argument(
        "--task",
        type=str,
        default='video',
        choices=['video', 'frame'],
        help="Task to perform: 'video' to generate video, 'frame' to print a specific frame's data",
    )
    args = parser.parse_args()
    main(args.task, args.name, args.config, args.seed, args.cache)