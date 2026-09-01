import os
import h5py
import numpy as np
import shutil
import cv2
from matplotlib import pyplot as plt
from typing import List, Dict
from multiprocessing import Pool
import json
from utils import gelsight_norm_stats

CROP_PARAMS = {
    1: {'i': 0, 'j': 312, 'h': 1080, 'w': 1296, 'size': (1080, 1920)},
    2: {'i': 108, 'j': 775, 'h': 755, 'w': 906, 'size': (1080, 1920)},
    3: {'i': 324, 'j': 768, 'h': 595, 'w': 714, 'size': (1080, 1920)},
    4: {'i': 360, 'j': 648, 'h': 560, 'w': 672, 'size': (1080, 1920)},
    5: {'i': 150, 'j': 350, 'h': 595, 'w': 714, 'size': (1080, 1920)},
    6: {'i': 212, 'j': 425, 'h': 375, 'w': 450, 'size': (800, 1200)},
}

MASK_VERTICIES = {5: [[0, 0.77], [0.0625, 1], [0, 1]],
                  6: [[0, 0.79], [0.0875, 1], [0, 1]]}

# mask out background of images (for where curtain was taken off)
def make_masks(image_size, verticies):
    masks = {1: None, 2: None, 3: None, 4: None, 5: None, 6: None}
    for cam in verticies:
        mask = np.ones((image_size[0], image_size[1]), dtype=np.uint8)
        pts = np.array(verticies[cam])*image_size
        pts = np.fliplr(pts.astype(np.int32)) # cv2 requires the points to be in the format (x, y), not (i, j)
        cv2.fillPoly(mask, [pts], 0)
        masks[cam] = mask
    return masks
        
def uncompress_data(source_folder, save_path, image_size = [400, 480], masks: Dict[str, np.ndarray] = {}, use_rot = False, gelsight_delay = 2):
    # First, copy the hdf files to the save_path
    # Find the hdf5 files in the source folder
    h5py_files = []
    for file in os.listdir(source_folder):
        if file.endswith('.hdf5'):
            h5py_files.append(file)

    assert len(h5py_files) == 1, f"Expected 1 hdf5 file, but found {len(h5py_files)}"


    # open the new hdf5 file
    with h5py.File(os.path.join(source_folder, h5py_files[0]), 'r') as old:
        with h5py.File(save_path, 'w') as new:
            # copy the attributes
            new.attrs['camera_names'] = old.attrs['camera_names']
            new.attrs['gelsight_height'] = old.attrs['gelsight_height']
            new.attrs['gelsight_width'] = old.attrs['gelsight_width']
            new.attrs['image_height'] = image_size[0]
            new.attrs['image_width'] = image_size[1]

            new.attrs['num_timesteps'] = old.attrs['num_timesteps'] - gelsight_delay
            new.attrs['sim'] = old.attrs['sim']
            new.attrs['use_gelsight'] = old.attrs['use_gelsight']


            position = old['observations/position'][:-gelsight_delay]
            velocity = old['observations/velocity'][:-gelsight_delay]
            action = old['goal_position'][:-gelsight_delay]
            if not use_rot:
                new.attrs['position_dim'] = 4
                new.attrs['velocity_dim'] = 4
                new.attrs['position_doc'] = "x, y, z, gripper"
                new.attrs['velocity_doc'] = "x_dot, y_dot, z_dot, gripper_vel"
                position = np.delete(position, [3, 4, 5], axis=1)
                velocity = np.delete(velocity, [3, 4, 5], axis=1)
                action = np.delete(action, [3, 4, 5], axis=1)
            else:
                new.attrs['position_dim'] = old.attrs['position_dim']
                new.attrs['velocity_dim'] = old.attrs['velocity_dim']
                new.attrs['position_doc'] = old.attrs['position_doc']
                new.attrs['velocity_doc'] = old.attrs['velocity_doc']
            


            # copy the datasets
            new.create_dataset('action', data=action, chunks=(1, action.shape[1]))
            
            obs = new.create_group('observations')
            obs.create_dataset('qpos', data=position, chunks=(1, position.shape[1]))
            obs.create_dataset('qvel', data=velocity, chunks=(1, velocity.shape[1]))

            # copy the gel sight data
            if old.attrs['use_gelsight']:
                gelsight = obs.create_group('gelsight')
                depth_strain = old['observations/gelsight/depth_strain_image'][gelsight_delay:]
                marker_data = old['observations/gelsight/marker_data'][gelsight_delay:]
                raw_image = old['observations/gelsight/raw_image'][gelsight_delay:]

                gelsight.create_dataset('depth_strain_image', data=depth_strain, 
                                        chunks=(1, depth_strain.shape[1], depth_strain.shape[2], 3))
                gelsight.create_dataset('marker_data', data=marker_data, 
                                        chunks=(1, marker_data.shape[1], marker_data.shape[2]))
                gelsight.create_dataset('raw_image', data=raw_image,
                                        chunks=(1, raw_image.shape[1], raw_image.shape[2], 3))
                
            
            # uncompress the images
            # save each camera image
            image_group = obs.create_group('images')
            for cam_name in old.attrs['camera_names']:
                # open the video file
                video_path = os.path.join(source_folder, f'cam-{cam_name}.avi')
                cap = cv2.VideoCapture(video_path)
                # get the number of frames
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                assert num_frames == old.attrs['num_timesteps'], f"Number of frames in video {num_frames} does not match number of timesteps in hdf5 file {old.attrs['num_timesteps']}"

                crop = CROP_PARAMS[int(cam_name)]
                images = np.empty((num_frames, image_size[0], image_size[1], 3), dtype=np.uint8)

                # loop through the frames and save them in the hdf5 file
                for i in range(num_frames):
                    ret, frame = cap.read()
                    # crop the frame
                    frame = frame[crop['i']:crop['i']+crop['h'], crop['j']:crop['j']+crop['w']]
                    # resize the frame and save
                    frame = cv2.resize(frame, (image_size[1], image_size[0]))
                    # apply the mask 
                    if int(cam_name) in masks and masks[int(cam_name)] is not None:
                        frame = cv2.bitwise_and(frame, frame, mask=masks[int(cam_name)])
                    images[i] = frame

                cap.release()
                
                # save the images in the hdf5 file
                image_group.create_dataset(name=f'{cam_name}', dtype='uint8', 
                                        chunks=(1, image_size[0], image_size[1], 3),
                                        data=images[:-gelsight_delay])
                
            
        



    
    # # Copy the hdf5 file to the save_path
    # shutil.copy(os.path.join(source_folder, h5py_files[0]), save_path)

    # # Open the hdf5 file
    # with h5py.File(save_path, 'a') as hdf5_file:

    #     # rename the datasets to be in the format that the ACT dataloader expects
    #     if 'observations/position' in hdf5_file:
    #         hdf5_file.move('observations/position', 'observations/qpos')
    #     else:
    #         print(f'No position dataset in file {file}')
    #     if 'observations/velocity' in hdf5_file:
    #         hdf5_file.move('observations/velocity', 'observations/qvel')
    #     else:
    #         print(f'No velocity dataset in file {file}')
    #     if '/goal_position' in hdf5_file:
    #         hdf5_file.move('/goal_position', '/action')

    #     # if the data is not using rotation, removed 3:6 from qpos, action, and qvel
    #     if not use_rot:
    #         for key in ['observations/qpos', 'observations/qvel', 'action']:
    #             data = hdf5_file[key][()]
    #             # print("before delte:", data.shape)
    #             data = np.delete(data, [3, 4, 5], axis=1)
    #             # print("after delte:", data.shape)
    #             del hdf5_file[key]
    #             hdf5_file.create_dataset(name=key, data=data, chunks=(1, data.shape[1]))
            
    #         # change position_dim and velocity_dim attrs to 4
    #         hdf5_file.attrs['position_dim'] = 4
    #         hdf5_file.attrs['velocity_dim'] = 4

    #     # delete image_sizes atribute:
    #     del hdf5_file.attrs['image_sizes']

    #     # save the image height and width
    #     hdf5_file.attrs['image_height'] = image_size[0]
    #     hdf5_file.attrs['image_width'] = image_size[1]

    #     cam_names = hdf5_file.attrs['camera_names']
    #     # save each camera image
    #     for cam_name in cam_names:
    #         # open the video file
    #         video_path = os.path.join(source_folder, f'cam-{cam_name}.avi')
    #         cap = cv2.VideoCapture(video_path)
    #         # get the number of frames
    #         num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #         assert num_frames == hdf5_file.attrs['num_timesteps'], f"Number of frames in video {num_frames} does not match number of timesteps in hdf5 file {hdf5_file.attrs['num_timesteps']}"

    #         crop = CROP_PARAMS[int(cam_name)]
    #         images = np.empty((num_frames, image_size[0], image_size[1], 3), dtype=np.uint8)

    #         # loop through the frames and save them in the hdf5 file
    #         for i in range(num_frames):
    #             ret, frame = cap.read()
    #             # crop the frame
    #             frame = frame[crop['i']:crop['i']+crop['h'], crop['j']:crop['j']+crop['w']]
    #             # resize the frame and save
    #             frame = cv2.resize(frame, (image_size[1], image_size[0]))
    #             # apply the mask 
    #             if int(cam_name) in masks and masks[int(cam_name)] is not None:
    #                 frame = cv2.bitwise_and(frame, frame, mask=masks[int(cam_name)])
    #             images[i] = frame

    #         cap.release()
            
    #         # save the images in the hdf5 file
    #         hdf5_file.create_dataset(name=f'observations/images/{cam_name}', 
    #                                  dtype='uint8', 
    #                                  chunks=(1, image_size[0], image_size[1], 3),
    #                                  data=images)
            
        # # there is a dely in the gelsight video, so we will resave the gelsight images earlier in the hdf5 file
        # hdf5_file['observations/gelsight/depth_strain_image'][()] = hdf5_file['observations/gelsight/depth_strain_image'][gelsight_delay:]
        # hdf5_file['observations/gelsight/marker_data'][()] = hdf5_file['observations/gelsight/marker_data'][gelsight_delay:]
        # hdf5_file['observations/gelsight/raw_image'][()] = hdf5_file['observations/gelsight/raw_image'][gelsight_delay:]
        

            
def process_folder(source_folders, save_folder, image_size = [400, 480], masks = {}):
    # find all the episodes in the source folders recursively
    h5py_files = []
    for source_folder in source_folders:
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                if file.endswith('.hdf5'):
                    h5py_files.append(os.path.join(root, file))

    save_paths = []
    episode_folders = []
    for i, h5py_file in enumerate(h5py_files):

        save_paths.append(os.path.join(save_folder, f'episode_{i}.hdf5'))
        episode_folders.append(os.path.dirname(h5py_file)) # the episode folder will be the parent of the h5py file
    
    # uncompress the data, multiprocessed
    with Pool() as p:
        p.starmap(uncompress_data, zip(episode_folders, save_paths, [image_size]*len(save_paths), [masks]*len(save_paths)))

def save_norm_stats(save_folder):
    # get the number of episodes
    # find all the episodes in the source folder recursively
    h5py_files = []
    for root, dirs, files in os.walk(save_folder):
        for file in files:
            if file.endswith('.hdf5'):
                h5py_files.append(os.path.join(root, file))

    num_episodes = len(h5py_files)

    # get norm stats and save them
    gelsight_mean, gelsight_std = gelsight_norm_stats(save_folder, num_episodes)
    # convert form numpy to list for json serialization
    stats = {'gelsight_mean': gelsight_mean.tolist(), 'gelsight_std': gelsight_std.tolist()}
    with open(os.path.join(save_folder, 'gelsight_norm_stats.json'), 'w') as f:
        json.dump(stats, f)

            
if __name__ == "__main__":
    image_size = [400, 480]
    masks = make_masks(image_size, MASK_VERTICIES)

    source_folders = ['/home/aigeorge/research/TactileACT/data/original/camara_cage_2_new_mount/',
                    '/home/aigeorge/research/TactileACT/data/original/camara_cage_3/',
                    '/home/aigeorge/research/TactileACT/data/original/camara_cage_5_crack_gel/',
                    '/home/aigeorge/research/TactileACT/data/original/camara_cage_6_new_gel/']
    
    save_folder = '/home/aigeorge/research/TactileACT/data/camera_cage_new_mount/data'

    # process_folder(source_folders, save_folder, image_size, masks)

    save_norm_stats(save_folder)

    # source_file = '/home/aigeorge/research/TactileACT/data/original/camara_cage_1/run_0/episode_3'
    # save_file = '/home/aigeorge/research/TactileACT/test.hdf5'
    # uncompress_data(source_file, save_file, image_size, masks, use_rot=False)



