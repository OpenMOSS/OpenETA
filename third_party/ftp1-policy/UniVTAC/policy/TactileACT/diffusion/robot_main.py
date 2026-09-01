from predict_robot_actions import create_nets, diffuse_robot
from robot_operation_old import PreprocessData
import h5py
import numpy as np
import cv2
import os

from utils import get_norm_stats

from visualization import visualize




EXPECTED_CAMERA_NAMES = ['1','2','3','4','5','6','gelsight'] 


print("CAMNAMES:", EXPECTED_CAMERA_NAMES)

# pretend to be a robot by loading a dataset
data_dir = "/home/selamg/diffusion_plugging/demo_data/episode_3.hdf5"
with h5py.File(data_dir, 'r') as root:
    qpos = root['/observations/position'][()]
    gall_imagest_actions = root['/goal_position'][()]
    all_gelsight_data = root['observations/gelsight/depth_strain_image'][()]
    num_episodes = root.attrs['num_timesteps']
    num_episodes = 50

    all_images = {}
    for cam in root.attrs['camera_names']:
        video_images = []
        video_path = os.path.join(os.path.dirname(data_dir), f'cam-{cam}.avi')
        cap = cv2.VideoCapture(video_path)
        for i in range(num_episodes):
            ret, frame = cap.read()
            if not ret:
                break
            video_images.append(frame)
        
        all_images[cam] = np.array(video_images)
        cap.release()


images = {key: all_images[key][i] for key in all_images}
gelsight_data = all_gelsight_data[i]

data_dir = "/home/selamg/diffusion_plugging/Processed_Data_2/data"
num_episodes = 100


norm_stats  = get_norm_stats(data_dir,num_episodes)

print(norm_stats)
input()
preprocess = PreprocessData(norm_stats, EXPECTED_CAMERA_NAMES)

image_data, qpos_data = preprocess.process_data(images, gelsight_data, qpos[i])


enc_type = 'clip' #valid is 'clip' or 'resnet18'
#note that weights dir for clip is hard coded inside create_nets
# which is inside predict_robot_actions.py

nets = create_nets(enc_type)

image_encoder  = nets['image_encoder']
gelsight_encoder = nets['gelsight_encoder']
noise_pred_net = nets['noise_pred_net']

device = 'cuda'
#prediction horizon is 8 time-steps
actions = diffuse_robot(qpos_data,image_data,
                         image_encoder,gelsight_encoder,
                         noise_pred_net,
                         pred_horizon=8,device=device)

print(actions.shape)


