
from visualize_waypts import predict_diff_actions, visualize, load_models
from utils import get_norm_stats
import torch
import time
import json
import numpy as np

device = torch.device('cuda')

num_episodes=100
dataset_path = "/home/selamg/diffusion_plugging/Processed_Data_2/data"
dataset_dir = dataset_path

# weights_dir = \
# "/home/selamg/diffusion_plugging/checkpoints/2024-02-25_resnet18/resnet18_epoch3050_05:46:24_2024-02-26"
# "/media/selamg/DATA/diffusion_plugging_checkpoints/resnet18_epoch3500_14-41-09_2024-03-06_NOT_FXD20"

ckpt_dir = "/media/selamg/DATA/diffusion_plugging_checkpoints/"

# wt_name = "resnet18_epoch3500_11-29-28_2024-03-08_FXD_ABLATE_GEL"
# wt_name = "clip_epoch3500_11-21-45_2024-03-08_FXD_ABLATE_GEL"
# wt_name = "clip_epoch3500_03-00-06_2024-03-08_FXD_GEL_ONLY"
# wt_name = "resnet18_epoch3500_02-58-45_2024-03-08_FXD_GEL_ONLY"
wt_name = "resnet18_epoch3500_12-57-20_2024-03-08_FXD20"
# wt_name = "clip_epoch3500_12-41-13_2024-03-08_FXD_20"

weights_dir = ckpt_dir + wt_name

ABLATE_GEL = False
GEL_ONLY = False

pred_horizon = 20

camera_names = [1,2,3,4,5,6,"gelsight"]
if GEL_ONLY:
    camera_names = ["gelsight"]

# norm_stats = get_norm_stats(dataset_dir, num_episodes)
    
norm_stats_dir = "/home/selamg/diffusion_plugging/norm_stats_fixed.json"

with open(norm_stats_dir, 'r') as f:
    norm_stats = json.load(f)

for key in norm_stats:
    norm_stats[key] = np.array(norm_stats[key])

dataloader, model_dict = load_models(
    dataset_dir, weights_dir, norm_stats, camera_names, num_episodes,pred_horizon)

print("KEYS:", model_dict.keys())
for batch in dataloader:
    start = time.time()
    if ABLATE_GEL:
        del batch['gelsight']

    all_images,qpos,naction, gt = predict_diff_actions(batch, 
                                                    dataloader.dataset.action_qpos_normalize, 
                                                    model_dict, camera_names, 
                                                    device)
    
    # print(all_images[0].shape)
    # print(qpos.shape)
    end = time.time()
    print("action_shape:",naction.shape)
    print("actions:", naction)
    print("gt_shape:",gt.shape)
    

    print(f"{end-start:.3f}")
    # with 100 diffusion steps avg is 0.6-0.7s
    # with 10 steps avg is 0.1-0.15s
    visualize(all_images,qpos,naction,ground_truth=gt)
    # visualize(all_images,qpos,naction2,ground_truth=gt)
    # visualize(all_images, qpos,naction, ground_truth=naction2)


