from load_ACT import load_ACT
import h5py
import json
import numpy as np
from torchvision import transforms
from process_data_cage import MASK_VERTICIES, CROP_PARAMS, make_masks
from typing import Dict, List, Tuple
import cv2
import torch
import os
from frankapy import FrankaArm
from frankapy import FrankaConstants as FC
from robomail.motion import GotoPoseLive

from simple_gelsight import GelSightMultiprocessed, get_camera_id
from multiprocessed_cameras import MultiprocessedCameras

from utils import NormalizeActionQpos, NormalizeDeltaActionQpos
from rospy import Rate

from copy import deepcopy

def monitor_cameras(frames: Dict[str, np.ndarray], gelsight_frame: np.ndarray = None):
    # print('show cams')
    out_size = (900, 1800, 3)
    if gelsight_frame is not None:
        gelsight_frame = (visualize_gelsight_data(gelsight_frame)*255).astype(np.uint8)
    
    n_col = int(np.ceil(np.sqrt(len(frames)))) # 3
    n_row = int(np.ceil(len(frames) / n_col))

    # Create a grid of images
    tile_size = (int(out_size[0]/n_row), int(out_size[1]/n_col))
    grid = np.zeros(out_size, dtype=np.uint8)

    running_idx = 0
    for i, (name, frame) in enumerate(frames.items()):
        if name != 'gelsight':
            frame_cropped = frame[CROP_PARAMS[int(name)]['i']:CROP_PARAMS[int(name)]['i']+CROP_PARAMS[int(name)]['h'], 
                        CROP_PARAMS[int(name)]['j']:CROP_PARAMS[int(name)]['j']+CROP_PARAMS[int(name)]['w']].copy()
        if name == '6': # rotate the 6th camera
            frame_cropped = np.rot90(frame_cropped).copy()
        row = i // n_col
        col = i % n_col
        scale_factor = min(tile_size[0]/frame_cropped.shape[0], tile_size[1]/frame_cropped.shape[1])
        frame_cropped = cv2.resize(frame_cropped.copy(), (0, 0), fx=scale_factor, fy=scale_factor)
        if row == n_row - 1 and gelsight_frame is not None: # sqeeze the gelsight into the bottom row
            grid[row*tile_size[0]:row*tile_size[0]+frame_cropped.shape[0], 
                running_idx:running_idx+frame_cropped.shape[1]] = frame_cropped
            running_idx += frame_cropped.shape[1]
        else:
            grid[row*tile_size[0]:row*tile_size[0]+frame_cropped.shape[0], 
                col*tile_size[1]:col*tile_size[1]+frame_cropped.shape[1]] = frame_cropped
            
    if gelsight_frame is not None:
        scale_factor = min(tile_size[0]/gelsight_frame.shape[0], (grid.shape[1]-running_idx)/gelsight_frame.shape[1])
        gelsight_frame = cv2.resize(gelsight_frame, (0, 0), fx=scale_factor, fy=scale_factor)
        grid[-gelsight_frame.shape[0]:, -gelsight_frame.shape[1]:] = gelsight_frame
    
    cv2.imshow('cameras', grid)
    cv2.waitKey(1)

def visualize_gelsight_data(image):
    # Convert the image to LAB color space
    max_depth = 10
    max_strain = 30
    # Show all three using LAB color space
    image[:, :, 0] = np.clip(100*np.maximum(image[:, :, 0], 0)/max_depth, 0, 100)
    # normalized_depth = np.clip(100*(depth_image/depth_image.max()), 0, 100)
    image[:, :, 1:] = np.clip(128*(image[:, :, 1:]/max_strain), -128, 127)
    # return (image*np.array([2.5, 1, 1]) + np.array([0, 128, 128]))/255
    return cv2.cvtColor(image.astype(np.float32), cv2.COLOR_LAB2BGR)

image_size = (400, 480)

class PreprocessData:
    """Preprocesses the data for the ACT model. Behaves like the dataset class 
    used in the training loop, but does not inherit from torch.utils.data.Dataset."""

    def __init__(self, norm_stats, camera_names):
        self.image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        # self.normalizer = NormalizeActionQpos(norm_stats)
        self.normalizer = NormalizeDeltaActionQpos(norm_stats)
        self.gelsight_mean = norm_stats["gelsight_mean"]
        self.gelsight_std = norm_stats["gelsight_std"]
        self.masks = make_masks(image_size=image_size, verticies=MASK_VERTICIES)
        self.camera_names = camera_names

    def process_data(self, 
                     images: Dict[str, np.ndarray], 
                     gelsight: np.ndarray, 
                     qpos: np.ndarray) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        
        all_images = []
        for cam_name in self.camera_names:

            if cam_name == "gelsight":
                # process gelsight
                gelsight_data = (gelsight - self.gelsight_mean) / self.gelsight_std
                gelsight_data = torch.tensor(gelsight_data, dtype=torch.float32)
                gelsight_data = torch.einsum('h w c -> c h w', gelsight_data)
                all_images.append(gelsight_data.unsqueeze(0))

            elif cam_name == "blank":
                # add a blank image
                blank_image = np.zeros([image_size[0], image_size[1], 3], dtype=np.uint8)
                all_images.append(torch.tensor(blank_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0))

            elif cam_name in images.keys():
                # crop the images
                crop = CROP_PARAMS[int(cam_name)]

                # crop the image and resize
                image = images[cam_name]
                image = image[crop['i']:crop['i']+crop['h'], crop['j']:crop['j']+crop['w']]
                image = cv2.resize(image, (image_size[1], image_size[0]))


                #apply the masks
                if int(cam_name) in self.masks and self.masks[int(cam_name)] is not None:
                    image = cv2.bitwise_and(image, image, mask=self.masks[int(cam_name)])

                # convert to tensor and normalize
                image = torch.tensor(image, dtype=torch.float32)/255.0
                image = torch.einsum('h w c -> c h w', image) # change to c h w
                image = self.image_normalize(image)
                all_images.append(image.unsqueeze(0))

            else:
                raise ValueError(f"Camera name {cam_name} not found in images")
            
        # get rid of velocities
        if qpos.shape[0] == 7:
            qpos = np.concatenate([qpos[:3], qpos[6:]])
        qpos = self.normalizer.normalize_qpos(qpos)
        qpos_data = torch.from_numpy(qpos).float().unsqueeze(0)

        return all_images, qpos_data        

def visualize(images, qpos, actions, ground_truth=None):

    import matplotlib.pyplot as plt
    # Create a figure and axes
    fig = plt.figure(figsize=(10, 10), layout='tight')
    subfigs = fig.subfigures(1, 2, wspace=0.07)

    axs_left = subfigs[0].subplots(len(images), 1)
    if len(images) > 1:
        for i, image in enumerate(images):
            print(image.shape)
            axs_left[i].imshow(image)     
    else:
        axs_left.imshow(images[0])

    # Make a 3D scatter plot of the actions in the right subplot. Use cmaps to color the points based on the index
    c = np.arange(len(actions))
    ax2 = subfigs[1].add_subplot(111, projection='3d')
    # ax2.scatter(actions[:, 0], actions[:, 1], actions[:, 2], c='b', marker='o')
    sc = ax2.scatter(actions[:, 0], actions[:, 1], actions[:, 2], c=c, cmap='viridis', marker='x')
    if ground_truth is not None:
        ax2.scatter(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], c=np.arange(len(ground_truth)), cmap = 'viridis', marker='o')
    ax2.scatter(qpos[0], qpos[1], qpos[2], c='r', marker='o')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Actions and Qpos')
    cbar = fig.colorbar(sc, ax=ax2, label='Time', shrink=0.5)

    # Set the axis limits
    center = np.array([0.5, 0, 0.2])
    radius = 0.15
    ax2.set_xlim(center[0] - radius, center[0] + radius)
    ax2.set_ylim(center[1] - radius, center[1] + radius)
    ax2.set_zlim(center[2] - radius, center[2] + radius)

    plt.show()    

import threading
import queue

class AsyncInput:
    def __init__(self):
        self.input_queue = queue.Queue()
        self.thread = threading.Thread(target=self.get_input)
        self.thread.daemon = True
        self.thread.start()
        self.command = None

    def get_input(self):
        while True:
            self.command = input()
            self.input_queue.put(self.command)

    def get_latest_input(self):
        latest_input = self.input_queue.get()
        while not self.input_queue.empty():
            latest_input = self.input_queue.get()
        return latest_input
    
    def input(self, command):       
        # clear the cue
        while not self.input_queue.empty():
            self.input_queue.get()

        print(command)
        
        # return the responce
        return self.get_latest_input()
    
class syncInput:
    def __init__(self):
        self.command = None

    def input(self, command):
        self.command = input(command)
        return self.command


    

if __name__ == '__main__':
    print("starting")
    import time
    fa = FrankaArm()
    fa.reset_joints()
    print("resetting joints")
    fa.open_gripper()
    move_pose = FC.HOME_POSE
    # move_pose.translation = np.array([0.6, 0, 0.35])
    # fa.goto_pose(move_pose)
    default_impedances = np.array(FC.DEFAULT_TRANSLATIONAL_STIFFNESSES + FC.DEFAULT_ROTATIONAL_STIFFNESSES)
    new_impedances = np.copy(default_impedances)
    new_impedances[3:] = np.array([0.5, 2, 0.5])*new_impedances[3:] # reduce the rotational stiffnesses, default in gotopose live
    # new_impedances[:3] = 1.5*default_impedances[:3] # increase the translational stiffnesses
    new_impedances[:3] = np.array([1, 1, 1])*default_impedances[:3] # reduce the translational stiffnesse

    pose_controller = GotoPoseLive(cartesian_impedances=new_impedances.tolist(), step_size=0.05)


    # pose_controller = GotoPoseLive(step_size=0.05)
    pose_controller.set_goal_pose(move_pose)


    # model_path = "/home/abraham/TactileACT/data/final_trained_policies/fixed/pretrain_gel_only_20/policy_last.ckpt"
    # save_path = "data/final_trained_policies/fixed/pretrain_gel_only_20/run_data"
    # args_file = "data/final_trained_policies/fixed/pretrain_gel_only_20/args.json"

    # model_path = "/home/abraham/TactileACT/data/final_trained_policies/fixed/pretrain_both_20/policy_last.ckpt"
    # save_path = "/home/abraham/TactileACT/data/final_trained_policies/fixed/pretrain_both_20/run_data"
    # args_file = "/home/abraham/TactileACT/data/final_trained_policies/fixed/pretrain_both_20/args.json"

    # model_path = "/home/abraham/TactileACT/data/final_trained_policies/fixed/no_pretrain_both_20/policy_last.ckpt"
    # save_path = "/home/abraham/TactileACT/data/final_trained_policies/fixed/no_pretrain_both_20/run_data"
    # args_file = "/home/abraham/TactileACT/data/final_trained_policies/fixed/no_pretrain_both_20/args.json"

    # model_path = "/home/abraham/TactileACT/data/final_trained_policies/fixed/pretrain_vision_only_20/policy_last.ckpt"
    # save_path = "/home/abraham/TactileACT/data/final_trained_policies/fixed/pretrain_vision_only_20/run_data"
    # args_file = "/home/abraham/TactileACT/data/final_trained_policies/fixed/pretrain_vision_only_20/args.json"

    model_path = "/home/abraham/TactileACT/data/final_trained_policies/fixed/no_pretrain_vision_only_20/policy_last.ckpt"
    save_path = "/home/abraham/TactileACT/data/final_trained_policies/fixed/no_pretrain_vision_only_20/run_data"
    args_file = "/home/abraham/TactileACT/data/final_trained_policies/fixed/no_pretrain_vision_only_20/args.json"

    # model_path = "/home/abraham/TactileACT/data/fixed_scene/no_gelsight_pretrained/policy_epoch_3200_seed_0.ckpt"
    # save_path = "/home/abraham/TactileACT/data/fixed_scene/no_gelsight_pretrained/run_data"
    # args_file = "/home/abraham/TactileACT/data/fixed_scene/no_gelsight_pretrained/args.json"

    # model_path = "/home/abraham/TactileACT/data/fixed_scene/gelsight_only_fixed/policy_last.ckpt"
    # save_path = "/home/abraham/TactileACT/data/fixed_scene/gelsight_only_fixed/run_data"
    # args_file = "/home/abraham/TactileACT/data/fixed_scene/gelsight_only_fixed/args.json"

    # model_path = "/home/abraham/TactileACT/data/final_trained_policies/fixed/no_pretrain_gel_only_20/policy_last.ckpt"
    # save_path = "/home/abraham/TactileACT/data/final_trained_policies/fixed/no_pretrain_gel_only_20/run_data"
    # args_file = "/home/abraham/TactileACT/data/final_trained_policies/fixed/no_pretrain_gel_only_20/args.json"

    # model_path = "/home/abraham/TactileACT/data/final_trained_policies/fixed/pretrain_gel_only_20/policy_last.ckpt"
    # save_path = "/home/abraham/TactileACT/data/final_trained_policies/fixed/pretrain_gel_only_20/run_data"
    # args_file = "/home/abraham/TactileACT/data/final_trained_policies/fixed/pretrain_gel_only_20/args.json"

    # model_path = "/home/abraham/TactileACT/data/final_trained_policies/fixed/blank_image_20/policy_last.ckpt"
    # save_path = "/home/abraham/TactileACT/data/final_trained_policies/fixed/blank_image_20/run_data"
    # args_file = "/home/abraham/TactileACT/data/final_trained_policies/fixed/blank_image_20/args.json"

    # model_path = "/home/abraham/TactileACT/data/final_trained_policies/Non-Fixed/pretrain_both_20/policy_last.ckpt"
    # save_path = "/home/abraham/TactileACT/data/final_trained_policies/Non-Fixed/pretrain_both_20/run_data"
    # args_file = "/home/abraham/TactileACT/data/final_trained_policies/Non-Fixed/pretrain_both_20/args.json"

    # model_path = "/home/abraham/TactileACT/data/final_trained_policies/Non-Fixed/ACT/pretrained_both/policy_last.ckpt"
    # save_path = "/home/abraham/TactileACT/data/final_trained_policies/Non-Fixed/ACT/pretrained_both/run_data"
    # args_file = "/home/abraham/TactileACT/data/final_trained_policies/Non-Fixed/ACT/pretrained_both/args.json"


    """# pretend to be a robot by loading a dataset
    data_dir = "/mnt/ssd/abraham/Tactile_ACT_Data/camara_cage_1/run_0/episode_3/episode_3.hdf5"
    with h5py.File(data_dir, 'r') as root:
        qpos = root['/observations/position'][()]
        gt_actions = root['/goal_position'][()]
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
    """
    camera_id = get_camera_id('GelSight')
    gelsight = GelSightMultiprocessed(camera_id, use_gpu=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    args = json.load(open(args_file, 'r'))
    modified_args = args.copy()
    modified_args["gelsight_backbone_path"] = "none"
    modified_args["vision_backbone_path"] = "none"

    act = load_ACT(model_path, args_file, modified_args).to(device)
    act.eval()
    
    camera_nums = [1, 2, 3, 4, 5, 6]
    camera_sizes = [(1080, 1920), (1080, 1920), (1080, 1920), (1080, 1920), (1080, 1920), (800, 1280)]
    cameras = MultiprocessedCameras(camera_nums, camera_sizes, 30)
    min_gripper_width = 0.00375

    # get the norm stats from args_file, need to convert to numpy array

    norm_stats = {k: np.array(v) for k, v in args['norm_stats'].items()}
    print(norm_stats)

    n_discard = 0
    horizon = args['chunk_size'] - n_discard

    # create the fake dataset
    preprocess = PreprocessData(norm_stats, args['camera_names'])
    
    num_episodes = 1000
    grip_closed = False

    temporal_ensemble = True
    ADD_NOISE = True
    noise_std = 0.0025
    noise_mean = 0
    K = 0.25
    offset = np.array([0, 0, 0, 0]) 
    # offset = np.array([0, 0, -0.00285, 0])

    async_input = AsyncInput()
    # async_input = syncInput()

    SAVE_VIDEO = True

    run = 0
    rate = Rate(10)
    move_pose = FC.HOME_POSE
    move_pose.translation = np.array([0.6, 0, 0.35])
    fa.goto_pose(move_pose)
    while True:
        run += 1
        print(f'starting run {run}')
        fa.open_gripper()
        move_pose.translation = np.array([0.6, 0, 0.35])
        async_input.input("Press enter to continue")

        while np.linalg.norm(fa.get_pose().translation - move_pose.translation) > 0.01:
            pose_controller.step(goal_pose = move_pose)
            time.sleep(0.05)

        async_input.input("Press enter to continue")
        action_history = np.zeros([num_episodes + horizon, num_episodes, 4]) # prediction_time, time_preiction_was_made, action
        confidences = []

        run_images = []
        for camera_name in args['camera_names']:
            if camera_name != "gelsight":
                run_images.append(np.zeros([num_episodes, image_size[0], image_size[1], 3], dtype=np.float32))
            else:
                run_images.append(np.zeros([num_episodes, 240, 320, 3], dtype=np.float32))

        integral_term = np.zeros(3)
        moved_gripper = False
        last_position = np.copy(move_pose.translation)

        gelsight_strains = []
        gelsight_max_strains = []
        gelsight_data_rec = []
        total_timesteps = 0
        start_time = time.time()
        for i in range(num_episodes):
            total_timesteps += 1
            print(i)
            # images = {key: all_images[key][i] for key in all_images}
            # gelsight_data = all_gelsight_data[i]

            robo_data = fa.get_robot_state()
            current_pose = deepcopy(robo_data['pose']*fa._tool_delta_pose)
            # print('current_rot', current_pose.euler_angles)

            images = cameras.get_next_frames()
            frame, marker_data, depth, strain_x, strain_y = gelsight.get_next_frame()
            gelsight_data = np.stack([depth, strain_x, strain_y], axis=-1)

            gelsight_strains.append(np.mean(np.abs(gelsight_data), axis=(0, 1)))
            gelsight_max_strains.append(np.max(np.linalg.norm(gelsight_data[1:], axis=-1)))
            gelsight_data_rec.append(gelsight_data)
            # print("gelsight_strains", gelsight_strains[-1])

            # show the images
            monitor_cameras(deepcopy(images), np.copy(gelsight_data))

            
            finger_width = robo_data['gripper_width']
            qpos = np.zeros(4)
            qpos[:3] = current_pose.translation
            qpos[3] = finger_width
            # print("qpos", qpos)
            image_data, qpos_data = preprocess.process_data(images, gelsight_data, qpos)

            # get the action from the model
            qpos_data = qpos_data.to(device)
            image_data = [img.to(device) for img in image_data]
            start_time = time.time()
            # qpos_data = torch.zeros_like(qpos_data).to(device)
            with torch.no_grad():
                deltas:torch.Tensor = act(qpos_data, image_data)
            print('act inference time', time.time() - start_time)

            # unnormalize the actions and qpos 
            deltas = deltas.squeeze().detach().cpu().numpy()
            unnormalized_deltas = preprocess.normalizer.unnormalize_delta(deltas)
            # print("unnormalized_deltas", unnormalized_deltas)

            all_actions = unnormalized_deltas + qpos + offset
            all_actions[:, 3] = np.clip(unnormalized_deltas[:, 3], 0, 0.08) # use output grip width and clip it

            action_history[i:i+horizon, i] = all_actions[n_discard:, :] 

            if temporal_ensemble:
                ensembled_action = np.zeros(4)
                total_weight = 0
                time_step_actions = []
            
            
                confidences.insert(0, 1) # add the confidence to the front of the list (this time step)
                if len(confidences) > horizon:
                    confidences.pop() # remove the last confidence from the list 
                
                for t in range(min(i+1, horizon)):
                    ensembled_action += confidences[t]*action_history[i, i-t, :]
                    total_weight += confidences[t]
                    confidences[t] *= np.exp(-K) # update the confidence
                    time_step_actions.append(action_history[i, i-t, :])

                time_step_actions = np.array(time_step_actions)
                ensembled_action /= total_weight
                # print("time_step_actions", time_step_actions)
                # print('confidences', confidences)
                print('current_delta', unnormalized_deltas[0])
                print("current_pose", current_pose.translation)
                print("ensembled_action", ensembled_action)
                print('ensembled_delta', ensembled_action[:3] - qpos[:3])
                # print('current_action', all_actions[0])
                print('abs mean gelsight', np.mean(gelsight_data, axis=(0, 1)))
                print("K", K)

            # visualize the data
            vis_images = [image_data[j].squeeze().detach().cpu().numpy().transpose(1, 2, 0) for j in range(len(image_data))]
            
            for j, vis_image in enumerate(vis_images):
                run_images[j][i] = vis_image.copy()

            # move the robot:
            if np.linalg.norm(last_position - current_pose.translation) < 0.0025 and not moved_gripper: # if the robot hasn't moved much, add to the integral term (to deal with friction)
                integral_term += (ensembled_action[:3] - current_pose.translation)*0.25
            else:
                integral_term = np.zeros(3)

            integral_term = np.clip(integral_term, -0.015, 0.015)
            # integral_term = np.zeros(3)
            print("integral_term", integral_term)
            
            last_position = np.copy(current_pose.translation)

            integral_term[2] = max(integral_term[2], 0) # don't wind up the z term in the negative direction (contact with the table)

            

            move_pose = FC.HOME_POSE
            if ADD_NOISE:
                move_pose.translation = ensembled_action[:3] + integral_term + np.random.normal(noise_mean, noise_std, 3)
            else:
                move_pose.translation = ensembled_action[:3] + integral_term
            # move_pose.translation = ensembled_action[:3] 
            
            pose_controller.step(move_pose, current_pose)
            grip_command = ensembled_action[3]
            grip_command = np.clip(grip_command, 0, 0.08)
            start_command_time = time.time()
            if grip_command <= min_gripper_width:
                if not grip_closed:
                    grip_closed = True
                    fa.goto_gripper(min_gripper_width)
                    moved_gripper = True
                    print("closing gripper")
                else:
                    moved_gripper = False
            else:
                grip_closed = False
                fa.goto_gripper(grip_command, block=False, speed=0.15, force = 10)
                moved_gripper = True
                print("gripper width", grip_command)

            print('command time', time.time() - start_command_time)

            command = async_input.command
            if i >= 300:
                while command != 'q' and command != 's':
                    command = async_input.input('run timed out. Press q to quit, or s to save')
                break
            if command == 'q' or command == 's':
                break
            elif command == 'v':
                visualize(vis_images, qpos, all_actions, time_step_actions)

            # if i % 5 == 0:
            #     command = input("Press enter to continue, v to visualize, or q to quit, or s to save")
            #     if command == 'q' or command == 's':
            #         break
            #     elif command == 'v':
            #         visualize(vis_images, qpos, all_actions, time_step_actions)
                
            rate.sleep()
            print('total time', time.time() - start_time)


        if command == 's':
            # create a new folder for the run
            existing_folders = [int(f.split('_')[-1]) for f in os.listdir(save_path) if f.startswith('run')]
            new_folder = max(existing_folders, default=0) + 1
            os.makedirs(os.path.join(save_path, f'run_{new_folder}'))

            print(f'saving run {new_folder}')

            # save the data
            gelsight_strains = np.array(gelsight_strains)
            gelsight_max_strains = np.array(gelsight_max_strains)
            gelsight_data_rec = np.array(gelsight_data_rec)
            np.save(os.path.join(save_path, f'run_{new_folder}', 'gelsight_strains.npy'), gelsight_strains)
            np.save(os.path.join(save_path, f'run_{new_folder}', 'gelsight_max_strains.npy'), gelsight_max_strains)
            np.save(os.path.join(save_path, f'run_{new_folder}', 'gelsight_data.npy'), gelsight_data_rec)

            while True:
                was_successful = async_input.input('Was the run successful? (y/n)')
                if was_successful == 'y' or was_successful == 'n':
                    break


            run_stats = {"num_timesteps": total_timesteps, 
                         "horizon": horizon, 
                         "temporal_ensemble": temporal_ensemble, 
                         "ADD_NOISE": ADD_NOISE, 
                         "noise_std": noise_std,
                         "noise_mean": noise_mean,
                         "K": K, 
                         "was_successful": was_successful,
                         "training_args": args}
            
            with open(os.path.join(save_path, f'run_{new_folder}', 'run_stats.json'), 'w') as f:
                json.dump(run_stats, f)
            
            if SAVE_VIDEO:
                # make a video folder for the run
                os.makedirs(os.path.join(save_path, f'run_{new_folder}', 'videos'))
                run_video_path = os.path.join(save_path, f'run_{new_folder}', 'videos')

                # dump npy file:
                # start_time = time.time()
                # for cam_num, save_cam in enumerate(args['camera_names']):
                #     np.save(os.path.join(save_path, f'run_{new_folder}/videos', f'cam_{save_cam}.npy'), run_images[cam_num])
                # print('saving npy time', time.time() - start_time)

                print("saving video")
                fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
                # get the time for the video

                for cam_num, save_cam in enumerate(args['camera_names']):
                    print('saving', save_cam)
                    cam_size = run_images[cam_num][0].shape[:2][::-1]
                    out_mp4 = cv2.VideoWriter(os.path.join(run_video_path, f'cam_{save_cam}.mp4'), fourcc_mp4, 10.0, cam_size)
                    for images in run_images[cam_num]:
                        # unnormalize the image (standard resnet normalization)
                        if save_cam != "gelsight":
                            image = images * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                            # out_mp4.write((image*255).astype(np.uint8))
                            out_mp4.write((image*255).astype(np.uint8))
                        else:
                            image = images*norm_stats['gelsight_std'] + norm_stats['gelsight_mean']
                            out_mp4.write((visualize_gelsight_data(images)*255).astype(np.uint8))

                    out_mp4.release()
                
            run_images = []

        if async_input.input('Press d to quit, or enter to continue') == 'd':
            break

        print('next run')
    print("done")
