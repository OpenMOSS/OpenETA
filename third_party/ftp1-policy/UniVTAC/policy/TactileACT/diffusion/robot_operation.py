import numpy as np
from torchvision import transforms
from process_data_cage import MASK_VERTICIES, CROP_PARAMS, make_masks
from typing import Dict, List, Tuple
import cv2
import torch
import os
from dataset import NormalizeDiffusionActionQpos


#from rospy import Rate
from copy import deepcopy
import time
import json
# from rospy import Rate

def monitor_cameras(frames: Dict[str, np.ndarray], gelsight_frame: np.ndarray = None):
    print('show cams')
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
            frame = frame[CROP_PARAMS[int(name)]['i']:CROP_PARAMS[int(name)]['i']+CROP_PARAMS[int(name)]['h'], 
                        CROP_PARAMS[int(name)]['j']:CROP_PARAMS[int(name)]['j']+CROP_PARAMS[int(name)]['w']]
        if name == '6': # rotate the 6th camera
            frame = np.rot90(frame).copy()
        row = i // n_col
        col = i % n_col
        scale_factor = min(tile_size[0]/frame.shape[0], tile_size[1]/frame.shape[1])
        frame = cv2.resize(frame.copy(), (0, 0), fx=scale_factor, fy=scale_factor)
        if row == n_row - 1 and gelsight_frame is not None: # sqeeze the gelsight into the bottom row
            grid[row*tile_size[0]:row*tile_size[0]+frame.shape[0], 
                running_idx:running_idx+frame.shape[1]] = frame
            running_idx += frame.shape[1]
        else:
            grid[row*tile_size[0]:row*tile_size[0]+frame.shape[0], 
                col*tile_size[1]:col*tile_size[1]+frame.shape[1]] = frame
            
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
    return cv2.cvtColor(image.astype(np.float32), cv2.COLOR_LAB2BGR)

image_size = (400, 480)

class PreprocessData:
    """Preprocesses the data for the ACT model. Behaves like the dataset class 
    used in the training loop, but does not inherit from torch.utils.data.Dataset."""

    def __init__(self, norm_stats, camera_names):
        self.image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        # self.normalizer = NormalizeActionQpos(norm_stats)
        self.normalizer = NormalizeDiffusionActionQpos(norm_stats)
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
        qpos, _ = self.normalizer(qpos, qpos)
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
    

if __name__ == '__main__':

    use_real_robot = False
    if use_real_robot:
        from frankapy import FrankaArm
        from frankapy import FrankaConstants as FC
        from robomail.motion import GotoPoseLive
        from rospy import rate

        from simple_gelsight import GelSightMultiprocessed, get_camera_id
        from multiprocessed_cameras import MultiprocessedCameras
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
        new_impedances[3:] = new_impedances[3:]*0.5 # reduce the rotational stiffnesses, default in gotopose live
        # new_impedances[:3] = 1.5*default_impedances[:3] # increase the translational stiffnesses
        new_impedances[:3] = default_impedances[:3] # reduce the translational stiffnesse

        pose_controller = GotoPoseLive(cartesian_impedances=new_impedances.tolist(), step_size=0.05)
        pose_controller.set_goal_pose(move_pose)

        save_video_path = "data/videos"
        
        
        camera_id = get_camera_id('GelSight')
        gelsight = GelSightMultiprocessed(camera_id, use_gpu=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device {device}")
        
        camera_nums = [1, 2, 3, 4, 5, 6]
        camera_sizes = [(1080, 1920), (1080, 1920), (1080, 1920), (1080, 1920), (1080, 1920), (800, 1280)]
        cameras = MultiprocessedCameras(camera_nums, camera_sizes, 30)
        min_gripper_width = 0.004

    else:
        import h5py
        data_dir = "/media/selamg/DATA/new_notfixed_visualization_data/episode_1/episode_1.hdf5"
        # data_dir = "/home/abraham/GelSightTeleopDataCollection/ssd/camara_cage_7_nonfixed/run_4/episode_17/episode_17.hdf5"
        with h5py.File(data_dir, 'r') as root:
            all_qpos_7 = root['/observations/position'][()]
            all_qpos = np.empty([all_qpos_7.shape[0], 4])
            all_qpos[:, :3] = all_qpos_7[:, :3]
            all_qpos[:, 3] = all_qpos_7[:, 6]
            gt_actions = root['/goal_position'][()]
            all_gelsight_data = root['observations/gelsight/depth_strain_image'][()]
            num_episodes = root.attrs['num_timesteps']
            #num_episodes = 50

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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from predict_robot_actions import diffuse_robot

    # weights_dir = "/home/abraham/diffusion_plugging/data/fixed/resnet_H32_both/FXD32_resnet18_epoch3050_19-56-19_2024-03-01"
    # norm_stats_dir = "/home/abraham/diffusion_plugging/norm_stats_fixed.json"

    weights_dir = "/media/selamg/DATA/diffusion_plugging_checkpoints/resnet18_epoch3500_01-47-26_2024-03-14_NOTFXD_ABLATE_GEL"
    norm_stats_dir = "/home/selamg/diffusion_plugging/norm_stats_not_fixed.json"


    with open(norm_stats_dir, 'r') as f:
        norm_stats = json.load(f)

    # convert the norm stats to numpy arrays
    for key in norm_stats:
        norm_stats[key] = np.array(norm_stats[key])

    # create the fake dataset
    # EXPECTED_CAMERA_NAMES = ['1','2','3','4','5','6','gelsight'] 
    
    # for gel only:
    # EXPECTED_CAMERA_NAMES = ['gelsight']

    # for images only:
    EXPECTED_CAMERA_NAMES = ['1','2','3','4','5','6']

    preprocess = PreprocessData(norm_stats, EXPECTED_CAMERA_NAMES)

    print('start load')
    model_dict = torch.load(weights_dir, map_location=device)
    print('finish load')
    
    num_episodes = 1000
    grip_closed = False

    ADD_NOISE = True

    # offset = np.array([0, 0, -0.01])

    run = 0
    if use_real_robot:
        rate = Rate(10)
    while True:
        run += 1
        print(f'starting run {run}')
        # noise = OUNoise(3, theta=0.1, sigma=0.0005)
        if use_real_robot:
            fa.open_gripper()
            move_pose = FC.HOME_POSE
            move_pose.translation = np.array([0.6, 0, 0.35])

            print(fa.get_pose().translation)
            input("Press enter to continue")

            while np.linalg.norm(fa.get_pose().translation - move_pose.translation) > 0.02:
                pose_controller.step(goal_pose = move_pose)
                time.sleep(0.05)
                print('moving to home')

            input("Press enter to continue")

            run_images = []
            integral_term = np.zeros(3)
            moved_gripper = False
            last_position = np.copy(move_pose.translation)

        for i in range(num_episodes):
            print(i)
            # images = {key: all_images[key][i] for key in all_images}
            # gelsight_data = all_gelsight_data[i]

            if use_real_robot:
                images = cameras.get_next_frames()
                frame, marker_data, depth, strain_x, strain_y = gelsight.get_next_frame()
                gelsight_data = np.stack([depth, strain_x, strain_y], axis=-1)
            
            else:
                images = {key: all_images[key][i] for key in all_images}
                gelsight_data = all_gelsight_data[i]


            # show the images
            monitor_cameras(images, np.copy(gelsight_data))

            if use_real_robot:
                robo_data = fa.get_robot_state()
                current_pose = robo_data['pose']*fa._tool_delta_pose
                
                # cur_joints = robo_data['joints']
                # cur_vel = robo_data['joint_velocities']
                finger_width = robo_data['gripper_width']
                qpos = np.zeros(4)
                qpos[:3] = current_pose.translation
                qpos[3] = finger_width
                print("qpos", qpos)
            else:
                qpos = all_qpos[i]
                current_pose = None

            image_data, qpos_data = preprocess.process_data(images, gelsight_data, qpos)

            # get the action from the model
            qpos_data = qpos_data.to(device)
            image_data = [img.to(device) for img in image_data]
            for im in image_data:
                print(im.shape)
            start = time.time()
            # For gel only, image data should be of form: 
            norm_actions = diffuse_robot(qpos_data,image_data,EXPECTED_CAMERA_NAMES,model_dict,
                         pred_horizon=20,device=device)

            print('norm_actions', norm_actions)
            end = time.time()
            print('inference time', end-start)
            _, actions = preprocess.normalizer.unnormalize(qpos, norm_actions)

            actions = actions.squeeze().detach().cpu().numpy()

            # visualize the data
            vis_images = [image_data[j].squeeze().detach().cpu().numpy().transpose(1, 2, 0) for j in range(len(image_data))]

            if use_real_robot:
                run_images.append(vis_images)

            if not use_real_robot:
                if i %10 == 0:
                    visualize(vis_images, qpos, actions, ground_truth=gt_actions[i:i+len(actions)])
                continue

            command = input("Press enter to continue, v to visualize, or q to quit, or s to save")
            if command == 'q' or command == 's':
                break
            elif command == 'v':
                visualize(vis_images, qpos, actions)

            print('actions', actions)
            print('current pose', current_pose.translation)
            for move_action in actions[:8]:
                print("move_action", move_action)

                # move the robot:
                if np.linalg.norm(last_position - current_pose.translation) < 0.0025 and not moved_gripper: # if the robot hasn't moved much, add to the integral term (to deal with friction)
                    integral_term += (move_action[:3] - current_pose.translation)*0.25
                else:
                    integral_term = np.zeros(3)
                print("integral_term", integral_term)
                last_position = np.copy(current_pose.translation)

                move_pose = FC.HOME_POSE
                if ADD_NOISE:
                    move_pose.translation = move_action[:3] + integral_term + np.random.normal(0, 0.0025, 3)
                else:
                    move_pose.translation = move_action[:3] + integral_term
                # move_pose.translation = ensembled_action[:3] 
                
                current_pose = fa.get_pose()
                pose_controller.step(move_pose, current_pose)
                print("moving to", move_pose.translation)
                grip_command = move_action[3]
                grip_command = np.clip(grip_command, 0, 0.08)
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

                rate.sleep()



        if command == 's':
            # gelsight.close()
            # cameras.close()
            # save the video
            print("saving video")
            fourcc_avi = cv2.VideoWriter_fourcc(*'HFYU')
            fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
            # get the time for the video
            current_time = time.strftime("%Y%m%d-%H%M%S")
            os.makedirs(os.path.join(save_video_path, f'run_{current_time}'))
            run_video_path = os.path.join(save_video_path, f'run_{current_time}')
            for cam_num, save_cam in enumerate(EXPECTED_CAMERA_NAMES):
                print('saving', save_cam)
                cam_size = run_images[0][cam_num].shape[:2][::-1]
                out_avi = cv2.VideoWriter(os.path.join(run_video_path, f'cam_{save_cam}.avi'), fourcc_avi, 10.0, cam_size)
                out_mp4 = cv2.VideoWriter(os.path.join(run_video_path, f'cam_{save_cam}.mp4'), fourcc_mp4, 10.0, cam_size)
                for images in run_images:
                    # unnormalize the image (standard resnet normalization)
                    if save_cam != "gelsight":
                        image = images[cam_num] * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                        out_avi.write((image*255).astype(np.uint8))
                        out_mp4.write((image*255).astype(np.uint8))
                    else:
                        image = images[cam_num]*norm_stats['gelsight_std'] + norm_stats['gelsight_mean']
                        out_avi.write((visualize_gelsight_data(image)*255).astype(np.uint8))
                        out_mp4.write((visualize_gelsight_data(image)*255).astype(np.uint8))

                out_avi.release()
                out_mp4.release()
            
            run_images = []

        if input('Press q to quit, or enter to continue') == 'q':
            break

        print('next run')
    print("done")
