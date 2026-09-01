# from load_ACT import load_ACT
import h5py
import json
import numpy as np
from torchvision import transforms

from old_plug_code.process_data_cage import MASK_VERTICIES, CROP_PARAMS, make_masks
from typing import Dict, List, Tuple
import cv2
import torch

from dataset import NormalizeDiffusionActionQpos
# import os
# from frankapy import FrankaArm
# from frankapy import FrankaConstants as FC
# from robomail.motion import GotoPoseLive

# from simple_gelsight import GelSightMultiprocessed, get_camera_id
# from multiprocessed_cameras import MultiprocessedCameras

from utils import NormalizeActionQpos, NormalizeDeltaActionQpos

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
        # self.normalizer = NormalizeDeltaActionQpos(norm_stats)
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
    for i, image in enumerate(images):
        print(image.shape)
        axs_left[i].imshow(image)     

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
    import time
    fa = FrankaArm()
    fa.reset_joints()
    fa.open_gripper()
    move_pose = FC.HOME_POSE
    move_pose.translation = np.array([0.6, 0, 0.35])
    fa.goto_pose(move_pose)

    pose_controller = GotoPoseLive(step_size=0.05)
    pose_controller.set_goal_pose(move_pose)

    model_path = "/home/abraham/TactileACT/data/delta_1999_no_pretraining/policy_last.ckpt"
    save_video_path = "/home/abraham/TactileACT/data/delta_1999_no_pretraining/videos/"
    args_file = "/home/abraham/TactileACT/data/delta_1999_no_pretraining/args.json"

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
    act = load_ACT(model_path, args_file).to(device)
    
    camera_nums = [1, 2, 3, 4, 5, 6]
    camera_sizes = [(1080, 1920), (1080, 1920), (1080, 1920), (1080, 1920), (1080, 1920), (800, 1280)]
    cameras = MultiprocessedCameras(camera_nums, camera_sizes, 30)
    min_gripper_width = 0.0055

    # get the norm stats from args_file, need to convert to numpy array

    args = json.load(open(args_file, 'r'))
    norm_stats = {k: np.array(v) for k, v in args['norm_stats'].items()}

    horizon = args['chunk_size']

    # create the fake dataset
    preprocess = PreprocessData(norm_stats, args['camera_names'])
    
    num_episodes = 200
    grip_closed = False

    temporal_ensemble = True
    K = 0.25

    action_history = np.zeros([num_episodes + horizon, num_episodes, 4]) # prediction_time, time_preiction_was_made, action
    confidences = []
    # stack the weights to the correct shape

    # # pretend to be a robot by loading a dataset
    # data_dir = "//home/aigeorge/research/TactileACT/data/original/camara_cage_1/run_0/episode_3/episode_3.hdf5"
    # with h5py.File(data_dir, 'r') as root:
    #     qpos = root['/observations/position'][()]
    #     gt_actions = root['/goal_position'][()]
    #     all_gelsight_data = root['observations/gelsight/depth_strain_image'][()]
    #     num_episodes = root.attrs['num_timesteps']

    #     all_images = {}
    #     for cam in root.attrs['camera_names']:
    #         video_images = []
    #         video_path = os.path.join(os.path.dirname(data_dir), f'cam-{cam}.avi')
    #         cap = cv2.VideoCapture(video_path)
    #         for i in range(num_episodes):
    #             ret, frame = cap.read()
    #             if not ret:
    #                 break
    #             video_images.append(frame)
            
    #         all_images[cam] = np.array(video_images)
    #         cap.release()

    run_images = []
    for i in range(num_episodes):
        # images = {key: all_images[key][i] for key in all_images}
        # gelsight_data = all_gelsight_data[i]

        images = cameras.get_next_frames()
        frame, marker_data, depth, strain_x, strain_y = gelsight.get_next_frame()

        robo_data = fa.get_robot_state()
        current_pose = robo_data['pose']*fa._tool_delta_pose
        
        # cur_joints = robo_data['joints']
        # cur_vel = robo_data['joint_velocities']
        finger_width = robo_data['gripper_width']
        qpos = np.zeros(4)
        qpos[:3] = current_pose.translation
        qpos[3] = finger_width
        print("qpos", qpos)
        gelsight_data = np.stack([depth, strain_x, strain_y], axis=-1)
        image_data, qpos_data = preprocess.process_data(images, gelsight_data, qpos)

        # get the action from the model
        qpos_data = qpos_data.to(device)
        image_data = [img.to(device) for img in image_data]
        deltas:torch.Tensor = act(qpos_data, image_data)

        # unnormalize the actions and qpos 
        deltas = deltas.squeeze().detach().cpu().numpy()
        # print("deltas", deltas)
        unnormalized_deltas = preprocess.normalizer.unnormalize_delta(deltas)
        # print("unnormalized_deltas", unnormalized_deltas)
        all_actions = unnormalized_deltas + qpos
        all_actions[:, 3] = np.clip(unnormalized_deltas[:, 3], 0, 0.08) # use output grip width and clip it
        # print("all_actions", all_actions)
        action_history[i:i+horizon, i] = all_actions

        if temporal_ensemble:
            print('i', i)
            ensembled_action = np.zeros(4)
            total_weight = 0
            time_step_actions = []
            
            K = 0.1 + np.mean(np.mean(gelsight_data, axis=(0, 1))**2)
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
            print("time_step_actions", time_step_actions)
            print('confidences', confidences)
            print('current_delta', unnormalized_deltas[0])
            print("current_pose", current_pose.translation)
            print("ensembled_action", ensembled_action)
            print('current_action', all_actions[0])
            print('abs mean gelsight', np.mean(gelsight_data, axis=(0, 1)))
            print("K", K)

        # visualize the data
        vis_images = [image_data[j].squeeze().detach().cpu().numpy().transpose(1, 2, 0) for j in range(len(image_data))]
        run_images.append(vis_images)

        skip_amount = 0
        move_pose = FC.HOME_POSE
        move_pose.translation = ensembled_action[:3]
        pose_controller.step(move_pose, current_pose)
        grip_command = ensembled_action[3]
        grip_command = np.clip(grip_command, 0, 0.08)
        if grip_command <= min_gripper_width:
            if not grip_closed:
                grip_closed = True
                fa.goto_gripper(min_gripper_width)
                print("closing gripper")
        else:
            grip_closed = False
            fa.goto_gripper(grip_command)
            print("gripper width", grip_command)
        # if i % 5 == 0:
        #     visualize(vis_images, qpos, all_actions, time_step_actions)
        if i % 3 == 0:
            command = input("Press enter to continue, v to visualize, or q to quit")
            if command == 'q':
                break
            elif command == 'v':
                visualize(vis_images, qpos, all_actions, time_step_actions)

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
    for cam_num, save_cam in enumerate(args['camera_names']):
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
    print("done")