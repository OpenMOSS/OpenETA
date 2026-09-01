
import numpy as np
import h5py
from frankapy import FrankaArm
from frankapy import FrankaConstants as FC
from robomail.motion import GotoPoseLive
from rospy import Rate
import json
from simple_gelsight import GelSightMultiprocessed, get_camera_id

episode_dir = "/media/abraham/Crucial/Processed_Data_2/data"

# sample 20 values in range 0-100:
episode_nums = np.random.randint(0, 100, 20)

trajecotries = []
gripper_trajectories = []

for episode_num in episode_nums:
    episode_path = f"{episode_dir}/episode_{episode_num}.hdf5"
    print(episode_path)
    with h5py.File(episode_path, 'r') as f:
        trajecotries.append(f['action'][:, :3])
        gripper_trajectories.append(f['action'][:, 3])


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

camera_id = get_camera_id('GelSight')
gelsight = GelSightMultiprocessed(camera_id, use_gpu=True)

pose_controller = GotoPoseLive(cartesian_impedances=new_impedances.tolist(), step_size=0.05)

min_gripper_width = 0.004
rate = Rate(10)

ADD_NOISE = True
noise_mean = 0
noise_std = 0.0025

run_id = np.random.randint(0, 1000)
for i in range(len(trajecotries)):
    gelsight_strains = []
    grip_closed = False
    trajectory = trajecotries[i]
    gripper_trajectory = gripper_trajectories[i]

    pose = FC.HOME_POSE
    pose.translation = trajectory[0]
    for _ in range(20):
        pose_controller.step(pose)
        rate.sleep()

    input("Press enter to start")


    for j in range(len(gripper_trajectory)):
        frame, marker_data, depth, strain_x, strain_y = gelsight.get_next_frame()
        gelsight_data = np.stack([depth, strain_x, strain_y], axis=-1)
        gelsight_strains.append(np.mean(np.abs(gelsight_data), axis=(0, 1)))
        print("gelsight strain", gelsight_strains[-1])

        pose = FC.HOME_POSE
        pose.translation = trajectory[j]
        if ADD_NOISE:
            pose.translation += np.random.normal(noise_mean, noise_std, 3)
        pose_controller.step(pose)
        print("moving to", trajectory[j], "gripper width", gripper_trajectory[j])

        if gripper_trajectory[j] <= min_gripper_width + 0.01:
            if not grip_closed:
                grip_closed = True
                fa.goto_gripper(min_gripper_width)
                print("closing gripper")
            else:
                moved_gripper = False
        else:
            if grip_closed:
                print("----------------opening gripper!!!!!!!!!--------------------")
            grip_closed = False
            fa.goto_gripper(gripper_trajectory[j], block=False, speed=0.15, force = 10)
            moved_gripper = True
            print("gripper width", gripper_trajectory[j])

        rate.sleep()

    success = input("Was the grasp successful? (y/n)")
    gelsight_strains = np.array(gelsight_strains)
    np.save(f'replay_data/gelsight_strains_{i}_{run_id}.npy', gelsight_strains)


    run_stats = {"episode_idx": int(episode_nums[i]),
                 "success": success}
    
    with open(f'replay_data/run_data_{i}_{run_id}.json', 'w') as f:
        json.dump(run_stats, f)

