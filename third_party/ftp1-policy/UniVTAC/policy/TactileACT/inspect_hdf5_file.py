import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
This file contains helper functions to inspect the contents of an HDF5 file.
"""

def visualize_gelsight_data(image):
	# Convert the image to LAB color space
	max_depth = 10
	max_strain = 30
	# Show all three using LAB color space
	image[0] = np.clip(100*np.maximum(image[0], 0)/max_depth, 0, 100)
	# normalized_depth = np.clip(100*(depth_image/depth_image.max()), 0, 100)
	image[1:] = np.clip(128*(image[1:]/max_strain), -128, 127)
	return cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

def print_hdf5_info(group, indent=""):
    for key in group.keys():
        if isinstance(group[key], h5py.Group):
            print(indent + "Group:", key)
            print_hdf5_info(group[key], indent + "  ")
        elif isinstance(group[key], h5py.Dataset):
            dataset = group[key]
            print(indent + "Dataset:", key)
            print(indent + "  Shape:", dataset.shape)
            print(indent + "  Attributes:")
    for attr_name, attr_value in group.attrs.items():
        print(indent + f"    {attr_name}: {attr_value}")
    print()

def print_hdf5_file(filename):
    try:
        with h5py.File(filename, 'r') as f:
            print("Keys and shapes of datasets in HDF5 file:", filename)
            print("==============================================")
            print_hdf5_info(f)
    except IOError:
        print("Error: Unable to open file:", filename)
    except Exception as e:
        print("Error:", e)

def show_images_in_hdf5_file(filename):
    print('showing images in', filename)
    with h5py.File(filename, 'r') as f:
        for idx in range(f.attrs['num_timesteps']):
            print(idx)
            # tile the images (2x3) plus gelsight
            image_size = (f.attrs['image_height'], f.attrs['image_width'])
            gelsight_size = (f.attrs['gelsight_height'], f.attrs['gelsight_width'])
            grid = np.zeros([2*image_size[0] + gelsight_size[0], 3*image_size[1], 3], dtype=np.uint8)
            for i, key in enumerate(f['observations/images'].keys()):
                image_data = f['observations/images'][key][idx, :, :, :]
                grid[(i//3)*image_size[0]:(i//3+1)*image_size[0], (i%3)*image_size[1]:(i%3+1)*image_size[1], :] = image_data
            gelsight_data = f['observations/gelsight/depth_strain_image'][idx, :, :, :]
            gelsight_data = visualize_gelsight_data(gelsight_data)*255
            grid[image_size[0]*2:, :gelsight_size[1], :] = gelsight_data
            cv2.imshow('Images', grid)
            cv2.waitKey(0)
        cv2.waitKey(0)

def save_images_from_hdf5_file(source_file, save_folder):
    """
    Save the images and gelsight data from the HDF5 file to the save folder.
    Useful for making graphics and visualizations.
    """
    with h5py.File(source_file, 'r') as f:
        for idx in tqdm(range(f.attrs['num_timesteps'])):
            print(idx)
            # tile the images (2x3) plus gelsight
            for i, key in enumerate(f['observations/images'].keys()):
                image_data = f['observations/images'][key][idx, :, :, :]
                cv2.imwrite(f'{save_folder}/{idx}_{key}.png', image_data)
            
            gelsight_data = f['observations/gelsight/depth_strain_image'][idx, :, :, :]
            gelsight_data = visualize_gelsight_data(gelsight_data)*255
            cv2.imwrite(f'{save_folder}/{idx}_gelsight.png', gelsight_data)    


import os
if __name__ == "__main__":
    filename = "/home/aigeorge/research/TactileACT/data/camera_cage_new_mount/data/episode_55.hdf5"

    folder = "/home/aigeorge/research/TactileACT/data/camera_cage_new_mount/data"
    images_folder = "/home/aigeorge/research/TactileACT/data/camera_cage_new_mount/images"
    # all_files = []
    # for filename in os.listdir(folder):
    #     if filename.endswith(".hdf5"):
    #         all_files.append(filename)     
    # all_files.sort()

    for episode_num in [0, 1, 2]:
        filename = os.path.join(folder, f"episode_{episode_num}.hdf5")
        print_hdf5_file(os.path.join(folder, filename))
        save_images_in_hdf5_file(os.path.join(folder, filename), os.path.join(images_folder, str(episode_num)))
    exit()
    with h5py.File(filename, 'r') as f:           
        positions = f['observations/qpos'][:, :3]
        actions = f['action'][:, :3]

    # print(actions)
    # scatter plot of the positions and actions, with a color bar for index
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=range(positions.shape[0]), cmap='viridis') 
    ax.scatter(actions[:, 0], actions[:, 1], actions[:, 2], c=range(positions.shape[0]), cmap='viridis', marker='x')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()