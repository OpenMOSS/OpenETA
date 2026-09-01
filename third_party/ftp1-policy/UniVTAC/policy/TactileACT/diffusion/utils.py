import numpy as np
import torch
import os
import h5py
#from torch.utils.data import TensorDataset, DataLoader
import cv2
from torchvision import transforms
from tqdm import tqdm
import json

import IPython
e = IPython.embed

class NormalizeActionQpos:
    def __init__(self, norm_stats):
        # since the values of the qpos and action are tied together
        # (current position, goal position), we normalize them together
        self.mean = (norm_stats["qpos_mean"] + norm_stats["action_mean"])/2
        self.std = (norm_stats["qpos_std"] + norm_stats["action_std"])/2
    
    def __call__(self, qpos, action):
        qpos = (qpos - self.mean) / self.std
        action = (action - self.mean) / self.std
        return qpos, action
    
    def unnormalize(self, qpos, action):
        new_qpos = qpos * self.std + self.mean
        new_action = action * self.std + self.mean
        return new_qpos, new_action
    
class NormalizeDeltaActionQpos:
    def __init__(self, norm_stats):
        self.qpos_mean = norm_stats["qpos_mean"]
        self.qpos_std = norm_stats["qpos_std"]
        self.delta_mean = norm_stats["delta_mean"]
        self.delta_std = norm_stats["delta_std"]
    
    def __call__(self, qpos, action):
        delta = action - qpos
        delta[:, 3] = action[:, 3] # keep the gripper action the same

        # normalize the qpos and delta
        qpos = (qpos - self.qpos_mean) / self.qpos_std
        delta = (delta - self.delta_mean) / self.delta_std
        return qpos, delta
    
    def unnormalize(self, normalized_qpos, normalized_delta):
        qpos = normalized_qpos * self.qpos_std + self.qpos_mean
        delta = normalized_delta * self.delta_std + self.delta_mean
        action = qpos + delta
        action[:, 3] = delta[:, 3] # keep the gripper action the same
        return qpos, action

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, chunk_size, image_size = None):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names

        self.action_qpos_normalize = NormalizeActionQpos(norm_stats)

        self.is_sim = None
        self.image_size = image_size # image size in (H, W)
        if "gelsight_mean" in norm_stats: # if gelsight data exists
            self.gelsight_mean = norm_stats["gelsight_mean"]
            self.gelsight_std = norm_stats["gelsight_std"]

        self.chunk_size = chunk_size

        # image normalization for resnet. 
        self.image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.__getitem__(0) # initialize self.is_sim, self.image_size


        # image = normalize(image)

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            # qvel = root['/observations/qvel'][start_ts] # unused

            if self.image_size is None: # if image size is not specified, use the saved image size
                self.image_size = (root.attrs['image_height'], root.attrs['image_width'])

            all_cam_images = []

            for cam_name in self.camera_names:
                # seperate processing for gelsight:
                if cam_name == 'gelsight':
                    gelsight_data = root['observations/gelsight/depth_strain_image'][start_ts]
                    # gelsight_data = cv2.resize(gelsight_data, (self.image_size[1], self.image_size[0]))
                    # adjust gelsight data using the mean and std
                    gelsight_data = (gelsight_data - self.gelsight_mean) / self.gelsight_std
                    gelsight_data = torch.tensor(gelsight_data, dtype=torch.float32)
                    gelsight_data = torch.einsum('h w c -> c h w', gelsight_data) # change to c h w
                    all_cam_images.append(gelsight_data)
                
                else:
                    image = root[f'/observations/images/{cam_name}'][start_ts]
                    # resize image
                    if self.image_size != image.shape[:2]:
                        print('reshaping image')
                        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))

                    # normalize image
                    image = torch.tensor(image, dtype=torch.float32)/255.0
                    image = torch.einsum('h w c -> c h w', image) # change to c h w
                    image = self.image_normalize(image)
                    all_cam_images.append(image)


            # get all actions after and including start_ts, with the max length of chunk_size
            action_len = min(episode_len - start_ts, self.chunk_size) 
            action = root['/action'][start_ts:start_ts + action_len]

        # new axis for different cameras
        # image_data = torch.stack(all_cam_images, axis=0)

        # normalize action and qpos
        qpos, action = self.action_qpos_normalize(qpos=qpos, action=action)

        self.is_sim = is_sim
        padded_action = np.zeros([self.chunk_size, action.shape[1]], dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1

        # construct observations
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # normalize image and change dtype to float

        # if not self.use_rot:
        #     qpos_data = torch.cat((qpos_data[:3], qpos_data[6:]))
        #     action_data = torch.cat((action_data[:, :3], action_data[:, 6:]), dim=1)

        return all_cam_images, qpos_data, action_data, is_pad
    
# create a new class for the delta action - different normalization
class EpisodicDatasetDelta(EpisodicDataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, chunk_size, image_size = None):
        super(EpisodicDatasetDelta, self).__init__(episode_ids, dataset_dir, camera_names, norm_stats, chunk_size, image_size)
        self.action_qpos_normalize = NormalizeDeltaActionQpos(norm_stats)

def gelsight_norm_stats(dataset_dir, num_episodes) -> tuple:
    gelsight_means = [] 
    gelsight_stds = []
    for episode_idx in tqdm(range(num_episodes), desc="Get Gelsight Stats"):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:

            # if gelsight data exists, get the average
            if 'observations/gelsight/depth_strain_image' in root:
                use_gelsight = True
                gelsight_data = root['observations/gelsight/depth_strain_image'][()]
                gelsight_mean = np.mean(gelsight_data, axis=(1, 2))
                gelsight_std = np.std(gelsight_data, axis=(1, 2))   
                gelsight_means.extend(gelsight_mean)
                gelsight_stds.extend(gelsight_std)
    
    gelsight_mean = np.mean(np.array(gelsight_means), axis=0)
    gelsight_std = np.mean(np.array(gelsight_stds), axis=0)
    gelsight_std = np.clip(gelsight_std, 1e-2, np.inf)

    return gelsight_mean, gelsight_std


def get_norm_stats(dataset_dir, num_episodes, use_existing=True, chunk_size = 0):
    qpos_data_list = []
    action_data_list = []
    use_gelsight = False

    for episode_idx in tqdm(range(num_episodes), desc="Get Norm Stats"):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            #qvel = root['/observations/qvel'][()] # unused
            action = root['/action'][()]

            # if gelsight data exists, get the average
            if 'observations/gelsight/depth_strain_image' in root:
                use_gelsight = True

        qpos_data_list.append(qpos)
        action_data_list.append(action)
    all_qpos_data = np.concatenate(qpos_data_list, axis=0)
    all_action_data = np.concatenate(action_data_list, axis=0)

    # get mean of the action data
    action_mean = all_action_data.mean(axis=0)
    action_std = all_action_data.std(axis=0)
    action_std = np.clip(action_std, 1e-2, np.inf) # clipping


    action_min = all_action_data.min(axis=0) 
    action_max = all_action_data.max(axis=0)
    #actual gripper minmax
    action_min[3] = 0.0 
    action_max[3] = 0.08

    # get mean of the qpos data
    qpos_mean = all_qpos_data.mean(axis=0)
    qpos_std = all_qpos_data.std(axis=0)
    qpos_std = np.clip(qpos_std, 1e-2, np.inf) # clipping

    qpos_min = all_qpos_data.min(axis=0)
    qpos_max = all_qpos_data.max(axis=0)
    #actual gripper minmax
    qpos_min[3] = 0.0
    qpos_max[3] = 0.08

    stats = {"action_mean": action_mean, "action_std": action_std, "action_min":action_min, 
             "action_max":action_max,
             "qpos_min":qpos_min, "qpos_max":qpos_max,
            "qpos_mean": qpos_mean, "qpos_std": qpos_std}

    # check to see if norm stats already exists
    if use_gelsight:
        if use_existing and os.path.exists(os.path.join(dataset_dir, 'gelsight_norm_stats.json')):
            with open(os.path.join(dataset_dir, 'gelsight_norm_stats.json'), 'r') as f:
                gelsight_stats = json.load(f)
                gelsight_mean = np.array(gelsight_stats['gelsight_mean'])
                gelsight_std = np.array(gelsight_stats['gelsight_std'])
        else:
            gelsight_mean, gelsight_std = gelsight_norm_stats(dataset_dir, num_episodes)

        stats["gelsight_mean"] = gelsight_mean
        stats["gelsight_std"] = gelsight_std

    if chunk_size != 0:
        # calculate the mean and std of the delta (position) actions:
        all_deltas = []
        for episode in range(num_episodes):
            len_episode = len(action_data_list[episode])
            for t in range(len_episode - chunk_size):
                deltas = action_data_list[episode][t:t+chunk_size, 0:3] - qpos_data_list[episode][t][0:3]
            # print(all_deltas[-(len_episode - chunk_size):])/
            # for offset in range(chunk_size):
            #     deltas = action_data_list[episode, offset:] - qpos_data_list[episode, :-offset]
                all_deltas.append(deltas)

        all_deltas = np.concatenate(all_deltas, axis=0)
        delta_mean = all_deltas.mean(axis=0)
        delta_std = all_deltas.std(axis=0)
        delta_std = np.clip(delta_std, 1e-3, np.inf)
        stats["delta_mean"] = np.concatenate([delta_mean, [action_mean[3]]])
        stats["delta_std"] = np.concatenate([delta_std, [action_std[3]]])
            
    return stats


## helper functions, same as ACT

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    dataset_dir = "/home/aigeorge/research/TactileACT/data/camera_cage_new_mount/data"
    num_episodes = 101
    norm_stats = get_norm_stats(dataset_dir, num_episodes, use_existing=True, chunk_size=30)
    print(norm_stats)
    # dataset = EpisodicDataset(range(num_episodes), dataset_dir, camera_names, norm_stats, 10)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # for i, data in enumerate(dataloader):
    #     print(data)
    #     if i == 10:
    #         break
    # print("done")
