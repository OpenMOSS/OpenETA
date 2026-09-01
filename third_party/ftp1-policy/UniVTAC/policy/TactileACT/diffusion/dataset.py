from utils import EpisodicDataset
import torch
import numpy as np
import warnings

class NormalizeDiffusionActionQpos:
    def __init__(self, norm_stats):
        # since the values of the qpos and action are tied together
        # (current position, goal position), we normalize them together
        self.action_min = norm_stats["action_min"]
        self.action_max = norm_stats["action_max"]
        self.qpos_min = norm_stats["qpos_min"]
        self.qpos_max = norm_stats["qpos_max"]
    
    def __call__(self, qpos, action):
        qpos = (qpos - self.qpos_min) / (self.qpos_max - self.qpos_min)
        action = (action - self.action_min) / (self.action_max - self.action_min)

        qpos = (qpos*2)-1
        action = (action*2)-1

        assert np.min(action) >= -1 and np.max(action) <= 1
        assert np.min(qpos) >= -1 and np.max(qpos) <= 1

        if np.min(action) < -1 or np.max(action) > 1:
            warnings.warn(f"outside bounds for action min or max, got min, max action: {np.min(action),np.max(action)}")
            
        if np.min(qpos) < -1 or np.max(qpos) > 1:
            warnings.warn(f"outside bounds for qpos min or max, got min max qpos: {np.min(qpos),np.max(qpos)}")

        return qpos, action
    
    def unnormalize(self, qpos, action):

        new_qpos = (qpos + 1)/2 
        new_qpos = (new_qpos*(self.qpos_max-self.qpos_min))+self.qpos_min

        print("max qpos:", self.qpos_max)
        print("min qpos:", self.qpos_min)
        print("max action:", self.action_max)
        print("min action:", self.action_min)

        new_action = (action+1)/2
        new_action = (new_action*(self.action_max-self.action_min))+self.action_min

        return new_qpos, new_action


class DiffusionEpisodicDataset(EpisodicDataset):

    def __init__(self, episode_ids, dataset_dir, pred_horizon, camera_names,norm_stats,image_size=None):
        self.gel_idx = None
        super().__init__(episode_ids,dataset_dir,camera_names,norm_stats, pred_horizon,image_size)
        self.action_qpos_normalize = NormalizeDiffusionActionQpos(norm_stats)
        self.camera_names = camera_names

        for i in range(len(camera_names)):
            if camera_names[i] == 'gelsight':
                self.gel_idx = i
        if self.gel_idx == None: 
            raise ValueError("camera names must include gelsight")

    def __getitem__(self, index):        

        if self.gel_idx == None:
            return super().__getitem__(index)
        
        all_cam_images, qpos_data, action_data, is_pad = super().__getitem__(index)
        # because we used the super init, everything is already normalized

        nsample = dict()

        # change the padding behavior so the robot stays in the same position at the end
        if any(is_pad): 
            last_idx = torch.where(is_pad==0)[0][-1]
            last_action = action_data[last_idx]

            action_data[last_idx+1:] = last_action
        
        # add all cameras
        for i in range(len(self.camera_names)):
            if i == self.gel_idx:
                continue
            nsample[self.camera_names[i]] = torch.stack([all_cam_images[i],]) 

        nsample['gelsight'] = torch.stack([all_cam_images[self.gel_idx],])
        nsample['agent_pos'] = torch.stack([qpos_data,])
        nsample['action'] = action_data

        return nsample