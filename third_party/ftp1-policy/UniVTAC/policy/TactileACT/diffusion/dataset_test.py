    
from dataset import DiffusionEpisodicDataset 
from utils import get_norm_stats

import torch
import numpy as np

#obs_horizon = 1
pred_horizon = 5
#action_horizon = 4

dataset_path = "/home/selamg/diffusion_plugging/Processed_Data_2/data"

dataset_dir = dataset_path

# with open(os.path.join(save_dir, 'meta_data.json'), 'r') as f:
#     meta_data: Dict[str, Any] = json.load(f)
# task_name: str = meta_data['task_name']
# num_episodes: int = meta_data['num_episodes']
# # episode_len: int = meta_data['episode_length']
# camera_names: List[str] = meta_data['camera_names']
# is_sim: bool = meta_data['is_sim']
# state_dim:int = meta_data['state_dim']

num_episodes = 39

norm_stats = get_norm_stats(dataset_dir, num_episodes)
camera_names = [1,2,3,4,5,6,"gelsight"]

train_ratio = 0.8
shuffled_indices = np.random.permutation(num_episodes)
train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
val_indices = shuffled_indices[int(train_ratio * num_episodes):]

train_dataset = DiffusionEpisodicDataset(train_indices, dataset_dir, pred_horizon, camera_names, norm_stats)


dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=2,
    num_workers=4,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)


batch = next(iter(dataloader))

print(batch.keys())
print("batch['gelsight'].shape:", batch['gelsight'].shape)
print("batch[6].shape:", batch[6].shape)
print("batch['agent_pos'].shape:", batch['agent_pos'].shape)
print("batch['action'].shape:", batch["action"].shape)

#@markdown  - key `image`: shape (obs_horizon, 3, 96, 96)
#@markdown  - key `agent_pos`: shape (obs_horizon, 2)
#@markdown  - key `action`: shape (pred_horizon, 2)

#batch['image'].shape: torch.Size([1, 7, 3, 400, 480])
# batch['agent_pos'].shape: torch.Size([1, 4])
# batch['action'].shape: torch.Size([1, 5, 4])