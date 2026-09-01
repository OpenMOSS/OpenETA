from utils import get_norm_stats
import json

# dataset_dir = '/home/selamg/diffusion_plugging/Processed_Data_2/data'
dataset_dir = '/home/selam/data/camera_cage_new_notfixed/data'

norm_stats = get_norm_stats(dataset_dir,num_episodes=100)
print(norm_stats)

for i in norm_stats.keys():
    norm_stats[i] = norm_stats[i].tolist()

with open('norm_stats_not_fixed.json','w') as f:
    json.dump(norm_stats,f,indent=4)


