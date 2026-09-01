import h5py
import torch
from torch import nn
import os
from visualization import visualize

from network import get_resnet, replace_bn_with_gn, ConditionalUnet1D
    
from dataset import DiffusionEpisodicDataset, NormalizeDiffusionActionQpos
from utils import get_norm_stats

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler



def load_models(dataset_dir, 
                weights_dir, norm_stats, camera_names,num_episodes,pred_horizon):

    if not os.path.isfile(weights_dir):
        print("fileNotFound")

    model_dict = torch.load(weights_dir, map_location=torch.device('cuda:0'))
    
    indices = list(range(num_episodes))
    #make it deterministic
    #MUST BE SAME AS TRAIN PRED!!!!!

    entire_dataset = DiffusionEpisodicDataset(indices, dataset_dir, pred_horizon, camera_names, norm_stats)

    ## nqpos, naction = entire_dataset.action_qpos_normalize.unnormalize(batch["qpos"],batch["action"])

    dataloader = torch.utils.data.DataLoader(
        entire_dataset,
        batch_size=2,
        num_workers=1,
        shuffle=False,
        # accelerate cpu-gpu transfer
        pin_memory=False,
        # don't kill worker process afte each epoch
        persistent_workers=False
    )
    # image_encoders = {}
    # for i, cam_name in enumerate(camera_names):
    #     image_encoders = model_dict[f'{cam_name}_encoder']

    # gelsight_encoder = model_dict['gelsight_encoder']
    # noise_pred_net = model_dict['noise_pred_net']

    print("Model Loaded")
    return dataloader, model_dict

def predict_diff_actions(batch,unnormalizer:NormalizeDiffusionActionQpos,model_dict,camera_names,
                         device): 
    
    nets = nn.ModuleDict(model_dict)

    obs_horizon = 1 

    diff_iters=100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=diff_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )
    
    action_dim = 4
    obs_horizon = 1
    # vision_feature_dim = 512
    # lowdim_obs_dim = 4

    # obs_dim = vision_feature_dim*7 + lowdim_obs_dim
    
    with torch.no_grad(): 
        #set models to eval
        image_features = torch.Tensor().to(device)
        for cam_name in camera_names:

            if cam_name == 'gelsight':
                continue
            ncur:torch.Tensor = batch[cam_name][:,:obs_horizon].to(device)
            ncur_features:torch.Tensor = nets[f'{cam_name}_encoder'](
                ncur.flatten(end_dim=1))
            ncur_features = ncur_features.reshape(
                *ncur.shape[:2],-1)
            image_features = torch.cat([image_features, ncur_features], dim=-1)
        
        if "gelsight" in batch.keys():
            gel = batch["gelsight"][:,:obs_horizon].to(device)
            gel_features = nets['gelsight_encoder'](gel.flatten(end_dim=1))
            gel_features = gel_features.reshape(*gel.shape[:2],-1)
            image_features = torch.cat([image_features,gel_features],dim=-1)
        #from train should be 2,1,3584
        # print("image_features:",image_features.shape)
        #check!

        agent_pos=batch['agent_pos'].to(device)
        obs = torch.cat([image_features,agent_pos],dim=-1) 
        #2,1,3588
        obs_cond = obs.flatten(start_dim=1)
        #obs cond 2, 3588
        
        B = (agent_pos.shape[0])
        # print("B, obs_cond, agent_pos:",B,obs_cond.shape,agent_pos.shape)
        noisy_action = torch.randn(
        (B, batch['action'].shape[1], action_dim), device=device)

        naction = noisy_action #1,5,4
        #1,pred_horizon,4

        #FROM TRAINING:
        # obs_cond.shape
        # torch.Size([8, 3588])

        inference_iters = 100
        noise_scheduler.set_timesteps(inference_iters)
        
        for k in noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets['noise_pred_net'](
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample


        #again with 100 inference steps and compare
        
        # inference_iters = 10
        # noise_scheduler.set_timesteps(inference_iters)
        # naction2 = noisy_action #1,5,4

        
        # for k in noise_scheduler.timesteps:
        #     # predict noise
        #     noise_pred = nets['noise_pred_net'](
        #         sample=naction2,
        #         timestep=k,
        #         global_cond=obs_cond
        #     )

        #     # inverse diffusion step (remove noise)
        #     naction2 = noise_scheduler.step(
        #         model_output=noise_pred,
        #         timestep=k,
        #         sample=naction2
        #     ).prev_sample


        # naction2 = naction2.detach().to('cpu')
        naction = naction.detach().to('cpu')

        #unnormalize
        #naction.detach.to('cpu').numpy()
        naction = naction[0]
        # naction2 = naction2[0]

        gt = batch["action"][0]
        gt = gt.detach().to('cpu')
        
        qpos  = batch["agent_pos"][0]
        qpos = qpos.flatten()
        qpos = qpos.detach().to('cpu')

        # norm_mean = (norm_stats["action_mean"]+ norm_stats["qpos_mean"])/2
        # norm_std = (norm_stats["action_std"] + norm_stats["qpos_std"])/2
        
        qpos, naction = unnormalizer.unnormalize(qpos, naction)
        # _, naction2 = unnormalizer.unnormalize(qpos, naction2)
        # naction = (naction *norm_std) + norm_mean
        # qpos = (qpos*norm_std)+norm_mean
        _,gt = unnormalizer.unnormalize(qpos,gt)

        if "gelsight" in batch.keys():
            gel = batch["gelsight"][0].flatten(end_dim=1)
            gel = gel.permute(1,2,0)
            gel = gel.detach().to('cpu')

        all_images = []
        # TODO 
        #add gelsight
        for i in camera_names:
            if i == 'gelsight':
                continue
            image = batch[i][0]
            image = image.flatten(end_dim=1)
            image = image.permute(1,2,0)
            image = image.detach().to('cpu')
            all_images.append(image)
        
        if 'gelsight' in batch.keys():
            all_images.append(gel)        
        return all_images, qpos,naction,gt

        

