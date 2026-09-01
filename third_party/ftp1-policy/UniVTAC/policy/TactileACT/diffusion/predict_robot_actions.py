from network import DDPMScheduler
import torch
from torch import nn
from network import get_resnet, replace_bn_with_gn, ConditionalUnet1D
from clip_pretraining import modified_resnet18
import os

#from dataset import NormalizeDiffusionActionQpos
#EXPECTED_CAMERA_NAMES = [1,2,3,4,5,6,'gelsight'] 

device = 'cuda'

def diffuse_robot(qpos_data,image_data,camera_names,model_dict,
                         pred_horizon,device=device):  
    
    #unnormalizer = NormalizeDiffusionActionQpos
    #for now they are just not normalized but it could be handled here

    # unlike predict_diff_actions this function is specifically for taking in what the
    # "process data" class spits out
    
    nets = nn.ModuleDict(model_dict).to(device)

    #obs_horizon = 1 
    #map_indices:

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
    # obs_horizon = 1
    # vision_feature_dim = 512
    # lowdim_obs_dim = 4

    #obs_dim = vision_feature_dim*7 + lowdim_obs_dim
    for i in range(len(image_data)):
        image_data[i] = torch.stack([image_data[i],])
    
    qpos_data = torch.stack([qpos_data,])


    #this is dumb but stacking them just to make it work...
    
    with torch.no_grad(): 
        #set models to eval
        image_features = torch.Tensor().to(device)
        for i,cam_name in enumerate(camera_names):
            if cam_name == 'gelsight':
                gel = image_data[i].to(device)
                gel_features = nets['gelsight_encoder'](gel.flatten(end_dim=1))
                gel_features = gel_features.reshape(*gel.shape[:2],-1)
                image_features = torch.cat([image_features,gel_features],dim=-1)
                continue
            ncur:torch.Tensor = image_data[i].to(device) #implicitly in order....?
            ncur_features:torch.Tensor = nets[f'{cam_name}_encoder'](
                ncur.flatten(end_dim=1))
            ncur_features = ncur_features.reshape(
                *ncur.shape[:2],-1)
            image_features = torch.cat([image_features, ncur_features], dim=-1)

        #from train should be 2,1,3584
        # print("image_features:",image_features.shape)
        #check!

        agent_pos=qpos_data.to(device)
        obs = torch.cat([image_features,agent_pos],dim=-1) 
        #2,1,3588
        obs_cond = obs.flatten(start_dim=1)
        #print(obs_cond.shape)
        #obs cond 2, 3588
        B = (agent_pos.shape[0])
        # print("B, obs_cond, agent_pos:",B,obs_cond.shape,agent_pos.shape)
        noisy_action = torch.randn(
        (B, pred_horizon, action_dim), device=device)

        naction = noisy_action #1,5,4
        #1,pred_horizon,
        infer_iters = 10
        noise_scheduler.set_timesteps(infer_iters)
        
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

    
        naction = naction.detach().to('cpu')

        naction = naction[0]

        qpos  = qpos_data
        qpos = qpos.flatten() #?
        qpos = qpos.detach().to('cpu')
        
        #qpos, naction = unnormalizer.unnormalize(qpos, naction)
        
        return naction #not normalized

