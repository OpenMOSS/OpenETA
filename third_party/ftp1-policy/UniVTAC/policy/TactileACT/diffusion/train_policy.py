import os
# diffusion policy import
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from network import get_resnet, replace_bn_with_gn, ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from clip_pretraining import modified_resnet18

from dataset import DiffusionEpisodicDataset 
#from utils import get_norm_stats

from datetime import datetime

from visualization import debug, visualize
from visualize_waypts import predict_diff_actions

from train_args import CKPT_DIR, GELSIGHT_WEIGHTS_PATH, \
IMAGE_WEIGHTS_PATH, DEVICE_STR, ABLATE_GEL,\
START_TIME


#CKPT_DIR = '/media/selamg/DATA/diffusion_plugging_checkpoints/'


# device = torch.device('cuda')
# data_dir = "/home/selamg/diffusion_plugging/Processed_Data_2/data"
# pred_horizon = 8
# obs_horizon = 1
# #both obs and action horizon are 1 right now

def create_nets(enc_type,data_dir,norm_stats,camera_names,pred_horizon,
                num_episodes=100):
    if enc_type not in ['clip','resnet18']:
        raise ValueError("only 'clip' or 'resnet18' accepted as encoder types")
    obs_horizon=1

    if enc_type == 'clip':
    # load modified CLIP pretrained resnet 

        gelsight_weights = torch.load(GELSIGHT_WEIGHTS_PATH)
        image_weights = torch.load(IMAGE_WEIGHTS_PATH)

        gelsight_encoder = modified_resnet18()
        gelsight_encoder.load_state_dict(gelsight_weights)
        gelsight_encoder = nn.Sequential(gelsight_encoder,nn.AdaptiveAvgPool2d(output_size=1), nn.Flatten())

        image_encoders = []
        for i in range(len(camera_names)-1): #subtract one to account for gelsight
            image_encoders += [modified_resnet18()]
            image_encoders[i].load_state_dict(image_weights)
            image_encoders[i] = nn.Sequential(image_encoders[i],nn.AdaptiveAvgPool2d(output_size=1), nn.Flatten())

        # modified_resnet18 uses groupnorm instead of batch already

        
    elif enc_type == 'resnet18':
        # construct ResNet18 encoder
        # if you have multiple camera views, use seperate encoder weights for each view.
        if not ABLATE_GEL:
            gelsight_encoder = get_resnet('resnet18')
            gelsight_encoder = replace_bn_with_gn(gelsight_encoder)

        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        image_encoders = []
        for i in range(len(camera_names)-1):
            image_encoders += [get_resnet('resnet18')]
            image_encoders[i] = replace_bn_with_gn(image_encoders[i])



    # Encoders have output dim of 512
    vision_feature_dim = 512
    # agent_pos is 4 dimensional
    lowdim_obs_dim = 4
    # observation feature has [  ] dims in total per step
    #7 cameras including gelsight
    if ABLATE_GEL:
        obs_dim = vision_feature_dim*(len(camera_names)-1) + lowdim_obs_dim
    else:
        obs_dim = vision_feature_dim*len(camera_names) + lowdim_obs_dim

    action_dim = 4

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )

    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]
    
    train_dataset = DiffusionEpisodicDataset(train_indices,data_dir,pred_horizon,camera_names,norm_stats)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True,
        prefetch_factor=4
        # this leads to an error message about shutting down workers at the end but 
        # does not affect training/model output
    )

    val_dataset = DiffusionEpisodicDataset(val_indices,data_dir,pred_horizon,camera_names,norm_stats)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=2,
        num_workers=1,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    nets = nn.ModuleDict({
        'noise_pred_net': noise_pred_net
    })

    if not ABLATE_GEL:
        nets['gelsight_encoder'] = gelsight_encoder

    for i, cam_name in enumerate(camera_names):
        if cam_name == 'gelsight':
            continue
        nets[f"{cam_name}_encoder"] = image_encoders[i]
    
    print("modeldict keys:",nets.keys())

    return nets, train_dataloader, val_dataloader, enc_type

def _save_ckpt(start_time:datetime,epoch,enc_type,
               nets,train_losses,val_losses,test=False):
    
    ckpt_dir=CKPT_DIR
    # noise_pred_net
    model_checkpoint = {}
    for i in nets.keys():
        model_checkpoint[i] = nets[i]
        
    now = datetime.now()
    now_time = now.strftime("%H-%M-%S_%Y-%m-%d")
    today = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    ckpt_dir = ckpt_dir+today+'_'+enc_type
    os.makedirs(ckpt_dir,exist_ok=True)

    save_dir = os.path.join(ckpt_dir,f'{enc_type}_epoch{epoch}_{now_time}')
    torch.save(model_checkpoint, save_dir)
    
    np.save(
        os.path.join(ckpt_dir,f'{enc_type}_trainlosses_{today}.npy'),
        train_losses)
    np.save(
        os.path.join(ckpt_dir,f'{enc_type}_vallosses_{today}.npy'),
        val_losses)
    

    ##test
    if test:
        model_dict = torch.load(save_dir)

        model1 = model_dict['gelsight_encoder']
        model2 = nets['gelsight_encoder']

        bool = True
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if p1.data.ne(p2.data).sum() > 0:
                bool = False

        print("noise model is same as saved model:", bool)

        model_dict = torch.load('checkpoints/2024-02-25/clip_epoch0_2024-02-25_20:31:31')
        model1 = model_dict['noise_pred_net']

        bool = True
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if p1.data.ne(p2.data).sum() > 0:
                bool = False

        print("noise model is same as random model:", bool)
        input("press any key to continue")


noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

#START_TIME = datetime.now()



def train(num_epochs,camera_names,nets:nn.ModuleDict,train_dataloader,val_dataloader:torch.utils.data.dataloader,enc_type,device=torch.device(DEVICE_STR)):
    
    debug.print=True
    debug.plot=True 
    debug.dataset=('validation')
    today = START_TIME.strftime("%Y-%m-%d_%H-%M-%S")
    debugdir = CKPT_DIR+today+'_plots'+'_'+enc_type
    debug.visualizations_dir=debugdir


    # TODO: 
    # variable obs_horizon   
    obs_horizon =1

    #TODO:
    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights

    # ema = EMAModel(
    #     parameters=nets.parameters(),
    #     power=0.75)

    nets.to(device)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_dataloader) * num_epochs
    )
    

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        train_losses = list()
        val_losses = list()
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            nets.train()
            with tqdm(train_dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer          
                    if ABLATE_GEL: 
                        del nbatch['gelsight']              
                    
                    image_features = torch.Tensor().to(device)
                    for cam_name in camera_names:
                        if cam_name == 'gelsight':
                            continue
                        ncur = nbatch[cam_name][:,:obs_horizon].to(device)
                        ncur_features = nets[f'{cam_name}_encoder'](
                            ncur.flatten(end_dim=1))
                        ncur_features = ncur_features.reshape(
                            *ncur.shape[:2],-1)
                        image_features = torch.cat([image_features, ncur_features], dim=-1)
                    if not ABLATE_GEL:
                        nimage = nbatch['gelsight'][:,:obs_horizon].to(device)
                        # encoder vision features
                        gel_features = nets['gelsight_encoder'](
                            nimage.flatten(end_dim=1))
                        gel_features = gel_features.reshape(
                            *nimage.shape[:2],-1)
                        # (B,obs_horizon,D)
                        image_features = torch.cat([image_features,gel_features],dim=-1)

           
                    nagent_pos = nbatch['agent_pos'][:,:obs_horizon].to(device)
                    naction = nbatch['action'].to(device)
                    B = nagent_pos.shape[0]
                    
                    # concatenate vision feature and low-dim obs
                    obs_features = torch.cat([image_features, nagent_pos], dim=-1)

                    obs_cond = obs_features.flatten(start_dim=1)
                    # (B, obs_horizon * obs_dim)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()


                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)
                    
                    # import pdb; pdb.set_trace()
                    # predict the noise residual
                    noise_pred = nets['noise_pred_net'](
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # TODO:
                    # update Exponential Moving Average of the model weights
                    #ema.step(nets.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            

            tglobal.set_postfix(loss=np.mean(epoch_loss))
            
            train_losses.append(np.mean(epoch_loss))
            # TODO:
            if epoch_idx % 10 == 0:
                
                nets.eval()
                val_loss=list()
                with tqdm(val_dataloader, desc='Val_Batch', leave=False) as tepoch:
                    with torch.no_grad():
                        for i,nbatch in enumerate(tepoch):

                            # data normalized in dataset
                            # device transfer

                            image_features = torch.Tensor().to(device)
                            for cam_name in camera_names:
                                if cam_name == 'gelsight':
                                    continue
                                ncur = nbatch[cam_name][:,:obs_horizon].to(device)
                                ncur_features = nets[f'{cam_name}_encoder'](
                                    ncur.flatten(end_dim=1))
                                ncur_features = ncur_features.reshape(
                                    *ncur.shape[:2],-1)
                                image_features = torch.cat([image_features, ncur_features], dim=-1)

                            if not ABLATE_GEL:
                                nimage = nbatch['gelsight'][:,:obs_horizon].to(device)
                                # encoder vision features
                                gel_features = nets['gelsight_encoder'](
                                    nimage.flatten(end_dim=1))
                                gel_features = gel_features.reshape(
                                    *nimage.shape[:2],-1)
                                # (B,obs_horizon,D)
                                image_features = torch.cat([image_features,gel_features],dim=-1)

                            
                            nagent_pos = nbatch['agent_pos'][:,:obs_horizon].to(device)
                            naction = nbatch['action'].to(device)
                            B = nagent_pos.shape[0]
                            
                            # concatenate vision feature and low-dim obs
                            obs_features = torch.cat([image_features, nagent_pos], dim=-1)

                            obs_cond = obs_features.flatten(start_dim=1)
                            # (B, obs_horizon * obs_dim)

                            # sample noise to add to actions
                            noise = torch.randn(naction.shape, device=device)

                            # sample a diffusion iteration for each data point
                            timesteps = torch.randint(
                                0, noise_scheduler.config.num_train_timesteps,
                                (B,), device=device
                            ).long()


                            # add noise to the clean images according to the noise magnitude at each diffusion iteration
                            # (this is the forward diffusion process)
                            noisy_actions = noise_scheduler.add_noise(
                                naction, noise, timesteps)
                            
                            # predict the noise residual
                            noise_pred = nets['noise_pred_net'](
                                noisy_actions, timesteps, global_cond=obs_cond)

                            # L2 loss
                            loss = nn.functional.mse_loss(noise_pred, noise)
                            loss_cpu = loss.item()
                            val_loss.append(loss_cpu)
                            tepoch.set_postfix(loss=loss_cpu)

                            #save plot of first batch
                            if i == 0:
                                debug.epoch = epoch_idx
                                
                                mdict = dict()
                                for i in nets.keys():
                                    mdict[i] = nets[i]
                                if ABLATE_GEL:
                                    del nbatch['gelsight']

                                all_images,qpos,preds,gt= predict_diff_actions(nbatch,
                                    val_dataloader.dataset.action_qpos_normalize,
                                    mdict,
                                    camera_names,device
                                )
                                print('all_images',len(all_images),'0:',all_images[0].shape)
                                print('qpos',qpos.shape)
                                print('preds', preds.shape)
                                print('gt',gt.shape)
                                visualize(all_images,qpos,preds,gt)
            
            val_losses.append(np.mean(val_loss))
            

            if epoch_idx % 500 == 0: 
                _save_ckpt(START_TIME,epoch_idx,enc_type,nets,train_losses,val_losses)
   
    _save_ckpt(START_TIME,num_epochs,enc_type,nets,train_losses,val_losses) #final save
    
    
    # TODO:
    # Weights of the EMA model
    # is used for inference
    
    # ema_nets = nets
    # ema.copy_to(ema_nets.parameters())
                
    # return nets, train_losses,val_losses


