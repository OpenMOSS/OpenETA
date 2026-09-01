import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from utils import get_norm_stats, EpisodicDataset, EpisodicDatasetDelta # data functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy
from visualization_utils import visualize_data, debug

from typing import List, Dict, Tuple, Any
from pathlib import Path
import json

import IPython
e = IPython.embed

import cv2

# from visualization_utils import visualise_trajectory, z_slider

# Note about debug:
# we used the global variable debug to trigger plotting in the training loop. This is a bit of a hack, but it works.

DT = 0.01 # need to hardcode, used to be in constants.py
FREEZE_TACTILE = True

def main(args):

    # extract args
    is_eval:bool = args['eval']
    save_dir:str = args['save_dir']
    model_name:str = args['name']
    policy_class:str = args['policy_class']
    batch_size_train:int = args['batch_size']
    batch_size_val:int = args['batch_size']
    num_epochs:int = args['num_epochs']
    chunk_size:int = args['chunk_size']
    kl_weight:float = args['kl_weight']
    hidden_dim:int = args['hidden_dim']
    dim_feedforward:int = args['dim_feedforward']
    lr:float = args['lr']
    lr_backbone:float = args['lr_backbone']
    backbone_type:str = args['backbone']
    num_enc_layers:int = args['enc_layers']
    num_dec_layers:int = args['dec_layers']
    nheads:int = args['nheads']
    seed:int = args['seed']
    temporal_agg:bool = args['temporal_agg']
    ckpt_name:str = args['checkpoint']
    gpu:int = args['gpu']
    onscreen_render:bool = args['onscreen_render'] 
    position_embedding_type:str = args['position_embedding']
    masks:bool = args['masks']
    dilation:bool = args['dilation']
    dropout:float = args['dropout']
    pre_norm:bool = args['pre_norm']
    z_dimension:int = args['z_dimension']
    weight_decay:float = args['weight_decay']


    ckpt_dir: str = os.path.join('./act-ckpt/', model_name)
    dataset_dir: str = save_dir

    assert os.path.exists(save_dir), f'{save_dir} does not exist. Please select a valid directory.'

    # read the meta_data folder:
    with open(os.path.join(save_dir, 'norm_stats.json'), 'r') as f:
        meta_data: Dict[str, Any] = json.load(f)
    task_name: str = meta_data['task_name']
    num_episodes: int = meta_data['num_episodes']
    # episode_len: int = meta_data['episode_length']
    camera_names: List[str] = meta_data['camera'] + meta_data['tactile']
    print(f"camera_names: {camera_names}")
    is_sim: bool = True
    state_dim:int = len(meta_data['qpos_mean'])

    norm_stats = get_norm_stats(dataset_dir, num_episodes, chunk_size=chunk_size)

    # save norm stats to args. Need to convert from numpy to list
    args['norm_stats'] = {k: v.tolist() for k, v in norm_stats.items()}

    set_seed(1)
    if gpu != -1: # -1 == No GPU Specified
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # load pretrained backbones
    if args['backbone'] == "clip_backbone":
        # assert 'gelsight' in camera_names, 'Gelsight camera not found in camera_names. Please add it to the meta_data.json file.'
        from clip_pretraining import modified_resnet18
        gelsight_model = modified_resnet18()
        vision_model = modified_resnet18()
        camera_backbone_mapping = {cam_name: 0 for cam_name in camera_names}
        camera_backbone_mapping['cam_left_tactile'] = 1
        camera_backbone_mapping['cam_right_tactile'] = 1

        # if pretrained, load the weights
        if args['gelsight_backbone_path'] != 'none' and args['vision_backbone_path'] != 'none':
            vision_model.load_state_dict(torch.load(args['vision_backbone_path']))
            gelsight_model.load_state_dict(torch.load(args['gelsight_backbone_path']))
        elif args['gelsight_backbone_path'] != 'none' or args['vision_backbone_path'] != 'none':
            raise ValueError('Both vision and gelsight backbones must be specified if one is specified.')
        
        if FREEZE_TACTILE:
            gelsight_model.requires_grad_(False)
            print("Freezing tactile backbone")
        pretrained_backbones = [vision_model, gelsight_model]
    else:
        if args['gelsight_backbone_path'] != 'none' or args['vision_backbone_path'] != 'none':
            raise ValueError('A backbone path was specified, but the backbone type is not clip_backbone.')
        pretrained_backbones = None
        camera_backbone_mapping = None


    # make policy
    if policy_class == 'ACT':
        policy = ACTPolicy(state_dim=state_dim,
                           hidden_dim=hidden_dim,
                           position_embedding_type=position_embedding_type,
                           lr_backbone=lr_backbone,
                           masks=masks,
                           backbone_type=backbone_type,
                           dilation=dilation,
                           dropout=dropout,
                           nheads=nheads,
                           dim_feedforward=dim_feedforward,
                           num_enc_layers=num_enc_layers,
                           num_dec_layers=num_dec_layers,
                           pre_norm=pre_norm,
                           num_queries=chunk_size,
                           camera_names=camera_names,
                           z_dimension=z_dimension,
                           lr=lr,
                           weight_decay=weight_decay,
                           kl_weight=kl_weight,
                           pretrained_backbones=pretrained_backbones,
                           cam_backbone_mapping=camera_backbone_mapping
                           )

    else:
        raise NotImplementedError(f'Policy class {policy_class} not implemented')
    
    policy.cuda()
    

    if is_eval:
        raise NotImplementedError("We evaluated on hardware, not simulation. Please see robot_operation.py for the evaluation code. If you wish to evaluate in simulation, examine robot_operation.py or eval_bc() in the orgianl ACT")

    # If its not eval, then we are training. Make the ckpt_dir to save the model.
    if os.path.exists(ckpt_dir):
        n = 0
        while os.path.exists(ckpt_dir + f'_{n}'):
            n += 1
        Warning(f'{ckpt_dir} already exists. Renaming to {ckpt_dir}_{n}')
        ckpt_dir = ckpt_dir + f'_{n}'

    # make ckpt_dir
    os.makedirs(ckpt_dir)

    # set visualisation directory in debug
    debug.visualizations_dir = os.path.join(ckpt_dir, 'visualizations')

    # save args + meta_data
    combo_dict = {**args, **meta_data}
    with open(os.path.join(ckpt_dir, 'args.json'), 'w') as f:
        # use enters for better readability
        json.dump(combo_dict, f, indent=4)

    # load dataset
    # def load_data(config.dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')

    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # save dataset stats
    # obtain normalization stats for qpos and action
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(norm_stats, f)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, chunk_size=chunk_size)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, chunk_size=chunk_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=8)

    debug.action_qpos_normalizer = train_dataset.action_qpos_normalize

    best_ckpt_info = train_bc(policy=policy,
                              train_dataloader=train_dataloader,
                              val_dataloader=val_dataloader,
                              num_epochs=num_epochs,
                              ckpt_dir=ckpt_dir,
                              seed=seed,
                              )
                              

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print('checkpoint path', ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def train_bc(policy:ACTPolicy,
             train_dataloader: DataLoader,
             val_dataloader: DataLoader, 
             num_epochs: int,
             ckpt_dir: str,
             seed: int,
             ):
    
    # policy.cuda()
    plot_freq = 1000

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        # print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                # forward pass
                image_data, qpos_data, action_data, is_pad = data

                # plot the first batch of every plot_freq epochs
                if batch_idx == 0 and epoch%plot_freq == 0:
                    debug.plot = True
                    debug.print = False
                    debug.epoch = epoch
                    debug.batch = 0
                    debug.dataset = 'validation'

                qpos_data = qpos_data.cuda()
                image_data = image_data.cuda()
                action_data = action_data.cuda()
                is_pad = is_pad.cuda()
                # evaluation, so we want to ignore the latent variables (we still want them to compute the loss though)
                forward_dict = policy(qpos_data, image_data, action_data, is_pad, ignore_latent=True)

                if batch_idx == 0 and epoch%plot_freq == 0:
                    debug.plot = False
                    debug.print = False

                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        # print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        # print(summary_string)

        # training
        policy.train()
        policy.optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):

            image_data, qpos_data, action_data, is_pad = data
            if batch_idx == 0 and epoch%plot_freq == 0:
                debug.plot = True
                debug.print = False
                debug.epoch = epoch
                debug.batch = 0
                debug.dataset = 'train'

            qpos_data = qpos_data.cuda()
            image_data = image_data.cuda()
            action_data = action_data.cuda()
            is_pad = is_pad.cuda()
            forward_dict = policy(qpos_data, image_data, action_data, is_pad)

            if batch_idx == 0 and epoch%plot_freq == 0:
                debug.plot = False
                debug.print = False
            
            # backward
            loss = forward_dict['loss']
            loss.backward()
            policy.optimizer.step()
            policy.optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        # print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        # print(summary_string)

        if epoch % 1000 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    # save history:
    with open(os.path.join(ckpt_dir, 'train_history.pkl'), 'wb') as f:
        pickle.dump(train_history, f)
    with open(os.path.join(ckpt_dir, 'validation_history.pkl'), 'wb') as f:
        pickle.dump(validation_history, f)


    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(1, num_epochs-1, len(train_history)), train_values, label='train') # skip the first epoch
        plt.plot(np.linspace(1, num_epochs-1, len(validation_history)), val_values, label='validation')
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the JSON config file', default=None)
    args, _ = parser.parse_known_args()

    print('parsed config', args.config)

    # If a config file is provided, use that, overwritting any values specified
    # in the command line
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = json.load(f)
        print('loaded config', config)

        # For the parse to work, we need to specify the type of each argument
        key_types = {
            "eval": bool,
            "checkpoint": str,
            "onscreen_render": bool,
            "save_dir": str,
            "name": str,
            "policy_class": str,
            "batch_size": int,
            "seed": int,
            "num_epochs": int,
            "lr": float,
            "kl_weight": float,
            "chunk_size": int,
            "hidden_dim": int,
            "dim_feedforward": int,
            "temporal_agg": bool,
            "z_dimension": int,
            "gpu": int,
            "lr_backbone": float,
            "weight_decay": float,
            "backbone": str,
            "dilation": bool,
            "position_embedding": str,
            "enc_layers": int,
            "dec_layers": int,
            "dropout": float,
            "nheads": int,
            "pre_norm": bool,
            "masks": bool,
            "gelsight_backbone_path": str,
            "vision_backbone_path": str,
        }
        
        for k, v in key_types.items():
            parser.add_argument(f'--{k}', action='store', type=v, help=f'{k}', required=False, default=None)

        # override the default values with the values from the config file
        new_args = vars(parser.parse_args())
        for k, v in new_args.items():
            if v is not None:
                config[k] = v
                print('updated', k, 'to', v)

        print('new config', config)

        # make sure all arguments are present and of the correct type
        for k, v in config.items():
            if k in key_types:
                if not isinstance(v, key_types[k]):
                    raise ValueError(f'Expected {k} to be of type {key_types[k]}, but got {type(v)}')
            elif k != "config":
                # raise ValueError(f'Unexpected config parameter {k}')
                pass
            
        for k, v in key_types.items():
            if k not in config:
                raise ValueError(f'Expected {k} to be in the config file')

        # check to make sure that non-default values are provided for required arguments:
        if new_args['save_dir'] == "path/to/save_dir":
            raise ValueError('save_dir must be provided in the config file or as a command line argument')
        if new_args['name'] == "example_name":
            raise ValueError('name must be provided in the config file or as a command line argument') 
        if new_args['eval'] == True and new_args['checkpoint'] == "path/to/checkpoint":
            raise ValueError('checkpoint must be provided in the config file or as a command line argument')

            
        # run main with the config
        main(config)
        exit() # exit here, as main(config) will run the training loop

    # If no config file is provided, use the command line arguments  
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--checkpoint', action='store', type=str, help='checkpoint name', required=False, default='policy_best.ckpt')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--save_dir', action='store', type=str, help='Directory where the checkpoint director will be created, where the recoreded sim episodes are saved, and where the meta-data JSON is located', required=True)
    parser.add_argument('--name', action='store', type=str, help='name of the directory (located in save_dir) to store the training data', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=float, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help="Size of the embeddings (dimension of the transformer)", required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, default=2048, help="Intermediate size of the feedforward layers in the transformer blocks", required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--z_dimension', default=32, type=int, help='dimension of the latent space of the CVAE')
    # For chosing gpu
    parser.add_argument('--gpu', action='store', type=int, help='chose which gpu to use', required=False, default=-1)

    # from DETR -----------------------------------------------------------------------------------------------------
    parser.add_argument('--lr_backbone', default=1e-5, type=float) 
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str, 
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--gelsight_backbone_path', default='none', type=str, required=False)
    parser.add_argument('--vision_backbone_path', default='none', type=str, required=False)

    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int, 
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=7, type=int, 
                        help="Number of decoding layers in the transformer")
    
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    
    main(vars(parser.parse_args()))


