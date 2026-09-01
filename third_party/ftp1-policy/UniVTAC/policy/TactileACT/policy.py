import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

import IPython
e = IPython.embed

import os
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from detrvae import DETRVAE
from detr.models.transformer import Transformer, TransformerEncoder, TransformerEncoderLayer
from detr.models.backbone import Backbone, Joiner, PositionEmbeddingLearned, PositionEmbeddingSine
from typing import Dict, List, Tuple
from visualization_utils import visualize_data, debug

class MyJoiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list):
        xs = self[0](tensor_list)
        out = [xs]
        pos = [self[1](xs).to(xs.dtype)]

        return out, pos

class ACTPolicy(nn.Module):
    def __init__(self,
                 state_dim: int,
                 hidden_dim: int,
                 position_embedding_type: str,
                 lr_backbone: float,
                 masks: bool,
                 backbone_type: str,
                 dilation: bool,
                 dropout: float,
                 nheads: int,
                 dim_feedforward: int,
                 num_enc_layers: int,
                 num_dec_layers: int,
                 pre_norm: bool,
                 num_queries: int,
                 camera_names,
                 z_dimension: int,
                 lr: float,
                 weight_decay: float,
                 kl_weight: float,
                 pretrained_backbones = None,
                 cam_backbone_mapping = None,
                 ):
        
        super().__init__()

        
        if cam_backbone_mapping is None:
            cam_backbone_mapping = {cam_name: 0 for cam_name in camera_names}
            num_backbones = 1
        else:
            num_backbones = len(set(cam_backbone_mapping.values()))

        if pretrained_backbones is not None:
            num_backbones = len(pretrained_backbones)

        # build model:
        # Build backbones:
        backbones = []
        for i in range(num_backbones):
            N_steps = hidden_dim // 2
            if position_embedding_type in ('v2', 'sine'):
                # TODO find a better way of exposing other arguments
                position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
            elif position_embedding_type in ('v3', 'learned'):
                position_embedding = PositionEmbeddingLearned(N_steps)
            else:
                raise ValueError(f"not supported {position_embedding_type}")
            
            train_backbone = lr_backbone > 0

            if pretrained_backbones is None:
                backbone = Backbone(name=backbone_type, 
                                    train_backbone=train_backbone, 
                                    return_interm_layers=masks, 
                                    dilation=dilation)
                backbone_model = Joiner(backbone, position_embedding)
                backbone_model.num_channels = backbone.num_channels
            else:
                backbone = pretrained_backbones[i]
                backbone_model = MyJoiner(backbone, position_embedding)
                backbone_model.num_channels = 512 #resnet18
            
            backbones.append(backbone_model)

        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_enc_layers,
            num_decoder_layers=num_dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=False,
        )

        # build encoder
        activation = "relu"

        encoder_layer = TransformerEncoderLayer(hidden_dim, nheads, dim_feedforward,
                                                dropout, activation, pre_norm)
        encoder_norm = nn.LayerNorm(hidden_dim) if pre_norm else None
        encoder = TransformerEncoder(encoder_layer, num_enc_layers, encoder_norm)

        self.model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=state_dim,
            num_queries=num_queries,
            camera_names=camera_names,
            z_dimension=z_dimension,
            cam_backbone_mapping=cam_backbone_mapping,
        )

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("number of parameters: %.2fM" % (n_parameters/1e6,))
        self.model.cuda()

        # build optimizer
        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": lr_backbone,
            },
        ]
        self.optimizer = torch.optim.AdamW(param_dicts, lr=lr,
                                    weight_decay=weight_decay)
        
        self.kl_weight = kl_weight
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos:torch.Tensor, images, actions=None, is_pad=None, z=None, ignore_latent=False):
        global debug
        env_state = None
        if actions is not None: # training time
            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, images, env_state, actions, is_pad, debug=debug.print, ignore_latent=ignore_latent)
            
            visualize_data([img[0] for img in images], qpos[0], a_hat[0], is_pad[0], actions[0])

            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, images, env_state, z=z) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu:torch.Tensor, logvar:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class ACT:
    def __init__(self, args, backbones):
        act_args = {
            'state_dim': args['state_dim'],
            'hidden_dim': args['hidden_dim'],
            'position_embedding_type': args['position_embedding'],
            'lr_backbone': float(args['lr_backbone']),
            'masks': args['masks'],
            'backbone_type': args['backbone'],
            'dilation': args['dilation'],
            'dropout': args['dropout'],
            'nheads': args['nheads'],
            'dim_feedforward': args['dim_feedforward'],
            'num_enc_layers': args['enc_layers'],
            'num_dec_layers': args['dec_layers'],
            'pre_norm': args['pre_norm'],
            'num_queries': args['chunk_size'],
            'camera_names': args['camera_names'],
            'z_dimension': args['z_dimension'],
            'lr': float(args['lr']),
            'weight_decay': float(args['weight_decay']),
            'kl_weight': float(args['kl_weight']),
            'pretrained_backbones': backbones,
            'cam_backbone_mapping': args['cam_backbone_mapping']
        }

        self.policy = ACTPolicy(**act_args)
        self.device = torch.device(args["device"])
        self.policy.to(self.device)
        self.policy.eval()

        # Temporal aggregation settings
        self.temporal_agg = args.get("temporal_agg", False)
        self.num_queries = args["chunk_size"]
        self.state_dim = args.get("state_dim", 14)  # TacArena: read from args
        self.max_timesteps = 3000  # Large enough for deployment
        self.camera_names = args.get("camera_names", ["cam_high"])  # TacArena: read from args

        # Set query frequency based on temporal_agg - matching imitate_episodes.py logic
        self.query_frequency = self.num_queries
        if self.temporal_agg:
            self.query_frequency = 1
            # Initialize with zeros matching imitate_episodes.py format
            self.all_time_actions = torch.zeros([
                self.max_timesteps,
                self.max_timesteps + self.num_queries,
                self.state_dim,
            ]).to(self.device)
            print(f"Temporal aggregation enabled with {self.num_queries} queries")

        self.t = 0  # Current timestep

        # Load statistics for normalization
        ckpt_dir = args.get("ckpt_dir", "")
        if ckpt_dir:
            # Load dataset stats for normalization
            stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
            if os.path.exists(stats_path):
                with open(stats_path, "rb") as f:
                    self.stats = pickle.load(f)
                print(f"Loaded normalization stats from {stats_path}")
            else:
                print(f"Warning: Could not find stats file at {stats_path}")
                self.stats = None

            # Load policy weights
            ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
            print("current pwd:", os.getcwd())
            if os.path.exists(ckpt_path):
                loading_status = self.policy.load_state_dict(torch.load(ckpt_path))
                print(f"Loaded policy weights from {ckpt_path}")
                print(f"Loading status: {loading_status}")
            else:
                 print(f"Warning: Could not find policy checkpoint at {ckpt_path}")
        else:
            self.stats = None

    def pre_process(self, qpos):
        """Normalize input joint positions"""
        # if self.stats is not None:
        #     return (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        # return qpos
        # TODO: ?? mismatch with the training pipeline
        """Normalize input joint positions. 与训练一致：EpisodicDataset 使用 NormalizeActionQpos，qpos/action 用同一组 mean/std。"""
        if self.stats is not None:
            mean = (self.stats["qpos_mean"] + self.stats["action_mean"]) / 2
            std = (self.stats["qpos_std"] + self.stats["action_std"]) / 2
            return (qpos - mean) / std

    def post_process(self, action):
        """Denormalize model outputs"""
        # if self.stats is not None:
        #     return action * self.stats["action_std"] + self.stats["action_mean"]
        # return action
        # TODO: ?? mismatch with the training pipeline
        """Denormalize model outputs. 与训练一致：逆变换使用同一组 mean/std。"""
        if self.stats is not None:
            mean = (self.stats["qpos_mean"] + self.stats["action_mean"]) / 2
            std = (self.stats["qpos_std"] + self.stats["action_std"]) / 2
            return action * std + mean

    def get_action(self, obs=None):
        if obs is None:
            return None

        # Convert observations to tensors and normalize qpos - matching imitate_episodes.py
        qpos_numpy = np.array(obs["qpos"])
        qpos_normalized = self.pre_process(qpos_numpy)
        qpos = torch.from_numpy(qpos_normalized).float().to(self.device).unsqueeze(0)

        # Prepare images following imitate_episodes.py pattern
        # Stack images from all cameras
        cam_image = []
        for cam_name in self.camera_names:
            cam_image.append(obs[cam_name])
        cam_image = torch.stack(cam_image, dim=0).to(self.device).unsqueeze(0)

        with torch.no_grad():
            # Only query the policy at specified intervals - exactly like imitate_episodes.py
            if self.t % self.query_frequency == 0:
                self.all_actions = self.policy(qpos, cam_image)
                
                print(f"all_actions_shape: {self.all_actions.shape}, freq = {self.query_frequency}")
                # print(f"all_actions: {self.all_actions}")

            if self.temporal_agg:
                # Match temporal aggregation exactly from imitate_episodes.py
                self.all_time_actions[[self.t], self.t:self.t + self.num_queries] = (self.all_actions)
                actions_for_curr_step = self.all_time_actions[:, self.t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]

                # Use same weighting factor as in imitate_episodes.py
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = (torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1))

                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                # Direct action selection, same as imitate_episodes.py
                raw_action = self.all_actions[:, self.t % self.query_frequency]

        # Denormalize action
        raw_action = raw_action.cpu().numpy()
        action = self.post_process(raw_action)

        self.t += 1
        return action

    def reset(self):
        """Reset temporal aggregation state and timestep counter"""
        self.t = 0
        if self.temporal_agg:
            self.all_time_actions = torch.zeros([
                self.max_timesteps,
                self.max_timesteps + self.num_queries,
                self.state_dim,
            ]).to(self.device)