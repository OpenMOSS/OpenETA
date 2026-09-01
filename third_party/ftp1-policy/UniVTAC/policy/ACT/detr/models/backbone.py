# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
import os
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List, Literal
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython

e = IPython.embed


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                                             unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        backbone = getattr(torchvision.models,
                           name)(replace_stride_with_dilation=[False, False, dilation],
                                 pretrained=is_main_process(),
                                 norm_layer=FrozenBatchNorm2d)  # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):

    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_vision_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

class TactileBackbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str, ckpt: str, tac_names:list[str], train_backbone: bool, return_interm_layers: bool, position_embedding, tactile_type:Literal['feat', 'full']='feat'):
        super().__init__()
        
        self.train_backbone = train_backbone
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}

        from .network import Tactile

        self.tac_names = tac_names
        self.num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        backbone = Tactile(backbone='resnet18', supervise=[
            'marker', 'rgb', 'marked_rgb', 'pose', 'depth'
        ], latent_dims=self.num_channels, norm_layer=FrozenBatchNorm2d)
        if ckpt and Path(ckpt).exists():
            backbone.load_state_dict(torch.load(ckpt, weights_only=True))

        self.tactile_type = tactile_type
        if self.tactile_type == 'feat':
            self.backbone = backbone.backbone
            self.position_embedding = nn.Embedding(1, 512)
        else:
            self.backbone = IntermediateLayerGetter(backbone.backbone, return_layers=return_layers)
            self.position_embedding = position_embedding
 
    def forward(self, x):
        feat, pos = [], []
        if self.tactile_type == 'full':
            xs = self.backbone(x) # dict of feature maps [N, 512, 8, 8]
            for name, x in xs.items():
                feat.append(x)
                pos.append(self.position_embedding(x).to(x.dtype))
        else:
            xs = self.backbone(x) # [N, 512]
            feat.append(xs.unsqueeze(2).unsqueeze(3)) # [N, 512, 1, 1]
            pos.append(self.position_embedding.weight.unsqueeze(-1).unsqueeze(-1)) # [1, 512, 1, 1]
        return feat, pos

def build_tactile_backbone(args):
    train_backbone = args.lr_tactile_backbone > 0
    return_interm_layers = args.tactile_masks
    tactile_type = args.tactile_type if hasattr(args, 'tactile_type') else 'feat'
    position_embedding = build_position_encoding(args)
    backbone = TactileBackbone(args.tactile_backbone, args.tactile_ckpt, args.tactile_names, train_backbone, return_interm_layers, position_embedding, tactile_type)
    return backbone