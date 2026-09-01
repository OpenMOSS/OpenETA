import torch
import torch.nn as nn
from torchvision import models

from typing import Literal

class RGBDecoder(nn.Module):
    def __init__(self, latent_dims=512, output_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dims, 256 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1), # 8x8 -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 128, kernel_size=4, stride=2, padding=1), # 16x16 -> 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 128, kernel_size=4, stride=2, padding=1), # 32x32 -> 64x64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 128, kernel_size=4, stride=2, padding=1), # 64x64 -> 128x128
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear'), # 128x128 -> 256x256
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 8, 8)
        x = self.deconv(x)
        return x

class MarkerDecoder(nn.Module):
    def __init__(self, latent_dims=512, marker_nums=63):
        super().__init__()
        self.marker_nums = marker_nums
        self.ffn = nn.Sequential(
            nn.Linear(latent_dims, 256),
            nn.GELU(),
            nn.Linear(256, marker_nums * 2)
        )

    def forward(self, x):
        x = self.ffn(x)
        x = x.view(-1, self.marker_nums, 2)
        return x 

class PoseDecoder(nn.Module):
    def __init__(self, latent_dims=512, pose_dims=7):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(latent_dims, 256),
            nn.GELU(),
            nn.Linear(256, pose_dims)
        )

    def forward(self, x):
        x = self.ffn(x)
        return x

class Tactile(nn.Module):
    def __init__(
        self, backbone='resnet18', latent_dims=512,
        supervise:list[Literal['marker', 'rgb', 'marked_rgb', 'pose', 'depth']]=['marked_rgb'],
        norm_layer=nn.BatchNorm2d
    ):
        super().__init__()

        if backbone == 'resnet18':
            self.backbone = models.resnet18(num_classes=latent_dims, norm_layer=norm_layer)
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(num_classes=latent_dims, norm_layer=norm_layer)
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(num_classes=latent_dims, norm_layer=norm_layer)
        
        self.supervise = supervise
        self.decoders = nn.ModuleDict()
        if 'rgb' in self.supervise:
            self.decoders['rgb'] = RGBDecoder(latent_dims=latent_dims, output_channels=3)
        if 'marked_rgb' in self.supervise:
            self.decoders['marked_rgb'] = RGBDecoder(latent_dims=latent_dims, output_channels=3)
        if 'depth' in self.supervise:
            self.decoders['depth'] = RGBDecoder(latent_dims=latent_dims, output_channels=1)
        if 'marker' in self.supervise:
            self.decoders['marker'] = MarkerDecoder(latent_dims=latent_dims, marker_nums=63)
        if 'pose' in self.supervise:
            self.decoders['pose'] = PoseDecoder(latent_dims=latent_dims, pose_dims=7)
    
    def reconstruct(self, img):
        latent = self.forward(img)
        outputs = {}
        for key in self.supervise:
            outputs[key] = self.decoders[key](latent)
        return outputs     

    def forward(self, x):
        return self.backbone(x)
    
    def loss(self, outputs:dict, targets:dict, weights:dict=None):
        loss = 0
        loss_dict = {}
        criterion = nn.MSELoss()
        for key in self.supervise:
            part_loss = criterion(outputs[key], targets[key])
            loss += weights.get(key, 1.0) * part_loss
            loss_dict[key] = part_loss.item()
        loss_dict['total'] = loss.item()
        return loss, loss_dict