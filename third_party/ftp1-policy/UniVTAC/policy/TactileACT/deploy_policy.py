import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from .._base_policy import BasePolicy

import os
import cv2
import json
import numpy as np
import torch
from .policy import ACT
from torchvision import transforms
    
class Policy(BasePolicy):
    def __init__(self, args):
        """Initialize ACT policy for TacArena deployment"""
        # Construct checkpoint directory path
        ckpt_dir = Path(__file__).parent / "act-ckpt" / f"{args['task_name']}-{args['task_config']}"
        
        self.task_name = args['task_name']
        with open(Path(__file__).parent.parent / 'task_settings.json', 'r') as f:
            task_settings = json.load(f)
        assert self.task_name in task_settings, f"Task '{self.task_name}' not found in task_settings.json"
        self.camera_type = task_settings[self.task_name].get('camera_type', 'head')
        print(f"Using camera type '{self.camera_type}' for task '{self.task_name}'")
        
        args['ckpt_dir'] = str(ckpt_dir)
        args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.camera_type == 'all':
            args['camera_names'] = ['cam_high', 'cam_wrist', 'cam_left_tactile', 'cam_right_tactile']
            args['cam_backbone_mapping'] = {'cam_high': 0, 'cam_wrist': 0, 'cam_left_tactile': 1, 'cam_right_tactile': 1}
        else:
            args['camera_names'] = ['cam_high', 'cam_left_tactile', 'cam_right_tactile']
            args['cam_backbone_mapping'] = {'cam_high': 0, 'cam_left_tactile': 1, 'cam_right_tactile': 1}

        from clip_pretraining import modified_resnet18
        gelsight_model = modified_resnet18()
        vision_model = modified_resnet18()
        backbones = [vision_model, gelsight_model]

        self.model = ACT(args, backbones)

    def encode_obs(self, observation):
        def tactile_transform(img:torch.Tensor):
            img = transforms.Resize((256, 256))(img.permute(2, 0, 1)) / 255.0
            img = transforms.Normalize(
                mean=self.model.stats['gelsight_mean'],
                std=self.model.stats['gelsight_std']
            )(img)
            return img
        def camera_transform(img:torch.Tensor):
            img = transforms.Resize((256, 256))(img.permute(2, 0, 1)) / 255.0
            img = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(img)
            return img

        if self.camera_type == 'all':
            cam_high = camera_transform(observation["observation"]["head"]["rgb"])
            cam_wrist = camera_transform(observation["observation"]["wrist"]["rgb"])
        else:
            cam_high = camera_transform(observation["observation"][self.camera_type]["rgb"])
        left_tac = tactile_transform(observation["tactile"]["left_tactile"]["rgb_marker"])               # ???? TODO: original: left_gsmini ? mis-match with the sim
        right_tac = tactile_transform(observation["tactile"]["right_tactile"]["rgb_marker"])             # ???? TODO: original: right_gsmini ? mis-match with the sim
        
        # Extract joint positions (8D: 7 arm + 1 gripper)
        qpos = observation["embodiment"]["joint"][:8]

        ret = {
            "cam_high": cam_high,
            "cam_left_tactile": left_tac,
            "cam_right_tactile": right_tac,
            "qpos": qpos.cpu().numpy()
        }
        if self.camera_type == 'all':
            ret["cam_wrist"] = cam_wrist
        return ret

    def eval(self, task, observation):
        """
        Evaluate ACT policy on TacArena task
        
        Args:
            task: TacArena BaseTask instance
            observation: Current observation from environment
        """
        obs = self.encode_obs(observation)
        
        # Get action from ACT model (returns (1, 8) numpy array)
        with torch.no_grad():  # Disable gradient computation to prevent memory leak
            actions = self.model.get_action(obs)

        for action in actions:
            if self.model.t % 10 == 0:
                self.save(task.get_frame_shot(observation), task.take_action_cnt)
            # Execute action in environment
            action = torch.from_numpy(action).float().flatten().cuda()
            exec_succ, eval_succ = task.take_action(action, action_type='qpos')

    def reset(self):
        """Reset ACT model state (temporal aggregation and timestep counter)"""
        if hasattr(self.model, 'reset'):
            self.model.reset()

    def save(self, img, t):
        from PIL import Image
        from PIL import ImageDraw, ImageFont
        
        obs = Image.fromarray(img.cpu().numpy())

        draw = ImageDraw.Draw(obs)
        font = ImageFont.load_default()

        draw.text((obs.width-100, obs.height-60), f'{t:03d}', fill=(255, 0, 0), font=font)
        obs.save(f'TactileACT_{self.task_name}.png')
