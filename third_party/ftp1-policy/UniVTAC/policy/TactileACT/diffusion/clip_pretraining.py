import torchvision
from torch import nn
import torch
from typing import Tuple, Sequence, Dict, Union, Optional, Callable, List
from torch.utils.data import DataLoader

def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, 'resnet18')
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

# the projection head for CLIP. I'm using resnet's approach of an average pooling layer followed by a linear layer.
class clip_projection_head(nn.Module):
    def __init__(self, out_dim: int, conditioning_dim: int = 0, num_channels:int = 512):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1, -1)
        # print('conditioning_dim:', conditioning_dim)
        self.linear = nn.Linear(num_channels + conditioning_dim, out_dim)
    
    def forward(self, feature_map, conditioning=None) -> torch.Tensor:
        # print('feature_map:', feature_map.shape)
        x = self.pooling(feature_map)
        x = self.flatten(x)
        # print('post pooling:', x.shape)
        if conditioning is not None:
            x = torch.cat((x, conditioning), dim=-1)

        return self.linear(x)

def modified_resnet18(features_per_group=16) -> nn.Module:
    # get a resnet18 model
    resnet18 = getattr(torchvision.models, 'resnet18')()

    # remove the final fully connected layer and average pooling
    resnet18 = nn.Sequential(*list(resnet18.children())[:-2])

    # replace all BatchNorm with GroupNorm
    resnet18 = replace_bn_with_gn(resnet18, features_per_group=features_per_group)
    return resnet18    

import h5py
import os
import cv2
from torchvision.transforms import Normalize
import numpy as np

class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 episode_ids: List[int], 
                 dataset_dir: str, 
                 camera_names: List[str], 
                 norm_stats: Dict[str, Union[float, np.ndarray]],
                 image_size: Tuple[int, int] = None, 
                 gelsight_size: Tuple[int, int] = None):
        super(ClipDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.image_size = image_size # image size in (H, W)
        self.n_cameras = len(camera_names)

        assert "gelsight_mean" in norm_stats, "gelsight data must exist"

        gelsight_mean = norm_stats["gelsight_mean"]
        gelsight_std = norm_stats["gelsight_std"]
        self.position_mean = norm_stats["qpos_mean"]
        self.position_std = norm_stats["qpos_std"]

        # image normalization for resnet. 
        self.image_normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.gelsight_normalize = Normalize(mean=gelsight_mean, std=gelsight_std)

        # get the length of each episode
        self.episode_lengths = []
        for episode_id in self.episode_ids:
            dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
            with h5py.File(dataset_path, 'r') as root:
                self.episode_lengths.append(root.attrs['num_timesteps'])
                if self.image_size is None:
                    self.image_size = (root.attrs['image_height'], root.attrs['image_width'])
                if gelsight_size is None:
                    self.gelsight_size = (root.attrs['gelsight_height'], root.attrs['gelsight_width'])   

        # create a lookup table for episode_id and timestep given an index
        self.index_lookup = []
        for episode, length in zip(self.episode_ids, self.episode_lengths):
            for timestep in range(length):
                self.index_lookup.append((episode, timestep))

    def __len__(self):
        return len(self.index_lookup)

    def __getitem__(self, index):
        debug_state = "start"
        try:
            episode_id, timestep = self.index_lookup[index]
            dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
            debug_state = "before open file"
            with h5py.File(dataset_path, 'r') as root:
                debug_state = 'after open file'
                all_cam_images = []

                for cam_name in self.camera_names:
                    debug_state = f'before get image, cam {cam_name}, timestep {timestep}'
                    image = root[f'/observations/images/{cam_name}'][timestep]
                    debug_state = 'after get image'
                    # resize image
                    if self.image_size != image.shape[:2]:
                        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
                    debug_state = 'after resize image'
                    
                    # convert to tensor
                    image = torch.tensor(image, dtype=torch.float32)/255.0
                    image = torch.einsum('h w c -> c h w', image) # change to c h w
                    debug_state = 'after einsum'

                    # normalize image
                    image = self.image_normalize(image)
                    all_cam_images.append(image)
                debug_state = 'after loop through cameras'

                images = torch.stack(all_cam_images, axis=0)
                debug_state = 'after stack images'
                
                # get gelsight data
                gelsight_data = root['observations/gelsight/depth_strain_image'][timestep]
                debug_state = 'after get gelsight data'

                # resize gelsight data
                if self.gelsight_size != gelsight_data.shape[:2]:
                    gelsight_data = cv2.resize(gelsight_data, (self.gelsight_size[1], self.gelsight_size[0]))

                debug_state = 'after resize gelsight data'
                
                # convert to tensor
                gelsight_data = torch.tensor(gelsight_data, dtype=torch.float32)
                gelsight_data = torch.einsum('h w c -> c h w', gelsight_data) # change to c h w

                debug_state = 'after einsum gelsight data'

                # normalize gelsight data
                gelsight_data = self.gelsight_normalize(gelsight_data)

                debug_state = 'after normalize gelsight data'

                # get qpos and normalize
                position = root['observations/qpos'][timestep]
                position = (position - self.position_mean) / self.position_std

                debug_state = 'after get qpos'

                # don't include the last element, which is the gripper
                position = torch.tensor(position[:3], dtype=torch.float32)
                debug_state = 'after tensor qpos'
        except:
            print('Error in dataloar get item index:', index)
            print('Debug state:', debug_state)
            raise

        return images, gelsight_data, position

import torch.nn.functional as F
def clip_loss(image_embeddings, gelsight_embeddings, target_matrix, logit_scale = 1.0, visualize = False):
    image_targets = target_matrix
    gelsight_targets = target_matrix.T

    n_cameras = image_embeddings.shape[1]

    loss = torch.empty(n_cameras).to(image_embeddings.device)
    avg_softmax_maps = []
    for i in range(n_cameras):
        image_logits = logit_scale * image_embeddings[:, i] @ gelsight_embeddings.T
        gelsight_logits = logit_scale * gelsight_embeddings @ image_embeddings[:, i].T

        if visualize:
            image_softmax = F.softmax(image_logits.clone().detach(), dim=1).cpu().numpy()
            gelsight_softmax = F.softmax(gelsight_logits.clone().detach(), dim=1).cpu().numpy()
            avg_softmax_maps.append((image_softmax + gelsight_softmax)/2.0)

        image_loss = F.cross_entropy(image_logits, gelsight_targets)
        gelsight_loss = F.cross_entropy(gelsight_logits, image_targets)

        loss[i] = ((image_loss + gelsight_loss)/2.0).mean()

    return loss, avg_softmax_maps

from tqdm import tqdm

def clip_pretraining(train_loader: DataLoader,
                     test_loader: DataLoader,
                     device: torch.device,
                     clip_dim: int = 512,
                     features_per_group: int = 16,
                     resnet_lr: float = 1e-5,
                     projection_lr: float = 1e-4,):
    
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    # get the camera, gelsight, and state dimensions from the dataset
    dataset:ClipDataset = train_loader.dataset
    n_cameras = dataset.n_cameras
    camera_sizes = [dataset.image_size]*n_cameras
    gelsight_size = dataset.gelsight_size
    state_size = 3

    # get resnet models for each camera
    # get a resnet18 model
    vision_encoder = modified_resnet18(weights=None, features_per_group=features_per_group).to(device)

    # # check the output dimension of the resnet18
    # test_input = torch.randn(1, 3, camera_sizes[i][0], camera_sizes[i][1]).to(device)
    # with torch.no_grad():
    #     test_output = vision_encoder(test_input)
    # out_dim = test_output.shape[1]*test_output.shape[2]*test_output.shape[3]

    # create a projection head
    vision_projection = clip_projection_head(out_dim=clip_dim).to(device)

    # get a resnet18 model for gelsight
    gelsight_encoder = modified_resnet18(weights=None, features_per_group=features_per_group).to(device)

    # check the output dimension of the resnet18
    # test_input = torch.randn(1, 3, gelsight_size[0], gelsight_size[1]).to(device)
    # with torch.no_grad():
    #     test_output = gelsight_encoder(test_input)
    # print('gelsight test size', test_output.shape)
    # out_dim = test_output.shape[1]*test_output.shape[2]*test_output.shape[3]

    # create a projection head, conditioned on state
    gelsight_projection = clip_projection_head(out_dim=clip_dim, conditioning_dim=state_size).to(device)

    optim_params = [{"params": gelsight_encoder.parameters(), "lr": resnet_lr},
                    {"params": gelsight_projection.parameters(), "lr": projection_lr},
                    {"params": vision_encoder.parameters(), "lr": resnet_lr},
                    {"params": vision_projection.parameters(), "lr": projection_lr}]

    print('optim_params:', optim_params)

    optimizer = torch.optim.Adam(optim_params)
    
    n_epochs = 100
    training_losses = np.empty([n_epochs, n_cameras])
    testing_losses = np.empty([n_epochs, n_cameras])
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
    # train the model
        training_loss = np.zeros(n_cameras)

        gelsight_encoder.train()
        gelsight_projection.train()
        vision_encoder.train()
        vision_projection.train()
        for batch_idx, (images, gelsight, position) in tqdm(enumerate(train_loader), total=len(train_loader)):
            images = images.to(device)
            gelsight = gelsight.to(device)
            position = position.to(device)

            # forward pass
            
            batch_size = images.shape[0]
            # images are in form batch, camera, c, h, w. We want to flatten the batch and camera dimensions
            images = images.view(-1, images.shape[2], images.shape[3], images.shape[4])
            image_embeddings = vision_projection(vision_encoder(images))
            
            # now reshape the image_embeddings to be batch, camera, clip_dim
            image_embeddings = image_embeddings.view(batch_size, n_cameras, clip_dim)

            gelsight_embeddings = gelsight_projection(gelsight_encoder(gelsight), position)

            # calculate target matrix
            target_matrix = torch.eye(position.shape[0]).to(device)

            # calculate loss - vector of per-camera losses
            if batch_idx == 0: # visualize the first batch in each epoch
                loss, avg_softmax_maps = clip_loss(image_embeddings, gelsight_embeddings, target_matrix, visualize=True)
                try:
                    for cam_num, softmax_map in enumerate(avg_softmax_maps):
                        plt.figure()
                        plt.imshow(softmax_map)
                        plt.colorbar()
                        plt.title(f'Average Softmax Map, Epoch {epoch}, Cam {cam_num} - Train')
                        plt.savefig(f'clip_graphs/epoch_{epoch}_cam_{cam_num}_train.png')
                        plt.close()
                except:
                    print('Error in train plots')
                    raise
            else:
                loss, _ = clip_loss(image_embeddings, gelsight_embeddings, target_matrix, visualize=False)
            training_loss += loss.clone().detach().cpu().numpy()
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
        training_losses[epoch] = training_loss/len(train_loader)

        # test the model
        gelsight_encoder.eval()
        gelsight_projection.eval()
        vision_encoder.eval()
        vision_projection.eval()

        test_loss = np.zeros(n_cameras)
        with torch.no_grad():
            for batch_idx, (images, gelsight, position) in tqdm(enumerate(test_loader), total=len(test_loader)):
                images = images.to(device)
                gelsight = gelsight.to(device)
                position = position.to(device)

                # forward pass
                batch_size = images.shape[0]
                # images are in form batch, camera, c, h, w. We want to flatten the batch and camera dimensions
                images = images.view(-1, images.shape[2], images.shape[3], images.shape[4])
                image_embeddings = vision_projection(vision_encoder(images))
                # print('image_embeddings:', image_embeddings.shape)
                
                # now reshape the image_embeddings to be batch, camera, clip_dim
                image_embeddings = image_embeddings.view(batch_size, n_cameras, clip_dim)

                gelsight_embeddings = gelsight_projection(gelsight_encoder(gelsight), position)

                # calculate target matrix
                target_matrix = torch.eye(position.shape[0]).to(device)

                # calculate loss - vector of per-camera losses
                            # calculate loss - vector of per-camera losses
                if batch_idx == 0: # visualize the first batch in each epoch
                    loss, avg_softmax_maps = clip_loss(image_embeddings, gelsight_embeddings, target_matrix, visualize=True)
                    try:
                        for cam_num, softmax_map in enumerate(avg_softmax_maps):
                            plt.figure()
                            plt.imshow(softmax_map)
                            plt.colorbar()
                            plt.title(f'Average Softmax Map, Epoch {epoch}, Cam {cam_num} - Test')
                            plt.savefig(f'clip_graphs/epoch_{epoch}_cam_{cam_num}_test.png')
                            plt.close()
                    except:
                        print('Error in test plots')
                        raise
                else:
                    loss, _ = clip_loss(image_embeddings, gelsight_embeddings, target_matrix, visualize=False)
                test_loss += loss.clone().detach().cpu().numpy()
        testing_losses[epoch] = test_loss/len(test_loader)


        # plot the training and testing losses
        plt.figure()
        for i in range(n_cameras):
            plt.plot(training_losses[:epoch+1, i], label=f'camera {i+1} train', c=f'C{i}')
            plt.plot(testing_losses[:epoch+1, i], label=f'camera {i+1} test', linestyle='dashed', c=f'C{i}')
        plt.legend(loc='best')
        plt.title(f'Training and Testing Loss - Epoch {epoch+1}/{n_epochs}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('clip_graphs/training_loss.png')
        plt.close()

    return vision_encoder, vision_projection, gelsight_encoder, gelsight_projection
    

def test():
    from utils import get_norm_stats
    num_episodes = 20
    dataset_dir = "Processed_Data_2/data/"
    camera_names = ['1', '2', '3', '4', '5']
    norm_stats = get_norm_stats(dataset_dir, num_episodes, use_existing=True)
    batch_size_train = 10
    batch_size_test = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    train_dataset = ClipDataset(train_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    test_dataset = ClipDataset(val_indices, dataset_dir, camera_names, norm_stats)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    # test dataloader
    for i, (images, gelsight, position) in enumerate(train_dataloader):
        print(images.shape, gelsight.shape, position.shape)
        break
        
    #clip_pretraining(train_dataloader, test_dataloader, device, clip_dim=512, features_per_group=16)