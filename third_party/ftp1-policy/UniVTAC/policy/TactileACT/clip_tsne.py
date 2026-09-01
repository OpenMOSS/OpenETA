import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from clip_pretraining import modified_resnet18, ClipProjectionHead, ClipDataset
from utils import get_norm_stats
import torch
from torch.nn import functional as F
import h5py
from tqdm import tqdm
from typing import List

"""
This file is used to visualize the latent vectors of the images and gelsight 
data using t-SNE. Its useful to see how well the pre-training works.
"""

def plot_tsne(all_image_vectors:np.ndarray, gelsight_vectors:np.ndarray, timestamps):
    """
    Plots the t-SNE visualization of the latent vectors of the images and gelsight data.
    all_image_vectors: np.ndarray of shape (n_cam, episode_len, 512). The latent vectors of the images.
    gelsight_vectors: np.ndarray of shape (episode_len, 512). The latent vectors of the gelsight data.
    timestamps: np.ndarray of shape (episode_len). The timestamps of the data.
    """
    n_cam = all_image_vectors.shape[0]
    episode_len = len(timestamps)
    tsne = TSNE(n_components=2, random_state=10) # create TSNE object


    # concatenate all the latent vectors, including the gelsight vectors
    all_latent_vectors = np.concatenate([all_image_vectors[i] for i in range(n_cam)] + [gelsight_vectors], axis=0)

    # fit the t-SNE model to the latent vectors
    embedded = tsne.fit_transform(all_latent_vectors)

    # get the image and gelsight TSNE embeddings
    image_embedings = [embedded[i*episode_len:(i+1)*episode_len] for i in range(n_cam)]
    gelsight_embedded = embedded[n_cam*episode_len:]

    # plot the t-SNE embeddings, use color to represent the timestamps
    plt.figure(figsize=(8, 5))
    markers = ['o', 's', '*', 'D', 'P', 'H']
    for i, image_embedded in enumerate(image_embedings):
        plt.scatter(image_embedded[:, 0], image_embedded[:, 1], c=timestamps/len(timestamps), cmap='viridis', marker=markers[i], s=50, alpha=0.25, label=f'Image Vectors {i+1}')
    plt.scatter(gelsight_embedded[:, 0], gelsight_embedded[:, 1], c=timestamps/len(timestamps), cmap='viridis', marker='x', s=50, alpha=0.25, label='Gelsight Vectors')

    plt.title('t-SNE Visualization of Latent Vectors')
    # put legend outside of plot
    plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.colorbar(label='Timestamp')
    plt.grid(True)
    plt.tight_layout() 


def plot_run_similarity(all_vectors, name):
    """
    Plots intra-run similarity matrix of the latent vectors
    all_vectors: np.ndarray of shape (episode_len, 512). The latent vectors."""
    episode_len = all_vectors.shape[0]
    similarity = np.zeros((episode_len, episode_len))
    for i in range(episode_len):
        for j in range(episode_len):
            # cosine similarity
            normalized_i = all_vectors[i]/np.linalg.norm(all_vectors[i])
            normalized_j = all_vectors[j]/np.linalg.norm(all_vectors[j])
            similarity[i, j] = np.dot(normalized_i, normalized_j)
            

    plt.figure()
    plt.imshow(similarity, cmap='viridis')
    plt.colorbar()
    plt.title('Similarity Matrix for ' + name)
    plt.xlabel('Timestep')
    plt.ylabel('Timestep')


if __name__ == "__main__":
    
    # Control variables for the script
    encoder_pretrained = True # whether to use the pre-trained encoders. If False, the encoders will be resnet18
    projection_head_pretrained = True # whether to use the pre-trained projection heads. If False, the projection heads will be randomly initialized
    use_projection_head = True # whether to use the projection head. If False, will compare the direct resnet18 encodings, not the clip encodings
    use_act = False # whether to use the encoders fine-tuned with ACT

    # clip model paths
    image_encoder_path = "/home/aigeorge/research/TactileACT/data/camera_cage_new_mount/clip_models/normalized/4/epoch_1399_vision_encoder.pth"
    image_projection_head_path = "/home/aigeorge/research/TactileACT/data/camera_cage_new_mount/clip_models/normalized/4/epoch_1399_vision_projection.pth"
    gelsight_encoder_path = "/home/aigeorge/research/TactileACT/data/camera_cage_new_mount/clip_models/normalized/4/epoch_1399_gelsight_encoder.pth"
    gelsight_projection_head_path = "/home/aigeorge/research/TactileACT/data/camera_cage_new_mount/clip_models/normalized/4/epoch_1399_gelsight_projection.pth"

    # ACT model path. Used if you want to examine the fine-tuned encoders
    ACT_model_path = "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/ACT/pretrain_both_20/policy_last.ckpt"
    
    dataset_dir = "/home/aigeorge/research/TactileACT/data/camera_cage_new_mount/data"
    num_episodes = 100

    # name of the cameras in the dataset
    camera_names = ["1", "2", "3", "4", "5", "6"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create the vision and tactile encode
    if use_act:
        from load_ACT import load_ACT
        act = load_ACT(ACT_model_path)
        vision_encoder = act.model.backbones[0][0]
        gelsight_encoder = act.model.backbones[1][0]
        
    else:
        vision_encoder = modified_resnet18().to(device)
        if encoder_pretrained:
            vision_encoder.load_state_dict(torch.load(image_encoder_path, map_location=device))

        # create the gelsight encoder and projection head
        gelsight_encoder = modified_resnet18().to(device)
        if encoder_pretrained:
            gelsight_encoder.load_state_dict(torch.load(gelsight_encoder_path, map_location=device))

    if use_projection_head:
        vision_projection_head = ClipProjectionHead(512).to(device)
        if projection_head_pretrained:
            vision_projection_head.load_state_dict(torch.load(image_projection_head_path, map_location=device))

        gelsight_projection_head = ClipProjectionHead(512, 3).to(device)
        # gelsight_projection_head = ClipProjectionHead(512).to(device)
        if  projection_head_pretrained:
            gelsight_projection_head.load_state_dict(torch.load(gelsight_projection_head_path, map_location=device))

    vision_encoder.eval()
    gelsight_encoder.eval()
    if use_projection_head:
        vision_projection_head.eval()
        gelsight_projection_head.eval()

    norm_stats = get_norm_stats(dataset_dir, num_episodes, use_existing=True)
    dataset = ClipDataset(list(range(num_episodes)), dataset_dir, camera_names, norm_stats)

    episode_idxs = [0, 3] # 0 is training, 3 is testing
    for episode_idx in episode_idxs:
        episode_len = dataset.episode_lengths[episode_idx]

        all_image_vectors = np.empty((len(camera_names), episode_len, 512))
        gelsight_vectors = np.empty((episode_len, 512))
        all_image_encodings = np.empty((len(camera_names), episode_len, 512))
        gelsight_encodings = np.empty((episode_len, 512))
        timestamps = []

        with torch.no_grad():
            for t in tqdm(range(episode_len)):
                images = [dataset.get_image(episode_idx, t, cam) for cam in camera_names]
                gelsight = dataset.get_gelsight(episode_idx, t)
                position = dataset.get_position(episode_idx, t)
                # position = torch.zeros_like(position).to(device)

                # get the endodings and clip projections for the images
                for i, image in enumerate(images):
                    encoding = vision_encoder(image.to(device).unsqueeze(0))
                    if use_projection_head:
                        # get the encoding and pass through the projection head
                        image_vector = vision_projection_head(encoding)
                        all_image_vectors[i, t, :] = image_vector.detach().cpu().numpy().squeeze()
                    else:
                        # only save the encoding, passed through an average 
                        # pooling layer (first layer of projection head)
                        encoding = F.adaptive_avg_pool2d(encoding, (1, 1))
                        all_image_encodings[i, t, :] = encoding.detach().cpu().numpy().squeeze()

                # get the encoding and clip projection for the gelsight data
                encoding = gelsight_encoder(gelsight.to(device).unsqueeze(0))
                if use_projection_head:
                    gelsight_vector = gelsight_projection_head(encoding, position.to(device).unsqueeze(0))
                    gelsight_vectors[t, :] = gelsight_vector.detach().cpu().numpy().squeeze()
                else:
                    encoding = F.adaptive_avg_pool2d(encoding, (1, 1))
                    gelsight_encodings[t, :] = encoding.detach().cpu().numpy().squeeze()
                timestamps.append(t) # can be replaced with actual timestamps

        timestamps = np.array(timestamps)

        # if not using the projection head, use the encodings
        if not use_projection_head:
            all_image_vectors = all_image_encodings
            gelsight_vectors = gelsight_encodings

        # plot the similarity matrix for the latent vectors (intra-run similarity)
        plot_run_similarity(gelsight_vectors, 'Gelsight')
        for i, embedding in enumerate(all_image_vectors):
            plot_run_similarity(embedding, 'camera ' + str(i+1))

        # plot the t-SNE visualization of the latent vectors
        plot_tsne(all_image_vectors, gelsight_vectors, timestamps)
        plt.title(f't-SNE Visualization of Latent Vectors for Episode {episode_idx}')
        plt.show()
        
    plt.show()