# Visuo-Tactile Pretraining for Cable Plugging
This repo is the code for the paper found here: https://arxiv.org/abs/2403.11898

### Repo Structure
- ``imitate_episodes.py`` Train ACT, using either pretrained on non-pretrained encoders
- ``clip_pretraining.py`` Pretrains the Vision and Tactile Encoders using CLIP style contrastive loss
- ``robot_operation.py`` Executes trained policy on a Franka robot
- ``policy.py`` Creates the ACT policy
- ``clip_tsne.py`` Plots TSNE graphs of the pretrained embedding space.
- ``data_collection`` Folder containing data collection/processing scripts
- ``inspect_hdf5_file.py`` Contains helper functions for inspecting collected data.
- ``utils.py`` Dataloader + additional util functions
- ``visualization_utils.py`` Helper function to visualize trajectories durring training
- ``base_config.json`` Base config for training. Reduces the number of command line arguments needed. All values can be overridden in the command line.


### Installation
    conda create -n TactileACT python=3.8
    conda activate TactileACT
    pip install torchvision
    pip install torch
    pip install pyyaml
    pip install pexpect
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    pip install tqdm
    pip install opencv-python
    cd detr && pip install -e .

### Example Usages

To train ACT:

python imitate_episodes.py --config base_config.json --save_dir data/data_dir --name pretrained_vision_tactile --batch_size 4 --kl_weight 10 --z_dimension 32 --num_epochs 4000 --dropout 0.025 --chunk_size 30 --backbone clip_backbone --gelsight_backbone_path data/clip_models/gelsight_encoder.pth --vision_backbone_path data/clip_models/vision_encoder.pth


### Notes:
As the paper is under review, this repo is still under development and may change, and the code may not be fully documented.
If you have any questions on the repo, or want any advise on using visuo-tacitle pretraining for your own project, please do not hesitate to reach out to aigeorge@andrew.cmu.edu.
Enjoy!

