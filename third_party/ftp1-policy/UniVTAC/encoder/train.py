import json
import time
import torch
import torch.nn as nn
from pathlib import Path

from torchvision import transforms
import matplotlib.pyplot as plt

from network import *
from dataloader import *
from torch.utils.data import random_split

data_length = 1000
backbone = 'resnet18'
ckpt_root = 'ablation/data_1000'
weights = {}
timestr = time.strftime(r"%Y%m%d-%H%M%S", time.localtime())
def log(msg):
    global backbone, timestr, ckpt_root
    save_path = Path(f'{ckpt_root}/{backbone}/{timestr}/log.log')
    with open(save_path, 'a') as f:
        f.write(msg + '\n')

def train(
    train_loader:DataLoader,
    valid_loader:DataLoader,
    epoches=5,
    lr=1e-4,
    loss_weights:dict={'rgb_marked': 1.0},
):
    global backbone, timestr, ckpt_root, weights
    save_root = Path(f'{ckpt_root}/{backbone}/{timestr}')
    save_root.mkdir(parents=True, exist_ok=True)
    log(json.dumps({
        'status': 'config',
        'backbone': backbone,
        'epoches': epoches,
        'lr': lr,
        'loss_weights': loss_weights,
    }, ensure_ascii=False))

    device = torch.device('cuda')
    supervise = list(loss_weights.keys())
    model = Tactile(backbone=backbone, supervise=supervise).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    valid_losses = []
    best_loss = 1e9
    
    for epoch in range(epoches):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        pbar.set_description(f'Epoch {epoch+1:4d}/{epoches:4d}, Training')
        for idx, d in pbar:
            x:torch.Tensor = d['marked_rgb'].to(device)
            y = {s: d[s].to(device) for s in supervise}
            outputs = model.reconstruct(x)

            loss, loss_dict = model.loss(outputs, y, weights=loss_weights)
            train_loss += loss_dict['total']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': loss_dict['total']})
            if idx % 20 == 0:
                log(json.dumps({
                    'status': 'train',
                    'epoch': epoch + 1,
                    'batch': idx + 1,
                    'loss': loss_dict['total'],
                    'losses': loss_dict
                }, ensure_ascii=False))
            
        train_losses.append(train_loss / idx)

        valid_loss = 0.0
        model.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=True)
            pbar.set_description(f'Epoch {epoch+1:4d}/{epoches:4d}, Validating')
            for idx, d in pbar:
                x:torch.Tensor = d['marked_rgb'].to(device)
                y = {s: d[s].to(device) for s in supervise}
                outputs = model.reconstruct(x)

                loss, loss_dict = model.loss(outputs, y, weights=loss_weights)
                valid_loss += loss_dict['total']
                
                pbar.set_postfix({'loss': loss_dict['total']})
                if idx % 20 == 0:
                    log(json.dumps({
                        'status': 'eval',
                        'epoch': epoch + 1,
                        'batch': idx + 1,
                        'loss': loss_dict['total'],
                        'losses': loss_dict
                    }, ensure_ascii=False))
        
        torch.save(model.state_dict(), str(save_root / f'ep{epoch}.pth'))
        valid_losses.append(valid_loss / idx)
        if valid_loss / idx < best_loss:
            best_loss = valid_loss / idx
            torch.save(model.state_dict(), str(save_root / 'best.pth'))
            print(f'  Best model saved with recon loss {best_loss:.6f}')

def main():
    global data_length
    prism_names = ['CircleShell', 'Cross', 'Cubehole', 'Cuboid', 'Cylinder', 'Doubleslope', 'Hemisphere', 'Line', 'Pacman', 'S', 'Sphere', 'Star', 'Tetrahedron', 'Torus']
    hdf5_paths = []
    for name in prism_names:
        l = list(Path(f'../data/contact-gs/{name}/hdf5').glob('*.hdf5'))
        hdf5_paths.extend(l)
    print(f'Found {len(hdf5_paths)} hdf5 files.')
    data = HDF5Dataset(hdf5_paths)
    data._data_metadata = list(np.array(data._data_metadata)[
        np.random.choice(len(data._data_metadata), data_length, replace=False)])
    print(f'Dataset length: {len(data)}')
    
    batch_size = 64
    valid_size = int(len(data) * 0.2)
    generator = torch.Generator().manual_seed(42)
    train_size = len(data) - valid_size
    train_data, valid_data = random_split(
        data, [train_size, valid_size], generator=generator)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=8,
        persistent_workers=True, worker_init_fn=worker_init_fn, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, num_workers=8,
        persistent_workers=True, worker_init_fn=worker_init_fn, pin_memory=True
    )
    train(train_loader, valid_loader, epoches=5, lr=1e-3, loss_weights=weights)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('config', type=str, default='all')
    parser.add_argument('data_num', type=int, default=1000)
    args = parser.parse_args()
    
    backbone = 'resnet18'
    if args.config == 'shape_pathway':
        ckpt_root = 'ablation/shape_pathway'
        weights = {
            'marked_rgb': 1.0,
            'rgb': 1.0,
            'depth': 0.0,
            'marker': 0.0,
            'pose': 0.0
        }
    elif args.config == 'contact_pathway':
        ckpt_root = 'ablation/contact_pathway'
        weights = {
            'marked_rgb': 0.0,
            'rgb': 0.0,
            'depth': 1.0,
            'marker': 1.0,
            'pose': 0.0
        }
    elif args.config == 'contact_shape':
        ckpt_root = 'ablation/contact_shape'
        weights = {
            'marked_rgb': 1.0,
            'rgb': 1.0,
            'depth': 0.5,
            'marker': 0.5,
            'pose': 0.0
        }
    else:
        ckpt_root = f'ablation/data_{args.data_num}'
        data_length = args.data_num
        weights = {
            'marked_rgb': 1.0,
            'rgb': 1.0,
            'depth': 0.5,
            'marker': 0.5,
            'pose': 0.5
        }
    main()
