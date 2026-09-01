from policy import ACTPolicy
import json
import torch
import os
from copy import deepcopy

def load_ACT(model_path, args_file:str = None) -> ACTPolicy:
    """
    Load the ACT model from the checkpoint file.
    model_path: str, path to the model checkpoint file.
    args_file: str, path to the args.json file. If None, it will look for the args.json file in the same directory as the model_path."""
    if args_file is None:
        args_file = os.path.join(os.path.dirname(model_path), 'args.json')
    args = json.load(open(args_file, 'r'))

    if override_args is not None:
        for key, value in override_args.items():
            args[key] = value


    # load pretrained backbones
    if args['backbone'] == "clip_backbone":
        # assert statement to check if gelsight is in camera_names. Its not necessary, but useful to double check.
        # assert 'gelsight' in args['camera_names'], 'Gelsight camera not found in camera_names. Please add it to the meta_data.json file.'
        from clip_pretraining import modified_resnet18
        gelsight_model = modified_resnet18()
        vision_model = modified_resnet18()
        camera_backbone_mapping = {cam_name: 0 for cam_name in args['camera_names']}
        camera_backbone_mapping['gelsight'] = 1

        # if pretrained, load the weights
        if args['gelsight_backbone_path'] != 'none' and args['vision_backbone_path'] != 'none':
            vision_model.load_state_dict(torch.load(args['vision_backbone_path']))
            gelsight_model.load_state_dict(torch.load(args['gelsight_backbone_path']))
        elif args['gelsight_backbone_path'] != 'none' or args['vision_backbone_path'] != 'none':
            raise ValueError('Both vision and gelsight backbones must be specified if one is specified.')
        
        pretrained_backbones = [vision_model, gelsight_model]
    else:
        pretrained_backbones = None
        camera_backbone_mapping = None

    act = ACTPolicy(state_dim = args['state_dim'],
                    hidden_dim = args['hidden_dim'],
                    position_embedding_type = args['position_embedding'],
                    lr_backbone = args['lr_backbone'],
                    masks = args['masks'],
                    backbone_type = args['backbone'],
                    dilation =  args['dilation'],
                    dropout = args['dropout'],
                    nheads = args['nheads'],
                    dim_feedforward = args['dim_feedforward'],
                    num_enc_layers = args['enc_layers'],
                    num_dec_layers = args['dec_layers'],
                    pre_norm = args['pre_norm'],
                    num_queries = args['chunk_size'],
                    camera_names = args['camera_names'],
                    z_dimension = args['z_dimension'],
                    lr = args['lr'],
                    pretrained_backbones = pretrained_backbones,
                    cam_backbone_mapping = camera_backbone_mapping)
    
    # verify that all of the weight parameters are loaded (they should change if the model is loaded)
    pre_load_params = deepcopy(list(act.parameters()))
    act.load_state_dict(torch.load(model_path))
    post_load_params = list(act.parameters())
    for i in range(len(pre_load_params)):
        if torch.equal(pre_load_params[i], post_load_params[i]):
            print(f"Parameter {i} was not loaded correctly.")
        # assert not torch.equal(pre_load_params[i], post_load_params[i]), f"The model was not loaded correctly. {i} parameter is the same before and after loading."
    return act

if __name__ == '__main__':
    model_path = "/home/aigeorge/research/TactileACT/data/camera_cage/pretrained_1999_melted/policy_last.ckpt"
    args_file = "/home/aigeorge/research/TactileACT/data/camera_cage/pretrained_1999_melted_0/args.json"
    act = load_ACT(model_path)
    print(act)
    print('Done')