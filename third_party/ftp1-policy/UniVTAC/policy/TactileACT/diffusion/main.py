from utils import get_norm_stats

from datetime import datetime

import shutil
import os

# This version trains the model with clip or resnet on any dataset
from train_args import START_TIME, CKPT_DIR, DATA_DIR, ENC_TYPE, CODE_DIR,\
    PRED_HORIZON, GEL_ONLY


######################################
## BEFORE RUNNING EDIT AND CONFIRM  ##
## TRAIN ARGS IN train_args.py      ##
######################################

now_time = START_TIME.strftime("%H-%M-%S_%Y-%m-%d")

thisfile = os.path.abspath(__file__)

thisfile_name = os.path.basename(__file__)




#this dir made by train_args.py
#shutil.copytree('/home/selam/diffusion_plugging', code_dir+'/diffusion_plugging')
shutil.copyfile(thisfile, CODE_DIR+'/'+thisfile_name)

num_episodes = 100

cams = [1,2,3,4,5,6,'gelsight']

if GEL_ONLY:
    cams = ['gelsight']


norm_stats = get_norm_stats(DATA_DIR, num_episodes)

#CHOOSE THE CORRECT TRAIN FILE!!! 
print(norm_stats)

from train_policy import train, create_nets

nets, train_dataloader, val_dataloader, enc_type = create_nets(
    ENC_TYPE,DATA_DIR,norm_stats,cams,num_episodes=num_episodes,pred_horizon=PRED_HORIZON)

train(3500, cams, nets,train_dataloader,val_dataloader, enc_type)

