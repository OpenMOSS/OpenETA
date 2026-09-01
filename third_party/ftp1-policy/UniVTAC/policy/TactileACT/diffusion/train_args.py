from datetime import datetime
import shutil
import os

START_TIME = datetime.now()

#CLUSTER:

# # FIXED 
# DATA_TYPE = "FIXED"
# CKPT_DIR = '/home/selam/checkpoints/'
# #for pretrained clip head
# GELSIGHT_WEIGHTS_PATH = '/home/selam/data/epoch_1399_gelsight_encoder.pth'
# IMAGE_WEIGHTS_PATH = '/home/selam/data/epoch_1399_vision_encoder.pth'
# DATA_DIR = "/home/selam/data/camera_cage_new_mount/"
# CODE_START_DIR = '/home/selam/diffusion_plugging/'
# ENC_TYPE = 'resnet18'  # weights above not actually used here
# DEVICE_STR = 'cuda:2'
# PRED_HORIZON = 20
# ABLATE_GEL = False 
# GEL_ONLY = False

# # NOT FIXED DATASET
DATA_TYPE = "NOT_FIXED"
# CKPT_DIR = '/home/selam/checkpoints/'
CKPT_DIR = 'policy/TactileACT/dp-ckpt'
#for pretrained clip head
# GELSIGHT_WEIGHTS_PATH = '/home/selam/data/camera_cage_new_notfixed/clip_models/trained/epoch_1499_gelsight_encoder.pth'
# IMAGE_WEIGHTS_PATH = '/home/selam/data/camera_cage_new_notfixed/clip_models/trained/epoch_1499_vision_encoder.pth'
# DATA_DIR = "/home/selam/data/camera_cage_new_notfixed/data"
GELSIGHT_WEIGHTS_PATH = 'policy/TactileACT/clip_models/insert_hole-clean-50/0/epoch_1499_gelsight_encoder.pth'
IMAGE_WEIGHTS_PATH = 'policy/TactileACT/clip_models/insert_hole-clean-50/0/epoch_1499_vision_encoder.pth'
DATA_DIR = 'policy/TactileACT/data'
# CODE_START_DIR = '/home/selam/diffusion_plugging/'
ENC_TYPE = 'clip'  # weights above not actually used here
DEVICE_STR = 'cuda:4'
PRED_HORIZON = 20
ABLATE_GEL = False 
GEL_ONLY = False


# PC: 
# DATA_TYPE = "FIXED"
# CKPT_DIR = '/media/selamg/DATA/diffusion_plugging_checkpoints/'
# #for pretrained clip head
# GELSIGHT_WEIGHTS_PATH = '/home/selamg/diffusion_plugging/Processed_Data_2/epoch_1399_gelsight_encoder.pth'
# IMAGE_WEIGHTS_PATH = '/home/selamg/diffusion_plugging/Processed_Data_2/epoch_1399_vision_encoder.pth'
# DATA_DIR = "/home/selamg/diffusion_plugging/Processed_Data_2/data"
# #doing this just bc datadir is inside of diffusion_plugging right now...
# CODE_START_DIR = "/home/selamg/diffusion_plugging/analysis"
# ENC_TYPE = 'resnet18'
# DEVICE_STR = 'cuda:0'
# PRED_HORIZON = 20
# ABLATE_GEL = False
# GEL_ONLY = False


######### shutil moves code ###############

assert  not(ABLATE_GEL == True and GEL_ONLY == True)

print(f"{START_TIME}__STARTING TASK: {DATA_TYPE} WITH {ENC_TYPE} CUDA {DEVICE_STR} HORIZON {PRED_HORIZON}")

if ABLATE_GEL:
    print("ABLATING GELSIGHT")

if GEL_ONLY:
    print("GEL ONLY ABLATING IMAGES")

now_time = START_TIME.strftime("%H-%M-%S_%Y-%m-%d")

CODE_DIR = CKPT_DIR+'/code_'+now_time+'_'+ ENC_TYPE + DATA_TYPE


thisfile = os.path.abspath(__file__)
thisfile_name = os.path.basename(__file__)


os.makedirs(CODE_DIR,exist_ok=True)


shutil.copytree(CODE_START_DIR, CODE_DIR+'/diffusion_plugging')
shutil.copyfile(thisfile, CODE_DIR+'/'+thisfile_name)
