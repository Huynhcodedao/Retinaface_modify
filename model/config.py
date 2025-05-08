import os

PROJECT             = 'Retina-Face'
RUN_NAME            = 'retina-face-detector'
# DATA config ================
DATA_PATH           = '.'
N_CLASSES           = 2
TRAIN_PATH          = 'train'
VALID_PATH          = 'eval'
TEST_PATH           = './test'
SAVE_PATH           = './model'

DATASET             = 'WIDER'
DVERSION            = 'latest'
INPUT_SIZE          = 640
BATCH_SIZE          = 8
RANDOM_SEED         = 42
NUM_WORKERS         = 2

# MODEL config ==============
EPOCHS              = 10
START_FRAME         = 32
LEARNING_RATE       = 0.01
LR_MILESTONE        = [1, 3, 5]
WEIGHT_DECAY        = 5e-4
MOMENTUM            = 0.9

IN_CHANNELS         = 32
OUT_CHANNELS        = 256

# MobileNetV1
FEATURE_MAP_MOBN1   = {'4':2, '8':3, '15':4, '15':5, '15':6}
RETURN_MAP_MOBN1    = {'stage1': 4, 'stage2': 8, 'stage3': 15}

# MobileNetV2
FEATURE_MAP_MOBN2   = {'features.3':2, 'features.6':3, 'features.13':4, 'features.13':5, 'features.13':6}

# Resnet50
FEATURE_MAP         = {'layer1':2, 'layer2':3, 'layer3':4, 'layer4':5, 'layer4':6}
RETURN_MAP          = {'layer1':1, 'layer2':2, 'layer3':3, 'layer4':4}

# CRITERION config ==========
OVERLAP_THRES       = 0.35
BKG_LABEL           = 0
NEG_MINING          = True
NEG_OVERLAP         = 0.35

MIN_SIZES           = [[16, 32], [64, 128], [256, 512]]
RGB_MEANS           = (104, 117, 123)
VARIANCE            = [0.1, 0.2]

USE_PRUNED_BACKBONE = True
