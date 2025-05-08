#!/bin/bash

# Run training with latent_75.npy files
# This script assumes you are in the root directory of the retina-face-detector-main

# Set the dataset root path
DATASET_ROOT="./retina-modify"  # Change this if needed

# Set training parameters
MODEL="resnet50"
EPOCHS=20
BATCH_SIZE=8
LEARNING_RATE=0.001
DEVICE=""  # Set to "cuda:0" for GPU training

# Run the training
python train_retinaface.py \
    --dataset_root $DATASET_ROOT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --model $MODEL \
    --device $DEVICE \
    --no_wandb

echo "Training complete!" 