#!/usr/bin/env python
# coding: utf-8

"""
Training script for RetinaFace model with latent inputs
"""

import os
import sys
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description='Train RetinaFace with latent representations')
    parser.add_argument('--dataset_root', type=str, default='.', help='Root directory of the dataset')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--model', type=str, default='resnet50', help='Backbone model (resnet50, resnet34, resnet18)')
    parser.add_argument('--device', type=str, default='', help='Device to train on (e.g. cuda:0)')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--freeze', action='store_true', help='Freeze backbone weights')
    parser.add_argument('--weight', type=str, default=None, help='Path to pretrained weights')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup paths
    train_latent_dir = os.path.join(args.dataset_root, 'train/latent')
    val_latent_dir = os.path.join(args.dataset_root, 'val/latent')
    train_label_file = os.path.join(args.dataset_root, 'train/labels.txt')
    val_label_file = os.path.join(args.dataset_root, 'val/labels.txt')
    
    # Check if paths exist
    if not os.path.exists(train_latent_dir):
        print(f"Error: Training latent directory not found at {train_latent_dir}")
        return
    
    if not os.path.exists(train_label_file):
        print(f"Error: Training label file not found at {train_label_file}")
        return
    
    # Build command
    cmd = [
        'python', 'train.py',
        '--use_latent',
        '--latent_dir', train_latent_dir,
        '--val_latent_dir', val_latent_dir,
        '--train_label_file', train_label_file,
        '--val_label_file', val_label_file,
        '--latent_type', 'latent_75.npy',
        '--model', args.model,
        '--epoch', str(args.epochs),
        '--batchsize', str(args.batch_size),
        '--lr', str(args.lr),
        '--device', args.device
    ]
    
    # Add optional arguments
    if args.no_wandb:
        cmd.append('--no_wandb')
    
    if args.freeze:
        cmd.append('--freeze')
    
    if args.weight:
        cmd.extend(['--weight', args.weight])
    
    # Print the command
    print("Running command:")
    print(' '.join(cmd))
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running training command: {e}")
    except KeyboardInterrupt:
        print("Training interrupted by user")

if __name__ == "__main__":
    main() 