#!/usr/bin/env python
# coding: utf-8

"""
Script for training RetinaFace with latent representations on Kaggle
"""

import os
import glob
import argparse
import subprocess
import sys

def check_dataset_structure(dataset_root):
    """Check if the dataset has the expected structure"""
    train_latent_dir = os.path.join(dataset_root, "train/latent")
    train_label_file = os.path.join(dataset_root, "train/labels.txt")
    
    print(f"Train latent directory exists: {os.path.exists(train_latent_dir)}")
    print(f"Train label file exists: {os.path.exists(train_label_file)}")
    
    # Count the number of image directories
    if os.path.exists(train_latent_dir):
        image_dirs = [d for d in os.listdir(train_latent_dir) 
                    if os.path.isdir(os.path.join(train_latent_dir, d))]
        print(f"Found {len(image_dirs)} image directories")
        
        # Check a sample directory for latent_75.npy files
        if image_dirs:
            sample_dir = os.path.join(train_latent_dir, image_dirs[0])
            print(f"Sample directory: {sample_dir}")
            print(f"Files in sample directory: {os.listdir(sample_dir)}")
            print(f"latent_75.npy exists: {os.path.exists(os.path.join(sample_dir, 'latent_75.npy'))}")
    
    return os.path.exists(train_latent_dir) and os.path.exists(train_label_file)

def parse_args():
    parser = argparse.ArgumentParser(description='Train RetinaFace with latent representations on Kaggle')
    parser.add_argument('--dataset_root', type=str, default='./retina-modify', help='Root directory of the dataset')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model', type=str, default='resnet50', help='Model architecture')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--check_only', action='store_true', help='Only check dataset structure without training')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=== Checking dataset structure ===")
    if not check_dataset_structure(args.dataset_root):
        print("Error: Dataset does not have the expected structure!")
        return
        
    if args.check_only:
        print("Dataset check completed. Exiting as requested.")
        return
    
    print("\n=== Starting training ===")
    train_cmd = [
        sys.executable, "train.py",
        "--use_latent",
        "--latent_dir", os.path.join(args.dataset_root, "train/latent"),
        "--val_latent_dir", os.path.join(args.dataset_root, "val/latent"),
        "--train_label_file", os.path.join(args.dataset_root, "train/labels.txt"),
        "--val_label_file", os.path.join(args.dataset_root, "val/labels.txt"),
        "--latent_type", "latent_75.npy",
        "--model", args.model,
        "--epoch", str(args.epochs),
        "--batchsize", str(args.batch_size),
        "--lr", str(args.lr),
        "--device", args.device,
        "--no_wandb"
    ]
    
    print(f"Running command: {' '.join(train_cmd)}")
    subprocess.run(train_cmd, check=True)
    
    print("\n=== Training complete ===")
    
    # Check for saved models
    exp_dirs = glob.glob('exp-*')
    if exp_dirs:
        latest_exp = max(exp_dirs, key=os.path.getmtime)
        print(f"Latest experiment directory: {latest_exp}")
        print(f"Files in experiment directory: {os.listdir(latest_exp)}")
    else:
        print("No experiment directories found")

if __name__ == "__main__":
    main() 