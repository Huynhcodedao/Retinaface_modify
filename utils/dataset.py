import os
import torch
import wandb
import numpy as np
from utils.data_augment import WiderFacePreprocess
from model.config import INPUT_SIZE, TRAIN_PATH, VALID_PATH
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class WiderFaceDataset(Dataset):
    """
    Dataset cho cấu trúc latent nhiều thư mục con, đọc label từ file tổng hợp.
    """
    def __init__(self, root_path, input_size=INPUT_SIZE, is_train=True, label_file='labels.txt'):
        self.latent_paths = []
        self.latent_keys = []
        self.is_train = is_train

        if is_train: 
            self.path = os.path.join(root_path, TRAIN_PATH)
        else: 
            self.path = os.path.join(root_path, VALID_PATH)

        latent_root = os.path.join(self.path, 'latent')
        self.label_dict = {}

        # Đọc file label tổng hợp
        label_path = os.path.join(self.path, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            img_name = lines[i].strip()
            num_bbox = int(lines[i+1].strip())
            bboxes = []
            for j in range(num_bbox):
                bboxes.append(lines[i+2+j].strip())
            # Lấy key là tên file không đuôi, không thư mục
            base = os.path.splitext(os.path.basename(img_name))[0]
            self.label_dict[base] = bboxes
            i = i + 2 + num_bbox

        # Duyệt toàn bộ file latent_75.npy
        for folder in os.listdir(latent_root):
            folder_path = os.path.join(latent_root, folder)
            if not os.path.isdir(folder_path):
                continue
            latent_file = os.path.join(folder_path, 'latent_75.npy')
            if os.path.isfile(latent_file):
                self.latent_paths.append(latent_file)
                self.latent_keys.append(folder)  # folder name là key

    def __len__(self):
        return len(self.latent_paths)

    def __getitem__(self, index):
        latent = np.load(self.latent_paths[index])  # shape [1, 256, 40, 40]
        key = self.latent_keys[index]
        bboxes = self.label_dict.get(key, [])
        annotations = np.zeros((len(bboxes), 15)) 
        for idx, line in enumerate(bboxes):
            line = [float(x) for x in line.strip().split()]
            # bbox
            annotations[idx, 0] = line[0]               # x1
            annotations[idx, 1] = line[1]               # y1
            annotations[idx, 2] = line[0] + line[2]     # x2
            annotations[idx, 3] = line[1] + line[3]     # y2
            # Nếu có landmark, bạn có thể bổ sung ở đây (nếu không thì giữ nguyên)
            # Nếu không có landmark, các cột còn lại giữ 0
            annotations[idx, 14] = 1  # label hợp lệ
        return latent, annotations

def log_dataset(use_artifact, 
        artifact_name, 
        artifact_path, dataset_name, 
        job_type='preprocess dataset', 
        project_name='Content-based RS'):

    run = wandb.init(project=project_name, job_type=job_type)
    run.use_artifact(use_artifact)
    artifact = wandb.Artifact(artifact_name, dataset_name)

    if os.path.isdir(artifact_path):
        artifact.add_dir(artifact_path)
    else:
        artifact.add_file(artifact_path)
    run.log_artifact(artifact)


def detection_collate(batch):
    """Custom collate fn cho latent representation."""
    targets = []
    latents = []

    for _, (latent, target) in enumerate(batch):
        latent = torch.from_numpy(latent).float()  # [1, 256, 40, 40]
        latent = latent.squeeze(0)  # [256, 40, 40]
        target = torch.from_numpy(target).to(dtype=torch.float)

        latents.append(latent)
        targets.append(target)

    return (torch.stack(latents, dim=0), targets)