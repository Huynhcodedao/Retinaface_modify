import os
import torch
import wandb
import numpy as np
from utils.data_augment import WiderFacePreprocess
from model.config import INPUT_SIZE, TRAIN_PATH, VALID_PATH
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class NestedLatentWiderFaceDataset(Dataset):
    """
    Dataset for latent representations of Wider Face with nested directory structure.
    
    Args:
        root_path (string): Path to dataset directory (e.g., 'train/latent')
        label_file (string): Path to the label file (e.g., 'train/labels.txt')
        latent_type (string): Type of latent file to use (e.g., 'latent_75.npy')
        is_train (bool): Train dataset or test dataset
    """
    def __init__(self, root_path, label_file, latent_type='latent_75.npy', is_train=True):
        self.root_path = root_path
        self.label_file = label_file
        self.latent_type = latent_type
        self.is_train = is_train
        
        # Get all subdirectories (image folders)
        self.image_dirs = [d for d in os.listdir(root_path) 
                         if os.path.isdir(os.path.join(root_path, d))]
        
        # Parse labels
        self.label_dict = self._parse_label_file(label_file)
        
        # Filter to only include directories that have the specified latent type
        self.valid_dirs = []
        for img_dir in self.image_dirs:
            latent_path = os.path.join(root_path, img_dir, latent_type)
            if os.path.exists(latent_path):
                self.valid_dirs.append(img_dir)
        
        print(f"Found {len(self.valid_dirs)} valid directories with {latent_type}")
    
    def _parse_label_file(self, label_file):
        """Parse the label file to match image directories with bounding boxes"""
        label_dict = {}
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
            i = 0
            while i < len(lines):
                # Each image annotation starts with the image name
                img_name = lines[i].strip()
                i += 1
                
                # Get base name (without extension) to match directory name
                base_name = os.path.splitext(os.path.basename(img_name))[0]
                
                # Next line has the number of faces
                if i >= len(lines):
                    break
                    
                try:
                    face_count = int(lines[i].strip())
                    i += 1
                except ValueError:
                    # If line is not a number, skip this entry
                    continue
                
                # Skip images with no faces
                if face_count == 0:
                    # Skip the blank line that follows zero faces
                    if i < len(lines):
                        i += 1
                    continue
                
                # Parse all face bounding boxes for this image
                annotations = []
                for j in range(face_count):
                    if i >= len(lines):
                        break
                        
                    try:
                        values = lines[i].strip().split()
                        i += 1
                        
                        # Check if we have sufficient values
                        if len(values) < 4:
                            continue
                            
                        # Get bounding box coordinates (x, y, w, h)
                        x = float(values[0])
                        y = float(values[1])
                        w = float(values[2])
                        h = float(values[3])
                        
                        # Convert to (x1, y1, x2, y2) format
                        box = [x, y, x + w, y + h]
                        
                        # Add landmark if available (assuming 5 landmarks with (x,y) coordinates)
                        landmarks = []
                        landmarks_available = True
                        
                        if len(values) >= 14:  # 4 bbox + 10 landmark coordinates
                            for k in range(4, 14, 2):
                                lx = float(values[k])
                                ly = float(values[k+1])
                                if lx < 0 or ly < 0:
                                    landmarks_available = False
                                    break
                                landmarks.extend([lx, ly])
                        else:
                            landmarks_available = False
                            
                        if not landmarks_available:
                            landmarks = [-1.0] * 10  # Fill with -1 if landmarks not available
                            
                        # Combine box and landmarks
                        annotation = box + landmarks + [1.0 if landmarks_available else -1.0]
                        annotations.append(annotation)
                        
                    except (ValueError, IndexError) as e:
                        # Skip malformed lines
                        continue
                
                # Store annotations for this image
                if annotations:
                    label_dict[base_name] = np.array(annotations)
        
        return label_dict
    
    def __len__(self):
        return len(self.valid_dirs)
    
    def __getitem__(self, index):
        # Get the image directory
        img_dir = self.valid_dirs[index]
        
        # Load the latent representation
        latent_path = os.path.join(self.root_path, img_dir, self.latent_type)
        latent = np.load(latent_path)
        
        # Ensure latent has proper dimensions [256, 40, 40]
        if len(latent.shape) == 4 and latent.shape[0] == 1:  # If shape is [1, 256, 40, 40]
            latent = latent.squeeze(0)  # Convert to [256, 40, 40]
            
        # Get annotations for this image
        if img_dir in self.label_dict:
            annotations = self.label_dict[img_dir]
        else:
            # If no annotations found, return empty annotation tensor
            annotations = np.zeros((0, 15))
            
        # Convert latent to tensor
        latent_tensor = torch.from_numpy(latent).float()
        
        return latent_tensor, annotations

class LatentWiderFaceDataset(Dataset):
    """
    Dataset for latent representations of Wider Face.
    
    Args:
        root_path (string): Path to dataset directory containing .npy latent files
        label_file (string): Path to the WIDER FACE bounding box labels file
        is_train (bool): Train dataset or test dataset
    """
    def __init__(self, root_path, label_file, is_train=True):
        self.latent_path = root_path
        self.label_file = label_file
        self.is_train = is_train
        
        # Load all latent file paths
        self.latent_files = [f for f in os.listdir(root_path) if f.endswith('.npy')]
        self.label_dict = self._parse_label_file(label_file)
        
    def _parse_label_file(self, label_file):
        """Parse the WIDER FACE label file to match latent files with bounding boxes"""
        label_dict = {}
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                # Get image name
                img_name = lines[i].strip()
                i += 1
                
                # Get number of faces
                face_count = int(lines[i].strip())
                i += 1
                
                # Skip images with no faces
                if face_count == 0:
                    i += 1
                    continue
                    
                # Parse all face bounding boxes for this image
                annotations = []
                for j in range(face_count):
                    if i >= len(lines):
                        break
                        
                    values = lines[i].strip().split()
                    i += 1
                    
                    # Check if we have sufficient values
                    if len(values) < 4:
                        continue
                        
                    # Get bounding box coordinates (x, y, w, h)
                    try:
                        x = float(values[0])
                        y = float(values[1])
                        w = float(values[2])
                        h = float(values[3])
                        
                        # Convert to (x1, y1, x2, y2) format
                        box = [x, y, x + w, y + h]
                        
                        # Add landmark if available (assuming 5 landmarks with (x,y) coordinates)
                        landmarks = []
                        landmarks_available = True
                        
                        if len(values) >= 14:  # 4 bbox + 10 landmark coordinates
                            for k in range(4, 14, 2):
                                lx = float(values[k])
                                ly = float(values[k+1])
                                if lx < 0 or ly < 0:
                                    landmarks_available = False
                                    break
                                landmarks.extend([lx, ly])
                        else:
                            landmarks_available = False
                            
                        if not landmarks_available:
                            landmarks = [-1.0] * 10  # Fill with -1 if landmarks not available
                            
                        # Combine box and landmarks
                        annotation = box + landmarks + [1.0 if landmarks_available else -1.0]
                        annotations.append(annotation)
                        
                    except (ValueError, IndexError):
                        # Skip malformed lines
                        continue
                
                # Store annotations for this image
                if annotations:
                    # Extract base filename without extension
                    base_name = os.path.splitext(os.path.basename(img_name))[0]
                    label_dict[base_name] = np.array(annotations)
                    
        return label_dict
    
    def __len__(self):
        return len(self.latent_files)
    
    def __getitem__(self, index):
        # Get the latent file name
        latent_file = self.latent_files[index]
        base_name = os.path.splitext(latent_file)[0]
        
        # Load the latent representation [256, 40, 40]
        latent = np.load(os.path.join(self.latent_path, latent_file))
        
        # Ensure latent has proper dimensions [256, 40, 40]
        if len(latent.shape) == 4 and latent.shape[0] == 1:  # If shape is [1, 256, 40, 40]
            latent = latent.squeeze(0)  # Convert to [256, 40, 40]
            
        # Get annotations for this image
        if base_name in self.label_dict:
            annotations = self.label_dict[base_name]
        else:
            # If no annotations found, return empty annotation tensor
            annotations = np.zeros((0, 15))
            
        # Convert latent to tensor
        latent_tensor = torch.from_numpy(latent).float()
        
        return latent_tensor, annotations

class WiderFaceDataset(Dataset):
    """
    Wider Face custom dataset.
    Args:
        root_path (string): Path to dataset directory
        is_train (bool): Train dataset or test dataset
        transform (function): whether to apply the data augmentation scheme
                mentioned in the paper. Only applied on the train split.
    """

    def __init__(self, root_path, input_size=INPUT_SIZE, is_train=True):
        self.ids       = []
        self.transform = WiderFacePreprocess(image_size=input_size)
        self.is_train  = is_train

        if is_train: 
            self.path = os.path.join(root_path, TRAIN_PATH)
        else: 
            self.path = os.path.join(root_path, VALID_PATH)
        
        for dirname in os.listdir(os.path.join(self.path, 'images')):
            for file in os.listdir(os.path.join(self.path, 'images', dirname)):
                self.ids.append(os.path.join(dirname, file)[:-4])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.path, 'images', self.ids[index]+'.jpg'))
        img = np.array(img)

        f = open(os.path.join(self.path, 'labels', self.ids[index]+'.txt'), 'r')
        lines = f.readlines()

        annotations = np.zeros((len(lines), 15)) 

        if len(lines) == 0:
            return annotations
        
        for idx, line in enumerate(lines):
            line = line.strip().split()
            line = [float(x) for x in line]

            # bbox
            annotations[idx, 0] = line[0]               # x1
            annotations[idx, 1] = line[1]               # y1
            annotations[idx, 2] = line[0] + line[2]     # x2
            annotations[idx, 3] = line[1] + line[3]     # y2

            if self.is_train:
                # landmarks
                annotations[idx, 4] = line[4]               # l0_x
                annotations[idx, 5] = line[5]               # l0_y
                annotations[idx, 6] = line[7]               # l1_x
                annotations[idx, 7] = line[8]               # l1_y
                annotations[idx, 8] = line[10]              # l2_x
                annotations[idx, 9] = line[11]              # l2_y
                annotations[idx, 10] = line[13]             # l3_x
                annotations[idx, 11] = line[14]             # l3_y
                annotations[idx, 12] = line[16]             # l4_x
                annotations[idx, 13] = line[17]             # l4_y

                if (annotations[idx, 4]<0):
                    annotations[idx, 14] = -1
                else:
                    annotations[idx, 14] = 1
            
            else:
                annotations[idx, 14] = 1

        if self.transform is not None:
            img, annotations = self.transform(image=img, targets=annotations)

        return img, annotations

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
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []

    for _, (image, target) in enumerate(batch):
        image  = torch.from_numpy(image)
        target = torch.from_numpy(target).to(dtype=torch.float)

        imgs.append(image)
        targets.append(target)

    return (torch.stack(imgs, dim=0), targets)