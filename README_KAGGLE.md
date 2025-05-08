# Training RetinaFace with Latent Representations on Kaggle

This README provides instructions for running the modified RetinaFace model with latent representations on Kaggle, specifically with the provided dataset structure.

## Dataset Structure

The dataset is structured as follows:
```
retina-modify/
├── train/
│   ├── latent/
│   │   ├── 0_Parade_Parade_0_1014/
│   │   │   ├── latent_100.npy
│   │   │   ├── latent_12.npy
│   │   │   ├── latent_25.npy
│   │   │   ├── latent_50.npy
│   │   │   └── latent_75.npy
│   │   ├── <more directories>/
│   │   └── labels.txt
│   └── labels.txt
├── val/
│   ├── latent/
│   │   └── <similar structure as train/latent/>
│   └── labels.txt
└── train_retinaface.py
```

Each directory under `latent/` represents an image, and contains different latent representations. For this project, we will use the `latent_75.npy` files.

## Setup Instructions

### Step 1: Move the code into place

Ensure that all the modified code files are in the appropriate places:
- `model/common.py` - Contains the BridgeModule and other model components
- `model/model.py` - Contains the modified RetinaFace model that uses latent input
- `utils/dataset.py` - Contains the NestedLatentWiderFaceDataset class
- `train.py` - Modified training script
- `train_retinaface.py` - Wrapper script for easy training

### Step 2: Install Dependencies

```bash
pip install torch torchvision numpy opencv-python matplotlib wandb
```

### Step 3: Run the Training Script

For training with the default settings:

```bash
python train_retinaface.py --dataset_root ./retina-modify --no_wandb
```

### Advanced Usage

You can customize the training with various parameters:

```bash
python train_retinaface.py \
    --dataset_root ./retina-modify \
    --epochs 30 \
    --batch_size 16 \
    --lr 0.001 \
    --model resnet50 \
    --device cuda:0 \
    --no_wandb \
    --freeze
```

### Parameters

- `--dataset_root`: Root directory of the dataset (default: '.')
- `--epochs`: Number of epochs to train (default: 20)
- `--batch_size`: Batch size for training (default: 8)
- `--lr`: Initial learning rate (default: 0.001)
- `--model`: Backbone model architecture (options: resnet18, resnet34, resnet50; default: resnet50)
- `--device`: Device to train on (e.g., 'cuda:0', default: '')
- `--no_wandb`: Disable Weights & Biases logging
- `--freeze`: Freeze backbone weights
- `--weight`: Path to pretrained weights to start from

## Model Architecture

The modified RetinaFace model uses a bridge module to convert latent representations [256, 40, 40] to match the expected input size for the ResNet-50 model (after stage 1). The key components are:

1. **Bridge Module**:
   - Converts latent [256, 40, 40] to [256, 160, 160]
   - Uses bilinear upsampling and convolution layers

2. **Pruned ResNet-50**:
   - Skips stage 0 and stage 1
   - Starts from stage 2 (layer2)

3. **FPN and Head Modules**:
   - Same as the original RetinaFace architecture

## Working with Specific Latent Files

This implementation is set to use `latent_75.npy` files by default. If you want to use a different latent file type:

```bash
python train.py --use_latent \
    --latent_dir ./retina-modify/train/latent \
    --val_latent_dir ./retina-modify/val/latent \
    --train_label_file ./retina-modify/train/labels.txt \
    --val_label_file ./retina-modify/val/labels.txt \
    --latent_type latent_100.npy \
    --model resnet50
```

## Saving and Loading Models

Models are automatically saved after each epoch to the created experiment directory. The final model is saved as `weight_final.pth`.

To use a saved model for inference:

```bash
python detection.py \
    --latent /path/to/latent.npy \
    --weight /path/to/weight_final.pth \
    --network resnet50 \
    --conf 0.6
``` 