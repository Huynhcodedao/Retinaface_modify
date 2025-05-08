# RetinaFace with Latent Representation Input

This repository contains a modified version of the RetinaFace face detector that accepts latent representations as input instead of RGB images. The model has been modified to skip the first two stages of ResNet50 and use a bridge module to convert latent representations to the appropriate size.

## Requirements

- Python 3.6+
- PyTorch 1.0+
- NumPy
- OpenCV
- Matplotlib
- WandB (for logging)

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Input Preparation

The model accepts latent representations with shape `[256, 40, 40]` stored as `.npy` files. These should be placed in a specified directory (default: `./latent/`).

### Labels Format

The bounding box labels should be in the WIDER FACE format, with each face annotation defined by:
- `(x, y, w, h)` - Bounding box coordinates on the original 640x640 image
- Optionally, facial landmarks can be included

## Training

To train the model with latent representations, use the following command:

```bash
python train.py --use_latent \
    --latent_dir /path/to/latent/files \
    --label_file /path/to/wider_face_train_bbx_gt.txt \
    --model resnet50 \
    --epoch 20 \
    --lr 0.001 \
    --batchsize 8
```

### Training Arguments

- `--use_latent`: Flag to use latent representations as input
- `--latent_dir`: Directory containing latent .npy files
- `--label_file`: Path to the WIDER FACE bounding box labels file
- `--model`: Backbone model name (default: 'resnet18', options: 'resnet50', 'resnet34', etc.)
- `--epoch`: Number of training epochs
- `--lr`: Learning rate
- `--batchsize`: Batch size
- `--device`: Device selection ('0', '1', 'cpu', etc.)
- `--freeze`: Flag to freeze backbone weights

## Detection

To run detection on a single latent file:

```bash
python detection.py \
    --latent /path/to/latent.npy \
    --weight /path/to/model.pth \
    --network resnet50 \
    --conf 0.6 \
    --output results.jpg
```

### Detection Arguments

- `--latent`: Path to a latent .npy file
- `--weight`: Path to the trained model weights
- `--network`: Network architecture (should match the training model)
- `--conf`: Confidence threshold for face detection
- `--nms`: NMS threshold
- `--topk`: Top-k before NMS
- `--keep`: Maximum number of detections to keep

## Model Architecture

The modified RetinaFace model has the following architecture:

1. **Bridge Module**:
   - Converts latent [256, 40, 40] to [256, 160, 160]
   - Uses upsampling and convolution layers

2. **Pruned ResNet-50**:
   - Skips stage 0 and stage 1
   - Starts from stage 2 (layer2)
   - Produces feature maps:
     - C2: [512, 80, 80] (stride 8)
     - C3: [1024, 40, 40] (stride 16)
     - C4: [2048, 20, 20] (stride 32)

3. **FPN**:
   - Creates pyramid features:
     - P2: [256, 160, 160] (stride 4)
     - P3: [256, 80, 80] (stride 8)
     - P4: [256, 40, 40] (stride 16)
     - P5: [256, 20, 20] (stride 32)
     - P6: [256, 10, 10] (stride 64)

4. **Head Modules**:
   - Classification, Bounding Box Regression, and Landmark Regression heads
   - Same as the original RetinaFace architecture

## References

- [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641) 