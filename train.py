import os 
import time
from utils.generate import select_device
import wandb
import torch
import argparse
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from model.config import *
from model.anchor import Anchors
from utils.mlops_tool import use_data_wandb
from model.model import RetinaFace, forward
from utils.data_tool import create_exp_dir
from model.multibox_loss import MultiBoxLoss
from model.metric import calculate_map, calculate_running_map
from utils.dataset import WiderFaceDataset, detection_collate

def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train image segmentation')
    parser.add_argument('--run', type=str, default=RUN_NAME, help="run name")
    parser.add_argument('--epoch', type=int, default=EPOCHS, help="number of epoch")
    parser.add_argument('--image_size', type=int, default=INPUT_SIZE, help='input size')
    parser.add_argument('--model', type=str, default='resnet18', help='select model')
    parser.add_argument('--freeze', action='store_true', help="freeze model backbone")
    parser.add_argument('--weight', type=str, default=None, help='path to pretrained weight')
    parser.add_argument('--weight_decay', type=int, default=WEIGHT_DECAY, help="weight decay of optimizer")
    parser.add_argument('--momentum', type=int, default=MOMENTUM, help="momemtum of optimizer")
    parser.add_argument('--startfm', type=int, default=START_FRAME, help="architecture start frame")
    parser.add_argument('--batchsize', type=int, default=BATCH_SIZE, help="total batch size for all GPUs (default:")
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help="init learning rate (default: 0.0001)")
    parser.add_argument('--download', action='store_true', help="download dataset from Wandb Database")
    parser.add_argument('--tuning', action='store_true', help="no plot image for tuning")
    parser.add_argument('--device', type=str, default='', help="no plot image for tuning")

    args = parser.parse_args()
    return args

def train(model, anchors, trainloader, optimizer, loss_function, device='cpu'):
    model.train()
    loss_cls, loss_box, loss_pts = 0, 0, 0
    
    for i, (input, targets) in enumerate(trainloader):
        # Debug: Kiểm tra dữ liệu đầu vào
        if i == 0:  # Chỉ in lần đầu để tránh quá nhiều log
            print(f"[DEBUG] Input shape: {input.shape}")
            for j, target in enumerate(targets[:2]):  # Chỉ in 2 targets đầu tiên
                print(f"[DEBUG] Target {j} shape: {target.shape}")
                if target.shape[0] > 0:  # Nếu có bbox
                    print(f"[DEBUG] Target {j} bbox example: {target[0][:4]}")
                    print(f"[DEBUG] Target {j} label: {target[0][-1]}")
                else:
                    print(f"[DEBUG] Target {j} is empty!")
        
        # load data into cuda
        input   = input.to(device)
        targets = [annos.to(device) for annos in targets]
        
        # Debug: Kiểm tra targets sau khi chuyển sang device
        if i == 0:
            for j, target in enumerate(targets[:2]):
                if target.shape[0] > 0:
                    print(f"[DEBUG] Target {j} on device shape: {target.shape}")
                    print(f"[DEBUG] Target {j} bbox sum: {target[:, :4].sum()}")
                else:
                    print(f"[DEBUG] Target {j} on device is empty!")
        
        predict = model(input)
        print(f"[DEBUG] anchors shape: {anchors.shape}")
        print(f"[DEBUG] bbox regression output shape: {predict[0].shape}")
        
        # Debug: Kiểm tra đầu ra của model
        if i == 0:
            for j, pred in enumerate(predict):
                print(f"[DEBUG] Prediction {j} shape: {pred.shape}")
                print(f"[DEBUG] Prediction {j} min: {pred.min().item()}, max: {pred.max().item()}, mean: {pred.mean().item()}")
                # Kiểm tra NaN hoặc Inf
                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    print(f"[DEBUG] WARNING: Prediction {j} contains NaN or Inf!")
        
        # forward + backpropagation + step
        loss_l, loss_c, loss_landm = forward(model, input, targets, anchors, loss_function, optimizer)
        
        # Debug: Kiểm tra loss
        if i == 0:
            print(f"[DEBUG] Loss values - box: {loss_l}, cls: {loss_c}, landmark: {loss_landm}")
        
        # In ra tất cả các giá trị loss
        print(f"[DEBUG] Batch {i} - Loss values - box: {loss_l}, cls: {loss_c}, landmark: {loss_landm}")

        # metric
        loss_cls += loss_c
        loss_box += loss_l 
        loss_pts += loss_landm

        # free after backward
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    
    # cls = classification; box = box regressionl; pts = landmark regression
    loss_cls = loss_cls/len(trainloader)
    loss_box = loss_box/len(trainloader)
    loss_pts = loss_pts/len(trainloader)

    return loss_cls, loss_box, loss_pts

def evaluate(model, anchors, validloader, loss_function, best_box, device='cpu'):
    model.eval()
    loss_cls, loss_box, loss_pts = 0, 0, 0
    count_img, count_target = 0, 0
    ap_5, ap_5_95 = 0, 0

    with torch.no_grad():
        for i, (input, targets) in enumerate(validloader):
            # Debug: Kiểm tra dữ liệu validation
            if i == 0:
                print(f"[DEBUG] Val input shape: {input.shape}")
                print(f"[DEBUG] Val targets len: {len(targets)}")
                if len(targets) > 0 and targets[0].shape[0] > 0:
                    print(f"[DEBUG] Val target example: {targets[0][0][:4]}")
                else:
                    print(f"[DEBUG] Val target is empty!")
            
            # load data into cuda
            input   = input.to(device)
            targets = [annos.to(device) for annos in targets]

            # forward
            predict = model(input)
            loss_l, loss_c, loss_landm = loss_function(predict, anchors, targets)
            
            # Debug: Kiểm tra loss trong validation
            if i == 0:
                print(f"[DEBUG] Val loss - box: {loss_l.item()}, cls: {loss_c.item()}, landmark: {loss_landm.item()}")

            # metric
            loss_cls += loss_c
            loss_box += loss_l 
            loss_pts += loss_landm

            # bap_5, bap_5_95 = calculate_running_map(targets, predict)
            # ap_5    += bap_5
            # ap_5_95 += bap_5_95

            # summary
            count_img += input.shape[0]
            for target in targets:
                count_target += target.shape[0]
    
    loss_cls = loss_cls/len(validloader)
    loss_box = loss_box/len(validloader)
    loss_pts = loss_pts/len(validloader)

    epoch_ap_5 = ap_5/len(validloader)
    epoch_ap_5_95 = ap_5_95/len(validloader)

    epoch_summary = [count_img, count_target, epoch_ap_5, epoch_ap_5_95]

    if loss_box>best_box:
    # export to onnx + pt
        torch.save(model.state_dict(), os.path.join(save_dir, 'weight.pth'))

    return loss_cls, loss_box, loss_pts, epoch_summary

if __name__ == '__main__':
    args = parse_args()

    # init wandb
    config = dict(
        epoch           = args.epoch,
        weight_decay    = args.weight_decay,
        momentum        = args.momentum,
        lr              = args.lr,
        batchsize       = args.batchsize,
        startfm         = args.startfm,
        input_size      = args.image_size
    )
    
    # log experiments to
    run = wandb.init(project='retina-face-detector-main', config=config)
    
    # use artifact
    #use_data_wandb(run=run, data_name=DATASET, download=args.download)

    # train on device
    device = select_device(args.device, args.batchsize)

    # get dataloader
    train_set = WiderFaceDataset(root_path=DATA_PATH, input_size=args.image_size, is_train=True)
    valid_set = WiderFaceDataset(root_path=DATA_PATH, input_size=args.image_size, is_train=False)
    
    print(f"\tNumber of training example: {len(train_set)}\n\tNumber of validation example: {len(valid_set)}")
    
    # Debug: Kiểm tra một mẫu dữ liệu đầu tiên
    print(f"[DEBUG] Checking first sample data...")
    sample_data, sample_target = train_set[0]
    print(f"[DEBUG] Sample data shape: {sample_data.shape}")
    print(f"[DEBUG] Sample target shape: {sample_target.shape}")
    print(f"[DEBUG] Sample target content: {sample_target}")

    torch.manual_seed(RANDOM_SEED)

    trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=detection_collate)
    validloader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=detection_collate)

    n_classes = N_CLASSES
    epochs = args.epoch
    # create dir for save weight
    save_dir = create_exp_dir()

    # get model and define loss func, optimizer
    model = RetinaFace(model_name=args.model, freeze_backbone=args.freeze, use_latent=True).to(device)
    if args.weight is not None and os.path.isfile(args.weight):
        checkpoint = torch.load(args.weight)
        model.load_state_dict(checkpoint)
        print(f'\tWeight located in {args.weight} have been loaded')

    cudnn.benchmark = True
    import numpy as np
    feat_shape = [
        (80, 80),   # feature_1
        (40, 40),   # feature_2
        (20, 20),   # feature_3
        (20, 20),   # feature_4
        (10, 10)    # feature_5
    ]
    with torch.no_grad():
        anchors = Anchors(
            pyramid_levels=[2, 3, 4, 5, 6],
            image_size=(640, 640),
            ratios=[1.0],
            scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
            feat_shape=feat_shape
        ).forward().to(device)
        
        # Debug: Kiểm tra anchors
        print(f"[DEBUG] Anchors shape: {anchors.shape}")
        print(f"[DEBUG] Anchors min: {anchors.min().item()}, max: {anchors.max().item()}")
        print(f"[DEBUG] First 3 anchors: {anchors[:3]}")

    # optimizer + citeration
    optimizer   = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion   = MultiBoxLoss(N_CLASSES, 
                    overlap_thresh=OVERLAP_THRES, 
                    prior_for_matching=True, 
                    bkg_label=BKG_LABEL, neg_pos=True, 
                    neg_mining=NEG_MINING, neg_overlap=NEG_OVERLAP, 
                    encode_target=False, device=device)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_MILESTONE, gamma=0.7)

    # wandb watch
    run.watch(models=model, criterion=criterion, log='all', log_freq=10)

    # training
    best_ap = -1

    for epoch in range(epochs):
        print(f'\n\tEpoch\tbox\t\tlandmarks\tcls\t\ttotal')
        t0 = time.time()
        loss_cls, loss_box, loss_pts = train(model, anchors, trainloader, optimizer, criterion, device)
        t1 = time.time()

        total_loss = loss_box + loss_pts + loss_cls
        # epoch
        wandb.log({'loss_cls': loss_cls, 'loss_box': loss_box, 'loss_landmark': loss_pts}, step=epoch)
        print(f'\t{epoch+1}/{epochs}\t{loss_box:.5f}\t\t{loss_pts:.5f}\t\t{loss_cls:.5f}\t\t{total_loss:.5f}\t\t{(t1-t0):.2f}s')
        
        # summary [count_img, count_target, epoch_ap_5, epoch_ap_5_95]
        t0 = time.time()
        loss_cls, loss_box, loss_pts, summary = evaluate(model, anchors, validloader, criterion, loss_box, device)
        t1 = time.time()

        # images, labels, P, R, map_5, map_95
        print(f'\n\tImages\tLabels\t\tbox\t\tlandmarks\tcls\t\tmAP@.5\t\tmAP.5.95')
        # print(f'\t{summary[0]}\t{summary[1]}\t\t{summary[2]}\t\t{summary[3]}')
        print(f'\t{summary[0]}\t{summary[1]}\t\t{loss_box:.5f}\t\t{loss_pts:.3f}\t\t{loss_cls:.5f}\t\t{(t1-t0):.2f}s')
    
        wandb.log({'val.loss_cls': loss_cls, 'val.loss_box': loss_box, 'val.loss_landmark': loss_pts}, step=epoch)
        wandb.log({'metric.map@.5': summary[2], 'metric.map@.5:.95': summary[3]}, step=epoch)
        wandb.log({"lr": scheduler.get_last_lr()[0]}, step=epoch)
        
        # decrease lr
        scheduler.step()

        # Wandb summary
        # if summary[2] > best_ap:
        #     best_ap = summary[2] 
        #     wandb.run.summary["best_accuracy"] = best_ap

    if not args.tuning:
        trained_weight = wandb.Artifact(args.run, type='WEIGHTS')
        # trained_weight.add_file(os.path.join(save_dir, 'weight.onnx'))
        trained_weight.add_file(os.path.join(save_dir, 'weight.pth'))
        wandb.log_artifact(trained_weight)