import os
import cv2
import torch
import time
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from math import sqrt

from model.anchor import Anchors
from model.config import *
from model.model import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.nms import py_cpu_nms
from utils.generate import PriorBox, select_device

def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run detection on an image')
    parser.add_argument('-i', '--input', type=str, default=None, help='image for detection')
    parser.add_argument('-l', '--latent', type=str, default=None, help='latent .npy file for detection')
    parser.add_argument('-o', '--output', type=str, default=None, help='output filename')
    parser.add_argument('--weight', type=str, default=None, help='snapshot model file')
    parser.add_argument('--network', type=str, default='resnet18', help="input network model")
    parser.add_argument('--conf', type=float, default=0.6, help="confidence threshold")
    parser.add_argument('--nms', type=float, default=0.4, help="nms threshold")
    parser.add_argument('--topk', type=int, default=10, help="top-k before nms")
    parser.add_argument('--visualize', action='store_true', help="visualize results")
    parser.add_argument('--keep', type=int, default=750, help="maximum results")
    parser.add_argument('--cpu', action='store_true', help="use cpu")
    parser.add_argument('-s', '--size', type=int, default=840, help="image size")
    parser.add_argument('--scale', type=float, default=1.0, help="scale")
    parser.add_argument('--device', type=str, default=None, help="device selection")

    args = parser.parse_args()
    return args

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def detect_latent(latent_path, output_path=None, weight='weights/mobilenet0.25_Final.pth',
                network='resnet50', confidence_threshold=0.6, nms_threshold=0.4, top_k=5000, 
                keep_top_k=750, visualize=False, cpu=False):

    torch.set_grad_enabled(False)
    device = select_device(args.device)

    # load latent representation
    latent = np.load(latent_path)
    
    # Ensure latent has proper dimensions [256, 40, 40]
    if len(latent.shape) == 4 and latent.shape[0] == 1:  # If shape is [1, 256, 40, 40]
        latent = latent.squeeze(0)  # Convert to [256, 40, 40]
        
    # Add batch dimension
    latent = torch.from_numpy(latent).unsqueeze(0).float().to(device)
    
    # Load model
    print(f'Creating {network}')
    net = RetinaFace(model_name=network, is_train=False, use_latent_input=True)
    net = load_model(net, weight, cpu)
    net.eval()
    
    print('Finished loading model.')
    net = net.to(device)
    
    # Forward pass
    loc, conf, landms = net(latent)
    
    # Get priors
    priorbox = PriorBox(image_size=(640, 640))
    priors = priorbox.forward().to(device)
    prior_data = priors.data
    
    # Post-process detections
    boxes = decode(loc.data.squeeze(0), prior_data, [0.1, 0.2])
    boxes = boxes.cpu().numpy()
    
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    
    landms = decode_landm(landms.data.squeeze(0), prior_data, [0.1, 0.2])
    landms = landms.cpu().numpy()
    
    # Ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]
    
    # Keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]
    
    # Do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]
    
    # Keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]
    
    # Return detections
    return dets, landms

if __name__ == '__main__':
    if args.input is not None:
        # Process image input
        image_path = args.input
        output_path = args.output if args.output else 'results.jpg'
        weight_path = args.weight if args.weight else 'weights/mobilenet0.25_Final.pth'
        
        # ... (existing image processing code)
        
    elif args.latent is not None:
        # Process latent input
        latent_path = args.latent
        output_path = args.output if args.output else 'results.jpg'
        weight_path = args.weight if args.weight else 'weights/resnet50_latent.pth'
        
        # Detect from latent
        dets, landms = detect_latent(
            latent_path=latent_path,
            output_path=output_path,
            weight=weight_path,
            network=args.network,
            confidence_threshold=args.conf,
            nms_threshold=args.nms,
            top_k=args.topk,
            keep_top_k=args.keep,
            visualize=args.visualize,
            cpu=args.cpu
        )
        
        # Print results
        print(f"Found {len(dets)} faces from latent input")
        for i, (box, score) in enumerate(zip(dets[:, :4], dets[:, 4])):
            print(f"Face {i+1}: confidence = {score:.4f}, bbox = {box}")
    
    else:
        print("Please provide either --input for image input or --latent for latent input")

    torch.set_grad_enabled(False)
    args = parse_args()

    # train on device
    device = torch.device("cpu")

    if args.device is not None:
        device = torch.device(args.device)
    print(f"\tCurrent training device {torch.cuda.get_device_name(device)}")
    
    # net & model
    model = RetinaFace(model_name=args.model, is_train=False).to(device)
    if args.weight is not None and os.path.isfile(args.weight):
        checkpoint = torch.load(args.weight, map_location=device)
        model.load_state_dict(checkpoint)
        print(f'\tWeight located in {args.weight} have been loaded')

    model.eval()

    cudnn.benchmark = True

    # load image
    img_raw = cv2.imread(args.path, cv2.IMREAD_COLOR)

    img = np.float32(img_raw)

    im_height, im_width, _ = img.shape
    # scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    scale = torch.Tensor([im_height, im_width, im_height, im_width])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)
    
    resize = 1

    # prediction
    tic = time.time()
    loc, conf, landms = model(img)  # forward pass
    print('\tModel forward time: {:.4f}'.format(time.time() - tic))

    input_size = np.array([im_height, im_width])
    # input_size = np.array([im_width, im_height])
    anchors = Anchors(image_size=input_size, pyramid_levels=model.feature_map).forward().to(device)

    boxes = decode(loc.data.squeeze(), anchors, [0.1, 0.2])
    # from IPython import embed
    # embed()
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), anchors, [0.1, 0.2])
    # scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
    #                         img.shape[3], img.shape[2], img.shape[3], img.shape[2],
    #                         img.shape[3], img.shape[2]])
    scale1 = torch.Tensor([im_height, im_width, im_height, im_width,
                            im_height, im_width, im_height, im_width,
                            im_height, im_width])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    # show image
    if args.save_image:
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
        # save image

        name = "test.jpg"
        cv2.imwrite(name, img_raw)

    