import torch
import torch.nn as nn
from math import sqrt, pow
import torch.nn.functional as F

from model.config import *
from model._utils import IntermediateLayerGetter
from model.common import FPN, SSH, MobileNetV1, BridgeModule

class ClassHead(nn.Module):
    def __init__(self, in_channels, num_anchors=3):
        """
        Face classification 
        """
        super(ClassHead, self).__init__()
        self.conv        = nn.Conv2d(in_channels, num_anchors*2, 1)

    def forward(self, input):
        out = self.conv(input)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self, in_channels, num_anchors=3):
        """
        Face bounding box
        """
        super(BboxHead, self).__init__()
        self.conv        = nn.Conv2d(in_channels, num_anchors*4, 1)

    def forward(self, input):
        out = self.conv(input)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self, in_channels, num_anchors=3):
        """
        Facial landmark
        """
        super(LandmarkHead, self).__init__()
        # 5 (x, y) refer to coordinate of 5 landmarks
        self.conv = nn.Conv2d(in_channels, num_anchors*10, 1)

    def forward(self, input):
        out = self.conv(input)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, model_name='resnet50', freeze_backbone=False, pretrain_path=None, is_train=True, use_latent=True):
        """
        Model RetinaFace for face recognition based on:
        `"RetinaFace: Single-stage Dense Face Localisation in the Wild" <https://arxiv.org/abs/1905.00641>`_.
        """
        super(RetinaFace, self).__init__()
        self.is_train = is_train
        self.use_latent = use_latent
        self.bridge = BridgeModule() if use_latent else None
        self.fpn = None
        self.ssh = None
        self.body = None
        self.feature_map = None
        self.pruned_resnet = None
        if use_latent:
            import torchvision.models as models
            backbone = models.resnet50(pretrained=True)
            # Prune 11 lớp đầu: giữ lại layer2, layer3, layer4
            self.pruned_resnet = nn.Sequential(
                backbone.layer2,  # [B, 512, 80, 80]
                backbone.layer3,  # [B, 1024, 40, 40]
                backbone.layer4   # [B, 2048, 20, 20]
            )
            self.feature_map = [2, 3, 4, 5, 6]  # giữ nguyên cho FPN
            in_channels_list = [512, 1024, 2048, 2048]
            self.fpn = FPN(in_channels_list=in_channels_list, out_channels=256)
            self.ssh = SSH(in_channels=256, out_channels=256)
        else:
            # load backbone
            backbone = None
            if model_name == 'mobilenet0.25':
                backbone            = MobileNetV1(start_frame=START_FRAME)
                return_feature      = RETURN_MAP_MOBN1
                self.feature_map    = FEATURE_MAP_MOBN1
                
                if not pretrain_path is None:
                    pretrain_weight = torch.load(pretrain_path)
                    backbone.load_state_dict(pretrain_weight)

            elif model_name == 'mobilenetv2':
                return_feature = FEATURE_MAP_MOBN2
                backbone = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
            
            elif 'resnet' in model_name:
                if '18' in  model_name:
                    backbone = models.resnet18(pretrained=True)

                elif '34' in  model_name:
                    backbone = models.resnet34(pretrained=True)

                elif '50' in model_name:
                    backbone = models.resnet50(pretrained=True)
                
                return_feature      = RETURN_MAP
                self.feature_map    = FEATURE_MAP

            else:
                print(f'Unable to select {model_name}.')

            num_fpn             = len(self.feature_map)

            # frozen pre-trained backbone
            self.body = IntermediateLayerGetter(backbone, return_feature)

            if freeze_backbone:
                for param in self.body.parameters():
                    param.requires_grad = False
                print('\tBackbone freezed')

            in_channels_list = [IN_CHANNELS*2, IN_CHANNELS*4, IN_CHANNELS*8, IN_CHANNELS*16]
            self.fpn = FPN(in_channels_list=in_channels_list, out_channels=OUT_CHANNELS)
            self.ssh = SSH(in_channels=OUT_CHANNELS, out_channels=OUT_CHANNELS)

        num_fpn = 5
        self.ClassHead      = self._make_class_head(inchannels=256, anchor_num=3, fpn_num=num_fpn)
        self.BboxHead       = self._make_bbox_head(inchannels=256, anchor_num=3, fpn_num=num_fpn)
        self.LandmarkHead   = self._make_landmark_head(inchannels=256, anchor_num=3, fpn_num=num_fpn)

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self, input):
        """
        The input to the RetinaFace is expected to be a Tensor

        Args:
            input (Tensor): image(s) for feed forward
        """
        if self.use_latent:
            # input: [B, 256, 40, 40]
            x = self.bridge(input)  # [B, 256, 160, 160]
            # Pruned ResNet-50
            c2 = self.pruned_resnet[0](x)  # [B, 512, 80, 80]
            c3 = self.pruned_resnet[1](c2) # [B, 1024, 40, 40]
            c4 = self.pruned_resnet[2](c3) # [B, 2048, 20, 20]
            # FPN expects dict of features
            features = {'c2': c2, 'c3': c3, 'c4': c4, 'c5': c4}  # c5 = c4 (no extra layer)
            fpn = self.fpn(features)
            feature_1 = self.ssh(fpn[0])
            feature_2 = self.ssh(fpn[1])
            feature_3 = self.ssh(fpn[2])
            feature_4 = self.ssh(fpn[3])
            feature_5 = self.ssh(fpn[4])
            print('[DEBUG] feature_1:', feature_1.shape)
            print('[DEBUG] feature_2:', feature_2.shape)
            print('[DEBUG] feature_3:', feature_3.shape)
            print('[DEBUG] feature_4:', feature_4.shape)
            print('[DEBUG] feature_5:', feature_5.shape)
            features = [feature_1, feature_2, feature_3, feature_4, feature_5]
        else:
            out = self.body(input)

            # Feature Pyramid Net
            fpn = self.fpn(out)

            # Single-stage headless
            feature_1 = self.ssh(fpn[0])
            feature_2 = self.ssh(fpn[1])
            feature_3 = self.ssh(fpn[2])
            feature_4 = self.ssh(fpn[3])
            feature_5 = self.ssh(fpn[4])
            print('[DEBUG] feature_1:', feature_1.shape)
            print('[DEBUG] feature_2:', feature_2.shape)
            print('[DEBUG] feature_3:', feature_3.shape)
            print('[DEBUG] feature_4:', feature_4.shape)
            print('[DEBUG] feature_5:', feature_5.shape)
            features = [feature_1, feature_2, feature_3, feature_4, feature_5]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications  = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions  = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.is_train:
            output = (bbox_regressions, classifications, ldm_regressions)
        else: 
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)

        return output

def forward(model, input, targets, anchors, loss_function, optimizer):
    """Due to the probability of OOM problem can happen, which might
    cause the "CUDA out of memory". I've passed all require grad into
    a function to free it while there is nothing refer to it.
    """
    print("[DEBUG] forward function - Computing loss and backprop")
    predict = model(input)
    
    # Debug predictions before loss computation
    for i, pred in enumerate(predict):
        print(f"[DEBUG] forward function - prediction[{i}] shape: {pred.shape}")
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            print(f"[DEBUG] WARNING: Prediction {i} contains NaN or Inf!")
        print(f"[DEBUG] forward function - prediction[{i}] min: {pred.min().item()}, max: {pred.max().item()}, mean: {pred.mean().item()}")
    
    # Debug targets before loss computation
    print(f"[DEBUG] forward function - targets length: {len(targets)}")
    for i, target in enumerate(targets[:2]):  # Debug first 2 targets
        if target.shape[0] > 0:
            print(f"[DEBUG] forward function - target[{i}] shape: {target.shape}")
            print(f"[DEBUG] forward function - target[{i}] bbox example: {target[0, :4]}")
        else:
            print(f"[DEBUG] forward function - target[{i}] is empty!")
    
    # Debug anchors
    print(f"[DEBUG] forward function - anchors shape: {anchors.shape}")
    print(f"[DEBUG] forward function - anchors min: {anchors.min().item()}, max: {anchors.max().item()}")
    
    # Compute loss
    loss_l, loss_c, loss_landm = loss_function(predict, anchors, targets)
    loss = 1.3*loss_l + loss_c + loss_landm
    
    # Debug loss values
    print(f"[DEBUG] forward function - Loss components - box: {loss_l.item()}, cls: {loss_c.item()}, landmark: {loss_landm.item()}")
    print(f"[DEBUG] forward function - Total loss: {loss.item()}")

    loss_l      = loss_l.item()
    loss_c      = loss_c.item()
    loss_landm  = loss_landm.item()

    # zero the gradient + backprpagation + step
    optimizer.zero_grad()
    
    # Kiểm tra đầu ra 
    print(f"[DEBUG] forward function - Before backward, loss requires_grad: {loss.requires_grad}")
    
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    
    # Tiến hành backpropagation
    loss.backward()
    
    # Debug gradients
    print("[DEBUG] forward function - After backward, checking gradients...")
    total_norm = 0
    param_count = 0
    zero_grad_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count += 1
            if param.grad is None:
                print(f"[DEBUG] WARNING: {name} has no gradient!")
                zero_grad_count += 1
            else:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"[DEBUG] forward function - Total gradient norm: {total_norm}")
    print(f"[DEBUG] forward function - Parameters with no gradients: {zero_grad_count}/{param_count}")
    
    # Tiến hành bước optimizer
    optimizer.step()
    
    # Debug parameters after step
    print("[DEBUG] forward function - After optimizer step, checking parameter changes...")
    for name, param in list(model.named_parameters())[:5]:  # Kiểm tra 5 tham số đầu tiên
        if param.requires_grad:
            print(f"[DEBUG] forward function - {name} stats - mean: {param.data.mean().item()}, std: {param.data.std().item()}")

    del predict
    del loss

    return loss_l, loss_c, loss_landm
