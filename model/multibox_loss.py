# author: https://github.com/biubug6/
# license: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.num_classes    = num_classes
        self.threshold      = overlap_thresh
        self.background_label = bkg_label
        self.encode_target  = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining  = neg_mining
        self.negpos_ratio   = neg_pos
        self.neg_overlap    = neg_overlap
        self.variance       = [0.1, 0.2]
        self.device         = device

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data, landm_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))
        
        # Debug print
        print(f"[DEBUG] MultiboxLoss: loc_data: {loc_data.shape}, conf_data: {conf_data.shape}, landm_data: {landm_data.shape}")
        print(f"[DEBUG] MultiboxLoss: priors: {priors.shape}, num_priors: {num_priors}")
        
        # Kiểm tra targets
        print(f"[DEBUG] MultiboxLoss: targets length: {len(targets)}")
        for i, target in enumerate(targets[:2]):  # Chỉ in 2 targets đầu tiên
            if target.shape[0] > 0:
                print(f"[DEBUG] MultiboxLoss: target[{i}] shape: {target.shape}")
                print(f"[DEBUG] MultiboxLoss: target[{i}] example: {target[0]}")
            else:
                print(f"[DEBUG] MultiboxLoss: target[{i}] is empty!")

        # match priors (default boxes) and ground truth boxes
        loc_t   = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t  = torch.LongTensor(num, num_priors)
        for idx in range(num):
            if idx < 2:  # Debug cho 2 mẫu đầu tiên
                print(f"[DEBUG] Processing batch idx: {idx}")
            
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            landms = targets[idx][:, 4:14].data
            defaults = priors.data
            
            if idx < 2:  # Debug cho 2 mẫu đầu tiên
                print(f"[DEBUG] truths shape: {truths.shape}, labels shape: {labels.shape}")
                print(f"[DEBUG] landms shape: {landms.shape}, defaults shape: {defaults.shape}")
                if truths.shape[0] > 0:
                    print(f"[DEBUG] truths[0]: {truths[0]}, labels[0]: {labels[0]}")
                    
                    # Kiểm tra giá trị truths
                    if torch.isnan(truths).any():
                        print(f"[DEBUG] WARNING: truths contains NaN values!")
                    if torch.isinf(truths).any():
                        print(f"[DEBUG] WARNING: truths contains Inf values!")
                    
                    # Kiểm tra kích thước bbox có hợp lệ không
                    width = truths[:, 2] - truths[:, 0]
                    height = truths[:, 3] - truths[:, 1]
                    if (width <= 0).any() or (height <= 0).any():
                        print(f"[DEBUG] WARNING: Invalid bbox dimensions found!")
                        print(f"[DEBUG] Invalid widths: {width[width <= 0]}")
                        print(f"[DEBUG] Invalid heights: {height[height <= 0]}")
            
            loc_t, conf_t, landm_t = match(self.threshold, 
                                        truths, defaults, 
                                        self.variance, labels, 
                                        landms, loc_t, conf_t, 
                                        landm_t, idx)
            
            if idx < 2:  # Debug cho 2 mẫu đầu tiên sau khi match
                print(f"[DEBUG] After match - loc_t[{idx}] non-zero: {(loc_t[idx] != 0).sum().item()}")
                print(f"[DEBUG] After match - conf_t[{idx}] non-zero: {(conf_t[idx] != 0).sum().item()}")
                print(f"[DEBUG] After match - landm_t[{idx}] non-zero: {(landm_t[idx] != 0).sum().item()}")

            loc_t   = loc_t.to(self.device)
            conf_t  = conf_t.to(self.device)
            landm_t = landm_t.to(self.device)

        zeros = torch.tensor(0).to(self.device)
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = conf_t > zeros
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        
        # Debug landm loss
        print(f"[DEBUG] landm loss - pos1 sum: {pos1.sum().item()}, num_pos_landm: {num_pos_landm.sum().item()}, N1: {N1.item()}")
        
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 10)
        landm_t = landm_t[pos_idx1].view(-1, 10)
        
        # Debug landm values
        if landm_p.shape[0] > 0:
            print(f"[DEBUG] landm_p shape: {landm_p.shape}, min: {landm_p.min().item()}, max: {landm_p.max().item()}")
            print(f"[DEBUG] landm_t shape: {landm_t.shape}, min: {landm_t.min().item()}, max: {landm_t.max().item()}")
        else:
            print(f"[DEBUG] WARNING: No positive landm samples found!")
        
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')

        pos = conf_t != zeros
        conf_t[pos] = 1
        
        # Debug pos
        print(f"[DEBUG] pos sum: {pos.sum().item()}")

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        
        # Debug loc values
        if loc_p.shape[0] > 0:
            print(f"[DEBUG] loc_p shape: {loc_p.shape}, min: {loc_p.min().item()}, max: {loc_p.max().item()}")
            print(f"[DEBUG] loc_t shape: {loc_t.shape}, min: {loc_t.min().item()}, max: {loc_t.max().item()}")
        else:
            print(f"[DEBUG] WARNING: No positive loc samples found!")
        
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        
        # Debug hard negative mining
        print(f"[DEBUG] num_pos: {num_pos.sum().item()}, num_neg: {num_neg.sum().item()}, neg sum: {neg.sum().item()}")

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        
        # Debug conf values
        if conf_p.shape[0] > 0:
            print(f"[DEBUG] conf_p shape: {conf_p.shape}, targets_weighted shape: {targets_weighted.shape}")
        else:
            print(f"[DEBUG] WARNING: No samples for confidence loss!")
        
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1
        
        # Final loss values
        print(f"[DEBUG] Final losses - loc: {loss_l.item()}, conf: {loss_c.item()}, landm: {loss_landm.item()}")

        return loss_l, loss_c, loss_landm