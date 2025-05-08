import torch
import numpy as np
import torch.nn as nn

from model.config import *
from utils.box_utils import point_form

class Anchors(nn.Module):
    def __init__(self, pyramid_levels=[2, 3, 4, 5, 6], image_size=(640, 640), 
                 ratios=[1.0], scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
                 feat_shape=None):
        """
        Sample anchorbox for RetinaNet.

        Args:
            pyramid_levels (list): Feature pyramid levels to use
            image_size (tuple): (h, w) of image size
            ratios (list): List of ratios, each have 3 scales
            scales (list): List of scales for each ratio

        Returns:
            anchors: (h, w, 4) Generated anchor box of (x, y, w, h)
        """
        super(Anchors, self).__init__()
        self.pyramid_levels = pyramid_levels
        self.image_size = image_size
        
        # Debug trước khi đổi
        print(f"[DEBUG] Anchors init - Original ratios: {ratios}, scales: {scales}")
        
        # FIX: Thêm nhiều ratios hơn để bắt được các khuôn mặt có nhiều dạng khác nhau
        # Faces thường có ratio từ 1:1 đến 1:1.5
        self.ratios = [0.75, 1.0, 1.33]  # thay vì chỉ [1.0]
        
        # FIX: Đa dạng hóa scales để bắt được faces ở nhiều kích thước khác nhau
        # Với input size 640x640, faces có thể chiếm từ 5% đến 30% kích thước ảnh
        self.scales = [2 ** (-0.5), 2 ** 0, 2 ** 0.5]  # thêm scale nhỏ hơn và lớn hơn
        
        # Debug sau khi đổi
        print(f"[DEBUG] Anchors init - New ratios: {self.ratios}, scales: {self.scales}")
        
        self.strides = [2 ** x for x in self.pyramid_levels]
        self.feat_shape = feat_shape
        self.sizes = []
        for i in range(len(self.pyramid_levels)):
            if self.feat_shape is not None:
                self.sizes.append([self.feat_shape[i][0], self.feat_shape[i][1]])
            else:
                self.sizes.append([image_size[0] // self.strides[i], image_size[1] // self.strides[i]])

    def forward(self):
        anchors = []
        for idx, p in enumerate(self.pyramid_levels):
            # for all anchors in one pyramid level (e.g. 80x80, 40x40, 20x20, 10x10)
            x_size, y_size = self.sizes[idx]
            grid_height, grid_width = y_size, x_size

            strides = self.strides[idx]
            stride = torch.tensor([strides, strides])

            # generate center offset for each pixel 
            shift_x = torch.arange(0, grid_width) + 0.5
            shift_y = torch.arange(0, grid_height) + 0.5

            # repeat to get meshgrid
            shift_x = shift_x.unsqueeze(0).repeat(grid_height, 1).view(-1)
            shift_y = shift_y.unsqueeze(1).repeat(1, grid_width).view(-1)

            # all grid points
            shift = torch.stack([shift_x, shift_y], dim=1)
            
            # Normalize coordinates to [0-1] range
            shifts = shift * stride / torch.tensor(self.image_size)
            
            # Generate all box combinations for one image
            # box_widths = []
            # box_heights = []
            # box_scales = []
            # box_ratios = []

            # for ratio in self.ratios:
            #     for scale in self.scales:
            #         box_scales.append(scale)
            #         box_ratios.append(ratio)

            #         # convert ratio to height and width ratio
            #         ratio_sqrt = np.sqrt(ratio)
            #         box_widths.append(scale / ratio_sqrt)
            #         box_heights.append(scale * ratio_sqrt)

            boxes = []
            # for each scale, ratio, we generate one box for each pixel
            for scale in self.scales:
                for ratio in self.ratios:
                    # FIX: Tính toán kích thước box từ ratio và scale
                    # Scale là hệ số điều chỉnh kích thước chung của box
                    # Ratio là tỷ lệ width/height, cần điều chỉnh riêng cho width và height
                    ratio_sqrt = torch.sqrt(torch.tensor(ratio))
                    base_anchor_size = 0.15  # FIX: base size tương đương với 15% kích thước ảnh
                    width = base_anchor_size * scale / ratio_sqrt
                    height = base_anchor_size * scale * ratio_sqrt
                    boxes.append(torch.tensor([0, 0, width, height]))
            
            # cat and reshape
            boxes = torch.stack(boxes, dim=0)
            all_anchors = shifts.unsqueeze(1) + torch.cat([torch.zeros_like(boxes[:, :2]), boxes[:, 2:]], dim=1)
            
            # FIX: Chuyển từ center-size format sang x1,y1,x2,y2 format
            all_anchors = torch.cat([
                all_anchors[:, :2] - all_anchors[:, 2:] / 2,  # x1, y1
                all_anchors[:, :2] + all_anchors[:, 2:] / 2   # x2, y2
            ], dim=1)
            
            # Kiểm tra xem có anchor nào bị âm hoặc lớn hơn 1 không
            if (all_anchors < 0).any() or (all_anchors > 1).any():
                print(f"[DEBUG] Warning: Some anchors are outside [0, 1] range")
                # Đảm bảo anchors nằm trong khoảng [0, 1]
                all_anchors = torch.clamp(all_anchors, 0, 1)
            
            anchors.append(all_anchors.reshape(-1, 4))
            
        anchors = torch.cat(anchors, dim=0)
        return anchors

def generate_anchors(num_anchors=None, base_size=8, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = [0.5, 1, 2]

    if scales is None:
        scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    if num_anchors == None:
        num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 3] = np.sqrt(areas / np.repeat(ratios, len(scales))) # h
    anchors[:, 2] = anchors[:, 3] * np.repeat(ratios, len(scales))  # w

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    # anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    # anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    # keep it form (0, 0, w, h)
    return anchors

def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]

    re_anchors  = anchors.reshape((1, A, 4))
    shifted     = shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    shifted[:,:,2:] = 0 # format (x_c, y_c, w, h) need to maintain w, h

    all_anchors = re_anchors + shifted
    all_anchors = all_anchors.reshape((K * A, 4))
    all_anchors = point_form(all_anchors)

    return all_anchors