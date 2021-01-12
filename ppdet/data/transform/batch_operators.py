# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import logging
import cv2
import numpy as np

from .operators import register_op, BaseOperator
from .op_helper import jaccard_overlap, gaussian2D, overlap_matrix
#from ppdet.modeling.ops import YOLOAnchorGenerator, Max_IoU_Assigner
logger = logging.getLogger(__name__)

__all__ = [
    'PadBatch',
    'RandomShape',
    'PadMultiScaleTest',
    'Gt2YoloTarget',
    'Gt2YoloTarget_1vN',
    'Gt2YoloTarget_topk',
    'Gt2FCOSTarget',
    'Gt2TTFTarget',
]


@register_op
class PadBatch(BaseOperator):
    """
    Pad a batch of samples so they can be divisible by a stride.
    The layout of each image should be 'CHW'.
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
    """

    def __init__(self, pad_to_stride=0, use_padded_im_info=True):
        super(PadBatch, self).__init__()
        self.pad_to_stride = pad_to_stride
        self.use_padded_im_info = use_padded_im_info

    def __call__(self, samples, context=None):
        """
        Args:
            samples (list): a batch of sample, each is dict.
        """
        coarsest_stride = self.pad_to_stride
        if coarsest_stride == 0:
            return samples
        max_shape = np.array([data['image'].shape for data in samples]).max(
            axis=0)

        if coarsest_stride > 0:
            max_shape[1] = int(
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(
                np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)

        padding_batch = []
        for data in samples:
            im = data['image']
            im_c, im_h, im_w = im.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = im
            data['image'] = padding_im
            if self.use_padded_im_info:
                data['im_info'][:2] = max_shape[1:3]
            if 'semantic' in data.keys() and data['semantic'] is not None:
                semantic = data['semantic']
                padding_sem = np.zeros(
                    (1, max_shape[1], max_shape[2]), dtype=np.float32)
                padding_sem[:, :im_h, :im_w] = semantic
                data['semantic'] = padding_sem

        return samples


@register_op
class RandomShape(BaseOperator):
    """
    Randomly reshape a batch. If random_inter is True, also randomly
    select one an interpolation algorithm [cv2.INTER_NEAREST, cv2.INTER_LINEAR,
    cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]. If random_inter is
    False, use cv2.INTER_NEAREST.
    Args:
        sizes (list): list of int, random choose a size from these
        random_inter (bool): whether to randomly interpolation, defalut true.
    """

    def __init__(self, sizes=[], random_inter=False, resize_box=False):
        super(RandomShape, self).__init__()
        self.sizes = sizes
        self.random_inter = random_inter
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ] if random_inter else []
        self.resize_box = resize_box

    def __call__(self, samples, context=None):
        shape = np.random.choice(self.sizes)
        method = np.random.choice(self.interps) if self.random_inter \
            else cv2.INTER_NEAREST
        for i in range(len(samples)):
            im = samples[i]['image']
            h, w = im.shape[:2]
            scale_x = float(shape) / w
            scale_y = float(shape) / h
            im = cv2.resize(
                im, None, None, fx=scale_x, fy=scale_y, interpolation=method)
            samples[i]['image'] = im
            if self.resize_box and 'gt_bbox' in samples[i] and len(samples[0][
                    'gt_bbox']) > 0:
                scale_array = np.array([scale_x, scale_y] * 2, dtype=np.float32)
                samples[i]['gt_bbox'] = np.clip(samples[i]['gt_bbox'] *
                                                scale_array, 0,
                                                float(shape) - 1)
        return samples


@register_op
class PadMultiScaleTest(BaseOperator):
    """
    Pad the image so they can be divisible by a stride for multi-scale testing.
 
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
    """

    def __init__(self, pad_to_stride=0):
        super(PadMultiScaleTest, self).__init__()
        self.pad_to_stride = pad_to_stride

    def __call__(self, samples, context=None):
        coarsest_stride = self.pad_to_stride
        if coarsest_stride == 0:
            return samples

        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        if len(samples) != 1:
            raise ValueError("Batch size must be 1 when using multiscale test, "
                             "but now batch size is {}".format(len(samples)))
        for i in range(len(samples)):
            sample = samples[i]
            for k in sample.keys():
                # hard code
                if k.startswith('image'):
                    im = sample[k]
                    im_c, im_h, im_w = im.shape
                    max_h = int(
                        np.ceil(im_h / coarsest_stride) * coarsest_stride)
                    max_w = int(
                        np.ceil(im_w / coarsest_stride) * coarsest_stride)
                    padding_im = np.zeros(
                        (im_c, max_h, max_w), dtype=np.float32)

                    padding_im[:, :im_h, :im_w] = im
                    sample[k] = padding_im
                    info_name = 'im_info' if k == 'image' else 'im_info_' + k
                    # update im_info
                    sample[info_name][:2] = [max_h, max_w]
        if not batch_input:
            samples = samples[0]
        return samples


@register_op
class Gt2YoloTarget(BaseOperator):
    """
    Generate YOLOv3 targets by groud truth data, this operator is only used in
    fine grained YOLOv3 loss mode
    """

    def __init__(self,
                 anchors,
                 anchor_masks,
                 downsample_ratios,
                 num_classes=80,
                 iou_thresh=1.):
        super(Gt2YoloTarget, self).__init__()
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh

    def __call__(self, samples, context=None):
        assert len(self.anchor_masks) == len(self.downsample_ratios), \
            "anchor_masks', and 'downsample_ratios' should have same length."

        h, w = samples[0]['image'].shape[1:3]
        an_hw = np.array(self.anchors) / np.array([[w, h]])
        for sample in samples:
            # im, gt_bbox, gt_class, gt_score = sample
            im = sample['image']
            gt_bbox = sample['gt_bbox']
            gt_class = sample['gt_class']
            gt_score = sample['gt_score']
            for i, (
                    mask, downsample_ratio
            ) in enumerate(zip(self.anchor_masks, self.downsample_ratios)):
                grid_h = int(h / downsample_ratio)
                grid_w = int(w / downsample_ratio)
                target = np.zeros(
                    (len(mask), 6 + self.num_classes, grid_h, grid_w),
                    dtype=np.float32)
                for b in range(gt_bbox.shape[0]):
                    gx, gy, gw, gh = gt_bbox[b, :]
                    cls = gt_class[b]
                    score = gt_score[b]
                    if gw <= 0. or gh <= 0. or score <= 0.:
                        continue

                    # find best match anchor index
                    best_iou = 0.
                    best_idx = -1
                    for an_idx in range(an_hw.shape[0]):
                        iou = jaccard_overlap(
                            [0., 0., gw, gh],
                            [0., 0., an_hw[an_idx, 0], an_hw[an_idx, 1]])
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = an_idx

                    gi = int(gx * grid_w)
                    gj = int(gy * grid_h)

                    # gtbox should be regresed in this layes if best match 
                    # anchor index in anchor mask of this layer
                    if best_idx in mask:
                        best_n = mask.index(best_idx)

                        # x, y, w, h, scale
                        target[best_n, 0, gj, gi] = gx * grid_w - gi
                        target[best_n, 1, gj, gi] = gy * grid_h - gj
                        target[best_n, 2, gj, gi] = np.log(
                            gw * w / self.anchors[best_idx][0])
                        target[best_n, 3, gj, gi] = np.log(
                            gh * h / self.anchors[best_idx][1])
                        target[best_n, 4, gj, gi] = 2.0 - gw * gh

                        # objectness record gt_score
                        target[best_n, 5, gj, gi] = score

                        # classification
                        target[best_n, 6 + cls, gj, gi] = 1.

                    # For non-matched anchors, calculate the target if the iou 
                    # between anchor and gt is larger than iou_thresh
                    if self.iou_thresh < 1:
                        for idx, mask_i in enumerate(mask):
                            if mask_i == best_idx: continue
                            iou = jaccard_overlap(
                                [0., 0., gw, gh],
                                [0., 0., an_hw[mask_i, 0], an_hw[mask_i, 1]])
                            if iou > self.iou_thresh:
                                # x, y, w, h, scale
                                target[idx, 0, gj, gi] = gx * grid_w - gi
                                target[idx, 1, gj, gi] = gy * grid_h - gj
                                target[idx, 2, gj, gi] = np.log(
                                    gw * w / self.anchors[mask_i][0])
                                target[idx, 3, gj, gi] = np.log(
                                    gh * h / self.anchors[mask_i][1])
                                target[idx, 4, gj, gi] = 2.0 - gw * gh

                                # objectness record gt_score
                                target[idx, 5, gj, gi] = score

                                # classification
                                target[idx, 6 + cls, gj, gi] = 1.
                #np.save('c_target{}'.format(i), target)
                sample['target{}'.format(i)] = target
        assert len(self.anchor_masks) != len(self.downsample_ratios), \
            "end assigner"       
        return samples

@register_op
class Gt2YoloTarget_1vN(BaseOperator):
    """
    Generate YOLOv3 targets by groud truth data, this operator is only used in
    fine grained YOLOv3 loss mode
    """

    def __init__(self,
                 anchors,
                 anchor_masks,
                 downsample_ratios,
                 num_classes=80,
                 iou_thresh=1.,
                 eps=0.000001):
        super(Gt2YoloTarget_1vN, self).__init__()
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        #self.assigner = 
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh
        self.eps = eps
        
    def _get_reg_target(self, bboxes, gt_bboxes, stride):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.
        Args:
            bboxes (torch.Tensor): Source boxes, e.g., anchors.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.
            stride (torch.Tensor | int): Stride of bboxes.
        Returns:
            torch.Tensor: Box transformation deltas
        """

        assert bboxes.shape[0] == gt_bboxes.shape[0]
        assert bboxes.shape[-1] == gt_bboxes.shape[-1] == 4
        x_center_gt = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) * 0.5
        y_center_gt = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) * 0.5
        w_gt = gt_bboxes[..., 2] - gt_bboxes[..., 0]
        h_gt = gt_bboxes[..., 3] - gt_bboxes[..., 1]
        x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
        y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
        w = bboxes[..., 2] - bboxes[..., 0]
        h = bboxes[..., 3] - bboxes[..., 1]
        w_target = np.log((w_gt / w).clip(min=self.eps))
        h_target = np.log((h_gt / h).clip(min=self.eps))
        x_center_target = ((x_center_gt - x_center) / stride + 0.5).clip(
            self.eps, 1 - self.eps)
        y_center_target = ((y_center_gt - y_center) / stride + 0.5).clip(
            self.eps, 1 - self.eps)
        reg_targets = np.stack(
            [x_center_target, y_center_target, w_target, h_target], axis=-1)
        return reg_targets

    def __call__(self, samples, context=None):
        assert len(self.anchor_masks) == len(self.downsample_ratios), \
            "anchor_masks', and 'downsample_ratios' should have same length."
        h, w = samples[0]['image'].shape[1:3]
        an_hw = np.array(self.anchors) / np.array([[w, h]])
        for sample in samples:
            # im, gt_bbox, gt_class, gt_score = sample
            im = sample['image']
            gt_bbox = sample['gt_bbox']
            gt_class = sample['gt_class']
            gt_score = sample['gt_score']
            base_anchors = YOLOAnchorGenerator(base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]], strides=self.downsample_ratios)
            featmap_sizes = [(int(h/downsample_ratio), int(h/downsample_ratio)) for downsample_ratio in self.downsample_ratios]
            anchor_list = base_anchors.grid_anchors(featmap_sizes)
            anchors = np.concatenate((anchor_list[0],anchor_list[1],anchor_list[2]))
            num_level_bboxes = [len(anchor_list[0]),len(anchor_list[1]),len(anchor_list[2])]
            length = [anchor_list[i].shape[0] for i in range(3)]
            gt_bbox_filted = gt_bbox[(gt_bbox != 0).any(axis=1)]
            bboxes = gt_bbox_filted.copy()
            gt_class = gt_class[:len(gt_bbox_filted)]
            gt_score = gt_score[:len(gt_bbox_filted)]
            bboxes[:, 0] = (gt_bbox_filted[:,0] - gt_bbox_filted[:,2]/2) * w
            bboxes[:, 1] = (gt_bbox_filted[:,1] - gt_bbox_filted[:,3]/2) * h
            bboxes[:, 2] = (gt_bbox_filted[:,0] + gt_bbox_filted[:,2]/2) * w
            bboxes[:, 3] = (gt_bbox_filted[:,1] + gt_bbox_filted[:,3]/2) * h
            #assigner = TopK_Assigner(anchors=anchors,
                                        #gts=bboxes,
                                        #k=1)
            assigner = ATSS_Assigner(anchors=anchors,
                                    gts=bboxes,
                                    num_level_bboxes=num_level_bboxes,
                                    topk=9)
            assigned_result = assigner.assign()
            pos_idx = assigned_result>=0
            gt_bboxes = bboxes[assigned_result]
            target_boxes = [gt_bboxes[0:length[0]],gt_bboxes[length[0]:length[0]+length[1]],gt_bboxes[length[0]+length[1]:]]
            xywh_gts = gt_bbox_filted[assigned_result]
            scales = 2.0 - xywh_gts[:,2] * xywh_gts[:,3]
            #print('scales:', scales)
            gt_class = gt_class[:len(gt_bbox_filted)]
            gt_score = gt_score[:len(gt_bbox_filted)]
            gt_labels = gt_class[assigned_result]
            gt_scores = gt_score[assigned_result]
            reg_target = np.concatenate([self._get_reg_target(bboxes=anchor_list[i],gt_bboxes=target_boxes[i],stride=self.downsample_ratios[i]) for i in range(3)])
            reg_target[assigned_result<0] = 0
            #print('reg_target.shape:', reg_target[0].shape)
            scales_target = np.zeros((anchors.shape[0]),dtype=np.float32)
            scales_target[pos_idx] = scales[pos_idx]
            obj_target = np.zeros((anchors.shape[0]),dtype=np.float32)
            obj_target[pos_idx] = gt_scores[pos_idx]
            #print("obj_target:",obj_target[pos_idx].sum())
            label_target = np.eye(self.num_classes)[gt_labels]
            label_target[assigned_result<0] = 0
            for i, (
                    mask, downsample_ratio
            ) in enumerate(zip(self.anchor_masks, self.downsample_ratios)):
                grid_h = int(h / downsample_ratio)
                grid_w = int(w / downsample_ratio)
                target = np.zeros(
                    (len(mask), 6 + self.num_classes, grid_h, grid_w),
                    dtype=np.float32)
                if i==0:
                    start = 0
                    end = grid_h * grid_w * 3
                elif i==1:
                    start = int(grid_h * grid_w * 3 / 4) 
                    end = int(grid_h * grid_w * 5 * 3 / 4)
                else:
                    start = - grid_h * grid_w * 3
                    end = anchors.shape[0]
                target[:,0:4,...] = reg_target[start:end].reshape((grid_w,grid_h,len(mask),4)).transpose(2,3,0,1)
                target[:,4,...] = scales_target[start:end].reshape((grid_w,grid_h,len(mask))).transpose(2,0,1)
                target[:,5,...] = obj_target[start:end].reshape((grid_w,grid_h,len(mask))).transpose(2,0,1)
                target[:,6:,...] = label_target[start:end].reshape((grid_w,grid_h,len(mask),80)).transpose(2,3,0,1)
                sample['target{}'.format(i)] = target
        return samples

@register_op
class Gt2YoloTarget_topk(BaseOperator):
    """
    Generate YOLOv3 targets by groud truth data, this operator is only used in
    fine grained YOLOv3 loss mode
    """
    def __init__(self,
                 anchors,
                 anchor_masks,
                 downsample_ratios,
                 num_classes=80,
                 iou_thresh=1.,
                 eps=0.000001):
        super(Gt2YoloTarget_topk, self).__init__()
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh
        self.eps = eps
        
    def _get_reg_target(self, bboxes, gt_bboxes, stride):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.
        Args:
            bboxes (torch.Tensor): Source boxes, e.g., anchors.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.
            stride (torch.Tensor | int): Stride of bboxes.
        Returns:
            torch.Tensor: Box transformation deltas
        """

        assert bboxes.shape[0] == gt_bboxes.shape[0]
        assert bboxes.shape[-1] == gt_bboxes.shape[-1] == 4
        x_center_gt = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) * 0.5
        y_center_gt = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) * 0.5
        w_gt = gt_bboxes[..., 2] - gt_bboxes[..., 0]
        h_gt = gt_bboxes[..., 3] - gt_bboxes[..., 1]
        x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
        y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
        w = bboxes[..., 2] - bboxes[..., 0]
        h = bboxes[..., 3] - bboxes[..., 1]
        w_target = np.log((w_gt / w).clip(min=self.eps))
        h_target = np.log((h_gt / h).clip(min=self.eps))
        x_center_target = ((x_center_gt - x_center) / stride + 0.5).clip(
            self.eps, 1 - self.eps)
        y_center_target = ((y_center_gt - y_center) / stride + 0.5).clip(
            self.eps, 1 - self.eps)
        reg_targets = np.stack(
            [x_center_target, y_center_target, w_target, h_target], axis=-1)
        return reg_targets

    def __call__(self, samples, context=None):
        assert len(self.anchor_masks) == len(self.downsample_ratios), \
            "anchor_masks', and 'downsample_ratios' should have same length."
        h, w = samples[0]['image'].shape[1:3]
        an_hw = np.array(self.anchors) / np.array([[w, h]])
        for sample in samples:
            # im, gt_bbox, gt_class, gt_score = sample
            im = sample['image']
            gt_bbox = sample['gt_bbox']
            gt_class = sample['gt_class']
            gt_score = sample['gt_score']
            base_anchors = YOLOAnchorGenerator(base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]], strides=self.downsample_ratios)
            featmap_sizes = [(int(h/downsample_ratio), int(h/downsample_ratio)) for downsample_ratio in self.downsample_ratios]
            anchor_list = base_anchors.grid_anchors(featmap_sizes)
            anchors = np.concatenate((anchor_list[0],anchor_list[1],anchor_list[2]))
            length = [anchor_list[i].shape[0] for i in range(3)]
            gt_bbox_filted = gt_bbox[(gt_bbox != 0).any(axis=1)]
            bboxes = gt_bbox_filted.copy()
            gt_class = gt_class[:len(gt_bbox_filted)]
            gt_score = gt_score[:len(gt_bbox_filted)]
            bboxes[:, 0] = (gt_bbox_filted[:,0] - gt_bbox_filted[:,2]/2) * w
            bboxes[:, 1] = (gt_bbox_filted[:,1] - gt_bbox_filted[:,3]/2) * h
            bboxes[:, 2] = (gt_bbox_filted[:,0] + gt_bbox_filted[:,2]/2) * w
            bboxes[:, 3] = (gt_bbox_filted[:,1] + gt_bbox_filted[:,3]/2) * h
            assigner = TopK_Assigner(anchors=anchors,
                                        gts=bboxes,
                                        k=5)
            #assigner = Max_IoU_Assigner(anchors=anchors,
                                            #gts=bboxes,
                                            #pos_thr=0.5,
                                            #neg_thr=0.5)
            assigned_result, assigned_iou = assigner.assign()
            #print('assigned_iou:', assigned_iou[assigned_iou>0])
            pos_idx = assigned_result>=0
            #print("$$$$$$$$, out side the assigner, $$$$$$$$$")
            #print("pos_idx:",np.argwhere(assigned_result>=0))
            gt_bboxes = bboxes[assigned_result]
            target_boxes = [gt_bboxes[0:length[0]],gt_bboxes[length[0]:length[0]+length[1]],gt_bboxes[length[0]+length[1]:]]
            xywh_gts = gt_bbox_filted[assigned_result]
            scales = 2.0 - xywh_gts[:,2] * xywh_gts[:,3]
            #print('scales:', scales)
            gt_class = gt_class[:len(gt_bbox_filted)]
            gt_score = gt_score[:len(gt_bbox_filted)]
            gt_labels = gt_class[assigned_result]
            gt_scores = gt_score[assigned_result]
            reg_target = np.concatenate([self._get_reg_target(bboxes=anchor_list[i],gt_bboxes=target_boxes[i],stride=self.downsample_ratios[i]) for i in range(3)])
            reg_target[assigned_result<0] = 0
            #print('reg_target.shape:', reg_target[0].shape)
            scales_target = np.zeros((anchors.shape[0]),dtype=np.float32)
            scales_target[pos_idx] = scales[pos_idx]
            obj_target = np.zeros((anchors.shape[0]),dtype=np.float32)
            obj_target[pos_idx] = gt_scores[pos_idx] * assigned_iou[pos_idx]
            label_target = np.eye(self.num_classes)[gt_labels]
            label_target[assigned_result<0] = 0
            for i, (
                    mask, downsample_ratio
            ) in enumerate(zip(self.anchor_masks, self.downsample_ratios)):
                grid_h = int(h / downsample_ratio)
                grid_w = int(w / downsample_ratio)
                target = np.zeros(
                    (len(mask), 6 + self.num_classes, grid_h, grid_w),
                    dtype=np.float32)
                if i==0:
                    start = 0
                    end = grid_h * grid_w * 3
                elif i==1:
                    start = int(grid_h * grid_w * 3 / 4) 
                    end = int(grid_h * grid_w * 5 * 3 / 4)
                else:
                    start = - grid_h * grid_w * 3
                    end = anchors.shape[0]
                target[:,0:4,...] = reg_target[start:end].reshape((grid_w,grid_h,len(mask),4)).transpose(2,3,0,1)
                target[:,4,...] = scales_target[start:end].reshape((grid_w,grid_h,len(mask))).transpose(2,0,1)
                target[:,5,...] = obj_target[start:end].reshape((grid_w,grid_h,len(mask))).transpose(2,0,1)
                target[:,6:,...] = label_target[start:end].reshape((grid_w,grid_h,len(mask),80)).transpose(2,3,0,1)
                sample['target{}'.format(i)] = target
        return samples

@register_op
class Gt2FCOSTarget(BaseOperator):
    """
    Generate FCOS targets by groud truth data
    """

    def __init__(self,
                 object_sizes_boundary,
                 center_sampling_radius,
                 downsample_ratios,
                 norm_reg_targets=False):
        super(Gt2FCOSTarget, self).__init__()
        self.center_sampling_radius = center_sampling_radius
        self.downsample_ratios = downsample_ratios
        self.INF = np.inf
        self.object_sizes_boundary = [-1] + object_sizes_boundary + [self.INF]
        object_sizes_of_interest = []
        for i in range(len(self.object_sizes_boundary) - 1):
            object_sizes_of_interest.append([
                self.object_sizes_boundary[i], self.object_sizes_boundary[i + 1]
            ])
        self.object_sizes_of_interest = object_sizes_of_interest
        self.norm_reg_targets = norm_reg_targets

    def _compute_points(self, w, h):
        """
        compute the corresponding points in each feature map
        :param h: image height
        :param w: image width
        :return: points from all feature map
        """
        locations = []
        for stride in self.downsample_ratios:
            shift_x = np.arange(0, w, stride).astype(np.float32)
            shift_y = np.arange(0, h, stride).astype(np.float32)
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shift_x = shift_x.flatten()
            shift_y = shift_y.flatten()
            location = np.stack([shift_x, shift_y], axis=1) + stride // 2
            locations.append(location)
        num_points_each_level = [len(location) for location in locations]
        locations = np.concatenate(locations, axis=0)
        return locations, num_points_each_level

    def _convert_xywh2xyxy(self, gt_bbox, w, h):
        """
        convert the bounding box from style xywh to xyxy
        :param gt_bbox: bounding boxes normalized into [0, 1]
        :param w: image width
        :param h: image height
        :return: bounding boxes in xyxy style
        """
        bboxes = gt_bbox.copy()
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * w
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * h
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        return bboxes

    def _check_inside_boxes_limited(self, gt_bbox, xs, ys,
                                    num_points_each_level):
        """
        check if points is within the clipped boxes
        :param gt_bbox: bounding boxes
        :param xs: horizontal coordinate of points
        :param ys: vertical coordinate of points
        :return: the mask of points is within gt_box or not
        """
        bboxes = np.reshape(
            gt_bbox, newshape=[1, gt_bbox.shape[0], gt_bbox.shape[1]])
        bboxes = np.tile(bboxes, reps=[xs.shape[0], 1, 1])
        ct_x = (bboxes[:, :, 0] + bboxes[:, :, 2]) / 2
        ct_y = (bboxes[:, :, 1] + bboxes[:, :, 3]) / 2
        beg = 0
        clipped_box = bboxes.copy()
        for lvl, stride in enumerate(self.downsample_ratios):
            end = beg + num_points_each_level[lvl]
            stride_exp = self.center_sampling_radius * stride
            clipped_box[beg:end, :, 0] = np.maximum(
                bboxes[beg:end, :, 0], ct_x[beg:end, :] - stride_exp)
            clipped_box[beg:end, :, 1] = np.maximum(
                bboxes[beg:end, :, 1], ct_y[beg:end, :] - stride_exp)
            clipped_box[beg:end, :, 2] = np.minimum(
                bboxes[beg:end, :, 2], ct_x[beg:end, :] + stride_exp)
            clipped_box[beg:end, :, 3] = np.minimum(
                bboxes[beg:end, :, 3], ct_y[beg:end, :] + stride_exp)
            beg = end
        l_res = xs - clipped_box[:, :, 0]
        r_res = clipped_box[:, :, 2] - xs
        t_res = ys - clipped_box[:, :, 1]
        b_res = clipped_box[:, :, 3] - ys
        clipped_box_reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)
        inside_gt_box = np.min(clipped_box_reg_targets, axis=2) > 0
        return inside_gt_box

    def __call__(self, samples, context=None):
        assert len(self.object_sizes_of_interest) == len(self.downsample_ratios), \
            "object_sizes_of_interest', and 'downsample_ratios' should have same length."

        for sample in samples:
            # im, gt_bbox, gt_class, gt_score = sample
            im = sample['image']
            im_info = sample['im_info']
            bboxes = sample['gt_bbox']
            gt_class = sample['gt_class']
            gt_score = sample['gt_score']
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * np.floor(im_info[1]) / \
                np.floor(im_info[1] / im_info[2])
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * np.floor(im_info[0]) / \
                np.floor(im_info[0] / im_info[2])
            # calculate the locations
            h, w = sample['image'].shape[1:3]
            points, num_points_each_level = self._compute_points(w, h)
            object_scale_exp = []
            for i, num_pts in enumerate(num_points_each_level):
                object_scale_exp.append(
                    np.tile(
                        np.array([self.object_sizes_of_interest[i]]),
                        reps=[num_pts, 1]))
            object_scale_exp = np.concatenate(object_scale_exp, axis=0)

            gt_area = (bboxes[:, 2] - bboxes[:, 0]) * (
                bboxes[:, 3] - bboxes[:, 1])
            xs, ys = points[:, 0], points[:, 1]
            xs = np.reshape(xs, newshape=[xs.shape[0], 1])
            xs = np.tile(xs, reps=[1, bboxes.shape[0]])
            ys = np.reshape(ys, newshape=[ys.shape[0], 1])
            ys = np.tile(ys, reps=[1, bboxes.shape[0]])

            l_res = xs - bboxes[:, 0]
            r_res = bboxes[:, 2] - xs
            t_res = ys - bboxes[:, 1]
            b_res = bboxes[:, 3] - ys
            reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)
            if self.center_sampling_radius > 0:
                is_inside_box = self._check_inside_boxes_limited(
                    bboxes, xs, ys, num_points_each_level)
            else:
                is_inside_box = np.min(reg_targets, axis=2) > 0
            # check if the targets is inside the corresponding level
            max_reg_targets = np.max(reg_targets, axis=2)
            lower_bound = np.tile(
                np.expand_dims(
                    object_scale_exp[:, 0], axis=1),
                reps=[1, max_reg_targets.shape[1]])
            high_bound = np.tile(
                np.expand_dims(
                    object_scale_exp[:, 1], axis=1),
                reps=[1, max_reg_targets.shape[1]])
            is_match_current_level = \
                (max_reg_targets > lower_bound) & \
                (max_reg_targets < high_bound)
            points2gtarea = np.tile(
                np.expand_dims(
                    gt_area, axis=0), reps=[xs.shape[0], 1])
            points2gtarea[is_inside_box == 0] = self.INF
            points2gtarea[is_match_current_level == 0] = self.INF
            points2min_area = points2gtarea.min(axis=1)
            points2min_area_ind = points2gtarea.argmin(axis=1)
            labels = gt_class[points2min_area_ind] + 1
            labels[points2min_area == self.INF] = 0
            reg_targets = reg_targets[range(xs.shape[0]), points2min_area_ind]
            ctn_targets = np.sqrt((reg_targets[:, [0, 2]].min(axis=1) / \
                                  reg_targets[:, [0, 2]].max(axis=1)) * \
                                  (reg_targets[:, [1, 3]].min(axis=1) / \
                                   reg_targets[:, [1, 3]].max(axis=1))).astype(np.float32)
            ctn_targets = np.reshape(
                ctn_targets, newshape=[ctn_targets.shape[0], 1])
            ctn_targets[labels <= 0] = 0
            pos_ind = np.nonzero(labels != 0)
            reg_targets_pos = reg_targets[pos_ind[0], :]
            split_sections = []
            beg = 0
            for lvl in range(len(num_points_each_level)):
                end = beg + num_points_each_level[lvl]
                split_sections.append(end)
                beg = end
            labels_by_level = np.split(labels, split_sections, axis=0)
            reg_targets_by_level = np.split(reg_targets, split_sections, axis=0)
            ctn_targets_by_level = np.split(ctn_targets, split_sections, axis=0)
            for lvl in range(len(self.downsample_ratios)):
                grid_w = int(np.ceil(w / self.downsample_ratios[lvl]))
                grid_h = int(np.ceil(h / self.downsample_ratios[lvl]))
                if self.norm_reg_targets:
                    sample['reg_target{}'.format(lvl)] = \
                        np.reshape(
                            reg_targets_by_level[lvl] / \
                            self.downsample_ratios[lvl],
                            newshape=[grid_h, grid_w, 4])
                else:
                    sample['reg_target{}'.format(lvl)] = np.reshape(
                        reg_targets_by_level[lvl],
                        newshape=[grid_h, grid_w, 4])
                sample['labels{}'.format(lvl)] = np.reshape(
                    labels_by_level[lvl], newshape=[grid_h, grid_w, 1])
                sample['centerness{}'.format(lvl)] = np.reshape(
                    ctn_targets_by_level[lvl], newshape=[grid_h, grid_w, 1])
        return samples


@register_op
class Gt2TTFTarget(BaseOperator):
    """
    Gt2TTFTarget
    Generate TTFNet targets by ground truth data
    
    Args:
        num_classes(int): the number of classes.
        down_ratio(int): the down ratio from images to heatmap, 4 by default.
        alpha(float): the alpha parameter to generate gaussian target.
            0.54 by default.
    """

    def __init__(self, num_classes, down_ratio=4, alpha=0.54):
        super(Gt2TTFTarget, self).__init__()
        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.alpha = alpha

    def __call__(self, samples, context=None):
        output_size = samples[0]['image'].shape[1]
        feat_size = output_size // self.down_ratio
        for sample in samples:
            heatmap = np.zeros(
                (self.num_classes, feat_size, feat_size), dtype='float32')
            box_target = np.ones(
                (4, feat_size, feat_size), dtype='float32') * -1
            reg_weight = np.zeros((1, feat_size, feat_size), dtype='float32')

            gt_bbox = sample['gt_bbox']
            gt_class = sample['gt_class']

            bbox_w = gt_bbox[:, 2] - gt_bbox[:, 0] + 1
            bbox_h = gt_bbox[:, 3] - gt_bbox[:, 1] + 1
            area = bbox_w * bbox_h
            boxes_areas_log = np.log(area)
            boxes_ind = np.argsort(boxes_areas_log, axis=0)[::-1]
            boxes_area_topk_log = boxes_areas_log[boxes_ind]
            gt_bbox = gt_bbox[boxes_ind]
            gt_class = gt_class[boxes_ind]

            feat_gt_bbox = gt_bbox / self.down_ratio
            feat_gt_bbox = np.clip(feat_gt_bbox, 0, feat_size - 1)
            feat_hs, feat_ws = (feat_gt_bbox[:, 3] - feat_gt_bbox[:, 1],
                                feat_gt_bbox[:, 2] - feat_gt_bbox[:, 0])

            ct_inds = np.stack(
                [(gt_bbox[:, 0] + gt_bbox[:, 2]) / 2,
                 (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2],
                axis=1) / self.down_ratio

            h_radiuses_alpha = (feat_hs / 2. * self.alpha).astype('int32')
            w_radiuses_alpha = (feat_ws / 2. * self.alpha).astype('int32')

            for k in range(len(gt_bbox)):
                cls_id = gt_class[k]
                fake_heatmap = np.zeros((feat_size, feat_size), dtype='float32')
                self.draw_truncate_gaussian(fake_heatmap, ct_inds[k],
                                            h_radiuses_alpha[k],
                                            w_radiuses_alpha[k])

                heatmap[cls_id] = np.maximum(heatmap[cls_id], fake_heatmap)
                box_target_inds = fake_heatmap > 0
                box_target[:, box_target_inds] = gt_bbox[k][:, None]

                local_heatmap = fake_heatmap[box_target_inds]
                ct_div = np.sum(local_heatmap)
                local_heatmap *= boxes_area_topk_log[k]
                reg_weight[0, box_target_inds] = local_heatmap / ct_div
            sample['ttf_heatmap'] = heatmap
            sample['ttf_box_target'] = box_target
            sample['ttf_reg_weight'] = reg_weight
        return samples

    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = gaussian2D((h, w), sigma_x, sigma_y)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom, w_radius -
                                   left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            heatmap[y - top:y + bottom, x - left:x + right] = np.maximum(
                masked_heatmap, masked_gaussian)
        return heatmap

class YOLOAnchorGenerator(object):
    
    def __init__(self, strides, base_sizes):
        self.strides = [(stride, stride) for stride in strides]
        self.centers = [(stride[0] / 2., stride[1] / 2.)
                        for stride in self.strides]
        self.base_sizes = []
        num_anchor_per_level = len(base_sizes[0])
        for base_sizes_per_level in base_sizes:
            assert num_anchor_per_level == len(base_sizes_per_level)
            self.base_sizes.append(
                [base_size for base_size in base_sizes_per_level])
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_levels(self):
        return len(self.base_sizes)

    def gen_base_anchors(self):
        multi_level_base_anchors = []
        for i, base_sizes_per_level in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(base_sizes_per_level,
                                                   center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self, base_sizes_per_level, center=None):
        x_center, y_center = center
        base_anchors = []
        for base_size in base_sizes_per_level:
            w, h = base_size
            # use float anchor and the anchor's center is aligned with the
            # pixel center
            base_anchor = np.array([
                x_center - 0.5 * w, y_center - 0.5 * h, x_center + 0.5 * w,
                y_center + 0.5 * h
            ])
            base_anchors.append(base_anchor)
        base_anchors = np.stack(base_anchors, axis=0)

        return base_anchors
    
    def _meshgrid(self, x, y, row_major=True):
        xx = np.tile(x,len(y))
        yy = y.reshape(-1,1).repeat(len(x),1).reshape(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx
        
    def grid_anchors(self, featmap_sizes):
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(
                self.base_anchors[i],
                featmap_sizes[i],
                self.strides[i])
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_anchors(self,
                                  base_anchors,
                                  featmap_size,
                                  stride=(16, 16)):
        feat_h, feat_w = featmap_size
        # convert Tensor to int, so that we can covert to ONNX correctlly
        feat_h = int(feat_h)
        feat_w = int(feat_w)
        shift_x = np.arange(0, feat_w) * stride[0]
        shift_y = np.arange(0, feat_h) * stride[1]

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = np.stack([shift_xx, shift_yy, shift_xx, shift_yy], axis=-1)
        shifts = shifts.astype(base_anchors.dtype)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.reshape(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

class ATSS_Assigner(object):
    def __init__(self,
                 anchors,
                 gts,
                 num_level_bboxes,
                 topk):
        super(ATSS_Assigner, self).__init__()
        self.anchors = anchors
        self.gts = gts
        self.num_level_bboxes = num_level_bboxes
        self.topk = topk
    
    def assign(self):
        distances = self.center_distance(self.gts, self.anchors)
        overlaps = self.overlap_matrix(self.gts, self.anchors)
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(self.num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[:, start_idx:end_idx]
            selectable_k = min(self.topk, bboxes_per_level)
            topk_idxs_per_level = np.argpartition(distances_per_level, selectable_k, axis=1)[:, 0:selectable_k]
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = np.concatenate(candidate_idxs, axis=1)
        #print(distances.shape)
        #print(candidate_idxs.shape)
        candidate_overlaps = np.zeros(candidate_idxs.shape)
        for i in range(overlaps.shape[0]):
            candidate_overlaps[i] = overlaps[i][candidate_idxs[i]]
        #candidate_overlaps = overlaps[candidate_idxs, np.arange(len(self.gts))]
        overlaps_mean_per_gt = np.mean(candidate_overlaps,axis=0)
        overlaps_std_per_gt = np.std(candidate_overlaps,axis=0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt
        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]
        
        overlaps_inf = np.zeros(overlaps.shape).T.reshape(-1)-500
        index = candidate_idxs.reshape(-1)[is_pos.reshape(-1)]
        overlaps_inf[index] = overlaps.T.reshape(-1)[index]
        overlaps_inf = overlaps_inf.reshape(len(self.gts), -1).T
        num_anchors = self.anchors.shape[0]
        assigned_gt_inds = np.zeros((num_anchors), dtype=np.int) - 1
        max_overlaps = overlaps_inf.max(axis=1)
        argmax_overlaps = overlaps_inf.argmax(axis=1)
        assigned_gt_inds[max_overlaps != -500] = argmax_overlaps[max_overlaps != -500] 
        return assigned_gt_inds

    def overlap_matrix(self, bbox, gt):
        lt = np.maximum(bbox[:, None, :2], gt[:, :2])  # left_top (x, y)
        rb = np.minimum(bbox[:, None, 2:], gt[:, 2:])  # right_bottom (x, y)
        wh = np.maximum(rb - lt + 1, 0)                # inter_area (w, h)
        inter_areas = wh[:, :, 0] * wh[:, :, 1]        # shape: (n, m)
        box_areas = (bbox[:, 2] - bbox[:, 0] + 1) * (bbox[:, 3] - bbox[:, 1] + 1)
        gt_areas = (gt[:, 2] - gt[:, 0] + 1) * (gt[:, 3] - gt[:, 1] + 1)
        unioun_areas = box_areas[:, None] + gt_areas - inter_areas
        IoU = np.divide(inter_areas, unioun_areas, out=np.zeros_like(inter_areas), where=unioun_areas!=0) 
        return IoU
    
    def center_distance(self, bbox, gt):
        center_x1 = (bbox[:, 2:3] + bbox[:, 0:1]) / 2 
        center_y1 = (bbox[:, 3:4] + bbox[:, 1:2]) / 2 
        center_x2 = (gt[:, 2:3] + gt[:, 0:1]) / 2
        center_y2 = (gt[:, 3:4] + gt[:, 1:2]) / 2
        center_distance = np.power(center_x1-center_x2.T,2)+np.power(center_y1-center_y2.T,2)
        return center_distance

class TopK_Assigner(object):
    def __init__(self,
                 anchors,
                 gts,
                 k):
        super(TopK_Assigner, self).__init__()
        self.anchors = anchors
        self.gts = gts
        self.k = k
    
    def assign(self):
        #print("#######################")
        #print("Topk_Assigner")
        num_anchors = self.anchors.shape[0]
        assigned_gt_inds = np.zeros((num_anchors), dtype=np.int) - 1
        assigned_iou = np.zeros((num_anchors), dtype=np.float)
        overlaps = self.diou_matrix(self.gts, self.anchors)
        overlaps_mask = np.zeros(overlaps.shape, dtype=np.float32)
        indices = np.argpartition(-overlaps, self.k, axis=1)[:, 0:self.k]
        overlaps_mask[np.repeat(np.arange(overlaps.shape[0]), self.k), indices.ravel()] = 1
        new_overlaps = overlaps * overlaps_mask
        max_overlaps = new_overlaps.max(axis=0)
        argmax_overlaps = new_overlaps.argmax(axis=0)
        #gt_argmax_overlaps = overlaps.argmax(axis=1)
        pos_inds = max_overlaps > 0.
        norm_ious = np.zeros(num_anchors) 
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds]
        gt_max_overlap = overlaps.max(axis=1)
        overlaps = np.multiply(overlaps.T,1/gt_max_overlap).T
        for i in range(len(argmax_overlaps)):                                                                                                               
            norm_ious[i] = overlaps[:,i][argmax_overlaps[i]]
        assigned_iou[pos_inds] = norm_ious[pos_inds]
        return assigned_gt_inds, assigned_iou

    def overlap_matrix(self, bbox, gt):
        lt = np.maximum(bbox[:, None, :2], gt[:, :2])  # left_top (x, y)
        rb = np.minimum(bbox[:, None, 2:], gt[:, 2:])  # right_bottom (x, y)
        wh = np.maximum(rb - lt + 1, 0)                # inter_area (w, h)
        inter_areas = wh[:, :, 0] * wh[:, :, 1]        # shape: (n, m)
        box_areas = (bbox[:, 2] - bbox[:, 0] + 1) * (bbox[:, 3] - bbox[:, 1] + 1)
        gt_areas = (gt[:, 2] - gt[:, 0] + 1) * (gt[:, 3] - gt[:, 1] + 1)
        unioun_areas = box_areas[:, None] + gt_areas - inter_areas
        IoU = np.divide(inter_areas, unioun_areas, out=np.zeros_like(inter_areas), where=unioun_areas!=0) 

        center_x1 = (bbox[:, 2:3] + bbox[:, 0:1]) / 2 
        center_y1 = (bbox[:, 3:4] + bbox[:, 1:2]) / 2 
        center_x2 = (gt[:, 2:3] + gt[:, 0:1]) / 2
        center_y2 = (gt[:, 3:4] + gt[:, 1:2]) / 2
        center_distance = abs(center_x1-center_x2.T)+abs(center_y1-center_y2.T)+5
        close = 1 / center_distance
        return IoU+close

    def diou_matrix(self, bbox, gt):
        lt = np.maximum(bbox[:, None, :2], gt[:, :2])  # left_top (x, y)
        rb = np.minimum(bbox[:, None, 2:], gt[:, 2:])  # right_bottom (x, y)
        wh = np.maximum(rb - lt + 1, 0)                # inter_area (w, h)
        inter_areas = wh[:, :, 0] * wh[:, :, 1]        # shape: (n, m)
        box_areas = (bbox[:, 2] - bbox[:, 0] + 1) * (bbox[:, 3] - bbox[:, 1] + 1)
        gt_areas = (gt[:, 2] - gt[:, 0] + 1) * (gt[:, 3] - gt[:, 1] + 1)
        unioun_areas = box_areas[:, None] + gt_areas - inter_areas
        IoU = np.divide(inter_areas, unioun_areas, out=np.zeros_like(inter_areas), where=unioun_areas!=0) 

        center_x1 = (bbox[:, 2:3] + bbox[:, 0:1]) / 2 
        center_y1 = (bbox[:, 3:4] + bbox[:, 1:2]) / 2 
        center_x2 = (gt[:, 2:3] + gt[:, 0:1]) / 2
        center_y2 = (gt[:, 3:4] + gt[:, 1:2]) / 2
        
        lt = np.minimum(bbox[:, None, :2], gt[:, :2])  # left_top (x, y)
        rb = np.maximum(bbox[:, None, 2:], gt[:, 2:])  # right_bottom (x, y)
        center_distance = np.power(center_x1-center_x2.T,2)+np.power(center_y1-center_y2.T,2)
        ltrb_distance = np.power(lt[:,0]-rb[:,0],2)+np.power(lt[:,1]-rb[:,1],2)
        diou_term =  center_distance / ltrb_distance
        #print(diou_term)
        dIoU = (IoU - diou_term).clip(-1,1)
        return dIoU
