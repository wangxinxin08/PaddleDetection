# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register

from ..utils import decode_yolo, xywh2xyxy, iou_similarity
import os
import numpy as np

__all__ = ['YOLOv3Loss']


@register
class YOLOv3Loss(nn.Layer):

    __inject__ = ['iou_loss', 'iou_aware_loss']
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 ignore_thresh=0.7,
                 label_smooth=False,
                 downsample=[32, 16, 8],
                 scale_x_y=1.,
                 iou_loss=None,
                 iou_aware_loss=None):
        super(YOLOv3Loss, self).__init__()
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.label_smooth = label_smooth
        self.downsample = downsample
        self.scale_x_y = scale_x_y
        self.iou_loss = iou_loss
        self.iou_aware_loss = iou_aware_loss

    def obj_loss(self, pbox, gbox, pobj, tobj, anchor, downsample):
        b, h, w, na = pbox.shape[:4]
        pbox = decode_yolo(pbox, anchor, downsample)
        pbox = pbox.reshape((b, -1, 4))
        pbox = xywh2xyxy(pbox)
        gbox = xywh2xyxy(gbox)

        iou = iou_similarity(pbox, gbox)
        iou.stop_gradient = True
        iou_max = iou.max(2)  # [N, M1]
        iou_mask = paddle.cast(iou_max <= self.ignore_thresh, dtype=pbox.dtype)
        iou_mask.stop_gradient = True

        pobj = pobj.reshape((b, -1))
        tobj = tobj.reshape((b, -1))
        obj_mask = paddle.cast(tobj > 0, dtype=pbox.dtype)
        obj_mask.stop_gradient = True

        loss_obj = F.binary_cross_entropy_with_logits(
            pobj, obj_mask, reduction='none')
        loss_obj_pos = (loss_obj * tobj)
        loss_obj_neg = (loss_obj * (1 - obj_mask) * iou_mask)
        return loss_obj_pos + loss_obj_neg

    def cls_loss(self, pcls, tcls):
        if self.label_smooth:
            delta = min(1. / self.num_classes, 1. / 40)
            pos, neg = 1 - delta, delta
            # 1 for positive, 0 for negative
            tcls = pos * paddle.cast(
                tcls > 0., dtype=tcls.dtype) + neg * paddle.cast(
                    tcls <= 0., dtype=tcls.dtype)

        loss_cls = F.binary_cross_entropy_with_logits(
            pcls, tcls, reduction='none')
        return loss_cls

    def yolov3_loss(self,
                    x,
                    t,
                    gt_box,
                    anchor,
                    downsample,
                    i,
                    scale=1.,
                    eps=1e-10):
        na = len(anchor)
        b, c, h, w = x.shape
        if self.iou_aware_loss:
            ioup, x = x[:, 0:na, :, :], x[:, na:, :, :]
        x = x.reshape((b, na, -1, h, w)).transpose((0, 3, 4, 1, 2))
        xy, wh = x[:, :, :, :, 0:2], x[:, :, :, :, 2:4]
        obj, pcls = x[:, :, :, :, 4:5], x[:, :, :, :, 5:]

        t = t.transpose((0, 3, 4, 1, 2))
        txy, twh, tscale = t[:, :, :, :, 0:2], t[:, :, :, :, 2:4], t[:, :, :, :,
                                                                     4:5]
        tobj, tcls = t[:, :, :, :, 5:6], t[:, :, :, :, 6:]

        tscale_obj = tscale * tobj
        loss = dict()
        if abs(scale - 1.) < eps:
            loss_xy = tscale_obj * F.binary_cross_entropy_with_logits(
                xy, txy, reduction='none')
        else:
            xy = scale * F.sigmoid(xy) - 0.5 * (scale - 1.)
            loss_xy = tscale_obj * paddle.abs(xy - txy)

        loss_xy = loss_xy.sum([1, 2, 3, 4]).mean()
        loss_wh = tscale_obj * paddle.abs(wh - twh)
        loss_wh = loss_wh.sum([1, 2, 3, 4]).mean()

        loss['loss_xy'] = loss_xy
        loss['loss_wh'] = loss_wh
        x[:, :, :, :, 0:2] = scale * F.sigmoid(x[:, :, :, :, 0:2]) - 0.5 * (
            scale - 1.)
        if self.iou_loss is not None:
            box, tbox = x[:, :, :, :, 0:4], t[:, :, :, :, 0:4]
            # self.loss_ious.append(box)
            # print('loss_iou box.stop_gradient:', box.stop_gradient)
            loss_iou, pbox = self.iou_loss(box, tbox, anchor, downsample, i,
                                           self.scale_x_y)
            self.loss_ious.append(pbox)
            # print('loss_iou loss_iou.stop_gradient:', loss_iou.stop_gradient)
            loss_iou = loss_iou * tscale_obj.reshape((b, -1))
            # self.loss_ious.append(loss_iou)
            loss_iou = loss_iou.sum(-1).mean()
            path = os.path.join('grad', 'loss_iou_{}.npy'.format(i))
            np.save(path, loss_iou.numpy())
            # self.loss_ious.append(loss_iou)
            loss['loss_iou'] = loss_iou

        # if self.iou_aware_loss is not None:
        #     box, tbox = x[:, :, :, :, 0:4], t[:, :, :, :, 0:4]
        #     loss_iou_aware = self.iou_aware_loss(ioup, box, tbox, anchor,
        #                                          downsample)
        #     loss_iou_aware = loss_iou_aware * tobj.squeeze(-1).transpose(
        #         (0, 3, 1, 2))
        #     loss_iou_aware = loss_iou_aware.sum([1, 2, 3]).mean()
        #     loss['loss_iou_aware'] = loss_iou_aware

        # box = x[:, :, :, :, 0:4]
        # loss_obj = self.obj_loss(box, gt_box, obj, tobj, anchor, downsample)
        # loss_obj = loss_obj.sum(-1).mean()
        # loss['loss_obj'] = loss_obj
        # loss_cls = self.cls_loss(pcls, tcls) * tobj
        # loss_cls = loss_cls.sum([1, 2, 3, 4]).mean()
        # loss['loss_cls'] = loss_cls
        return loss

    def forward(self, inputs, targets, anchors):
        self.loss_ious = []
        np = len(inputs)
        gt_targets = [targets['target{}'.format(i)] for i in range(np)]
        gt_box = targets['gt_bbox']
        yolo_losses = dict()
        for i, (
                x, t, anchor, downsample
        ) in enumerate(zip(inputs, gt_targets, anchors, self.downsample)):
            yolo_loss = self.yolov3_loss(x, t, gt_box, anchor, downsample, i,
                                         self.scale_x_y)
            for k, v in yolo_loss.items():
                if k in yolo_losses:
                    yolo_losses[k] += v
                else:
                    yolo_losses[k] = v

        loss = 0
        for k, v in yolo_losses.items():
            loss += v

        yolo_losses['loss'] = loss
        return yolo_losses
