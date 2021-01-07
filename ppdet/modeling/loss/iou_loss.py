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
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ..utils import xywh2xyxy, bbox_iou, decode_yolo
import numpy as np
import os

__all__ = ['IouLoss']


@register
@serializable
class IouLoss(object):
    """
    iou loss, see https://arxiv.org/abs/1908.03851
    loss = 1.0 - iou * iou
    Args:
        loss_weight (float): iou loss weight, default is 2.5
        max_height (int): max height of input to support random shape input
        max_width (int): max width of input to support random shape input
        ciou_term (bool): whether to add ciou_term
        loss_square (bool): whether to square the iou term
    """

    def __init__(self,
                 loss_weight=2.5,
                 giou=False,
                 diou=False,
                 ciou=False,
                 loss_square=True):
        self.loss_weight = loss_weight
        self.giou = giou
        self.diou = diou
        self.ciou = ciou
        self.loss_square = loss_square

    def __call__(self, pbox, gbox, anchor, downsample, i, scale_x_y):
        pbox = self.bbox_transfrom(
            pbox, anchor, downsample, scale_x_y, is_gt=False)
        gbox = self.bbox_transfrom(gbox, anchor, downsample)
        print('loss_iou pbox.stop_gradient:', pbox.stop_gradient)
        print('loss_iou gbox.stop_gradient:', gbox.stop_gradient)
        iou = bbox_iou(
            pbox, gbox, giou=self.giou, diou=self.diou, ciou=self.ciou)
        print('loss_iou iou.stop_gradient:', iou.stop_gradient)
        # iou = self._iou(pbox, gbox, anchor, downsample)
        path = os.path.join('grad', 'iou_{}.npy'.format(i))
        np.save(path, iou.numpy())
        if self.loss_square:
            loss_iou = 1 - iou * iou
        else:
            loss_iou = 1 - iou

        loss_iou = loss_iou * self.loss_weight
        return loss_iou, pbox

    def bbox_transfrom(self, box, anchor, downsample, scale_x_y=1.0,
                       is_gt=True):
        b = box.shape[0]
        box = decode_yolo(box, anchor, downsample, scale_x_y, is_gt=is_gt)
        box = box.reshape((b, -1, 4))
        box = xywh2xyxy(box)
        return box

    # def _iou(self, pbox, gbox, anchor, downsample):
    #     b = pbox.shape[0]
    #     pbox = decode_yolo(pbox, anchor, downsample)
    #     pbox = pbox.reshape((b, -1, 4))
    #     gbox = decode_yolo(gbox, anchor, downsample)
    #     gbox = gbox.reshape((b, -1, 4))
    #     pbox = xywh2xyxy(pbox)
    #     gbox = xywh2xyxy(gbox)
    #     print('pbox.stop_gradient:', pbox.stop_gradient)
    #     print('gbox.stop_gradient:', gbox.stop_gradient)
    #     iou = bbox_iou(
    #         pbox, gbox, giou=self.giou, diou=self.diou, ciou=self.ciou)
    #     return iou
