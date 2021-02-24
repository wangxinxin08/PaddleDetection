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
import numpy as np
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import NumpyArrayInitializer

from paddle import fluid
from ppdet.core.workspace import register, serializable
from .iou_loss import IouLoss

__all__ = ['IouAwareLoss']


@register
@serializable
class IouAwareLoss(IouLoss):
    """
    iou aware loss, see https://arxiv.org/abs/1912.05992
    Args:
        loss_weight (float): iou aware loss weight, default is 1.0
        max_height (int): max height of input to support random shape input
        max_width (int): max width of input to support random shape input
    """

    def __init__(self,
                 loss_weight=1.0,
                 max_height=608,
                 max_width=608,
                 ciou_term=False,
                 eiou_term=False):
        super(IouAwareLoss, self).__init__(
            loss_weight=loss_weight,
            max_height=max_height,
            max_width=max_width,
            ciou_term=ciou_term,
            eiou_term=eiou_term)

    def __call__(self,
                 ioup,
                 x,
                 y,
                 w,
                 h,
                 tx,
                 ty,
                 tw,
                 th,
                 anchors,
                 downsample_ratio,
                 batch_size,
                 scale_x_y,
                 tobj,
                 obj_mask,
                 eps=1.e-10):
        '''
        Args:
            ioup ([Variables]): the predicted iou
            x  | y | w | h  ([Variables]): the output of yolov3 for encoded x|y|w|h
            tx |ty |tw |th  ([Variables]): the target of yolov3 for encoded x|y|w|h
            anchors ([float]): list of anchors for current output layer
            downsample_ratio (float): the downsample ratio for current output layer
            batch_size (int): training batch size
            eps (float): the decimal to prevent the denominator eqaul zero
        '''

        pred = self._bbox_transform(x, y, w, h, anchors, downsample_ratio,
                                    batch_size, False, scale_x_y, eps)
        gt = self._bbox_transform(tx, ty, tw, th, anchors, downsample_ratio,
                                  batch_size, True, scale_x_y, eps)
        iouk = self._iou(pred, gt, ioup, eps)
        iouk = iouk * obj_mask
        iouk = fluid.layers.reshape(iouk, [batch_size, -1])
        iouk.stop_gradient = True
        weight = tobj + (1 - obj_mask)
        weight = fluid.layers.reshape(weight, [batch_size, -1])
        weight.stop_gradient = True
        ioup = fluid.layers.reshape(ioup, [batch_size, -1])
        loss_iou_aware = weight * fluid.layers.sigmoid_cross_entropy_with_logits(
            ioup, iouk)
        loss_iou_aware = loss_iou_aware * self._loss_weight
        return loss_iou_aware
