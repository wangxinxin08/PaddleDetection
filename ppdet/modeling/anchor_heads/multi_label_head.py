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

import numpy as np
import math

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

from ppdet.modeling.ops import MultiClassNMS, MultiClassSoftNMS, MatrixNMS
from ppdet.modeling.losses.yolo_loss import YOLOv3Loss
from ppdet.core.workspace import register
from ppdet.modeling.ops import DropBlock
from .iou_aware import get_iou_aware_score
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
from ppdet.utils.check import check_version

@register
class MultiLabelHead(object):
    def __init__(self,
                 num_classes=80,
                 second_head=False):
        check_version("1.8.4")
        self.num_classes = num_classes
        self.second_head = second_head
    def _get_outputs(self, input, is_train=True):
        """
        Get YOLOv3 head output

        Args:
            input (list): List of Variables, output of backbone stages
            is_train (bool): whether in train or test mode

        Returns:
            outputs (list): Variables of each output layer
        """
        out_layer_num = 3
        blocks = input[-1:-out_layer_num - 1:-1]
        pool = fluid.layers.pool2d(input=blocks[0], pool_size=19, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        fc_param_attr = fluid.param_attr.ParamAttr(
            initializer=fluid.initializer.Uniform(-stdv, stdv), trainable=True)
        outputs = fluid.layers.fc(
            input=pool, size=self.num_classes, act=None, param_attr=fc_param_attr)
        return outputs

    def get_loss(self, loss, body_feats, multi_label_target):
        outputs = self._get_outputs(body_feats, is_train=True)
        losses = fluid.layers.sigmoid_cross_entropy_with_logits(outputs, multi_label_target)
        loss['loss_multilLabel'] = fluid.layers.reduce_sum(losses)
        return loss

