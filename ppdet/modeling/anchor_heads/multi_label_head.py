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

    def _conv_bn(self,
                 input,
                 ch_out,
                 filter_size,
                 stride,
                 padding,
                 act='leaky',
                 name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            param_attr=ParamAttr(name=name + ".conv.weights"),
            bias_attr=False)

        bn_name = name + ".bn"
        bn_param_attr = ParamAttr(
            regularizer=L2Decay(0.), name=bn_name + '.scale')
        bn_bias_attr = ParamAttr(
            regularizer=L2Decay(0.), name=bn_name + '.offset')
        out = fluid.layers.batch_norm(
            input=conv,
            act=None,
            param_attr=bn_param_attr,
            bias_attr=bn_bias_attr,
            moving_mean_name=bn_name + '.mean',
            moving_variance_name=bn_name + '.var')

        if act == 'leaky':
            out = fluid.layers.leaky_relu(x=out, alpha=0.1)
        return out

    def _get_outputs(self, input, is_train=True):
        """
        Get YOLOv3 head output

        Args:
            input (list): List of Variables, output of backbone stages
            is_train (bool): whether in train or test mode

        Returns:
            outputs (list): Variables of each output layer
        """
        #out_layer_num = 3
        #blocks = input[-1:-out_layer_num - 1:-1]
        #pool = fluid.layers.pool2d(input=blocks[0], pool_size=19, pool_type='avg', global_pooling=True)
        outputs = []
        for i, fpn_feature in enumerate(input):
            multiLabel_conv = self._conv_bn(
                fpn_feature,
                fpn_feature.shape[1]*2,
                filter_size=3,
                stride=1,
                padding=1,
                name='multiLabel{}'.format(i))
            pool = fluid.layers.pool2d(input=multiLabel_conv, pool_size=19, pool_type='avg', global_pooling=True)
            stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
            fc_param_attr = fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv), trainable=True)
            output = fluid.layers.fc(
                input=pool, size=self.num_classes, act=None, param_attr=fc_param_attr)
            outputs.append(output)
        return outputs

    def get_loss(self, loss, body_feats, multi_label_target):
        outputs = self._get_outputs(body_feats, is_train=True)
        if len(outputs) == 1:
            losses = fluid.layers.sigmoid_cross_entropy_with_logits(outputs[0], multi_label_target)
            loss['loss_multiLabel'] = fluid.layers.reduce_sum(losses) / fluid.layers.reduce_sum(multi_label_target)
        else:
            for i, output in enumerate(outputs):
                losses = fluid.layers.sigmoid_cross_entropy_with_logits(output, multi_label_target)
                loss['loss_multiLabel{}'.format(i)] = fluid.layers.reduce_sum(losses) / fluid.layers.reduce_sum(multi_label_target)
        return loss

