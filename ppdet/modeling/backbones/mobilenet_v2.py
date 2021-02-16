# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import MSRAInitializer

from ppdet.experimental import mixed_precision_global_state
from ppdet.core.workspace import register

__all__ = ['MobileNetV2']


def conv_bn(input,
            ch_out,
            filter_size,
            stride,
            padding,
            groups=1,
            act='relu',
            name=''):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=ch_out,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        groups=groups,
        act=None,
        param_attr=ParamAttr(
            initializer=MSRAInitializer(uniform=False),
            name=name + ".conv.weights"),
        bias_attr=False)

    bn_name = name + ".bn"
    bn_param_attr = ParamAttr(name=bn_name + '.scale')
    bn_bias_attr = ParamAttr(name=bn_name + '.offset')

    out = fluid.layers.batch_norm(
        input=conv,
        act=None,
        param_attr=bn_param_attr,
        bias_attr=bn_bias_attr,
        moving_mean_name=bn_name + '.mean',
        moving_variance_name=bn_name + '.var')

    if act == 'leaky':
        out = fluid.layers.leaky_relu(x=out, alpha=0.1)
    elif act == 'relu':
        out = fluid.layers.relu(out)

    return out


def ibottleneck(input,
                ch_outs,
                filter_sizes,
                strides,
                shortcut=True,
                act='relu',
                name=''):
    conv1 = conv_bn(
        input=input,
        ch_out=ch_outs[0],
        filter_size=filter_sizes[0],
        stride=strides[0],
        padding=filter_sizes[0] // 2,
        groups=1,
        act=act,
        name=name + '.expand')
    conv2 = conv_bn(
        input=conv1,
        ch_out=ch_outs[1],
        filter_size=filter_sizes[1],
        stride=strides[1],
        padding=filter_sizes[1] // 2,
        groups=ch_outs[1],
        act=act,
        name=name + '.depthwise')
    conv3 = conv_bn(
        input=conv2,
        ch_out=ch_outs[2],
        filter_size=filter_sizes[2],
        stride=strides[2],
        padding=filter_sizes[2] // 2,
        groups=1,
        act=None,
        name=name + '.pointwise')
    if shortcut:
        out = fluid.layers.elementwise_add(x=input, y=conv3)
    else:
        out = conv3
    return out, conv1


@register
class MobileNetV2(object):
    __shared__ = ['norm_type', 'weight_prefix_name']

    def __init__(self,
                 layer_cfg=None,
                 return_idx=[9, 12],
                 act='relu',
                 norm_type='bn',
                 weight_prefix_name=''):
        if layer_cfg is None:
            self.layer_cfg = [
                # n, func, [ch_outs, filter_sizes, strides, shortcut, act]
                [
                    1, ibottleneck,
                    [[32, 32, 16], [3, 3, 1], [2, 1, 1], False, act]
                ],  #0 /2
                [
                    1, ibottleneck, [[96, 96, 24], [1, 3, 1], [1, 2, 1], False,
                                     act]
                ],  #1 /4
                [
                    1, ibottleneck, [[144, 144, 24], [1, 3, 1], [1, 1, 1], True,
                                     act]
                ],
                [
                    1, ibottleneck, [[144, 144, 32], [1, 3, 1], [1, 2, 1],
                                     False, act]
                ],  #3 /8
                [
                    2, ibottleneck, [[192, 192, 32], [1, 3, 1], [1, 1, 1], True,
                                     act]
                ],
                [
                    1, ibottleneck, [[192, 192, 64], [1, 3, 1], [1, 2, 1],
                                     False, act]
                ],  #5 /16
                [
                    3, ibottleneck, [[384, 384, 64], [1, 3, 1], [1, 1, 1], True,
                                     act]
                ],
                [
                    1, ibottleneck, [[384, 384, 96], [1, 3, 1], [1, 1, 1],
                                     False, act]
                ],
                [
                    2, ibottleneck, [[576, 576, 96], [1, 3, 1], [1, 1, 1], True,
                                     act]
                ],
                [
                    1, ibottleneck, [[576, 576, 160], [1, 3, 1], [1, 2, 1],
                                     False, act]
                ],  #9 /32
                [
                    2, ibottleneck, [[960, 960, 160], [1, 3, 1], [1, 1, 1],
                                     True, act]
                ],
                [
                    1, ibottleneck, [[960, 960, 320], [1, 3, 1], [1, 1, 1],
                                     False, act]
                ],
                [1, conv_bn, [1280, 1, 1, 0, 1, act]]
            ]
        else:
            self.layer_cfg = layer_cfg

        self.return_idx = return_idx
        self.norm_type = norm_type
        self.act = act
        self.weight_prefix_name = weight_prefix_name

    def __call__(self, input):
        layer_num = len(self.layer_cfg)
        x = input
        outputs = []
        for i, (n, func, args) in enumerate(self.layer_cfg):
            name = self.weight_prefix_name + '.{}'.format(i)
            for j in range(n):
                if isinstance(x, tuple):
                    x = func(x[0], *args, name=name + '.{}'.format(j))
                else:
                    x = func(x, *args, name=name + '.{}'.format(j))
            if i in self.return_idx:
                outputs.append(x if i == layer_num - 1 else x[1])
            x = x if i == layer_num - 1 else x[0]

        return outputs
