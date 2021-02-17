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

import numpy as np

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import MSRAInitializer, ConstantInitializer

from ppdet.modeling.ops import MultiClassNMS, MultiClassSoftNMS, MatrixNMS
from ppdet.modeling.losses.yolo_loss import YOLOv3Loss
from ppdet.core.workspace import register
from collections import Sequence
from ppdet.utils.check import check_version

__all__ = ['YOLOLiteHead']


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
    elif act == 'relu6':
        out = fluid.layers.relu6(out)

    return out


def conv(input,
         ch_out,
         filter_size,
         stride,
         padding,
         groups=1,
         act=None,
         name=''):
    out = fluid.layers.conv2d(
        input=input,
        num_filters=ch_out,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        groups=groups,
        act=act,
        param_attr=ParamAttr(
            initializer=MSRAInitializer(uniform=False),
            name=name + ".conv.weights"),
        bias_attr=ParamAttr(name=name + ".conv.bias"))
    return out


def add(inputs, name=''):
    return fluid.layers.elementwise_add(x=inputs[0], y=inputs[1], name=name)


def upsample(input, scale=2, name=""):
    out = fluid.layers.resize_nearest(
        input=input, scale=float(scale), name=name)
    out = fluid.layers.pool2d(
        input=out,
        pool_size=2,
        pool_stride=1,
        pool_padding=[1, 0, 1, 0],
        ceil_mode=False,
        pool_type='max')
    return out


@register
class YOLOLiteHead(object):
    __inject__ = ['yolo_loss', 'nms']
    __shared__ = ['num_classes', 'weight_prefix_name']

    def __init__(self,
                 neck_cfg=None,
                 head_cfg=None,
                 act='relu',
                 num_classes=80,
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 yolo_loss="YOLOv3Loss",
                 nms=MultiClassNMS(
                     score_threshold=0.01,
                     nms_top_k=1000,
                     keep_top_k=100,
                     nms_threshold=0.45,
                     background_label=-1).__dict__,
                 weight_prefix_name='',
                 downsample=[32, 16],
                 scale_x_y=1.0,
                 clip_bbox=True):
        check_version("1.8.4")
        self.num_classes = num_classes
        self.anchor_masks = anchor_masks
        self._parse_anchors(anchors)
        self.yolo_loss = yolo_loss
        self.nms = nms
        self.prefix_name = weight_prefix_name
        if isinstance(nms, dict):
            self.nms = MultiClassNMS(**nms)
        self.downsample = downsample
        self.scale_x_y = scale_x_y
        self.clip_bbox = clip_bbox
        if neck_cfg is None:
            # [block5, block4]
            self.neck_cfg = [
                # from, func, args
                # block5
                # block4
                [0, conv_bn, [1280, 3, 1, 1, 1280, act]],  # 2
                [-1, conv_bn, [576, 1, 1, 0, 1, act]],  # 3 C5
                [-1, upsample, [2, ]],  # 4
                [1, conv_bn, [576, 3, 1, 1, 576, act]],  # 5
                [-1, conv_bn, [576, 1, 1, 0, 1, act]],  # 6
                [[-1, 4], add, []],  # 7
                [-1, conv_bn, [576, 3, 1, 1, 576, act]],  # 8
                [-1, conv_bn, [576, 1, 1, 0, 1, act]]  # 9 C4
            ]
        else:
            self.neck_cfg = neck_cfg

        if head_cfg is None:
            out_channels = [
                len(anchor_mask) * (num_classes + 5)
                for anchor_mask in anchor_masks
            ]
            self.head_cfg = [[3, conv, [out_channels[0], 1, 1, 0, 1]],
                             [9, conv, [out_channels[1], 1, 1, 0, 1]]]
        else:
            self.head_cfg = head_cfg

    def _parse_anchors(self, anchors):
        """
        Check ANCHORS/ANCHOR_MASKS in config and parse mask_anchors

        """
        self.anchors = []
        self.mask_anchors = []

        assert len(anchors) > 0, "ANCHORS not set."
        assert len(self.anchor_masks) > 0, "ANCHOR_MASKS not set."

        for anchor in anchors:
            assert len(anchor) == 2, "anchor {} len should be 2".format(anchor)
            self.anchors.extend(anchor)

        anchor_num = len(anchors)
        for masks in self.anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, "anchor mask index overflow"
                self.mask_anchors[-1].extend(anchors[mask])

    def _get_outputs(self, input, is_train=True):
        """
        Get YOLOv3 head output

        Args:
            input (list): List of Variables, output of backbone stages
            is_train (bool): whether in train or test mode

        Returns:
            outputs (list): Variables of each output layer
        """

        outputs = []

        # get last out_layer_num blocks in reverse order
        out_layer_num = len(self.anchor_masks)
        blocks = input[-1:-out_layer_num - 1:-1]

        for i, (from_idx, module, args) in enumerate(self.neck_cfg):
            module = eval(module) if isinstance(module, str) else module
            name = self.prefix_name + 'yolo_neck.{}'.format(i)
            if isinstance(from_idx, int):
                inputs = blocks[from_idx]
            else:
                inputs = [blocks[idx] for idx in from_idx]

            output = module(inputs, *args, name=name)
            blocks.append(output)

        for i, (from_idx, module, args) in enumerate(self.head_cfg):
            module = eval(module) if isinstance(module, str) else module
            name = self.prefix_name + 'yolo_head.{}'.format(i)
            if isinstance(from_idx, int):
                inputs = blocks[from_idx]
            else:
                inputs = [blocks[idx] for idx in from_idx]
            output = module(inputs, *args, name=name)
            outputs.append(output)

        return outputs

    def get_loss(self, input, gt_box, gt_label, gt_score, targets):
        """
        Get final loss of network of YOLOv3.

        Args:
            input (list): List of Variables, output of backbone stages
            gt_box (Variable): The ground-truth boudding boxes.
            gt_label (Variable): The ground-truth class labels.
            gt_score (Variable): The ground-truth boudding boxes mixup scores.
            targets ([Variables]): List of Variables, the targets for yolo
                                   loss calculatation.

        Returns:
            loss (Variable): The loss Variable of YOLOv3 network.

        """
        outputs = self._get_outputs(input, is_train=True)

        return self.yolo_loss(outputs, gt_box, gt_label, gt_score, targets,
                              self.anchors, self.anchor_masks,
                              self.mask_anchors, self.num_classes,
                              self.prefix_name)

    def get_prediction(self, input, im_size, exclude_nms=False):
        """
        Get prediction result of YOLOv3 network

        Args:
            input (list): List of Variables, output of backbone stages
            im_size (Variable): Variable of size([h, w]) of each image

        Returns:
            pred (Variable): The prediction result after non-max suppress.

        """

        outputs = self._get_outputs(input, is_train=False)

        boxes = []
        scores = []
        for i, output in enumerate(outputs):
            scale_x_y = self.scale_x_y if not isinstance(
                self.scale_x_y, Sequence) else self.scale_x_y[i]
            box, score = fluid.layers.yolo_box(
                x=output,
                img_size=im_size,
                anchors=self.mask_anchors[i],
                class_num=self.num_classes,
                conf_thresh=self.nms.score_threshold,
                downsample_ratio=self.downsample[i],
                name=self.prefix_name + "yolo_box" + str(i),
                clip_bbox=self.clip_bbox,
                scale_x_y=scale_x_y)
            boxes.append(box)
            scores.append(fluid.layers.transpose(score, perm=[0, 2, 1]))

        yolo_boxes = fluid.layers.concat(boxes, axis=1)
        yolo_scores = fluid.layers.concat(scores, axis=2)

        # Only for benchmark, postprocess(NMS) is not needed
        if exclude_nms:
            return {'bbox': yolo_scores}

        if type(self.nms) is MultiClassSoftNMS:
            yolo_scores = fluid.layers.transpose(yolo_scores, perm=[0, 2, 1])
        pred = self.nms(bboxes=yolo_boxes, scores=yolo_scores)
        return {'bbox': pred}
