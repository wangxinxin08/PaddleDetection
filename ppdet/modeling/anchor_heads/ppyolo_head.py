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

__all__ = ['PPYOLOHead']


def create_tensor_from_numpy(numpy_array):
    paddle_array = fluid.layers.create_global_var(
        shape=numpy_array.shape, value=0., dtype=numpy_array.dtype)
    fluid.layers.assign(numpy_array, paddle_array)
    return paddle_array


def add_coord(input, is_test=True):

    # NOTE: here is used for exporting model for TensorRT inference,
    #       only support batch_size=1 for input shape should be fixed,
    #       and we create tensor with fixed shape from numpy array
    if is_test and input.shape[2] > 0 and input.shape[3] > 0:
        batch_size = 1
        grid_x = int(input.shape[3])
        grid_y = int(input.shape[2])
        idx_i = np.array(
            [[i / (grid_x - 1) * 2.0 - 1 for i in range(grid_x)]],
            dtype='float32')
        gi_np = np.repeat(idx_i, grid_y, axis=0)
        gi_np = np.reshape(gi_np, newshape=[1, 1, grid_y, grid_x])
        gi_np = np.tile(gi_np, reps=[batch_size, 1, 1, 1])

        x_range = self._create_tensor_from_numpy(gi_np.astype(np.float32))
        x_range.stop_gradient = True

        idx_j = np.array(
            [[j / (grid_y - 1) * 2.0 - 1 for j in range(grid_y)]],
            dtype='float32')
        gj_np = np.repeat(idx_j, grid_x, axis=1)
        gj_np = np.reshape(gj_np, newshape=[1, 1, grid_y, grid_x])
        gj_np = np.tile(gi_np, reps=[batch_size, 1, 1, 1])
        y_range = self._create_tensor_from_numpy(gj_np.astype(np.float32))
        y_range.stop_gradient = True

    # NOTE: in training mode, H and W is variable for random shape,
    #       implement add_coord with shape as Variable
    else:
        input_shape = fluid.layers.shape(input)
        b = input_shape[0]
        h = input_shape[2]
        w = input_shape[3]

        x_range = fluid.layers.range(0, w, 1, 'float32') / ((w - 1.) / 2.)
        x_range = x_range - 1.
        x_range = fluid.layers.unsqueeze(x_range, [0, 1, 2])
        x_range = fluid.layers.expand(x_range, [b, 1, h, 1])
        x_range.stop_gradient = True

        y_range = fluid.layers.range(0, h, 1, 'float32') / ((h - 1.) / 2.)
        y_range = y_range - 1.
        y_range = fluid.layers.unsqueeze(y_range, [0, 1, 3])
        y_range = fluid.layers.expand(y_range, [b, 1, 1, w])
        y_range.stop_gradient = True

    return fluid.layers.concat([input, x_range, y_range], axis=1)


def conv_bn(input, ch_out, filter_size, stride, padding, act='silu', name=None):
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
    bn_param_attr = ParamAttr(regularizer=L2Decay(0.), name=bn_name + '.scale')
    bn_bias_attr = ParamAttr(regularizer=L2Decay(0.), name=bn_name + '.offset')
    out = fluid.layers.batch_norm(
        input=conv,
        act=None,
        param_attr=bn_param_attr,
        bias_attr=bn_bias_attr,
        moving_mean_name=bn_name + '.mean',
        moving_variance_name=bn_name + '.var')
    if act == 'leaky':
        out = fluid.layers.leaky_relu(x=out, alpha=0.1)
    elif act == 'mish':
        out = fluid.layers.mish(out)
    elif act == 'silu':
        out = out * fluid.layers.sigmoid(out)
    return out


def coord_conv(input,
               ch_out,
               filter_size,
               stride,
               padding,
               act='silu',
               is_test=True,
               name=None):
    conv = add_coord(input, is_test=is_test)
    out = conv_bn(
        conv, ch_out, filter_size, stride, padding, act=act, name=name)
    return out


def basic_block(input, ch_out, e=0.5, act='silu', shortcut=True, name=""):
    c_ = int(ch_out * e)
    conv1 = conv_bn(input, c_, 1, 1, 0, act=act, name=name + '.conv1')
    conv2 = conv_bn(conv1, ch_out, 3, 1, 1, act=act, name=name + ".conv2")
    if shortcut:
        ch_in = input.shape[1]
        if ch_in != ch_out:
            short = conv_bn(
                input, ch_out, 1, 1, 0, act=act, name=name + '.short')
        else:
            short = input
        output = short + conv2
    else:
        output = conv2
    return output


def spp(input, ch_out, act='silu', shortcut=False, name=""):
    output1 = input
    output2 = fluid.layers.pool2d(
        input=output1,
        pool_size=5,
        pool_stride=1,
        pool_padding=2,
        ceil_mode=False,
        pool_type='max')
    output3 = fluid.layers.pool2d(
        input=output1,
        pool_size=9,
        pool_stride=1,
        pool_padding=4,
        ceil_mode=False,
        pool_type='max')
    output4 = fluid.layers.pool2d(
        input=output1,
        pool_size=13,
        pool_stride=1,
        pool_padding=6,
        ceil_mode=False,
        pool_type='max')
    output = fluid.layers.concat(
        input=[output1, output2, output3, output4], axis=1)
    output = conv_bn(output, ch_out, 1, 1, 0, act=act, name=name)
    if shortcut:
        output = output + input
    return output


def upsample(input, scale=2, name=""):
    out = fluid.layers.resize_nearest(
        input=input, scale=float(scale), name=name)
    return out


def concat(inputs, axis, name=None):
    return fluid.layers.concat(inputs, axis=axis, name=name)


def stack_conv(input, *args, **kwargs):
    name = kwargs['name']
    output = input
    for i, (ch_out, filter_size, stride, padding, act) in enumerate(args):
        output = conv_bn(output, ch_out, filter_size, stride, padding, act,
                         name + ".{}".format(i))
    return output


@register
class PPYOLOHead(object):
    """
    Head block for YOLOv3 network

    Args:
        conv_block_num (int): number of conv block in each detection block
        norm_decay (float): weight decay for normalization layer weights
        num_classes (int): number of output classes
        anchors (list): anchors
        anchor_masks (list): anchor masks
        nms (object): an instance of `MultiClassNMS`
    """
    __inject__ = ['yolo_loss', 'nms']
    __shared__ = ['num_classes', 'weight_prefix_name']

    def __init__(self,
                 neck_cfg=None,
                 save_idx=None,
                 head_cfg=None,
                 act='silu',
                 norm_decay=0.,
                 num_classes=80,
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 iou_aware=False,
                 iou_aware_factor=0.4,
                 yolo_loss="PPYOLOLoss",
                 nms=MultiClassNMS(
                     score_threshold=0.01,
                     nms_top_k=1000,
                     keep_top_k=100,
                     nms_threshold=0.45,
                     background_label=-1).__dict__,
                 weight_prefix_name='',
                 downsample=[32, 16, 8],
                 clip_bbox=True,
                 scale_x_y=1.0):
        check_version("1.8.4")
        self.name = weight_prefix_name
        if neck_cfg is None:
            self.neck_cfg = [
                # 0 block3
                # 1 block4
                # 2 block5
                [-1, conv_bn, [512, 1, 1, 0, act]],  #3
                [-1, basic_block, [512, 2, act, True]],  #4
                [-1, spp, [512, act, False]],  #5
                [-1, basic_block, [512, 2, act, True]],  #6 P5
                [-1, conv_bn, [256, 1, 1, 0, act]],  #7 transition
                [-1, upsample, [2]],  #8 upsample
                [[-1, 1], concat, [1]],  #9 concat
                [-1, conv_bn, [256, 1, 1, 0, act]],  #10
                [-1, basic_block, [256, 2, act, True]],  #11
                [-1, basic_block, [256, 2, act, True]],  #12 P4
                [-1, conv_bn, [128, 1, 1, 0, act]],  #13 transition
                [-1, upsample, [2]],  #14 upsample
                [[-1, 0], concat, [1]],  #15 concat
                [-1, conv_bn, [128, 1, 1, 0, act]],  #16
                [-1, basic_block, [128, 2, act, True]],  #17
                [-1, basic_block, [128, 2, act, True]],  #18, C3
                [-1, conv_bn, [256, 3, 2, 1, act]],  #19 downsample
                [[-1, 4], concat, [1]],  #20 concat
                [-1, conv_bn, [256, 1, 1, 0, act]],  #21
                [-1, basic_block, [256, 2, act, True]],  #22
                [-1, basic_block, [256, 2, act, True]],  #23, C4
                [-1, conv_bn, [512, 3, 2, 1, act]],  #24 downsample
                [[-1, 3], concat, [1]],  #25 concat
                [-1, conv_bn, [512, 1, 1, 0, act]],  #26
                [-1, basic_block, [512, 2, act, True]],  #27
                [-1, basic_block, [512, 2, act, True]],  #28, C5
            ]
            self.save_idx = [6, 12, 18, 23, 28]
        else:
            self.neck_cfg = neck_cfg
            self.save_idx = save_idx

        self.out_channels = [
            len(anchor_mask) * (num_classes + 6) for anchor_mask in anchor_masks
        ]
        if head_cfg is None:
            self.head_cfg = [
                # ch_outs
                [7, conv_bn, [1024, 3, 1, 1, act]],
                [6, conv_bn, [512, 3, 1, 1, act]],
                [5, conv_bn, [256, 3, 1, 1, act]],
            ]
        else:
            self.head_cfg = head_cfg

        self.norm_decay = norm_decay
        self.num_classes = num_classes
        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor
        self.anchor_masks = anchor_masks
        self._parse_anchors(anchors)
        self.yolo_loss = yolo_loss
        self.nms = nms
        if isinstance(nms, dict):
            self.nms = MultiClassNMS(**nms)
        self.downsample = downsample
        self.scale_x_y = scale_x_y
        self.clip_bbox = clip_bbox

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
        layers = input[:out_layer_num]

        # for i in range(out_layer_num):
        #     fluid.layers.Print(layers[i])

        # neck
        pre = layers[-1]
        for i, (f, m, args) in enumerate(self.neck_cfg):
            if isinstance(f, int):
                inputs = pre if f == -1 else layers[i]
            else:
                inputs = [pre if idx == -1 else layers[idx] for idx in f]
            layer = m(inputs, *args, name=self.name + 'yolo_neck.{}'.format(i))
            pre = layer
            if i + 3 in self.save_idx:
                # fluid.layers.Print(layer)
                layers.append(layer)

        # head
        for i, (f, m, args) in enumerate(self.head_cfg):
            if isinstance(f, int):
                inputs = layers[f]
            else:
                inputs = [layers[idx] for idx in f]

            layer = m(inputs, *args, name=self.name + 'yolo_head.{}'.format(i))
            layer = fluid.layers.conv2d(
                input=layer,
                num_filters=self.out_channels[i],
                filter_size=1,
                stride=1,
                padding=0,
                act=None,
                param_attr=ParamAttr(
                    name=self.name + "yolo_output.{}.conv.weights".format(i)),
                bias_attr=ParamAttr(
                    regularizer=L2Decay(0.),
                    name=self.name + "yolo_output.{}.conv.bias".format(i)))
            # fluid.layers.Print(layer)
            outputs.append(layer)

        return outputs

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
            if self.iou_aware:
                output = get_iou_aware_score(output,
                                             len(self.anchor_masks[i]),
                                             self.num_classes,
                                             self.iou_aware_factor)
            box, score = fluid.layers.yolo_box(
                x=output,
                img_size=im_size,
                anchors=self.mask_anchors[i],
                class_num=self.num_classes,
                conf_thresh=self.nms.score_threshold,
                downsample_ratio=self.downsample[i],
                name=self.name + "yolo_box" + str(i),
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
                              self.mask_anchors, self.num_classes, self.name)
