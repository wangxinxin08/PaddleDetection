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

__all__ = ['PPYOLOHead', 'yolo_box']


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

        x_range = _create_tensor_from_numpy(gi_np.astype(np.float32))
        x_range.stop_gradient = True

        idx_j = np.array(
            [[j / (grid_y - 1) * 2.0 - 1 for j in range(grid_y)]],
            dtype='float32')
        gj_np = np.repeat(idx_j, grid_x, axis=1)
        gj_np = np.reshape(gj_np, newshape=[1, 1, grid_y, grid_x])
        gj_np = np.tile(gi_np, reps=[batch_size, 1, 1, 1])
        y_range = _create_tensor_from_numpy(gj_np.astype(np.float32))
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
    conv1 = conv_bn(input, c_, 1, 1, 0, act=act, name=name + '.0')
    conv2 = conv_bn(conv1, ch_out, 3, 1, 1, act=act, name=name + ".1")
    if shortcut:
        ch_in = input.shape[1]
        if ch_in != ch_out:
            short = conv_bn(
                input, ch_out, 1, 1, 0, act=act, name=name + '.short')
        else:
            short = input
        output = fluid.layers.elementwise_add(x=short, y=conv2)
    else:
        output = conv2
    return output


def spp_module(input, ch_out, e=0.5, act='silu', shortcut=True, name=""):
    c_ = int(ch_out * e)
    output1 = conv_bn(input, c_, 1, 1, 0, act=act, name=name + '.0')
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
    output = conv_bn(output, ch_out, 1, 1, 0, act=act, name=name + '.1')
    if shortcut:
        output = fluid.layers.elementwise_add(x=output, y=input)
    else:
        output = output
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


def C3(input,
       ch_out,
       act,
       use_spp=False,
       drop_block=False,
       is_test=False,
       name=""):
    c_ = ch_out // 2
    cv1 = conv_bn(input, c_, 1, 1, 0, act, name=name + ".left")
    cv2 = conv_bn(input, c_, 1, 1, 0, act, name=name + ".right")
    if use_spp:
        cfg = [['basic_block', [c_, 1, act, False]],
               ['spp_module', [c_, 1, act, False]],
               ['basic_block', [c_, 1, act, False]]]
    else:
        cfg = [['basic_block', [c_, 1, act, False]],
               ['basic_block', [c_, 1, act, False]],
               ['basic_block', [c_, 1, act, False]]]

    for i, (m, args) in enumerate(cfg):
        cv1 = eval(m)(cv1, *args, name=name + ".left.{}".format(i))
        if drop_block and i == 1:
            cv1 = DropBlock(cv1, 3, 0.9, is_test=is_test)

    cv = fluid.layers.concat([cv1, cv2], 1)
    cv = conv_bn(cv, ch_out, 1, 1, 0, act, name=name)
    return cv


def bottle_block(input,
                 ch_out,
                 e=4,
                 act='silu',
                 use_spp=False,
                 shortcut=True,
                 name=""):
    c_ = int(ch_out * e)
    conv1 = conv_bn(input, ch_out, 1, 1, 0, act=act, name=name + '.0')
    if use_spp:
        conv2 = spp(conv1, name=name + ".1")
    else:
        conv2 = conv_bn(conv1, ch_out, 3, 1, 1, act=act, name=name + ".1")
    conv3 = conv_bn(conv2, c_, 1, 1, 0, act=None, name=name + ".2")
    if shortcut:
        ch_in = input.shape[1]
        if ch_in != ch_out:
            short = conv_bn(input, c_, 1, 1, 0, act=None, name=name + '.short')
        else:
            short = input
        output = fluid.layers.elementwise_add(
            x=short, y=conv3, act=act, name=name + ".add")
    else:
        output = conv3
    return output


def B3(input,
       ch_out,
       e=4,
       act='silu',
       use_spp=False,
       drop_block=False,
       is_test=False,
       name=""):
    if use_spp:
        cfg = [['bottle_block', [ch_out, 4, act, False]],
               ['bottle_block', [ch_out, 4, act, True]],
               ['bottle_block', [ch_out, 4, act, False]]]
    else:
        cfg = [['bottle_block', [ch_out, 4, act, False]],
               ['bottle_block', [ch_out, 4, act, False]],
               ['bottle_block', [ch_out, 4, act, False]]]

    output = input
    for i, (m, args) in enumerate(cfg):
        output = eval(m)(output, *args, name=name + ".{}".format(i))
        if drop_block and i == 1:
            output = DropBlock(output, 3, 0.9, is_test=is_test)

    return output


def B2(input,
       ch_out,
       e=4,
       act='silu',
       use_spp=False,
       drop_block=False,
       is_test=False,
       name=""):
    if use_spp:
        cfg = [['bottle_block', [ch_out, 4, act, True]],
               ['bottle_block', [ch_out, 4, act, False]]]
    else:
        cfg = [['bottle_block', [ch_out, 4, act, False]],
               ['bottle_block', [ch_out, 4, act, False]]]

    output = input
    for i, (m, args) in enumerate(cfg):
        output = eval(m)(output, *args, name=name + ".{}".format(i))
        if drop_block and i == 1:
            output = DropBlock(output, 3, 0.9, is_test=is_test)

    return output


def spp(input, name=""):
    output1 = fluid.layers.pool2d(
        input=input,
        pool_size=5,
        pool_stride=1,
        pool_padding=2,
        ceil_mode=False,
        pool_type='max')
    output2 = fluid.layers.pool2d(
        input=input,
        pool_size=9,
        pool_stride=1,
        pool_padding=4,
        ceil_mode=False,
        pool_type='max')
    output3 = fluid.layers.pool2d(
        input=input,
        pool_size=13,
        pool_stride=1,
        pool_padding=6,
        ceil_mode=False,
        pool_type='max')
    output = fluid.layers.concat(
        input=[input, output1, output2, output3], axis=1)
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
                 fpn_cfg=None,
                 pan_cfg=None,
                 act='silu',
                 norm_decay=0.,
                 num_classes=80,
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 iou_aware=False,
                 iou_aware_factor=0.4,
                 drop_block=False,
                 block_size=3,
                 keep_prob=0.9,
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
                 fpn_head=False,
                 scale_x_y=1.0):
        check_version("1.8.4")
        self.name = weight_prefix_name
        if fpn_cfg is None:
            self.fpn_cfg = [
                [[2, 'C3', [1024, act, True, False]]  # P3
                 ],
                [
                    [-1, 'conv_bn', [512, 1, 1, 0, act]],
                    [-1, 'upsample', [2]],
                    [[-1, 1], 'concat', [1]],
                    [-1, 'C3', [512, act, False, False]]  # P4
                ],
                [[-1, 'conv_bn', [256, 1, 1, 0, act]], [-1, 'upsample', [2]],
                 [[-1, 0], 'concat', [1]]]
            ]
        else:
            self.fpn_cfg = fpn_cfg

        if pan_cfg is None:
            self.pan_cfg = [
                [[0, 'C3', [256, act, False, False]]  # C3
                 ],
                [
                    [-1, 'conv_bn', [256, 3, 2, 1, act]],
                    [[-1, 1], 'concat', [1]],
                    [-1, 'C3', [512, act, False, False]],  # C4
                ],
                [
                    [-1, 'conv_bn', [512, 3, 2, 1, act]],
                    [[-1, 2], 'concat', [1]],
                    [-1, 'C3', [1024, act, False, False]]  # C5
                ]
            ]
        else:
            self.pan_cfg = pan_cfg

        self.out_channels = [
            len(anchor_mask) * (num_classes + 6) for anchor_mask in anchor_masks
        ]

        self.norm_decay = norm_decay
        self.num_classes = num_classes
        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor
        self.drop_block = drop_block
        self.keep_prob = keep_prob
        self.block_size = block_size
        self.anchor_masks = anchor_masks
        self._parse_anchors(anchors)
        self.yolo_loss = yolo_loss
        self.nms = nms
        if isinstance(nms, dict):
            self.nms = MultiClassNMS(**nms)
        self.downsample = downsample
        self.scale_x_y = scale_x_y
        self.fpn_head = fpn_head
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

        # fpn
        for i, layer_cfg in enumerate(self.fpn_cfg):
            for j, (f, m, args) in enumerate(layer_cfg):
                if isinstance(f, int):
                    inputs = output if f == -1 else layers[f]
                else:
                    inputs = [output if idx == -1 else layers[idx] for idx in f]
                if m in ['B2', 'B3', 'C3']:
                    output = eval(m)(
                        inputs,
                        *args,
                        is_test=(not is_train),
                        name=self.name + 'yolo_fpn.{}.{}'.format(i, j))
                else:
                    output = eval(m)(
                        inputs,
                        *args,
                        name=self.name + 'yolo_fpn.{}.{}'.format(i, j))
            if self.drop_block and i < 2:
                output = DropBlock(
                    output,
                    self.block_size,
                    self.keep_prob,
                    is_test=(not is_train))
            layers[-i - 1] = output

        # fpn head
        if self.fpn_head:
            fpn_outputs = []
            for i, inputs in enumerate(layers[::-1]):
                layer = fluid.layers.conv2d(
                    input=inputs,
                    num_filters=self.out_channels[i],
                    filter_size=1,
                    stride=1,
                    padding=0,
                    act=None,
                    param_attr=ParamAttr(
                        name=self.name +
                        "yolo_fpn_output.{}.conv.weights".format(i)),
                    bias_attr=ParamAttr(
                        regularizer=L2Decay(0.),
                        name=self.name +
                        "yolo_fpn_output.{}.conv.bias".format(i)))
                fpn_outputs.append(layer)

        for i, layer_cfg in enumerate(self.pan_cfg):
            for j, (f, m, args) in enumerate(layer_cfg):
                if isinstance(f, int):
                    inputs = output if f == -1 else layers[f]
                else:
                    inputs = [output if idx == -1 else layers[idx] for idx in f]
                if m in ['B2', 'B3', 'C3']:
                    output = eval(m)(
                        inputs,
                        *args,
                        is_test=(not is_train),
                        name=self.name + 'yolo_pan.{}.{}'.format(i, j))
                else:
                    output = eval(m)(
                        inputs,
                        *args,
                        name=self.name + 'yolo_pan.{}.{}'.format(i, j))
            if self.drop_block:
                output = DropBlock(
                    output,
                    self.block_size,
                    self.keep_prob,
                    is_test=(not is_train))
            layers[i] = output

        # pan head
        for i, inputs in enumerate(layers[::-1]):
            layer = fluid.layers.conv2d(
                input=inputs,
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
            outputs.append(layer)

        return outputs, fpn_outputs if self.fpn_head else None

    def get_prediction(self, input, im_size, exclude_nms=False):
        """
        Get prediction result of YOLOv3 network

        Args:
            input (list): List of Variables, output of backbone stages
            im_size (Variable): Variable of size([h, w]) of each image

        Returns:
            pred (Variable): The prediction result after non-max suppress.

        """

        outputs, _ = self._get_outputs(input, is_train=False)

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
            box, score = yolo_box(
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
        outputs, fpn_outputs = self._get_outputs(input, is_train=True)

        loss = self.yolo_loss(outputs, gt_box, gt_label, gt_score, targets,
                              self.anchors, self.anchor_masks,
                              self.mask_anchors, self.num_classes, self.name)
        if self.fpn_head:
            fpn_loss = self.yolo_loss(
                fpn_outputs,
                gt_box,
                gt_label,
                gt_score,
                targets,
                self.anchors,
                self.anchor_masks,
                self.mask_anchors,
                self.num_classes,
                self.name,
                fpn_loss=True)
            for k in fpn_loss:
                loss[k] = loss[k] + fpn_loss[k]

        return loss


def yolo_box(x,
             img_size,
             anchors,
             class_num,
             conf_thresh,
             downsample_ratio,
             clip_bbox,
             scale_x_y,
             name=''):
    shape = fluid.layers.shape(x)
    b, c, h, w = shape[0], shape[1], shape[2], shape[3]
    na = len(anchors) // 2
    no = class_num + 5
    x = fluid.layers.reshape(x, [b, na, no, h, w])
    x = fluid.layers.transpose(x, perm=[0, 1, 3, 4, 2])
    anchors = np.array(anchors).reshape((1, na, 1, 1, 2))
    anchors = _create_tensor_from_numpy(
        anchors.astype(np.float32), name='anchors' + name)
    grid = _make_grid(w, h)
    bias_x_y = 0.5 * (scale_x_y - 1)
    im_h = fluid.layers.reshape(img_size[:, 0:1], (b, 1, 1, 1, 1))
    im_w = fluid.layers.reshape(img_size[:, 1:2], (b, 1, 1, 1, 1))
    xc = (scale_x_y * fluid.layers.sigmoid(x[:, :, :, :, 0:1]) - bias_x_y +
          grid[:, :, :, :, 0:1]) / w
    yc = (scale_x_y * fluid.layers.sigmoid(x[:, :, :, :, 1:2]) - bias_x_y +
          grid[:, :, :, :, 1:2]) / h
    wc = (2 * fluid.layers.sigmoid(x[:, :, :, :, 2:3]))**2
    wc = wc * anchors[:, :, :, :, 0:1] / (w * downsample_ratio)
    hc = (2 * fluid.layers.sigmoid(x[:, :, :, :, 3:4]))**2
    hc = hc * anchors[:, :, :, :, 1:2] / (h * downsample_ratio)
    x1 = xc - wc * 0.5
    y1 = yc - hc * 0.5
    x2 = xc + wc * 0.5
    y2 = yc + hc * 0.5
    if clip_bbox:
        x1 = fluid.layers.clip(x1, 0., 1.)
        y1 = fluid.layers.clip(y1, 0., 1.)
        x2 = fluid.layers.clip(x2, 0., 1.)
        y2 = fluid.layers.clip(y2, 0., 1.)
    x1 = x1 * im_w
    y1 = y1 * im_h
    x2 = x2 * im_w
    y2 = y2 * im_h
    bbox = fluid.layers.concat([x1, y1, x2, y2], axis=-1)
    conf = fluid.layers.sigmoid(x[:, :, :, :, 4:5])
    mask = fluid.layers.cast(conf >= conf_thresh, 'float32')
    conf = conf * mask
    score = fluid.layers.sigmoid(x[:, :, :, :, 5:]) * conf
    bbox = bbox * mask
    bbox = fluid.layers.reshape(bbox, (b, -1, 4))
    score = fluid.layers.reshape(score, (b, -1, class_num))
    return bbox, score


def _make_grid(nx, ny):
    yv, xv = fluid.layers.meshgrid([
        fluid.layers.range(0, ny, 1, 'float32'),
        fluid.layers.range(0, nx, 1, 'float32')
    ])
    grid = fluid.layers.stack([xv, yv], axis=2)
    return fluid.layers.reshape(grid, (1, 1, ny, nx, 2))


def _create_tensor_from_numpy(numpy_array, name=None):
    paddle_array = fluid.layers.create_global_var(
        shape=numpy_array.shape, value=0., dtype=numpy_array.dtype, name=name)
    fluid.layers.assign(numpy_array, paddle_array)
    return paddle_array
