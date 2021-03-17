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

__all__ = ['YOLOv3HeadPAN']


@register
class YOLOv3HeadPAN(object):
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
                 conv_block_num=2,
                 norm_decay=0.,
                 num_classes=80,
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 drop_block=False,
                 coord_conv=False,
                 iou_aware=False,
                 iou_aware_factor=0.4,
                 block_size=3,
                 keep_prob=0.9,
                 act='leaky',
                 spp_stage=5,
                 yolo_loss="YOLOv3Loss",
                 spp=False,
                 nms=MultiClassNMS(
                     score_threshold=0.01,
                     nms_top_k=1000,
                     keep_top_k=100,
                     nms_threshold=0.45,
                     background_label=-1).__dict__,
                 weight_prefix_name='',
                 downsample=[32, 16, 8],
                 scale_x_y=1.0,
                 clip_bbox=True):
        check_version("1.8.4")
        self.conv_block_num = conv_block_num
        self.norm_decay = norm_decay
        self.num_classes = num_classes
        self.anchor_masks = anchor_masks
        self._parse_anchors(anchors)
        self.yolo_loss = yolo_loss
        self.nms = nms
        self.prefix_name = weight_prefix_name
        self.drop_block = drop_block
        self.iou_aware = iou_aware
        self.coord_conv = coord_conv
        self.iou_aware_factor = iou_aware_factor
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.act = act
        self.use_spp = spp
        self.spp_stage = spp_stage
        if isinstance(nms, dict):
            self.nms = MultiClassNMS(**nms)
        self.downsample = downsample
        self.scale_x_y = scale_x_y
        self.clip_bbox = clip_bbox

    def _create_tensor_from_numpy(self, numpy_array):
        paddle_array = fluid.layers.create_global_var(
            shape=numpy_array.shape, value=0., dtype=numpy_array.dtype)
        fluid.layers.assign(numpy_array, paddle_array)
        return paddle_array

    def _add_coord(self, input, is_test=True):
        if not self.coord_conv:
            return input

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
            regularizer=L2Decay(self.norm_decay), name=bn_name + '.scale')
        bn_bias_attr = ParamAttr(
            regularizer=L2Decay(self.norm_decay), name=bn_name + '.offset')
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

    def _softplus(self, input):
        expf = fluid.layers.exp(input)
        return fluid.layers.log(1 + expf)

    # def _mish(self, input):
    #     return input * fluid.layers.tanh(self._softplus(input))

    def _spp_module(self, input, name=""):
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
        return output

    def stack_conv(self,
                   input,
                   ch_list=[512, 1024, 512],
                   filter_list=[1, 3, 1],
                   stride=1,
                   is_test=True,
                   name=None):
        conv = input
        for i, (ch_out, f_size) in enumerate(zip(ch_list, filter_list)):
            padding = 1 if f_size == 3 else 0
            conv = self._conv_bn(
                conv,
                ch_out=ch_out,
                filter_size=f_size,
                stride=stride,
                padding=padding,
                act=self.act,
                name='{}.{}'.format(name, i))
            if i == 0:
                short = conv
            if i == 2:
                residual = conv
                conv = fluid.layers.elementwise_add(
                    x=short, y=residual, name='{}.{}.add'.format(name, 1))
                short = conv
        residual = conv
        conv = fluid.layers.elementwise_add(
            x=short, y=residual, name='{}.{}.add'.format(name, 2))
        if self.drop_block:
            conv = DropBlock(
                conv,
                block_size=self.block_size,
                keep_prob=self.keep_prob,
                is_test=is_test)
        return conv

    def spp_module(self,
                   input,
                   channel=512,
                   conv_block_num=2,
                   is_test=True,
                   name=None):
        conv = input
        for j in range(conv_block_num):
            conv = self._conv_bn(
                conv,
                channel,
                filter_size=1,
                stride=1,
                padding=0,
                act=self.act,
                name='{}.{}.0'.format(name, j))
            if j == 0:
                short = conv
            if j == 1:
                residual = conv
                conv = fluid.layers.elementwise_add(
                    x=short, y=residual, name='{}.{}.add'.format(name, 1))
                conv = self._spp_module(conv, name="spp")
                conv = self._conv_bn(
                    conv,
                    512,
                    filter_size=1,
                    stride=1,
                    padding=0,
                    act=self.act,
                    name='{}.{}.spp.conv'.format(name, j))
                short = conv
            conv = self._conv_bn(
                conv,
                channel * 2,
                filter_size=3,
                stride=1,
                padding=1,
                act=self.act,
                name='{}.{}.1'.format(name, j))

        conv = self._conv_bn(
            conv,
            channel,
            filter_size=1,
            stride=1,
            padding=0,
            act=self.act,
            name='{}.2'.format(name))
        residual = conv
        conv = fluid.layers.elementwise_add(
            x=short, y=residual, name='{}.{}.add'.format(name, 2))
        if self.drop_block:
            conv = DropBlock(
                conv,
                block_size=self.block_size,
                keep_prob=self.keep_prob,
                is_test=is_test)
        return conv

    def pan_module(self, input, filter_list, is_test=True, name=None):
        for i in range(1, len(input)):
            conv_left = input[i]
            # ch_out = input[i].shape[1] // 4
            # conv_left = self._conv_bn(
            #   input[i],
            #   ch_out=ch_out,
            #   filter_size=1,
            #   stride=1,
            #   padding=0,
            #   name=name+'.{}.left'.format(i))
            ch_out = input[i - 1].shape[1] // 2
            conv_right = self._add_coord(input[i - 1], is_test=is_test)
            conv_right = self._conv_bn(
                conv_right,
                ch_out=ch_out,
                filter_size=1,
                stride=1,
                padding=0,
                act=self.act,
                name=name + '.{}.right'.format(i))
            conv_right = self._upsample(conv_right)
            pan_out = fluid.layers.concat([conv_left, conv_right], axis=1)
            ch_list = [ch_out * k for k in [1, 2, 1, 2, 1]]
            input[i] = self.stack_conv(
                pan_out,
                ch_list=ch_list,
                filter_list=filter_list,
                is_test=is_test,
                name=name + '.stack_conv.{}'.format(i))
        return input

    def _upsample(self, input, scale=2, name=None):
        out = fluid.layers.resize_nearest(
            input=input, scale=float(scale), name=name)
        return out

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
            input (list): List of Variables, output of backbonxe stages
            is_train (bool): whether in train or test mode
        Returns:
            outputs (list): Variables of each output layer
        """

        outputs = []
        filter_list = [1, 3, 1, 3, 1]
        spp_stage = self.spp_stage
        # spp_stage = len(input) - self.spp_stage
        # get last out_layer_num blocks in reverse order
        out_layer_num = len(self.anchor_masks)
        blocks = input[-1:-out_layer_num - 1:-1]
        # SPP needs to be modified
        blocks[spp_stage] = self.spp_module(
            blocks[spp_stage],
            is_test=(not is_train),
            name=self.prefix_name + "spp_module")
        blocks = self.pan_module(
            blocks,
            filter_list=filter_list,
            is_test=(not is_train),
            name=self.prefix_name + "pan_module")

        # whether add reverse
        # reverse order back to input
        blocks = blocks[::-1]

        # first block should be 19x19
        route = None
        for i, block in enumerate(blocks):
            if i > 0:  # perform concat in first 2 detection_block
                # downsample
                ch_in = route.shape[1]
                route = self._add_coord(route, is_test=(not is_train))
                route = self._conv_bn(
                    route,
                    ch_out=ch_in * 2,
                    filter_size=3,
                    stride=2,
                    padding=1,
                    act=self.act,
                    name=self.prefix_name + 'yolo_block.route.{}'.format(i))
                block = fluid.layers.concat(input=[route, block], axis=1)
                ch_list = [block.shape[1] // 2 * k for k in [1, 2, 1, 2, 1]]
                block = self.stack_conv(
                    block,
                    ch_list=ch_list,
                    filter_list=filter_list,
                    is_test=(not is_train),
                    name=self.prefix_name +
                    'yolo_block.stack_conv.{}'.format(i))
            route = block

            block = self._add_coord(block, is_test=(not is_train))
            block_out = self._conv_bn(
                block,
                ch_out=block.shape[1] * 2,
                filter_size=3,
                stride=1,
                padding=1,
                act=self.act,
                name=self.prefix_name + 'yolo_output.{}.conv.0'.format(i))

            # out channel number = mask_num * (5 + class_num)
            if self.iou_aware:
                num_filters = len(self.anchor_masks[i]) * (self.num_classes + 6)
            else:
                num_filters = len(self.anchor_masks[i]) * (self.num_classes + 5)
            with fluid.name_scope('yolo_output'):
                block_out = fluid.layers.conv2d(
                    input=block_out,
                    num_filters=num_filters,
                    filter_size=1,
                    stride=1,
                    padding=0,
                    act=None,
                    param_attr=ParamAttr(
                        name=self.prefix_name +
                        "yolo_output.{}.conv.1.weights".format(i)),
                    bias_attr=ParamAttr(
                        regularizer=L2Decay(0.),
                        name=self.prefix_name +
                        "yolo_output.{}.conv.1.bias".format(i)))
                outputs.append(block_out)

        outputs = outputs[::-1]
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
            if self.iou_aware:
                output = get_iou_aware_score(output,
                                             len(self.anchor_masks[i]),
                                             self.num_classes,
                                             self.iou_aware_factor)
            scale_x_y = self.scale_x_y if not isinstance(
                self.scale_x_y, Sequence) else self.scale_x_y[i]
            box, score = self.yolo_box(
                x=output,
                img_size=im_size,
                anchors=self.mask_anchors[i],
                class_num=self.num_classes,
                conf_thresh=self.nms.score_threshold,
                downsample_ratio=self.downsample[i],
                clip_bbox=True,
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

    def _make_grid(self, nx, ny):
        yv, xv = fluid.layers.meshgrid([
            fluid.layers.range(0, ny, 1, 'float32'),
            fluid.layers.range(0, nx, 1, 'float32')
        ])
        grid = fluid.layers.stack([xv, yv], axis=2)
        return fluid.layers.reshape(grid, (1, 1, ny, nx, 2))

    def yolo_box(self, x, img_size, anchors, class_num, conf_thresh,
                 downsample_ratio, clip_bbox, scale_x_y):
        shape = fluid.layers.shape(x)
        b, c, h, w = shape[0], shape[1], shape[2], shape[3]
        na = len(anchors) // 2
        no = class_num + 5
        x = fluid.layers.reshape(x, [b, na, no, h, w])
        x = fluid.layers.transpose(x, perm=[0, 1, 3, 4, 2])
        anchors = np.array(anchors).reshape((1, na, 1, 1, 2))
        anchors = self._create_tensor_from_numpy(anchors)
        grid = self._make_grid(w, h)
        bias_x_y = 0.5 * (scale_x_y - 1)
        im_h = fluid.layers.reshape(img_size[:, 0:1], (b, 1, 1, 1, 1))
        im_w = fluid.layers.reshape(img_size[:, 1:2], (b, 1, 1, 1, 1))
        xc = (scale_x_y * x[:, :, :, :, 0:1] - bias_x_y +
              grid[:, :, :, :, 0:1]) / w
        yc = (scale_x_y * x[:, :, :, :, 1:2] - bias_x_y +
              grid[:, :, :, :, 1:2]) / h
        wc = fluid.layers.exp(x[:, :, :, :, 2:3]) * anchors[:, :, :, :, 0:1] / (
            w * downsample_ratio)
        hc = fluid.layers.exp(x[:, :, :, :, 3:4]) * anchors[:, :, :, :, 1:2] / (
            h * downsample_ratio)
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
