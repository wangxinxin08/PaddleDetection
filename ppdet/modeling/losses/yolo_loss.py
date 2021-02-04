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

from paddle import fluid
import paddle
from ppdet.core.workspace import register
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import logging
import numpy as np
logger = logging.getLogger(__name__)

__all__ = ['YOLOv3Loss']


@register
class YOLOv3Loss(object):
    """
    Combined loss for YOLOv3 network

    Args:
        train_batch_size (int): training batch size
        ignore_thresh (float): threshold to ignore confidence loss
        label_smooth (bool): whether to use label smoothing
        use_fine_grained_loss (bool): whether use fine grained YOLOv3 loss
                                      instead of fluid.layers.yolov3_loss
    """
    __inject__ = ['iou_loss', 'iou_aware_loss']
    __shared__ = ['use_fine_grained_loss', 'train_batch_size']

    def __init__(
            self,
            train_batch_size=8,
            batch_size=-1,  # stub for backward compatable
            ignore_thresh=0.7,
            label_smooth=True,
            use_fine_grained_loss=False,
            iou_loss=None,
            iou_aware_loss=None,
            downsample=[32, 16, 8],
            scale_x_y=1.,
            match_score=False):
        self._train_batch_size = train_batch_size
        self._ignore_thresh = ignore_thresh
        self._label_smooth = label_smooth
        self._use_fine_grained_loss = use_fine_grained_loss
        self._iou_loss = iou_loss
        self._iou_aware_loss = iou_aware_loss
        self.downsample = downsample
        self.scale_x_y = scale_x_y
        self.match_score = match_score

        if batch_size != -1:
            logger.warn(
                "config YOLOv3Loss.batch_size is deprecated, "
                "training batch size should be set by TrainReader.batch_size")

    def __call__(self, outputs, gt_box, gt_label, gt_score, targets, anchors,
                 anchor_masks, mask_anchors, num_classes, prefix_name, second_head=False):
        if self._use_fine_grained_loss:
            return self._get_fine_grained_loss(
                outputs, targets, gt_box, gt_label, self._train_batch_size, num_classes,
                mask_anchors, self._ignore_thresh, second_head)
        else:
            losses = []
            for i, output in enumerate(outputs):
                scale_x_y = self.scale_x_y if not isinstance(
                    self.scale_x_y, Sequence) else self.scale_x_y[i]
                anchor_mask = anchor_masks[i]
                loss = fluid.layers.yolov3_loss(
                    x=output,
                    gt_box=gt_box,
                    gt_label=gt_label,
                    gt_score=gt_score,
                    anchors=anchors,
                    anchor_mask=anchor_mask,
                    class_num=num_classes,
                    ignore_thresh=self._ignore_thresh,
                    downsample_ratio=self.downsample[i],
                    use_label_smooth=self._label_smooth,
                    scale_x_y=scale_x_y,
                    name=prefix_name + "yolo_loss" + str(i))

                losses.append(fluid.layers.reduce_mean(loss))

            return {'loss': sum(losses)}

    def _get_fine_grained_loss(self,
                               outputs,
                               targets,
                               gt_box,
                               gt_label,
                               train_batch_size,
                               num_classes,
                               mask_anchors,
                               ignore_thresh,
                               second_head,
                               eps=1.e-10):
        """
        Calculate fine grained YOLOv3 loss

        Args:
            outputs ([Variables]): List of Variables, output of backbone stages
            targets ([Variables]): List of Variables, The targets for yolo
                                   loss calculatation.
            gt_box (Variable): The ground-truth boudding boxes.
            train_batch_size (int): The training batch size
            num_classes (int): class num of dataset
            mask_anchors ([[float]]): list of anchors in each output layer
            ignore_thresh (float): prediction bbox overlap any gt_box greater
                                   than ignore_thresh, objectness loss will
                                   be ignored.

        Returns:
            Type: dict
                xy_loss (Variable): YOLOv3 (x, y) coordinates loss
                wh_loss (Variable): YOLOv3 (w, h) coordinates loss
                obj_loss (Variable): YOLOv3 objectness score loss
                cls_loss (Variable): YOLOv3 classification loss

        """

        assert len(outputs) == len(targets), \
            "YOLOv3 output layer number not equal target number"
        loss_xys, loss_whs, loss_objs, loss_clss = [], [], [], []
        if self._iou_loss is not None:
            loss_ious = []
        if self._iou_aware_loss is not None:
            loss_iou_awares = []
        for i, (output, target,
                anchors) in enumerate(zip(outputs, targets, mask_anchors)):
            downsample = self.downsample[i]
            an_num = len(anchors) // 2
            if self._iou_aware_loss is not None:
                ioup, output = self._split_ioup(output, an_num, num_classes)
            x, y, w, h, obj, cls = self._split_output(output, an_num,
                                                      num_classes)
            tx, ty, tw, th, tscale, tobj, tcls = self._split_target(target)

            # POTO: modify tobj
            tobj = self._poto(output, cls, obj, tobj, gt_label, gt_box, self._train_batch_size, anchors,
                       num_classes, downsample, self._ignore_thresh, scale_x_y=1.0)

            tscale_tobj = tscale * tobj

            scale_x_y = self.scale_x_y if not isinstance(
                self.scale_x_y, Sequence) else self.scale_x_y[i]

            #if (abs(scale_x_y - 1.0) < eps):
            if False:
                loss_x = fluid.layers.sigmoid_cross_entropy_with_logits(
                    x, tx) * tscale_tobj
                loss_x = fluid.layers.reduce_sum(loss_x, dim=[1, 2, 3])
                loss_y = fluid.layers.sigmoid_cross_entropy_with_logits(
                    y, ty) * tscale_tobj
                loss_y = fluid.layers.reduce_sum(loss_y, dim=[1, 2, 3])
            else:
                # dx = scale_x_y * fluid.layers.sigmoid(x) - 0.5 * (scale_x_y -
                #                                                   1.0)
                # dy = scale_x_y * fluid.layers.sigmoid(y) - 0.5 * (scale_x_y -
                #                                                   1.0)
                loss_x = fluid.layers.abs(x - tx) * tscale_tobj
                loss_x = fluid.layers.reduce_sum(loss_x, dim=[1, 2, 3])
                loss_y = fluid.layers.abs(y - ty) * tscale_tobj
                loss_y = fluid.layers.reduce_sum(loss_y, dim=[1, 2, 3])

            # NOTE: we refined loss function of (w, h) as L1Loss
            #fluid.layers.Print(gt_box)
            loss_w = fluid.layers.abs(w - tw) * tscale_tobj
            loss_w = fluid.layers.reduce_sum(loss_w, dim=[1, 2, 3])
            loss_h = fluid.layers.abs(h - th) * tscale_tobj
            loss_h = fluid.layers.reduce_sum(loss_h, dim=[1, 2, 3])
            if self._iou_loss is not None:
                loss_iou = self._iou_loss(x, y, w, h, tx, ty, tw, th, anchors,
                                          downsample, self._train_batch_size,
                                          scale_x_y)
                loss_iou = loss_iou * tscale_tobj
                loss_iou = fluid.layers.reduce_sum(loss_iou, dim=[1, 2, 3])
                loss_ious.append(fluid.layers.reduce_mean(loss_iou))

            if self._iou_aware_loss is not None:
                loss_iou_aware = self._iou_aware_loss(
                    ioup, x, y, w, h, tx, ty, tw, th, anchors, downsample,
                    self._train_batch_size, scale_x_y)
                loss_iou_aware = loss_iou_aware * tobj
                loss_iou_aware = fluid.layers.reduce_sum(
                    loss_iou_aware, dim=[1, 2, 3])
                loss_iou_awares.append(fluid.layers.reduce_mean(loss_iou_aware))
            loss_obj_pos, loss_obj_neg = self._calc_obj_loss(
                output, obj, tobj, gt_box, self._train_batch_size, anchors,
                num_classes, downsample, self._ignore_thresh, scale_x_y)

            loss_cls = fluid.layers.sigmoid_cross_entropy_with_logits(cls, tcls)
            #fluid.layers.Print(fluid.layers.reshape(x=loss_cls,shape=[-1,80]))
            #fluid.layers.Print(fluid.layers.reshape(x=tobj,shape=[-1,1]))

            # fg_num = fluid.layers.reduce_sum(tcls)                                                                                                         
            # fg_num = fluid.layers.cast(fg_num, dtype="int32")                                                                                              
            # tcls = fluid.layers.argmax(x=tcls, axis=4) + 1                                                                                                 
            # tcls = fluid.layers.cast(tcls, dtype="int32")                                                                                                  
            # tcls = fluid.layers.unsqueeze(tcls, axes=[4])                                                                                                  
            # r_cls=fluid.layers.reshape(x=cls,shape=[-1,fluid.layers.shape(cls)[-1]])                                                                       
            # r_tcls=fluid.layers.reshape(x=tcls,shape=[-1,1])                                                                                               
            # r_tobj=fluid.layers.reshape(tobj,(-1,1))                                                                                                       
            # loss_cls = fluid.layers.sigmoid_focal_loss(r_cls, r_tcls, fg_num)                                                                              
            # loss_cls = fluid.layers.elementwise_mul(loss_cls, r_tobj, axis=0)                                                                              
            # loss_cls = fluid.layers.reduce_sum(loss_cls, dim=[0,1]) 

            loss_cls = fluid.layers.elementwise_mul(loss_cls, tobj, axis=0)
            loss_cls = fluid.layers.reduce_sum(loss_cls, dim=[1, 2, 3, 4])

            loss_xys.append(fluid.layers.reduce_mean(loss_x + loss_y))
            loss_whs.append(fluid.layers.reduce_mean(loss_w + loss_h))
            loss_objs.append(
                fluid.layers.reduce_mean(loss_obj_pos + loss_obj_neg))
            loss_clss.append(fluid.layers.reduce_mean(loss_cls))
        
        if second_head==True:
            losses_all = {
            "loss_xy2": fluid.layers.sum(loss_xys),
            "loss_wh2": fluid.layers.sum(loss_whs),
            "loss_obj2": fluid.layers.sum(loss_objs),
            "loss_cls2": fluid.layers.sum(loss_clss),
            }
            if self._iou_loss is not None:
                losses_all["loss_iou2"] = fluid.layers.sum(loss_ious)
            if self._iou_aware_loss is not None:
                losses_all["loss_iou_aware2"] = fluid.layers.sum(loss_iou_awares)
        else:
            losses_all = {
            "loss_xy": fluid.layers.sum(loss_xys),
            "loss_wh": fluid.layers.sum(loss_whs),
            "loss_obj": fluid.layers.sum(loss_objs),
            "loss_cls": fluid.layers.sum(loss_clss),
            }
            if self._iou_loss is not None:
                losses_all["loss_iou"] = fluid.layers.sum(loss_ious)
            if self._iou_aware_loss is not None:
                losses_all["loss_iou_aware"] = fluid.layers.sum(loss_iou_awares)
        return losses_all

    def _split_ioup(self, output, an_num, num_classes):
        """
        Split output feature map to output, predicted iou
        along channel dimension
        """
        ioup = fluid.layers.slice(output, axes=[1], starts=[0], ends=[an_num])
        ioup = fluid.layers.sigmoid(ioup)
        oriout = fluid.layers.slice(
            output,
            axes=[1],
            starts=[an_num],
            ends=[an_num * (num_classes + 6)])
        return (ioup, oriout)

    def _split_output(self, output, an_num, num_classes):
        """
        Split output feature map to x, y, w, h, objectness, classification
        along channel dimension
        """
        x = fluid.layers.strided_slice(
            output,
            axes=[1],
            starts=[0],
            ends=[output.shape[1]],
            strides=[5 + num_classes])
        y = fluid.layers.strided_slice(
            output,
            axes=[1],
            starts=[1],
            ends=[output.shape[1]],
            strides=[5 + num_classes])
        w = fluid.layers.strided_slice(
            output,
            axes=[1],
            starts=[2],
            ends=[output.shape[1]],
            strides=[5 + num_classes])
        h = fluid.layers.strided_slice(
            output,
            axes=[1],
            starts=[3],
            ends=[output.shape[1]],
            strides=[5 + num_classes])
        obj = fluid.layers.strided_slice(
            output,
            axes=[1],
            starts=[4],
            ends=[output.shape[1]],
            strides=[5 + num_classes])
        clss = []
        stride = output.shape[1] // an_num
        for m in range(an_num):
            clss.append(
                fluid.layers.slice(
                    output,
                    axes=[1],
                    starts=[stride * m + 5],
                    ends=[stride * m + 5 + num_classes]))
        cls = fluid.layers.transpose(
            fluid.layers.stack(
                clss, axis=1), perm=[0, 1, 3, 4, 2])

        return (x, y, w, h, obj, cls)

    def _split_target(self, target):
        """
        split target to x, y, w, h, objectness, classification
        along dimension 2

        target is in shape [N, an_num, 6 + class_num, H, W]
        """
        tx = target[:, :, 0, :, :]
        ty = target[:, :, 1, :, :]
        tw = target[:, :, 2, :, :]
        th = target[:, :, 3, :, :]

        tscale = target[:, :, 4, :, :]
        tobj = target[:, :, 5, :, :]

        tcls = fluid.layers.transpose(
            target[:, :, 6:, :, :], perm=[0, 1, 3, 4, 2])
        tcls.stop_gradient = True

        return (tx, ty, tw, th, tscale, tobj, tcls)

    def _poto(self, output, cls, obj, tobj, gt_label, gt_box, batch_size, anchors,
                       num_classes, downsample, ignore_thresh, scale_x_y):
        #return a new tobj
        one_hot_label = fluid.one_hot(gt_label,depth=81,allow_out_of_range=False)[:,:,1:]
        #1. calculate reg_loss_matrix
        bbox, prob = self.yolo_box(
            x=output,
            img_size=fluid.layers.ones(
                shape=[batch_size, 2], dtype="int32"),
            anchors=anchors,
            class_num=num_classes,
            conf_thresh=0.,
            downsample_ratio=downsample,
            clip_bbox=False,
            scale_x_y=scale_x_y)
        
        # 2. split pred bbox and gt bbox by sample, calculate IoU between pred bbox
        #    and gt bbox in each sample
        if batch_size > 1:
            preds = fluid.layers.split(bbox, batch_size, dim=0)
            gts = fluid.layers.split(gt_box, batch_size, dim=0)
        else:
            preds = [bbox]
            gts = [gt_box]
            probs = [prob]
        ious = []
        for pred, gt in zip(preds, gts):

            def box_xywh2xyxy(box):
                x = box[:, 0]
                y = box[:, 1]
                w = box[:, 2]
                h = box[:, 3]
                return fluid.layers.stack(
                    [
                        x - w / 2.,
                        y - h / 2.,
                        x + w / 2.,
                        y + h / 2.,
                    ], axis=1)

            pred = fluid.layers.squeeze(pred, axes=[0])
            gt = box_xywh2xyxy(fluid.layers.squeeze(gt, axes=[0]))
            ious.append(fluid.layers.iou_similarity(pred, gt))

        iou = fluid.layers.stack(ious, axis=0)
        cls_matrix_list = []
        pred_cls = fluid.layers.reshape(cls, (batch_size, -1, num_classes))
        for b in range(batch_size):
            pred_matrix_per_batch = []
            for i in range(50):
                pred_matrix_per_batch.append(pred_cls[b,:,gt_label[b,i]])
            cls_loss_per_batch = fluid.layers.stack(pred_matrix_per_batch, axis=1)
            cls_matrix_list.append(cls_loss_per_batch)
        pred_matrix = fluid.layers.stack(cls_matrix_list, axis=0)
        quality = fluid.layers.pow(iou,factor=0.8) * fluid.layers.pow(fluid.layers.sigmoid(pred_matrix),factor=0.2)
        spatial_prior = fluid.layers.reshape(tobj, shape=[batch_size,-1,1])
        ex_spatial_prior = fluid.layers.expand(spatial_prior, expand_times=[1,1,50])
        quality = fluid.layers.elementwise_mul(quality, ex_spatial_prior)
        quality_transposed = fluid.layers.transpose(quality, perm=[0, 2, 1])
        top1_values, top1_indices = fluid.layers.topk(quality_transposed, 1)

        an_num = fluid.layers.shape(pred_cls)[1]
        gt_num = fluid.layers.shape(one_hot_label)[1]
        update_ones = fluid.layers.ones(shape=[batch_size*50],dtype='float32')
        index = fluid.layers.squeeze(input=top1_indices,axes=[2])
        index = fluid.layers.reshape(index, [-1,1])

        tmp = fluid.layers.arange(0,batch_size, dtype='int64')
        tmp = fluid.layers.reshape(tmp, shape=[batch_size, 1])
        tmp = fluid.layers.expand(tmp,[1, gt_num])
        tmp = fluid.layers.reshape(tmp, [-1,1])
        idx = fluid.layers.concat(input=[tmp,index],axis=-1)
        out = fluid.layers.scatter_nd(idx, update_ones, shape=[batch_size, an_num])
        poto_mask = fluid.layers.cast(out > 0., dtype="float32")
        poto_mask = fluid.layers.reshape(poto_mask,shape=fluid.layers.shape(tobj))
        '''
        # get target by loss
        reg_loss_matrix = -fluid.layers.log(iou+0.000001)
        
        pred_cls = fluid.layers.reshape(cls, (batch_size, -1, num_classes))
        an_num = fluid.layers.shape(pred_cls)[1]
        gt_num = fluid.layers.shape(one_hot_label)[1]
        one_hot_label = fluid.layers.reshape(one_hot_label, shape = [batch_size, 1, gt_num, num_classes])
        ex_one_hot_label = fluid.layers.expand(one_hot_label, expand_times=[1, an_num, 1, 1])

        pred_cls = fluid.layers.reshape(pred_cls, shape=[batch_size, an_num, 1, num_classes])
        ex_pred_cls = fluid.layers.expand(pred_cls, expand_times=[1, 1, gt_num, 1])
        ex_cls_matrix = fluid.layers.sigmoid_cross_entropy_with_logits(ex_pred_cls, ex_one_hot_label)


        cls_matrix_list = []
        #fluid.layers.Print(gt_label[0])
        for b in range(batch_size):
            cls_matrix_per_batch = []
            for i in range(50):
                #fluid.layers.Print(ex_cls_matrix[b,:,i,gt_label[b,i]])
                cls_matrix_per_batch.append(ex_cls_matrix[b,:,i,gt_label[b,i]])
            cls_loss_per_batch = fluid.layers.stack(cls_matrix_per_batch, axis=1)
            cls_matrix_list.append(cls_loss_per_batch)
        
            #fluid.layers.Print(cls_loss_per_batch)
        
        cls_matrix = fluid.layers.stack(cls_matrix_list, axis=0)
        loss_matrix = fluid.layers.pow(reg_loss_matrix,factor=0.8) * fluid.layers.pow(cls_matrix,factor=0.2)
        spatial_prior = fluid.layers.reshape(tobj, shape=[batch_size,-1,1])
        ex_spatial_prior = fluid.layers.expand(spatial_prior, expand_times=[1,1,50])
        loss_transposed = fluid.layers.transpose(loss_matrix, perm=[0, 2, 1])
        top1_values, top1_indices = fluid.layers.topk(-loss_transposed, 1)
        loss_matrix = fluid.layers.elementwise_mul(-loss_matrix, ex_spatial_prior)

        minimal_loss_list = []
        #for b in range(batch_size):
        input_zeros = fluid.layers.zeros_like(an_num)
        update_ones = fluid.layers.ones(shape=[batch_size*50],dtype='float32')
        index = fluid.layers.squeeze(input=top1_indices,axes=[2])
        index = fluid.layers.reshape(index, [-1,1])

        tmp = fluid.layers.arange(0,batch_size, dtype='int64')
        tmp = fluid.layers.reshape(tmp, shape=[batch_size, 1])
        tmp = fluid.layers.expand(tmp,[1, gt_num])
        tmp = fluid.layers.reshape(tmp, [-1,1])
        idx = fluid.layers.concat(input=[tmp,index],axis=-1)
        #fluid.layers.Print(idx)
        #fluid.layers.Print(tmp)
            #index_transposed = fluid.layers.transpose(index, perm=[1, 0])
            #fluid.layers.Print(index)
            #fluid.layers.Print(update_ones)
        out = fluid.layers.scatter_nd(idx, update_ones, shape=[batch_size, an_num])
        poto_mask = fluid.layers.cast(out > 0., dtype="float32")
        poto_mask = fluid.layers.reshape(poto_mask,shape=fluid.layers.shape(tobj))
        '''
        return tobj*poto_mask

    def _calc_obj_loss(self, output, obj, tobj, gt_box, batch_size, anchors,
                       num_classes, downsample, ignore_thresh, scale_x_y):
        # A prediction bbox overlap any gt_bbox over ignore_thresh, 
        # objectness loss will be ignored, process as follows:

        # 1. get pred bbox, which is same with YOLOv3 infer mode, use yolo_box here
        # NOTE: img_size is set as 1.0 to get noramlized pred bbox
        bbox, prob = self.yolo_box(
            x=output,
            img_size=fluid.layers.ones(
                shape=[batch_size, 2], dtype="int32"),
            anchors=anchors,
            class_num=num_classes,
            conf_thresh=0.,
            downsample_ratio=downsample,
            clip_bbox=False,
            scale_x_y=scale_x_y)
        # 2. split pred bbox and gt bbox by sample, calculate IoU between pred bbox
        #    and gt bbox in each sample
        if batch_size > 1:
            preds = fluid.layers.split(bbox, batch_size, dim=0)
            gts = fluid.layers.split(gt_box, batch_size, dim=0)
        else:
            preds = [bbox]
            gts = [gt_box]
            probs = [prob]
            #fluid.layers.Print(probs)
        ious = []
        for pred, gt in zip(preds, gts):

            def box_xywh2xyxy(box):
                x = box[:, 0]
                y = box[:, 1]
                w = box[:, 2]
                h = box[:, 3]
                return fluid.layers.stack(
                    [
                        x - w / 2.,
                        y - h / 2.,
                        x + w / 2.,
                        y + h / 2.,
                    ], axis=1)

            pred = fluid.layers.squeeze(pred, axes=[0])
            gt = box_xywh2xyxy(fluid.layers.squeeze(gt, axes=[0]))
            ious.append(fluid.layers.iou_similarity(pred, gt))
        iou = fluid.layers.stack(ious, axis=0)
        #fluid.layers.Print(iou)
        # 3. Get iou_mask by IoU between gt bbox and prediction bbox,
        #    Get obj_mask by tobj(holds gt_score), calculate objectness loss

        max_iou = fluid.layers.reduce_max(iou, dim=-1)
        iou_mask = fluid.layers.cast(max_iou <= ignore_thresh, dtype="float32")
        if self.match_score:
            max_prob = fluid.layers.reduce_max(prob, dim=-1)
            iou_mask = iou_mask * fluid.layers.cast(
                max_prob <= 0.25, dtype="float32")
        output_shape = fluid.layers.shape(output)
        an_num = len(anchors) // 2
        iou_mask = fluid.layers.reshape(iou_mask, (-1, an_num, output_shape[2],
                                                   output_shape[3]))
        iou_mask.stop_gradient = True

        # NOTE: tobj holds gt_score, obj_mask holds object existence mask
        obj_mask = fluid.layers.cast(tobj > 0., dtype="float32")
        obj_mask.stop_gradient = True

        # For positive objectness grids, objectness loss should be calculated
        # For negative objectness grids, objectness loss is calculated only iou_mask == 1.0
        loss_obj = fluid.layers.sigmoid_cross_entropy_with_logits(obj, obj_mask)
        loss_obj_pos = fluid.layers.reduce_sum(loss_obj * tobj, dim=[1, 2, 3])
        loss_obj_neg = fluid.layers.reduce_sum(
            loss_obj * (1.0 - obj_mask) * iou_mask, dim=[1, 2, 3])
        # focal Loss
        #fg_num = fluid.layers.reduce_sum(obj_mask)
        #fg_num = fluid.layers.cast(fg_num, dtype="int32")
        #r_obj = fluid.layers.reshape(x=obj, shape=[-1,1])
        #r_tobj = fluid.layers.reshape(x=obj_mask, shape=[-1, 1])
        #r_tobj = fluid.layers.cast(r_tobj, dtype="int32")
        #loss_obj = fluid.layers.sigmoid_focal_loss(r_obj, r_tobj, fg_num) * fg_num
        #loss_obj_pos = fluid.layers.reduce_sum(loss_obj * r_tobj)
        #loss_obj_neg = fluid.layers.reduce_sum(loss_obj * (1.0 - r_tobj))
        return loss_obj_pos, loss_obj_neg

    def _make_grid(self, nx, ny):
        yv, xv = fluid.layers.meshgrid([
            fluid.layers.range(0, ny, 1, 'float32'),
            fluid.layers.range(0, nx, 1, 'float32')
        ])
        grid = fluid.layers.stack([xv, yv], axis=2)
        return fluid.layers.reshape(grid, (1, 1, ny, nx, 2))
    
    def _create_tensor_from_numpy(self, numpy_array):
        paddle_array = fluid.layers.create_global_var(
            shape=numpy_array.shape, value=0., dtype=numpy_array.dtype)
        fluid.layers.assign(numpy_array, paddle_array)
        return paddle_array

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