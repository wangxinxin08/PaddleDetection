#!/usr/bin/env python
# coding: utf-8
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

import argparse
import os
import csv
import numpy as np
import cv2
import sys
import shutil
import json
from tqdm import tqdm


class SKU2COCO:
    def __init__(self):
        self.img2id = dict()
        self.cat2id = dict()
        self.obj_id = -1
        self.image_paths = dict()
        self.anno_dicts = dict()

    def get_annonation(self, obj_id, img_id, cat_id, points):
        annotation = dict()
        seg_points = np.asarray(points).copy()
        seg_points[1, :] = np.asarray(points)[2, :]
        seg_points[2, :] = np.asarray(points)[1, :]
        annotation['segmentation'] = [seg_points.flatten().tolist()]
        annotation['iscrowd'] = 0
        annotation['image_id'] = img_id
        annotation['bbox'] = list(
            map(float, [
                points[0][0], points[0][1], points[1][0] - points[0][0], points[1][
                    1] - points[0][1]
            ]))
        annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
        annotation['category_id'] = cat_id
        annotation['id'] = obj_id
        return annotation

    def get_image(self, img_id, w, h, img_name):
        image = dict()
        image['id'] = img_id
        image['width'] = w
        image['height'] = h
        image['file_name'] = img_name
        return image

    def get_categories(self):
        categories = []
        for name, cat_id in self.cat2id.items():
            categories.append({'id': cat_id, 'name': name, 'supercategory': 'retail'})
        return categories

    def save_annos(self, root, anno_dir):
        for split, anno_dict in self.anno_dicts.items():
            anno_file = os.path.join(root, os.path.join(anno_dir, split + '.json'))
            with open(anno_file, 'w') as f:
                json.dump(anno_dict, f)


    def forward_once(self, root, image_dir, split, label_file):
        self.image_paths[split] = []
        self.anno_dicts[split] = dict()
        annotations = []
        images = []
        with open(label_file, 'r') as f:
            lines = [line.strip() for line in f]
        for line in tqdm(lines, desc=split):
            self.obj_id += 1
            img_name, x1, y1, x2, y2, cat, w, h = line.split(',')
            x1, y1, x2, y2, w, h = map(int, [x1, y1, x2, y2, w, h])
            if img_name not in self.img2id:
                self.image_paths[split].append(
                    os.path.join(root, os.path.join(image_dir, img_name))
                )
                self.img2id[img_name] = len(self.img2id) # start from 0
                img_id = self.img2id[img_name]
                images.append(self.get_image(img_id, w, h, img_name))
            # get annonation
            obj_id = self.obj_id
            if cat not in self.cat2id:
                self.cat2id[cat] = len(self.cat2id) + 1 # start from 1
            cat_id = self.cat2id[cat]
            if x1 > x2 or y1 > y2:
                raise ValueError('bbox coordinate error: x1 > x2 or y1 > y2')
            points = [[x1, y1], [x2, y2], [x1, y2], [x2, y1]]
            annotations.append(self.get_annonation(
                obj_id, img_id, cat_id, points 
            ))
        self.anno_dicts[split]['images'] = images
        self.anno_dicts[split]['annotations'] = annotations

    def __call__(self, root, image_dir, anno_dir, label_files, clean=False):
        for split, label_file in label_files.items():
            self.forward_once(root, image_dir, split, label_file)
        
        categories = self.get_categories()

        for split in label_files:
            self.anno_dicts[split]['categories'] = categories

        # save annotation files
        self.save_annos(root, anno_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('sku110 to coco')
    parser.add_argument('--root', default='SKU110K_fixed', type=str, help='dataset root dir')
    parser.add_argument('--image_dir', default='images', type=str, help='image dir')
    parser.add_argument('--anno_dir', default='annotations', type=str, help='annotation dir')
    parser.add_argument('--label_files', default='annotations_train.csv,annotations_val.csv,annotations_test.csv', type=str, help='label files:train,val,test')
    parser.add_argument('--splits', default='train,val,test', type=str, help='split:train,val,test, corresponding to label_files')
    parser.add_argument('--clean', action='store_true', help='whether clean the data dir')
    args = parser.parse_args()
    clean = args.clean
    root, image_dir, anno_dir = args.root, args.image_dir, args.anno_dir
    label_files = dict()
    for split, label_file in zip(args.splits.split(','), args.label_files.split(',')):
        label_file = os.path.join(root, os.path.join(anno_dir, label_file))
        label_files[split] = label_file

    SKU2COCO()(root, image_dir, anno_dir, label_files, clean=clean)





            

    