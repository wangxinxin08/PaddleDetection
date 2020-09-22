import argparse
import os
import sys
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
from copy import deepcopy, copy

sys.path.append('.')
from ppdet.data.transform.operators import *
from ppdet.utils.colormap import colormap

def draw_box(im, bbox, label, cnames):
    draw_thickness = min(im.size) // 160
    draw = ImageDraw.Draw(im)
    catid2color = {}
    color_list = colormap(rgb=True)[:40]
    for dt, catid in zip(bbox, label):
        bbox = dt
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        if catid not in catid2color:
            idx = np.random.randint(len(color_list))
            catid2color[catid] = color_list[idx]
        color = tuple(catid2color[catid])

        # draw bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=draw_thickness,
            fill=color)

        # draw label
        text = "{}".format(cnames[catid])
        tw, th = draw.textsize(text)
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))

    return im


def draw_coco_json(coco_json_file, image_dir, output_dir, sample=10):
    coco = COCO(coco_json_file)
    img_ids = coco.getImgIds()
    if sample == -1:
        sample = len(img_ids)
    sample = min(len(img_ids), sample)
    img_ids = np.random.choice(img_ids, size=sample, replace=False)
    cat_ids = coco.getCatIds()
    catid2clsid = dict({
            catid: i
            for i, catid in enumerate(cat_ids)
        })
    cnames = [coco.loadCats(catid)[0]['name'] for catid in cat_ids]
    for img_id in img_ids:
        img_id = int(img_id)
        anno_ids = coco.getAnnIds(img_id, iscrowd=False)
        insts = coco.loadAnns(anno_ids)
        boxes = []
        for inst in insts:
            x, y, w, h = inst['bbox']
            clsid = catid2clsid[inst['category_id']]
            score = 1.0
            boxes.append([float(clsid), score, x, y, x + w, y + h])

        boxes = np.array(boxes)[:, 2:6]
        img_path = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(image_dir, img_path)
        im = Image.open(img_path)
        sample = {'image': np.array(im), 'gt_bbox': boxes}
        im1 = draw_box(im, boxes)
        im1.save(os.path.join(output_dir, '{:08d}.jpg'.format(img_id)))
        ops = [Rotate(10, 1.0), RandomRotate(10, 0.0), Shear((10, 10)), RandomShear(10, 10), Translate(0.05), RandomTranslate(0.1, (-0.1, 0.1)), Scale((1.2, 1.2)), RandomScale((0.8, 1.2), (0.8, 1.2))]
        names = ['r10', 'rr10', 's10', 'rs10', 't10', 'rt10', 's1.2', 'rs1.2']
        for op, name in zip(ops, names):
            sample1 = op(deepcopy(sample))
            im2 = Image.fromarray(sample1['image'], 'RGB')
            im2 = draw_box(im2, sample1['gt_bbox'])
            im2.save(os.path.join(output_dir, '{:08d}_{}.jpg'.format(img_id, name)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('draw a coco json')
    parser.add_argument('--root', default='dataset/coco', type=str, help='root dir')
    parser.add_argument('--image_dir', default='val2017', type=str, help='image dir')
    parser.add_argument('--json_name', default='annotations/instances_val2017_debug_139.json', type=str, help='json file name')
    parser.add_argument('--output_dir', default='vis', type=str, help='visualize path')
    parser.add_argument('--sample', default=1, type=int, help='sample num')
    args = parser.parse_args()
    root, sample = args.root, args.sample
    image_dir = os.path.join(root, args.image_dir)
    output_dir = os.path.join(root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    coco_json_file = os.path.join(root, args.json_name)
    draw_coco_json(coco_json_file, image_dir, output_dir, sample)
