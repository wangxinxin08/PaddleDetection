import argparse
import os
import sys
from pycocotools.coco import COCO
import numpy as np
from PIL import Image

sys.path.append('.')
from ppdet.data.transform.operators import *

def draw_box(im, np_boxes):
    draw_thickness = min(im.size) // 320
    draw = ImageDraw.Draw(im)

    for dt in np_boxes:
        bbox = dt
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        color = (0, 255, 0)

        # draw bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=draw_thickness,
            fill=color)

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
        sample = {'image': im, 'gt_bbox': boxes}
        sample = Rotate(10, 1.0)(sample)
        im = draw_box(sample['image'], sample['gt_bbox'])
        im.save(os.path.join(output_dir, '{:08d}.jpg'.format(img_id)))

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
