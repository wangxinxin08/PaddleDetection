import json
import os
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
from multiprocessing import Pool

def merge(train_json, val_json):

    # load train json
    with open(train_json, 'r') as f:
        train_dicts = json.load(f)
    
    # load val json
    with open(val_json, 'r') as f:
        val_dicts = json.load(f)
    
    # trainval 
    trainval_dicts = dict()
    trainval_dicts['categories'] = train_dicts['categories']
    for k in ['images', 'annotations']:
        trainval_dicts[k] = train_dicts[k] + val_dicts[k]
    
    trainval_file = os.path.join(os.path.split(train_json)[0], 'trainval.json')
    with open(trainval_file, 'w') as f:
        json.dump(trainval_dicts, f)
    
    del train_dicts, val_dicts

    return trainval_dicts, trainval_file

def sample_once(trainval_coco, trainval_dicts, annoids2idx, new_idxes, i):
    try:
        d = trainval_dicts['images'][i]
        annos = []
        for anno_id in trainval_coco.getAnnIds(int(d['id'])):
            annos.append(trainval_dicts['annotations'][annoids2idx[anno_id]])
        is_train = i in new_idxes
    except Exception as e:
        print(e)
        raise('error occurs: {}'.format(e))
    return i, d, annos, is_train


def sample(trainval_dicts, trainval_file):
    # trainval_coco
    trainval_coco = COCO(trainval_file)
    # idxes -> imgids
    idxes, imgids = [], []
    for i, d in enumerate(trainval_dicts['images']):
        idxes.append(i)
        imgids.append(d['id'])
    print(len(idxes))
    # annoid2idx
    annoids2idx = dict()
    for i, d in enumerate(trainval_dicts['annotations']):
        annoids2idx[d['id']] = i

    # get freq
    nums = []
    for imgid in imgids:
        anno_ids = trainval_coco.getAnnIds(int(imgid))
        nums.append(len(anno_ids))
    
    nums = np.array(nums)
    total = nums.sum()
    start, end = nums.min(), nums.max()
    for i in range(start, end + 1, 10):
        mask = (nums >= i) & (nums < i + 10)
        nums[mask] = len(mask)
    nums = total / nums
    nums = nums / nums.sum()

    # sample idxes
    new_idxes = np.random.choice(idxes, size=int(len(idxes) * 0.6), replace=False, p=nums)
    new_train_dicts = {'images': [], 'annotations': [], 'categories': trainval_dicts['categories']}
    new_val_dicts = {'images': [], 'annotations': [], 'categories': trainval_dicts['categories']}

    def callback(res):
        i, d, annos, is_train = res
        if is_train:
            new_train_dicts['images'].append(d)
            new_train_dicts['annotations'].extend(annos)
        else:
            new_val_dicts['images'].append(d)
            new_val_dicts['annotations'].extend(annos)
        

    for i in tqdm(idxes):
        res = sample_once(trainval_coco, trainval_dicts, annoids2idx, new_idxes, i)
        callback(res)

    # with Pool(8) as pool:
    #     for i in idxes:
    #         pool.apply_async(sample_once, 
    #         args=(trainval_coco, trainval_dicts, annoids2idx, new_idxes, i), callback=callback)
    
    # save json
    new_train_file = os.path.join(os.path.split(train_json)[0], 'new_train.json')
    new_val_file = os.path.join(os.path.split(train_json)[0], 'new_val.json')
    with open(new_train_file, 'w') as f:
        json.dump(new_train_dicts, f)
    
    with open(new_val_file, 'w') as f:
        json.dump(new_val_dicts, f)

if __name__ == "__main__":
    train_json = 'dataset/SKU110K_fixed/annotations/train.json'
    val_json = 'dataset/SKU110K_fixed/annotations/val.json'
    trainval_dicts, trainval_file = merge(train_json, val_json)
    sample(trainval_dicts, trainval_file)





    
