# 给数据集标注增加COCO数据集必须的关键字如， iscrowd和num_keypoints
from my_coco_tools import COCO
import json
import numpy as np
from tqdm import tqdm
import os
import cv2

def get_num_keypoints(kpts):
    kpts_np = np.array(kpts).reshape((-1, 3))
    num_kepoints = int(sum(kpts_np[:, 2] > 0))
    return num_kepoints

def add_keys(ann_file, img_dir):
    with open(ann_file, 'r') as fd:
        json_dict = json.load(fd)
    
    for ann_info in tqdm(json_dict['annotations'], desc=ann_file):
        ann_info['iscrowd'] = 0   # 是否为拥挤人群
        ann_info['area'] = round(0.8 * ann_info['bbox'][2] * ann_info['bbox'][3], 1)   # 面积
        ann_info['segmentation'] = []
        ann_info['num_keypoints'] = get_num_keypoints(ann_info['keypoints'])  # 有效关键点的个数

    for img_info in tqdm(json_dict['images'], desc=img_dir):
        img_file = os.path.join(img_dir, img_info['file_name'])
        img = cv2.imread(img_file)
        height, width = img.shape[:2]
        img_info['width'] = width
        img_info['height'] = height
    
    with open(ann_file, 'w') as fd:
        json.dump(json_dict, fd, indent=4)


if __name__ == '__main__':
    ann_files = ["annotations/posetrack_train.json", "annotations/posetrack_val.json"]
    img_path = ["images/train", "images/val"]
    for file, img_dir in zip(ann_files, img_path):
        add_keys(ann_file=file, img_dir=img_dir)