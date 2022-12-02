# 这个文件通过可视化图片和标注检查生成的数据集是否正确。
import numpy as np
import cv2
import json
from video_pose_utils import pose_structure
from my_coco_tools import COCO
from pathlib import Path


def load_json(json_file):
    with open(json_file, 'r') as fd:
        json_dict = json.load(fd)
    return json_dict


if __name__ == '__main__':
    # dataset = load_json(json_file="merge_datasets/annotations/video_keypoints.json")
    cv2.namedWindow("check up", cv2.WINDOW_KEEPRATIO)
    img_path = Path("yizhi_train_datasets/images/")
    coco = COCO(annotation_file="yizhi_train_datasets/annotations/video_keypoints.json")
    imgIds = coco.getImgIds()
    try:
        for img_id in imgIds:
            img_info = coco.loadImgs(img_id)[0]
            img_file = str(img_path.joinpath(img_info['file_name']))
            img = cv2.imread(img_file)

            annIds = coco.getAnnIds(imgIds=img_id)
            for ann_id in annIds:
                ann_info = coco.loadAnns(ann_id)[0]
                img = coco.draw_pose(img, kpts=ann_info['keypoints'], bbox=ann_info['bbox'], kpt_type='pose12')

            cv2.imshow("check up", img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
