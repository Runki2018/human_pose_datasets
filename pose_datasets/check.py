#
# 有的数据集可能缺少一些数据项，通过该文件补充缺少的数据项。
#
import json
from pycocotools.coco import COCO
from pathlib import Path
import cv2
from tqdm import tqdm


def check(ann_file, img_path):
    dataset = COCO(ann_file)
    imgIds = dataset.getImgIds()
    cv2.namedWindow("images")
    try:
        for i in imgIds:
            img_info = dataset.loadImgs(i)[0]
            annIds = dataset.getAnnIds(i)
            for j in annIds:
                ann_info = dataset.loadAnns(j)[0]
                if "area" not in ann_info.keys():
                    img_file = str(img_path.joinpath(img_info['file_name']))
                    img = cv2.imread(img_file)
                    cv2.imshow("images", img)
                    print(f"{img_info=}")
                    print(f"{ann_info=}")
                    key = cv2.waitKey()
                    if key & 0xFFFF == ord('q'):
                        return
    finally:
        cv2.destroyAllWindows()

def add_key(ann_file):
    with open(ann_file, 'r') as fd:
        json_dict = json.load(fd)

    for ann_info in tqdm(json_dict['annotations']):
        if "arae" not in ann_info.keys():
            ann_info['area'] = ann_info['bbox'][2] * ann_info['bbox'][3] * 0.8

    with open(ann_file, 'w') as fd:
        json.dump(json_dict, fd, indent=4)

if __name__ == '__main__':
    val_annot_files = [
        "coco/annotations/person_keypoints_val2017.json",
        "ochuman/annotations/ochuman_coco_format_test_range_0.00_1.00.json",
        "yizhi_yt20221028/val/annotations/video_keypoints.json",
        "crowdpose/crowdpose_test.json",
    ]
    # val_json = "/home/huangzhiyong/Project/kapao_with_kp_conf/data/datasets/pose12/annotations/val_pose12.json"
    # img_path = Path("/home/huangzhiyong/Project/kapao_with_kp_conf/data/datasets/pose12/images/val")
    # check(ann_file=val_json, img_path=img_path)
    for val_json in val_annot_files:
        add_key(val_json)
