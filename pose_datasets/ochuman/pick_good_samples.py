# 由于OCHuman数据集中存在大量漏标的样本，可以使用该脚本从该数据集中挑选质量高的样本
import cv2
import numpy as np
import os 
from pathlib import Path
import json
import sys
sys.path.insert(0, os.path.abspath('.'))
print(sys.path)
from my_coco_tools import COCO

def pick_samples(annotation_file, image_path, new_img_id=0, new_ann_id=0):
    """
    annotation_file: coco格式json标注文件的路径
    image_path: 图片路径
    new_img_id: 新的标注文件起始图片信息id 
    new_ann_id: 新的标注文件起始标注信息id 
    """
    print("press key 'q' to quit.!")  # 退出
    print("press key 's' to keep current image and coresponding annotations.")    # 保留当前图片
    print("press other keys to skip current image and coresponding annotations.")  # 舍弃当前图片
    # help(COCO)   # 可以把该行注释去掉，了解COCO类的用法
    dataset = COCO(annotation_file=annotation_file)
    imgIds = dataset.getImgIds()
    image_path = Path(image_path)
    images, annotations = [], []
    try:
        for i in imgIds:
            img_info = dataset.loadImgs(i)[0]
            img_info['id'] = new_img_id
            img_file = str(image_path.joinpath(img_info['file_name']))
            img = cv2.imread(img_file)

            annIds = dataset.getAnnIds(imgIds=i)
            ann_list = []
            for j in annIds:
                ann_info = dataset.loadAnns(j)[0]
                ann_info['id'] = new_ann_id
                ann_info['image_id'] = new_img_id
                ann_info['category_id'] = 1
                img = dataset.draw_pose(img, 
                                        kpts=ann_info['keypoints'],
                                        bbox=ann_info['bbox'],
                                        kpt_type='coco')
                ann_list.append(ann_info)
                new_ann_id +=1
            new_img_id += 1

            cv2.imshow("images", img)
            key = cv2.waitKey()
            if key & 0xFF == ord('q'):   # quit
                print("quit!")
                break
            elif key & 0xFF == ord('s'):  # save
                print(f"keep current image, {new_img_id=}")
                images.append(img_info)
                annotations += ann_list
            else:
                print("next")
        return dataset.dataset, images, annotations, new_img_id, new_ann_id
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    annotation_files = ["ochuman/annotations/ochuman_coco_format_test_range_0.00_1.00.json",
                        "ochuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json"]
    image_path = "ochuman/images"
    save_file = "ochuman/annotations/pick_samples.json"
    
    datasets, images, annotations = [], [], []
    new_img_id, new_ann_id = 0, 0
    for af in annotation_files:
        _dataset, _images, _annotations, new_img_id, new_ann_id = \
            pick_samples(af, image_path, new_img_id, new_ann_id)
        
        datasets.append(_dataset)
        images += _images
        annotations += _annotations
    
    for d in datasets:
        print(len(d['images']))
    
    dataset = datasets[0]
    dataset['images'] = images
    dataset['annotations'] = annotations

    with open(save_file, 'w') as fd:
        json.dump(dataset, fd, indent=4)
    print(f" Saved annotation file => {save_file}")


