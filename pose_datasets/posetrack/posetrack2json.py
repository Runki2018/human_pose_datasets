import cv2
import matplotlib.pyplot as plt 
from pathlib import Path 
import os 
import numpy as np 
from my_coco_tools import COCO
from tqdm import tqdm
import shutil
import json

"""
    该脚本用于提取PoseTrack中有效的样本,并输出为COCO文件目录格式
    |-posetrack
        |-annotations
            |-posetrack_train.json
            |-posetrack_bal.json
        |-images
            |-tarin
                |-1.jpg
                |-2.jpg
                |-...
            |-val
                |-1.jpg
                |-2.jpg
                |-...
"""

categories = [
        {
            "supercategory": "person", 
            "id": 1, 
            "name": "person", 
            "keypoints": ['nose', 'head_bottom', 'head_top',
                        'left_ear', 'right_ear', 
                        'left_shoulder', 'right_shoulder',
                        'left_elbow', 'right_elbow',
                        'left_wrist', 'right_wrist',
                        'left_hip', 'right_hip',
                        'left_knee','right_knee', 
                        'left_ankle', 'right_ankle'], 
            "skeleton": [[16, 14], [14, 12], [17, 15],
                        [15, 13], [12, 13], [6, 12],
                        [7, 13], [6, 7], [6, 8], 
                        [7, 9], [8, 10], [9, 11],
                        [2, 3], [1, 2], [1, 3], 
                        [2, 4], [3, 5], [4, 6], [5, 7]]
        }
    ]
skeleton = np.array(categories[0]['skeleton']) - 1

def get_num_keypoints(kpts):
    kpts_np = np.array(kpts).reshape((-1, 3))
    num_kepoints = int(sum(kpts_np[:, 2] > 0))
    return num_kepoints

def _int(v):
    return [int(vi) for vi in v]

def draw_pose(img, kpts, bbox, bbox_head, thickness=2, color=(0, 255, 0), skeleton=None):
    """
        用于在图片上绘制姿态信息
        img: cv2.imread
        kpt: ann_info['keypoints']
        bbox:ann_info['bbox']
        kpt_type: 'coco' or 'mixed', default to 'coco'
    """
    kpts = np.array(kpts).reshape((-1, 3))
    # 1 keypoints
    for x, y, vis in kpts:
        if vis > 0:
            cv2.circle(img, (int(x), int(y)), thickness+2, color, cv2.FILLED)

    # 2 bbox
    left_top = bbox[:2]
    right_bottom = bbox[0] + bbox[2], bbox[1] + bbox[3]
    cv2.rectangle(img, _int(left_top), _int(right_bottom), (0, 0, 255), thickness=thickness)  

    # 2 bbox_head
    left_top = bbox_head[:2]
    right_bottom = bbox_head[0] + bbox_head[2], bbox_head[1] + bbox_head[3]
    cv2.rectangle(img, _int(left_top), _int(right_bottom), (255, 0, 255), thickness=thickness)  

    if skeleton is not None:
        for pairs in skeleton:
            if kpts[pairs[0], 2] > 0 and kpts[pairs[1], 2] > 0:
                pt1 = kpts[pairs[0], :2].astype(np.int32).tolist()
                pt2 = kpts[pairs[1], :2].astype(np.int32).tolist()
                cv2.line(img, pt1, pt2, color, thickness)
    return img


def load_data(file, img_dir, image_id=0, annotations_id=0, display=False):
    """
        加载数据, 保留有标注的图片和相应的标注信息
        file: json标注文件, 每个json文件对于一个视频,每个视频只有中间片段有标注
        img_dir: 视频截图目录
        image_id: 保留有效图片的id
        annotations_id: 有效图片上的实例id
        display: 可视化图片标注
    """
    if display:
        print(f"Press 'q' to quit!")
    img_file_list = []
    images, annotations = [], []
    if img_dir:
        dataset = COCO(str(file))
        imgIds = dataset.getImgIds()
        for img_id in imgIds:
        # for img_id in tqdm(imgIds, desc=f"{file.stem}"):
            img_info = dataset.loadImgs(img_id)[0]
            img_file = img_dir.parents[2].joinpath(img_info['file_name'])

            if img_file.exists():
                img = cv2.imread(str(img_file))
                annIds = dataset.getAnnIds(imgIds=img_id)
                if len(annIds) == 0:
                    continue
                
                isEmpty = True
                for ann_id in annIds:
                    ann_info = dataset.loadAnns(ann_id)[0]
                    kpts = ann_info.get('keypoints', False)
                    bbox = ann_info.get('bbox', False)
                    bbox_head = ann_info.get('bbox_head', False)
                    if kpts and bbox and bbox_head:
                        isEmpty = False
                        ann_info['category_id'] = 1
                        ann_info['image_id'] = image_id
                        ann_info['id'] = annotations_id
                        ann_info['iscrowd'] = 0   # 是否为拥挤人群
                        ann_info['num_keypoints'] = get_num_keypoints(ann_info['keypoints'])  # 有效关键点的个数
                        annotations_id += 1
                        annotations.append(ann_info)
                        if display:
                            img = draw_pose(img, kpts, bbox, bbox_head, skeleton=skeleton)

                if not isEmpty:
                    height, width = img.shape[:2]
                    img_info['width'] = width
                    img_info['height'] = height
                    img_info['file_name'] = str(image_id) + '.jpg'
                    img_info['id'] = image_id
                    image_id += 1
                    images.append(img_info)
                    img_file_list.append((img_file, img_info['file_name']))  # (src_img_path, dst_img_name)
                if display:
                    cv2.imshow("image", img)
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        return 
            else:
                print(f"img_file not exists!")
        cv2.destroyAllWindows()
    return images, annotations, image_id, annotations_id, img_file_list

def generate_dataset(data_type, img_path, ann_path):
    assert data_type in ['train', 'val']
    img_path = Path(img_path).joinpath(data_type)
    ann_path = Path(ann_path).joinpath(data_type)
    assert ann_path.exists() and img_path.exists()
    ann_files = list(ann_path.glob("*.json"))
    img_dirs = {d.stem:d for d in img_path.glob("*") if d.is_dir()}
    
    # 筛选出有效样本
    img_id, ann_id = 0, 0
    images, annotations, file_list = [], [], []
    for file in tqdm(ann_files, desc=f"Scaning {data_type} data"):
        img_dir = img_dirs.get(file.stem, False)
        # print(f"{img_dir=}")
        imgs, anns, img_id, ann_id, img_file_list = \
            load_data(file, img_dir, img_id, ann_id, display=False)

        images += imgs
        annotations += anns
        file_list += img_file_list

    # 设置输出路径
    ann_path = Path("posetrack/annotations")    # 保存json文件的目录
    img_path = Path(f"posetrack/images/{data_type}")   # 保存图片文件的目录
    if not ann_path.exists():
        ann_path.mkdir(mode=0o777, parents=True, exist_ok=True)
    if not img_path.exists():
        img_path.mkdir(mode=0o777, parents=True, exist_ok=True)
    output_json_file = ann_path.joinpath(f"posetrack_{data_type}.json")
    
    # 生成数据集json文件
    json_dict = dict(images=images, annotations=annotations, categories=categories)
    with output_json_file.open('w') as fd:
        json.dump(json_dict, fd, indent=4)
    print(f"Save json file => {str(output_json_file)}")
    
    # 拷贝图片
    for src_file, dst_file_name in tqdm(file_list, desc=f"Copy images to {str(img_path)}"):
        dst_file = img_path.joinpath(dst_file_name)
        shutil.copyfile(str(src_file), str(dst_file))


if __name__ == '__main__':
    data_type = "train"   # train, val, test
    for data_type in ['train', 'val']:
        generate_dataset(data_type,
                         img_path="posetrack18_images/images",
                         ann_path="posetrack_data/annotations")
