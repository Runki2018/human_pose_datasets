import glob
import os
from scipy.io import loadmat
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import json
from tqdm import tqdm

colors = [(50, 0, 0), (100, 0, 0), (150, 0, 0), (200, 0, 0), (250, 0, 0), (0, 50, 0), (0, 100, 0),(0, 150, 0), (0, 200, 0),(0, 250, 0), (0, 0, 50), (0, 0, 100), (0, 0, 150), (0, 0, 200), (0, 0, 250), (50, 50, 50)]

def save_joints(mat_path,image_path,save_path, json_format=False, vis=False):
    """
    mat_path: LSP数据集(2000张图片, 每张图片只标注一个人)中mat文件的路径
    image_path: 图片根目录
    save_path: mat文件转换成txt文件要保存的路径
    json_format: 将标注转换成一个COCO标注格式的json文件。
    vis: 可视化标注信息
    """
    img_id, ann_id = 0, 0  # 由于一张图片中只有一个目标，所以数值上图片id == 目标id
    images, annotations = [], []
    save_path = Path(save_path)
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    joints = loadmat(mat_path)
    joints = joints["joints"].transpose(2,0,1)
    joints = joints[:,:2,:]

    num = 0
    for img_path in  tqdm(glob.glob("%s/*.jpg" %image_path), desc="Processing "):
        img_name = img_path.split("\\")[-1]
        img = Image.open(img_path)
        img = np.array(img,dtype=np.uint8)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        cen_points = joints[num,...]
        points_num = cen_points.shape[-1]
        point_dict = {}
        for points_ in range(points_num):
            point_x = cen_points[0,points_]
            point_y = cen_points[1,points_]
            point_dict[str(points_)] = [int(point_x),int(point_y)]
            if vis:
                img = cv2.circle(img, (int(point_x), int(point_y)), 5, colors[points_], 
                                thickness=-1)
                img = cv2.putText(img, str(points_),
                                (int(point_x) + 10, int(point_y)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, colors[points_], 1)
        if not json_format:
            txt_file = save_path.joinpath(img_name.split(".")[0]+".txt")
            with txt_file.open(mode='w') as img_txt:
                img_txt.write(str(point_dict))
        else:
            h, w = img.shape[:2]
            kp = []
            for i in range(14):
                kp += [point_dict[str(i)][0], point_dict[str(i)][1], 1]   # x, y, vis
            kpt = np.array(kp).reshape((-1, 3))
            lt = kpt.min(axis=0)[:2]
            rb = kpt.max(axis=0)[:2]
            # 近似计算人体框
            x1, y1 = lt / 2
            x2, y2 = (rb + np.array([w, h])) / 2          
            bbox = [x1, y1, x2-x1, y2-y1]
            area = bbox[2] * bbox[3]

            images.append(dict(id=img_id, file_name=img_name, width=w, height=h))
            annotations.append(dict(category_id=1, image_id=img_id, id=ann_id, num_keypoints=14,
                                    keypoints=list(kp), iscrowd=0, bbox=bbox, area=area, segmentation=[]))
            img_id += 1
            ann_id += 1
            if vis:
                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), thickness=2)
        num += 1
        if vis:
            cv2.imshow("img",img)
            cv2.waitKey()

    if json_format:
        json_dict = dict(
            images=images,
            annotations=annotations,
            categories=[{"supercategory": "person",
                         "id": 1,
                         "name": "person",
                         "keypoints":["right_ankle", "right_knee", "right_hip", "left_hip", "left_knee", "left_ankle", "right_wrist",
                                      "right_elbow", "right_shoulder", "left_shoulder", "left_elbow","left_wrist", "neck", "head_top"],
                         "skeleton": [[12, 13], [9, 12], [9, 10], [10, 11], [8, 12], [8, 7], [7, 6],
                                      [9, 3], [8, 2], [3, 4], [4, 5], [2, 1], [1, 0]]
                         }]
            )
        json_file = save_path.joinpath("lsp.json")
        print(f"save json file => {json_file}.")
        with json_file.open(mode='w') as fd:
            json.dump(json_dict, fd, indent=4)


if __name__ == '__main__':
    save_joints(mat_path="joints.mat", 
                image_path="images/",
                save_path="./annotations",
                json_format=True, 
                vis=False)
