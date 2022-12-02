#
# 作用：将MPII的Matlab标注文件转换成COCO数据集标注格式的Json文件
#
import os
import numpy as np
from PIL import Image
import cv2
import json
from pathlib import Path
from scipy.io import loadmat
from tqdm import tqdm

colors = [(50, 0, 0), (100, 0, 0), (150, 0, 0), (200, 0, 0), (250, 0, 0), (0, 50, 0), (0, 100, 0), (0, 150, 0),
          (0, 200, 0),(0, 250, 0), (0, 0, 50), (0, 0, 100), (0, 0, 150), (0, 0, 200), (0, 0, 250), (50, 50, 50)]
num_joints = 16  # MPII

def get_bbox(joints, head_box, img_size, scale_factor=(0.1, 0.2)):
    """获取近似人体框"""
    joints_vis = joints[joints[:, 2] > 0]
    x1, y1 = joints_vis.min(axis=0)[:2]
    x2, y2 = joints_vis.max(axis=0)[:2]
    x_pad, y_pad = scale_factor[0] * (x2 - x1), scale_factor[1] * (y2 - y1)
    
    # 如果头部框在边缘则取头部框的坐标
    x1 = head_box[0] if x1 > head_box[0] else max(0, x1 - x_pad)
    right_wrist, left_wrist = joints[10], joints[15]
    if (y1 == left_wrist[1] or y1 == right_wrist[1]) and (left_wrist[1] - right_wrist[1] < (y2-y1)*0.2):  # wrist点是最高点，且两点接近，表示举起双手姿势。
        y1 = max(0, y1 - y_pad * 0.4)
    elif y1 == joints[9, 1]:   # 头顶点9就是最高点，站立姿势
        y1 = head_box[1] if y1 > head_box[1] else y1
    else:
        y1 = head_box[1] if y1 > head_box[1] else max(0, y1 - y_pad)
    x2 = head_box[2] if x2 < head_box[2] else min(img_size[0] - 1, x2 + x_pad)
    right_ankle, left_ankle = joints[0], joints[5]
    right_knee, left_knee = joints[1], joints[4]
    if (y2 == left_ankle[1] or y2 == right_ankle[1]) and (left_ankle[1] - right_ankle[1] < (y2-y1)*0.2):  # ankle点是最低点 且 两点接近表示站立状态。
        y2 = min(img_size[1] - 1, y2 + y_pad * 0.4)
    elif (y2 == left_knee[1] or y2 == right_knee[1]) and (left_knee[1] - right_knee[1] < (y2-y1)*0.1):  # knee点为最低点，且两点接近表示站立状态。
        y2 = min(img_size[1] - 1, y2 + y_pad * 1.5)
    else:
        y2 = head_box[3] if y2 < head_box[3] else min(img_size[1] - 1, y2 + y_pad*1.1)
    assert y2 > y1 and x2 > x1, f"{[x1, y1, x2, y2]=}"
    return [x1, y1, x2 - x1, y2 - y1]  # [lx, ly, w, h]


def save_joints(mat_path, image_path, save_path, json_format=False, visualization=False):
    """
    Args:
        mat_path (str): MPII数据集mat文件的路径
        image_path (str): MPII数据集图片根目录
        save_path (str): txt文件保存的路径
        json_format (bool, optional): 输出为一个COCO数据集标注格式的json文件. Defaults to False.
        visualization (bool, optional): 可视化标注信息. Defaults to False.
    """
    save_path =  Path(save_path)
    mpii_images = Path(image_path)
    mat = loadmat(mat_path)
    wh_list = []  # 用于统计图片宽高的范围 
    num_samples = dict(train=0, test=0, missing=0)  # 样本数：训练|测试|丢失
    img_id, ann_id = 0, 0  # 图片id, 标注id, 对于多人数据集 一个img_id 对应多个 ann_id
    images, annotations = [], []

    for i, (anno, train_flag) in tqdm(enumerate(zip(mat['RELEASE']['annolist'][0, 0][0], 
                                               mat['RELEASE']['img_train'][0, 0][0])), desc="Processing "):
        img_name = anno['image']['name'][0, 0][0]
        img_path = mpii_images.joinpath(img_name)
        if not os.path.exists(img_path):
            # print("error, not exist", img_path)
            num_samples['missing'] += 1
            continue
        img = Image.open(img_path)
        img = np.array(img,dtype=np.uint8)
        img1 = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        height, width, _ = img1.shape
        wh_list.append([width, height])

        train_flag = int(train_flag)
        if not train_flag:
            # print(f"{train_flag=}, skip this image ({img_name}) due to no annotations.")
            num_samples['test'] += 1
            continue
        num_samples['train'] += 1
        
        if json_format:
            images.append(dict(id=img_id, file_name=img_name, width=width, height=height))

        if 'x1' in str(anno['annorect'].dtype):
            head_box = zip(
                [x1[0, 0] for x1 in anno['annorect']['x1'][0]],
                [y1[0, 0] for y1 in anno['annorect']['y1'][0]],
                [x2[0, 0] for x2 in anno['annorect']['x2'][0]],
                [y2[0, 0] for y2 in anno['annorect']['y2'][0]])

        if 'annopoints' in str(anno['annorect'].dtype):
            # only one person
            annopoints = anno['annorect']['annopoints'][0]
            head_x1s = anno['annorect']['x1'][0]
            head_y1s = anno['annorect']['y1'][0]
            head_x2s = anno['annorect']['x2'][0]
            head_y2s = anno['annorect']['y2'][0]

            image_write = ""
            for annopoint, head_x1, head_y1, head_x2, head_y2 in zip(
                    annopoints, head_x1s, head_y1s, head_x2s, head_y2s):
                if annopoint != []:
                    head_box = [float(head_x1[0, 0]),
                                 float(head_y1[0, 0]),
                                 float(head_x2[0, 0]),
                                 float(head_y2[0, 0])]

                    # joint coordinates
                    annopoint = annopoint['point'][0, 0]
                    j_id = [str(j_i[0, 0]) for j_i in annopoint['id'][0]]
                    if len(j_id) == 0:
                        if json_format:
                            images.pop()
                            img_id -= 1
                        continue
                    # assert len(j_id) == num_joints, f"{len(j_id)=} < {num_joints=}!!!"
                    x = [x[0, 0] for x in annopoint['x'][0]]
                    y = [y[0, 0] for y in annopoint['y'][0]]
                    joint_pos = {str(nj): [0, 0] for nj in range(num_joints)}
                    for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
                        joint_pos[str(_j_id)] = [float(_x), float(_y)]

                    # visiblity list , 在MPII中如果有标注则表明关键点在图片上，vis=0表示遮挡，vis=1表示可见。
                    if 'is_visible' in str(annopoint.dtype):
                        vis = [v[0] if v.size > 0 else [0] for v in annopoint['is_visible'][0]]
                        vis = dict([(k, int(v[0])) if len(v) > 0 else v for k, v in zip(j_id, vis)])
                        for nj in range(num_joints):
                            if str(nj) not in vis.keys():
                                vis[str(nj)] = -1
                    else:
                        # vis = None
                        vis = dict()
                        for nj in range(num_joints):
                            vis[str(nj)] = 0 if str(nj) not in j_id else 1

                    if visualization:
                        img1 = cv2.rectangle(img1, (int(head_box[0]), int(head_box[1])),
                                            (int(head_box[2]), int(head_box[3])),
                                            color=(255, 0, 0), thickness=4)
                        for j in range(len(joint_pos)):
                            px, py = int(joint_pos[str(list(joint_pos.keys())[j])][0]), int(joint_pos[str(list(joint_pos.keys())[j])][1])
                            img1 = cv2.circle(img1, (px, py), 10, colors[j], thickness=-1)           
                            img1 = cv2.putText(img1, str(j), (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[j], 1)

                    if not json_format:
                        data = {
                            'filename': img_name,
                            'train': train_flag,
                            'head_box': head_box,
                            'is_visible': vis,
                            'joint_pos': joint_pos
                            }
                        image_write = image_write + str(data) + "\n"
                    else:
                        # print(f"{i=}\t{vis=}\n")
                        # vis += 1 转换成 coco格式 -1, 0, 1 => 0, 1, 2 == 不在视野中，遮挡， 可见
                        vis = np.array([[vis[str(nj)]] for nj in range(num_joints)]) + 1 if isinstance(vis, dict) else vis
                        joints = np.array([joint_pos[str(nj)] for nj in range(num_joints)])
                        joints = np.concatenate([joints, vis], axis=1)
                        bbox = get_bbox(joints, head_box, img_size=(width, height), scale_factor=(0.1, 0.2))
                        area = bbox[2] * bbox[3]
                        annotations.append(dict(category_id=1,
                                                image_id=img_id,
                                                id=ann_id,
                                                num_keypoints=num_joints,
                                                keypoints=joints.flatten().tolist(),
                                                iscrowd=0,
                                                bbox=bbox,
                                                head_box=head_box,
                                                area=area,
                                                segmentation=[]))

                        ann_id += 1
                    if visualization:
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                        img1 = cv2.rectangle(img1, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), thickness=2)
            
            if not json_format:
                txt_file = save_path.joinpath(img_name.split(".")[0]+".txt")
                with txt_file.open(mode='w') as fd:
                    fd.write(image_write)
                # print(f"Save text file => {img_name.split('.')[0]}.txt")
            else:
                img_id += 1

        if visualization:
            cv2.imshow("img_video", img1)
            cv2.waitKey()
            cv2.destroyAllWindows()
    
    if json_format:
        json_dict = dict(
            images=images,
            annotations=annotations,
            categories=[{"supercategory": "person",
                         "id": 1,
                         "name": "person",
                         "keypoints":["right_ankle", "right_knee", "right_hip", "left_hip", "left_knee", "left_ankle", "pelvis", "neck", "upper_neck", "head_top",
                                      "right_wrist", "right_elbow", "right_shoulder", "left_shoulder", "left_elbow", "left_wrist"],
                         "skeleton": [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [6, 7], [7, 12], [12, 11], [11, 10], [7, 13], [13, 14], [14, 15], [7, 8], [8, 9]]
                         }]
            )
        json_file = save_path.joinpath("mpii.json")
        print(f"Save json file => {json_file}")
        with json_file.open(mode='w') as fd:
            json.dump(json_dict, fd, indent=4)

    wh_list = np.array(wh_list)
    wh_min = wh_list.min(axis=0)
    wh_max = wh_list.max(axis=0)
    print(f"The range of image size => w: {wh_min[0]}~{wh_max[0]}, h: {wh_min[1]}~{wh_max[1]}")
    print(f"The number of image sample is {num_samples}")

if __name__ == "__main__":
    save_joints(mat_path="mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat",
                image_path="images", 
                save_path="./",
                json_format=True,
                visualization=False)
