# author: huangzhiyong
# date: 2022/10/19

import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import json
import mediapipe as mp 
import math
from pathlib import  Path
from tqdm import tqdm
import os
import shutil
from get_video_pose import pose_structure
from argparse import ArgumentParser


def vis_annotations(img, bbox, kpts):
    for i in range(0, len(kpts), 3):
        cv2.circle(img, (int(kpts[i]), int(kpts[i+1])), 6, (0, 255, 0), cv2.FILLED)
    left_top = bbox[:2]
    right_bottom = bbox[0] + bbox[2], bbox[1] + bbox[3]
    # rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img
    cv2.rectangle(img, left_top, right_bottom, (0, 255, 0), thickness=2)  
    return img


def check_keypoints(kpts, width, height, thr=0.2):
    kpts = np.array(kpts).reshape((-1, 3))
    kpts[(kpts[:, 0] > width) & (kpts[:, 0] < 0)] = [0, 0, 0]  # 去除超出边界的关键点
    kpts[(kpts[:, 1] > height) & (kpts[:, 1] < 0)] = [0, 0, 0]  # 去除超出边界的关键点
    kpts[kpts[:, 2] < 0.2] = [0, 0, 0]  # 得分低于阈值的去除
    num_keypoints = np.sum(kpts[:, 2] > 0)
    return kpts.flatten().tolist(), num_keypoints


def merge_remained_video_data(video_root='videos',              # 存放视频的根目录
                        img_root='images',                # 存放高质量可视化图片的根目录,先使用window图片查看器去除低质量检测图片
                        ann_root='annotations',           # 存放标注文件的更目录
                        output_root='merge_datasets',     # 输出图片和标注文件的根目录。根据img_root中保留的图片,生成相应帧图片和标注文件
                        show=False,                       # 可视化帧与标注,查看是否正确匹配
                        ):
    """
    生成最终数据集
    1. 使用Windows图片查看器去除检测结果质量低的图片,快捷键: '→'键 下一张, 'Del'键 删除当前图片
    2. 使用merge_remained_video_data函数,根据保留下来的高质量检测结果,生成相应帧图片和标注文件
    """
    # 输入目录
    video_root = Path(video_root)
    img_dirs = os.listdir(img_root)   # 图片目录中我们手工删除检测质量较差的结果
    ann_dirs = os.listdir(ann_root)
    img_dirs = [d for d in img_dirs if os.path.isdir(os.path.join(img_root, d))]
    ann_dirs = [d for d in ann_dirs if os.path.isdir(os.path.join(img_root, d))]
    # print(f"{img_dirs=}\t{ann_dirs=}")
    dirs = list(set(img_dirs) and set(ann_dirs))  # 共同视频目录
    video_names = {name.split('.')[0]:name for name in os.listdir(video_root) if name.split('.')[0] in dirs}
    # print(f"common directories: {dirs}")

    # 输出目录
    output_root = Path(output_root)
    out_img_path = output_root.joinpath('images')
    out_ann_path = output_root.joinpath('annotations')
    if not out_img_path.exists():
        out_img_path.mkdir(mode=777, parents=True, exist_ok=True)
    if not out_ann_path.exists():
        out_ann_path.mkdir(mode=777, parents=True, exist_ok=True)

    print(f"Save images\t\t=> {out_img_path}")
    print(f"Save annotations\t=> {out_ann_path}")

    # 开始比对,如果合并保留下来的高质量检测结果。
    img_id, ann_id = 0, 0
    images, annotations = [], []   # Json标注文件中的
    num_images_dir = len(dirs)
    for idx, _dir in enumerate(dirs):
        try:
            img_files = os.listdir(os.path.join(img_root, _dir))
            ann_files = os.listdir(os.path.join(ann_root, _dir))
            hash_table = {a.strip('.json'):a for a in ann_files if '.json' in a}
            cap = cv2.VideoCapture(str(video_root.joinpath(video_names[_dir])))

            # 按帧序号升序排序
            img_files = sorted(img_files, key=lambda x:int(x.strip('.jpg')))

            # 匹配和生成标注
            for file in tqdm(img_files, desc=f"{idx+1}/{num_images_dir}: {_dir}"):
                file_name = file.strip('.jpg')
                ann_file = hash_table.get(file_name, None)
                if ann_file != None:
                    with open(os.path.join(ann_root, _dir, ann_file), 'r') as fd:
                        ann_dict = json.load(fd)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(file_name))
                    success, img = cap.read()
                    if not success:
                        continue

                    dst_file = str(img_id) + '.jpg'
                    dst_img = out_img_path / dst_file
                    cv2.imwrite(str(dst_img), img)

                    kpts, num_keypoints = check_keypoints(ann_dict['keypoints'], ann_dict['width'], ann_dict['height'], thr=0.2)
                    images.append(dict(
                        id=img_id,
                        file_name=dst_file,
                        width=ann_dict['width'],
                        height=ann_dict['height'],
                    ))
                    annotations.append(dict(
                        id=ann_id,
                        image_id=img_id,
                        category_id=1,
                        bbox=ann_dict['bbox'],
                        area=int(ann_dict['area']),
                        num_keypoints=int(num_keypoints),
                        keypoints=kpts,
                        iscrowd=0,
                    ))
                    img_id += 1
                    ann_id += 1

                    if show:
                        vis_annotations(img, ann_dict['bbox'], ann_dict['keypoints'])
                        cv2.imshow('image', img)
                        cv2.waitKey(1)
        finally:
            cv2.destroyAllWindows()
            cap.release()

    # 保存COCO数据集标注格式的JSON标注文件
    json_file = out_ann_path.joinpath("video_keypoints.json")
    print(f"Saving annotation file\t=> {json_file}")
    with json_file.open('w') as fd:
        json_dict = dict(
            categories=[{
                "supercategory": "person",
                "id": 1,
                "name": "person",
                "keypoints": [
                    "left_shoulder", "right_shoulder",
                    "left_elbow","right_elbow",
                    "left_wrist","right_wrist",
                    "left_hip","right_hip",
                    "left_knee","right_knee",
                    "left_ankle","right_ankle"
                ],
                "skeleton": [[0, 1], [6, 7],
                            [0, 2], [1, 3],
                            [2, 4], [3, 5],
                            [0, 6], [1, 7],
                            [6, 8], [7, 9],
                            [8, 10], [9, 11]]
            }],
            images=images,
            annotations=annotations)
        json.dump(json_dict, fd, indent=4)
    print(f"the number of images is {len(images)}")
    print(f"Done!")


def merge_remained_image_data(input_root='videos',        # 存放原图片的目录
                        img_root='images',                # 存放高质量可视化图片的根目录,先使用window图片查看器去除低质量检测图片
                        ann_root='annotations',           # 存放标注文件的更目录
                        output_root='merge_datasets',     # 输出图片和标注文件的根目录。根据img_root中保留的图片,生成相应帧图片和标注文件
                        show=False,                       # 可视化帧与标注,查看是否正确匹配
                        ):
    """
    生成最终数据集
    1. 使用Windows图片查看器去除检测结果质量低的可视化图片,快捷键: '→'键 下一张, 'Del'键 删除当前图片
    2. 使用merge_remained_image_data函数,根据保留下来的高质量检测结果, 生成相应原图和标注文件
    """
    # 输入目录
    input_root = Path(input_root)
    img_dirs = os.listdir(img_root)   # 注意：图片目录中我们要手工删除检测质量较差的结果, 只保留高质量的可视化图片
    ann_dirs = os.listdir(ann_root)
    img_dirs = [d for d in img_dirs if os.path.isdir(os.path.join(img_root, d))]
    ann_dirs = [d for d in ann_dirs if os.path.isdir(os.path.join(img_root, d))]
    # print(f"{img_dirs=}\t{ann_dirs=}")
    dirs = list(set(img_dirs) and set(ann_dirs))  # 共同目录
    # video_names = {name.split('.')[0]:name for name in os.listdir(input_root) if name.split('.')[0] in dirs}
    # print(f"common directories: {dirs}")

    # 输出目录
    output_root = Path(output_root)
    out_img_path = output_root.joinpath('images')
    out_ann_path = output_root.joinpath('annotations')
    if not out_img_path.exists():
        out_img_path.mkdir(mode=777, parents=True, exist_ok=True)
    if not out_ann_path.exists():
        out_ann_path.mkdir(mode=777, parents=True, exist_ok=True)
    print(f"Save images\t\t=> {out_img_path}")
    print(f"Save annotations\t=> {out_ann_path}")

    # 开始比对,如果合并保留下来的高质量检测结果。
    img_id, ann_id = 0, 0
    images, annotations = [], []   # Json标注文件中的
    num_images_dir = len(dirs)
    IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.tif', '.bmp', '.tif', '.tiff', '.dng', '.webp', '.mpo']
    for idx, _dir in enumerate(dirs):
        try:
            img_files = list(Path(img_root).joinpath(_dir).glob('*'))
            ann_files = list(Path(ann_root).joinpath(_dir).glob('*'))
            hash_table = {a.stem:a.name for a in ann_files if a.suffix == '.json'}

            # 可视化图片按修改时间升序排序，有利于按顺序处理同一批数据
            print(f"Sorting image files...")
            img_files = sorted(img_files, key=lambda x:Path(x).stat().st_mtime)  

            # 匹配和生成标注
            for file in tqdm(img_files, desc=f"{idx}/{num_images_dir}: {_dir}"):
                # print(f"{file=}\t{file.suffix=}")
                if file.suffix not in IMAGE_FORMATS:
                    continue
                ann_file = hash_table.get(file.stem, None)
                # print(f"{ann_file=}")
                if ann_file != None:
                    with open(os.path.join(ann_root, _dir, ann_file), 'r') as fd:
                        ann_dict = json.load(fd)
                    original_file = input_root.joinpath(file.name) 
                    img = cv2.imread(str(original_file))
                    if img is None:
                        raise ValueError(f"ERROR: cv2.imread => {file}")

                    dst_file = str(img_id) + '.jpg'
                    dst_img = out_img_path / dst_file
                    cv2.imwrite(str(dst_img), img)

                    kpts, num_keypoints = check_keypoints(ann_dict['keypoints'], ann_dict['width'], ann_dict['height'], thr=0.2)
                    images.append(dict(
                        id=img_id,
                        file_name=dst_file,
                        width=ann_dict['width'],
                        height=ann_dict['height'],
                    ))
                    annotations.append(dict(
                        id=ann_id,
                        image_id=img_id,
                        category_id=1,
                        bbox=ann_dict['bbox'],
                        area=int(ann_dict['area']),
                        num_keypoints=int(num_keypoints),
                        keypoints=kpts,
                        iscrowd=0,
                    ))
                    img_id += 1
                    ann_id += 1

                    if show:
                        vis_annotations(img, ann_dict['bbox'], ann_dict['keypoints'])
                        cv2.imshow('image', img)
                        cv2.waitKey(1)
        finally:
            cv2.destroyAllWindows()

    # 保存COCO数据集标注格式的JSON标注文件
    json_file = out_ann_path.joinpath("video_keypoints.json")
    print(f"Saving annotation file\t=> {json_file}")
    with json_file.open('w') as fd:
        json_dict = dict(
            categories=[{
                "supercategory": "person",
                "id": 1,
                "name": "person",
                "keypoints": [
                    "left_shoulder", "right_shoulder",
                    "left_elbow","right_elbow",
                    "left_wrist","right_wrist",
                    "left_hip","right_hip",
                    "left_knee","right_knee",
                    "left_ankle","right_ankle"
                ],
                "skeleton": [[0, 1], [6, 7],
                            [0, 2], [1, 3],
                            [2, 4], [3, 5],
                            [0, 6], [1, 7],
                            [6, 8], [7, 9],
                            [8, 10], [9, 11]]
            }],
            images=images,
            annotations=annotations)
        json.dump(json_dict, fd, indent=4)
    num_images = len(list(input_root.glob('*')))
    print(f"images: high-qulity | low-qulity | full  => {len(images)} | {num_images-len(images)} | {num_images}")
    print(f"Done!")


if __name__ == '__main__':
    # 步骤2：生成最终数据集
    # 1. 使用Windows图片查看器去除检测结果质量低的图片，快捷键: '→'键 下一张, 'Del'键 删除当前图片
    # 2. 完成步骤1以及去除低质量帧后，使用merge_remained_data函数，根据保留下来的高质量检测结果，生成相应帧图片和标注文件
    parser = ArgumentParser()
    parser.add_argument('--input-dir', '-i', type=str, default='videos/', help='A directory to loading original videos or images')  # 原视频或原图的目录
    parser.add_argument('--image-type', '-t', action='store_true', help='input data type is images, default to videos type')   # 默认输入目录下读入视频，加上该参数是读入图片
    parser.add_argument('--output-dir', type=str, default='merge_datasets', help='A directory to output dataset')
    parser.add_argument('--show', action='store_true', help='To visualize the processing for checking pose on each frame.')
    parser.add_argument('--img-root', type=str, default='vis_images', help='A directory to visual image files')
    parser.add_argument('--ann-root', type=str, default='annotations', help='A directory to annotation files')
    args = parser.parse_args()

    if args.image_type:
        merge_remained_image_data(
            input_root=args.input_dir,
            img_root=args.img_root,
            ann_root=args.ann_root,
            output_root=args.output_dir,
            show=args.show,
        )
    else:
        merge_remained_video_data(
            video_root=args.input_dir,
            img_root=args.img_root,
            ann_root=args.ann_root,
            output_root=args.output_dir,
            show=args.show,
        )
