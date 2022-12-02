# + author: huangzhiyong
# + function: 用于合并多个数据集
#   对于每个数据集的json文件，每个目标仅仅保留12个关键点信息。
#   将每个数据集相应的图片拷贝到指定目录
import numpy as np
import json
from my_coco_tools import COCO
from pathlib import Path
from tqdm import tqdm
import shutil
import os
from datetime import  datetime as dt


class pick_keypoints:
    def __init__(self, src_cat, dst_cat):
        self.src_cat = src_cat
        self.dst_cat = dst_cat
        self.src_cat2id = {k:v for v, k in enumerate(src_cat)}
        self.keep_cat = [self.src_cat2id[cc] for cc in self.dst_cat]

    def dst_keypoints(self, src_keypoints):
        assert 3*len(self.src_cat) == len(src_keypoints)
        src_keypoints = np.array(src_keypoints).reshape((-1, 3))
        dst_keypoints = src_keypoints[self.keep_cat]     
        num_keypoints = int(sum(dst_keypoints[:, 2] > 0))   # 可见点的个数
        return dst_keypoints.flatten().tolist(), num_keypoints
    
def merge_datasets(keep_categories, img_roots, ann_files,
                   save_file="coco_crowd_mpii_lsp_12keypoints.json",
                   copy_img=True, copy_img_dir="all_images"):
    if not isinstance(ann_files, (list, tuple)):
        ann_files = [ann_files]   # ["1.json", "2.json"]
    if not isinstance(img_roots, (list, tuple)):
        img_roots = [img_roots]
    assert len(ann_files) == len(img_roots)
    out_file = Path(save_file)  # 合并数据集和修改关键点后的输出json文件的路径
    copy_img_dir = Path(copy_img_dir)  # 各个数据集的图片都复制到该目录，在copy_img==True时生效
    if not out_file.parent.exists():
        old_mask = os.umask(000)
        out_file.parent.mkdir(mode=0o777, parents=True, exist_ok=True)
        os.umask(old_mask)
    if copy_img and not copy_img_dir.exists():
        old_mask = os.umask(000)
        copy_img_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
        os.umask(old_mask)
    conflict_dict = dict()  #  记录出现过的图片名，防止图片名冲突 

    img_index, ann_index = 0, 0  # 图片和标注编号
    images, annotations = [], []
    record = dict(num_imgs=[], num_anns=[], num_keep=[0] * len(ann_files), wh_range=[])  # 记录每个标注文件的图片总数、标注总数、保留的图片数, 图片宽高范围
    for i, (_img_root, _ann_file) in enumerate(zip(img_roots, ann_files)):
        _img_root = Path(_img_root)  # 图片根目录
        _ann_file = Path(_ann_file)  # json标注文件的路径
        w_range, h_range = [], []

        dataset = COCO(str(_ann_file))
        record['num_imgs'].append(len(dataset.imgs))
        record['num_anns'].append(len(dataset.anns))
        src_cat = dataset.loadCats(1)[0]["keypoints"]
        dst_cat = keep_categories[0]["keypoints"]  # 保留12个关键点

        # 更新图片和标注信息
        pick_kpt = pick_keypoints(src_cat=src_cat, dst_cat=dst_cat)
        imgIds = dataset.getImgIds()
        for img_id in tqdm(imgIds, desc="Processing <<{}>> image ".format(_ann_file.stem)):
            annIds = dataset.getAnnIds(imgIds=img_id)
            img_info = dataset.loadImgs(img_id)[0]
            src_img_file = _img_root.joinpath(img_info['file_name'])
            if not src_img_file.exists():
                continue  # missing
            
            # 记录图片宽高范围
            w_range.append(img_info['width'])
            h_range.append(img_info['height'])
            w_range = [min(w_range), max(w_range)]
            h_range = [min(h_range), max(h_range)]

            # 更新标注信息
            is_person = False
            for ann_id in annIds:
                ann_info = dataset.loadAnns(ann_id)[0]
                if ann_info['category_id'] != 1:
                    continue
                is_person = True
                scr_kpt = ann_info['keypoints']
                dst_kpt, num_keypoints = pick_kpt.dst_keypoints(src_keypoints=scr_kpt)
                ann_info['image_id'] = img_index
                ann_info['id'] = ann_index
                ann_info['keypoints'] = dst_kpt
                ann_info['num_keypoints'] = num_keypoints
                annotations.append(ann_info)
                ann_index += 1

            # 更新图片信息
            if is_person:
                if copy_img:
                    inc = 0  # 防止文件名冲突的自增序号
                    file_name = img_info['file_name']
                    while conflict_dict.get(file_name, False):
                        # print(f"conflict image name: {img_info['file_name']}")
                        file_name = str(inc) + '_conflict_' + img_info['file_name']
                        inc += 1
                    img_info['file_name'] = file_name
                    conflict_dict[img_info['file_name']] = True
                    dst_img_file = copy_img_dir.joinpath(img_info['file_name'])
                    shutil.copyfile(src_img_file, dst_img_file)
                img_info['id'] = img_index
                img_info['from'] = _ann_file.name  # 图片的标注文件来源
                images.append(img_info)  
                img_index += 1
                record['num_keep'][i] += 1

        record['wh_range'].append([w_range, h_range])

    # 打印和保存record日志
    #  record = dict(num_imgs=[], num_anns=[], num_keep=[0] * len(ann_files), wh_range=[])  # 记录每个标注文件的图片总数、标注总数、保留的图片数, 图片宽高范围
    logfile = out_file.with_name("merge_dataset_{}.log".format(dt.strftime(dt.now(), "%Y%m%d%H%M%S")))
    with logfile.open('w') as fd:
        for i in range(len(ann_files)):
            message = f"@{i:3d}, {ann_files[i]:25s}\n \
                =>num_images:\t\t{record['num_imgs'][i]:8d} \n \
                =>num_annotations:\t{record['num_anns'][i]:8d}\n \
                =>num_keep:\t\t{record['num_keep'][i]:8d}\n \
                =>num_missing:\t\t{record['num_imgs'][i]-record['num_keep'][i]:8d}\n \
                =>w_range:\t\t{record['wh_range'][i][0]}\n \
                =>h_range:\t\t{record['wh_range'][i][1]}\n \
                "
            fd.write(message)
            print(message)

        message = f"@all dataset:\n \
                =>num_images:\t\t{sum(record['num_imgs'])} \n \
                =>num_annotations:\t{sum(record['num_anns'])}\n \
                =>num_keep:\t\t{sum(record['num_keep'])}\n \
                =>num_missing:\t\t{sum(record['num_imgs'])-sum(record['num_keep']):8d}\n \
                "
        fd.write(message)
        fd.write(f"Saving log file => {str(logfile)} ...\n")
        fd.write(f"Saving annotation file => {str(out_file)} ...")
        print(message)
        print(f"Saving log file => {str(logfile)} ...\n")
        print(f"Saving annotation file => {str(out_file)} ...")

    # 保存新的标注文件
    out_dict = dict(categories=keep_categories, images=images, annotations=annotations)
    with out_file.open(mode='w') as fd:
        json.dump(out_dict, fd, indent=4)
    print(f"Done!")


if __name__ == '__main__':
    # + 设置各个数据集的图片根目录和标注文件
    # + 使用MPII、CrowdPose、LSP的所有数据，以及COCO train2017进行训练， COCO val2017作为验证集和测试集不变。

    # COCO 17个关键点
    # CrowdPose 14 个关键点
    # LSP 14 个关键点
    # MPII 16 个关键点
    # ...
    # 保留的12个关键点信息如下所示
    keep_categories = [{
        "supercategory": "person",
        "id": 1,
        "name": "person",
        "keypoints": [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        ],
        "skeleton": [[0, 1], [6, 7],
                     [0, 2], [1, 3],
                     [2, 4], [3, 5],
                     [0, 6], [1, 7],
                     [6, 8], [7, 9],
                     [8, 10], [9, 11]]  # 关键点骨骼连线
    }]

    # train --------------------
    train_image_roots = [
        "lsp_dataset/images",
        "mpii/images",
        "crowdpose/images",
        "coco/images/train2017",
        "video_pose/images/train",
        "ochuman/images",
        "fall106_datasets/images",
        "yizhi_yt20221028/train/images",
        "PoseTrack/images/train",
        "PoseTrack/images/val"
    ]
    train_annot_files = [
        "lsp_dataset/annotations/lsp.json",
        "mpii/mpii.json",
        "crowdpose/crowdpose_trainval.json",
        "coco/annotations/person_keypoints_train2017.json",
        "video_pose/annotations/video_keypoints.json",
        "ochuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json",
        "fall106_datasets/annotations/video_keypoints.json",
        "yizhi_yt20221028/train/annotations/video_keypoints.json",
        "PoseTrack/annotations/posetrack_train.json",
        "PoseTrack/annotations/posetrack_val.json",
    ]
    assert len(train_image_roots) == len(train_annot_files)

    merge_datasets(keep_categories=keep_categories,
              img_roots=train_image_roots,
              ann_files=train_annot_files,
              save_file="/home/huangzhiyong/Project/kapao/data/datasets/_pose12/annotations/train_pose12.json",
              copy_img=True,
              copy_img_dir="/home/huangzhiyong/Project/kapao/data/datasets/_pose12/images/train")

    # val -----------------
    val_image_roots = [
        "coco/images/val2017",
        "ochuman/images",
        "yizhi_yt20221028/val/images",
        "crowdpose/images",
    ]
    val_annot_files = [
        "coco/annotations/person_keypoints_val2017.json",
        "ochuman/annotations/ochuman_coco_format_test_range_0.00_1.00.json",
        "yizhi_yt20221028/val/annotations/video_keypoints.json",
        "crowdpose/crowdpose_test.json",
    ]
    assert len(val_image_roots) == len(val_annot_files)

    merge_datasets(keep_categories=keep_categories,
                   img_roots=val_image_roots,
                   ann_files=val_annot_files,
                   save_file="/home/huangzhiyong/Project/kapao/data/datasets/_pose12/annotations/val_pose12.json",
                   copy_img=True,
                   copy_img_dir="/home/huangzhiyong/Project/kapao/data/datasets/_pose12/images/val")
