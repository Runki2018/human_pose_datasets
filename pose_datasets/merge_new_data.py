# 这个文件用于将新的数据融合到已有的数据集中
from my_coco_tools import COCO
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime as dt
import shutil

def merge_new_data(src_img_path,   # 新数据的图片目录
                   src_json_file,  # 新数据的标注文件路径
                   dst_img_path,   # 已有数据集的图片目录
                   dst_json_file,  # 已有数据集的标注距离
                   dst_log_path=None,  # 将合并信息添加到已有log文件中
                   img_prefix='X',  # 给新数据的图片名添加前缀,可以防止命名冲突
                   out_json_file=None,  # 合并后的标注文件名,
                   copy_img=True, # copy src_images to destination image directory
                    overwirte_dst_file=False
                   ): 
    """
    将新数据从src合并到dst中
    """
    src_img_path = Path(src_img_path)
    dst_img_path = Path(dst_img_path)
    assert src_img_path.exists() and dst_img_path.exists(), "image path not exists!"
    
    src_dataset = COCO(src_json_file)
    dst_dataset = COCO(dst_json_file)
    
    src_imgIds = src_dataset.getImgIds()
    dst_imgIds = dst_dataset.getImgIds()
    
    # 已有图片名的hash表用于防止命名冲突和非法重写
    img_hash = dict()
    for img_id in tqdm(dst_imgIds, desc="Gengerating image hash"):
        img_info = dst_dataset.loadImgs(img_id)[0]
        img_hash[img_info['file_name']] = True
    
    # 已有数据集的最大图片序号和最大标注序号
    max_dst_img_id = max(dst_dataset.imgs.keys())
    max_dst_ann_id = max(dst_dataset.anns.keys())

    # 新数据的图片序号和标注序号
    src_images, src_annotations = [], []
    new_img_id, new_ann_id = max_dst_img_id + 1, max_dst_ann_id + 1
    for img_id in tqdm(src_imgIds, desc="Merging new data"):
        # 检查文件名
        img_info = src_dataset.loadImgs(img_id)[0]
        file_name = img_prefix + img_info['file_name']
        while img_hash.get(file_name, False):
            file_name = dt.strftime(dt.now(), "%Y%m%d%H%M%S") + file_name
        # # 拷贝图片
        src_file = src_img_path.joinpath(img_info['file_name'])
        dst_file = dst_img_path.joinpath(file_name)
        if copy_img:
            shutil.copyfile(src_file, dst_file)
        # 修改图片信息
        img_info['id'] = new_img_id
        img_info['file_name'] = file_name
        src_images.append(img_info)
        
        # 修改标注信息
        annIds = src_dataset.getAnnIds(img_id)
        for ann_id in annIds:
            ann_info = src_dataset.loadAnns(ann_id)[0]
            ann_info['id'] = new_ann_id
            ann_info['image_id'] = new_img_id
            src_annotations.append(ann_info)
            new_ann_id += 1
        new_img_id += 1

    # 保存标注文件
    dst_json_file = Path(dst_json_file)
    if overwirte_dst_file:
        out_json_file = dst_json_file
    elif out_json_file != None and out_json_file.split('.')[-1] == 'json':
        out_json_file = dst_json_file.with_name(out_json_file)
    else:
        out_json_file = dt.strftime(dt.now(), "%Y%m%d") + '_' + dst_json_file.name
        out_json_file = dst_json_file.with_name(out_json_file)

    json_dict = dst_dataset.dataset
    dst_num_imgs = len(json_dict['images'])
    dst_num_anns = len(json_dict['annotations'])
    src_num_imgs = len(src_images)
    src_num_anns = len(src_annotations)
    
    json_dict['images'].extend(src_images)
    json_dict['annotations'].extend(src_annotations)
    with out_json_file.open('w') as fd:
        json.dump(json_dict, fd, indent=4)

    # 读入旧的日志信息
    if dst_log_path != None:
        dst_log_path = Path(dst_log_path)
        with dst_log_path.open('r') as fd:
            lines = fd.readlines()

    # 写入新日志
    new_log_file = dst_log_path.with_name(
        dt.strftime(dt.now(), "%Y%m%d%H%M%S")+ '_' + dst_log_path.name)
    with new_log_file.open('w') as fd:
        data_time = dt.strftime(dt.now(), "%Y-%m-%d, %H:%M:%S")
        message = f"\n\n@ Merging new data\n \
                =>num_images:\t\t{dst_num_imgs + src_num_imgs} = {dst_num_imgs} + {src_num_imgs}\n \
                =>num_annotations:\t{dst_num_anns + src_num_anns} = {dst_num_anns} + {src_num_anns}\n \
                =>date time:\t\t{data_time} \n \
                merge_new_data.py \n \
                img_prefix: {img_prefix}:\n \
                src_img_path:  {str(src_img_path.absolute())}\n \
                src_json_file: {src_json_file}\n \
                dst_img_path: {(dst_img_path.absolute())}\n \
                dst_json_file: {dst_json_file}\n \
                dst_log_path:  {dst_log_path}\n \
                Save log file => {str(new_log_file)} \n \
                Save new json file => {str(out_json_file)} \n"
        fd.writelines(lines)
        fd.write(message)
        print(message)


if __name__ == '__main__':
    merge_new_data(src_img_path="test2/images",   # 新数据的图片目录
                   src_json_file="test2/annotations/video_keypoints.json",  # 新数据的标注文件路径
                   dst_img_path="/data/zrway/huangzhiyong/fall_detection/pose/datasets/pose12/images/val",   # 已有数据集的图片目录
                   dst_json_file="/data/zrway/huangzhiyong/fall_detection/pose/datasets/pose12/annotations/val_pose12.json",  # 已有数据集的标注file
                   dst_log_path="/data/zrway/huangzhiyong/fall_detection/pose/datasets/pose12/annotations/merge_dataset_20221107163020.log",   # 将新增数据的信息记录到已有日志文件中
                   img_prefix='test2',  # 给新数据的图片名添加前缀,可以防止命名冲突
                   out_json_file=None,  # 合并后的标注文件名,
                   copy_img=True,
                   overwirte_dst_file=True,
                   )
