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
from argparse import ArgumentParser
from extract_videos import extract_videos, pose_similarity, pose_structure



class poseDetector():
    """
        可视化视频、并保存可视化后的每一帧，用于后续处理。
        关键点蓝色点表示置信度高, 粉色点表示置信度低。
        目前仅仅适合单人视频检测, 可以获取关键点和边界框, 通过修改也可以获取图像分割掩码信息。
    """
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detection_confidence=0.5, track_confidence=0.5, frame_size=(640, 640)):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth   # True 减少抖动
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        # MediaPipe Pose检测器参数,关闭防抖动,因为防抖动可能会降低动作大幅度变化时的检测精度
        self.pose = self.mpPose.Pose(static_image_mode=True, # True：图片模式, false 视频流模式
                                    model_complexity=2,       # 0, 1, 2, 越大精度越高,延迟越大, 注意设置为0或2时需要下载模型
                                    smooth_landmarks=False,    # 减少关键点抖动,只对视频流有效
                                    enable_segmentation=True,  # 生成图像分割掩码, 这里用于获取人体边界框
                                    smooth_segmentation=False,   # 减少分割掩码抖动,只对视频流有效
                                    min_detection_confidence=0.5,  # 最小检测阈值, 预测结果大于该值保留
                                    min_tracking_confidence=0.5)   # 最小跟踪阈值, 检测结果和跟踪预测结果相似度
        self.grid = np.arange(frame_size[0]*frame_size[1]).reshape(frame_size)  # 网格序号,用于找出mask的区域,从而计算bbox
        self.mask_thr = 0.2   # 分割得分阈值,大于该值为人体像素
        self.thickness = 2
        self.text_position = (10, 10)

    def findPose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        if results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(
                img, 
                results.pose_landmarks,
                self.mpPose.POSE_CONNECTIONS)
        return img, results

    def findPosition(self, img, results, pose_structure, draw=True):
        kpts, xyxy, area = [], [], 0
        if results.pose_landmarks:
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                vis = round(lm.visibility, 4)
                # print(f"{cx=}\t{cy=}\t{vis=}")
                kpts.append([cx, cy, vis])
                if draw and idx in pose_structure['request_kpt_indices']:
                    color = (255, 0, 0) if vis > 0.8 else (128, 12, 255)
                    cv2.circle(img, (cx, cy), self.thickness+2, color, cv2.FILLED)

            kpts = np.array(kpts)[pose_structure['request_kpt_indices']]
            if draw:
                for pair, color in zip(pose_structure['skeleton'], pose_structure['line_color']):
                    # line(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img
                    pt1 = int(kpts[pair[0], 0]), int(kpts[pair[0], 1])
                    pt2 = int(kpts[pair[1], 0]), int(kpts[pair[1], 1])
                    cv2.line(img, pt1, pt2, color, thickness=self.thickness)

            area = (results.segmentation_mask > self.mask_thr).size  # 人体分割图面积
            if area > 0:
                # print(results.segmentation_mask.shape)   # h, w
                assert results.segmentation_mask.shape == self.grid.shape, f"{results.segmentation_mask.shape} != {self.grid.shape}"
                mask = self.grid[results.segmentation_mask > self.mask_thr]   # mask_thr 是正样本的阈值, 大于该值为人体区域, 小于则为背景区域
                x = mask % self.grid.shape[1]  
                y = mask // self.grid.shape[1]
                xy = np.concatenate([x[:, None], y[:, None]], axis=-1)
                left_top = xy.min(axis=0)
                right_bottom = xy.max(axis=0)
                xyxy = [*left_top, *right_bottom]
                cv2.rectangle(img, left_top, right_bottom, (255, 0, 0), thickness=self.thickness)  # rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img
                
        return kpts, xyxy, area


    def findAngle(self, img, kpts, p1, p2, p3, draw=True):
        x1, y1 = kpts[p1][1:]
        x2, y2 = kpts[p2][1:]
        x3, y3 = kpts[p3][1:]
        
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        angle += 360 if angle < 0 else 0
        assert  0 <= angle <= 360
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
        return angle        


def save_annotations(filename, kpts, bbox, height, width, area):
    json_dict = dict(
        height=height,
        width=width,
        area=area,
        keypoints=kpts.flatten().tolist(),
        bbox=bbox   # xywh
    )
    with open(str(filename), 'w') as fd:
        json.dump(json_dict, fd, indent=4)
    

def main(img_path:str,                              # 图片文件 或 图片目录
         pose_dict,                                 # 需要保留和可视化的关键点结构
         interval=1,                                # 取帧间隔
         auto_interval=True,                            # 自动跳过相似帧
         save=False,                                # 是否保存图片和标注
         save_img_root='vis_images',                # 保存图片的根目录, 最终目录是 save_img_root/<video_name>, 图片保存为 jpg格式        
         save_ann_root='annotations',               # 保存标注文件的更目录, 最终目录是 save_ann_root/<video_name>, 标注保存为json格式
         show=True):                                # 是否显示可视化

    try:
        img_path = Path(img_path)
        if not img_path.exists():
            raise ValueError(f"Error: img_path not eists! {img_path=}")
        
        # 检查和创建输出目录
        if save:
            save_img_path = Path(save_img_root).joinpath(img_path.stem)
            if not save_img_path.exists():
                save_img_path.mkdir(mode=777, parents=True, exist_ok=True)
                print(f"Save visual images to {str(save_img_path)}")
            else:
                raise ValueError(f"Error: path already exist! {save_img_root=}")

            save_ann_path = Path(save_ann_root).joinpath(img_path.stem)
            if not save_ann_path.exists():
                save_ann_path.mkdir(mode=777, parents=True, exist_ok=True)
                print(f"Save annotations to {str(save_ann_path)}")
            else:
                raise ValueError(f"Error: path already exist! {save_ann_root=}")

        # 生成待处理图像列表
        IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.tif', '.bmp', '.tif', '.tiff', '.dng', '.webp', '.mpo']
        print("Sorting image list...")
        if img_path.is_file() and img_path.suffix in IMAGE_FORMATS:
            img_files = [img_path]
        elif img_path.is_dir():
            images = img_path.glob('*')
            img_files = [img for img in images if img.suffix in IMAGE_FORMATS]
            img_files.sort(key=lambda x: x.stat().st_mtime)  # 图片按修改时间升序排序
        else:
            raise ValueError(f"{img_path=}")

        if show:
            cv2.namedWindow("images", cv2.WINDOW_KEEPRATIO)

        # 初始化检测器和总图片数
        detector = poseDetector()
        frames_count = len(img_files)
        # 初始化前一帧的关键点和边界框
        pos = (30, 30)  # 文字框的左上角坐标
        pre_kpts = [0] * len(pose_dict['request_kpt_indices']) * 3
        pre_bbox = [0, 0, 0, 0]
        for frame_idx in tqdm(range(frames_count), desc="Processing images"):
            file = str(img_files[frame_idx])
            img = cv2.imread(file)
            if img is not None:
                if frame_idx % interval != 0:
                    continue
                height, width = img.shape[:2]
                detector.grid = np.arange(height*width).reshape((height, width))
                img, results = detector.findPose(img, draw=False)
                kpts, xyxy, area = detector.findPosition(img, results, pose_structure=pose_dict, draw=True)  # landmark
                # 计算姿态相似度，用于去除相似连续帧
                if auto_interval and xyxy != []:
                    iou, pck = pose_similarity(kp1=pre_kpts, kp2=kpts, bbox1=pre_bbox, bbox2=xyxy)
                    pre_kpts, pre_bbox = kpts, xyxy
                    if iou > 0.9 and pck > 0.5:
                        continue
                    # cv2.putText(img, str(iou), (pos[0], pos[1]+20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    # cv2.putText(img, str(pck), (pos[0], pos[1]+40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                
                # add info on image
                cv2.putText(img, str(frame_idx), pos, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

                if xyxy != []:
                    bbox = [int(b) for b in xyxy]
                    bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]   # x1y1x2y2 -> x1y1wh
                    # if bbox[2]/bbox[3] > 1.2:   # 宽大于高度, 认为是跌倒
                    #     cv2.putText(img, "Fall", (pos[0], pos[1]+20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    # else:
                    #     cv2.putText(img, "Normal", (pos[0], pos[1]+20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

                if save and not isinstance(kpts, list):
                    img_file = save_img_path.joinpath(img_files[frame_idx].name)
                    cv2.imwrite(str(img_file), img)
                    ann_file = save_ann_path.joinpath(img_files[frame_idx].stem + '.json')
                    save_annotations(ann_file, kpts, bbox, height, width, area)
                if show:
                    cv2.imshow("images", img)
                    cv2.waitKey(1)
                
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--img', type=str, help='The path of a image file or a directory to image files.')
    parser.add_argument('--interval', '-i', type=int, default=1, help='The interval for frame detection')
    parser.add_argument('--auto-interval', action='store_true', help='Auto skip the images having similar pose with previous images')
    parser.add_argument('--show', action='store_true', help='To visualize the processing for getting pose on each frame.')
    parser.add_argument('--save', '-s', action='store_true', help='To save the visual images and annotations.')
    parser.add_argument('--save-img-root', type=str, default='vis_images', help='A directory to saving visual images')
    parser.add_argument('--save-ann-root', type=str, default='annotations', help='A directory to saving annotations')
    args = parser.parse_args()

    """
    目录结构：
    + ./vis_images: 存放视频可视化的帧图像(jpg格式)
    + ./annotations: 存放可视化帧的标注文件(json格式)
    """
    print(f"{args.auto_interval=}")
    if args.auto_interval:
        args.interval = 1
    main(
        img_path=args.img,                   # 图片文件或图片目录
        pose_dict=pose_structure,            # 需要保留和可视化的关键点结构
        interval=args.interval,              # 取帧间隔
        auto_interval=args.auto_interval,    # 自动跳过相似帧
        save=args.save,                      # 是否保存图片和标注
        save_img_root=args.save_img_root,    # 保存图片的根目录, 最终目录是 save_img_root/<video_name>, 图片保存为 jpg格式
        save_ann_root=args.save_ann_root,    # 保存标注文件的更目录, 最终目录是 save_ann_root/<video_name>, 标注保存为json格式
        show=args.show                       # 是否显示可视化
    )
    # \\192.168.16.105/data_huangzhiyong/datasets/fall_datasets/fall106/images



