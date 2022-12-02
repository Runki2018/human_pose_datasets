from pathlib import Path
from video_pose_utils import *
from extract_videos import extract_videos

"""
    目录结构：
    + ./images: 存放视频可视化的帧图像(jpg格式)
    + ./annotations: 存放可视化帧的标注文件(json格式)
    + ./videos: 存放所有待处理的视频
    + ./merge_datasets: 存放合并后的最终数据集
"""

if __name__ == '__main__':
    # 步骤1：从视频中获取可视化帧，每帧上带有预测的关键点和边界框信息。
    # just_one_video = True  # 处理单个视频：
    # if just_one_video:
    #     main(
    #         video_path="videos/20221019111006video (48).avi",    # 视频路径
    #         pose_dict=pose_structure,     # 需要保留和可视化的关键点结构
    #         interval=4,                   # 取帧间隔
    #         save=True,                    # 是否保存图片和标注
    #         save_img_root='images',       # 保存图片的根目录, 最终目录是 save_img_root/<video_name>, 图片保存为 jpg格式
    #         save_ann_root='annotations',  # 保存标注文件的更目录, 最终目录是 save_ann_root/<video_name>, 标注保存为json格式
    #         show=True                    # 是否显示可视化
    #     )
    # else:
    #     # 多个视频的根目录，提取到一个目录下
    #     videos_root="FallDataset"
    #     output_dir = 'videos'
    #     video_format = '.avi'   # 原视频目录下的视频格式: '.avi' or ['.avi', '.mp4', ...]
    #     # extract_videos(videos_root=videos_root,
    #     #                output_dir=output_dir,
    #     #                video_format=video_format)

    #     output_dir = Path(output_dir)
    #     for video in output_dir.glob('**/*' + video_format):
    #         main(
    #             video_path=str(video),        # 视频路径
    #             pose_dict=pose_structure,     # 需要保留和可视化的关键点结构
    #             interval=4,                   # 取帧间隔
    #             save=True,                   # 是否保存图片和标注
    #             save_img_root='images',       # 保存图片的根目录, 最终目录是 save_img_root/<video_name>, 图片保存为 jpg格式
    #             save_ann_root='annotations',  # 保存标注文件的更目录, 最终目录是 save_ann_root/<video_name>, 标注保存为json格式
    #             show=False                     # 是否显示可视化
    #         )

    # 步骤2：生成最终数据集
    # 1. 使用Windows图片查看器去除检测结果质量低的图片，快捷键: '→'键 下一张, 'Del'键 删除当前图片
    # 2. 完成步骤1以及去除低质量帧后，使用merge_remained_data函数，根据保留下来的高质量检测结果，生成相应帧图片和标注文件

    merge_remained_data(
        video_root="videos",
        img_root='images',
        ann_root='annotations',
        output_root='merge_datasets',
        show=False,
    )

