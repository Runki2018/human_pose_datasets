import shutil
from pathlib import Path
from datetime import datetime as dt

def extract_videos(videos_root, output_dir):
    """将指定根目录下面的所有视频重命名(确保命名不冲突), 并输出到指定目录
    videos_root (str)
    output_dir (str)
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(mode=777, parents=True, exist_ok=True)
    print(f"Extracting videos from {videos_root} to {str(output_dir)}")

    # 多个视频的根目录
    video_path = Path(videos_root)
    # 视频的格式如： '.avi', '.mp4'
    video_format = ['mov', 'avi', 'mp4','mpg','mpeg','m4v','mkv','wmv']
    print(f"Supported Video Formats => {video_format}")
    for vf in video_format:
        for video in video_path.glob('**/*' + vf):
            src_video = str(video)
            dst_video = dt.strftime(dt.now(), "%Y%m%d%H%M%S") + video.name
            dst_video = str(output_dir.joinpath(dst_video))
            shutil.copyfile(src_video, dst_video)
            print(f"src_video => dst_video: {src_video} => {dst_video}")
