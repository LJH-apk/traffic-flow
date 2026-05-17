"""
视频读写工具函数。封装 OpenCV VideoCapture / VideoWriter，消除各模块重复样板代码。
"""
import cv2
from pathlib import Path
from typing import Generator


def open_video(path: str | Path) -> cv2.VideoCapture:
    """打开视频文件并校验可用性。

    Args:
        path: 视频文件路径。

    Returns:
        已打开的 VideoCapture 对象。

    Raises:
        FileNotFoundError: 文件不存在。
        IOError: VideoCapture 无法打开。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"视频文件不存在: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {path}")
    return cap


def video_meta(cap: cv2.VideoCapture) -> dict:
    """读取视频元数据。

    Args:
        cap: 已打开的 VideoCapture 对象。

    Returns:
        包含 width / height / fps / frame_count 的字典。
    """
    return {
        "width":       int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":         cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def make_writer(
    path: str | Path,
    fps: float,
    width: int,
    height: int,
    fourcc: str = "mp4v",
) -> cv2.VideoWriter:
    """创建 VideoWriter，自动创建父目录。

    Args:
        path:   输出文件路径。
        fps:    帧率。
        width:  帧宽（像素）。
        height: 帧高（像素）。
        fourcc: 编码器四字节码，默认 mp4v。

    Returns:
        cv2.VideoWriter 对象。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cc = cv2.VideoWriter_fourcc(*fourcc)
    return cv2.VideoWriter(str(path), cc, fps, (width, height))


def iter_frames(
    cap: cv2.VideoCapture,
    max_frames: int | None = None,
) -> Generator[tuple[int, any], None, None]:
    """逐帧迭代视频，返回 (frame_index, frame) 元组。

    Args:
        cap:        已打开的 VideoCapture 对象。
        max_frames: 最大帧数限制，None 表示读完整个视频。

    Yields:
        (frame_index, frame): 帧序号（从 0 开始）与 BGR 图像数组。
    """
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield idx, frame
        idx += 1
        if max_frames is not None and idx >= max_frames:
            break
