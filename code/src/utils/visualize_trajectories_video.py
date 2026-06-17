"""
车辆轨迹视频叠加：将历史轨迹路径实时叠加在原视频上，输出带轨迹的 MP4。

每帧会画出该帧之前所有已记录的轨迹点（渐隐效果：超过 N 秒的线段逐渐变淡）。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import cv2
import numpy as np
import pandas as pd
from collections import defaultdict

from src.config.settings import OUTPUT_DIR, VIDEO_PATH

CSV_PATH = OUTPUT_DIR / "trajectory.csv"
OUT_PATH = OUTPUT_DIR / "trajectory_video.mp4"

MAX_FRAMES  = 1000    # 只处理前 N 帧，None = 全部
FADE_SECS   = 3.0     # 轨迹尾迹保留时长（秒），超出后渐隐

# BGR 颜色
CLASS_COLORS = {
    "car":        (0,   210,   0),
    "truck":      (255, 128,   0),
    "bus":        (0,   128, 255),
    "motorcycle": (255,   0, 200),
    "bicycle":    (200, 255,   0),
    "person":     (0,   220, 220),
}


def main():
    df = pd.read_csv(CSV_PATH)
    # 按 track_id 建立 {track_id: [(frame_id, cx, cy), ...]} 字典
    track_points: dict[int, list] = defaultdict(list)
    track_class:  dict[int, str]  = {}
    for row in df.itertuples():
        track_points[row.track_id].append((row.frame_id, row.cx, row.cy))
        track_class[row.track_id] = row.class_name

    # 预先排序
    for tid in track_points:
        track_points[tid].sort(key=lambda x: x[0])

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if MAX_FRAMES is not None:
        total = min(total, MAX_FRAMES)

    fade_frames = int(FADE_SECS * fps)   # 多少帧内完全淡出

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_PATH), fourcc, fps, (width, height))

    # 为每条轨迹建立 overlay 画布（持久累积）
    # 用一张单独的 overlay 图，每帧在上面画，再 addWeighted 融合
    # 为实现渐隐：每帧对 overlay 做一次轻微衰减
    overlay = np.zeros((height, width, 3), dtype=np.uint8)

    # 按 frame_id 构建：该帧应绘制哪些线段
    # 格式：{frame_id: [(x1,y1,x2,y2,color), ...]}
    frame_segments: dict[int, list] = defaultdict(list)
    for tid, pts in track_points.items():
        color = CLASS_COLORS.get(track_class.get(tid, ""), (200, 200, 200))
        for i in range(1, len(pts)):
            fid_prev, x0, y0 = pts[i - 1]
            fid_curr, x1, y1 = pts[i]
            # 把线段放到"从上一采样帧到当前采样帧"中间每一帧都画
            # 简化：只在 fid_curr 这一帧触发绘制
            frame_segments[fid_curr].append((x0, y0, x1, y1, color))

    frame_idx = 0
    decay = 1.0 - (1.0 / max(fade_frames, 1))   # 每帧衰减系数

    print(f"生成轨迹视频：{width}x{height}  {fps:.0f}fps  共{total}帧")
    while True:
        ret, frame = cap.read()
        if not ret or (MAX_FRAMES is not None and frame_idx >= MAX_FRAMES):
            break

        # 1. overlay 衰减（实现渐隐效果）
        overlay = (overlay * decay).astype(np.uint8)

        # 2. 在 overlay 上绘制当前帧新增的线段
        for seg in frame_segments.get(frame_idx, []):
            x0, y0, x1, y1, color = seg
            cv2.line(overlay, (x0, y0), (x1, y1), color, thickness=2,
                     lineType=cv2.LINE_AA)
            cv2.circle(overlay, (x1, y1), 3, color, -1, lineType=cv2.LINE_AA)

        # 3. 融合 overlay 与原始帧
        # 只在 overlay 非零处叠加
        mask = overlay.any(axis=2)
        out  = frame.copy()
        out[mask] = cv2.addWeighted(frame, 0.35, overlay, 0.65, 0)[mask]

        # 4. 左上角信息
        cv2.putText(out, f"Frame {frame_idx:4d}  Tracks: {len(track_points)}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        writer.write(out)
        frame_idx += 1

        if frame_idx % 100 == 0:
            pct = frame_idx / total * 100
            print(f"  [{pct:5.1f}%] {frame_idx}/{total}", flush=True)

    cap.release()
    writer.release()
    print(f"\n完成，输出: {OUT_PATH}")


if __name__ == "__main__":
    main()
