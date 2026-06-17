"""
车辆轨迹可视化：将 trajectory.csv 中的路径叠加在视频第一帧上，输出 PNG 图。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

from src.config.settings import OUTPUT_DIRCSV_PATH = OUTPUT_DIR / "trajectory.csv"
OUT_PNG = OUTPUT_DIR / "trajectory_map.png"

# 各类别颜色（RGB, 0-1）
CLASS_COLORS = {
    "car":        (0.2,  0.85, 0.2),
    "truck":      (0.1,  0.5,  1.0),
    "bus":        (1.0,  0.5,  0.1),
    "motorcycle": (1.0,  0.2,  1.0),
    "bicycle":    (0.2,  1.0,  1.0),
    "person":     (1.0,  1.0,  0.2),
}

MIN_POINTS = 3   # 至少有这么多轨迹点才画


def load_background(video_path: Path, scale: float = 0.4) -> np.ndarray:
    """读取视频第一帧作为背景，缩放后返回 RGB 数组。"""
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise IOError(f"无法读取视频: {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w  = frame.shape[:2]
    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame


def main():
    df  = pd.read_csv(CSV_PATH)
    bg  = load_background()
    H, W = bg.shape[:2]

    # 原始视频分辨率（用于坐标缩放）
    orig_w, orig_h = 3840, 2160
    sx = W / orig_w
    sy = H / orig_h

    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    ax.imshow(bg, alpha=0.55)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)   # 图像坐标系：y 轴向下
    ax.axis("off")

    plotted = 0
    for tid, grp in df.groupby("track_id"):
        if len(grp) < MIN_POINTS:
            continue
        grp     = grp.sort_values("frame_id")
        cls     = grp["class_name"].iloc[0]
        color   = CLASS_COLORS.get(cls, (1, 1, 1))
        xs      = grp["cx"].values * sx
        ys      = grp["cy"].values * sy

        # 渐变透明度：越新的点越不透明（用 LineCollection 实现）
        points  = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segs    = np.concatenate([points[:-1], points[1:]], axis=1)
        n       = len(segs)
        alphas  = np.linspace(0.15, 0.9, n)
        colors  = [(*color, a) for a in alphas]
        lc      = LineCollection(segs, colors=colors, linewidths=1.5)
        ax.add_collection(lc)

        # 终点标记（最新位置）
        ax.plot(xs[-1], ys[-1], "o", color=color, markersize=3, alpha=0.9)
        plotted += 1

    # 图例
    legend_patches = [
        mpatches.Patch(color=c, label=cls)
        for cls, c in CLASS_COLORS.items()
        if cls in df["class_name"].values
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              fontsize=9, framealpha=0.7, markerscale=1.5)

    n_tracks = df["track_id"].nunique()
    ax.set_title(
        f"车辆轨迹图  |  共 {n_tracks} 条轨迹（{plotted} 条已绘制，≥{MIN_POINTS} 点）  "
        f"|  前 1000 帧",
        fontsize=11, pad=8,
    )

    plt.tight_layout(pad=0.3)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {OUT_PNG}  ({plotted} 条轨迹)")


if __name__ == "__main__":
    import matplotlib
    matplotlib.rcParams["font.family"] = "Arial Unicode MS"
    main()
