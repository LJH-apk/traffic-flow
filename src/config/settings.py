"""
全局配置常量。业务代码只 import 本模块的常量，禁止硬编码路径或超参数。
"""
from pathlib import Path

# ── 路径 ──────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).parents[2]                     # 项目根目录

VIDEO_PATH  = _ROOT / "data" / "test_video.mp4"            # 比赛测试视频
OUTPUT_DIR  = _ROOT / "outputs"                             # 所有生成文件的输出目录
MODEL_DIR   = _ROOT                                         # .pt 权重所在目录（暂留根目录）

# ── 模型 ──────────────────────────────────────────────────────────────────────
MODEL_NAME      = "yolo26m.pt"   # 主推理模型，可切换 yolo26n/s/m/l/x.pt
MODEL_NAME_GT   = "yolo26x.pt"  # 伪GT生成器（eval 用）
DEVICE          = "mps"          # 推理设备：mps / cpu / cuda

# ── 类别 ──────────────────────────────────────────────────────────────────────
VEHICLE_CLASSES: dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car":        (0,   255,   0),
    "truck":      (0,   128, 255),
    "bus":        (255, 128,   0),
    "motorcycle": (255,   0, 255),
    "bicycle":    (0,   255, 255),
}

# ── 检测超参 ──────────────────────────────────────────────────────────────────
CONF_THRESH     = 0.25   # 默认置信度阈值
IOU_THRESH      = 0.45   # NMS IoU 阈值

# ── 评测超参 ──────────────────────────────────────────────────────────────────
CONF_GT         = 0.25   # 伪GT置信度阈值
CONF_PRED       = 0.25   # 待评测模型置信度阈值
SAMPLE_EVERY    = 10     # eval_on_video 每隔N帧采样一帧
MAX_SAMPLES     = 200    # eval_on_video 最大采样帧数

# ── 轨迹采样 ─────────────────────────────────────────────────────────────────
TRAJ_SAMPLE_FPS = 1      # 轨迹采样频率（每秒采样次数）
TRAJ_CSV_PATH   = OUTPUT_DIR / "trajectory.csv"  # 轨迹输出文件

# ── 断面过车检测 ──────────────────────────────────────────────────────────────
# 格式: (name, lx1, ly1, lx2, ly2, dir_pos_label, dir_neg_label)
# dir_pos: 叉积正→负方向（水平线：车辆向下/向南 = "到达"）
SECTION_LINES: list[tuple[str, int, int, int, int, str, str]] = [
    ("断面A", 600,  1300, 3200, 1300, "到达", "离去"),
    ("断面B", 1800, 1650, 3800, 1650, "到达", "离去"),
]

# 单应矩阵：像素坐标 → 路面世界坐标（单位：米）
# 由 zebra_detector 自动计算，或运行 calibrate.py 手动标定后粘贴到此处
# None = 退化到 PIXELS_PER_METER 估算
import numpy as np
HOMOGRAPHY_MATRIX: np.ndarray | None = None

# H 不可用时的兜底系数（路口平面粗估：约85px≈1m）
PIXELS_PER_METER: float = 85.0

CROSS_SECTION_CSV_PATH = OUTPUT_DIR / "cross_section.csv"
