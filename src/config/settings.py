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
MODEL_NAME      = "yolo26n.pt"   # 主推理模型，可切换 yolo26n/s/m/l/x.pt
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
# dir_pos: 叉积正方向（水平线从左→右，车辆向下 = 叉积为正 = "到达"）
# 主断面：停止线上游 15m，覆盖全部车道宽度
# 右转断面：从 Lane4 停止线端点沿车道方向向路口内延伸，检测右转车辆
SECTION_LINES_MAP: dict[str, list[tuple[str, int, int, int, int, str, str]]] = {
    "北进口": [
        # H 矩阵精确标定，15m 由单应矩阵计算
        ("北进口主断面", 1497, 551,  2190, 534,  "到达", "离去"),
        ("北进口右转",   2493, 731,  2781, 929,  "右转", "直行"),
    ],
    "南进口": [
        # 15m 近似值，依赖北进口 H 估算
        ("南进口主断面", 1637, 501,  2302, 501,  "到达", "离去"),
        ("南进口掉头",   2382, 655,  2585, 939,  "掉头", "直行"),
        ("南进口右转",   1542, 630,  1403, 951,  "右转", "直行"),
    ],
    "东进口": [
        # 独立摄像头，2026-05-21 车道线标注标定
        # 斜向断面：内侧L1底端→外侧L4底端，对应停止线上游约15m
        # 来车从上方（远端）驶向下方（停止线），叉积负→正触发 dir_pos
        ("东进口主断面", 1157, 1442, 3080, 1302, "到达", "离去"),
    ],
}

# 兜底：未能识别进口时使用全部断面线（向后兼容）
SECTION_LINES: list[tuple[str, int, int, int, int, str, str]] = [
    line for lines in SECTION_LINES_MAP.values() for line in lines
]

# 单应矩阵：像素坐标 → 路面世界坐标（单位：米）
# 由 zebra_detector 自动计算，或运行 calibrate.py 手动标定后粘贴到此处
# None = 退化到 PIXELS_PER_METER 估算
import numpy as np
HOMOGRAPHY_MATRIX: np.ndarray | None = None

# H 不可用时的兜底系数（路口平面粗估：约85px≈1m）
PIXELS_PER_METER: float = 85.0

CROSS_SECTION_CSV_PATH = OUTPUT_DIR / "cross_section.csv"

# ── 车道标定 ──────────────────────────────────────────────────────────────────
CALIBRATIONS_DIR = _ROOT / "calibrations"

# ── 光照预设（按时段或检测结果切换检测参数）───────────────────────────────────
LIGHTING_PRESETS: dict[str, dict] = {
    'morning_peak': dict(conf_thresh=0.30, clahe=False, dehaze=False),
    'off_peak':     dict(conf_thresh=0.25, clahe=False, dehaze=False),
    'evening_peak': dict(conf_thresh=0.30, clahe=False, dehaze=False),
    'dusk':         dict(conf_thresh=0.20, clahe=True,  dehaze=False),
    'night':        dict(conf_thresh=0.15, clahe=True,  dehaze=True),
}

# 进口名规范化（统一映射到中文）
ENTRANCE_ALIASES: dict[str, str] = {
    '北进口': '北进口', 'north': '北进口', 'N': '北进口',
    '南进口': '南进口', 'south': '南进口', 'S': '南进口',
    '东进口': '东进口', 'east':  '东进口', 'E': '东进口',
}

# ── 测速 ─────────────────────────────────────────────────────────────────────
SPEED_WINDOW_FRAMES = 15                              # 滑动窗口长度（帧），25fps 下约 0.6s
SPEED_MIN_DIST_M    = 0.1                             # 小于此距离视为静止（米）
SPEED_MIN_SAMPLES   = 5                               # 写入 vehicle_stats.csv 的最少采样数

VEHICLE_STATS_CSV_PATH = OUTPUT_DIR / "vehicle_stats.csv"

# ── 统计报表 ──────────────────────────────────────────────────────────────────
VEHICLE_LENGTHS_M: dict[str, float] = {
    "car": 4.5, "truck": 8.0, "bus": 12.0, "motorcycle": 2.0, "bicycle": 1.8,
}
SECTION_ROAD_LENGTH_M: float = 50.0   # 断面监测路段默认长度（米）
QUEUE_SPEED_THRESH_KMH: float = 10.0  # 低于此速度视为排队状态（km/h）
QUEUE_GAP_M: float = 1.5              # 排队车辆平均车间距（米）
EXCEL_REPORT_PATH = OUTPUT_DIR / "traffic_report.xlsx"

# ── 轨迹分组 ──────────────────────────────────────────────────────────────────
TRAJ_GROUP_INTERVAL_S    = 15.0    # 批量分组触发间隔（秒）
TRAJ_GROUP_COS_THRESH    = 0.70    # 余弦相似度阈值（0.70–0.95）
TRAJ_GROUP_JSD_THRESH    = 0.70    # JS散度相似度阈值（1-JSD，0.70–0.95）
TRAJ_GROUP_EUC_THRESH    = 0.70    # 欧氏距离相似度阈值（对角线归一化，[0,1]，越高越相似）
TRAJ_GROUP_MIN_FRAMES    = 8       # 短于此帧数的轨迹片段丢弃（遮挡尾迹）
TRAJ_GROUP_CSV_PATH      = OUTPUT_DIR / "trajectory_groups.csv"
# 各进口车辆行驶参考方向（图像坐标，Y轴向下）
ENTRANCE_TRAVEL_DIR: dict[str, tuple[float, float]] = {
    "北进口": (0.0,  1.0),   # 向南（图像向下）
    "南进口": (0.0, -1.0),   # 向北（图像向上）
    "东进口": (-1.0, 0.0),   # 向西（图像向左）
}
