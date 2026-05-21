"""
光照与天气模式判断模块。

由两部分独立组合：
1. 时段：从视频起始时间推断（晨/午/晚/夜）
2. 画面：从第一帧像素统计推断（亮度、雨噪声）

最终的 preset 名是 LIGHTING_PRESETS 字典里的键，
对应一组检测参数（CONF_THRESH、是否启用 CLAHE 等）。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.config.settings import LIGHTING_PRESETS

# 雨天判定：边缘密度阈值（雨水模糊导致高频特征下降）
_RAIN_EDGE_DENSITY_THRESH = 0.025

# 亮度判定阈值（V 通道均值）
_NIGHT_V_MAX  = 60   # < 60 视为夜间
_DUSK_V_MAX   = 100  # < 100 视为黄昏/暮色


def get_lighting_preset(start_time: datetime) -> str:
    """根据视频起始时刻推断光照预设名。

    时段划分：
      07:00 - 09:00  早高峰 → morning_peak
      17:00 - 19:00  晚高峰 → evening_peak
      19:00 - 22:00  黄昏    → dusk
      22:00 - 05:00  夜间    → night
      其他           平峰    → off_peak
    """
    h = start_time.hour
    if 7 <= h < 9:
        return 'morning_peak'
    if 17 <= h < 19:
        return 'evening_peak'
    if 19 <= h < 22:
        return 'dusk'
    if h >= 22 or h < 5:
        return 'night'
    return 'off_peak'


def detect_lighting_from_frame(frame: np.ndarray) -> str:
    """从单帧画面统计推断光照预设名。

    完全不依赖时间信息，仅用画面亮度统计：
      平均亮度 < 60  → night
      平均亮度 < 100 → dusk
      其他           → off_peak（白天泛指）
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_v = float(np.mean(gray))
    if mean_v < _NIGHT_V_MAX:
        return 'night'
    if mean_v < _DUSK_V_MAX:
        return 'dusk'
    return 'off_peak'


def detect_rain(frame: np.ndarray) -> bool:
    """雨天画面纹理被雨水模糊，高频边缘密度显著下降。

    Returns:
        True  → 疑似雨天
        False → 正常
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))
    return edge_density < _RAIN_EDGE_DENSITY_THRESH


def apply_preset(preset_name: str) -> dict:
    """获取指定预设的参数字典。

    返回 dict 包含 conf_thresh / clahe / dehaze 等键。
    若预设名不存在，回退到 off_peak。
    """
    return dict(LIGHTING_PRESETS.get(preset_name, LIGHTING_PRESETS['off_peak']))


def decide_preset(start_time: datetime | None, frame: np.ndarray | None = None) -> str:
    """综合时段和画面统计决定最终预设。

    策略：
      - 优先使用时段判断
      - 若画面统计也表明夜间（且时段不是夜间），以画面为准（更准确）
      - frame is None 时只用时间
      - start_time is None 时只用画面
    """
    if start_time is None and frame is None:
        return 'off_peak'
    if frame is None:
        return get_lighting_preset(start_time)
    if start_time is None:
        return detect_lighting_from_frame(frame)

    time_preset  = get_lighting_preset(start_time)
    frame_preset = detect_lighting_from_frame(frame)

    # 画面表明夜间但时段不是 → 信任画面（可能是阴天/隧道等）
    if frame_preset == 'night' and time_preset not in ('night', 'dusk'):
        return 'night'
    return time_preset
