"""
断面过车检测模块。

输出字段（cross_section.csv）::

    frame_id, timestamp_s, section, track_id, class_name,
    color, direction, speed_kmh, headway_s, spacing_m
"""
import math
from collections import defaultdict, deque
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.cross_section.speed_estimator import SpeedEstimator


def detect_color(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> str:
    """采样 bbox 中央 60%×60%，按 HSV 中位数分类车身颜色。

    Args:
        frame: BGR 帧。
        x1,y1,x2,y2: 车辆检测框坐标。

    Returns:
        颜色名称字符串（黑色/白色/银色/灰色/红色/黄色/绿色/蓝色/其他）。
    """
    fh, fw = frame.shape[:2]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw = (x2 - x1) * 0.3
    hh = (y2 - y1) * 0.3
    sx1 = max(0, int(cx - hw))
    sy1 = max(0, int(cy - hh))
    sx2 = min(fw, int(cx + hw))
    sy2 = min(fh, int(cy + hh))

    if sx2 <= sx1 or sy2 <= sy1:
        return "其他"

    crop = frame[sy1:sy2, sx1:sx2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    H = int(np.median(pixels[:, 0]))
    S = int(np.median(pixels[:, 1]))
    V = int(np.median(pixels[:, 2]))

    if V < 50:
        return "黑色"
    if S < 35 and V > 180:
        return "白色"
    if S < 55 and V > 130:
        return "银色"
    if S < 55:
        return "灰色"
    if H < 10 or H > 160:
        return "红色"
    if 10 <= H < 35:
        return "黄色"
    if 35 <= H < 85:
        return "绿色"
    if 85 <= H <= 130:
        return "蓝色"
    return "其他"


class CrossSectionDetector:
    """断面过车事件检测器。

    通过叉积符号翻转检测车辆穿越虚拟断面线的事件，
    输出速度（世界坐标米制）、车头时距、间距等字段。

    Args:
        lines:            断面线配置列表，格式：
                          [(name, lx1, ly1, lx2, ly2, dir_pos, dir_neg), ...]
                          dir_pos: 叉积由正变负时的方向标签（车辆穿越线从正侧到负侧）。
        homography:       像素→世界坐标单应矩阵（None 时退化到 pixels_per_meter）。
        pixels_per_meter: H 不可用时的兜底系数（像素/米）。
        video_fps:        视频帧率，用于速度和时距计算。
    """

    CSV_FIELDS = [
        "frame_id", "timestamp_s", "section", "track_id", "class_name",
        "color", "direction", "speed_kmh", "headway_s", "spacing_m",
    ]

    def __init__(
        self,
        lines: list[tuple[str, int, int, int, int, str, str]],
        homography: np.ndarray | None,
        pixels_per_meter: float,
        video_fps: float,
        speed_estimator: "SpeedEstimator | None" = None,
    ) -> None:
        self._lines = lines
        self._H = homography
        self._ppm = pixels_per_meter
        self._fps = video_fps
        self._speed_estimator = speed_estimator

        # track_id -> deque of (frame_idx, wx, wy)，保留最近15帧用于速度估算
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=15))

        # (track_id, line_name) -> 上一帧叉积符号 (-1/0/+1)
        self._prev_sign: dict[tuple[int, str], int] = {}

        # line_name -> {direction: (timestamp_s, speed_kmh, wx)} 上一辆过线车
        self._last_crossing: dict[str, dict[str, tuple[float, float, float]]] = (
            defaultdict(dict)
        )

    def _to_world(self, cx: float, cy: float) -> tuple[float, float]:
        """像素坐标转世界坐标（米）。"""
        if self._H is not None:
            pt = np.float32([[[cx, cy]]])
            wp = cv2.perspectiveTransform(pt, self._H)[0][0]
            return float(wp[0]), float(wp[1])
        return cx / self._ppm, cy / self._ppm

    def _estimate_speed(self, tid: int) -> float:
        """估算瞬时速度（km/h）。

        优先用外部 SpeedEstimator（保持算法一致性）；
        否则用内部 _history（向后兼容）。
        """
        if self._speed_estimator is not None:
            v = self._speed_estimator.instant_speed(tid)
            return round(v, 1) if v is not None else 0.0

        hist = self._history[tid]
        if len(hist) < 2:
            return 0.0
        f0, wx0, wy0 = hist[0]
        f1, wx1, wy1 = hist[-1]
        dist_m = math.hypot(wx1 - wx0, wy1 - wy0)
        time_s = (f1 - f0) / self._fps
        if time_s <= 0:
            return 0.0
        return round(dist_m / time_s * 3.6, 1)

    @staticmethod
    def _cross_sign(px: float, py: float,
                    lx1: int, ly1: int, lx2: int, ly2: int) -> int:
        """计算点相对于有向线段的叉积符号。

        Returns:
            +1（点在线左侧/上方）、-1（右侧/下方）或 0（在线上）。
        """
        dx, dy = lx2 - lx1, ly2 - ly1
        cross = dx * (py - ly1) - dy * (px - lx1)
        if cross > 0:
            return 1
        if cross < 0:
            return -1
        return 0

    def update(
        self,
        frame_idx: int,
        timestamp_s: float,
        tid: int,
        class_name: str,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
    ) -> list[dict]:
        """处理单辆车单帧，返回本帧触发的过线事件列表（通常为空）。

        Args:
            frame_idx:   视频绝对帧号。
            timestamp_s: 当前时间戳（秒）。
            tid:         跟踪 ID。
            class_name:  车辆类别名称。
            frame:       当前 BGR 帧（用于颜色检测）。
            x1,y1,x2,y2: 车辆检测框坐标。

        Returns:
            触发的过线事件列表，每个事件为符合 CSV_FIELDS 的字典。
        """
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # 更新世界坐标历史（供速度估算）
        wx, wy = self._to_world(cx, cy)
        self._history[tid].append((frame_idx, wx, wy))

        events = []
        for line in self._lines:
            name, lx1, ly1, lx2, ly2, dir_pos, dir_neg = line
            sign = self._cross_sign(cx, cy, lx1, ly1, lx2, ly2)

            key = (tid, name)
            prev = self._prev_sign.get(key, sign)

            # 仅在符号非零时更新（避免静止压线时状态混乱）
            if sign != 0:
                self._prev_sign[key] = sign

            # 符号翻转 → 过线事件
            if prev != 0 and sign != 0 and prev != sign:
                # sign 由正变负：车辆从正侧穿越到负侧，对应 dir_pos
                direction = dir_pos if sign < 0 else dir_neg
                speed = self._estimate_speed(tid)
                color = detect_color(frame, x1, y1, x2, y2)

                # 车头时距 & 间距
                last = self._last_crossing[name].get(direction)
                if last is not None:
                    last_ts, last_speed, last_wx = last
                    headway_s = round(timestamp_s - last_ts, 2)
                    spacing_m = (
                        round(last_speed / 3.6 * headway_s, 1)
                        if last_speed > 0 else 0.0
                    )
                else:
                    headway_s = 0.0
                    spacing_m = 0.0

                self._last_crossing[name][direction] = (timestamp_s, speed, wx)

                events.append({
                    "frame_id":    frame_idx,
                    "timestamp_s": timestamp_s,
                    "section":     name,
                    "track_id":    tid,
                    "class_name":  class_name,
                    "color":       color,
                    "direction":   direction,
                    "speed_kmh":   speed,
                    "headway_s":   headway_s,
                    "spacing_m":   spacing_m,
                })

        return events
