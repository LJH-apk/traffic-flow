"""
车辆测速模块（滑动窗口）。

A 模式：每帧返回当前 track 的瞬时速度（窗口首尾两点）
C 模式：track 消失时返回整段路径的平均/最大/最小速度

像素 → 世界坐标：
- 优先用单应矩阵 H（cv2.perspectiveTransform）
- H 不可用时退化到 PIXELS_PER_METER 兜底
"""
from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np


class SpeedEstimator:
    def __init__(
        self,
        homography: np.ndarray | None,
        fps: float,
        window: int = 15,
        pixels_per_meter: float = 85.0,
        min_dist_m: float = 0.1,
    ) -> None:
        self.H = homography
        self.fps = fps
        self.window = window
        self.ppm = pixels_per_meter
        self.min_dist_m = min_dist_m

        # tid -> deque[(frame_idx, wx, wy)]
        self._tracks: dict[int, deque] = {}
        # tid -> {'sum','count','max','min','first_frame','last_frame'}
        self._stats: dict[int, dict] = {}

    def _pix_to_world(self, cx: float, cy: float) -> tuple[float, float]:
        if self.H is not None:
            pt = np.float32([[[cx, cy]]])
            wp = cv2.perspectiveTransform(pt, self.H)[0][0]
            return float(wp[0]), float(wp[1])
        return cx / self.ppm, cy / self.ppm

    def update(self, frame_idx: int, tid: int, cx: float, cy: float) -> None:
        wx, wy = self._pix_to_world(cx, cy)
        hist = self._tracks.setdefault(tid, deque(maxlen=self.window))
        hist.append((frame_idx, wx, wy))

        # 累计统计
        v = self.instant_speed(tid)
        if v is None or v <= 0:
            return
        s = self._stats.setdefault(tid, {
            'sum': 0.0, 'count': 0,
            'max': 0.0, 'min': float('inf'),
            'first_frame': frame_idx,
            'last_frame': frame_idx,
        })
        s['sum'] += v
        s['count'] += 1
        s['max'] = max(s['max'], v)
        s['min'] = min(s['min'], v)
        s['last_frame'] = frame_idx

    def instant_speed(self, tid: int) -> float | None:
        hist = self._tracks.get(tid)
        if not hist or len(hist) < 2:
            return None
        f0, wx0, wy0 = hist[0]
        f1, wx1, wy1 = hist[-1]
        dist = math.hypot(wx1 - wx0, wy1 - wy0)
        if dist < self.min_dist_m:
            return 0.0
        dt = (f1 - f0) / self.fps
        if dt <= 0:
            return None
        return dist / dt * 3.6

    def get_stats(self, tid: int) -> dict | None:
        s = self._stats.get(tid)
        if not s or s['count'] == 0:
            return None
        return {
            'avg_kmh': s['sum'] / s['count'],
            'max_kmh': s['max'],
            'min_kmh': s['min'],
            'n_samples': s['count'],
            'first_frame': s['first_frame'],
            'last_frame': s['last_frame'],
        }

    def finalize(self, tid: int) -> dict | None:
        stats = self.get_stats(tid)
        self._tracks.pop(tid, None)
        self._stats.pop(tid, None)
        return stats
