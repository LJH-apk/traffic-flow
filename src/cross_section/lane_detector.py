"""
车道线自动检测。

基于 Hough 变换检测路面白色/黄色标线，按线段角度聚类后
以不同颜色区分车道方向，供叠加到视频帧上显示。

检测在第一帧运行一次，结果静态复用到后续所有帧（零额外推理开销）。
"""
import math

import cv2
import numpy as np

# 每个角度分组对应的 BGR 颜色（6个桶，每桶30°）
_CLUSTER_COLORS: list[tuple[int, int, int]] = [
    (0,   165, 255),   # 橙色  —  0°~30°
    (0,   255, 128),   # 黄绿  — 30°~60°
    (255,  64, 128),   # 紫粉  — 60°~90°
    (128, 255,   0),   # 亮绿  — 90°~120°
    (255, 128,   0),   # 天蓝  — 120°~150°
    (0,   128, 255),   # 深橙  — 150°~180°
]

_ALPHA = 0.65  # 叠加不透明度


class LaneDetector:
    """基于霍夫变换的路面车道线检测器。

    内部将帧缩小到 1/4 进行处理（4K→960×540），检测结果坐标还原到
    原始分辨率。同方向内距离相近的线段通过极坐标 NMS 合并为一条，
    避免同一条车道线被重复检测。

    Args:
        scale:            处理前缩放比例（默认 0.25）。
        hough_thresh:     HoughLinesP 最小投票数（越大越严格）。
        min_length:       缩小分辨率下最小线段长度（像素）。
        max_gap:          缩小分辨率下最大线段间隔（像素）。
        min_cluster_size: 某角度分组合并后最少线段数，少于此值则丢弃该方向。
        merge_dist:       原始分辨率下，同方向线段合并距离阈值（像素）。
                          小于此距离的平行线段合并为最长一条。
    """

    def __init__(
        self,
        scale: float = 0.25,
        hough_thresh: int = 80,
        min_length: int = 100,
        max_gap: int = 25,
        min_cluster_size: int = 5,
        merge_dist: float = 60.0,
    ) -> None:
        self.scale = scale
        self.hough_thresh = hough_thresh
        self.min_length = min_length
        self.max_gap = max_gap
        self.min_cluster_size = min_cluster_size
        self.merge_dist = merge_dist

    def detect(
        self,
        frame: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
    ) -> list[tuple[int, int, int, int, tuple[int, int, int]]]:
        """检测帧中的车道线段。

        Args:
            frame: BGR 帧（原始分辨率）。
            roi:   可选 (x1,y1,x2,y2) 感兴趣区域，None 则搜索全帧。

        Returns:
            线段列表，每条为 (x1, y1, x2, y2, color_bgr)，
            坐标为原始帧坐标系，已完成去重合并。
        """
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            work = frame[ry1:ry2, rx1:rx2]
            ox, oy = rx1, ry1
        else:
            work = frame
            ox, oy = 0, 0

        # ── 降采样 ───────────────────────────────────────────────────────────
        sh = max(1, int(work.shape[0] * self.scale))
        sw = max(1, int(work.shape[1] * self.scale))
        small = cv2.resize(work, (sw, sh), interpolation=cv2.INTER_AREA)
        inv_s = 1.0 / self.scale

        # ── 白/黄颜色掩膜 ─────────────────────────────────────────────────────
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv,
                                 np.array([0,   0, 190]),
                                 np.array([180, 40, 255]))
        yellow_mask = cv2.inRange(hsv,
                                  np.array([10,  90, 120]),
                                  np.array([38, 255, 255]))
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # ── 边缘检测 ──────────────────────────────────────────────────────────
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        masked = cv2.bitwise_and(gray, gray, mask=color_mask)
        blurred = cv2.GaussianBlur(masked, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 160)

        # ── 霍夫变换 ──────────────────────────────────────────────────────────
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_thresh,
            minLineLength=self.min_length,
            maxLineGap=self.max_gap,
        )
        if lines is None:
            return []

        # ── 还原坐标 + 计算角度 ───────────────────────────────────────────────
        segments: list[tuple[float, int, int, int, int]] = []
        for line in lines:
            sx1, sy1, sx2, sy2 = line[0]
            x1 = int(sx1 * inv_s) + ox
            y1 = int(sy1 * inv_s) + oy
            x2 = int(sx2 * inv_s) + ox
            y2 = int(sy2 * inv_s) + oy
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180.0
            segments.append((angle, x1, y1, x2, y2))

        # ── 按角度分桶（每桶30°）─────────────────────────────────────────────
        buckets: dict[int, list[tuple[int, int, int, int]]] = {}
        for angle, x1, y1, x2, y2 in segments:
            bucket = int(angle // 30) % 6
            buckets.setdefault(bucket, []).append((x1, y1, x2, y2))

        # ── 每桶内极坐标 NMS 合并 + 过滤稀疏方向 ─────────────────────────────
        result: list[tuple[int, int, int, int, tuple[int, int, int]]] = []
        for bucket_idx, segs in buckets.items():
            merged = self._merge_cluster(segs, bucket_idx)
            if len(merged) < self.min_cluster_size:
                continue
            color = _CLUSTER_COLORS[bucket_idx]
            for x1, y1, x2, y2 in merged:
                result.append((x1, y1, x2, y2, color))

        return result

    def _merge_cluster(
        self,
        segs: list[tuple[int, int, int, int]],
        bucket_idx: int,
    ) -> list[tuple[int, int, int, int]]:
        """极坐标 NMS：同方向内距离相近的线段只保留最长一条。

        以该桶的代表角度（桶中心角）计算法线方向，将各线段中点
        投影到法线轴得到 rho，按 rho 排序后滑窗合并。
        """
        if len(segs) <= 1:
            return segs

        approx_angle = math.radians(bucket_idx * 30 + 15)
        normal_angle = approx_angle + math.pi / 2
        cos_n = math.cos(normal_angle)
        sin_n = math.sin(normal_angle)

        items: list[tuple[float, float, int, int, int, int]] = []
        for x1, y1, x2, y2 in segs:
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            rho = mx * cos_n + my * sin_n
            length = math.hypot(x2 - x1, y2 - y1)
            items.append((rho, length, x1, y1, x2, y2))

        items.sort(key=lambda t: t[0])

        merged: list[tuple[int, int, int, int]] = []
        i = 0
        while i < len(items):
            ref_rho = items[i][0]
            best = items[i]
            j = i + 1
            while j < len(items) and abs(items[j][0] - ref_rho) < self.merge_dist:
                if items[j][1] > best[1]:   # 保留最长的
                    best = items[j]
                j += 1
            merged.append((best[2], best[3], best[4], best[5]))
            i = j

        return merged

    @staticmethod
    def draw(frame: np.ndarray,
             lanes: list[tuple[int, int, int, int, tuple[int, int, int]]],
             alpha: float = _ALPHA,
             thickness: int = 3) -> np.ndarray:
        """将车道线半透明叠加到帧上（原地修改）。"""
        if not lanes:
            return frame
        overlay = frame.copy()
        for x1, y1, x2, y2, color in lanes:
            cv2.line(overlay, (x1, y1), (x2, y2), color, thickness)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        return frame
