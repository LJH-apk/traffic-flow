"""
车道线检测模块（双实现）。

── 运行时检测器 ─────────────────────────────────────────────────────────────────
LaneDetector（Hough 变换）：在 tracker.py 主循环中首帧运行一次，结果静态复用。
  - 零额外推理开销，适合实时叠加显示。

── 标定辅助算法 ─────────────────────────────────────────────────────────────────
基于背景白色标线的车道线自动检测（行扫描 + 多项式 + 样条拟合）。

主要用途：辅助生成初始车道标定（人工再微调），
而非作为运行期主车道检测器（运行期请用 LaneDetector 加载已标定数据）。

思路：
  1. 中位数背景消除运动车辆，得到静态路面标线
  2. 白色掩码提取标线
  3. 对 ROI 每一行扫描白色像素 x 坐标，按列近邻聚类成各条车道线
  4. 两遍中位数曲率过滤 + 间距合并
  5. degree-3 拟合 + 切线方向远端追踪
  6. UnivariateSpline 平滑（BoundedCurve 避免外推震荡）

运行：
  python3 -m src.cross_section.lane_detector
  python3 -m src.cross_section.lane_detector --frame 2675
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.config.settings import OUTPUT_DIR


# ══════════════════════════════════════════════════════════════════════════════
# 运行时检测器（Hough 变换，供 tracker.py 使用）
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# 标定辅助算法（行扫描 + 多项式 + 样条拟合，用于生成初始标定数据）
# ══════════════════════════════════════════════════════════════════════════════

class BoundedCurve:
    """样条/多项式包装器，求值时自动截断到数据范围内，避免外推震荡。"""
    def __init__(self, fn, y_min: float, y_max: float):
        self.fn = fn
        self.y_min = y_min
        self.y_max = y_max

    def __call__(self, y):
        return self.fn(np.clip(y, self.y_min, self.y_max))

    def derivative(self):
        return self.fn.derivative()


# 标定辅助算法专用视频路径（默认使用修复版视频）
_CALIB_VIDEO = "test_video_fixed.mp4"

# ── ROI ─────────────────────────────────────────────────────────────────────
ROI_Y_START = 0.10
ROI_Y_END   = 0.30
ROI_X_START = 0.35
ROI_X_END   = 0.64

# ── 白色掩码 ─────────────────────────────────────────────────────────────────
WHITE_V_MIN = 155
WHITE_S_MAX = 40

# ── 行扫描聚类参数 ───────────────────────────────────────────────────────────
ROW_STEP       = 4      # 每隔多少行扫描一次（加速）
CLUSTER_GAP    = 45     # 同行内超过此距离视为不同车道线（px）
TRACK_GAP      = 80     # 跨行追踪：上一行中心与当前中心距离阈值（px）
MIN_ROW_COUNT  = 40     # 一条车道线至少出现的行数（过滤噪声）
MIN_ROW_COUNT_FAR = 15  # 远端（ROI 上半段）最少行数
ROI_FAR_SPLIT     = 0.5 # ROI y 范围内远/近分界（0 = 顶部，1 = 底部）
MIN_LANE_WIDTH    = 150 # 相邻两条线最小横向间距（px），小于此视为同一条线重复检测

# ── 多项式次数 ───────────────────────────────────────────────────────────────
POLY_DEG = 2

# ── 曲率过滤：二次项系数 a 允许偏离中位数的最大值 ────────────────────────────
CURVATURE_TOL = 2e-3    # 单位：px^-1，实测 a ~ 1e-3，容差约 ±100%
POLY_SEARCH_R = 60   # 多项式引导搜索横向半径（px）
EXTEND_SEARCH_R = 40 # 切线追踪时每行搜索横向半径（px）
EXTEND_SLOPE_N  = 5  # 用最近 N 个检测点估计当前斜率
EXTEND_FAR_W    = 8  # 远端点在 polyfit 中的权重倍数（相对近端点=1）
SPLINE_S_PER_PT = 200 # 样条平滑因子（每点），远端稀疏时需要更大值防震荡
BLIND_SEARCH_R    = 60   # 盲搜横向半径（px）
BLIND_MIN_ROWS    = 12   # 盲搜需要找到的最少行数
BLIND_BOUNDARY_EXCL = 30   # 盲搜边界排除距离：忽略贴近已知线的白色像素（px）

# ── 车道颜色表（BGR）────────────────────────────────────────────────────────
LANE_COLORS = [
    (180,  60,  60), (60, 180,  60), (60,  60, 180),
    (180, 180,  60), (60, 180, 180), (180,  60, 180),
    (120, 200,  80), (200, 120,  80), (80, 120, 200),
    (200,  80, 120), (80, 200, 120),
]


# ── 背景模型 ─────────────────────────────────────────────────────────────────
def build_background(video_path: str, target_frame: int,
                     n_samples: int = 20, spread: int = 200):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(
        max(0, target_frame - spread),
        min(total - 1, target_frame + spread),
        n_samples, dtype=int,
    )
    grays = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, f = cap.read()
        if ret:
            grays.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()

    if not grays:
        return None, None

    bg = np.median(np.stack(grays, axis=0), axis=0).astype(np.uint8)

    cap2 = cv2.VideoCapture(video_path)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    _, frame = cap2.read()
    cap2.release()
    return frame, bg


def white_mask_calib(img_bgr: np.ndarray) -> np.ndarray:
    """提取白色标线掩码（标定辅助专用）。"""
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, WHITE_V_MIN), (180, WHITE_S_MAX, 255))
    k    = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask,  k, iterations=1)
    mask = cv2.dilate(mask, k, iterations=2)
    return mask


# ── 行扫描 → 车道线点集 ──────────────────────────────────────────────────────
def scan_lane_points(mask: np.ndarray, h: int, w: int) -> list[np.ndarray]:
    """
    对 ROI 内每行扫描白色像素，按列追踪聚类成各条车道线。
    返回列表，每项为 shape=(N,2) 的点集 [[y, x], ...]，已按全图坐标。
    """
    roi_y1 = int(h * ROI_Y_START)
    roi_y2 = int(h * ROI_Y_END)
    roi_x1 = int(w * ROI_X_START)
    roi_x2 = int(w * ROI_X_END)

    # 活跃轨迹：{id: {"cx": float, "pts": [(y,x), ...]}}
    tracks: dict[int, dict] = {}
    next_id = 0

    for abs_y in range(roi_y1, roi_y2, ROW_STEP):
        row = mask[abs_y, roi_x1:roi_x2]
        xs  = np.where(row > 0)[0] + roi_x1   # 全图 x

        if len(xs) == 0:
            continue

        # 将同行白色像素按间距分组
        groups = []
        grp = [xs[0]]
        for x in xs[1:]:
            if x - grp[-1] <= CLUSTER_GAP:
                grp.append(x)
            else:
                groups.append(int(np.mean(grp)))
                grp = [x]
        groups.append(int(np.mean(grp)))

        for cx in groups:
            best_id, best_dist = None, TRACK_GAP
            for tid, tk in tracks.items():
                d = abs(cx - tk["cx"])
                if d < best_dist:
                    best_dist, best_id = d, tid
            if best_id is not None:
                tracks[best_id]["pts"].append((abs_y, cx))
                tracks[best_id]["cx"] = cx
            else:
                tracks[next_id] = {"cx": cx, "pts": [(abs_y, cx)]}
                next_id += 1

    # 过滤短轨迹
    roi_y1 = int(h * ROI_Y_START)
    roi_y2 = int(h * ROI_Y_END)
    far_thresh_y = roi_y1 + (roi_y2 - roi_y1) * ROI_FAR_SPLIT

    lines = []
    for tk in tracks.values():
        if not tk["pts"]:
            continue
        median_y = float(np.median([p[0] for p in tk["pts"]]))
        thresh = MIN_ROW_COUNT_FAR if median_y < far_thresh_y else MIN_ROW_COUNT
        if len(tk["pts"]) >= thresh:
            lines.append(np.array(tk["pts"], dtype=np.float32))

    # 按中位 x 排序
    lines.sort(key=lambda pts: np.median(pts[:, 1]))
    return lines


# ── 多项式拟合 ───────────────────────────────────────────────────────────────
def fit_curves(lines: list[np.ndarray], roi_y1: int, roi_y2: int
               ) -> list[np.poly1d]:
    """对每条车道线点集拟合 x = poly(y)，返回多项式列表。"""
    polys = []
    for pts in lines:
        ys_pts = pts[:, 0]
        xs_pts = pts[:, 1]
        try:
            coeffs = np.polyfit(ys_pts, xs_pts, POLY_DEG)
            polys.append(np.poly1d(coeffs))
        except Exception:
            pass
    return polys


# ── 曲率过滤 ─────────────────────────────────────────────────────────────────
def filter_by_curvature(polys: list[np.poly1d],
                        lines: list[np.ndarray],
                        tol: float = CURVATURE_TOL
                        ) -> tuple[list[np.poly1d], list[np.ndarray]]:
    """
    只保留二次项系数 a 与"密集区中位数"接近的曲线。

    原理（两遍中位数）：
      1. 第一遍：对全部 a 取粗略中位数（可能被噪声拉偏）
      2. 取离粗略中位数最近的 50% 作为"真实线密集区"
      3. 第二遍：在密集区内重算中位数，锁定在真实线中心
      4. 用 tol 过滤离中位数过远的曲线
    """
    if not polys:
        return polys, lines

    a_values = np.array([p.c[0] for p in polys])

    coarse = float(np.median(a_values))

    distances = np.abs(a_values - coarse)
    half = max(1, len(a_values) // 2)
    densest_idx = np.argsort(distances)[:half]

    median_a = float(np.median(a_values[densest_idx]))

    kept_polys = []
    kept_lines = []
    for poly, line, a in zip(polys, lines, a_values):
        if abs(a - median_a) <= tol:
            kept_polys.append(poly)
            kept_lines.append(line)
        else:
            print(f"  ✗ 剔除曲率异常线：a={a:.2e}  (median={median_a:.2e})")

    return kept_polys, kept_lines


# ── 最小间距过滤：合并重复检测的同一条线 ─────────────────────────────────────
def merge_close_lines(
    polys: list[np.poly1d],
    lines: list[np.ndarray],
    ref_y: float,
) -> tuple[list[np.poly1d], list[np.ndarray]]:
    """
    相邻两条线在 ref_y 处的 x 距离 < MIN_LANE_WIDTH 时，
    保留点数更多的那条，丢弃另一条。
    """
    if len(polys) < 2:
        return polys, lines

    discard = set()
    for i in range(len(polys) - 1):
        if i in discard:
            continue
        j = i + 1
        while j < len(polys):
            if j in discard:
                j += 1
                continue
            dist = abs(polys[i](ref_y) - polys[j](ref_y))
            if dist < MIN_LANE_WIDTH:
                drop = i if len(lines[i]) < len(lines[j]) else j
                discard.add(drop)
                print(f"  ↔ 合并过近线：[{i}]x={polys[i](ref_y):.0f} 与 [{j}]x={polys[j](ref_y):.0f}，间距={dist:.0f}px，丢弃[{drop}]")
            j += 1

    new_polys = [p for k, p in enumerate(polys) if k not in discard]
    new_lines = [l for k, l in enumerate(lines) if k not in discard]
    return new_polys, new_lines


# ── 多项式引导远端窄带扫描 ───────────────────────────────────────────────────
def refine_with_poly_guide(
    mask: np.ndarray,
    polys: list[np.poly1d],
    lines: list[np.ndarray],
    h: int,
    w: int,
) -> tuple[list[np.poly1d], list[np.ndarray]]:
    """
    沿每条已有多项式曲线，在远端 ROI 做窄带像素验证，
    把找到的点追加到点集后重新拟合。
    """
    roi_y1 = int(h * ROI_Y_START)
    roi_y2 = int(h * ROI_Y_END)
    roi_x1 = int(w * ROI_X_START)
    roi_x2 = int(w * ROI_X_END)
    far_thresh_y = roi_y1 + (roi_y2 - roi_y1) * ROI_FAR_SPLIT

    new_polys = []
    new_lines = []

    for poly_idx, (poly, pts) in enumerate(zip(polys, lines)):
        extra_pts = list(pts)
        hits = 0

        for abs_y in range(roi_y1, int(far_thresh_y), ROW_STEP):
            pred_x = poly(abs_y)
            x_lo = int(max(roi_x1, pred_x - POLY_SEARCH_R))
            x_hi = int(min(roi_x2, pred_x + POLY_SEARCH_R))
            if x_lo >= x_hi:
                continue

            row_slice = mask[abs_y, x_lo:x_hi]
            white_xs  = np.where(row_slice > 0)[0] + x_lo
            if len(white_xs) > 0:
                cx = int(np.mean(white_xs))
                extra_pts.append([float(abs_y), float(cx)])
                hits += 1
        print(f"  线[{poly_idx}] 远端扫描命中 {hits} 行（共扫 {(int(far_thresh_y)-roi_y1)//ROW_STEP} 行）")

        if len(extra_pts) < MIN_ROW_COUNT_FAR:
            new_polys.append(poly)
            new_lines.append(pts)
            continue

        arr = np.array(extra_pts, dtype=np.float32)
        try:
            coeffs = np.polyfit(arr[:, 0], arr[:, 1], POLY_DEG)
            new_polys.append(np.poly1d(coeffs))
            new_lines.append(arr)
        except Exception:
            new_polys.append(poly)
            new_lines.append(pts)

    return new_polys, new_lines


# ── 切线方向逐行追踪远端虚线 ─────────────────────────────────────────────────
def extend_to_far_field(
    mask: np.ndarray,
    polys: list[np.poly1d],
    lines: list[np.ndarray],
    h: int,
    w: int,
) -> tuple[list[np.poly1d], list[np.ndarray]]:
    """
    从近端轨迹的最高点出发，沿切线方向逐行向远端追踪虚线。

    每步只外推一行（ROW_STEP px），以实际检测到的像素更新当前 x，
    空隙期间沿最近 N 个检测点估计的斜率继续推进，误差不会累积。
    """
    roi_y1 = int(h * ROI_Y_START)
    roi_x1 = int(w * ROI_X_START)
    roi_x2 = int(w * ROI_X_END)

    new_polys = []
    new_lines = []

    for poly, pts in zip(polys, lines):
        top_idx = np.argmin(pts[:, 0])
        start_y = float(pts[top_idx, 0])
        start_x = float(pts[top_idx, 1])

        deriv = poly.deriv()
        slope = float(deriv(start_y))

        recent: list[tuple[float, float]] = [(start_y, start_x)]
        cx = start_x
        far_pts: list[list[float]] = []

        for abs_y in range(int(start_y) - ROW_STEP, roi_y1 - 1, -ROW_STEP):
            pred_x = cx + slope * (-ROW_STEP)
            x_lo = int(max(roi_x1, pred_x - EXTEND_SEARCH_R))
            x_hi = int(min(roi_x2, pred_x + EXTEND_SEARCH_R))

            if x_lo < x_hi:
                row_slice = mask[abs_y, x_lo:x_hi]
                white_xs = np.where(row_slice > 0)[0] + x_lo
            else:
                white_xs = np.array([])

            if len(white_xs) > 0:
                cx = float(np.mean(white_xs))
                far_pts.append([float(abs_y), cx])
                recent.append((float(abs_y), cx))
                if len(recent) > EXTEND_SLOPE_N:
                    recent.pop(0)
                if len(recent) >= 2:
                    dy = recent[-1][0] - recent[0][0]
                    dx = recent[-1][1] - recent[0][1]
                    slope = dx / dy if dy != 0 else slope
            else:
                cx = pred_x

        if not far_pts:
            new_polys.append(poly)
            new_lines.append(pts)
            continue

        far_arr = np.array(far_pts, dtype=np.float32)
        all_pts = np.vstack([pts, far_arr])
        weights = np.concatenate([
            np.ones(len(pts)),
            np.full(len(far_arr), EXTEND_FAR_W),
        ])
        try:
            coeffs = np.polyfit(all_pts[:, 0], all_pts[:, 1], POLY_DEG, w=weights)
            new_polys.append(np.poly1d(coeffs))
            new_lines.append(all_pts)
            print(f"  切线追踪：新增 {len(far_pts)} 个远端点（权重×{EXTEND_FAR_W}）")
        except Exception:
            new_polys.append(poly)
            new_lines.append(pts)

    return new_polys, new_lines


# ── 盲搜补全漏检虚线 ─────────────────────────────────────────────────────────
def blind_search_missing(
    mask: np.ndarray,
    polys: list[np.poly1d],
    lines: list[np.ndarray],
    h: int,
    w: int,
) -> tuple[list[np.poly1d], list[np.ndarray]]:
    """
    在相邻两条已知线中间做盲搜，找漏检的虚线。
    找到后插入到结果列表的对应位置。
    """
    if len(polys) < 2:
        return polys, lines

    roi_y1 = int(h * ROI_Y_START)
    roi_y2 = int(h * ROI_Y_END)
    roi_x1 = int(w * ROI_X_START)
    roi_x2 = int(w * ROI_X_END)

    insert_polys = list(polys)
    insert_lines = list(lines)
    offset = 0

    for i in range(len(polys) - 1):
        p_left  = polys[i]
        p_right = polys[i + 1]

        mid_pts = []
        for abs_y in range(roi_y1, roi_y2, ROW_STEP):
            mid_x  = (p_left(abs_y) + p_right(abs_y)) / 2.0
            x_lo   = int(max(roi_x1, mid_x - BLIND_SEARCH_R))
            x_hi   = int(min(roi_x2, mid_x + BLIND_SEARCH_R))
            if x_lo >= x_hi:
                continue

            row_slice = mask[abs_y, x_lo:x_hi]
            white_xs  = np.where(row_slice > 0)[0] + x_lo
            if len(white_xs) > 0:
                cx = int(np.mean(white_xs))
                if abs(cx - p_left(abs_y)) > BLIND_BOUNDARY_EXCL and abs(cx - p_right(abs_y)) > BLIND_BOUNDARY_EXCL:
                    mid_pts.append([float(abs_y), float(cx)])

        if len(mid_pts) < BLIND_MIN_ROWS:
            continue

        arr = np.array(mid_pts, dtype=np.float32)
        try:
            coeffs = np.polyfit(arr[:, 0], arr[:, 1], POLY_DEG)
            new_poly = np.poly1d(coeffs)
            insert_idx = i + 1 + offset
            insert_polys.insert(insert_idx, new_poly)
            insert_lines.insert(insert_idx, arr)
            offset += 1
            print(f"  ✓ 盲搜补全：在 L{i+1}/L{i+2} 之间找到新车道线")
        except Exception:
            pass

    return insert_polys, insert_lines


# ── 可视化 ───────────────────────────────────────────────────────────────────
def draw_lanes_calib(frame: np.ndarray,
                     polys: list,
                     lines: list[np.ndarray]) -> np.ndarray:
    """将标定辅助算法检测到的车道线可视化叠加到帧上。"""
    h, w = frame.shape[:2]
    roi_y1 = int(h * ROI_Y_START)
    roi_y2 = int(h * ROI_Y_END)
    roi_x1 = int(w * ROI_X_START)
    roi_x2 = int(w * ROI_X_END)

    ys = np.arange(roi_y1, roi_y2)

    overlay = frame.copy()

    for i in range(len(polys) - 1):
        p_left  = polys[i]
        p_right = polys[i + 1]

        xs_left  = np.clip(p_left(ys),  roi_x1, roi_x2).astype(np.int32)
        xs_right = np.clip(p_right(ys), roi_x1, roi_x2).astype(np.int32)

        if np.median(xs_right - xs_left) < 20:
            continue

        pts_left  = np.stack([xs_left,  ys], axis=1)
        pts_right = np.stack([xs_right, ys], axis=1)[::-1]
        poly_pts  = np.vstack([pts_left, pts_right]).astype(np.int32)

        color = LANE_COLORS[i % len(LANE_COLORS)]
        cv2.fillPoly(overlay, [poly_pts], color)

        cx = int(np.median((xs_left + xs_right) / 2))
        cy = (roi_y1 + roi_y2) // 2
        cv2.putText(overlay, f"L{i+1}", (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    vis = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

    for poly in polys:
        xs = np.clip(poly(ys), roi_x1, roi_x2).astype(np.int32)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], False, (0, 220, 255), 2)

    for pts in lines:
        for y, x in pts:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 80), -1)

    cv2.rectangle(vis, (roi_x1, roi_y1), (roi_x2, roi_y2), (120, 120, 120), 2)
    cv2.putText(vis, f"lane lines: {len(polys)}  lanes: {len(polys)-1}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return vis


# ── 主流程（标定辅助，命令行直接运行）────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="车道线标定辅助：基于背景中位数 + 行扫描拟合（非实时）"
    )
    parser.add_argument("--frame", type=int, default=2675)
    parser.add_argument("--video", default=_CALIB_VIDEO)
    args = parser.parse_args()

    print(f"构建背景模型（帧 {args.frame}）...")
    frame, bg = build_background(args.video, args.frame)
    if frame is None:
        print("读取帧失败"); return

    h, w = frame.shape[:2]
    bg_bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    mask   = white_mask_calib(bg_bgr)

    roi_y1 = int(h * ROI_Y_START)
    roi_y2 = int(h * ROI_Y_END)

    print("行扫描聚类...")
    lines = scan_lane_points(mask, h, w)
    print(f"检测到车道线: {len(lines)} 条")

    polys = fit_curves(lines, roi_y1, roi_y2)
    print(f"拟合曲线: {len(polys)} 条（过滤前）")

    polys, lines = filter_by_curvature(polys, lines)
    print(f"保留曲线: {len(polys)} 条（曲率过滤后）")

    ref_y = (roi_y1 + roi_y2) / 2
    polys, lines = merge_close_lines(polys, lines, ref_y)
    print(f"合并后: {len(polys)} 条")

    # 第二阶段：用 degree-3 重拟合（保留已识别的正确线，提升精度）
    polys = []
    for ln in lines:
        try:
            coeffs = np.polyfit(ln[:, 0], ln[:, 1], 3)
            polys.append(np.poly1d(coeffs))
        except Exception:
            coeffs = np.polyfit(ln[:, 0], ln[:, 1], 2)
            polys.append(np.poly1d(coeffs))
    print("degree-3 重拟合完成")

    print("切线方向追踪远端虚线...")
    polys, lines = extend_to_far_field(mask, polys, lines, h, w)
    print(f"追踪后曲线: {len(polys)} 条")

    # 最终换成平滑样条（避免高次多项式边界震荡）
    from scipy.interpolate import UnivariateSpline
    splines = []
    for ln in lines:
        order = np.argsort(ln[:, 0])
        ys_ln, xs_ln = ln[order, 0], ln[order, 1]
        _, uid = np.unique(ys_ln, return_index=True)
        ys_u, xs_u = ys_ln[uid], xs_ln[uid]
        s_val = SPLINE_S_PER_PT * len(ys_u)
        try:
            sp = UnivariateSpline(ys_u, xs_u, k=3, s=s_val)
        except Exception:
            sp = np.poly1d(np.polyfit(ys_u, xs_u, 2))
        splines.append(BoundedCurve(sp, float(ys_u[0]), float(ys_u[-1])))
    polys = splines

    # print("盲搜补全漏检线...")
    # polys, lines = blind_search_missing(mask, polys, lines, h, w)
    # polys, lines = filter_by_curvature(polys, lines, tol=CURVATURE_TOL * 1.5)
    print(f"最终曲线: {len(polys)} 条  →  划分车道: {len(polys)-1} 个")

    vis = draw_lanes_calib(frame, polys, lines)

    cv2.imwrite(str(OUTPUT_DIR / f"lane_bg_f{args.frame}.jpg"),   bg_bgr)
    cv2.imwrite(str(OUTPUT_DIR / f"lane_mask_f{args.frame}.jpg"), mask)
    out = OUTPUT_DIR / f"lane_divide_f{args.frame}.jpg"
    cv2.imwrite(str(out), vis)
    print(f"已保存: {out}")


if __name__ == "__main__":
    main()
