"""
基于背景白色标线划分车道（曲线拟合版）。

思路：
  1. 中位数背景消除运动车辆，得到静态路面标线
  2. 白色掩码提取标线
  3. 对 ROI 每一行扫描白色像素 x 坐标，按列近邻聚类成各条车道线
  4. 对每条线做 x = poly(y) 二次多项式拟合（随透视变形弯曲）
  5. 用曲线多边形填色划分车道区域，叠加车道编号

运行：
  python3 cv_lane_divide.py
  python3 cv_lane_divide.py --frame 2675
"""
import argparse
import cv2
import numpy as np
from pathlib import Path

VIDEO   = "test_video_fixed.mp4"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

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
CLUSTER_GAP    = 60     # 同行内超过此距离视为不同车道线（px）
TRACK_GAP      = 80     # 跨行追踪：上一行中心与当前中心距离阈值（px）
MIN_ROW_COUNT  = 40     # 一条车道线至少出现的行数（过滤噪声）
MIN_ROW_COUNT_FAR = 15  # 远端（ROI 上半段）最少行数
ROI_FAR_SPLIT     = 0.5 # ROI y 范围内远/近分界（0 = 顶部，1 = 底部）

# ── 多项式次数 ───────────────────────────────────────────────────────────────
POLY_DEG = 2

# ── 曲率过滤：二次项系数 a 允许偏离中位数的最大值 ────────────────────────────
CURVATURE_TOL = 1e-3    # 单位：px^-1，实测 a ~ 1e-3，容差约 ±30%
POLY_SEARCH_R = 25   # 多项式引导搜索横向半径（px）
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


def white_mask(img_bgr: np.ndarray) -> np.ndarray:
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

        matched = set()
        for cx in groups:
            best_id, best_dist = None, TRACK_GAP
            for tid, tk in tracks.items():
                d = abs(cx - tk["cx"])
                if d < best_dist:
                    best_dist, best_id = d, tid
            if best_id is not None:
                tracks[best_id]["pts"].append((abs_y, cx))
                tracks[best_id]["cx"] = cx
                matched.add(best_id)
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
        # 用轨迹点的中位 y 判断远/近端
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
    ys    = np.arange(roi_y1, roi_y2)
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
    只保留二次项系数 a 与中位数接近的曲线。

    原理：
      x = a·y² + b·y + c 中，a 决定弯曲方向和程度。
      同一路口所有车道线透视弯曲方向一致，所以真实车道线的 a 应聚集在
      一个窄区间里。取所有 a 的中位数，把 |a - median_a| > tol 的踢掉。
    """
    if not polys:
        return polys, lines

    # 提取每条曲线的二次项系数（poly1d 系数从高次到低次排列）
    # POLY_DEG=2 时：coeffs = [a, b, c]，故 poly.c[0] 就是 a
    a_values = np.array([p.c[0] for p in polys])
    median_a = np.median(a_values)

    kept_polys = []
    kept_lines = []
    for poly, line, a in zip(polys, lines, a_values):
        if abs(a - median_a) <= tol:
            kept_polys.append(poly)
            kept_lines.append(line)
        else:
            print(f"  ✗ 剔除曲率异常线：a={a:.2e}  (median={median_a:.2e})")

    return kept_polys, kept_lines


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

    for poly, pts in zip(polys, lines):
        extra_pts = list(pts)  # 复制原有点集

        # 只在远端（y < far_thresh_y）做引导搜索
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
    offset = 0  # 插入后索引偏移

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
                # 排除贴近已知线的点（避免把边界线误认为中间线）
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
def draw_lanes(frame: np.ndarray,
               polys: list[np.poly1d],
               lines: list[np.ndarray]) -> np.ndarray:
    h, w = frame.shape[:2]
    roi_y1 = int(h * ROI_Y_START)
    roi_y2 = int(h * ROI_Y_END)
    roi_x1 = int(w * ROI_X_START)
    roi_x2 = int(w * ROI_X_END)

    ys = np.arange(roi_y1, roi_y2)

    overlay = frame.copy()

    # 车道 = 相邻两条检测线夹住的区域（不用 ROI 边缘凑数）
    for i in range(len(polys) - 1):
        p_left  = polys[i]
        p_right = polys[i + 1]

        xs_left  = np.clip(p_left(ys),  roi_x1, roi_x2).astype(np.int32)
        xs_right = np.clip(p_right(ys), roi_x1, roi_x2).astype(np.int32)

        # 太窄跳过
        if np.median(xs_right - xs_left) < 20:
            continue

        # 多边形点：左边界从上到下，右边界从下到上
        pts_left  = np.stack([xs_left,  ys], axis=1)
        pts_right = np.stack([xs_right, ys], axis=1)[::-1]
        poly_pts  = np.vstack([pts_left, pts_right]).astype(np.int32)

        color = LANE_COLORS[i % len(LANE_COLORS)]
        cv2.fillPoly(overlay, [poly_pts], color)

        # 车道编号（中心位置）
        cx = int(np.median((xs_left + xs_right) / 2))
        cy = (roi_y1 + roi_y2) // 2
        cv2.putText(overlay, f"L{i+1}", (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    vis = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

    # 绘制拟合曲线（黄色）
    for poly in polys:
        xs = np.clip(poly(ys), roi_x1, roi_x2).astype(np.int32)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], False, (0, 220, 255), 2)

    # 原始扫描点（绿色小点，调试用）
    for pts in lines:
        for y, x in pts:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 80), -1)

    # ROI 框
    cv2.rectangle(vis, (roi_x1, roi_y1), (roi_x2, roi_y2), (120, 120, 120), 2)
    cv2.putText(vis, f"lane lines: {len(polys)}  lanes: {len(polys)-1}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return vis


# ── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, default=2675)
    parser.add_argument("--video", default=VIDEO)
    args = parser.parse_args()

    print(f"构建背景模型（帧 {args.frame}）...")
    frame, bg = build_background(args.video, args.frame)
    if frame is None:
        print("读取帧失败"); return

    h, w = frame.shape[:2]
    bg_bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    mask   = white_mask(bg_bgr)

    roi_y1 = int(h * ROI_Y_START)
    roi_y2 = int(h * ROI_Y_END)

    print("行扫描聚类...")
    lines = scan_lane_points(mask, h, w)
    print(f"检测到车道线: {len(lines)} 条")

    polys = fit_curves(lines, roi_y1, roi_y2)
    print(f"拟合曲线: {len(polys)} 条（过滤前）")

    polys, lines = filter_by_curvature(polys, lines)
    print(f"保留曲线: {len(polys)} 条（曲率过滤后）")

    print("多项式引导远端扫描...")
    polys, lines = refine_with_poly_guide(mask, polys, lines, h, w)
    print(f"精修后曲线: {len(polys)} 条")

    print("盲搜补全漏检线...")
    polys, lines = blind_search_missing(mask, polys, lines, h, w)
    # 补全后再过一次曲率过滤，剔除噪声
    polys, lines = filter_by_curvature(polys, lines, tol=CURVATURE_TOL * 1.5)
    print(f"最终曲线: {len(polys)} 条  →  划分车道: {len(polys)-1} 个")

    vis = draw_lanes(frame, polys, lines)

    cv2.imwrite(str(OUT_DIR / f"lane_bg_f{args.frame}.jpg"),   bg_bgr)
    cv2.imwrite(str(OUT_DIR / f"lane_mask_f{args.frame}.jpg"), mask)
    out = OUT_DIR / f"lane_divide_f{args.frame}.jpg"
    cv2.imwrite(str(out), vis)
    print(f"已保存: {out}")


if __name__ == "__main__":
    main()
