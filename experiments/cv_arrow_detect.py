"""
传统 CV 路面箭头检测 —— 独立可视化脚本，不修改任何已有代码。

思路：
  1. ROI 裁切（排除天空、建筑、图像边缘）
  2. 白色掩码（路面标线为白色）
  3. 形态学清理
  4. 轮廓过滤（面积 + 凸缺陷数量，箭头形状有特定凸缺陷）
  5. 绘制候选区域可视化

运行：
  python3 cv_arrow_detect.py
  python3 cv_arrow_detect.py --frame 500   # 指定帧号
"""
import argparse
import cv2
import numpy as np
from pathlib import Path

VIDEO   = "test_video_fixed.mp4"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# ── 参数 ──────────────────────────────────────────────────────────────────────
# ROI：取画面中下部分（路面区域），排除天空和远景
ROI_Y_START = 0.1   # 从画面高度 35% 开始
ROI_Y_END   = 0.35
ROI_X_START = 0.05
ROI_X_END   = 0.85   # 右侧绿化带排除

# 白色掩码阈值（HSV）
WHITE_V_MIN = 155    # 亮度下限（灰度背景图适当降低）
WHITE_S_MAX = 40     # 饱和度上限

# 轮廓过滤
AREA_MIN       = 2000   # 最小面积（像素²），过滤噪声
AREA_MAX       = 40000  # 最大面积，过滤大块连通区域
SOLIDITY_MIN   = 0.35   # 凸包填充率下限（箭头有缺口）
SOLIDITY_MAX   = 0.88   # 凸包填充率上限（实心矩形排除）
DEFECT_MIN     = 1      # 凸缺陷数量下限（箭头至少1个缺陷）


def white_mask(frame: np.ndarray) -> np.ndarray:
    """提取路面白色标线掩码。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       (0,   0,   WHITE_V_MIN),
                       (180, WHITE_S_MAX, 255))
    # 形态学：先腐蚀去噪，再膨胀恢复形状
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask,  kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def count_convexity_defects(cnt) -> int:
    """计算轮廓的凸缺陷数量（箭头因箭尾缺口会有明显缺陷）。"""
    if len(cnt) < 5:
        return 0
    hull   = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0
    try:
        defects = cv2.convexityDefects(cnt, hull)
    except Exception:
        return 0
    if defects is None:
        return 0
    # 只统计深度足够大的缺陷（过滤微小毛刺）
    significant = sum(1 for d in defects if d[0][3] / 256.0 > 8)
    return significant


def detect_arrows(frame: np.ndarray,
                  override_mask: np.ndarray | None = None) -> tuple[np.ndarray, list[dict]]:
    """在帧上检测箭头候选轮廓，返回可视化图和候选列表。"""
    h, w = frame.shape[:2]
    vis  = frame.copy()

    # ① ROI 掩码
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    y1 = int(h * ROI_Y_START)
    y2 = int(h * ROI_Y_END)
    x1 = int(w * ROI_X_START)
    x2 = int(w * ROI_X_END)
    roi_mask[y1:y2, x1:x2] = 255

    # ② 白色掩码（或外部传入的组合掩码）
    if override_mask is not None:
        wmask = override_mask
    else:
        wmask = white_mask(frame)
        wmask = cv2.bitwise_and(wmask, roi_mask)

    # ③ 找轮廓
    contours, _ = cv2.findContours(wmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (AREA_MIN <= area <= AREA_MAX):
            continue

        hull    = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity  = area / hull_area if hull_area > 0 else 0

        if not (SOLIDITY_MIN <= solidity <= SOLIDITY_MAX):
            continue

        n_defects = count_convexity_defects(cnt)
        if n_defects < DEFECT_MIN:
            continue

        rx, ry, rw, rh = cv2.boundingRect(cnt)
        aspect = rw / rh if rh > 0 else 0

        candidates.append({
            "cnt":      cnt,
            "area":     int(area),
            "solidity": round(solidity, 2),
            "defects":  n_defects,
            "bbox":     (rx, ry, rw, rh),
            "aspect":   round(aspect, 2),
        })

    # ④ 可视化
    # 白色掩码叠加（半透明蓝色）
    mask_color = np.zeros_like(frame)
    mask_color[wmask > 0] = (180, 120, 0)
    vis = cv2.addWeighted(vis, 1.0, mask_color, 0.3, 0)

    # ROI 边界
    cv2.rectangle(vis, (x1, y1), (x2, y2), (100, 100, 100), 2)

    # 候选轮廓
    for c in candidates:
        rx, ry, rw, rh = c["bbox"]
        # 轮廓填充
        cv2.drawContours(vis, [c["cnt"]], -1, (0, 255, 0), 2)
        # 外接矩形
        cv2.rectangle(vis, (rx, ry), (rx+rw, ry+rh), (0, 200, 255), 1)
        # 标注
        label = f"a={c['area']} s={c['solidity']} d={c['defects']}"
        cv2.putText(vis, label, (rx, ry - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    cv2.putText(vis, f"candidates: {len(candidates)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return vis, candidates


def build_static_mask(video_path: str, target_frame: int,
                       n_samples: int = 15, spread: int = 150) -> np.ndarray:
    """
    用多帧取中位数构建背景图，再与目标帧做差，得到静止区域掩码。
    静止的路面标线在各帧中像素值稳定，运动车辆像素值变化大。
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 在目标帧附近均匀采样
    indices = np.linspace(
        max(0, target_frame - spread),
        min(total - 1, target_frame + spread),
        n_samples, dtype=int
    )

    frames_gray = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, f = cap.read()
        if ret:
            frames_gray.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()

    if not frames_gray:
        return None

    # 中位数背景（消除运动车辆）
    stack   = np.stack(frames_gray, axis=0)
    bg      = np.median(stack, axis=0).astype(np.uint8)

    # 目标帧与背景做差：差值小 = 静止区域
    cap2 = cv2.VideoCapture(video_path)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    _, frame = cap2.read()
    cap2.release()

    cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff     = cv2.absdiff(cur_gray, bg)
    # 差值小于阈值的区域 = 静止（路面标线）
    _, static_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY_INV)

    return frame, bg, static_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, default=0, help="帧号")
    parser.add_argument("--video", default=VIDEO)
    args = parser.parse_args()

    print("构建背景模型（多帧中位数）...")
    result = build_static_mask(args.video, args.frame)
    if result is None:
        print("读取帧失败")
        return
    frame, bg, static_mask = result

    # 直接在背景图（灰度→BGR）上检测，车辆已消除
    bg_bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(OUT_DIR / f"cv_bg_f{args.frame}.jpg"), bg_bgr)

    vis, candidates = detect_arrows(bg_bgr)

    out_path = OUT_DIR / f"cv_arrow_f{args.frame}.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"检测候选数: {len(candidates)}")
    for i, c in enumerate(candidates):
        print(f"  [{i}] bbox={c['bbox']}  area={c['area']}  "
              f"solidity={c['solidity']}  defects={c['defects']}  aspect={c['aspect']}")
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
