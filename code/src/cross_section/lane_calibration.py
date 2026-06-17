"""
车道标定管理模块。

启动流程：
  1. 解析视频文件名 → 提取进口名和起止时间
  2. 查找标定目录中的 {entrance}/lanes.json
  3. 有 → 直接加载返回
  4. 没有 → 跑背景建模 + 交互标注 → 保存

主要接口：
  parse_video_filename(path)  → dict | None
  find_calibration(entrance)  → dict | None
  save_calibration(...)
  get_calibration(video_path, auto_annotate=True) → CalibrationData

独立运行：
  python3 -m src.cross_section.lane_calibration --video <path>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.config.settings import (
    CALIBRATIONS_DIR,
    ENTRANCE_ALIASES,
    VIDEO_PATH,
)
from src.cross_section.lane_annotator import (
    annotate as run_annotator,
    annotate_homography,
)
from src.cross_section.zebra_detector import ZebraDetector
from src.cross_section.lane_detector import build_background
from src.cross_section.lane_lighting import decide_preset


_FILENAME_RE = re.compile(
    r'(北进口|南进口|东进口|north|south|east)'
    r'_?(\d{14})至(\d{14})',
    re.IGNORECASE,
)


@dataclass
class CalibrationData:
    """运行时持有的标定信息。"""
    entrance: str                          # 规范化中文进口名
    lanes: dict[int, list[tuple[int, int]]]  # 车道线点集
    ref_image: np.ndarray                  # 背景参考图（用于可视化对比）
    metadata: dict                         # 元信息（标定时间、源视频等）
    start_time: datetime | None = None     # 当前视频起始时间（从文件名解析）
    end_time:   datetime | None = None
    lighting_preset: str = 'off_peak'      # 根据时段+画面综合判断的光照预设
    homography: np.ndarray | None = None
    homography_method: str = "fallback_ppm"


# ── 文件名解析 ───────────────────────────────────────────────────────────────
def parse_video_filename(path: str | Path) -> dict | None:
    """从视频文件名解析进口与时间。

    支持格式：
      北进口_20260420075959至20260420081500.mp4
      南进口20260420075959至20260420081500.mp4
      south_20260420075959至20260420081500.mp4

    Returns:
        {'entrance': '北进口',
         'start': datetime(2026,4,20,7,59,59),
         'end':   datetime(2026,4,20,8,15,00)}
        或 None（不匹配）
    """
    p = Path(path)
    m = _FILENAME_RE.search(p.stem)
    if not m:
        return None
    raw_entrance = m.group(1)
    entrance = ENTRANCE_ALIASES.get(raw_entrance.lower(),
                                     ENTRANCE_ALIASES.get(raw_entrance))
    if not entrance:
        return None
    try:
        start = datetime.strptime(m.group(2), '%Y%m%d%H%M%S')
        end   = datetime.strptime(m.group(3), '%Y%m%d%H%M%S')
    except ValueError:
        return None
    return {'entrance': entrance, 'start': start, 'end': end}


# ── 已有标定加载 ────────────────────────────────────────────────────────────
def find_calibration(entrance: str) -> dict | None:
    """查找标定目录中的 {entrance}/lanes.json。

    兼容两种 lanes.json 格式：
      - 带顶层 'lanes' 键的包装格式（来自 lane_annotator 输出）：
          {"image": "...", "scale": 0.4, "lanes": {"1": [...], ...}}
      - 裸 dict 格式：
          {"1": [...], "2": [...], ...}

    Returns:
        {'lanes': {...}, 'ref_image': ndarray, 'metadata': {...}}
        或 None
    """
    entrance = ENTRANCE_ALIASES.get(entrance, entrance)
    cal_dir = CALIBRATIONS_DIR / entrance
    lanes_path    = cal_dir / 'lanes.json'
    ref_path      = cal_dir / 'ref.jpg'
    metadata_path = cal_dir / 'metadata.json'

    if not lanes_path.exists():
        return None

    lanes_raw = json.loads(lanes_path.read_text(encoding='utf-8'))
    # 兼容包装格式（有 'lanes' 键）和裸 dict 格式
    raw_lanes_dict = lanes_raw.get('lanes', lanes_raw)
    # 过滤掉非 int 可转换的键（例如 'image'、'scale' 等残留顶层字段）
    lanes = {}
    for k, v in raw_lanes_dict.items():
        try:
            lane_id = int(k)
        except (ValueError, TypeError):
            continue
        lanes[lane_id] = [tuple(p) for p in v]

    ref_image = cv2.imread(str(ref_path)) if ref_path.exists() else None
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) \
               if metadata_path.exists() else {}

    H, method = _load_homography(cal_dir)
    return {
        'lanes': lanes,
        'ref_image': ref_image,
        'metadata': metadata,
        'homography': H,
        'homography_method': method,
    }


# ── 单应矩阵辅助函数 ────────────────────────────────────────────────────────

# 标准机动车道宽度（米，GB/T 14886）
LANE_WIDTH_M = 3.5

# 路面导向箭头国标尺寸（GB 5768.3-2009，60km/h 设计速度）
ARROW_LONG_M  = 6.0   # 箭头总长（米）
ARROW_SHORT_M = 0.4   # 箭头轴宽（米）


def _lane_annotation_homography(
    bg_image: np.ndarray,
    lanes: dict[int, list[tuple[int, int]]],
) -> np.ndarray | None:
    """用已标注车道线 + 用户输入纵向距离计算单应矩阵 H。

    横向：相邻车道线间距 = LANE_WIDTH_M（3.5m，国标）
    纵向：用户输入"标注区域沿行驶方向覆盖多少米"

    取相邻两条标注最完整的线，在 y_near 和 y_far 处各取一点，
    共 4 个像素点对应世界矩形，调用 findHomography。

    Returns:
        H（3×3）或 None
    """
    from scipy.interpolate import UnivariateSpline

    # 找点数最多的两条相邻线
    sorted_lanes = sorted(
        [(lid, pts) for lid, pts in lanes.items() if len(pts) >= 4],
        key=lambda t: t[0],
    )
    if len(sorted_lanes) < 2:
        print("[H] 车道线不足 2 条，无法用车道线标定")
        return None

    lid1, pts1 = sorted_lanes[0]
    lid2, pts2 = sorted_lanes[1]

    def make_spline(pts):
        arr = np.array(pts, dtype=np.float64)
        ys, xs = arr[:, 1], arr[:, 0]
        order = np.argsort(ys)
        ys_u, xs_u = ys[order], xs[order]
        _, uid = np.unique(ys_u, return_index=True)
        ys_u, xs_u = ys_u[uid], xs_u[uid]
        return UnivariateSpline(ys_u, xs_u, k=3, s=200 * len(ys_u)), float(ys_u[0]), float(ys_u[-1])

    sp1, y1_min, y1_max = make_spline(pts1)
    sp2, y2_min, y2_max = make_spline(pts2)

    y_far  = max(y1_min, y2_min)   # 远端（共同覆盖的顶端）
    y_near = min(y1_max, y2_max)   # 近端（共同覆盖的底端）

    if y_near - y_far < 50:
        print("[H] 两条车道线 y 重叠范围太小，无法标定")
        return None

    # 4 个像素坐标
    src = np.float32([
        [sp1(y_near), y_near],   # 左近
        [sp2(y_near), y_near],   # 右近
        [sp2(y_far),  y_far ],   # 右远
        [sp1(y_far),  y_far ],   # 左远
    ])

    # 横向世界距离（已知）
    lat_m = LANE_WIDTH_M

    # 纵向世界距离（用户输入）
    print(f"\n[H 车道线标定] 使用线 {lid1} 和线 {lid2}")
    print(f"  标注 y 范围：{y_far:.0f}（远端）→ {y_near:.0f}（近端）")
    print(f"  请估算这段路（远端到近端）沿车辆行驶方向的实际长度")
    try:
        long_m = float(input("  纵向距离（米，例如 30）: "))
    except (ValueError, EOFError):
        print("[H] 输入无效，取消")
        return None

    if long_m <= 0:
        print("[H] 距离必须为正数")
        return None

    dst = np.float32([
        [0,     0      ],   # 左近
        [lat_m, 0      ],   # 右近
        [lat_m, long_m ],   # 右远
        [0,     long_m ],   # 左远
    ])

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        print("[H] findHomography 计算失败")
    else:
        print(f"[H] ✓ 车道线标定完成（横向 {lat_m}m × 纵向 {long_m}m）")
    return H


def _detect_arrow_homography(bg_image: np.ndarray) -> np.ndarray | None:
    """全图检测路面导向箭头，用国标尺寸计算单应矩阵 H。

    检测流程：
    1. HSV 白色掩码 + 形态学清理
    2. 轮廓过滤（面积 + 凸包填充率 + 凸缺陷数）
    3. 取最大面积候选，求最小外接旋转矩形（minAreaRect）
    4. 长边 → ARROW_LONG_M，短边 → ARROW_SHORT_M
    5. findHomography

    Returns:
        H（3×3）或 None（未检出或计算失败）
    """
    import cv2 as _cv2

    hsv  = _cv2.cvtColor(bg_image, _cv2.COLOR_BGR2HSV)
    mask = _cv2.inRange(hsv, (0, 0, 155), (180, 40, 255))
    k    = _cv2.getStructuringElement(_cv2.MORPH_RECT, (3, 3))
    mask = _cv2.erode(mask,  k, iterations=1)
    mask = _cv2.dilate(mask, k, iterations=2)

    contours, _ = _cv2.findContours(mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = _cv2.contourArea(cnt)
        if not (2000 <= area <= 80000):
            continue
        hull      = _cv2.convexHull(cnt)
        hull_area = _cv2.contourArea(hull)
        solidity  = area / hull_area if hull_area > 0 else 0
        if not (0.35 <= solidity <= 0.88):
            continue
        hull_idx = _cv2.convexHull(cnt, returnPoints=False)
        try:
            defects = _cv2.convexityDefects(cnt, hull_idx)
        except Exception:
            continue
        if defects is None:
            continue
        n_def = sum(1 for d in defects if d[0][3] / 256.0 > 8)
        if n_def < 1:
            continue
        candidates.append((area, cnt))

    if not candidates:
        return None

    # 取面积最大的候选
    _, best_cnt = max(candidates, key=lambda t: t[0])

    # 最小外接旋转矩形 → 4 角点
    rect   = _cv2.minAreaRect(best_cnt)
    box    = _cv2.boxPoints(rect).astype(np.float32)  # shape (4,2)
    w_px, h_px = rect[1]   # 宽、高（可能需要对调以确定长边）

    if w_px < h_px:          # 确保 w_px 是长边
        w_px, h_px = h_px, w_px
        # box 顺序也随之调整（旋转 90°）
        box = np.roll(box, 1, axis=0)

    long_m, short_m = ARROW_LONG_M, ARROW_SHORT_M

    # 目标世界坐标（以箭头左下角为原点，沿车道方向为 y 轴）
    dst_pts = np.float32([
        [0,       0      ],
        [short_m, 0      ],
        [short_m, long_m ],
        [0,       long_m ],
    ])

    H, mask_h = _cv2.findHomography(box, dst_pts, _cv2.RANSAC, 5.0)
    return H


def _compute_homography(
    bg_image: np.ndarray,
    lanes: dict[int, list[tuple[int, int]]] | None = None,
) -> tuple[np.ndarray | None, str]:
    """优先斑马线，其次车道线标定，最后人工 4 点备用。"""
    # 1. 斑马线自动检测（≥3 条纹 且 y 跨度 < 画面高度 30%）
    print("[H] 尝试自动检测斑马线...")
    zresult = ZebraDetector(min_stripes=3).detect(bg_image)
    if zresult is not None:
        H, n_stripes, rects = zresult
        img_h = bg_image.shape[0]
        y_vals = [r[1] for r in rects]
        y_span = (max(y_vals) - min(y_vals)) / img_h
        if y_span < 0.30:
            print(f"[H] ✓ 斑马线检测成功（{n_stripes} 条条纹，y 跨度 {y_span:.1%}）")
            return H, "auto_zebra"
        else:
            print(f"[H] ⚠ 检测到 {n_stripes} 条纹但分布太散（y 跨度 {y_span:.1%}），忽略")

    # 2. 已标注车道线 + 用户输入纵向距离
    if lanes:
        print("[H] 尝试用已标注车道线标定（横向 3.5m + 纵向距离输入）...")
        H = _lane_annotation_homography(bg_image, lanes)
        if H is not None:
            return H, "lane_annotation"

    # 3. 人工 4 点备用
    print("[H] ⚠ 进入备用人工 4 点标定")
    manual = annotate_homography(bg_image)
    if manual is None:
        print("[H] ✗ 备用标定取消，将使用 PIXELS_PER_METER 兜底")
        return None, "fallback_ppm"
    return manual['H'], "manual"


def _save_homography(cal_dir: Path, H: np.ndarray, method: str, n_stripes: int = 0) -> None:
    """保存 H 矩阵到 homography.json。"""
    data = {
        'H': H.tolist(),
        'method': method,
        'n_stripes': n_stripes,
        'calibration_date': datetime.now().isoformat(),
    }
    (cal_dir / 'homography.json').write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )


def _load_homography(cal_dir: Path) -> tuple[np.ndarray | None, str]:
    """从 homography.json 读取 H。失败返回 (None, 'fallback_ppm')。"""
    path = cal_dir / 'homography.json'
    if not path.exists():
        return None, 'fallback_ppm'
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return np.array(data['H'], dtype=np.float64), data.get('method', 'unknown')
    except Exception:
        return None, 'fallback_ppm'


# ── 保存标定 ────────────────────────────────────────────────────────────────
def save_calibration(entrance: str,
                     lanes: dict[int, list[tuple[int, int]]],
                     ref_image: np.ndarray,
                     metadata: dict) -> Path:
    """保存标定到标定目录的 {entrance}/。"""
    entrance = ENTRANCE_ALIASES.get(entrance, entrance)
    cal_dir = CALIBRATIONS_DIR / entrance
    cal_dir.mkdir(parents=True, exist_ok=True)

    (cal_dir / 'lanes.json').write_text(
        json.dumps({'lanes': {str(k): list(v) for k, v in lanes.items()}},
                   indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    cv2.imwrite(str(cal_dir / 'ref.jpg'), ref_image)
    (cal_dir / 'metadata.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding='utf-8',
    )
    return cal_dir


# ── 主入口：编排所有部分 ────────────────────────────────────────────────────
def get_calibration(video_path: str | Path,
                    auto_annotate: bool = True) -> CalibrationData | None:
    """标定主入口。

    Args:
        video_path:    待处理视频路径
        auto_annotate: 找不到标定时是否自动启动标注工具

    Returns:
        CalibrationData（含车道线、参考图、元信息、光照预设）
        None（无法识别进口或用户取消标注）
    """
    video_path = Path(video_path)
    parsed = parse_video_filename(video_path)

    if not parsed:
        print(f"⚠ 文件名 {video_path.name} 不符合"
              f"'北/南/东进口_YYYYMMDDHHMMSS至YYYYMMDDHHMMSS' 格式")
        return None

    entrance   = parsed['entrance']
    start_time = parsed['start']
    end_time   = parsed['end']
    print(f"✓ 解析视频：进口={entrance}  时段={start_time} → {end_time}")

    # 加载已有标定
    existing = find_calibration(entrance)
    if existing is not None:
        print(f"✓ 已加载 {entrance} 的标定（{len(existing['lanes'])} 条车道线）")
        # 顺便判断光照
        first_frame, _ = build_background(str(video_path), target_frame=100)
        preset = decide_preset(start_time, first_frame)
        H = existing.get('homography')
        method = existing.get('homography_method', 'fallback_ppm')
        if H is None:
            print(f"⚠ {entrance} 暂无 H 矩阵，使用 PIXELS_PER_METER 兜底")
        else:
            print(f"✓ H 矩阵已加载（method={method}）")
        return CalibrationData(
            entrance=entrance,
            lanes=existing['lanes'],
            ref_image=existing['ref_image'],
            metadata=existing['metadata'],
            start_time=start_time,
            end_time=end_time,
            lighting_preset=preset,
            homography=H,
            homography_method=method,
        )

    # 无标定 → 启动标注流程
    if not auto_annotate:
        print(f"✗ 未找到 {entrance} 的标定，且 auto_annotate=False")
        return None

    print(f"⚠ 未找到 {entrance} 的标定，进入交互式标注模式...")
    print("   1. 正在生成背景图（消除运动车辆）...")
    first_frame, bg_gray = build_background(str(video_path), target_frame=100)
    if first_frame is None or bg_gray is None:
        print("✗ 背景建模失败")
        return None

    bg_bgr = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2BGR)
    print("   2. 启动标注工具...")
    lanes = run_annotator(bg_bgr, n_lanes=4)
    if lanes is None or not any(lanes.values()):
        print("✗ 用户取消标注")
        return None

    metadata = {
        'entrance':         entrance,
        'calibration_date': datetime.now().isoformat(),
        'source_video':     video_path.name,
        'source_frame':     100,
        'lane_count':       sum(1 for v in lanes.values() if v),
    }
    cal_dir = save_calibration(entrance, lanes, bg_bgr, metadata)
    print(f"✓ 车道线已保存到 {cal_dir}")

    H, method = _compute_homography(bg_bgr, lanes=lanes)
    if H is not None:
        _save_homography(cal_dir, H, method)
        print(f"✓ H 矩阵已保存（method={method}）")

    preset = decide_preset(start_time, first_frame)
    return CalibrationData(
        entrance=entrance,
        lanes=lanes,
        ref_image=bg_bgr,
        metadata=metadata,
        start_time=start_time,
        end_time=end_time,
        lighting_preset=preset,
        homography=H,
        homography_method=method,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="车道标定管理：解析视频名 → 加载/创建标定"
    )
    parser.add_argument("--video", default=str(VIDEO_PATH),
                        help="视频文件路径")
    parser.add_argument("--no-annotate", action="store_true",
                        help="不触发交互标注（仅查询）")
    args = parser.parse_args()

    cal = get_calibration(args.video, auto_annotate=not args.no_annotate)
    if cal is None:
        print("✗ 无标定数据")
        sys.exit(1)

    print(f"\n=== 标定信息 ===")
    print(f"进口:       {cal.entrance}")
    print(f"时段:       {cal.start_time} → {cal.end_time}")
    print(f"光照预设:    {cal.lighting_preset}")
    print(f"车道线数:    {len(cal.lanes)}")
    for lid, pts in sorted(cal.lanes.items()):
        print(f"  线{lid}: {len(pts)} 点")
    print(f"H 矩阵:     {'已加载（' + cal.homography_method + '）' if cal.homography is not None else '无（兜底 PPM）'}")


if __name__ == "__main__":
    main()
