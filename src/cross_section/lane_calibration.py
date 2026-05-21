"""
车道标定管理模块。

启动流程：
  1. 解析视频文件名 → 提取进口名和起止时间
  2. 查找 calibrations/{entrance}/lanes.json
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
    """查找 calibrations/{entrance}/lanes.json。

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
def _compute_homography(bg_image: np.ndarray) -> tuple[np.ndarray | None, str]:
    """优先用 ZebraDetector 自动算 H，失败则进入备用标注流程。"""
    print("[H] 尝试自动检测斑马线...")
    zresult = ZebraDetector().detect(bg_image)
    if zresult is not None:
        H, n_stripes, _rects = zresult
        print(f"[H] ✓ 自动检测成功（{n_stripes} 条条纹）")
        return H, "auto_zebra"

    print("[H] ⚠ 自动检测失败，进入备用人工标定")
    manual = annotate_homography(bg_image)
    if manual is None:
        print("[H] ✗ 备用标定也取消，将使用 PIXELS_PER_METER 兜底")
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
    """保存标定到 calibrations/{entrance}/。"""
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

    H, method = _compute_homography(bg_bgr)
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
