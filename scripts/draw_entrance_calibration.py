"""在相机参考帧 ref.jpg 上绘制某个进口的车道标定示意图。

读取 {calib_dir}/ 下的 ref.jpg、lanes.json、stop_lines.json、sections.json，
按统一图例样式叠加：
  - 车道边界   蓝色实线（lanes.json 每条折线）
  - 车道中心线 黄色虚线（相邻边界线插值取中）
  - 停止线     红色实线（stop_lines.json）
  - 断面线     绿色虚线（sections.json）
  - 导向箭头   白色（沿各车道中心线指向停止线方向）
  - 关键点     红色菱形（边界远端、停止线两端、主断面两端）

坐标均为参考帧原始分辨率下的像素坐标。

用法：
  python3 scripts/draw_entrance_calibration.py src/assets/calibrations/南进口 \
      -o outputs/南进口_calibration.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# OpenCV BGR
BLUE = (255, 145, 25)
YELLOW = (0, 220, 255)
RED = (20, 20, 255)
GREEN = (70, 220, 60)
WHITE = (255, 255, 255)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _pts(seq) -> list[tuple[int, int]]:
    return [(int(round(x)), int(round(y))) for x, y in seq]


def draw_polyline(image, points, color, thickness):
    cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, color,
                  thickness, cv2.LINE_AA)


def draw_dashed_segment(image, start, end, color, thickness, dash, gap):
    p1 = np.asarray(start, dtype=float)
    p2 = np.asarray(end, dtype=float)
    vec = p2 - p1
    length = float(np.linalg.norm(vec))
    if length == 0:
        return
    unit = vec / length
    cursor = 0.0
    while cursor < length:
        q1 = p1 + unit * cursor
        q2 = p1 + unit * min(cursor + dash, length)
        cv2.line(image, tuple(np.rint(q1).astype(int)),
                 tuple(np.rint(q2).astype(int)), color, thickness, cv2.LINE_AA)
        cursor += dash + gap


def draw_dashed_polyline(image, points, color, thickness, dash, gap):
    for start, end in zip(points, points[1:]):
        draw_dashed_segment(image, start, end, color, thickness, dash, gap)


def interpolate_polyline(left, right, ratio=0.5):
    """在两条边界折线间按比例取中线（按较短的点数对齐）。"""
    n = min(len(left), len(right))
    return [
        (round(lx + (rx - lx) * ratio), round(ly + (ry - ly) * ratio))
        for (lx, ly), (rx, ry) in zip(left[:n], right[:n])
    ]


def fill_lane(image, left, right, color, alpha=0.32):
    """把相邻两条边界折线围成的车道面填充为半透明色。"""
    n = min(len(left), len(right))
    if n < 2:
        return
    poly = np.asarray(left[:n] + right[:n][::-1], dtype=np.int32)
    overlay = image.copy()
    cv2.fillPoly(overlay, [poly], color, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def draw_arrow(image, start, end, scale):
    cv2.arrowedLine(image, tuple(map(int, start)), tuple(map(int, end)),
                    WHITE, max(2, round(5 * scale)), cv2.LINE_AA, tipLength=0.35)


def draw_diamond(image, point, scale):
    x, y = int(round(point[0])), int(round(point[1]))
    r = max(5, round(11 * scale))
    vertices = np.asarray([(x, y - r), (x + r, y), (x, y + r), (x - r, y)],
                          dtype=np.int32)
    cv2.fillConvexPoly(image, vertices, RED, cv2.LINE_AA)
    cv2.polylines(image, [vertices], True, WHITE, max(1, round(1.5 * scale)),
                  cv2.LINE_AA)


def load_font(size):
    for path in ("/System/Library/Fonts/STHeiti Light.ttc",
                 "/System/Library/Fonts/PingFang.ttc",
                 "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def add_legend(image, scale, title):
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    canvas = Image.fromarray(rgba)
    draw = ImageDraw.Draw(canvas, "RGBA")

    x0, y0 = round(40 * scale), round(36 * scale)
    box_w, box_h = round(430 * scale), round(360 * scale)
    draw.rounded_rectangle((x0, y0, x0 + box_w, y0 + box_h),
                           radius=round(16 * scale), fill=(0, 0, 0, 190),
                           outline=(255, 255, 255, 160), width=max(1, round(2 * scale)))

    font = load_font(round(30 * scale))
    legend = [
        ("车道边界", "solid", (25, 145, 255, 255)),
        ("车道中心线", "dashed", (255, 220, 0, 255)),
        ("停止线", "solid", (255, 20, 20, 255)),
        ("断面线", "dashed", (60, 220, 70, 255)),
        ("导向箭头", "arrow", (255, 255, 255, 255)),
        ("关键点", "diamond", (255, 20, 20, 255)),
    ]
    sample_x1 = x0 + round(36 * scale)
    sample_x2 = x0 + round(150 * scale)
    text_x = x0 + round(180 * scale)
    row_y = y0 + round(48 * scale)
    row_gap = round(50 * scale)
    line_w = max(2, round(4 * scale))

    for i, (label, kind, color) in enumerate(legend):
        y = row_y + i * row_gap
        if kind == "solid":
            draw.line((sample_x1, y, sample_x2, y), fill=color, width=line_w)
        elif kind == "dashed":
            for dx in range(sample_x1, sample_x2, round(22 * scale)):
                draw.line((dx, y, min(dx + round(13 * scale), sample_x2), y),
                          fill=color, width=line_w)
        elif kind == "arrow":
            draw.line((sample_x1, y, sample_x2 - round(12 * scale), y),
                      fill=color, width=line_w)
            draw.polygon([(sample_x2, y),
                          (sample_x2 - round(20 * scale), y - round(11 * scale)),
                          (sample_x2 - round(20 * scale), y + round(11 * scale))],
                         fill=color)
        else:
            r = round(11 * scale)
            cx = (sample_x1 + sample_x2) // 2
            draw.polygon([(cx, y - r), (cx + r, y), (cx, y + r), (cx - r, y)],
                         fill=color)
        draw.text((text_x, y - round(20 * scale)), label, font=font,
                  fill=(255, 255, 255, 255))

    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGBA2BGR)


def draw_calibration(calib_dir: Path, bg_path: Path | None = None) -> np.ndarray:
    src = bg_path if bg_path is not None else calib_dir / "ref.jpg"
    ref = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if ref is None:
        raise FileNotFoundError(f"无法读取参考帧：{src}")
    result = ref.copy()
    scale = result.shape[1] / 2000.0  # 以 2000px 宽为基准

    lanes_data = _load_json(calib_dir / "lanes.json")["lanes"]
    # 按车道 id 升序，得到有序边界折线
    boundaries = [_pts(lanes_data[k]) for k in sorted(lanes_data, key=int)]

    # 先铺半透明车道面（蓝 / 青交替，区分相邻车道），再压边界与中心线
    LANE_FILLS = [(255, 170, 40), (230, 200, 60)]  # BGR: 蓝、青
    for i, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        fill_lane(result, left, right, LANE_FILLS[i % len(LANE_FILLS)])

    for line in boundaries:
        draw_polyline(result, line, BLUE, max(2, round(4 * scale)))

    # 相邻边界之间的中心线（黄色虚线）
    centers = []
    for left, right in zip(boundaries, boundaries[1:]):
        center = interpolate_polyline(left, right, 0.5)
        centers.append(center)
        draw_dashed_polyline(result, center, YELLOW, max(2, round(3 * scale)),
                             dash=round(26 * scale), gap=round(16 * scale))

    # 停止线（红色实线）
    stop_ends = []
    for sl in _load_json(calib_dir / "stop_lines.json")["stop_lines"]:
        p1, p2 = _pts(sl["points"])[:2]
        cv2.line(result, p1, p2, RED, max(3, round(6 * scale)), cv2.LINE_AA)
        stop_ends.extend([p1, p2])

    # 断面线（绿色虚线）；第一条视为主断面，记录端点做关键点
    section_endpoints = []
    for idx, sec in enumerate(_load_json(calib_dir / "sections.json")["sections"]):
        p1, p2 = _pts(sec["points"])[:2]
        draw_dashed_segment(result, p1, p2, GREEN, max(2, round(4 * scale)),
                            dash=round(22 * scale), gap=round(13 * scale))
        if idx == 0:
            section_endpoints.extend([p1, p2])

    # 导向箭头：沿每条车道中心线，从远端指向停止线端
    for center in centers:
        if len(center) < 2:
            continue
        # center[0] 为近端（靠停止线，y 较大），末点为远端
        near, far = center[0], center[-1]
        # 取中段作为箭头位置，指向近端方向
        mid_i = len(center) // 2
        a_start = center[min(mid_i + 1, len(center) - 1)]
        a_end = center[max(mid_i - 1, 0)]
        # 确保指向近端（y 更大的一侧）
        if a_end[1] < a_start[1]:
            a_start, a_end = a_end, a_start
        draw_arrow(result, a_start, a_end, scale)

    # 关键点：各边界远端 + 停止线两端 + 主断面两端
    for line in boundaries:
        draw_diamond(result, line[-1], scale)
    for p in stop_ends + section_endpoints:
        draw_diamond(result, p, scale)

    return add_legend(result, scale, title=calib_dir.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calib_dir", type=Path, help="进口标定目录（含 ref.jpg 与各 json）")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="输出 PNG 路径，默认 outputs/{进口}_calibration.png")
    parser.add_argument("--bg", type=Path, default=None,
                        help="自定义背景图（如彩色帧），默认用标定目录的 ref.jpg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = draw_calibration(args.calib_dir, bg_path=args.bg)
    output = args.output or Path("outputs") / f"{args.calib_dir.name}_calibration.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), result):
        raise OSError(f"无法写入输出图：{output}")
    print(output.resolve())


if __name__ == "__main__":
    main()
