"""断面线手动标注工具。

用法：
  python3 src/cross_section/annotate_sections.py --entrance 北进口
  python3 src/cross_section/annotate_sections.py --entrance 东进口 --names 东进口主断面,东进口右转

每条断面线点 2 个端点，按 S 保存，按 Q 或 ESC 取消。
输出: calibrations/<进口>/sections.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.settings import CALIBRATIONS_DIR
from src.cross_section.lane_annotator import annotate
from src.cross_section.section_calibration import sections_path


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="标注一条或多条断面线，每条线需要 2 个端点。")
    parser.add_argument("--entrance", required=True, help="进口名，例如：北进口、南进口、东进口")
    parser.add_argument("--image", type=Path, help="参考图路径，默认 calibrations/<进口>/ref.jpg")
    parser.add_argument("--out", type=Path, help="输出 JSON，默认 calibrations/<进口>/sections.json")
    parser.add_argument("--names", help="逗号分隔的断面名；默认 <进口>主断面")
    parser.add_argument("--dir-pos", default="到达", help="叉积正方向标签，默认 到达")
    parser.add_argument("--dir-neg", default="离去", help="叉积负方向标签，默认 离去")
    args = parser.parse_args()

    image_path = args.image or CALIBRATIONS_DIR / args.entrance / "ref.jpg"
    out_path = args.out or sections_path(args.entrance)
    names = _split_csv(args.names) if args.names else [f"{args.entrance}主断面"]
    if len(names) > 6:
        raise SystemExit("一次最多标注 6 条断面线。")

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"读取失败: {image_path}")

    print("操作：每条断面线点 2 个端点；1-6 切换断面；滚轮缩放；中键拖拽；S 保存。")
    for idx, name in enumerate(names, start=1):
        print(f"  {idx}: {name}")

    result = annotate(image, n_lanes=len(names))
    if result is None:
        raise SystemExit("已取消，未保存。")

    sections = []
    for idx, name in enumerate(names, start=1):
        points = result.get(idx, [])
        if len(points) != 2:
            raise SystemExit(f"{name} 需要正好 2 个端点，当前 {len(points)} 个点。请重新标注。")
        sections.append(
            {
                "name": name,
                "points": points,
                "dir_pos": args.dir_pos,
                "dir_neg": args.dir_neg,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"image": str(image_path), "sections": sections}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"断面线已保存: {out_path}")


if __name__ == "__main__":
    main()
