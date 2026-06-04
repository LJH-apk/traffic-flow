"""
停车线手动标注工具。

用法：
  python3 src/cross_section/annotate_stop_line.py --entrance 东进口 --image calibrations/东进口/ref.jpg

操作与车道线标注器一致，只需要在 Lane1 上点停车线两个端点，然后按 S 保存。
输出: calibrations/<进口>/stop_lines.json
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


def main() -> None:
    parser = argparse.ArgumentParser(description="标注停车线两个端点。")
    parser.add_argument("--entrance", required=True, help="进口名，例如：东进口")
    parser.add_argument("--image", type=Path, help="参考图路径，默认 calibrations/<进口>/ref.jpg")
    parser.add_argument("--out", type=Path, help="输出 JSON，默认 calibrations/<进口>/stop_lines.json")
    parser.add_argument("--name", default="停车线", help="停车线名称")
    args = parser.parse_args()

    image_path = args.image or CALIBRATIONS_DIR / args.entrance / "ref.jpg"
    out_path = args.out or CALIBRATIONS_DIR / args.entrance / "stop_lines.json"

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"读取失败: {image_path}")

    result = annotate(image, n_lanes=1)
    if result is None:
        raise SystemExit("已取消，未保存。")

    points = result.get(1, [])
    if len(points) != 2:
        raise SystemExit(f"停车线需要正好 2 个端点，当前 {len(points)} 个点。请重新标注。")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "image": str(image_path),
                "stop_lines": [
                    {
                        "name": args.name,
                        "points": points,
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"停车线已保存: {out_path}")


if __name__ == "__main__":
    main()
