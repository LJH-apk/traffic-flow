"""
手动标注多条车道线脚本。

用法：
  python3 annotate_lane.py
  python3 annotate_lane.py --image outputs/lane_bg_f2675.jpg

操作：
  1-5            — 切换当前标注的车道线编号
  左键单击       — 在当前线添加点
  z              — 撤销当前线最后一个点
  c              — 清空当前线所有点
  +/=            — 放大（以画面中心为基准）
  -              — 缩小
  中键拖拽/方向键 — 平移画面
  s              — 保存并退出
  q              — 不保存退出
"""
import argparse
import json
import cv2
import numpy as np
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    _FONT = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", 18)
    _PIL_OK = True
except Exception:
    _PIL_OK = False

INIT_SCALE = 0.4
LINE_COLORS = {
    1: (0, 255,   0),    # 绿
    2: (0, 200, 255),    # 黄
    3: (255,  80, 200),  # 紫粉
    4: (255, 200,   0),  # 青蓝
    5: (80,  200, 255),  # 橙
}
LANE_NAMES = {
    1: "线1", 2: "线2", 3: "线3", 4: "线4", 5: "线5(非机动车)",
}
POINT_R = 5


def put_cn(img_bgr: np.ndarray, text: str, pos: tuple, color: tuple,
           size: int = 18) -> np.ndarray:
    """用 PIL 在 BGR 图上绘制中文文字，返回修改后的 BGR 图。"""
    if not _PIL_OK:
        cv2.putText(img_bgr, text, (pos[0], pos[1] + size),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        return img_bgr
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", size)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="outputs/lane_bg_f2675.jpg")
    parser.add_argument("--out",   default="outputs/annotated_lanes.json")
    args = parser.parse_args()

    orig = cv2.imread(args.image)
    if orig is None:
        print(f"读取失败: {args.image}"); return
    oh, ow = orig.shape[:2]

    # ── 视口状态 ──────────────────────────────────────────────────────────────
    scale   = INIT_SCALE          # 当前缩放比例
    pan_x   = 0                   # 视口左上角在原图中的 x 偏移（像素）
    pan_y   = 0                   # 视口左上角在原图中的 y 偏移（像素）
    WIN_W   = int(ow * INIT_SCALE)
    WIN_H   = int(oh * INIT_SCALE)

    lanes: dict[int, list[tuple[int, int]]] = {i: [] for i in range(1, 6)}
    current = 1
    dragging = False
    drag_start = (0, 0)
    pan_start  = (0, 0)

    def clamp_pan():
        nonlocal pan_x, pan_y
        vis_w = WIN_W / scale
        vis_h = WIN_H / scale
        pan_x = max(0, min(pan_x, ow - vis_w))
        pan_y = max(0, min(pan_y, oh - vis_h))

    def orig_to_disp(ox: float, oy: float) -> tuple[int, int]:
        """原图坐标 → 显示窗口坐标。"""
        return (int((ox - pan_x) * scale), int((oy - pan_y) * scale))

    def disp_to_orig(dx: int, dy: int) -> tuple[int, int]:
        """显示窗口坐标 → 原图坐标。"""
        return (int(dx / scale + pan_x), int(dy / scale + pan_y))

    def draw() -> np.ndarray:
        # 裁出可见区域并缩放
        vis_w = int(WIN_W / scale)
        vis_h = int(WIN_H / scale)
        x0 = max(0, int(pan_x))
        y0 = max(0, int(pan_y))
        x1 = min(ow, x0 + vis_w)
        y1 = min(oh, y0 + vis_h)
        crop = orig[y0:y1, x0:x1]
        vis  = cv2.resize(crop, (WIN_W, WIN_H), interpolation=cv2.INTER_LINEAR)

        # 绘制各条线
        for lid, pts in lanes.items():
            color = LINE_COLORS[lid]
            disp  = [orig_to_disp(x, y) for x, y in pts]
            for i, (dx, dy) in enumerate(disp):
                cv2.circle(vis, (dx, dy), POINT_R, color, -1)
                if lid == current:
                    cv2.putText(vis, str(i), (dx + 6, dy - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            if len(disp) >= 2:
                for a, b in zip(disp[:-1], disp[1:]):
                    cv2.line(vis, a, b, color, 2 if lid == current else 1)

        # HUD（中文提示）
        vis = put_cn(vis, f"当前: {LANE_NAMES[current]}  "
                          f"缩放:{scale:.2f}x  "
                          f"[1-5切换  +/-缩放  右键拖移  左键添加  z撤销  c清空  s保存  q退出]",
                     (8, 4), (255, 255, 255), size=16)
        for i, lid in enumerate(range(1, 6)):
            color = LINE_COLORS[lid]
            mark  = "●" if lid == current else "○"
            vis = put_cn(vis, f"{mark} {LANE_NAMES[lid]}: {len(lanes[lid])}点",
                         (8, 28 + i * 24), color, size=16)
        return vis

    def on_mouse(event, mx, my, flags, param):
        nonlocal current, pan_x, pan_y, dragging, drag_start, pan_start

        if event == cv2.EVENT_LBUTTONDOWN:
            ox, oy = disp_to_orig(mx, my)
            lanes[current].append((ox, oy))
            cv2.imshow("annotate", draw())

        elif event == cv2.EVENT_RBUTTONDOWN:
            dragging   = True
            drag_start = (mx, my)
            pan_start  = (pan_x, pan_y)

        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            dx = (mx - drag_start[0]) / scale
            dy = (my - drag_start[1]) / scale
            pan_x = pan_start[0] - dx
            pan_y = pan_start[1] - dy
            clamp_pan()
            cv2.imshow("annotate", draw())

        elif event == cv2.EVENT_RBUTTONUP:
            dragging = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            # 滚轮缩放（以鼠标位置为中心）
            nonlocal_scale = scale
            factor = 1.15 if flags > 0 else 1 / 1.15
            new_scale = max(0.1, min(4.0, nonlocal_scale * factor))
            # 保持鼠标指向的原图点不动
            ox, oy = disp_to_orig(mx, my)
            update_scale(new_scale)
            pan_x = ox - mx / scale
            pan_y = oy - my / scale
            clamp_pan()
            cv2.imshow("annotate", draw())

    def update_scale(new_scale: float):
        nonlocal scale
        scale = new_scale

    cv2.namedWindow("annotate", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("annotate", WIN_W, WIN_H)
    cv2.setMouseCallback("annotate", on_mouse)
    cv2.imshow("annotate", draw())

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
            current = int(chr(key))
            cv2.imshow("annotate", draw())

        elif key in (ord('+'), ord('=')):        # 放大
            new_s = min(4.0, scale * 1.2)
            cx_orig = pan_x + (WIN_W / 2) / scale
            cy_orig = pan_y + (WIN_H / 2) / scale
            scale = new_s
            pan_x = cx_orig - (WIN_W / 2) / scale
            pan_y = cy_orig - (WIN_H / 2) / scale
            clamp_pan()
            cv2.imshow("annotate", draw())

        elif key == ord('-'):                    # 缩小
            new_s = max(0.1, scale / 1.2)
            cx_orig = pan_x + (WIN_W / 2) / scale
            cy_orig = pan_y + (WIN_H / 2) / scale
            scale = new_s
            pan_x = cx_orig - (WIN_W / 2) / scale
            pan_y = cy_orig - (WIN_H / 2) / scale
            clamp_pan()
            cv2.imshow("annotate", draw())

        elif key == 81 or key == 2:              # ← 左箭头
            pan_x -= 50 / scale; clamp_pan()
            cv2.imshow("annotate", draw())
        elif key == 83 or key == 3:              # → 右箭头
            pan_x += 50 / scale; clamp_pan()
            cv2.imshow("annotate", draw())
        elif key == 82 or key == 0:              # ↑ 上箭头
            pan_y -= 50 / scale; clamp_pan()
            cv2.imshow("annotate", draw())
        elif key == 84 or key == 1:              # ↓ 下箭头
            pan_y += 50 / scale; clamp_pan()
            cv2.imshow("annotate", draw())

        elif key == ord('z') and lanes[current]:
            lanes[current].pop()
            cv2.imshow("annotate", draw())

        elif key == ord('c'):
            lanes[current] = []
            cv2.imshow("annotate", draw())

        elif key == ord('s'):
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "image": args.image,
                "scale": INIT_SCALE,
                "lanes": {str(k): v for k, v in lanes.items() if v},
            }
            out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            total = sum(len(v) for v in lanes.values())
            print(f"已保存 {len(data['lanes'])} 条线共 {total} 个点 → {out}")
            break

        elif key == ord('q'):
            print("退出，未保存")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
