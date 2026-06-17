"""
车道线交互式标注工具（支持缩放和平移）。

替代项目早期的 annotate_lane.py，支持：
- 鼠标滚轮缩放（聚焦光标）
- 中键拖拽平移
- 最多 6 条车道线并行标注
- 实时坐标显示

接口：annotate(image) -> 标注 dict
独立运行：python3 -m src.cross_section.lane_annotator --image <path>

注意：OpenCV putText 不支持中文，所有 UI 标签使用英文显示。
"""
import argparse
import json
from pathlib import Path
from typing import Optional
import cv2
import numpy as np


LANE_COLORS = {
    1: (0, 255,   0),   # green
    2: (0, 200, 255),   # yellow
    3: (255, 80, 200),  # purple-pink
    4: (255, 200,  0),  # cyan-blue
    5: (80, 200, 255),  # orange
    6: (180, 120, 255), # violet
}

LANE_NAMES = {i: f"Lane{i}" for i in range(1, 7)}

ZOOM_MIN = 0.05
ZOOM_MAX = 8.0
ZOOM_STEP = 1.2


class ZoomPanAnnotator:
    """
    交互式车道线标注器。

    视口状态：
        zoom    - 当前缩放比例（原图像素 → 屏幕像素的比例）
        view_x  - 视口左上角在原图中的 x 坐标
        view_y  - 视口左上角在原图中的 y 坐标
        win_w   - 显示窗口宽度（屏幕像素）
        win_h   - 显示窗口高度（屏幕像素）
    """

    def __init__(self, image: np.ndarray, n_lanes: int = 4,
                 win_w: int = 1280, win_h: int = 800):
        self.orig = image.copy()
        self.h, self.w = image.shape[:2]
        self.n_lanes = min(n_lanes, 6)

        self.win_w = win_w
        self.win_h = win_h

        # 初始缩放：适应窗口（letter-box 方式）
        self.zoom = min(win_w / self.w, win_h / self.h)
        # 居中
        self.view_x = (self.w - win_w / self.zoom) / 2
        self.view_y = (self.h - win_h / self.zoom) / 2

        # 标注数据 {lane_id: [(x,y), ...]}
        self.lanes: dict[int, list[tuple[int, int]]] = {i: [] for i in range(1, self.n_lanes + 1)}
        self.current_lane = 1

        # 鼠标状态
        self._mouse_sx = 0
        self._mouse_sy = 0
        # 中键拖拽
        self._pan_active = False
        self._pan_start_sx = 0
        self._pan_start_sy = 0
        self._pan_start_vx = 0.0
        self._pan_start_vy = 0.0
        # 空格+左键拖拽
        self._space_held = False
        self._space_pan_active = False
        self._space_pan_start_sx = 0
        self._space_pan_start_sy = 0
        self._space_pan_start_vx = 0.0
        self._space_pan_start_vy = 0.0

        self._result: Optional[dict] = None   # None = cancel, dict = save
        self._done = False

        self.win_name = "Lane Annotator"

    # ------------------------------------------------------------------
    # 坐标映射
    # ------------------------------------------------------------------

    def screen_to_orig(self, sx: float, sy: float) -> tuple[float, float]:
        """屏幕坐标 → 原图浮点坐标"""
        return self.view_x + sx / self.zoom, self.view_y + sy / self.zoom

    def orig_to_screen(self, ox: float, oy: float) -> tuple[int, int]:
        """原图坐标 → 屏幕整数坐标"""
        return int((ox - self.view_x) * self.zoom), int((oy - self.view_y) * self.zoom)

    # ------------------------------------------------------------------
    # 视口管理
    # ------------------------------------------------------------------

    def _clip_viewport(self):
        """将 view_x/y clip 到合法范围，使视口不完全跑出图像。"""
        view_w = self.win_w / self.zoom
        view_h = self.win_h / self.zoom
        # 允许图像覆盖不满窗口（小图时居中），但不让图像完全消失
        self.view_x = max(-(view_w * 0.9), min(self.w - view_w * 0.1, self.view_x))
        self.view_y = max(-(view_h * 0.9), min(self.h - view_h * 0.1, self.view_y))

    def _on_wheel(self, sx: int, sy: int, direction: int):
        """滚轮缩放，保持光标下原图点不动。"""
        ox, oy = self.screen_to_orig(sx, sy)
        self.zoom *= ZOOM_STEP if direction > 0 else 1.0 / ZOOM_STEP
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom))
        self.view_x = ox - sx / self.zoom
        self.view_y = oy - sy / self.zoom
        self._clip_viewport()

    def _reset_viewport(self):
        """重置到适应窗口的初始视口。"""
        self.zoom = min(self.win_w / self.w, self.win_h / self.h)
        self.view_x = (self.w - self.win_w / self.zoom) / 2
        self.view_y = (self.h - self.win_h / self.zoom) / 2

    # ------------------------------------------------------------------
    # 鼠标回调
    # ------------------------------------------------------------------

    def _mouse_cb(self, event, x, y, flags, param):
        self._mouse_sx = x
        self._mouse_sy = y

        # 中键按下 → 开始平移
        if event == cv2.EVENT_MBUTTONDOWN:
            self._pan_active = True
            self._pan_start_sx = x
            self._pan_start_sy = y
            self._pan_start_vx = self.view_x
            self._pan_start_vy = self.view_y

        # 中键松开
        elif event == cv2.EVENT_MBUTTONUP:
            self._pan_active = False

        # 移动
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._pan_active:
                dx = (x - self._pan_start_sx) / self.zoom
                dy = (y - self._pan_start_sy) / self.zoom
                self.view_x = self._pan_start_vx - dx
                self.view_y = self._pan_start_vy - dy
                self._clip_viewport()
            if self._space_pan_active:
                dx = (x - self._space_pan_start_sx) / self.zoom
                dy = (y - self._space_pan_start_sy) / self.zoom
                self.view_x = self._space_pan_start_vx - dx
                self.view_y = self._space_pan_start_vy - dy
                self._clip_viewport()

        # 左键点击 → 添加点（空格拖拽时不添加）
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self._space_held:
                self._space_pan_active = True
                self._space_pan_start_sx = x
                self._space_pan_start_sy = y
                self._space_pan_start_vx = self.view_x
                self._space_pan_start_vy = self.view_y
            else:
                ox, oy = self.screen_to_orig(x, y)
                ox = int(round(max(0, min(self.w - 1, ox))))
                oy = int(round(max(0, min(self.h - 1, oy))))
                self.lanes[self.current_lane].append((ox, oy))

        elif event == cv2.EVENT_LBUTTONUP:
            self._space_pan_active = False

        # 滚轮
        elif event == cv2.EVENT_MOUSEWHEEL:
            direction = 1 if flags > 0 else -1
            self._on_wheel(x, y, direction)

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------

    def _render(self) -> np.ndarray:
        """渲染当前帧到 win_w×win_h 的 canvas。"""
        view_w = self.win_w / self.zoom
        view_h = self.win_h / self.zoom

        # 计算裁剪区域（可能超出图像边界）
        x0 = self.view_x
        y0 = self.view_y
        x1 = x0 + view_w
        y1 = y0 + view_h

        # 原图裁剪的有效区域
        ix0 = int(max(0, x0))
        iy0 = int(max(0, y0))
        ix1 = int(min(self.w, x1))
        iy1 = int(min(self.h, y1))

        # 创建黑色 canvas
        canvas = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)

        if ix0 < ix1 and iy0 < iy1:
            crop = self.orig[iy0:iy1, ix0:ix1]
            # 目标屏幕区域
            sx0 = int((ix0 - x0) * self.zoom)
            sy0 = int((iy0 - y0) * self.zoom)
            sx1 = int((ix1 - x0) * self.zoom)
            sy1 = int((iy1 - y0) * self.zoom)
            sx0 = max(0, sx0); sy0 = max(0, sy0)
            sx1 = min(self.win_w, sx1); sy1 = min(self.win_h, sy1)
            if sx1 > sx0 and sy1 > sy0:
                resized = cv2.resize(crop, (sx1 - sx0, sy1 - sy0), interpolation=cv2.INTER_LINEAR)
                canvas[sy0:sy1, sx0:sx1] = resized

        # 绘制标注点和连线
        pt_radius = max(3, int(6 * self.zoom ** 0.5))
        pt_radius = min(pt_radius, 20)
        font_scale = max(0.35, min(0.7, 0.4 * self.zoom ** 0.3))
        thickness = max(1, int(2 * self.zoom ** 0.3))

        for lane_id in range(1, self.n_lanes + 1):
            pts = self.lanes[lane_id]
            if not pts:
                continue
            color = LANE_COLORS[lane_id]
            # 高亮当前线（稍亮）
            draw_color = color if lane_id != self.current_lane else tuple(
                min(255, int(c * 1.15)) for c in color)

            screen_pts = []
            for i, (ox, oy) in enumerate(pts):
                sx, sy = self.orig_to_screen(ox, oy)
                screen_pts.append((sx, sy))
                # 只绘制在窗口范围内或附近的点
                if -pt_radius * 2 <= sx <= self.win_w + pt_radius * 2 and \
                   -pt_radius * 2 <= sy <= self.win_h + pt_radius * 2:
                    cv2.circle(canvas, (sx, sy), pt_radius, draw_color, -1)
                    cv2.circle(canvas, (sx, sy), pt_radius, (255, 255, 255), 1)
                    # 编号
                    num_str = str(i + 1)
                    (tw, th), _ = cv2.getTextSize(num_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                    cv2.putText(canvas, num_str,
                                (sx - tw // 2, sy + th // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                (0, 0, 0), max(1, thickness), cv2.LINE_AA)

            # 连线（只绘制相邻点之间）
            if len(screen_pts) >= 2:
                for i in range(len(screen_pts) - 1):
                    p1, p2 = screen_pts[i], screen_pts[i + 1]
                    cv2.line(canvas, p1, p2, draw_color, thickness, cv2.LINE_AA)

        # HUD 信息
        self._draw_hud(canvas)
        return canvas

    def _draw_hud(self, canvas: np.ndarray):
        """在左上角绘制 HUD 信息。"""
        ox, oy = self.screen_to_orig(self._mouse_sx, self._mouse_sy)
        ox = int(round(ox)); oy = int(round(oy))

        lines = []
        # 当前车道和各线点数
        lane_counts = " | ".join(
            f"L{i}:{len(self.lanes[i])}" for i in range(1, self.n_lanes + 1))
        lines.append(f"Current: {LANE_NAMES[self.current_lane]}  [{lane_counts}]")
        lines.append(f"Zoom: {self.zoom * 100:.0f}%   Cursor: ({ox}, {oy})")
        lines.append("[1-6:lane | +/-:zoom | MBtn or Space+LBtn:pan]")
        lines.append("[Z:undo | C:clear | R:reset | S:save | Q:quit]")

        font = cv2.FONT_HERSHEY_SIMPLEX
        fscale = 0.52
        fthick = 1
        pad = 6
        line_h = 20

        # 半透明背景
        bg_w = 500
        bg_h = len(lines) * line_h + pad * 2
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (bg_w, bg_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        for i, line in enumerate(lines):
            y = pad + (i + 1) * line_h - 4
            # 当前车道行高亮
            color = (220, 220, 220)
            if i == 0:
                color = LANE_COLORS[self.current_lane]
            cv2.putText(canvas, line, (pad, y), font, fscale, color, fthick, cv2.LINE_AA)

        # 右侧显示车道色块
        for lane_id in range(1, self.n_lanes + 1):
            color = LANE_COLORS[lane_id]
            sx = bg_w + 10 + (lane_id - 1) * 40
            sy = 8
            cv2.rectangle(canvas, (sx, sy), (sx + 30, sy + 18),
                          color if lane_id != self.current_lane else (255, 255, 255), -1)
            cv2.rectangle(canvas, (sx, sy), (sx + 30, sy + 18), (255, 255, 255), 1)
            label = str(lane_id)
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(canvas, label, (sx + 15 - tw // 2, sy + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self, initial: Optional[dict] = None) -> Optional[dict]:
        """
        启动标注窗口。

        Args:
            initial: 可选初始标注，格式 {lane_id: [(x,y), ...]}

        Returns:
            保存时返回 {1: [...], 2: [...], ...}，取消时返回 None。
        """
        if initial:
            for lane_id, pts in initial.items():
                lid = int(lane_id)
                if 1 <= lid <= self.n_lanes:
                    self.lanes[lid] = list(pts)

        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, self.win_w, self.win_h)
        cv2.setMouseCallback(self.win_name, self._mouse_cb)

        while not self._done:
            frame = self._render()
            cv2.imshow(self.win_name, frame)
            key = cv2.waitKey(16) & 0xFF  # ~60fps

            if ord('1') <= key <= ord('6'):
                lane_id = key - ord('0')
                if self.n_lanes >= lane_id:
                    self.current_lane = lane_id
            elif key == ord('z'):
                if self.lanes[self.current_lane]:
                    self.lanes[self.current_lane].pop()
            elif key == ord('c'):
                self.lanes[self.current_lane].clear()
            elif key == ord('r'):
                self._reset_viewport()
            elif key == ord('+') or key == ord('='):
                # 以光标位置为锚点放大，光标不在窗口内时退化到画面中心
                sx = self._mouse_sx if 0 <= self._mouse_sx < self.win_w else self.win_w // 2
                sy = self._mouse_sy if 0 <= self._mouse_sy < self.win_h else self.win_h // 2
                self._on_wheel(sx, sy, +1)
            elif key == ord('-') or key == ord('_'):
                sx = self._mouse_sx if 0 <= self._mouse_sx < self.win_w else self.win_w // 2
                sy = self._mouse_sy if 0 <= self._mouse_sy < self.win_h else self.win_h // 2
                self._on_wheel(sx, sy, -1)
            elif key == ord('s'):
                self._result = {k: list(v) for k, v in self.lanes.items()}
                self._done = True
            elif key == ord('q') or key == 27:  # q or ESC
                self._result = None
                self._done = True
            elif key == 32:  # Space — toggle space-held flag
                # waitKey 不能直接检测 keyUp，用状态机模拟：
                # 按下时设 True，但因为 waitKey 是轮询，只在按下帧为 True
                # 此处仅做 toggle，实际 pan 依赖鼠标左键状态
                self._space_held = not self._space_held
            elif key == 255 or key == -1:
                pass  # no key

        cv2.destroyWindow(self.win_name)
        return self._result


# --------------------------------------------------------------------------
# 公开接口
# --------------------------------------------------------------------------

def annotate(image: np.ndarray, n_lanes: int = 4,
             initial: Optional[dict] = None,
             win_w: int = 1280, win_h: int = 800
             ) -> Optional[dict[int, list[tuple[int, int]]]]:
    """
    交互式标注车道线。

    Args:
        image:     原图（任意分辨率，4K 也可）
        n_lanes:   车道线数量上限（最多 6）
        initial:   可选初始标注，用于编辑现有标定
        win_w:     窗口宽度（默认 1280）
        win_h:     窗口高度（默认 800）

    Returns:
        {1: [(x,y), ...], 2: [...], ...}  保存退出（按 s）
        None                              用户取消（按 q 或 ESC）
    """
    annotator = ZoomPanAnnotator(image, n_lanes=n_lanes, win_w=win_w, win_h=win_h)
    return annotator.run(initial)


# --------------------------------------------------------------------------
# CLI 入口
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Lane annotator with zoom/pan support"
    )
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--out", default="outputs/annotated_lanes.json",
                        help="Output JSON path (default: outputs/annotated_lanes.json)")
    parser.add_argument("--initial", default=None,
                        help="Path to existing annotation JSON to load for editing")
    parser.add_argument("--n-lanes", type=int, default=4,
                        help="Number of lanes (1-6, default 4)")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERROR] Cannot read image: {args.image}")
        return

    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    print("Controls: 1-6 switch lane | Wheel zoom | MBtn pan | Space+LBtn pan")
    print("          Z undo | C clear lane | R reset view | S save | Q quit")

    initial = None
    if args.initial:
        try:
            data = json.loads(Path(args.initial).read_text())
            lanes_raw = data.get("lanes", {})
            initial = {int(k): [tuple(p) for p in v] for k, v in lanes_raw.items()}
            print(f"Loaded initial annotation from {args.initial}")
        except Exception as e:
            print(f"[WARN] Cannot load initial annotation: {e}")

    result = annotate(img, n_lanes=args.n_lanes, initial=initial)

    if result is None:
        print("Cancelled.")
        return

    # 过滤空线
    result_out = {k: v for k, v in result.items() if v}
    total_pts = sum(len(v) for v in result_out.values())
    print(f"Saved {len(result_out)} lanes, {total_pts} points total.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"image": str(Path(args.image).resolve()),
             "lanes": {str(k): v for k, v in result_out.items()}},
            indent=2, ensure_ascii=False
        )
    )
    print(f"Saved to {args.out}")


def annotate_homography(
    image: np.ndarray,
) -> dict | None:
    """备用单应矩阵标定：用户点 4 个角点 + 输入物理长宽。

    复用 ZoomPanAnnotator 的视口，把车道线标注改成 4 点矩形标注。
    使用 n_lanes=1，让用户在 Lane 1 上点 4 个点（矩形 4 角：左上→右上→右下→左下）。

    Returns:
        {'src_pts': [(x,y)*4], 'world_w': float, 'world_h': float, 'H': 3x3 ndarray}
        或 None（用户取消或点数不足 4）
    """
    print("\n[H 标定] 请在弹出窗口中为 Lane 1 点 4 个角点（左上→右上→右下→左下）")
    print("           可以是斑马线一条条纹、停止线一段、或任何已知尺寸的矩形")
    result = annotate(image, n_lanes=1)
    if result is None or 1 not in result or len(result[1]) != 4:
        print("[H 标定] 未获得 4 个角点，取消")
        return None

    src_pts = np.float32(result[1])
    try:
        world_w = float(input("\n请输入这个矩形的宽度（米，沿水平方向）: "))
        world_h = float(input("请输入这个矩形的高度（米，沿垂直方向）: "))
    except ValueError:
        print("[H 标定] 输入无效，取消")
        return None

    dst_pts = np.float32([
        [0,       0      ],
        [world_w, 0      ],
        [world_w, world_h],
        [0,       world_h],
    ])
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("[H 标定] 单应矩阵计算失败")
        return None

    return {
        'src_pts': [tuple(p) for p in result[1]],
        'world_w': world_w,
        'world_h': world_h,
        'H': H,
    }


if __name__ == "__main__":
    main()
