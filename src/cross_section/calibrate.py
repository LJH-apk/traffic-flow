"""
交互式斑马线标定（备用方案）。

当自动检测失败时，运行此脚本手动点击4角点，
输出可直接粘贴入 settings.py 的 HOMOGRAPHY_MATRIX 定义。

运行方式::

    python3 src/cross_section/calibrate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import cv2
import numpy as np

from src.config.settings import VIDEO_PATH

_INSTRUCTION = """\
[calibrate] 请依次点击斑马线4角点：
  1. 最顶条纹 左上角
  2. 最顶条纹 右上角
  3. 最底条纹 右下角
  4. 最底条纹 左下角
按 'r' 重置，点完4点后按任意键继续输入物理尺寸，'q' 退出。"""

_clicks: list[tuple[int, int]] = []


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(_clicks) < 4:
        _clicks.append((x, y))
        print(f"  点 {len(_clicks)}: ({x}, {y})")


def run(video_path=VIDEO_PATH, frame_idx: int = 0):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[calibrate] 读取帧失败，请检查视频路径。")
        return

    win = "Calibrate – click 4 corners (r=reset, q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    cv2.setMouseCallback(win, _on_mouse)

    print(_INSTRUCTION)

    while True:
        vis = frame.copy()
        for i, (x, y) in enumerate(_clicks):
            cv2.circle(vis, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(vis, str(i + 1), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        if len(_clicks) == 4:
            pts = np.array(_clicks, dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            print("[calibrate] 已退出，未保存。")
            break
        elif key == ord('r'):
            _clicks.clear()
            print("[calibrate] 重置，请重新点击。")
        elif len(_clicks) == 4:
            cv2.destroyAllWindows()
            # 询问物理尺寸
            try:
                width_m = float(input("\n请输入斑马线物理宽度（米，例如10.0）: "))
            except ValueError:
                width_m = 6.0
                print(f"[calibrate] 输入无效，使用默认值 {width_m}m")
            try:
                n_stripes = int(input("请输入斑马线条纹数（整数，例如5）: "))
            except ValueError:
                n_stripes = 5
                print(f"[calibrate] 输入无效，使用默认值 {n_stripes} 条")

            world_height = (2 * n_stripes - 1) * 0.40
            src_pts = np.float32(_clicks)
            dst_pts = np.float32([
                [0,       0           ],
                [width_m, 0           ],
                [width_m, world_height],
                [0,       world_height],
            ])
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None:
                print("[calibrate] 单应矩阵计算失败，请重试。")
                _clicks.clear()
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win, 1280, 720)
                cv2.setMouseCallback(win, _on_mouse)
                continue

            arr_str = np.array2string(H, separator=", ", precision=8)
            print("\n===== 复制以下内容粘贴到 src/config/settings.py =====")
            print(f"HOMOGRAPHY_MATRIX = np.array({arr_str})")
            print("=" * 60)
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
