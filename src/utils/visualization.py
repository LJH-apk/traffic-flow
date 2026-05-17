"""
绘图工具函数。供 detector 和 tracker 共用，避免重复实现绘框逻辑。
"""
import cv2
import numpy as np
from src.config.settings import CLASS_COLORS

try:
    from PIL import Image, ImageDraw, ImageFont as _ImageFont
    _pil_font = _ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", 18)
    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False


def _has_cjk(text: str) -> bool:
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)


def put_text(frame, text, pos, color, font_scale=0.55, thickness=2):
    """含中文走 Pillow，否则走 cv2.putText。pos 为左上角坐标。"""
    if _PIL_AVAILABLE and _has_cjk(text):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(img).text(pos, text,
                                 font=_pil_font,
                                 fill=(color[2], color[1], color[0]))
        frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(frame, text, (pos[0], pos[1] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def draw_boxes(
    frame: np.ndarray,
    boxes_xyxy: list[tuple[int, int, int, int]],
    labels: list[str],
    confs: list[float],
    track_ids: list[int | None] | None = None,
    plates: list[str] | None = None,
) -> np.ndarray:
    """在帧上绘制检测框和标签。

    Args:
        frame:      BGR 图像（原地修改）。
        boxes_xyxy: 检测框列表，格式 [(x1,y1,x2,y2), ...]。
        labels:     每个框对应的类别名称。
        confs:      每个框对应的置信度。
        track_ids:  可选，每个框对应的跟踪 ID；None 表示不显示。
        plates:     可选，每个框对应的车牌号；None 表示不显示。

    Returns:
        绘制了检测框的帧（与输入同一对象）。
    """
    for i, (box, label, conf) in enumerate(zip(boxes_xyxy, labels, confs)):
        x1, y1, x2, y2 = box
        color = CLASS_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        tid   = track_ids[i] if track_ids is not None else None
        plate = plates[i]    if plates    is not None else ""
        base  = f"#{tid} {label} {conf:.2f}" if tid is not None else f"{label} {conf:.2f}"
        text  = f"{base} | {plate}" if plate else base
        put_text(frame, text, (x1, y1 - 22), color)
    return frame


def put_fps_text(frame: np.ndarray, fps: float, n_det: int) -> np.ndarray:
    """在帧左上角叠加 FPS 和检测数量文字。

    Args:
        frame: BGR 图像（原地修改）。
        fps:   当前推理帧率。
        n_det: 当前帧检测到的目标数量。

    Returns:
        叠加了文字的帧。
    """
    cv2.putText(frame, f"FPS: {fps:.1f}  Det: {n_det}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return frame


def put_track_text(frame: np.ndarray, track_id: int, plate: str,
                   pos: tuple[int, int]) -> np.ndarray:
    """在跟踪框上方叠加车牌号文字（仅当车牌有效时）。

    Args:
        frame:    BGR 图像（原地修改）。
        track_id: 跟踪 ID。
        plate:    识别到的车牌字符串，空字符串表示未识别。
        pos:      文字绘制位置 (x, y)。

    Returns:
        叠加了文字的帧。
    """
    if plate:
        text = f"#{track_id} {plate}"
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return frame
