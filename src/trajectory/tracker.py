"""
轨迹数据提取模块：ByteTrack 跟踪 + 车牌识别 + CSV 输出。

输出字段（trajectory.csv）::

    frame_id, timestamp_s, track_id, class_name, cx, cy, x1, y1, x2, y2, plate

运行示例::

    python3 -u src/trajectory/tracker.py
"""
# 确保从任意目录运行时均可找到项目根（src 包）
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parents[2]))

import csv
import re
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from src.config.settings import (
    VEHICLE_CLASSES,
    DEVICE,
    CONF_THRESH,
    VIDEO_PATH,
    OUTPUT_DIR,
    MODEL_DIR,
    MODEL_NAME,
    TRAJ_SAMPLE_FPS,
    TRAJ_CSV_PATH,
)
from src.utils.video_io import open_video, video_meta, make_writer, iter_frames
from src.utils.visualization import draw_boxes, put_fps_text, put_text

# ── 运行时配置 ────────────────────────────────────────────────────────────────
_MODEL       = MODEL_DIR / MODEL_NAME          # 推理模型路径
_START_FRAME = 2250                               # 起始帧号（含），0 = 视频开头
_END_FRAME   = 3250                            # 终止帧号（不含），None = 视频结尾
_OUTPUT_VID  = OUTPUT_DIR / "trajectory.mp4"  # 带轨迹标注的输出视频

# 车牌正则：省份简称 + 字母 + 5位字母/数字（标准蓝牌 / 新能源绿牌格式）
# 第1位：中文省份简称
# 第2位：A-Z（发牌城市代码）
# 第3-7位：A-Z 或 0-9（流水号，新能源末位可为字母，共5位）
_PROVINCE = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁"
# 完整车牌：省份简称 + 城市码(A-Z) + 5位流水号
_PLATE_FULL = re.compile(rf"[{_PROVINCE}][A-Z][A-Z0-9]{{5}}")
# 降级正则：省份字符被 OCR 误读时，从末尾取城市码 + 5位流水号（共6字符）
# 不加边界限制，直接取字符串最后6位，避免 'MA8R5Z9' 中 'A' 被前置 'M' 阻断
_PLATE_BODY = re.compile(r"[A-Z][A-Z0-9]{5}$")
_PLATE_CONF_THRESH = 0.6  # HyperLPR3 置信度阈值，低于此值的结果丢弃


def _rel_to_abs(
    rel_box: tuple[float, float, float, float],
    x1: int, y1: int, x2: int, y2: int,
) -> tuple[int, int, int, int]:
    """将相对坐标（在车辆 bbox 内的比例）转换为当前帧的绝对像素坐标。"""
    rx1, ry1, rx2, ry2 = rel_box
    vw, vh = x2 - x1, y2 - y1
    return (x1 + int(rx1 * vw), y1 + int(ry1 * vh),
            x1 + int(rx2 * vw), y1 + int(ry2 * vh))


# ─────────────────────────────────────────────────────────────────────────────
class PlateRecognizer:
    """车牌识别器：使用 HyperLPR3 检测并识别车牌，首次成功后缓存结果。

    缓存值为 (plate_str, box_xyxy)，其中 box_xyxy 是车牌在完整帧中的坐标，
    供上层绘制真实车牌框使用。

    Attributes:
        _cache:   track_id -> (plate_str, rel_box) 的识别缓存，每辆车仅推理一次。
                  rel_box 为车牌在车辆 bbox 内的相对坐标 (rx1,ry1,rx2,ry2)，
                  每帧用当前 bbox 还原绝对坐标，从而跟随车辆移动。
        _enabled: False 表示 hyperlpr3 不可用，recognize() 直接返回空结果。
    """

    def __init__(self) -> None:
        self._cache:   dict[int, tuple[str, tuple[float, float, float, float] | None]] = {}
        self._catcher = None
        self._enabled = False

        try:
            import hyperlpr3 as lpr3  # type: ignore
            self._catcher = lpr3.LicensePlateCatcher()
            self._enabled = True
            print("[PlateRecognizer] HyperLPR3 加载成功")
        except ImportError:
            print("[PlateRecognizer] 未找到 hyperlpr3，车牌识别已禁用")

    def recognize(
        self,
        frame: np.ndarray,
        track_id: int,
        box_xyxy: tuple[int, int, int, int],
    ) -> tuple[str, tuple[float, float, float, float] | None]:
        """识别车牌，已成功识别的 track_id 直接返回缓存。

        Args:
            frame:    完整 BGR 帧。
            track_id: 跟踪 ID。
            box_xyxy: 车辆检测框 (x1, y1, x2, y2)，传入整个车辆区域供检测。

        Returns:
            (plate_str, rel_box)；rel_box 为车牌在车辆 bbox 内的相对坐标
            (rx1, ry1, rx2, ry2)，未识别时返回 ("", None)。
        """
        if track_id in self._cache:
            return self._cache[track_id]
        if not self._enabled:
            return "", None

        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = box_xyxy
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(fw, x2), min(fh, y2)
        crop_w, crop_h = cx2 - cx1, cy2 - cy1
        if crop_w <= 0 or crop_h <= 0:
            return "", None
        crop = frame[cy1:cy2, cx1:cx2]

        try:
            results = self._catcher(crop)
        except Exception:  # noqa: BLE001
            return "", None

        best_plate: str = ""
        best_box:   tuple[float, float, float, float] | None = None
        best_conf:  float = _PLATE_CONF_THRESH

        for item in (results or []):
            text, conf = item[0], float(item[1])
            if conf < best_conf:
                continue
            plate = self._match_plate(text)
            if not plate:
                continue
            # item[3] 是车牌在 crop 内的绝对像素坐标，转为相对比例存储，
            # 使得后续每帧可根据当前 bbox 还原绝对坐标，车牌框跟随车辆移动。
            px1, py1, px2, py2 = item[3]
            rel_box = (px1 / crop_w, py1 / crop_h, px2 / crop_w, py2 / crop_h)
            best_plate = plate
            best_box   = rel_box
            best_conf  = conf

        result: tuple[str, tuple[float, float, float, float] | None] = (best_plate, best_box)
        if best_plate:
            self._cache[track_id] = result
        return result

    @staticmethod
    def _match_plate(text: str) -> str:
        """从 OCR 文字中提取车牌号。

        优先匹配完整格式（省份+城市码+5位），
        降级匹配城市码+5位（省份被误读时）。

        Args:
            text: OCR 识别的原始文字。

        Returns:
            提取的车牌字符串，无匹配返回空字符串。
        """
        clean = text.replace(" ", "").upper()
        m = _PLATE_FULL.search(clean)
        if m:
            return m.group()
        m = _PLATE_BODY.search(clean)
        if m:
            return m.group()   # 缺省份字符的车牌，仍有意义
        return ""


# ─────────────────────────────────────────────────────────────────────────────
class TrajectoryTracker:
    """ByteTrack 车辆跟踪器，按固定频率采样轨迹并写出 CSV。

    每隔 ``sample_interval`` 帧记录一次各车辆的中心坐标、检测框和车牌，
    最终输出结构化 CSV 文件，供后续统计分析使用。

    Attributes:
        model_path:      YOLO 权重路径。
        device:          推理设备。
        conf:            置信度阈值。
        sample_interval: 每隔多少帧采样一次（根据视频 FPS 和 TRAJ_SAMPLE_FPS 计算）。
    """

    _CSV_FIELDS = [
        "frame_id", "timestamp_s", "track_id", "class_name",
        "cx", "cy", "x1", "y1", "x2", "y2", "plate",
    ]

    def __init__(
        self,
        model_path: str | Path = _MODEL,
        device: str = DEVICE,
        conf: float = CONF_THRESH,
    ) -> None:
        """初始化跟踪器，加载 YOLO 模型和车牌识别器。

        Args:
            model_path: YOLO 权重文件路径。
            device:     推理设备（mps / cpu / cuda）。
            conf:       置信度阈值。
        """
        self.model_path = Path(model_path)
        self.device     = device
        self.conf       = conf
        self._model     = YOLO(str(self.model_path))
        self._model.to(device)
        self._plate_rec = PlateRecognizer()
        print(f"[Tracker] 模型: {self.model_path.name}  设备: {device}")

    def run(
        self,
        video_path: str | Path = VIDEO_PATH,
        output_video: str | Path = _OUTPUT_VID,
        csv_path: str | Path = TRAJ_CSV_PATH,
        start_frame: int = _START_FRAME,
        end_frame: Optional[int] = _END_FRAME,
        sample_fps: int = TRAJ_SAMPLE_FPS,
    ) -> Path:
        """运行跟踪并输出轨迹 CSV 和标注视频。

        Args:
            video_path:   输入视频路径。
            output_video: 输出带标注视频路径。
            csv_path:     输出轨迹 CSV 路径。
            start_frame:  起始帧号（含），默认 0。
            end_frame:    终止帧号（不含），None 表示处理到视频结尾。
            sample_fps:   每秒采样次数（轨迹记录频率）。

        Returns:
            写出的 CSV 文件路径。
        """
        cap  = open_video(video_path)
        meta = video_meta(cap)

        video_fps       = meta["fps"] if meta["fps"] > 0 else 25.0
        sample_interval = max(1, round(video_fps / sample_fps))   # 采样间隔（帧数）

        # 计算实际处理范围
        total_video = meta["frame_count"]
        start_frame = max(0, min(start_frame, total_video - 1))
        end_frame   = total_video if end_frame is None else min(end_frame, total_video)
        n_frames    = max(0, end_frame - start_frame)

        # Seek 到起始帧
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        writer = make_writer(output_video, meta["fps"], meta["width"], meta["height"])

        print(f"[Tracker] 视频: {meta['width']}x{meta['height']}  {video_fps:.1f}fps  "
              f"处理帧: {start_frame} ~ {end_frame}（共{n_frames}帧）  "
              f"采样间隔: {sample_interval}帧")

        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        class_ids    = list(VEHICLE_CLASSES.keys())
        fps_list: list[float] = []
        rows_written = 0

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(f, fieldnames=self._CSV_FIELDS)
            writer_csv.writeheader()

            for local_idx, frame in iter_frames(cap, n_frames):
                frame_idx = start_frame + local_idx   # 视频中的绝对帧号
                t0 = time.perf_counter()

                # ByteTrack 跟踪推理（persist=True 保持跨帧状态）
                results = self._model.track(
                    frame,
                    classes=class_ids,
                    device=self.device,
                    conf=self.conf,
                    persist=True,
                    verbose=False,
                )[0]

                cur_fps = 1.0 / (time.perf_counter() - t0)
                fps_list.append(cur_fps)

                timestamp_s   = round(frame_idx / video_fps, 3)
                should_sample = (frame_idx % sample_interval == 0)

                boxes_xyxy, labels, confs, track_ids, plates, plate_boxes = [], [], [], [], [], []

                boxes_data = results.boxes
                if boxes_data is not None and len(boxes_data):
                    ids = (boxes_data.id.int().tolist()
                           if boxes_data.id is not None
                           else [None] * len(boxes_data))

                    for i, box in enumerate(boxes_data):
                        cls_id = int(box.cls[0])
                        if cls_id not in VEHICLE_CLASSES:
                            continue
                        label = VEHICLE_CLASSES[cls_id]
                        conf  = float(box.conf[0])
                        tid   = ids[i]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        boxes_xyxy.append((x1, y1, x2, y2))
                        labels.append(label)
                        confs.append(conf)
                        track_ids.append(tid)

                        # 每帧：从缓存取已确认车牌，用当前 bbox 还原绝对坐标
                        cached = (
                            self._plate_rec._cache.get(tid, ("", None))
                            if tid is not None else ("", None)
                        )
                        plate_display, cached_rel_box = cached
                        plate_box_display = (
                            _rel_to_abs(cached_rel_box, x1, y1, x2, y2)
                            if cached_rel_box is not None else None
                        )

                        # 轨迹采样记录
                        if should_sample and tid is not None:
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            plate_display, rel_box = self._plate_rec.recognize(
                                frame, tid, (x1, y1, x2, y2)
                            )
                            plate_box_display = (
                                _rel_to_abs(rel_box, x1, y1, x2, y2)
                                if rel_box is not None else None
                            )
                            writer_csv.writerow({
                                "frame_id":   frame_idx,
                                "timestamp_s": timestamp_s,
                                "track_id":   tid,
                                "class_name": label,
                                "cx":         cx,
                                "cy":         cy,
                                "x1":         x1,
                                "y1":         y1,
                                "x2":         x2,
                                "y2":         y2,
                                "plate":      plate_display,
                            })
                            rows_written += 1

                        plates.append(plate_display)
                        plate_boxes.append(plate_box_display)

                # 画真实车牌框和识别标签（含中文省份字符，走 PIL 渲染）
                for plate_str, plate_box in zip(plates, plate_boxes):
                    if not plate_str or plate_box is None:
                        continue
                    px1, py1, px2, py2 = plate_box
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
                    put_text(frame, plate_str, (px1, max(0, py1 - 22)), (0, 255, 255))

                # 绘制带跟踪ID和车牌的标注框
                draw_boxes(frame, boxes_xyxy, labels, confs, track_ids, plates)
                put_fps_text(frame, cur_fps, len(boxes_xyxy))
                writer.write(frame)

                if (local_idx + 1) % 30 == 0:
                    avg30 = sum(fps_list[-30:]) / min(len(fps_list), 30)
                    pct   = (local_idx + 1) / n_frames * 100
                    print(f"[{pct:5.1f}%] 帧 {frame_idx:4d}/{end_frame-1}  "
                          f"近30帧均FPS: {avg30:.1f}  "
                          f"已记录轨迹行: {rows_written}", flush=True)

        cap.release()
        writer.release()

        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0.0
        print(f"\n=== 轨迹提取完成 ===")
        print(f"处理帧数    : {len(fps_list)}")
        print(f"平均推理FPS : {avg_fps:.1f}")
        print(f"轨迹记录行  : {rows_written}")
        print(f"CSV 输出    : {csv_path}")
        print(f"视频输出    : {output_video}")
        return csv_path


if __name__ == "__main__":
    tracker = TrajectoryTracker()
    tracker.run(start_frame=_START_FRAME, end_frame=_END_FRAME)
