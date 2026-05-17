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
from src.utils.visualization import draw_boxes, put_fps_text

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
_OCR_CONF_THRESH = 0.2   # OCR 置信度阈值，低于此值的文字行丢弃


# ─────────────────────────────────────────────────────────────────────────────
class PlateRecognizer:
    """车牌识别器：首次识别成功后缓存结果，避免对同一车辆重复 OCR。

    OCR 引擎优先级：RapidOCR（ONNX）→ PaddleOCR → 禁用。
    RapidOCR 在 macOS M1 使用 CPU ONNX Runtime；迁移到 NVIDIA 时只需将
    ``onnxruntime`` 替换为 ``onnxruntime-gpu``，业务代码零改动。

    Attributes:
        _cache:   track_id -> plate_str 的识别缓存，每辆车仅 OCR 一次。
        _engine:  当前使用的引擎名称（"rapid" / "paddle" / None）。
        _enabled: False 表示无可用 OCR 引擎，recognize() 直接返回空字符串。
    """

    def __init__(self) -> None:
        """按优先级初始化 OCR 引擎，均不可用时降级为空实现。"""
        self._cache:   dict[int, str] = {}
        self._ocr      = None
        self._engine:  str | None = None
        self._enabled: bool = False

        # 优先尝试 RapidOCR（轻量，跨平台 ONNX）
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
            self._ocr     = RapidOCR()
            self._engine  = "rapid"
            self._enabled = True
            print("[PlateRecognizer] RapidOCR 加载成功")
            return
        except ImportError:
            pass

        # 其次尝试 PaddleOCR（NVIDIA 生产环境备选）
        try:
            from paddleocr import PaddleOCR  # type: ignore
            self._ocr     = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
            self._engine  = "paddle"
            self._enabled = True
            print("[PlateRecognizer] PaddleOCR 加载成功")
            return
        except ImportError:
            pass

        print("[PlateRecognizer] 未找到 OCR 引擎，车牌识别已禁用")

    def recognize(
        self,
        frame: np.ndarray,
        track_id: int,
        box_xyxy: tuple[int, int, int, int],
    ) -> str:
        """识别车牌，已成功识别的 track_id 直接返回缓存。

        Args:
            frame:    完整 BGR 帧。
            track_id: 跟踪 ID。
            box_xyxy: 车辆检测框 (x1, y1, x2, y2)。

        Returns:
            车牌字符串；未识别返回空字符串。
        """
        if track_id in self._cache:
            return self._cache[track_id]
        if not self._enabled:
            return ""

        crop = self._crop_plate_region(frame, box_xyxy)
        if crop is None:
            return ""

        crop = self._preprocess(crop)
        plate = self._run_ocr(crop)
        if plate:
            self._cache[track_id] = plate
        return plate

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _crop_plate_region(
        self,
        frame: np.ndarray,
        box_xyxy: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        """从车辆框中裁剪车牌候选区域。

        取框高底部 30%、宽度中间 80% 的矩形，并向下延伸 5% 补偿框偏紧情况。

        Args:
            frame:    完整 BGR 帧。
            box_xyxy: 车辆检测框 (x1, y1, x2, y2)。

        Returns:
            裁剪后的 BGR 图像；框过小时返回 None。
        """
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = box_xyxy
        h, w = y2 - y1, x2 - x1

        if h < 20 or w < 20:   # 车辆框过小，无法识别车牌
            return None

        # 底部 30% + 向下延伸 5%
        crop_y1 = max(0,  y2 - int(h * 0.30))
        crop_y2 = min(fh, y2 + int(h * 0.05))

        # 宽度收窄到中间 80%（去掉左右边缘遮挡）
        margin   = int(w * 0.10)
        crop_x1  = max(0,  x1 + margin)
        crop_x2  = min(fw, x2 - margin)

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        return crop if crop.size > 0 else None

    @staticmethod
    def _preprocess(crop: np.ndarray) -> np.ndarray:
        """对裁剪区域做图像增强，提升 OCR 准确率。

        流程：上采样（≥48px 高）→ 灰度化 → CLAHE 对比度增强 → 转回 BGR。

        Args:
            crop: BGR 裁剪图像。

        Returns:
            增强后的 BGR 图像。
        """
        # 上采样：OCR 对高度 < 48px 的图像准确率明显下降
        h, w = crop.shape[:2]
        if h < 48:
            scale = 48 / h
            crop  = cv2.resize(crop, (int(w * scale), 48),
                               interpolation=cv2.INTER_LINEAR)

        # CLAHE 对比度均衡化
        gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray  = clahe.apply(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _run_ocr(self, crop: np.ndarray) -> str:
        """对预处理后的图像运行 OCR，返回匹配车牌格式的字符串。

        Args:
            crop: 预处理后的 BGR 图像。

        Returns:
            匹配车牌正则且置信度达标的字符串，否则返回空字符串。
        """
        try:
            if self._engine == "rapid":
                result, _ = self._ocr(crop)
                if not result:
                    return ""
                for item in result:
                    # RapidOCR 格式: [box_points, text_str, score_str]
                    text  = item[1]
                    score = float(item[2])
                    if score < _OCR_CONF_THRESH:
                        continue
                    plate = self._match_plate(text)
                    if plate:
                        return plate

            elif self._engine == "paddle":
                result = self._ocr.ocr(crop, cls=True)
                if not result or not result[0]:
                    return ""
                for line in result[0]:
                    # PaddleOCR 格式: [box_points, (text_str, score_float)]
                    text  = line[1][0]
                    score = float(line[1][1])
                    if score < _OCR_CONF_THRESH:
                        continue
                    plate = self._match_plate(text)
                    if plate:
                        return plate

        except Exception:  # noqa: BLE001
            pass
        return ""

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

                boxes_xyxy, labels, confs, track_ids = [], [], [], []

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

                        # 轨迹采样记录
                        if should_sample and tid is not None:
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            plate = self._plate_rec.recognize(
                                frame, tid, (x1, y1, x2, y2)
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
                                "plate":      plate,
                            })
                            rows_written += 1

                # 绘制带跟踪ID的标注框
                draw_boxes(frame, boxes_xyxy, labels, confs, track_ids)
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
