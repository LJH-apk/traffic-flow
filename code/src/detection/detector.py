"""
VehicleDetector：基于 YOLO 的车辆逐帧检测，支持流式视频写出。

运行示例::

    python3 -u src/detection/detector.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parents[2]))

import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.config.settings import (
    VEHICLE_CLASSES,
    DEVICE,
    CONF_THRESH,
    DATA_DIR,
    OUTPUT_DIR,
    MODEL_DIR,
    MODEL_NAME,
)
from src.utils.video_io import open_video, video_meta, make_writer, iter_frames
from src.utils.visualization import draw_boxes, put_fps_text

# ── 运行时配置（可在此处直接修改） ───────────────────────────────────────────
_MODEL      = MODEL_DIR / MODEL_NAME          # 推理模型路径
_MAX_FRAMES = 1000                            # None = 处理全部帧
_OUTPUT     = OUTPUT_DIR / "detection.mp4"   # 输出视频路径


class VehicleDetector:
    """逐帧目标检测器，内部调用 YOLO 推理并流式写出带标注视频。

    Attributes:
        model_path: YOLO 权重路径。
        device:     推理设备（mps / cpu / cuda）。
        conf:       置信度阈值。
    """

    def __init__(
        self,
        model_path: str | Path = _MODEL,
        device: str = DEVICE,
        conf: float = CONF_THRESH,
    ) -> None:
        """初始化检测器并加载模型。

        Args:
            model_path: YOLO 权重文件路径。
            device:     推理设备。
            conf:       置信度阈值。
        """
        self.model_path = Path(model_path)
        self.device     = device
        self.conf       = conf
        self._model     = YOLO(str(self.model_path))
        self._model.to(device)
        print(f"[Detector] 模型: {self.model_path.name}  设备: {device}")

    def run(
        self,
        video_path: str | Path | None = None,
        output_path: str | Path = _OUTPUT,
        max_frames: int | None = _MAX_FRAMES,
    ) -> dict:
        """运行检测并将结果写出为视频。

        Args:
            video_path:  输入视频路径。
            output_path: 输出视频路径。
            max_frames:  最大处理帧数，None 表示全量。

        Returns:
            包含 avg_fps / min_fps / max_fps / frame_count / vehicle_counts 的统计字典。
        """
        if video_path is None:
            mp4s = sorted(DATA_DIR.glob("*.mp4"))
            video_path = mp4s[0] if mp4s else DATA_DIR / "test_video.mp4"
        cap    = open_video(video_path)
        meta   = video_meta(cap)
        writer = make_writer(output_path, meta["fps"], meta["width"], meta["height"])

        total  = meta["frame_count"] if max_frames is None else min(meta["frame_count"], max_frames)
        print(f"[Detector] 视频: {meta['width']}x{meta['height']}  "
              f"{meta['fps']:.1f}fps  共{total}帧")

        fps_list       = []
        vehicle_counts = {cls: 0 for cls in VEHICLE_CLASSES.values()}
        class_ids      = list(VEHICLE_CLASSES.keys())

        for frame_idx, frame in iter_frames(cap, max_frames):
            t0      = time.perf_counter()
            results = self._model(
                frame,
                classes=class_ids,
                device=self.device,
                conf=self.conf,
                verbose=False,
            )[0]
            cur_fps = 1.0 / (time.perf_counter() - t0)
            fps_list.append(cur_fps)

            # 提取检测结果
            boxes_xyxy, labels, confs = [], [], []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue
                label = VEHICLE_CLASSES[cls_id]
                boxes_xyxy.append(tuple(map(int, box.xyxy[0])))
                labels.append(label)
                confs.append(float(box.conf[0]))
                vehicle_counts[label] += 1

            draw_boxes(frame, boxes_xyxy, labels, confs)
            put_fps_text(frame, cur_fps, len(results.boxes))
            writer.write(frame)

            if (frame_idx + 1) % 30 == 0:
                avg30 = sum(fps_list[-30:]) / min(len(fps_list), 30)
                pct   = (frame_idx + 1) / total * 100
                print(f"[{pct:5.1f}%] 帧 {frame_idx+1:4d}/{total}  "
                      f"近30帧均FPS: {avg30:.1f}", flush=True)

        cap.release()
        writer.release()

        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0.0
        stats = {
            "frame_count":    len(fps_list),
            "avg_fps":        round(avg_fps, 2),
            "min_fps":        round(min(fps_list), 2) if fps_list else 0.0,
            "max_fps":        round(max(fps_list), 2) if fps_list else 0.0,
            "vehicle_counts": vehicle_counts,
        }

        print(f"\n=== 检测完成 ===")
        print(f"处理帧数    : {stats['frame_count']}")
        print(f"平均推理FPS : {stats['avg_fps']:.1f}  "
              f"(min {stats['min_fps']:.1f} / max {stats['max_fps']:.1f})")
        print(f"比赛要求    : "
              f"{'✓ 达标(≥15fps)' if avg_fps >= 15 else '✗ 未达标(<15fps)'}")
        print(f"\n各类别累计检测数:")
        for cls, cnt in vehicle_counts.items():
            if cnt > 0:
                print(f"  {cls:12s}: {cnt}")
        print(f"\n输出视频: {output_path}")
        return stats


if __name__ == "__main__":
    det = VehicleDetector()
    det.run()
