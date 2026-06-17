"""
轨迹数据提取模块：ByteTrack 跟踪 + 车牌识别 + CSV 输出。

输出字段（trajectory.csv）::

    frame_id, timestamp_s, track_id, class_name, cx, cy, x1, y1, x2, y2, speed_kmh, plate

运行示例::

    python3 -u src/trajectory/tracker.py
"""
# 确保从任意目录运行时均可找到项目根（src 包）
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parents[2]))

import csv
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
    OUTPUT_DIR,
    MODEL_DIR,
    MODEL_NAME,
    DATA_DIR,
    TRAJ_SAMPLE_FPS,
    TRAJ_CSV_PATH,
    HOMOGRAPHY_MATRIX,
    PIXELS_PER_METER,
    CROSS_SECTION_CSV_PATH,
    CROSS_EVENT_COOLDOWN_FRAMES,
    SPEED_WINDOW_FRAMES,
    SPEED_MIN_DIST_M,
    SPEED_MIN_SAMPLES,
    VEHICLE_STATS_CSV_PATH,
    EXCEL_REPORT_PATH,
    TRAJ_GROUP_INTERVAL_S,
    TRAJ_GROUP_CSV_PATH,
)
from src.cross_section.counter import CrossSectionDetector
from src.cross_section.speed_estimator import SpeedEstimator
from src.cross_section.lane_calibration import get_calibration
from src.trajectory import lane_assignment
from src.trajectory.plate_recognizer import PLATE_VEHICLE_CLASSES, PlateRecognizer, rel_to_abs
from src.trajectory.traffic_report import (
    backfill_plates,
    export_excel,
    rewrite_dict_csv,
    write_vehicle_stats_row,
)
from src.trajectory.tracker_runtime import TrackerOutputLock, resolve_section_lines
from src.trajectory.traj_grouper import TrajGrouper
from src.utils.video_io import open_video, video_meta, make_writer, iter_frames, AsyncWriter
from src.utils.visualization import draw_boxes, put_fps_text, put_text

_MODEL       = MODEL_DIR / MODEL_NAME          # 推理模型路径
_START_FRAME = 0                               # 起始帧号（含），0 = 视频开头
_END_FRAME   = 9000                            # 终止帧号（不含），None = 视频结尾
_GRACE_FRAMES = 10                              # track消失后保留帧数，防止碎片化
_OUTPUT_VID  = OUTPUT_DIR / "trajectory.mp4"  # 带轨迹标注的输出视频
_LIVE_PREVIEW_FPS = 18.0                       # Web 实时预览发布帧率


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
        "frame_id", "timestamp_s", "track_id", "class_name", "lane_id", "lane_type",
        "cx", "cy", "x1", "y1", "x2", "y2", "speed_kmh", "plate",
    ]
    _CROSS_CSV_FIELDS = CrossSectionDetector.CSV_FIELDS

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
        if device == "cuda":
            self._model.model.half()
        self._plate_rec = PlateRecognizer()
        self._bbox_smooth: dict[int, tuple[int, int, int, int]] = {}
        print(f"[Tracker] 模型: {self.model_path.name}  设备: {device}")

    @staticmethod
    def _build_lane_overlay(
        height: int, width: int,
        lanes: dict[int, list[tuple[int, int]]],
    ) -> np.ndarray:
        return lane_assignment.build_lane_overlay(height, width, lanes)

    @staticmethod
    def _assign_lane(
        cx: float, cy: float,
        lanes: dict[int, list[tuple[int, int]]],
    ) -> int | str | None:
        return lane_assignment.assign_lane(cx, cy, lanes)

    @staticmethod
    def _build_lane_splines(
        lanes: dict[int, list[tuple[int, int]]],
    ) -> dict[int, tuple[float, float, object]]:
        return lane_assignment.build_lane_splines(lanes)

    @staticmethod
    def _project_point_to_segment(
        px: float,
        py: float,
        ax: float,
        ay: float,
        bx: float,
        by: float,
    ) -> tuple[float, float]:
        return lane_assignment.project_point_to_segment(px, py, ax, ay, bx, by)

    @staticmethod
    def _lane_boundary_xs_at_y(
        y: float,
        lane_splines: dict[int, tuple[float, float, object]],
    ) -> dict[int, float]:
        return lane_assignment.lane_boundary_xs_at_y(y, lane_splines)

    @staticmethod
    def _compute_lane_boundary_cross_t(
        section_line: tuple,
        lane_splines: dict[int, tuple[float, float, object]],
    ) -> dict[int, float]:
        return lane_assignment.compute_lane_boundary_cross_t(section_line, lane_splines)

    @staticmethod
    def _assign_section_event_lane(
        section_name: str,
        class_name: str,
        direction: str,
        cross_t: float | None,
        boundary_t: dict[int, float] | None,
    ) -> int | str | None:
        return lane_assignment.assign_section_event_lane(
            section_name, class_name, direction, cross_t, boundary_t,
        )
    @staticmethod
    def _trajectory_matches_turn_section(
        section_name: str,
        class_name: str,
        traj_points: list[tuple[float, float]],
    ) -> bool:
        return lane_assignment.trajectory_matches_turn_section(section_name, class_name, traj_points)

    @staticmethod
    def _rewrite_dict_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
        rewrite_dict_csv(path, fields, rows)

    @staticmethod
    def _backfill_plates(
        traj_records: list[dict],
        cross_events: list[dict],
        plate_cache: dict[int, tuple[str, tuple[float, float, float, float] | None]],
    ) -> int:
        return backfill_plates(traj_records, cross_events, plate_cache)

    def run(
        self,
        video_path: str | Path | None = None,
        output_video: str | Path = _OUTPUT_VID,
        csv_path: str | Path = TRAJ_CSV_PATH,
        start_frame: int = _START_FRAME,
        end_frame: Optional[int] = _END_FRAME,
        sample_fps: int = TRAJ_SAMPLE_FPS,
        live_publisher=None,
        stop_event=None,
    ) -> Path:
        """运行跟踪并输出轨迹 CSV 和标注视频。

        Args:
            video_path:   输入视频路径。
            output_video: 输出带标注视频路径。
            csv_path:     输出轨迹 CSV 路径。
            start_frame:  起始帧号（含），默认 0。
            end_frame:    终止帧号（不含），None 表示处理到视频结尾。
            sample_fps:   每秒采样次数（轨迹记录频率）。
            live_publisher: 可选实时发布器，接收标注帧、进度和统计快照。
            stop_event:     可选停止信号，用于 Web 控制台请求优雅退出。

        Returns:
            写出的 CSV 文件路径。
        """
        if video_path is None:
            mp4s = sorted(DATA_DIR.glob("*.mp4"))
            video_path = mp4s[0] if mp4s else DATA_DIR / "test_video.mp4"
        _output_lock = TrackerOutputLock(OUTPUT_DIR / ".tracker.lock").__enter__()
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

        writer = AsyncWriter(make_writer(output_video, meta["fps"], 1920, 1080))

        print(f"[Tracker] 视频: {meta['width']}x{meta['height']}  {video_fps:.1f}fps  "
              f"处理帧: {start_frame} ~ {end_frame}（共{n_frames}帧）  "
              f"采样间隔: {sample_interval}帧")

        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        class_ids    = list(VEHICLE_CLASSES.keys())
        fps_list: list[float] = []
        rows_written = 0
        _t_infer = _t_post = _t_cross = _t_draw = 0.0  # 各阶段累计耗时

        # 内存缓存，供最终 Excel 导出
        _all_traj_records: list[dict] = []
        _all_cross_events: list[dict] = []

        cal = get_calibration(video_path)
        if cal is None:
            print("[标定] ⚠ 加载失败，退化到默认值")
            _H = HOMOGRAPHY_MATRIX
            _lanes: dict[int, list] = {}
            _lane_types: dict[str, str] = {}
            _lane_overlay: np.ndarray | None = None
        else:
            _H = cal.homography
            _lanes = cal.lanes
            _lane_types: dict[str, str] = cal.metadata.get("lane_types", {})
            print(f"[标定] ✓ {cal.entrance} | 车道线 {len(_lanes)} 条 | "
                  f"H={cal.homography_method} | 光照={cal.lighting_preset}")
            _lane_overlay = self._build_lane_overlay(
                meta["height"], meta["width"], _lanes
            )
        _lane_splines = self._build_lane_splines(_lanes) if _lanes else {}

        # 测速器
        speed_est = SpeedEstimator(
            homography=_H,
            fps=video_fps,
            window=SPEED_WINDOW_FRAMES,
            pixels_per_meter=PIXELS_PER_METER,
            min_dist_m=SPEED_MIN_DIST_M,
        )

        # 轨迹分组器
        _entrance = cal.entrance if cal is not None else "未知"
        _grouper = TrajGrouper(
            interval_s=TRAJ_GROUP_INTERVAL_S,
            csv_path=TRAJ_GROUP_CSV_PATH,
            excel_path=EXCEL_REPORT_PATH,
            frame_width=meta["width"],
            frame_height=meta["height"],
        )

        # 断面检测器（复用同一 SpeedEstimator）
        _section_lines = resolve_section_lines(video_path)
        _section_lines_by_name = {line[0]: line for line in _section_lines}
        # 预计算每个断面的车道分界线 t 阈值
        _boundary_t_by_section: dict[str, dict[int, float]] = {}
        if _lane_splines:
            for sec_name, sec_line in _section_lines_by_name.items():
                bt = self._compute_lane_boundary_cross_t(sec_line, _lane_splines)
                if bt:
                    _boundary_t_by_section[sec_name] = bt
        section_det = CrossSectionDetector(
            _section_lines, _H, PIXELS_PER_METER, video_fps,
            speed_estimator=speed_est,
            cooldown_frames=CROSS_EVENT_COOLDOWN_FRAMES,
        )
        _section_hit: dict[str, int] = {}
        _HIGHLIGHT_FRAMES = 8

        cross_path = Path(CROSS_SECTION_CSV_PATH)
        cross_path.parent.mkdir(parents=True, exist_ok=True)
        cross_f = cross_path.open("w", newline="", encoding="utf-8")
        cross_writer = csv.DictWriter(cross_f, fieldnames=self._CROSS_CSV_FIELDS)
        cross_writer.writeheader()

        stats_path = Path(VEHICLE_STATS_CSV_PATH)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_fh = open(stats_path, 'w', newline='', encoding='utf-8')
        stats_writer = csv.writer(stats_fh)
        stats_writer.writerow([
            "track_id", "first_frame", "last_frame", "lane_id",
            "avg_speed_kmh", "max_speed_kmh", "min_speed_kmh", "n_samples",
        ])

        last_known_lane: dict[int, int | str | None] = {}
        active_tids: set[int] = set()
        prev_active: set[int] = set()
        _traj_buf: dict[int, list[tuple[float, float]]] = {}
        _tid_cls: dict[int, str] = {}
        _grace_buf: dict[int, int] = {}   # tid → 剩余存活帧数
        _live_vehicle_ids: set[int] = set()
        _live_recent_events: list[dict] = []
        _LIVE_FRAME_INTERVAL = max(1, round(video_fps / _LIVE_PREVIEW_FPS))

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(f, fieldnames=self._CSV_FIELDS)
            writer_csv.writeheader()

            for local_idx, frame in iter_frames(cap, n_frames):
                if stop_event is not None and stop_event.is_set():
                    print("[Tracker] 收到停止请求，正在结束检测")
                    break

                frame_idx = start_frame + local_idx   # 视频中的绝对帧号
                t0 = time.perf_counter()

                # ByteTrack 跟踪推理（persist=True 保持跨帧状态）
                results = self._model.track(
                    frame,
                    classes=class_ids,
                    device=self.device,
                    conf=self.conf,
                    persist=True,
                    tracker="botsort.yaml",
                    verbose=False,
                    half=True,
                )[0]
                t1 = time.perf_counter()
                _t_infer += t1 - t0

                cur_fps = 1.0 / (t1 - t0)
                fps_list.append(cur_fps)

                timestamp_s   = round(frame_idx / video_fps, 3)
                should_sample = (frame_idx % sample_interval == 0)

                boxes_xyxy, labels, confs, track_ids, plates, plate_boxes = [], [], [], [], [], []
                _live_class_counts: dict[str, int] = {}
                _live_lane_counts: dict[str, int] = {}
                _live_speed_samples: list[float] = []

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

                        # 非机动车小框抖动大，用指数平滑稳定车道判定。
                        if tid is not None:
                            alpha = 0.3 if label in ("motorcycle", "bicycle") else 0.6
                            prev = self._bbox_smooth.get(tid)
                            if prev is not None:
                                x1 = int(alpha * x1 + (1 - alpha) * prev[0])
                                y1 = int(alpha * y1 + (1 - alpha) * prev[1])
                                x2 = int(alpha * x2 + (1 - alpha) * prev[2])
                                y2 = int(alpha * y2 + (1 - alpha) * prev[3])
                            self._bbox_smooth[tid] = (x1, y1, x2, y2)

                        # YOLO 在远处容易把电动车/摩托车误判为 car。
                        # 用 bbox 尺寸 + 长宽比信号纠正。
                        # 数据驱动阈值: motorcycle 面积中位数 6209, aspect 中位数 1.59
                        #               car 面积中位数 13965, aspect 中位数 0.82
                        if label == "car":
                            bw = x2 - x1
                            bh = y2 - y1
                            area = bw * bh
                            aspect = bh / max(bw, 1)
                            if area < 15000 and aspect > 1.2:
                                label = "motorcycle"

                        boxes_xyxy.append((x1, y1, x2, y2))
                        labels.append(label)
                        confs.append(conf)
                        track_ids.append(tid)
                        _live_class_counts[label] = _live_class_counts.get(label, 0) + 1

                        # 每帧：对未锁定车辆积累投票，锁定后不再调用 HyperLPR3
                        if (tid is not None
                                and label in PLATE_VEHICLE_CLASSES
                                and tid not in self._plate_rec.cache):
                            self._plate_rec.recognize(frame, tid, (x1, y1, x2, y2))

                        # 每帧：从缓存取已确认车牌，用当前 bbox 还原绝对坐标
                        cached = (
                            self._plate_rec.cache.get(tid, ("", None))
                            if tid is not None else ("", None)
                        )
                        plate_display, cached_rel_box = cached
                        plate_box_display = (
                            rel_to_abs(cached_rel_box, x1, y1, x2, y2)
                            if cached_rel_box is not None else None
                        )

                        # 测速 + 车道归属（必须在轨迹采样前完成，保证本帧速度/车道已更新）
                        cx_c = float((x1 + x2) / 2)
                        cy_b = float(y2)
                        v_now: float | None = None
                        lane_id: int | None = None
                        if tid is not None:
                            speed_est.update(frame_idx, tid, cx_c, cy_b)
                            v_now = speed_est.instant_speed(tid)
                            lane_id = self._assign_lane(cx_c, cy_b, _lanes) if _lanes else None
                            if lane_id is not None:
                                last_known_lane[tid] = lane_id
                                _live_lane_counts[str(lane_id)] = _live_lane_counts.get(str(lane_id), 0) + 1
                            active_tids.add(tid)
                            _live_vehicle_ids.add(tid)
                            if tid in _grace_buf:
                                del _grace_buf[tid]   # 从 grace 复活
                            _traj_buf.setdefault(tid, []).append((cx_c, float((y1 + y2) / 2)))
                            _tid_cls[tid] = label
                            if v_now is not None and v_now > 0:
                                _live_speed_samples.append(float(v_now))

                            # 速度+车道标签（bbox 下方，避免与类别标签重叠）
                            parts = []
                            if lane_id is not None:
                                parts.append(f"L{lane_id}")
                            if v_now is not None and v_now > 0:
                                parts.append(f"{v_now:.0f}km/h")
                            if parts:
                                label_speed = " ".join(parts)
                                put_text(frame, label_speed,
                                         (int(x1), min(frame.shape[0] - 20, int(y2) + 24)),
                                         color=(0, 255, 255),
                                         font_scale=0.75)

                        # 轨迹采样记录（speed/lane 已是本帧最新值）
                        # lane_id 为空表示对向/越界车辆，仍记录轨迹，但统计时会过滤
                        _cur_lane = last_known_lane.get(tid)
                        if should_sample and tid is not None:
                            cx = int(cx_c)
                            cy = (y1 + y2) // 2
                            plate_display, rel_box = self._plate_rec.cache.get(tid, ("", None))
                            plate_box_display = (
                                rel_to_abs(rel_box, x1, y1, x2, y2)
                                if rel_box is not None else None
                            )
                            _speed_kmh = round(v_now, 1) if v_now is not None else None
                            _lane_type = _lane_types.get(str(_cur_lane), "") if _cur_lane else ""
                            _row = {
                                "frame_id":    frame_idx,
                                "timestamp_s": timestamp_s,
                                "track_id":    tid,
                                "class_name":  label,
                                "lane_id":     _cur_lane,
                                "lane_type":   _lane_type,
                                "cx":          cx,
                                "cy":          cy,
                                "x1":          x1,
                                "y1":          y1,
                                "x2":          x2,
                                "y2":          y2,
                                "speed_kmh":   _speed_kmh,
                                "plate":       plate_display,
                            }
                            writer_csv.writerow(_row)
                            _all_traj_records.append(_row)
                            rows_written += 1

                        plates.append(plate_display)
                        plate_boxes.append(plate_box_display)

                t2 = time.perf_counter()
                _t_post += t2 - t1

                # 断面过车检测。
                for _box, _label, _tid in zip(boxes_xyxy, labels, track_ids):
                    if _tid is None:
                        continue
                    _bx1, _by1, _bx2, _by2 = _box
                    for ev in section_det.update(
                        frame_idx, timestamp_s, _tid, _label, last_known_lane.get(_tid),
                        frame, _bx1, _by1, _bx2, _by2,
                    ):
                        if not self._trajectory_matches_turn_section(
                            str(ev.get("section", "")),
                            _label,
                            _traj_buf.get(_tid, []),
                        ):
                            continue
                        # 主断面：直接用 _assign_lane() 在车辆实际y处的判定结果，
                        # 不再用断面投影重算（投影法已验证无效）。
                        # 转弯断面：仍用 _assign_section_event_lane 做几何约束。
                        sec_name = str(ev.get("section", ""))
                        if _lane_splines and "主断面" not in sec_name:
                            lane_override = self._assign_section_event_lane(
                                sec_name,
                                _label,
                                str(ev.get("direction", "")),
                                ev.get("cross_t"),
                                _boundary_t_by_section.get(sec_name),
                            )
                            if lane_override:
                                ev["lane_id"] = lane_override
                        # 将已识别车牌绑定到过线事件（按 track_id 查缓存）
                        ev["plate"] = self._plate_rec.cache.get(_tid, ("", None))[0]

                        ev["lane_rule"] = "bbox_bottom" if "主断面" in sec_name else ("cross_segment" if _boundary_t_by_section else "last_known")

                        cross_writer.writerow(ev)
                        _all_cross_events.append(ev)
                        _live_recent_events.append({
                            "timestamp_s": ev.get("timestamp_s", timestamp_s),
                            "section": ev.get("section", ""),
                            "direction": ev.get("direction", ""),
                            "class_name": ev.get("class_name", ""),
                            "speed_kmh": ev.get("speed_kmh", 0),
                            "plate": ev.get("plate", ""),
                        })
                        _live_recent_events = _live_recent_events[-8:]
                        _section_hit[ev["section"]] = frame_idx

                t3 = time.perf_counter()
                _t_cross += t3 - t2

                # 缩放至 1080p 再绘制，避免 4K 画图拖慢主流程。
                S = 0.5  # 3840→1920, 2160→1080
                frame = cv2.resize(frame, (1920, 1080))
                _lane_1080 = cv2.resize(_lane_overlay, (1920, 1080)) if _lane_overlay is not None else None
                _sec_lines_1080 = [(_n, int(_a*S), int(_b*S), int(_c*S), int(_d*S), _e, _f)
                                   for _n, _a, _b, _c, _d, _e, _f in _section_lines]
                _boxes_1080 = [(int(x1*S), int(y1*S), int(x2*S), int(y2*S)) for (x1, y1, x2, y2) in boxes_xyxy]
                _plates_1080 = [(int(p[0]*S), int(p[1]*S), int(p[2]*S), int(p[3]*S)) if p else None
                                for p in plate_boxes]

                # 断面线绘制在车辆框下层。
                for _name, _lx1, _ly1, _lx2, _ly2, _, _ in _sec_lines_1080:
                    _age = frame_idx - _section_hit.get(_name, -999)
                    _col = (0, 255, 255) if _age <= _HIGHLIGHT_FRAMES else (0, 0, 255)
                    _thi = 2              if _age <= _HIGHLIGHT_FRAMES else 1
                    cv2.line(frame, (_lx1, _ly1), (_lx2, _ly2), _col, _thi)
                    put_text(frame, _name, (_lx1, _ly1 - 12), _col)

                # 画真实车牌框和识别标签（含中文省份字符，走 PIL 渲染）
                for plate_str, plate_box in zip(plates, _plates_1080):
                    if not plate_str or plate_box is None:
                        continue
                    px1, py1, px2, py2 = plate_box
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 1)
                    put_text(frame, plate_str, (px1, max(0, py1 - 11)), (0, 255, 255))

                # 绘制带跟踪ID和车牌的标注框
                draw_boxes(frame, _boxes_1080, labels, confs, track_ids, plates)
                put_fps_text(frame, cur_fps, len(_boxes_1080))

                # 车道线 overlay 叠加
                annotated = frame
                if _lane_1080 is not None:
                    annotated = cv2.addWeighted(frame, 1.0, _lane_1080, 0.5, 0)

                if live_publisher is not None:
                    avg_live_speed = (
                        sum(_live_speed_samples) / len(_live_speed_samples)
                        if _live_speed_samples else 0.0
                    )
                    live_publisher.update_progress(frame_idx, timestamp_s, cur_fps)
                    live_publisher.publish_stats({
                        "vehicles": len(_live_vehicle_ids),
                        "events": len(_all_cross_events),
                        "avg_speed": round(avg_live_speed, 1),
                        "active_tracks": sum(1 for tid in track_ids if tid is not None),
                        "recent_events": list(reversed(_live_recent_events)),
                        "class_counts": _live_class_counts,
                        "lane_counts": _live_lane_counts,
                    })
                    if frame_idx % _LIVE_FRAME_INTERVAL == 0:
                        live_publisher.publish_frame(annotated, width=960, quality=68)

                # 消失的 track → 进入 grace buffer
                for tid in (prev_active - active_tids):
                    if tid not in _grace_buf:
                        _grace_buf[tid] = _GRACE_FRAMES

                # grace 到期 → 真正结束
                expired = [tid for tid, c in _grace_buf.items() if c <= 0]
                for tid in expired:
                    del _grace_buf[tid]
                    s = speed_est.finalize(tid)
                    write_vehicle_stats_row(stats_writer, tid, s, last_known_lane.get(tid), SPEED_MIN_SAMPLES)
                    _pts = _traj_buf.pop(tid, [])
                    _lid = last_known_lane.get(tid)
                    _grouper.on_track_end(
                        tid, _pts, _tid_cls.pop(tid, "car"),
                        _lid,
                        _lane_types.get(str(_lid), ""),
                        _entrance,
                    )
                    last_known_lane.pop(tid, None)

                # 递减未到期的 grace 计数
                for tid in list(_grace_buf):
                    _grace_buf[tid] -= 1

                prev_active = set(active_tids)
                active_tids.clear()

                writer.write(annotated)

                t4 = time.perf_counter()
                _t_draw += t4 - t3

                _grouper.tick(timestamp_s)

                if (local_idx + 1) % 30 == 0:
                    avg30 = sum(fps_list[-30:]) / min(len(fps_list), 30)
                    pct   = (local_idx + 1) / n_frames * 100
                    n = min(local_idx + 1, 30)
                    print(f"[{pct:5.1f}%] 帧 {frame_idx:4d}/{end_frame-1}  "
                          f"FPS:{avg30:.1f} "
                          f"推理:{_t_infer/n*1000:.0f}ms "
                          f"后处理:{_t_post/n*1000:.0f}ms "
                          f"断面:{_t_cross/n*1000:.0f}ms "
                          f"绘制:{_t_draw/n*1000:.0f}ms "
                          f"轨迹:{rows_written}", flush=True)
                    _t_infer = _t_post = _t_cross = _t_draw = 0.0

        # finalize grace buffer 中的 track（视频结束，强制到期）
        for tid in list(_grace_buf):
            del _grace_buf[tid]
            s = speed_est.finalize(tid)
            write_vehicle_stats_row(stats_writer, tid, s, last_known_lane.get(tid), SPEED_MIN_SAMPLES)
            _pts = _traj_buf.pop(tid, [])
            _lid = last_known_lane.get(tid)
            _grouper.on_track_end(
                tid, _pts, _tid_cls.pop(tid, "car"),
                _lid,
                _lane_types.get(str(_lid), ""),
                _entrance,
            )
            last_known_lane.pop(tid, None)

        # finalize 剩余 track
        for tid in list(speed_est._tracks.keys()):
            s = speed_est.finalize(tid)
            write_vehicle_stats_row(stats_writer, tid, s, last_known_lane.get(tid), SPEED_MIN_SAMPLES)
        stats_fh.close()
        print(f"[Stats] vehicle_stats.csv: {stats_path}")

        cap.release()
        writer.release()
        cross_f.close()

        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0.0
        print(f"\n=== 轨迹提取完成 ===")
        print(f"处理帧数    : {len(fps_list)}")
        print(f"平均推理FPS : {avg_fps:.1f}")
        print(f"轨迹记录行  : {rows_written}")
        print(f"CSV 输出    : {csv_path}")
        print(f"断面CSV     : {cross_path}")
        print(f"视频输出    : {output_video}")

        from collections import defaultdict as _dd
        _sec_cat: dict[str, dict[str, int]] = _dd(lambda: _dd(int))
        for _ev in _all_cross_events:
            _sec_cat[_ev["section"]][_ev.get("vehicle_category", "未知")] += 1
        unique_tids = len({r["track_id"] for r in _all_traj_records})
        total_motor    = sum(e.get("vehicle_category") == "机动车"  for e in _all_cross_events)
        total_nonmotor = sum(e.get("vehicle_category") == "非机动车" for e in _all_cross_events)

        print(f"\n=== 断面过车摘要 ===")
        for sec, cats in sorted(_sec_cat.items()):
            motor    = cats.get("机动车",  0)
            nonmotor = cats.get("非机动车", 0)
            print(f"  {sec:<18} 机动车 {motor:3d}辆   非机动车 {nonmotor:3d}辆   合计 {motor+nonmotor:3d}辆")
        print(f"跟踪车辆总数    : {unique_tids} 辆（唯一 track_id）")
        print(f"断面过车合计    : 机动车 {total_motor} 辆  /  非机动车 {total_nonmotor} 辆")

        filled_plates = self._backfill_plates(
            _all_traj_records,
            _all_cross_events,
            self._plate_rec.cache,
        )
        self._rewrite_dict_csv(csv_path, self._CSV_FIELDS, _all_traj_records)
        self._rewrite_dict_csv(cross_path, self._CROSS_CSV_FIELDS, _all_cross_events)
        if filled_plates:
            print(f"[Plate] 已按 track_id 回填车牌 {filled_plates} 处，并重写 CSV")

        export_excel(
            _all_traj_records, _all_cross_events,
            self._CSV_FIELDS, self._CROSS_CSV_FIELDS,
        )
        _grouper.finalize()
        _output_lock.__exit__(None, None, None)
        return csv_path

if __name__ == "__main__":
    tracker = TrajectoryTracker()
    tracker.run(start_frame=_START_FRAME, end_frame=_END_FRAME)
