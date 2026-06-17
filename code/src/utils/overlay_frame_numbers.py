"""
给标注视频叠加帧号和时间戳。

默认输出到 outputs/annotation_videos/<原文件名>_5min_framed_1080p.mp4，
便于人工过线标注时直接读取 gt_frame_id。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.video_io import make_writer, open_video, video_meta
from src.config.settings import CALIBRATIONS_DIR, DATA_DIR, OUTPUT_DIR, SECTION_LINES_MAP
from src.cross_section.section_calibration import load_section_lines, load_stop_lines
from src.utils.visualization import put_text


DEFAULT_OUTPUT_DIR = Path("outputs") / "annotation_videos"
DEFAULT_DURATION_S = 300.0
DEFAULT_WIDTH = 1920


def default_output_path(source: Path, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """根据输入视频路径生成默认帧号版输出路径。"""
    return output_dir / f"{source.stem}_5min_framed_1080p.mp4"


def default_reference_output_path(source: Path, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """根据输入视频路径生成断面/车道参考版输出路径。"""
    return output_dir / f"{source.stem}_5min_reference_1080p.mp4"


def infer_entrance(source: Path) -> str | None:
    """从视频文件名推断进口名；test_video_fixed.mp4 按用户说明视为南进口。"""
    name = source.name
    if source.name == "test_video_fixed.mp4":
        return "南进口"
    for entrance in SECTION_LINES_MAP:
        if entrance in name:
            return entrance
    return None


def load_lanes(entrance: str | None) -> dict[int, list[tuple[int, int]]]:
    """加载指定进口的车道线标定。"""
    if not entrance:
        return {}
    lanes_path = CALIBRATIONS_DIR / entrance / "lanes.json"
    if not lanes_path.exists():
        return {}
    raw = json.loads(lanes_path.read_text(encoding="utf-8"))
    raw_lanes = raw.get("lanes", raw)
    lanes: dict[int, list[tuple[int, int]]] = {}
    for key, pts in raw_lanes.items():
        try:
            lane_id = int(key)
        except (TypeError, ValueError):
            continue
        lanes[lane_id] = [tuple(map(int, pt)) for pt in pts]
    return lanes


def draw_frame_label(frame: np.ndarray, frame_idx: int, fps: float) -> np.ndarray:
    """在帧左上角叠加大号帧号和时间戳。"""
    timestamp_s = frame_idx / fps if fps > 0 else 0.0
    label = f"Frame {frame_idx}   Time {timestamp_s:.2f}s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 2
    x, y = 18, 34

    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    pad = 8
    x1, y1 = max(0, x - pad), max(0, y - th - pad)
    x2, y2 = min(frame.shape[1], x + tw + pad), min(frame.shape[0], y + baseline + pad)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    cv2.putText(frame, label, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(frame, label, (x, y), font, scale, (0, 210, 255), 1, cv2.LINE_AA)
    return frame


def draw_reference_overlay(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    section_lines: list[tuple[str, int, int, int, int, str, str]],
    lanes: dict[int, list[tuple[int, int]]],
    stop_lines: list[tuple[str, int, int, int, int]] | None = None,
) -> np.ndarray:
    """叠加帧号、断面线和车道线，供人工标注使用。"""
    draw_frame_label(frame, frame_idx, fps)

    lane_colors = {
        1: (0, 255, 0),
        2: (0, 200, 255),
        3: (255, 80, 200),
        4: (255, 200, 0),
        5: (80, 200, 255),
    }
    for lane_id, pts in sorted(lanes.items()):
        if len(pts) < 2:
            continue
        arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        color = lane_colors.get(lane_id, (220, 220, 220))
        cv2.polylines(frame, [arr], False, color, 3, cv2.LINE_AA)
        label_x, label_y = pts[min(len(pts) - 1, len(pts) // 2)]
        cv2.putText(
            frame,
            f"L{lane_id}",
            (label_x + 10, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )

    for name, lx1, ly1, lx2, ly2 in stop_lines or []:
        cv2.line(frame, (lx1, ly1), (lx2, ly2), (255, 255, 255), 4, cv2.LINE_AA)
        cv2.line(frame, (lx1, ly1), (lx2, ly2), (0, 180, 255), 2, cv2.LINE_AA)
        put_text(frame, name, (lx1, max(0, ly1 - 24)), (0, 180, 255), font_scale=0.45, thickness=1)

    for name, lx1, ly1, lx2, ly2, dir_pos, dir_neg in section_lines:
        cv2.line(frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, (lx1, ly1), 5, (0, 255, 255), -1)
        cv2.circle(frame, (lx2, ly2), 5, (0, 255, 255), -1)
        label = f"{name}  {dir_pos}/{dir_neg}"
        tx, ty = lx1, max(36, ly1 - 14)
        cv2.rectangle(frame, (tx - 6, ty - 28), (tx + 230, ty + 4), (0, 0, 0), -1)
        put_text(frame, label, (tx, ty - 24), (0, 255, 255), font_scale=0.45, thickness=1)
    return frame


def overlay_video(
    source: Path,
    output: Path | None = None,
    max_frames: int | None = None,
) -> Path:
    """读取视频，逐帧叠加帧号，写出新 MP4。"""
    source = Path(source)
    output = output or default_output_path(source)

    cap = open_video(source)
    meta = video_meta(cap)
    writer = make_writer(output, meta["fps"], meta["width"], meta["height"])

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break
            draw_frame_label(frame, frame_idx, meta["fps"])
            writer.write(frame)
            frame_idx += 1
            if frame_idx % 250 == 0:
                total = meta["frame_count"] or 1
                pct = min(frame_idx / total * 100, 100.0)
                print(f"[{pct:5.1f}%] {frame_idx}/{meta['frame_count']}  {source.name}", flush=True)
    finally:
        cap.release()
        writer.release()

    print(f"帧号视频已保存: {output}")
    return output


def overlay_reference_video_opencv(
    source: Path,
    output: Path | None = None,
    max_frames: int | None = None,
    duration_s: float | None = DEFAULT_DURATION_S,
    width: int | None = DEFAULT_WIDTH,
) -> Path:
    """逐帧叠加断面线、车道线、帧号，写出人工标注参考视频。"""
    source = Path(source)
    output = output or default_reference_output_path(source)
    output.parent.mkdir(parents=True, exist_ok=True)

    entrance = infer_entrance(source)
    lines = load_section_lines(entrance)
    lanes = load_lanes(entrance)
    stop_lines = load_stop_lines(entrance)

    cap = open_video(source)
    meta = video_meta(cap)
    fps = meta["fps"]
    out_width = width or meta["width"]
    scale = out_width / meta["width"]
    out_height = int(round(meta["height"] * scale / 2) * 2)
    writer = make_writer(output, fps, out_width, out_height)
    max_by_duration = int(duration_s * fps) if duration_s is not None and fps > 0 else None
    frame_limit = min(v for v in [max_frames, max_by_duration] if v is not None) if any(
        v is not None for v in [max_frames, max_by_duration]
    ) else None

    scaled_lines = [
        (name, int(lx1 * scale), int(ly1 * scale), int(lx2 * scale), int(ly2 * scale), dir_pos, dir_neg)
        for name, lx1, ly1, lx2, ly2, dir_pos, dir_neg in lines
    ]
    scaled_lanes = {
        lane_id: [(int(x * scale), int(y * scale)) for x, y in pts]
        for lane_id, pts in lanes.items()
    }
    scaled_stop_lines = [
        (name, int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale))
        for name, x1, y1, x2, y2 in stop_lines
    ]

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_limit is not None and frame_idx >= frame_limit:
                break
            if scale != 1.0:
                frame = cv2.resize(frame, (out_width, out_height))
            draw_reference_overlay(frame, frame_idx, fps, scaled_lines, scaled_lanes, scaled_stop_lines)
            writer.write(frame)
            frame_idx += 1
            if frame_idx % 250 == 0:
                print(f"[参考视频] {frame_idx}帧  {source.name}", flush=True)
    finally:
        cap.release()
        writer.release()

    print(f"标注参考视频已保存: {output}")
    return output


def overlay_video_ffmpeg(
    source: Path,
    output: Path | None = None,
    font_size: int = 52,
    duration_s: float | None = DEFAULT_DURATION_S,
    width: int | None = DEFAULT_WIDTH,
) -> Path:
    """使用 ffmpeg drawtext 快速叠加帧号。"""
    source = Path(source)
    output = output or default_output_path(source)
    output.parent.mkdir(parents=True, exist_ok=True)

    font = "/System/Library/Fonts/Supplemental/Arial.ttf"
    drawtext = (
        "drawtext="
        f"fontfile='{font}':"
        "text='Frame %{n}   Time %{pts\\:hms}':"
        "x=24:y=24:"
        f"fontsize={font_size}:"
        "fontcolor=yellow:"
        "box=1:boxcolor=black@0.62:boxborderw=14"
    )
    filters = [drawtext]
    if width:
        filters.append(f"scale={width}:-2")
    filter_chain = ",".join(filters)
    base_cmd = [
        "ffmpeg",
        "-y",
    ]
    if duration_s is not None:
        base_cmd.extend(["-t", f"{duration_s:g}"])
    base_cmd.extend([
        "-i", str(source),
        "-vf", filter_chain,
        "-an",
    ])
    hw_cmd = [
        *base_cmd,
        "-c:v", "h264_videotoolbox",
        "-b:v", "12M",
        "-pix_fmt", "yuv420p",
        str(output),
    ]
    cpu_cmd = [
        *base_cmd,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        str(output),
    ]
    try:
        subprocess.run(hw_cmd, check=True)
    except subprocess.CalledProcessError:
        subprocess.run(cpu_cmd, check=True)
    print(f"帧号视频已保存: {output}")
    return output


def _discover_videos() -> list[Path]:
    candidates = [
        OUTPUT_DIR / "trajectory.mp4",
        DATA_DIR / "test_video_fixed.mp4",
        *DATA_DIR.glob("*进口*.mp4"),
    ]
    seen: set[Path] = set()
    videos: list[Path] = []
    for path in candidates:
        if path.exists() and path not in seen:
            videos.append(path)
            seen.add(path)
    return videos


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="给人工标注视频叠加帧号。")
    parser.add_argument("videos", nargs="*", type=Path, help="要处理的视频；为空时自动处理 trajectory.mp4 和各进口视频")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="批量输出目录")
    parser.add_argument("--output", type=Path, help="单个视频时指定完整输出路径")
    parser.add_argument("--max-frames", type=int, help="只处理前 N 帧，用于快速预览")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S, help="输出前 N 秒；默认 300 秒")
    parser.add_argument("--full", action="store_true", help="输出完整视频，不截取 5 分钟")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="输出宽度，默认 1920；设为 0 保持原分辨率")
    parser.add_argument("--opencv", action="store_true", help="强制使用 OpenCV 逐帧写出；默认优先使用 ffmpeg")
    parser.add_argument("--reference", action="store_true", help="叠加断面线和车道线，生成标注参考视频")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    videos = args.videos or _discover_videos()
    if not videos:
        raise SystemExit("未找到可处理视频，请显式传入视频路径。")
    if args.output and len(videos) != 1:
        raise SystemExit("--output 只能在处理单个视频时使用。")

    for video in videos:
        output = args.output or (
            default_reference_output_path(video, args.output_dir)
            if args.reference
            else default_output_path(video, args.output_dir)
        )
        if args.reference:
            overlay_reference_video_opencv(
                video,
                output=output,
                max_frames=args.max_frames,
                duration_s=None if args.full else args.duration,
                width=args.width or None,
            )
            continue
        if not args.opencv and args.max_frames is None and shutil.which("ffmpeg"):
            overlay_video_ffmpeg(
                video,
                output=output,
                duration_s=None if args.full else args.duration,
                width=args.width or None,
            )
        else:
            overlay_video(video, output=output, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
