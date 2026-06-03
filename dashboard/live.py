"""Live preview state shared by the dashboard server and tracker."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Iterator

import cv2


class LiveState:
    """Thread-safe latest-frame and stats store for live dashboard preview."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._frame_ready = threading.Condition(self._lock)
        self._latest_frame: bytes | None = None
        self._frame_seq = 0
        self.stop_event = threading.Event()
        self._status = {
            "running": False,
            "stop_requested": False,
            "video": "",
            "started_at": None,
            "finished_at": None,
            "error": "",
            "total_frames": 0,
        }
        self._progress = {
            "frame_idx": 0,
            "timestamp_s": 0.0,
            "fps": 0.0,
            "percent": 0.0,
        }
        self._stats = self._empty_stats()

    @staticmethod
    def _empty_stats() -> dict:
        return {
            "vehicles": 0,
            "events": 0,
            "avg_speed": 0.0,
            "active_tracks": 0,
            "recent_events": [],
            "class_counts": {},
            "lane_counts": {},
        }

    def start(self, video: str | Path, total_frames: int = 0) -> None:
        with self._frame_ready:
            self.stop_event.clear()
            self._latest_frame = None
            self._frame_seq = 0
            self._status.update({
                "running": True,
                "stop_requested": False,
                "video": str(video),
                "started_at": time.time(),
                "finished_at": None,
                "error": "",
                "total_frames": int(total_frames or 0),
            })
            self._progress.update({
                "frame_idx": 0,
                "timestamp_s": 0.0,
                "fps": 0.0,
                "percent": 0.0,
            })
            self._stats = self._empty_stats()
            self._frame_ready.notify_all()

    def request_stop(self) -> None:
        with self._frame_ready:
            self.stop_event.set()
            self._status["stop_requested"] = True
            self._frame_ready.notify_all()

    def finish(self) -> None:
        with self._frame_ready:
            self._status["running"] = False
            self._status["finished_at"] = time.time()
            self._frame_ready.notify_all()

    def fail(self, message: str) -> None:
        with self._frame_ready:
            self._status["running"] = False
            self._status["error"] = message
            self._status["finished_at"] = time.time()
            self._frame_ready.notify_all()

    def update_progress(self, frame_idx: int, timestamp_s: float, fps: float) -> None:
        with self._lock:
            total = int(self._status.get("total_frames") or 0)
            percent = (frame_idx / total * 100.0) if total > 0 else 0.0
            self._progress.update({
                "frame_idx": int(frame_idx),
                "timestamp_s": round(float(timestamp_s), 3),
                "fps": round(float(fps), 1),
                "percent": round(max(0.0, min(100.0, percent)), 1),
            })

    def publish_stats(self, stats: dict) -> None:
        with self._lock:
            merged = self._empty_stats()
            merged.update(stats)
            self._stats = merged

    def publish_frame(
        self,
        frame,
        width: int = 960,
        quality: int = 72,
    ) -> bool:
        if frame is None:
            return False

        h, w = frame.shape[:2]
        if width > 0 and w > width:
            scale = width / w
            height = max(1, int(h * scale))
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        if not ok:
            return False

        with self._frame_ready:
            self._latest_frame = encoded.tobytes()
            self._frame_seq += 1
            self._frame_ready.notify_all()
        return True

    def latest_frame(self) -> bytes | None:
        with self._lock:
            return self._latest_frame

    def snapshot(self) -> dict:
        with self._lock:
            return {
                **self._status.copy(),
                "progress": self._progress.copy(),
                "stats": {
                    **self._stats,
                    "recent_events": list(self._stats.get("recent_events", [])),
                    "class_counts": dict(self._stats.get("class_counts", {})),
                    "lane_counts": dict(self._stats.get("lane_counts", {})),
                },
            }

    def mjpeg_chunks(self, timeout: float = 10.0) -> Iterator[bytes]:
        last_seq = -1
        while True:
            with self._frame_ready:
                ready = self._frame_ready.wait_for(
                    lambda: self._frame_seq != last_seq and self._latest_frame is not None,
                    timeout=timeout,
                )
                if not ready:
                    continue
                last_seq = self._frame_seq
                frame = self._latest_frame
            if frame is None:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                + frame
                + b"\r\n"
            )
