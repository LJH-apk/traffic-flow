#!/usr/bin/env python3
"""交通流仪表盘 HTTP 服务器（零依赖，内置 http.server）"""
import csv
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from dashboard.live import LiveState
except ImportError:  # 支持 python dashboard/server.py 直接运行
    from live import LiveState

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = 8765
HOST = "127.0.0.1"
DEFAULT_VIDEO = "北进口_20260420075959至20260420081500.mp4"


class DashboardApp:
    """后台检测任务控制器。"""

    def __init__(self, state=None, tracker_factory=None, video_path=DEFAULT_VIDEO):
        self.state = state or LiveState()
        self._tracker_factory = tracker_factory or self._default_tracker_factory
        self.video_path = video_path
        self._lock = threading.Lock()
        self._thread = None

    @staticmethod
    def _default_tracker_factory():
        from src.trajectory.tracker import TrajectoryTracker

        return TrajectoryTracker()

    def start_detection(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return {
                    "started": False,
                    "message": "Detection is already running",
                    "status": self.state.snapshot(),
                }

            self.state.start(self.video_path, self._total_frames())
            self._thread = threading.Thread(
                target=self._run_detection,
                name="dashboard-detection",
                daemon=True,
            )
            self._thread.start()

        return {
            "started": True,
            "message": "Detection started",
            "status": self.state.snapshot(),
        }

    def _total_frames(self):
        try:
            import cv2

            path = self.video_path
            if not os.path.isabs(path):
                path = os.path.join(BASE_DIR, path)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return 0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return max(0, total)
        except Exception:  # noqa: BLE001
            return 0

    def _run_detection(self):
        try:
            tracker = self._tracker_factory()
            tracker.run(
                video_path=self.video_path,
                live_publisher=self.state,
                stop_event=self.state.stop_event,
            )
        except Exception as exc:  # noqa: BLE001
            self.state.fail(str(exc))
        else:
            self.state.finish()

    def stop_detection(self):
        self.state.request_stop()
        return {
            "stop_requested": True,
            "status": self.state.snapshot(),
        }

    def join(self, timeout=None):
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)


APP = DashboardApp()


class Handler(BaseHTTPRequestHandler):
    app = APP

    def log_message(self, fmt, *args):
        pass  # 静默日志

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/" or path == "/index.html":
            self._serve_file(os.path.join(DASHBOARD_DIR, "index.html"), "text/html; charset=utf-8")
        elif path == "/api/trajectory":
            self._serve_csv("trajectory.csv")
        elif path == "/api/cross_section":
            self._serve_csv("cross_section.csv")
        elif path == "/api/live/status":
            self._json_response(self.app.state.snapshot())
        elif path == "/api/live/stats":
            self._json_response(self.app.state.snapshot()["stats"])
        elif path == "/api/live/stream.mjpg":
            self._serve_mjpeg()
        elif path == "/outputs/trajectory.mp4":
            self._serve_video(os.path.join(OUTPUTS_DIR, "trajectory.mp4"))
        elif path == "/outputs/trajectory_map.png":
            self._serve_file(os.path.join(OUTPUTS_DIR, "trajectory_map.png"), "image/png")
        else:
            self.send_error(404)

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/api/detect/start":
            self._json_response(self.app.start_detection())
        elif path == "/api/detect/stop":
            self._json_response(self.app.stop_detection())
        else:
            self.send_error(404)

    def _serve_csv(self, filename):
        filepath = os.path.join(OUTPUTS_DIR, filename)
        if not os.path.exists(filepath):
            self._json_response([])
            return
        with open(filepath, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self._json_response(rows)

    def _json_response(self, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            for chunk in self.app.state.mjpeg_chunks():
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _serve_file(self, filepath, content_type):
        if not os.path.exists(filepath):
            self.send_error(404)
            return
        with open(filepath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_video(self, filepath):
        if not os.path.exists(filepath):
            self.send_error(404)
            return
        file_size = os.path.getsize(filepath)
        range_header = self.headers.get("Range")

        if range_header:
            # 解析 Range: bytes=start-end
            ranges = range_header.replace("bytes=", "").split("-")
            start = int(ranges[0]) if ranges[0] else 0
            end = int(ranges[1]) if ranges[1] else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1

            self.send_response(206)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            try:
                with open(filepath, "rb") as f:
                    f.seek(start)
                    remaining = length
                    chunk = 64 * 1024
                    while remaining > 0:
                        data = f.read(min(chunk, remaining))
                        if not data:
                            break
                        self.wfile.write(data)
                        remaining -= len(data)
            except OSError:
                pass
        else:
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            try:
                with open(filepath, "rb") as f:
                    while True:
                        data = f.read(64 * 1024)
                        if not data:
                            break
                        self.wfile.write(data)
            except OSError:
                pass


def run():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"  仪表盘已启动: {url}")
    print("  按 Ctrl+C 停止服务器")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  服务器已停止")


if __name__ == "__main__":
    run()
