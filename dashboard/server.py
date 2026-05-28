#!/usr/bin/env python3
"""交通流仪表盘 HTTP 服务器（零依赖，内置 http.server）"""
import csv
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = 8765


class Handler(BaseHTTPRequestHandler):
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
        elif path == "/outputs/trajectory.mp4":
            self._serve_video(os.path.join(OUTPUTS_DIR, "trajectory.mp4"))
        elif path == "/outputs/trajectory_map.png":
            self._serve_file(os.path.join(OUTPUTS_DIR, "trajectory_map.png"), "image/png")
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
    server = HTTPServer(("localhost", PORT), Handler)
    url = f"http://localhost:{PORT}"
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"  仪表盘已启动: {url}")
    print("  按 Ctrl+C 停止服务器")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  服务器已停止")


if __name__ == "__main__":
    run()
