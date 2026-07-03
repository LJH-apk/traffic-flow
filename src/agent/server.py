#!/usr/bin/env python3
"""AI 交通分析智能体 HTTP 服务（零依赖 http.server，与 dashboard/server.py 同风格）。

端点：
  GET  /                          前端页面
  GET  /assets/*                  静态资源
  GET  /api/agent/health          key/数据/模型状态
  GET  /api/agent/summary         侧栏 KPI 数据
  GET  /api/agent/quick_questions 快捷问题列表
  GET  /api/agent/tools           已接入的分析工具清单（前端 /tools 内部命令用）
  GET  /api/agent/report/download 下载最近一次报告（.md）
  POST /api/agent/chat            {message} → SSE 事件流
  POST /api/agent/report          → SSE 事件流（生成完整报告）
  POST /api/agent/settings        {model, thinking_strength} → 更新模型设置
  POST /api/agent/reset           清空会话
  POST /api/agent/reload          重新加载 outputs 数据
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.config.settings import (
    AGENT_HOST,
    AGENT_PORT,
    AGENT_REPORT_PATH,
    DEEPSEEK_MODEL,
    DEEPSEEK_MODELS,
)
from src.agent.analyst import TrafficAnalyst
from src.agent.llm import DeepSeekClient
from src.agent.tools import TOOL_DEFINITIONS, TOOL_LABELS_ZH

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


class AgentApp:
    """持有唯一的 analyst 实例；忙锁保证同一时刻只跑一个生成任务。"""

    thinking_options = [
        {"value": "low", "label": "低", "description": "响应更快，适合简单问答"},
        {"value": "medium", "label": "中", "description": "默认平衡档"},
        {"value": "high", "label": "高", "description": "更充分推理，适合报告和复杂诊断"},
        {"value": "ultra", "label": "超高", "description": "最高推理投入，适合复杂论证"},
    ]

    def __init__(self, analyst: TrafficAnalyst | None = None):
        self.analyst = analyst or TrafficAnalyst()
        self.busy = threading.Lock()
        self.available_models = tuple(DEEPSEEK_MODELS) or (DEEPSEEK_MODEL,)

    def health(self) -> dict:
        store = self.analyst.executor.store
        return {
            "model": self.analyst.client.model,
            "available_models": list(self.available_models),
            "thinking_strength": self.analyst.client.thinking_strength,
            "thinking_options": self.thinking_options,
            "context": self.analyst.context_status(),
            "key_configured": self.analyst.client.available,
            "data_ready": store.data_ready,
            "report_exists": AGENT_REPORT_PATH.exists(),
        }

    def update_settings(self, model: str | None, thinking_strength: str | None) -> dict:
        if self.busy.locked():
            return {"ok": False, "error": "智能体正在处理请求，完成后再切换模型。"}
        model = (model or self.analyst.client.model).strip()
        if model not in self.available_models:
            return {"ok": False, "error": f"不支持的模型: {model}"}
        strength = DeepSeekClient._normalize_thinking_strength(
            thinking_strength or self.analyst.client.thinking_strength
        )
        changed = (
            model != self.analyst.client.model
            or strength != self.analyst.client.thinking_strength
        )
        self.analyst.client.configure(model=model, thinking_strength=strength)
        if changed:
            self.analyst.reset()
        return {"ok": True, "changed": changed, "settings": self.health()}


APP = AgentApp()


class Handler(BaseHTTPRequestHandler):
    app = APP

    def log_message(self, fmt, *args):
        pass  # 静默日志

    # ── GET ──────────────────────────────────────────────────────────────

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._serve_file(os.path.join(STATIC_DIR, "index.html"))
        elif path.startswith("/assets/"):
            asset = os.path.normpath(os.path.join(STATIC_DIR, path.lstrip("/")))
            if asset.startswith(STATIC_DIR) and os.path.isfile(asset):
                self._serve_file(asset)
            else:
                self.send_error(404)
        elif path == "/api/agent/health":
            self._json(self.app.health())
        elif path == "/api/agent/summary":
            self._json(self.app.analyst.executor.store.summary_for_ui())
        elif path == "/api/agent/quick_questions":
            self._json(self.app.analyst.quick_questions)
        elif path == "/api/agent/tools":
            self._json(self._tool_list())
        elif path == "/api/agent/report/download":
            self._serve_report()
        else:
            self.send_error(404)

    # ── POST ─────────────────────────────────────────────────────────────

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/api/agent/chat":
            body = self._read_json()
            message = (body.get("message") or "").strip()
            if not message:
                self._json({"error": "message 不能为空"}, status=400)
                return
            self._run_stream(lambda: self.app.analyst.chat_events(message))
        elif path == "/api/agent/report":
            self._run_stream(self.app.analyst.report_events)
        elif path == "/api/agent/settings":
            body = self._read_json()
            result = self.app.update_settings(body.get("model"), body.get("thinking_strength"))
            self._json(result, status=200 if result.get("ok") else 400)
        elif path == "/api/agent/reset":
            self.app.analyst.reset()
            self._json({"ok": True})
        elif path == "/api/agent/reload":
            self.app.analyst.executor.store.reload()
            self._json({"ok": True,
                        "summary": self.app.analyst.executor.store.summary_for_ui()})
        else:
            self.send_error(404)

    # ── 工具方法 ─────────────────────────────────────────────────────────

    @staticmethod
    def _tool_list() -> list[dict]:
        return [
            {"name": t["function"]["name"],
             "label": TOOL_LABELS_ZH.get(t["function"]["name"], t["function"]["name"]),
             "description": t["function"]["description"]}
            for t in TOOL_DEFINITIONS
        ]

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, OSError):
            return {}

    def _json(self, data, status=200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _run_stream(self, events_factory) -> None:
        """把事件生成器写成 SSE 流；忙时直接拒绝，避免并发会话串写历史。"""
        if not self.app.busy.acquire(blocking=False):
            self._json({"error": "智能体正在处理上一个请求，请稍候"}, status=409)
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()
            for ev in events_factory():
                payload = json.dumps(ev, ensure_ascii=False)
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # 客户端断开
        except Exception as exc:  # noqa: BLE001 — 兜底：把意外错误发给前端
            try:
                payload = json.dumps({"type": "error", "message": f"服务内部错误: {exc}"},
                                     ensure_ascii=False)
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
            except OSError:
                pass
        finally:
            self.app.busy.release()

    def _serve_file(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            self.send_error(404)
            return
        ext = os.path.splitext(filepath)[1].lower()
        with open(filepath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", _CONTENT_TYPES.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_report(self) -> None:
        if not AGENT_REPORT_PATH.exists():
            self.send_error(404, "报告尚未生成")
            return
        data = AGENT_REPORT_PATH.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/markdown; charset=utf-8")
        self.send_header("Content-Disposition",
                         'attachment; filename="ai_analysis_report.md"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def create_server(host: str = AGENT_HOST, port: int = AGENT_PORT) -> ThreadingHTTPServer:
    """建服务器；端口被占用时自动向后找 10 个端口。"""
    last_err: OSError | None = None
    for candidate in range(port, port + 10):
        try:
            return ThreadingHTTPServer((host, candidate), Handler)
        except OSError as exc:
            last_err = exc
    raise last_err  # type: ignore[misc]


def start_in_thread() -> tuple[ThreadingHTTPServer, str]:
    """后台线程启动，返回 (server, url)。桌面壳（app.py）使用。"""
    server = create_server()
    thread = threading.Thread(target=server.serve_forever,
                              name="agent-http", daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}"


def run() -> None:
    """浏览器模式入口：前台阻塞运行。"""
    import webbrowser

    server = create_server()
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}"
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"  AI 交通分析智能体已启动: {url}")
    print("  按 Ctrl+C 停止服务")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  服务已停止")


if __name__ == "__main__":
    run()
