"""
AI 交通分析智能体桌面应用入口：后台线程跑 HTTP 服务 + pywebview 原生窗口。

pywebview 未安装或窗口创建失败时，自动退化为浏览器模式（与仪表盘行为一致），
保证比赛演示现场任何情况下都能打开界面。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))


def main() -> None:
    try:
        import webview
    except ImportError:
        print("[INFO] 未安装 pywebview，以浏览器模式启动（pip install pywebview 可启用桌面窗口）")
        from src.agent.server import run

        run()
        return

    from src.agent.server import start_in_thread

    server, url = start_in_thread()
    print(f"  AI 交通分析智能体（桌面模式）: {url}")
    try:
        webview.create_window(
            "交通流 AI 分析决策系统",
            url,
            width=1440,
            height=900,
            min_size=(1080, 700),
            background_color="#05070d",
        )
        webview.start()
    except Exception as exc:  # noqa: BLE001 — 窗口创建失败退化为浏览器
        print(f"[WARN] 桌面窗口启动失败（{exc}），改用浏览器打开")
        import webbrowser

        webbrowser.open(url)
        try:
            while True:
                import time

                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
