"""项目主入口。"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config.settings import DATA_DIR, VIDEO_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="交通流检测项目主入口")
    parser.add_argument(
        "command",
        nargs="?",
        choices=[
            "detect",
            "track",
            "run-all",
            "build-dashboard",
            "dashboard",
            "agent",
            "eval-video",
            "eval-coco",
        ],
        help="不传 command 时进入交互菜单",
    )
    args = parser.parse_args()

    command = args.command or _choose_command()
    _run_command(command)


def _choose_command() -> str:
    items = [
        ("detect", "单视频检测"),
        ("track", "轨迹跟踪"),
        ("run-all", "三进口批量跟踪"),
        ("build-dashboard", "构建仪表盘数据"),
        ("dashboard", "启动仪表盘"),
        ("agent", "AI 分析智能体"),
        ("eval-video", "伪 GT 评测"),
        ("eval-coco", "COCO 评测"),
    ]
    print("\n交通流检测项目")
    for index, (_, label) in enumerate(items, start=1):
        print(f"{index}. {label}")
    while True:
        raw = input("请选择功能编号: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(items):
            return items[int(raw) - 1][0]
        print("输入无效，请重新选择。")


def _run_command(command: str) -> None:
    if command == "detect":
        from src.detection.detector import VehicleDetector

        VehicleDetector().run(video_path=VIDEO_PATH)
        return

    if command == "track":
        from src.trajectory.tracker import TrajectoryTracker

        video_path = _choose_video()
        TrajectoryTracker().run(video_path=video_path)
        return

    if command == "run-all":
        from src.trajectory.run_all_entrances import run_all

        run_all()
        return

    if command == "build-dashboard":
        from src.dashboard.build_data import main as build_dashboard

        build_dashboard()
        return

    if command == "dashboard":
        _ensure_dashboard_data()
        from src.dashboard.server import run

        run()
        return

    if command == "agent":
        from src.agent.app import main as agent_main

        agent_main()
        return

    if command == "eval-video":
        from src.evaluation.eval_on_video import main as eval_video

        eval_video()
        return

    if command == "eval-coco":
        from src.evaluation.eval_coco import run_val as eval_coco

        eval_coco()
        return

    raise ValueError(f"未知命令: {command}")


def _choose_video() -> Path:
    videos = [
        DATA_DIR / "北进口_20260420075959至20260420081500.mp4",
        DATA_DIR / "东进口_20260420075958至20260420081459.mp4",
        DATA_DIR / "南进口_20260420075959至20260420081500.mp4",
        VIDEO_PATH,
    ]
    existing = [path for path in videos if path.exists()]
    if not existing:
        return VIDEO_PATH

    print("\n可用视频")
    for index, path in enumerate(existing, start=1):
        print(f"{index}. {path.name}")
    raw = input("请选择视频编号，直接回车使用第 1 个: ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(existing):
        return existing[int(raw) - 1]
    return existing[0]


def _ensure_dashboard_data() -> None:
    from src.config.settings import OUTPUT_DIR

    meta_path = OUTPUT_DIR / "dashboard" / "meta.json"
    if meta_path.exists():
        print("[INFO] 仪表盘数据已存在，跳过构建")
        return
    print("[INFO] 仪表盘数据尚未构建，正在自动生成")
    from src.dashboard.build_data import main as build_dashboard

    build_dashboard()


if __name__ == "__main__":
    main()
