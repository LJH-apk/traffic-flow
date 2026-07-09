"""三进口批量跟踪入口。"""
from __future__ import annotations

import shutil
from pathlib import Path

from src.config.settings import DATA_DIR, OUTPUT_DIR
from src.trajectory.tracker import TrajectoryTracker

ENTRANCE_RUNS = [
    ("北进口", "北进口_20260420075959至20260420081500.mp4", "north"),
    ("东进口", "东进口_20260420075958至20260420081459.mp4", "east"),
    ("南进口", "南进口_20260420075959至20260420081500.mp4", "south"),
]


def run_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for index, (entrance, filename, suffix) in enumerate(ENTRANCE_RUNS, start=1):
        print(f"\n{'='*60}")
        print(f"[{index}/3] {entrance} ({filename})")
        print(f"{'='*60}")
        lock_path = OUTPUT_DIR / ".tracker.lock"
        if lock_path.exists():
            lock_path.unlink()

        tracker = TrajectoryTracker()
        tracker.run(video_path=DATA_DIR / filename, start_frame=0, end_frame=None)

        # 保存各进口独立输出，避免相互覆盖
        _copy_if_exists(OUTPUT_DIR / "cross_section.csv", OUTPUT_DIR / f"cross_section_{suffix}_full.csv")
        _copy_if_exists(OUTPUT_DIR / "trajectory.csv", OUTPUT_DIR / f"trajectory_{suffix}_full.csv")
        _copy_if_exists(OUTPUT_DIR / "vehicle_stats.csv", OUTPUT_DIR / f"vehicle_stats_{suffix}_full.csv")
        _copy_if_exists(OUTPUT_DIR / "trajectory_groups.csv", OUTPUT_DIR / f"trajectory_groups_{suffix}_full.csv")
        _copy_if_exists(OUTPUT_DIR / "trajectory.mp4", OUTPUT_DIR / f"trajectory_{suffix}_full.mp4")
        print(f"[{index}/3] {entrance} 完成\n")

    print("=" * 60)
    print("全部完成！输出文件：")
    for path in sorted(OUTPUT_DIR.glob("*_full.*")):
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {path.name}  ({size_mb:.1f} MB)")
    print("=" * 60)


def _copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        shutil.copyfile(source, target)


if __name__ == "__main__":
    run_all()
