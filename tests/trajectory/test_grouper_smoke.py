"""
独立冒烟测试：读取现有 trajectory.csv，模拟 TrajGrouper 工作流程。

运行：
    python3 tests/trajectory/test_grouper_smoke.py
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import csv
from src.trajectory.traj_grouper import TrajGrouper


def main():
    traj_csv = Path("outputs/trajectory.csv")
    if not traj_csv.exists():
        print(f"[smoke] 跳过：{traj_csv} 不存在")
        return

    # 从 CSV 重建每个 track_id 的坐标序列
    tracks: dict[int, list[tuple[float, float]]] = defaultdict(list)
    track_cls: dict[int, str] = {}
    track_lane: dict[int, int | None] = {}
    track_lane_type: dict[int, str] = {}
    with traj_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = int(row["track_id"])
            tracks[tid].append((float(row["cx"]), float(row["cy"])))
            track_cls[tid] = row["class_name"]
            lid = row.get("lane_id")
            track_lane[tid] = int(lid) if lid else None
            track_lane_type[tid] = row.get("lane_type", "")

    out_csv = Path("outputs/trajectory_groups_smoke.csv")
    grouper = TrajGrouper(csv_path=out_csv, excel_path=None)

    entrance = "北进口"
    for tid, pts in tracks.items():
        grouper.on_track_end(
            tid, pts,
            track_cls.get(tid, "car"),
            track_lane.get(tid),
            track_lane_type.get(tid, ""),
            entrance,
        )

    grouper.finalize()

    records = grouper.get_records()
    print(f"[smoke] 轨迹总数: {len(tracks)}")
    print(f"[smoke] 分组数: {len(records)}")
    sizes = [r["size"] for r in records]
    if sizes:
        print(f"[smoke] 最大组: {max(sizes)}辆  平均: {sum(sizes)/len(sizes):.1f}辆")

    turns = defaultdict(int)
    for r in records:
        turns[r["turn_type"]] += r["size"]
    print(f"[smoke] 转向分布: {dict(turns)}")
    print(f"[smoke] 输出: {out_csv}")


if __name__ == "__main__":
    main()
