#!/usr/bin/env python3
"""
将检测输出 CSV 预处理为前端展示用的静态 JSON。

用法：
    python3 -m src.dashboard.build_data

输入：outputs/ 下的 CSV 文件
输出：outputs/dashboard/ 下的 JSON 文件
    meta.json        — 元信息（视频参数、类别颜色、断面列表）
    overview.json    — 总览 KPI（累计车辆、事件数、分类计数等）
    timeline.json    — 每秒快照（演示模式时间同步用）
    charts.json      — 图表数据（饼图、直方图、柱状图）
    events.json      — 最近过车事件列表
    trajectories.json— 轨迹坐标（抽稀后，Canvas 绘制用）
    validation.json  — 精度校验摘要
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parents[2]
OUTPUTS = ROOT / "outputs"
DASHBOARD_OUT = OUTPUTS / "dashboard"

# 优先使用合并版数据，不存在则回退到分进口文件
TRAJECTORY_SOURCES = [
    OUTPUTS / "trajectory_merged.csv",
    OUTPUTS / "trajectory.csv",
]
CROSS_SECTION_SOURCES = [
    OUTPUTS / "cross_section_merged_with_lane.csv",
    OUTPUTS / "cross_section_merged.csv",
]
CROSS_SECTION_FALLBACKS = [
    OUTPUTS / "cross_section_north.csv",
    OUTPUTS / "cross_section_south.csv",
    OUTPUTS / "cross_section_east.csv",
]
VEHICLE_STATS_SOURCES = [
    OUTPUTS / "vehicle_stats_lane_fix.csv",
    OUTPUTS / "vehicle_stats.csv",
]
TRAJ_GROUPS_SOURCES = [
    OUTPUTS / "trajectory_groups_lane_fix.csv",
    OUTPUTS / "trajectory_groups.csv",
]
VALIDATION_SOURCES = [
    OUTPUTS / "validation_final" / "validation_summary.csv",
    OUTPUTS / "validation_latest" / "validation_summary.csv",
]

# ── 车型颜色（与 settings.py 对齐） ──────────────────────────────────
CLASS_COLORS: dict[str, str] = {
    "car":        "#00ff00",
    "truck":      "#0080ff",
    "bus":        "#ff8000",
    "motorcycle": "#ff00ff",
    "bicycle":    "#00ffff",
}

# ── 方向颜色 ─────────────────────────────────────────────────────────
DIRECTION_COLORS: dict[str, str] = {
    "到达": "#22d3ee",
    "离去": "#6366f1",
    "右转": "#f59e0b",
    "直行": "#10b981",
    "掉头": "#ef4444",
}

# ── 辅助函数 ──────────────────────────────────────────────────────────


def _find_first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _read_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _safe_int(v, default=0) -> int:
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default


def _safe_str(v, default="") -> str:
    if v is None:
        return default
    return str(v).strip()


def _pick(v, default):
    """返回非空值，否则默认值。"""
    if v is None or v == "":
        return default
    return v


# ── 主构建逻辑 ────────────────────────────────────────────────────────


def build_meta(traj_path: Path) -> dict:
    """生成 meta.json：视频参数、颜色、断面列表。"""
    traj_rows = _read_csv(traj_path)

    # 从 trajectory 推算视频参数
    total_frames = 0
    duration_s = 0.0
    if traj_rows:
        last = traj_rows[-1]
        total_frames = _safe_int(last.get("frame_id", 0)) + 1
        duration_s = _safe_float(last.get("timestamp_s", 0))

    # 收集断面列表（从多个来源合并）
    sections: list[dict] = []
    seen_sections: set[str] = set()

    cs_path = _find_first_existing(CROSS_SECTION_SOURCES)
    if cs_path:
        cs_rows = _read_csv(cs_path)
    else:
        cs_rows = []
        for fb in CROSS_SECTION_FALLBACKS:
            if fb.exists() and fb.stat().st_size > 0:
                cs_rows.extend(_read_csv(fb))

    for row in cs_rows:
        name = _safe_str(row.get("section", ""))
        if name and name not in seen_sections:
            seen_sections.add(name)
            sections.append({"name": name})

    try:
        from src.config.settings import SECTION_LINES_MAP
        sections_config = []
        for entrance, lines in SECTION_LINES_MAP.items():
            for line in lines:
                sections_config.append({
                    "name": line[0],
                    "entrance": entrance,
                })
        if sections_config:
            sections = sections_config
    except ImportError:
        pass

    return {
        "video": {
            "path": "/outputs/trajectory.mp4",
            "name": "trajectory.mp4",
            "duration_s": round(duration_s, 1),
            "fps": 25,
            "width": 3840,
            "height": 2160,
            "total_frames": total_frames,
        },
        "dataset": {
            "trajectory_rows": len(traj_rows),
            "cross_section_rows": len(cs_rows),
        },
        "generated_at": "",
        "class_colors": CLASS_COLORS,
        "direction_colors": DIRECTION_COLORS,
        "sections": sections,
    }


def build_timeline_and_events(
    traj_path: Path, cs_rows: list[dict]
) -> tuple[dict, list[dict]]:
    """
    生成 timeline.json 和 events.json。

    timeline: 每秒一个快照，包含累计/活跃指标。
    events: 全部过车事件（按时间排序）。
    """
    traj_rows = _read_csv(traj_path)

    # ── 按秒聚合 trajectory ──
    # 每秒：活跃 track 集合、新出现 track、车型分布
    sec_data: dict[int, dict] = defaultdict(lambda: {
        "active_tracks": set(),
        "class_counts": defaultdict(int),
        "lane_counts": defaultdict(int),
    })

    all_tracks_seen: set[int] = set()
    track_first_sec: dict[int, int] = {}  # track_id → 首次出现的秒

    for row in traj_rows:
        t = int(_safe_float(row.get("timestamp_s", 0)))
        tid = _safe_int(row.get("track_id", 0))
        cls = _safe_str(row.get("class_name", ""))
        lane = _safe_str(row.get("lane_id", "")) or "UNKNOWN"

        if tid == 0:
            continue

        sd = sec_data[t]
        sd["active_tracks"].add(tid)
        sd["class_counts"][cls] += 1
        sd["lane_counts"][lane] += 1

        if tid not in all_tracks_seen:
            all_tracks_seen.add(tid)
            track_first_sec[tid] = t

    # 按秒计算累计车辆数
    max_sec = max(sec_data.keys()) if sec_data else 0
    cumulative = 0
    cum_by_sec: dict[int, int] = {}

    for t in range(max_sec + 1):
        for tid, first_t in track_first_sec.items():
            if first_t == t:
                cumulative += 1
        cum_by_sec[t] = cumulative

    # ── 按秒聚合 cross_section ──
    cs_by_sec: dict[int, list[dict]] = defaultdict(list)
    for row in cs_rows:
        t = int(_safe_float(row.get("timestamp_s", 0)))
        cs_by_sec[t].append(row)

    cum_events = 0
    cs_cum_by_sec: dict[int, int] = {}
    section_cum: dict[str, int] = defaultdict(int)

    for t in range(max_sec + 1):
        for ev in cs_by_sec.get(t, []):
            cum_events += 1
            sec_name = _safe_str(ev.get("section", ""))
            if sec_name:
                section_cum[sec_name] += 1
        cs_cum_by_sec[t] = cum_events

    # ── 构建 timeline snapshots ──
    snapshots = []
    for t in range(max_sec + 1):
        sd = sec_data.get(t, {
            "active_tracks": set(),
            "class_counts": {},
            "lane_counts": {},
        })

        # 计算活跃均速
        speeds = []
        for row in traj_rows:
            rt = int(_safe_float(row.get("timestamp_s", 0)))
            if rt == t:
                spd = _safe_float(row.get("speed_kmh", 0))
                if spd > 0:
                    speeds.append(spd)

        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

        # 最近过车事件（本秒内）
        recent = cs_by_sec.get(t, [])
        recent_events = []
        for ev in recent[:5]:  # 最多取5条
            recent_events.append({
                "timestamp_s": _safe_float(ev.get("timestamp_s", 0)),
                "section": _safe_str(ev.get("section", "")),
                "track_id": _safe_int(ev.get("track_id", 0)),
                "class_name": _safe_str(ev.get("class_name", "")),
                "lane_id": _safe_str(ev.get("lane_id", "")) or "UNKNOWN",
                "color": _safe_str(ev.get("color", "")),
                "direction": _safe_str(ev.get("direction", "")),
                "speed_kmh": _safe_float(ev.get("speed_kmh", 0)),
            })

        snapshot = {
            "t": t,
            "frame_id": t * 25,  # 估算 frame_id（25fps）
            "active_tracks": len(sd["active_tracks"]),
            "cumulative_vehicles": cum_by_sec.get(t, 0),
            "cumulative_events": cs_cum_by_sec.get(t, 0),
            "avg_speed_kmh": round(avg_speed, 1),
            "class_counts_active": dict(sd["class_counts"]),
            "lane_counts_active": dict(sd["lane_counts"]),
            "section_counts_cumulative": dict(section_cum),
            "recent_events": recent_events,
        }
        snapshots.append(snapshot)

    timeline = {
        "bucket_s": 1,
        "total_seconds": max_sec + 1,
        "snapshots": snapshots,
    }

    # ── 构建 events 列表 ──
    all_events = []
    for row in cs_rows:
        all_events.append({
            "frame_id": _safe_int(row.get("frame_id", 0)),
            "timestamp_s": _safe_float(row.get("timestamp_s", 0)),
            "section": _safe_str(row.get("section", "")),
            "arrival_departure": _safe_str(row.get("arrival_departure", "")),
            "track_id": _safe_int(row.get("track_id", 0)),
            "plate": _safe_str(row.get("plate", "")),
            "class_name": _safe_str(row.get("class_name", "")),
            "lane_id": _safe_str(row.get("lane_id", "")) or "UNKNOWN",
            "vehicle_category": _safe_str(row.get("vehicle_category", "")),
            "color": _safe_str(row.get("color", "")),
            "direction": _safe_str(row.get("direction", "")),
            "speed_kmh": _safe_float(row.get("speed_kmh", 0)),
            "headway_s": _safe_float(row.get("headway_s", 0)),
            "spacing_m": _safe_float(row.get("spacing_m", 0)),
        })

    # 按时间排序
    all_events.sort(key=lambda e: e["timestamp_s"])

    return timeline, all_events


def build_overview(traj_rows: list[dict], events: list[dict],
                   vehicle_stats_rows: list[dict],
                   traj_groups_rows: list[dict]) -> dict:
    """生成 overview.json：总览 KPI。"""
    # 唯一车辆数
    unique_tracks: set[int] = set()
    class_counts: dict[str, int] = defaultdict(int)
    lane_counts: dict[str, int] = defaultdict(int)

    for row in traj_rows:
        tid = _safe_int(row.get("track_id", 0))
        cls = _safe_str(row.get("class_name", ""))
        lane = _safe_str(row.get("lane_id", "")) or "UNKNOWN"
        if tid:
            unique_tracks.add(tid)
        if cls:
            class_counts[cls] += 1
        lane_counts[lane] += 1

    # 归一化 class_counts（每个 track 只计一次）
    track_class: dict[int, str] = {}
    for row in traj_rows:
        tid = _safe_int(row.get("track_id", 0))
        cls = _safe_str(row.get("class_name", ""))
        if tid and tid not in track_class and cls:
            track_class[tid] = cls

    class_unique_counts: dict[str, int] = defaultdict(int)
    for cls in track_class.values():
        class_unique_counts[cls] += 1

    # 断面统计
    section_counts: dict[str, int] = defaultdict(int)
    direction_counts: dict[str, int] = defaultdict(int)
    total_speed = 0.0
    speed_count = 0

    for ev in events:
        sec = _safe_str(ev.get("section", ""))
        direc = _safe_str(ev.get("direction", ""))
        spd = _safe_float(ev.get("speed_kmh", 0))
        if sec:
            section_counts[sec] += 1
        if direc:
            direction_counts[direc] += 1
        if spd > 0:
            total_speed += spd
            speed_count += 1

    avg_speed = round(total_speed / speed_count, 1) if speed_count > 0 else 0.0

    # 最高速度
    max_speed = 0.0
    for row in vehicle_stats_rows:
        ms = _safe_float(row.get("max_speed_kmh", 0))
        if ms > max_speed:
            max_speed = ms

    # 时间范围
    first_ts = 0.0
    last_ts = 0.0
    if traj_rows:
        first_ts = _safe_float(traj_rows[0].get("timestamp_s", 0))
        last_ts = _safe_float(traj_rows[-1].get("timestamp_s", 0))

    return {
        "total_vehicles": len(unique_tracks),
        "unique_tracks": len(unique_tracks),
        "total_events": len(events),
        "avg_speed_kmh": avg_speed,
        "max_speed_kmh": round(max_speed, 1),
        "active_duration_s": round(last_ts - first_ts, 1),
        "time_range": {
            "start_s": round(first_ts, 1),
            "end_s": round(last_ts, 1),
        },
        "class_counts": dict(class_unique_counts),
        "section_counts": dict(section_counts),
        "lane_counts": dict(lane_counts),
        "direction_counts": dict(direction_counts),
        "trajectory_groups_count": len(traj_groups_rows),
    }


def build_charts(events: list[dict], vehicle_stats_rows: list[dict],
                 traj_groups_rows: list[dict]) -> dict:
    """生成 charts.json：饼图、直方图、柱状图等图表所需数据。"""
    # ── 车型分布饼图 ──
    class_pie: dict[str, int] = defaultdict(int)
    for ev in events:
        cls = _safe_str(ev.get("class_name", ""))
        if cls:
            class_pie[cls] += 1
    class_pie_data = [
        {"name": k, "value": v}
        for k, v in sorted(class_pie.items(), key=lambda x: -x[1])
    ]

    # ── 速度分布直方图 ──
    speeds = []
    for row in vehicle_stats_rows:
        spd = _safe_float(row.get("avg_speed_kmh", 0))
        if spd > 0:
            speeds.append(spd)

    bins = [0, 10, 20, 30, 40, 50, 60, 80, 100]
    bin_labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80+"]
    hist = [0] * len(bin_labels)
    for s in speeds:
        for i in range(len(bins) - 1):
            if bins[i] <= s < bins[i + 1]:
                hist[i] += 1
                break
        else:
            if s >= bins[-1]:
                hist[-1] += 1

    speed_histogram = {
        "bins": bin_labels,
        "counts": hist,
    }

    # ── 断面流量对比柱状图 ──
    section_names: list[str] = []
    section_direction: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ev in events:
        sec = _safe_str(ev.get("section", ""))
        direc = _safe_str(ev.get("direction", ""))
        if sec and direc:
            section_direction[sec][direc] += 1

    # 按断面名排序
    section_names = sorted(section_direction.keys())
    all_directions: set[str] = set()
    for sd in section_direction.values():
        all_directions.update(sd.keys())
    all_directions_sorted = sorted(all_directions)

    section_bar = {
        "categories": section_names,
        "series": [
            {
                "name": d,
                "data": [section_direction[s].get(d, 0) for s in section_names],
            }
            for d in all_directions_sorted
        ],
    }

    # ── 车道流量 ──
    lane_counts: dict[str, int] = defaultdict(int)
    for ev in events:
        lane = _safe_str(ev.get("lane_id", "")) or "UNKNOWN"
        lane_counts[lane] += 1
    lanes_sorted = sorted(lane_counts.keys())
    lane_heatmap = {
        "lanes": lanes_sorted,
        "counts": [lane_counts[l] for l in lanes_sorted],
    }

    # ── 转向分组统计 ──
    turn_counts: dict[str, int] = defaultdict(int)
    for row in traj_groups_rows:
        tt = _safe_str(row.get("turn_type", ""))
        if tt:
            turn_counts[tt] += 1
    turn_groups = {
        "categories": sorted(turn_counts.keys()),
        "counts": [turn_counts[k] for k in sorted(turn_counts.keys())],
    }

    return {
        "class_pie": class_pie_data,
        "speed_histogram": speed_histogram,
        "section_bar": section_bar,
        "lane_heatmap": lane_heatmap,
        "turn_groups": turn_groups,
    }


def build_trajectories(traj_rows: list[dict],
                       traj_groups_rows: list[dict]) -> dict:
    """生成 trajectories.json：轨迹坐标（抽稀后）。"""
    # 按 track_id 分组
    track_points: dict[int, list[dict]] = defaultdict(list)
    for row in traj_rows:
        tid = _safe_int(row.get("track_id", 0))
        if not tid:
            continue
        track_points[tid].append({
            "t": _safe_float(row.get("timestamp_s", 0)),
            "x": _safe_int(row.get("cx", 0)),
            "y": _safe_int(row.get("cy", 0)),
        })

    # 获取每个 track 的 class_name
    track_info: dict[int, str] = {}
    for row in traj_rows:
        tid = _safe_int(row.get("track_id", 0))
        cls = _safe_str(row.get("class_name", ""))
        if tid and tid not in track_info and cls:
            track_info[tid] = cls

    # 抽稀：每5帧取1个点（25fps下约5点/秒，足够）
    tracks = []
    for tid, pts in track_points.items():
        if len(pts) < 3:
            continue
        # 每隔 STEP 取一个点
        STEP = 5
        sampled = pts[::STEP]
        # 确保首尾点保留
        if len(pts) > STEP and sampled[-1] != pts[-1]:
            sampled.append(pts[-1])

        tracks.append({
            "track_id": tid,
            "class_name": track_info.get(tid, "unknown"),
            "lane_id": "UNKNOWN",
            "point_count": len(sampled),
            "points": [[p["x"], p["y"]] for p in sampled],
        })

    # 分组数据
    groups = []
    for row in traj_groups_rows:
        tid_str = _safe_str(row.get("track_ids", ""))
        try:
            track_id_list = [int(x.strip()) for x in tid_str.split(",") if x.strip()]
        except ValueError:
            track_id_list = []

        groups.append({
            "group_id": _safe_str(row.get("group_id", "")),
            "entrance": _safe_str(row.get("entrance", "")),
            "turn_type": _safe_str(row.get("turn_type", "")),
            "class_type": _safe_str(row.get("class_type", "")),
            "track_ids": track_id_list,
            "size": _safe_int(row.get("size", 0)),
            "window_start_s": _safe_float(row.get("window_start_s", 0)),
            "window_end_s": _safe_float(row.get("window_end_s", 0)),
        })

    return {
        "canvas": {"width": 3840, "height": 2160},
        "tracks": tracks,
        "groups": groups,
    }


def build_validation() -> dict | None:
    """生成 validation.json：精度校验摘要。"""
    vs_path = _find_first_existing(VALIDATION_SOURCES)
    if not vs_path:
        return None

    rows = _read_csv(vs_path)
    raw = []
    summary: dict[str, float] = {}
    key_metrics = [
        "event_precision", "event_recall", "event_f1",
        "lane_accuracy", "direction_accuracy", "class_accuracy",
        "crossing_time_mae_s", "headway_mae_s", "headway_mape",
        "spacing_consistency_pass_rate",
        "physical_anomaly_count", "total_anomaly_count",
    ]

    for row in rows:
        metric = _safe_str(row.get("metric", ""))
        value = _safe_float(row.get("value", 0))
        raw.append({"metric": metric, "value": value})
        if metric in key_metrics:
            summary[metric] = value

    # 计算 F1（如果原始数据有 tp/fp/fn）
    tp = summary.get("event_tp", 0)
    fp = summary.get("event_fp", 0)
    fn = summary.get("event_fn", 0)
    if tp + fp + fn > 0:
        # 从 raw 中查找
        for r in raw:
            if r["metric"] == "event_tp":
                tp = r["value"]
            elif r["metric"] == "event_fp":
                fp = r["value"]
            elif r["metric"] == "event_fn":
                fn = r["value"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        summary["event_f1"] = round(f1, 4)
        if "event_precision" not in summary:
            summary["event_precision"] = round(precision, 4)
        if "event_recall" not in summary:
            summary["event_recall"] = round(recall, 4)

    return {
        "summary": summary,
        "raw": raw,
    }


# ── 入口 ───────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  交通流检测数据 → 前端静态 JSON 构建器")
    print("=" * 60)

    # 1. 查找数据源
    traj_path = _find_first_existing(TRAJECTORY_SOURCES)
    if not traj_path:
        print("[ERROR] 找不到 trajectory CSV 文件！")
        print(f"  已搜索: {[str(p) for p in TRAJECTORY_SOURCES]}")
        sys.exit(1)

    cs_path = _find_first_existing(CROSS_SECTION_SOURCES)
    if cs_path:
        cs_rows = _read_csv(cs_path)
        print(f"[OK]  断面数据: {cs_path.name} ({len(cs_rows)} 行)")
    else:
        cs_rows = []
        for fb in CROSS_SECTION_FALLBACKS:
            if fb.exists() and fb.stat().st_size > 0:
                rows = _read_csv(fb)
                cs_rows.extend(rows)
                print(f"[OK]  断面数据(合并): {fb.name} ({len(rows)} 行)")
        if not cs_rows:
            print("[WARN] 未找到任何断面数据，部分 JSON 将为空")

    vs_path = _find_first_existing(VEHICLE_STATS_SOURCES)
    vs_rows = _read_csv(vs_path) if vs_path else []
    print(f"[OK]  车辆统计: {vs_path.name if vs_path else 'N/A'} ({len(vs_rows)} 行)")

    tg_path = _find_first_existing(TRAJ_GROUPS_SOURCES)
    tg_rows = _read_csv(tg_path) if tg_path else []
    print(f"[OK]  轨迹分组: {tg_path.name if tg_path else 'N/A'} ({len(tg_rows)} 行)")

    traj_rows = _read_csv(traj_path)
    print(f"[OK]  轨迹数据: {traj_path.name} ({len(traj_rows)} 行)")

    # 2. 创建输出目录
    DASHBOARD_OUT.mkdir(parents=True, exist_ok=True)

    # 3. 生成各个 JSON
    from datetime import datetime, timezone, timedelta

    # meta.json
    print("\n[1/7] 生成 meta.json ...")
    meta = build_meta(traj_path)
    tz = timezone(timedelta(hours=8))
    meta["generated_at"] = datetime.now(tz).isoformat()
    meta["dataset"]["vehicle_stats_rows"] = len(vs_rows)
    meta["dataset"]["trajectory_group_rows"] = len(tg_rows)
    _write_json("meta.json", meta)

    # timeline.json + events.json
    print("[2/7] 生成 timeline.json ...")
    print("[3/7] 生成 events.json ...")
    timeline, events = build_timeline_and_events(traj_path, cs_rows)
    _write_json("timeline.json", timeline)
    _write_json("events.json", {"items": events, "total": len(events)})

    # overview.json
    print("[4/7] 生成 overview.json ...")
    overview = build_overview(traj_rows, events, vs_rows, tg_rows)
    _write_json("overview.json", overview)

    # charts.json
    print("[5/7] 生成 charts.json ...")
    charts = build_charts(events, vs_rows, tg_rows)
    _write_json("charts.json", charts)

    # trajectories.json
    print("[6/7] 生成 trajectories.json ...")
    trajectories = build_trajectories(traj_rows, tg_rows)
    _write_json("trajectories.json", trajectories)

    # validation.json
    print("[7/7] 生成 validation.json ...")
    validation = build_validation()
    if validation:
        _write_json("validation.json", validation)
    else:
        print("  [SKIP] 未找到 validation 数据")

    # 4. 汇总
    print("\n" + "=" * 60)
    print(f"  构建完成！输出目录: {DASHBOARD_OUT}")
    print(f"  生成文件:")
    for f in sorted(DASHBOARD_OUT.glob("*.json")):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name}  ({size_kb:.1f} KB)")
    print("=" * 60)


def _write_json(filename: str, data: dict | list):
    path = DASHBOARD_OUT / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    size_kb = path.stat().st_size / 1024
    print(f"  -> {filename} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
