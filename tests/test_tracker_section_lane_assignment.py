import json
from pathlib import Path

from src.trajectory.tracker import TrajectoryTracker


def _load_north_lane_context():
    root = Path(__file__).resolve().parents[1]
    with (root / "calibrations" / "北进口" / "lanes.json").open(encoding="utf-8") as fh:
        lanes = json.load(fh)["lanes"]
    with (root / "calibrations" / "北进口" / "sections.json").open(encoding="utf-8") as fh:
        sections = json.load(fh)["sections"]
    lane_points = {int(k): [tuple(pt) for pt in v] for k, v in lanes.items()}
    lane_splines = TrajectoryTracker._build_lane_splines(lane_points)
    section_lines_by_name = {
        item["name"]: (
            item["name"],
            item["points"][0][0],
            item["points"][0][1],
            item["points"][1][0],
            item["points"][1][1],
            item["dir_pos"],
            item["dir_neg"],
        )
        for item in sections
    }
    return lane_splines, section_lines_by_name


def test_assign_section_event_lane_main_section():
    lane_splines, section_lines_by_name = _load_north_lane_context()

    sec_line = section_lines_by_name["北进口主断面"]
    boundary_t = TrajectoryTracker._compute_lane_boundary_cross_t(sec_line, lane_splines)
    # 预计算的 t 阈值（从 spline 导出）
    sorted_by_t = sorted(boundary_t.items(), key=lambda kv: kv[1], reverse=True)
    t_outer = sorted_by_t[0][1]   # L1 的 t（最大）
    t_inner = sorted_by_t[-1][1]  # L5 的 t（最小）

    # 到达方向 + cross_t 在各车道区间内
    # Lane 1: t 介于 L2 和 L1 之间
    t_l2 = sorted_by_t[1][1]
    t_mid_lane1 = (t_outer + t_l2) / 2
    assert TrajectoryTracker._assign_section_event_lane(
        "北进口主断面", "car", "到达", t_mid_lane1, boundary_t,
    ) == 1

    # OPPOSITE: cross_t >= t_outer
    assert TrajectoryTracker._assign_section_event_lane(
        "北进口主断面", "car", "到达", t_outer + 0.01, boundary_t,
    ) == "OPPOSITE"

    # 离去方向 → 直接返回 OPPOSITE（无论 cross_t 多少）
    assert TrajectoryTracker._assign_section_event_lane(
        "北进口主断面", "car", "离去", 0.5, boundary_t,
    ) == "OPPOSITE"


def test_assign_section_event_lane_turn_sections():
    lane_splines, section_lines_by_name = _load_north_lane_context()

    # 右转断面
    sec_line = section_lines_by_name["北进口右转专用道"]
    boundary_t = TrajectoryTracker._compute_lane_boundary_cross_t(sec_line, lane_splines)
    sorted_by_t = sorted(boundary_t.items(), key=lambda kv: kv[1], reverse=True)
    t_outer = sorted_by_t[0][1]
    t_inner = sorted_by_t[-1][1]

    # 主车道 (lane 1): t 在 inner 和 outer 之间
    t_mid = (t_outer + t_inner) / 2
    assert TrajectoryTracker._assign_section_event_lane(
        "北进口右转专用道", "car", "到达", t_mid, boundary_t,
    ) == 1
    # 外侧窄区 (lane 2): t >= outer
    assert TrajectoryTracker._assign_section_event_lane(
        "北进口右转专用道", "truck", "到达", t_outer + 0.01, boundary_t,
    ) == 2

    # 掉头断面
    sec_line2 = section_lines_by_name["北进口掉头车道"]
    boundary_t2 = TrajectoryTracker._compute_lane_boundary_cross_t(sec_line2, lane_splines)
    # 掉头无 special 处理，走通用转弯逻辑
    sorted_by_t2 = sorted(boundary_t2.items(), key=lambda kv: kv[1], reverse=True)
    t_outer2 = sorted_by_t2[0][1]
    t_inner2 = sorted_by_t2[-1][1]
    t_mid2 = (t_outer2 + t_inner2) / 2
    assert TrajectoryTracker._assign_section_event_lane(
        "北进口掉头车道", "truck", "到达", t_mid2, boundary_t2,
    ) == 1


def test_trajectory_matches_turn_section_filters_straight_motion():
    turning = [(2000 + i * 12, 780 + i * 20) for i in range(10)]
    straight = [(2000 + i * 2, 500 + i * 22) for i in range(10)]
    left_turn_like = [(2600 - i * 12, 820 + i * 8) for i in range(10)]

    assert TrajectoryTracker._trajectory_matches_turn_section("北进口右转专用道", "motorcycle", turning) is True
    assert TrajectoryTracker._trajectory_matches_turn_section("北进口右转专用道", "motorcycle", straight) is False
    assert TrajectoryTracker._trajectory_matches_turn_section("北进口右转专用道", "car", left_turn_like) is True
    assert TrajectoryTracker._trajectory_matches_turn_section("北进口右转专用道", "car", turning) is False
