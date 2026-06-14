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

    assert TrajectoryTracker._assign_section_event_lane(
        "北进口主断面", "car",
        1940, 430, 1980, 470,
        lane_splines, section_lines_by_name,
    ) == 1
    assert TrajectoryTracker._assign_section_event_lane(
        "北进口主断面", "car",
        2100, 430, 2140, 470,
        lane_splines, section_lines_by_name,
    ) == "OPPOSITE"


def test_assign_section_event_lane_turn_sections():
    lane_splines, section_lines_by_name = _load_north_lane_context()

    assert TrajectoryTracker._assign_section_event_lane(
        "北进口右转专用道", "car",
        2580, 920, 2620, 980,
        lane_splines, section_lines_by_name,
    ) == 1
    assert TrajectoryTracker._assign_section_event_lane(
        "北进口右转专用道", "truck",
        2830, 920, 2870, 980,
        lane_splines, section_lines_by_name,
    ) == 2
    assert TrajectoryTracker._assign_section_event_lane(
        "北进口掉头车道", "truck",
        2440, 630, 2480, 700,
        lane_splines, section_lines_by_name,
    ) == 1


def test_trajectory_matches_turn_section_filters_straight_motion():
    turning = [(2000 + i * 12, 780 + i * 20) for i in range(10)]
    straight = [(2000 + i * 2, 500 + i * 22) for i in range(10)]
    left_turn_like = [(2600 - i * 12, 820 + i * 8) for i in range(10)]

    assert TrajectoryTracker._trajectory_matches_turn_section("北进口右转专用道", "motorcycle", turning) is True
    assert TrajectoryTracker._trajectory_matches_turn_section("北进口右转专用道", "motorcycle", straight) is False
    assert TrajectoryTracker._trajectory_matches_turn_section("北进口右转专用道", "car", left_turn_like) is True
    assert TrajectoryTracker._trajectory_matches_turn_section("北进口右转专用道", "car", turning) is False
