"""车道归属和转弯断面过滤。"""
from __future__ import annotations

import cv2
import numpy as np

from src.cross_section.counter import CrossSectionDetector

LanePoints = dict[int, list[tuple[int, int]]]
LaneSplines = dict[int, tuple[float, float, object]]
SectionLine = tuple[str, int, int, int, int, str, str]


def build_lane_overlay(height: int, width: int, lanes: LanePoints) -> np.ndarray:
    from scipy.interpolate import UnivariateSpline

    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    lane_colors = {
        1: (0, 255, 0),
        2: (0, 200, 255),
        3: (255, 80, 200),
        4: (255, 200, 0),
        5: (80, 200, 255),
    }
    for lane_id, points in sorted(lanes.items()):
        if len(points) < 2:
            continue
        arr = np.array(points, dtype=np.float64)
        ys, xs = arr[:, 1], arr[:, 0]
        order = np.argsort(ys)
        ys_u, xs_u = ys[order], xs[order]
        _, uid = np.unique(ys_u, return_index=True)
        ys_u, xs_u = ys_u[uid], xs_u[uid]
        try:
            k = min(3, len(ys_u) - 1)
            spline = UnivariateSpline(ys_u, xs_u, k=k, s=200 * len(ys_u))
            y_lo, y_hi = int(ys_u[0]), int(ys_u[-1])
            y_values = np.arange(y_lo, y_hi + 1)
            x_values = np.clip(spline(y_values), 0, width - 1).astype(np.int32)
            curve = np.stack([x_values, y_values], axis=1).reshape(-1, 1, 2)
            cv2.polylines(overlay, [curve], False, lane_colors.get(lane_id, (200, 200, 200)), 4)
        except Exception:
            continue
    return overlay


def assign_lane(cx: float, cy: float, lanes: LanePoints) -> int | str | None:
    from scipy.interpolate import UnivariateSpline

    xs_at_cy: list[tuple[int, float]] = []
    for lane_id, points in sorted(lanes.items()):
        if len(points) < 2:
            continue
        arr = np.array(points, dtype=np.float64)
        ys, xs = arr[:, 1], arr[:, 0]
        order = np.argsort(ys)
        ys_u, xs_u = ys[order], xs[order]
        _, uid = np.unique(ys_u, return_index=True)
        ys_u, xs_u = ys_u[uid], xs_u[uid]
        if cy < float(ys_u[0]) or cy > float(ys_u[-1]):
            continue
        try:
            k = min(3, len(ys_u) - 1)
            spline = UnivariateSpline(ys_u, xs_u, k=k, s=200 * len(ys_u))
            xs_at_cy.append((lane_id, float(spline(cy))))
        except Exception:
            continue
    if len(xs_at_cy) < 2:
        return None

    sorted_by_x = sorted(dict(xs_at_cy).items(), key=lambda item: item[1], reverse=True)
    for idx in range(len(sorted_by_x) - 1):
        left_lane_id, left_x = sorted_by_x[idx]
        right_lane_id, right_x = sorted_by_x[idx + 1]
        if right_x <= cx <= left_x:
            return min(left_lane_id, right_lane_id)

    _, first_x = sorted_by_x[0]
    _, last_x = sorted_by_x[-1]
    if cx > first_x:
        return "OPPOSITE"
    if cx < last_x:
        return None
    return None


def build_lane_splines(lanes: LanePoints) -> LaneSplines:
    from scipy.interpolate import UnivariateSpline

    splines: LaneSplines = {}
    for lane_id, points in sorted(lanes.items()):
        if len(points) < 2:
            continue
        arr = np.array(points, dtype=np.float64)
        ys, xs = arr[:, 1], arr[:, 0]
        order = np.argsort(ys)
        ys_u, xs_u = ys[order], xs[order]
        _, uid = np.unique(ys_u, return_index=True)
        ys_u, xs_u = ys_u[uid], xs_u[uid]
        try:
            k = min(3, len(ys_u) - 1)
            splines[lane_id] = (
                float(ys_u[0]),
                float(ys_u[-1]),
                UnivariateSpline(ys_u, xs_u, k=k, s=200 * len(ys_u)),
            )
        except Exception:
            continue
    return splines


def project_point_to_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> tuple[float, float]:
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-6:
        return ax, ay
    t = ((px - ax) * vx + (py - ay) * vy) / denom
    t = max(0.0, min(1.0, t))
    return ax + t * vx, ay + t * vy


def lane_boundary_xs_at_y(y: float, lane_splines: LaneSplines) -> dict[int, float]:
    xs_by_lane_id: dict[int, float] = {}
    for lane_id, (y0, y1, spline) in lane_splines.items():
        if y < y0 - 60 or y > y1 + 260:
            continue
        try:
            xs_by_lane_id[lane_id] = float(spline(y))
        except Exception:
            continue
    return xs_by_lane_id


def compute_lane_boundary_cross_t(section_line: SectionLine, lane_splines: LaneSplines) -> dict[int, float]:
    _, lx1, ly1, lx2, ly2, _, _ = section_line
    boundaries: dict[int, float] = {}
    y_mid = (ly1 + ly2) / 2.0
    vx, vy = lx2 - lx1, ly2 - ly1
    denom_s = vx * vx + vy * vy
    if denom_s <= 1e-6:
        return boundaries
    for lane_id, (y0, y1, spline) in lane_splines.items():
        if not (y0 - 60 <= y_mid <= y1 + 260):
            continue
        try:
            x_at_y = float(spline(y_mid))
        except Exception:
            continue
        t = ((x_at_y - lx1) * vx + (y_mid - ly1) * vy) / denom_s
        boundaries[lane_id] = max(0.0, min(1.0, t))
    return boundaries


def assign_section_event_lane(
    section_name: str,
    class_name: str,
    direction: str,
    cross_t: float | None,
    boundary_t: dict[int, float] | None,
) -> int | str | None:
    _ = class_name
    if cross_t is None or boundary_t is None or len(boundary_t) < 2:
        return None

    if "主断面" in section_name:
        if direction == "离去":
            return "OPPOSITE"
        sorted_bounds = sorted(boundary_t.items(), key=lambda item: item[1], reverse=True)
        if len(sorted_bounds) < 2:
            return None
        _, t_outer = sorted_bounds[0]
        _, t_inner = sorted_bounds[-1]
        if cross_t >= t_outer:
            return "OPPOSITE"
        if cross_t < t_inner:
            return None
        for idx in range(len(sorted_bounds) - 1):
            left_lane_id, left_t = sorted_bounds[idx]
            right_lane_id, right_t = sorted_bounds[idx + 1]
            if right_t <= cross_t < left_t:
                return min(left_lane_id, right_lane_id)
        return None

    if "东进口" in section_name and "右转" in section_name:
        if not all(lane_id in boundary_t for lane_id in (5, 6)):
            return None
        t_l5 = boundary_t[5]
        t_l6 = boundary_t[6]
        if cross_t >= t_l5:
            return 2
        if t_l6 <= cross_t < t_l5:
            return 5
        return None

    if any(keyword in section_name for keyword in ("右转", "掉头", "左转", "转弯", "待转")):
        sorted_bounds = sorted(boundary_t.items(), key=lambda item: item[1], reverse=True)
        if len(sorted_bounds) < 2:
            return None
        _, t_outer = sorted_bounds[0]
        _, t_inner = sorted_bounds[-1]
        if cross_t >= t_outer:
            return 2
        if t_inner <= cross_t < t_outer:
            return 1
        return None

    return None


def trajectory_matches_turn_section(
    section_name: str,
    class_name: str,
    traj_points: list[tuple[float, float]],
) -> bool:
    if not CrossSectionDetector._is_turn_section(section_name) or not traj_points:
        return True

    recent = traj_points[-20:]
    if len(recent) < 8:
        return False

    xs = [point[0] for point in recent]
    ys = [point[1] for point in recent]
    dx = xs[-1] - xs[0]
    x_span = max(xs) - min(xs)
    positive_x_steps = sum((xs[idx + 1] - xs[idx]) > 2.0 for idx in range(len(xs) - 1))
    negative_x_steps = sum((xs[idx + 1] - xs[idx]) < -2.0 for idx in range(len(xs) - 1))
    dominant_step_ratio = max(positive_x_steps, negative_x_steps) / max(1, len(xs) - 1)
    negative_step_ratio = negative_x_steps / max(1, len(xs) - 1)
    end_x = xs[-1]
    end_y = ys[-1]
    is_motor = str(class_name or "").strip().lower() in {"car", "bus", "truck"}

    if section_name == "北进口右转专用道":
        if is_motor:
            return (
                x_span >= 55.0
                and negative_step_ratio >= 0.35
                and dx <= -20.0
                and end_y >= 700.0
                and end_x >= 1500.0
            )
        return (
            x_span >= 55.0
            and dominant_step_ratio >= 0.45
            and end_y >= 760.0
            and end_x >= 1750.0
        )

    if section_name == "北进口掉头车道":
        return (
            dx >= 45.0
            and x_span >= 60.0
            and dominant_step_ratio >= 0.55
        )

    return True
