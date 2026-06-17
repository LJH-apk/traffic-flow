"""断面线和停车线标定文件读写工具。"""
from __future__ import annotations

import json
from pathlib import Path

from src.config.settings import CALIBRATIONS_DIR, SECTION_LINES_MAP

SectionLine = tuple[str, int, int, int, int, str, str]
StopLine = tuple[str, int, int, int, int]


def _line_points_to_xyxy(points) -> tuple[int, int, int, int] | None:
    if not isinstance(points, list) or len(points) != 2:
        return None
    try:
        (x1, y1), (x2, y2) = points
        return int(x1), int(y1), int(x2), int(y2)
    except (TypeError, ValueError):
        return None


def sections_path(entrance: str, base_dir: Path = CALIBRATIONS_DIR) -> Path:
    return base_dir / entrance / "sections.json"


def stop_lines_path(entrance: str, base_dir: Path = CALIBRATIONS_DIR) -> Path:
    return base_dir / entrance / "stop_lines.json"


def load_section_lines(entrance: str | None, base_dir: Path = CALIBRATIONS_DIR) -> list[SectionLine]:
    """优先加载标定目录中的 sections.json，不存在时回退 settings.py。"""
    if not entrance:
        return []

    path = sections_path(entrance, base_dir)
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw_lines = raw.get("sections", raw)
        lines: list[SectionLine] = []
        for idx, item in enumerate(raw_lines, start=1):
            if not isinstance(item, dict):
                continue
            xyxy = _line_points_to_xyxy(item.get("points"))
            if xyxy is None:
                continue
            name = str(item.get("name") or f"{entrance}断面{idx}")
            dir_pos = str(item.get("dir_pos") or "到达")
            dir_neg = str(item.get("dir_neg") or "离去")
            lines.append((name, *xyxy, dir_pos, dir_neg))
        if lines:
            return lines

    return SECTION_LINES_MAP.get(entrance, [])


def load_all_section_lines(base_dir: Path = CALIBRATIONS_DIR) -> list[SectionLine]:
    """加载所有进口断面，优先使用各自 sections.json。"""
    lines: list[SectionLine] = []
    for entrance in SECTION_LINES_MAP:
        lines.extend(load_section_lines(entrance, base_dir))
    return lines


def load_stop_lines(entrance: str | None, base_dir: Path = CALIBRATIONS_DIR) -> list[StopLine]:
    """加载标定目录中的 stop_lines.json。"""
    if not entrance:
        return []
    path = stop_lines_path(entrance, base_dir)
    if not path.exists():
        return []

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw_lines = raw.get("stop_lines", raw)
    lines: list[StopLine] = []
    for idx, item in enumerate(raw_lines, start=1):
        name = item.get("name", f"停车线{idx}") if isinstance(item, dict) else f"停车线{idx}"
        points = item.get("points", item) if isinstance(item, dict) else item
        xyxy = _line_points_to_xyxy(points)
        if xyxy is None:
            continue
        lines.append((str(name), *xyxy))
    return lines
