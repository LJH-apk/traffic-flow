"""
人工复核集验证工具。

该脚本用于验证已有算法输出，而不是替代主办方隐藏评分。核心目标是把
cross_section.csv 与人工过线标注对齐，输出可写入报告的汇总指标，以及
后续绘图可直接使用的明细 CSV/JSON。

人工过线标注 CSV 至少包含：
    gt_frame_id, section, class_name, direction

建议额外包含 track_id；若缺失，匹配会退化为 section + frame window。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


DEFAULT_OUTPUT_DIR = Path("outputs") / "validation"


@dataclass(frozen=True)
class EventMatch:
    match_status: str
    pred_index: int | None
    gt_index: int | None
    section: str
    track_id: str
    class_name: str
    direction: str
    pred_frame_id: int | None
    gt_frame_id: int | None
    frame_error_s: float | None
    direction_correct: int | None
    class_correct: int | None


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_int(value: object, default: int | None = None) -> int | None:
    try:
        if value in (None, ""):
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _to_float(value: object, default: float | None = None) -> float | None:
    try:
        if value in (None, ""):
            return default
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _fmt(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return round(value, 4)
    return value


def _same_track(pred: dict[str, str], gt: dict[str, str]) -> bool:
    gt_tid = str(gt.get("track_id", "")).strip()
    if not gt_tid:
        return True
    return str(pred.get("track_id", "")).strip() == gt_tid


def _match_events(
    pred_events: list[dict[str, str]],
    gt_events: list[dict[str, str]],
    fps: float,
    frame_tolerance: int,
) -> list[EventMatch]:
    matched_pred: set[int] = set()
    matches: list[EventMatch] = []

    for gt_idx, gt in enumerate(gt_events):
        gt_frame = _to_int(gt.get("gt_frame_id", gt.get("frame_id")))
        section = str(gt.get("section", "")).strip()
        if gt_frame is None:
            continue

        best_idx: int | None = None
        best_err: int | None = None
        for pred_idx, pred in enumerate(pred_events):
            if pred_idx in matched_pred:
                continue
            if str(pred.get("section", "")).strip() != section:
                continue
            if not _same_track(pred, gt):
                continue
            pred_frame = _to_int(pred.get("frame_id"))
            if pred_frame is None:
                continue
            err = abs(pred_frame - gt_frame)
            if err > frame_tolerance:
                continue
            if best_err is None or err < best_err:
                best_idx = pred_idx
                best_err = err

        if best_idx is None:
            matches.append(EventMatch(
                match_status="FN",
                pred_index=None,
                gt_index=gt_idx,
                section=section,
                track_id=str(gt.get("track_id", "")),
                class_name=str(gt.get("class_name", "")),
                direction=str(gt.get("direction", "")),
                pred_frame_id=None,
                gt_frame_id=gt_frame,
                frame_error_s=None,
                direction_correct=None,
                class_correct=None,
            ))
            continue

        matched_pred.add(best_idx)
        pred = pred_events[best_idx]
        pred_frame = _to_int(pred.get("frame_id"))
        frame_error_s = abs((pred_frame or 0) - gt_frame) / fps
        pred_direction = str(pred.get("direction", ""))
        gt_direction = str(gt.get("direction", ""))
        pred_class = str(pred.get("class_name", ""))
        gt_class = str(gt.get("class_name", ""))
        matches.append(EventMatch(
            match_status="TP",
            pred_index=best_idx,
            gt_index=gt_idx,
            section=section,
            track_id=str(pred.get("track_id", "")),
            class_name=pred_class,
            direction=pred_direction,
            pred_frame_id=pred_frame,
            gt_frame_id=gt_frame,
            frame_error_s=frame_error_s,
            direction_correct=int(pred_direction == gt_direction) if gt_direction else None,
            class_correct=int(pred_class == gt_class) if gt_class else None,
        ))

    for pred_idx, pred in enumerate(pred_events):
        if pred_idx in matched_pred:
            continue
        matches.append(EventMatch(
            match_status="FP",
            pred_index=pred_idx,
            gt_index=None,
            section=str(pred.get("section", "")),
            track_id=str(pred.get("track_id", "")),
            class_name=str(pred.get("class_name", "")),
            direction=str(pred.get("direction", "")),
            pred_frame_id=_to_int(pred.get("frame_id")),
            gt_frame_id=None,
            frame_error_s=None,
            direction_correct=None,
            class_correct=None,
        ))

    return matches


def _event_match_rows(matches: Iterable[EventMatch]) -> list[dict]:
    rows = []
    for m in matches:
        rows.append({
            "match_status": m.match_status,
            "pred_index": _fmt(m.pred_index),
            "gt_index": _fmt(m.gt_index),
            "section": m.section,
            "track_id": m.track_id,
            "class_name": m.class_name,
            "direction": m.direction,
            "pred_frame_id": _fmt(m.pred_frame_id),
            "gt_frame_id": _fmt(m.gt_frame_id),
            "frame_error_s": _fmt(m.frame_error_s),
            "direction_correct": _fmt(m.direction_correct),
            "class_correct": _fmt(m.class_correct),
        })
    return rows


def _headway_details(gt_events: list[dict[str, str]], pred_events: list[dict[str, str]], fps: float) -> list[dict]:
    pred_by_key = {
        (str(r.get("section", "")), str(r.get("track_id", "")), _to_int(r.get("frame_id"))): r
        for r in pred_events
    }
    sorted_gt = sorted(
        gt_events,
        key=lambda r: (
            str(r.get("section", "")),
            str(r.get("direction", "")),
            _to_int(r.get("gt_frame_id", r.get("frame_id")), 0) or 0,
        ),
    )
    rows: list[dict] = []
    last_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for gt in sorted_gt:
        section = str(gt.get("section", ""))
        direction = str(gt.get("direction", ""))
        gt_frame = _to_int(gt.get("gt_frame_id", gt.get("frame_id")))
        if gt_frame is None:
            continue
        key = (section, direction)
        previous = last_by_key.get(key)
        last_by_key[key] = gt
        if previous is None:
            continue
        prev_frame = _to_int(previous.get("gt_frame_id", previous.get("frame_id")))
        if prev_frame is None:
            continue
        gt_headway = (gt_frame - prev_frame) / fps

        pred = None
        tid = str(gt.get("track_id", ""))
        for pred_key, candidate in pred_by_key.items():
            p_section, p_tid, p_frame = pred_key
            if p_section == section and p_tid == tid and p_frame is not None and abs(p_frame - gt_frame) <= int(round(fps)):
                pred = candidate
                break
        pred_headway = _to_float(pred.get("headway_s") if pred else None)
        abs_error = abs(pred_headway - gt_headway) if pred_headway is not None else None
        mape = abs_error / gt_headway if abs_error is not None and gt_headway > 0 else None
        rows.append({
            "section": section,
            "direction": direction,
            "track_id": tid,
            "gt_frame_id": gt_frame,
            "previous_gt_frame_id": prev_frame,
            "gt_headway_s": round(gt_headway, 4),
            "pred_headway_s": _fmt(pred_headway),
            "abs_error_s": _fmt(abs_error),
            "mape": _fmt(mape),
            "error_le_0_5s": int(abs_error <= 0.5) if abs_error is not None else "",
        })
    return rows


def _spacing_details(pred_events: list[dict[str, str]], tolerance_m: float = 2.0) -> list[dict]:
    rows: list[dict] = []
    last_by_key: dict[tuple[str, str], dict[str, str]] = {}
    ordered = sorted(
        pred_events,
        key=lambda r: (
            str(r.get("section", "")),
            str(r.get("direction", "")),
            _to_float(r.get("timestamp_s"), 0.0) or 0.0,
            _to_int(r.get("frame_id"), 0) or 0,
        ),
    )
    for event in ordered:
        section = str(event.get("section", ""))
        direction = str(event.get("direction", ""))
        key = (section, direction)
        previous = last_by_key.get(key)
        last_by_key[key] = event
        if previous is None:
            continue
        prev_speed = _to_float(previous.get("speed_kmh"), 0.0) or 0.0
        headway = _to_float(event.get("headway_s"))
        spacing = _to_float(event.get("spacing_m"))
        expected = prev_speed / 3.6 * headway if headway is not None else None
        error = abs(spacing - expected) if spacing is not None and expected is not None else None
        rows.append({
            "section": section,
            "direction": direction,
            "track_id": str(event.get("track_id", "")),
            "frame_id": _fmt(_to_int(event.get("frame_id"))),
            "previous_track_id": str(previous.get("track_id", "")),
            "previous_speed_kmh": _fmt(prev_speed),
            "headway_s": _fmt(headway),
            "spacing_m": _fmt(spacing),
            "expected_spacing_m": _fmt(expected),
            "spacing_error_m": _fmt(error),
            "passed": int(error <= tolerance_m) if error is not None else "",
        })
    return rows


def _anomaly_rows(pred_events: list[dict[str, str]], fps: float) -> list[dict]:
    rows: list[dict] = []
    sorted_events = sorted(
        enumerate(pred_events),
        key=lambda item: (
            str(item[1].get("track_id", "")),
            str(item[1].get("section", "")),
            _to_int(item[1].get("frame_id"), 0) or 0,
        ),
    )
    last_by_track_section: dict[tuple[str, str], tuple[int, dict[str, str]]] = {}

    def add(anomaly_type: str, idx: int, event: dict[str, str], detail: str) -> None:
        rows.append({
            "anomaly_type": anomaly_type,
            "detail": detail,
            "row_index": idx,
            "frame_id": event.get("frame_id", ""),
            "timestamp_s": event.get("timestamp_s", ""),
            "section": event.get("section", ""),
            "track_id": event.get("track_id", ""),
            "class_name": event.get("class_name", ""),
            "direction": event.get("direction", ""),
            "speed_kmh": event.get("speed_kmh", ""),
            "headway_s": event.get("headway_s", ""),
            "spacing_m": event.get("spacing_m", ""),
        })

    for idx, event in sorted_events:
        track_id = str(event.get("track_id", ""))
        section = str(event.get("section", ""))
        frame_id = _to_int(event.get("frame_id"))
        key = (track_id, section)
        previous = last_by_track_section.get(key)
        if previous is not None and frame_id is not None:
            prev_idx, prev_event = previous
            prev_frame = _to_int(prev_event.get("frame_id"))
            if prev_frame is not None and abs(frame_id - prev_frame) <= fps:
                add("duplicate_event", idx, event, f"previous_row_index={prev_idx}")
            if prev_event.get("direction") and event.get("direction") and prev_event.get("direction") != event.get("direction"):
                add("opposite_direction", idx, event, f"previous_row_index={prev_idx}")
        last_by_track_section[key] = (idx, event)

    for idx, event in enumerate(pred_events):
        speed = _to_float(event.get("speed_kmh"))
        headway = _to_float(event.get("headway_s"))
        spacing = _to_float(event.get("spacing_m"))
        if speed is not None and speed > 100:
            add("speed_out_of_range", idx, event, "speed_kmh > 100")
        if headway is not None and 0 < headway < 0.3:
            add("headway_too_short", idx, event, "0 < headway_s < 0.3")
        if spacing is not None and spacing < 0:
            add("negative_spacing", idx, event, "spacing_m < 0")
    return rows


def _summary(matches: list[EventMatch], headway_rows: list[dict], spacing_rows: list[dict], anomalies: list[dict]) -> dict[str, float | int | str]:
    tp = sum(1 for m in matches if m.match_status == "TP")
    fp = sum(1 for m in matches if m.match_status == "FP")
    fn = sum(1 for m in matches if m.match_status == "FN")
    frame_errors = [m.frame_error_s for m in matches if m.match_status == "TP" and m.frame_error_s is not None]
    direction_values = [m.direction_correct for m in matches if m.match_status == "TP" and m.direction_correct is not None]
    class_values = [m.class_correct for m in matches if m.match_status == "TP" and m.class_correct is not None]
    headway_errors = [_to_float(r.get("abs_error_s")) for r in headway_rows if _to_float(r.get("abs_error_s")) is not None]
    headway_mapes = [_to_float(r.get("mape")) for r in headway_rows if _to_float(r.get("mape")) is not None]
    spacing_pass = [_to_int(r.get("passed")) for r in spacing_rows if _to_int(r.get("passed")) is not None]
    duplicate_count = sum(1 for r in anomalies if r["anomaly_type"] == "duplicate_event")
    physical_count = sum(1 for r in anomalies if r["anomaly_type"] in {"speed_out_of_range", "headway_too_short", "negative_spacing"})

    return {
        "event_tp": tp,
        "event_fp": fp,
        "event_fn": fn,
        "event_precision": tp / (tp + fp) if tp + fp else 0.0,
        "event_recall": tp / (tp + fn) if tp + fn else 0.0,
        "crossing_time_mae_s": mean(frame_errors) if frame_errors else 0.0,
        "direction_accuracy": mean(direction_values) if direction_values else "",
        "class_accuracy": mean(class_values) if class_values else "",
        "headway_mae_s": mean(headway_errors) if headway_errors else 0.0,
        "headway_mape": mean(headway_mapes) if headway_mapes else "",
        "headway_error_le_0_5s_rate": mean([_to_int(r.get("error_le_0_5s")) for r in headway_rows if _to_int(r.get("error_le_0_5s")) is not None]) if headway_rows else "",
        "spacing_consistency_pass_rate": mean(spacing_pass) if spacing_pass else "",
        "duplicate_event_count": duplicate_count,
        "physical_anomaly_count": physical_count,
        "total_anomaly_count": len(anomalies),
    }


def validate_outputs(
    cross_section_csv: Path,
    manual_crossing_csv: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    fps: float = 25.0,
    frame_tolerance: int = 10,
) -> dict:
    """验证断面事件与交通参数一致性，并保存报告数据。

    Returns:
        汇总指标字典，内容同时写入 validation_summary.csv。
    """
    pred_events = _read_csv(cross_section_csv)
    gt_events = _read_csv(manual_crossing_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    matches = _match_events(pred_events, gt_events, fps=fps, frame_tolerance=frame_tolerance)
    match_rows = _event_match_rows(matches)
    headway_rows = _headway_details(gt_events, pred_events, fps=fps)
    spacing_rows = _spacing_details(pred_events)
    anomalies = _anomaly_rows(pred_events, fps=fps)
    summary = _summary(matches, headway_rows, spacing_rows, anomalies)

    generated_files: list[str] = []

    def write(name: str, rows: list[dict], fields: list[str]) -> None:
        _write_csv(output_dir / name, rows, fields)
        generated_files.append(name)

    summary_rows = [{"metric": key, "value": _fmt(value)} for key, value in summary.items()]
    write("validation_summary.csv", summary_rows, ["metric", "value"])
    write("event_matching_details.csv", match_rows, [
        "match_status", "pred_index", "gt_index", "section", "track_id",
        "class_name", "direction", "pred_frame_id", "gt_frame_id",
        "frame_error_s", "direction_correct", "class_correct",
    ])
    write("headway_details.csv", headway_rows, [
        "section", "direction", "track_id", "gt_frame_id", "previous_gt_frame_id",
        "gt_headway_s", "pred_headway_s", "abs_error_s", "mape", "error_le_0_5s",
    ])
    write("spacing_consistency_details.csv", spacing_rows, [
        "section", "direction", "track_id", "frame_id", "previous_track_id",
        "previous_speed_kmh", "headway_s", "spacing_m", "expected_spacing_m",
        "spacing_error_m", "passed",
    ])
    write("anomaly_events.csv", anomalies, [
        "anomaly_type", "detail", "row_index", "frame_id", "timestamp_s",
        "section", "track_id", "class_name", "direction", "speed_kmh",
        "headway_s", "spacing_m",
    ])

    log = {
        "cross_section_csv": str(cross_section_csv),
        "manual_crossing_csv": str(manual_crossing_csv),
        "output_dir": str(output_dir),
        "fps": fps,
        "frame_tolerance": frame_tolerance,
        "manual_event_count": len(gt_events),
        "pred_event_count": len(pred_events),
        "generated_files": generated_files,
        "summary": summary,
        "notes": [
            "速度、车头时距、车头间距均按可观测中间量复算或一致性检查。",
            "车头间距按 previous_speed / 3.6 * headway_s 做衍生一致性校验。",
            "该验证报告用于人工复核集，不代表主办方隐藏测试评分。",
        ],
    }
    log_path = output_dir / "validation_log.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    generated_files.append(log_path.name)
    log["generated_files"] = generated_files
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于人工复核集验证交通流算法输出。")
    parser.add_argument("--cross-section", type=Path, default=Path("outputs") / "cross_section.csv",
                        help="算法输出的 cross_section.csv")
    parser.add_argument("--manual-crossing", type=Path, required=True,
                        help="人工过线标注 CSV，包含 gt_frame_id/section/class_name/direction")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="验证报告与明细输出目录")
    parser.add_argument("--fps", type=float, default=25.0, help="视频 FPS")
    parser.add_argument("--frame-tolerance", type=int, default=10, help="事件匹配帧容差")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = validate_outputs(
        cross_section_csv=args.cross_section,
        manual_crossing_csv=args.manual_crossing,
        output_dir=args.output_dir,
        fps=args.fps,
        frame_tolerance=args.frame_tolerance,
    )
    print("=== 人工复核验证摘要 ===")
    for key, value in summary.items():
        print(f"{key}: {_fmt(value)}")
    print(f"验证明细已保存: {args.output_dir}")


if __name__ == "__main__":
    main()
