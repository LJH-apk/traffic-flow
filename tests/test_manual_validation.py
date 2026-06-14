import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.evaluation.manual_validation import export_normalized_manual_annotations, validate_outputs


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_validation_exports_plot_ready_details_and_log(tmp_path):
    cross_csv = tmp_path / "cross_section.csv"
    manual_csv = tmp_path / "manual_crossing_annotations.csv"
    output_dir = tmp_path / "validation"

    _write_csv(cross_csv, [
        {
            "frame_id": 25, "timestamp_s": 1.0, "section": "北进口主断面",
            "track_id": 1, "class_name": "car", "direction": "到达",
            "speed_kmh": 36.0, "headway_s": 0.0, "spacing_m": 0.0,
        },
        {
            "frame_id": 38, "timestamp_s": 1.52, "section": "北进口主断面",
            "track_id": 1, "class_name": "car", "direction": "到达",
            "speed_kmh": 36.0, "headway_s": 0.52, "spacing_m": 5.2,
        },
        {
            "frame_id": 75, "timestamp_s": 3.0, "section": "北进口主断面",
            "track_id": 2, "class_name": "car", "direction": "到达",
            "speed_kmh": 72.0, "headway_s": 2.0, "spacing_m": 20.0,
        },
        {
            "frame_id": 100, "timestamp_s": 4.0, "section": "北进口主断面",
            "track_id": 3, "class_name": "car", "direction": "离去",
            "speed_kmh": 120.0, "headway_s": 0.2, "spacing_m": -1.0,
        },
    ])
    _write_csv(manual_csv, [
        {
            "gt_frame_id": 26, "section": "北进口主断面", "track_id": 1,
            "class_name": "car", "direction": "到达",
        },
        {
            "gt_frame_id": 75, "section": "北进口主断面", "track_id": 2,
            "class_name": "car", "direction": "到达",
        },
    ])

    summary = validate_outputs(
        cross_section_csv=cross_csv,
        manual_crossing_csv=manual_csv,
        output_dir=output_dir,
        fps=25.0,
        frame_tolerance=10,
    )

    assert summary["event_precision"] == pytest.approx(2 / 3)
    assert summary["event_recall"] == pytest.approx(1.0)
    assert summary["duplicate_event_count"] == 0
    assert summary["physical_anomaly_count"] == 3
    assert summary["headway_mae_s"] == pytest.approx(0.04)
    assert summary["pred_dedup_count"] == 3
    assert summary["manual_in_scope_count"] == 2
    assert summary["overlap_max_frame"] == 75

    expected_files = {
        "validation_summary.csv",
        "event_matching_details.csv",
        "headway_details.csv",
        "spacing_consistency_details.csv",
        "anomaly_events.csv",
        "metric_breakdown.csv",
        "section_class_breakdown.csv",
        "section_lane_breakdown.csv",
        "validation_log.json",
    }
    assert expected_files.issubset({p.name for p in output_dir.iterdir()})

    matching = _read_csv(output_dir / "event_matching_details.csv")
    assert [row["match_status"] for row in matching].count("TP") == 2
    assert [row["match_status"] for row in matching].count("FP") == 1
    assert matching[0]["frame_error_s"] == "0.04"
    assert matching[0]["lane_id"] == ""

    spacing = _read_csv(output_dir / "spacing_consistency_details.csv")
    row = next(r for r in spacing if r["track_id"] == "2")
    assert row["expected_spacing_m"] == "20.0"
    assert row["spacing_error_m"] == "0.0"
    assert row["passed"] == "1"

    anomalies = _read_csv(output_dir / "anomaly_events.csv")
    assert {row["anomaly_type"] for row in anomalies} == {
        "speed_out_of_range", "headway_too_short", "negative_spacing",
    }

    log = json.loads((output_dir / "validation_log.json").read_text(encoding="utf-8"))
    assert log["fps"] == 25.0
    assert "event_matching_details.csv" in log["generated_files"]
    assert "metric_breakdown.csv" in log["generated_files"]
    assert log["manual_crossing_csv"] == str(manual_csv)


def test_export_normalized_manual_annotations_handles_xlsx_aliases(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")

    manual_xlsx = tmp_path / "manual.xlsx"
    normalized_csv = tmp_path / "manual_crossing.csv"

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "manual_crossing"
    ws.append(["entrance", "video_name", 0, "主", "到达", "car", "L2", "NC001", "小汽车", None])
    ws.append(["北进口", "北进口_20260420075959至20260420081500.mp4", 108, "右", "到达", "BUS", "L1", "NB001", "公交车", None])
    ws.append(["北进口", "北进口_20260420075959至20260420081500.mp5", 188, "北进口右转专用道", "到达", "YRU", "L1", "NT001", "货运皮卡", None])
    ws.append([None, None, "frame_id", "section_name", "direction", "class_name", "lane_id", "manual_vehicle_id", "note", None])
    wb.save(manual_xlsx)

    events = export_normalized_manual_annotations(manual_xlsx, normalized_csv)

    assert len(events) == 2
    assert events[0]["section"] == "北进口右转"
    assert events[0]["class_name"] == "bus"
    assert events[1]["section"] == "北进口右转"
    assert events[1]["class_name"] == "truck"

    rows = _read_csv(normalized_csv)
    assert len(rows) == 2
    assert rows[0]["track_id"] == "NB001"
