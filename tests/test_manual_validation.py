import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.evaluation.manual_validation import validate_outputs


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

    assert summary["event_precision"] == pytest.approx(0.5)
    assert summary["event_recall"] == pytest.approx(1.0)
    assert summary["duplicate_event_count"] == 1
    assert summary["physical_anomaly_count"] == 3
    assert summary["headway_mae_s"] == pytest.approx(0.04)

    expected_files = {
        "validation_summary.csv",
        "event_matching_details.csv",
        "headway_details.csv",
        "spacing_consistency_details.csv",
        "anomaly_events.csv",
        "validation_log.json",
    }
    assert expected_files.issubset({p.name for p in output_dir.iterdir()})

    matching = _read_csv(output_dir / "event_matching_details.csv")
    assert [row["match_status"] for row in matching].count("TP") == 2
    assert [row["match_status"] for row in matching].count("FP") == 2
    assert matching[0]["frame_error_s"] == "0.04"

    spacing = _read_csv(output_dir / "spacing_consistency_details.csv")
    row = next(r for r in spacing if r["track_id"] == "2")
    assert row["expected_spacing_m"] == "20.0"
    assert row["spacing_error_m"] == "0.0"
    assert row["passed"] == "1"

    anomalies = _read_csv(output_dir / "anomaly_events.csv")
    assert {row["anomaly_type"] for row in anomalies} == {
        "duplicate_event", "speed_out_of_range", "headway_too_short", "negative_spacing",
    }

    log = json.loads((output_dir / "validation_log.json").read_text(encoding="utf-8"))
    assert log["fps"] == 25.0
    assert "event_matching_details.csv" in log["generated_files"]
    assert log["manual_crossing_csv"] == str(manual_csv)
