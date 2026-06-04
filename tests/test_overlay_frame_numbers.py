import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.utils.overlay_frame_numbers import (
    default_output_path,
    default_reference_output_path,
    draw_frame_label,
    draw_reference_overlay,
    infer_entrance,
)
from src.cross_section.section_calibration import load_section_lines, load_stop_lines


def test_draw_frame_label_changes_hud_region():
    frame = np.zeros((240, 480, 3), dtype=np.uint8)

    draw_frame_label(frame, frame_idx=125, fps=25.0)

    assert frame[:90, :260].sum() > 0
    assert frame[120:, 300:].sum() == 0
    assert frame[:70, :210].sum() > 0
    assert frame[:90, 360:].sum() == 0


def test_default_output_path_adds_frame_index_suffix():
    source = Path("outputs/trajectory.mp4")

    result = default_output_path(source)

    assert result == Path("outputs/annotation_videos/trajectory_5min_framed_1080p.mp4")


def test_infer_entrance_treats_fixed_video_as_south_entrance():
    assert infer_entrance(Path("test_video_fixed.mp4")) == "南进口"
    assert infer_entrance(Path("东进口_20260420075958至20260420081459.mp4")) == "东进口"


def test_default_reference_output_path_adds_reference_suffix():
    source = Path("test_video_fixed.mp4")

    result = default_reference_output_path(source)

    assert result == Path("outputs/annotation_videos/test_video_fixed_5min_reference_1080p.mp4")


def test_draw_reference_overlay_changes_section_lane_and_stop_line_regions():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    lines = [("测试断面", 100, 100, 500, 100, "到达", "离去")]
    lanes = {1: [(300, 200), (320, 800)]}
    stop_lines = [("停车线", 120, 300, 520, 300)]

    draw_reference_overlay(
        frame,
        frame_idx=125,
        fps=25.0,
        section_lines=lines,
        lanes=lanes,
        stop_lines=stop_lines,
    )

    assert frame[:90, :360].sum() > 0
    assert frame[90:115, 95:505].sum() > 0
    assert frame[190:810, 290:330].sum() > 0
    assert frame[290:310, 115:525].sum() > 0


def test_draw_reference_overlay_renders_chinese_label_with_compact_text():
    frame = np.zeros((300, 800, 3), dtype=np.uint8)
    lines = [("南进口主断面", 100, 180, 500, 180, "到达", "离去")]

    draw_reference_overlay(frame, frame_idx=1, fps=25.0, section_lines=lines, lanes={})

    assert frame[115:180, 95:520].sum() > 0
    assert frame[:90, 360:].sum() == 0


def test_lane_annotator_supports_six_parallel_lines():
    from src.cross_section.lane_annotator import ZoomPanAnnotator

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    annotator = ZoomPanAnnotator(image, n_lanes=6)

    assert annotator.n_lanes == 6
    assert sorted(annotator.lanes.keys()) == [1, 2, 3, 4, 5, 6]


def test_load_section_lines_prefers_manual_sections_json(tmp_path):
    entrance_dir = tmp_path / "北进口"
    entrance_dir.mkdir()
    (entrance_dir / "sections.json").write_text(
        """
        {
          "image": "calibrations/北进口/ref.jpg",
          "sections": [
            {
              "name": "北进口主断面",
              "points": [[10, 20], [300, 40]],
              "dir_pos": "到达",
              "dir_neg": "离去"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    assert load_section_lines("北进口", tmp_path) == [
        ("北进口主断面", 10, 20, 300, 40, "到达", "离去")
    ]


def test_load_stop_lines_reads_manual_stop_line_json(tmp_path):
    entrance_dir = tmp_path / "南进口"
    entrance_dir.mkdir()
    (entrance_dir / "stop_lines.json").write_text(
        """
        {
          "image": "calibrations/南进口/ref.jpg",
          "stop_lines": [
            {"name": "南进口停车线", "points": [[11, 22], [333, 44]]}
          ]
        }
        """,
        encoding="utf-8",
    )

    assert load_stop_lines("南进口", tmp_path) == [("南进口停车线", 11, 22, 333, 44)]
