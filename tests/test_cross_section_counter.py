import numpy as np

from src.cross_section.counter import CrossSectionDetector


def test_cross_section_detector_suppresses_short_jitter_retrigger():
    detector = CrossSectionDetector(
        lines=[("北进口主断面", 0, 10, 100, 10, "到达", "离去")],
        homography=None,
        pixels_per_meter=10.0,
        video_fps=25.0,
        cooldown_frames=25,
    )
    frame = np.zeros((40, 120, 3), dtype=np.uint8)

    assert detector.update(0, 0.0, 1, "car", 1, frame, 40, 0, 60, 8) == []

    first = detector.update(1, 0.04, 1, "car", 1, frame, 40, 12, 60, 20)
    assert len(first) == 1
    assert first[0]["direction"] == "离去"
    assert first[0]["lane_id"] == 1

    suppressed = detector.update(2, 0.08, 1, "car", 1, frame, 40, 0, 60, 8)
    assert suppressed == []

    later = detector.update(40, 1.6, 1, "car", 1, frame, 40, 12, 60, 20)
    assert len(later) == 1


def test_turn_section_only_counts_first_arrival_crossing():
    detector = CrossSectionDetector(
        lines=[("北进口右转专用道", 0, 10, 100, 10, "到达", "离去")],
        homography=None,
        pixels_per_meter=10.0,
        video_fps=25.0,
        cooldown_frames=0,
    )
    frame = np.zeros((40, 120, 3), dtype=np.uint8)

    assert detector.update(0, 0.0, 9, "car", 1, frame, 40, 12, 60, 20) == []

    first = detector.update(1, 0.04, 9, "car", 1, frame, 40, 0, 60, 8)
    assert len(first) == 1
    assert first[0]["direction"] == "到达"

    back_swing = detector.update(2, 0.08, 9, "car", 1, frame, 40, 12, 60, 20)
    assert back_swing == []

    second_arrival = detector.update(3, 0.12, 9, "car", 1, frame, 40, 0, 60, 8)
    assert second_arrival == []
