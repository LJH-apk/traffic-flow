"""tests/trajectory/test_traj_grouper.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import pytest
from src.trajectory.traj_grouper import TrajGrouper


def _grouper():
    """返回使用默认阈值的 TrajGrouper 实例（不写文件）."""
    return TrajGrouper(
        cos_thresh=0.85,
        jsd_thresh=0.85,
        euc_thresh=0.85,
        interval_s=15.0,
        csv_path=None,
        excel_path=None,
    )


class TestComputeSimilarity:
    def test_identical_trajectories_score_one(self):
        g = _grouper()
        pts = [(100.0, 200.0), (110.0, 250.0), (120.0, 300.0)]
        cos_s, jsd_s, euc_s = g._compute_similarity(pts, pts, 3840, 2160)
        assert cos_s == pytest.approx(1.0, abs=1e-6)
        assert jsd_s == pytest.approx(1.0, abs=1e-6)
        assert euc_s == pytest.approx(1.0, abs=1e-6)  # 相同轨迹→距离0→相似度1

    def test_zero_pad_alignment(self):
        """短轨迹用零填充后，与自身仍应余弦相似度=1."""
        g = _grouper()
        pts_long  = [(float(i*10), float(i*20)) for i in range(10)]
        pts_short = pts_long[:5]
        cos_s, _, _ = g._compute_similarity(pts_long, pts_short, 3840, 2160)
        # 零填充后余弦相似度应接近1（方向基本一致）
        assert cos_s > 0.9

    def test_opposite_direction_has_low_cosine(self):
        g = _grouper()
        pts_a = [(0.0, float(i*10)) for i in range(5)]   # 向下移动
        pts_b = [(0.0, float(-i*10)) for i in range(5)]  # 向上移动
        cos_s, _, _ = g._compute_similarity(pts_a, pts_b, 3840, 2160)
        assert cos_s < 0.0   # 方向相反，余弦为负

    def test_nearby_parallel_trajectories_low_euc(self):
        """平行且接近的轨迹应有高欧氏相似度."""
        g = _grouper()
        pts_a = [(500.0 + i*5, 200.0 + i*10) for i in range(8)]
        pts_b = [(505.0 + i*5, 200.0 + i*10) for i in range(8)]  # x偏移5px
        _, _, euc_sim = g._compute_similarity(pts_a, pts_b, 3840, 2160)
        # 5px / diag ≈ 0.001，euc_sim ≈ 0.999
        assert euc_sim > 0.99

    def test_far_apart_trajectories_high_euc(self):
        """相距较远的轨迹应有低欧氏相似度."""
        g = _grouper()
        pts_a = [(100.0, 200.0 + i*10) for i in range(8)]
        pts_b = [(2000.0, 200.0 + i*10) for i in range(8)]  # x相差1900px
        _, _, euc_sim = g._compute_similarity(pts_a, pts_b, 3840, 2160)
        # 1900px / diag ≈ 0.431，euc_sim = 1/(1+0.431) ≈ 0.70
        assert euc_sim < 0.72


class TestInferTurnType:
    def test_straight_north_entrance(self):
        g = _grouper()
        pts = [(500.0, float(i * 50)) for i in range(10)]  # 向下移动（南向）
        assert g._infer_turn_type(pts, "motor", "北进口") == "straight"

    def test_left_turn_north_entrance(self):
        g = _grouper()
        # 大幅向左（西）偏移
        pts = [(500.0 - i * 50, 100.0 + i * 10) for i in range(10)]
        result = g._infer_turn_type(pts, "motor", "北进口")
        assert result in ("left_turn", "right_turn")

    def test_non_motor_lane_type(self):
        g = _grouper()
        pts = [(100.0, float(i * 30)) for i in range(5)]
        assert g._infer_turn_type(pts, "non-motor", "北进口") == "non_motor"

    def test_unknown_for_single_point(self):
        g = _grouper()
        assert g._infer_turn_type([(100.0, 200.0)], "motor", "北进口") == "unknown"


class TestGrouping:
    def test_identical_tracks_grouped_together(self):
        """完全相同的两条轨迹应归为同一组."""
        g = TrajGrouper(
            cos_thresh=0.85, jsd_thresh=0.85, euc_thresh=0.85,
            interval_s=15.0, csv_path=None, excel_path=None,
        )
        pts = [(float(i * 10), float(i * 20)) for i in range(15)]
        g.on_track_end(1, pts, "car", 1, "motor", "北进口")
        g.on_track_end(2, list(pts), "car", 1, "motor", "北进口")
        g._flush(0.0, 15.0)
        records = g.get_records()
        assert len(records) == 1
        assert records[0]["size"] == 2

    def test_far_apart_tracks_not_grouped(self):
        """空间上相距很远的轨迹不应归组."""
        g = TrajGrouper(
            cos_thresh=0.85, jsd_thresh=0.85, euc_thresh=0.85,
            interval_s=15.0, csv_path=None, excel_path=None,
        )
        pts_a = [(100.0, float(i * 20)) for i in range(15)]
        pts_b = [(3000.0, float(i * 20)) for i in range(15)]
        g.on_track_end(1, pts_a, "car", 1, "motor", "北进口")
        g.on_track_end(2, pts_b, "car", 2, "motor", "北进口")
        g._flush(0.0, 15.0)
        records = g.get_records()
        assert len(records) == 2
        assert all(r["size"] == 1 for r in records)

    def test_short_trajectory_discarded(self):
        """短于 min_frames 的轨迹应被丢弃."""
        g = TrajGrouper(
            cos_thresh=0.85, jsd_thresh=0.85, euc_thresh=0.85,
            interval_s=15.0, min_frames=8, csv_path=None, excel_path=None,
        )
        short_pts = [(float(i), float(i)) for i in range(5)]
        g.on_track_end(99, short_pts, "car", 1, "motor", "北进口")
        g._flush(0.0, 15.0)
        assert g.get_records() == []

    def test_turn_type_in_record(self):
        """分组记录应包含推断的转向类型."""
        g = TrajGrouper(
            cos_thresh=0.85, jsd_thresh=0.85, euc_thresh=0.85,
            interval_s=15.0, csv_path=None, excel_path=None,
        )
        pts = [(500.0, float(i * 50)) for i in range(15)]  # 直行
        g.on_track_end(1, pts, "car", 1, "motor", "北进口")
        g._flush(0.0, 15.0)
        assert g.get_records()[0]["turn_type"] == "straight"
