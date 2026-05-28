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
        assert euc_s == pytest.approx(0.0, abs=1e-6)

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
        """平行且接近的轨迹欧氏距离归一化后应 < 0.0085."""
        g = _grouper()
        pts_a = [(500.0 + i*5, 200.0 + i*10) for i in range(8)]
        pts_b = [(505.0 + i*5, 200.0 + i*10) for i in range(8)]  # x偏移5px
        _, _, euc_dist = g._compute_similarity(pts_a, pts_b, 3840, 2160)
        # 5px / 3840 ≈ 0.0013 << 0.0085
        assert euc_dist < 0.0085

    def test_far_apart_trajectories_high_euc(self):
        """相距较远的轨迹归一化欧氏距离应 > 0.0085."""
        g = _grouper()
        pts_a = [(100.0, 200.0 + i*10) for i in range(8)]
        pts_b = [(2000.0, 200.0 + i*10) for i in range(8)]  # x相差1900px
        _, _, euc_dist = g._compute_similarity(pts_a, pts_b, 3840, 2160)
        # 1900 / 3840 ≈ 0.495 >> 0.0085
        assert euc_dist > 0.1
