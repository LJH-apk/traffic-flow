# 车辆测速 + 单应矩阵自动标定 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给系统加上每辆车的实时速度显示（画面叠加）+ 每辆车的路径平均速度（CSV 输出），并把单应矩阵 H 的标定整合进 lane_calibration 流程，自动持久化。

**Architecture:** 新增 `SpeedEstimator`（滑动窗口、纯逻辑、无 IO）类负责所有测速。在 lane_calibration 里增加 H 标定步骤——优先用 ZebraDetector 自动检测，失败时调用 lane_annotator 备用流程让用户点 4 角 + 输入物理尺寸。tracker.py 主循环里调用 SpeedEstimator 每帧更新，画面叠加车道线 + 车辆框 + 速度标签，track 消失时写入 vehicle_stats.csv。

**Tech Stack:** Python 3.13, OpenCV (`cv2.perspectiveTransform`, `cv2.findHomography`), NumPy, ultralytics YOLO

---

## 文件改动清单

| 文件 | 类型 | 内容 |
|------|------|------|
| `src/config/settings.py` | 修改 | 新增 `VEHICLE_STATS_CSV_PATH`、`SPEED_WINDOW_FRAMES` |
| `src/cross_section/speed_estimator.py` | 新增 | `SpeedEstimator` 类 |
| `src/cross_section/lane_annotator.py` | 修改 | 新增 `annotate_homography()` 备用 H 标定 |
| `src/cross_section/lane_calibration.py` | 修改 | 标定流程末尾自动 H 计算 + 保存 + 读取 |
| `src/cross_section/counter.py` | 修改 | 接收外部 SpeedEstimator，删除内部 `_estimate_speed` |
| `src/trajectory/tracker.py` | 修改 | 集成 SpeedEstimator + 车道线可视化 + vehicle_stats.csv |

---

### Task 1: 新增配置常量

**Files:**
- Modify: `src/config/settings.py`（末尾追加）

- [ ] **Step 1: 在 settings.py 末尾追加新常量**

打开 `/Users/liujiahang/科研/交通流算法/src/config/settings.py`，在文件末尾追加：

```python

# ── 测速 ─────────────────────────────────────────────────────────────────────
SPEED_WINDOW_FRAMES = 15                              # 滑动窗口长度（帧），25fps 下约 0.6s
SPEED_MIN_DIST_M    = 0.1                             # 小于此距离视为静止（米）
SPEED_MIN_SAMPLES   = 5                               # 写入 vehicle_stats.csv 的最少采样数

VEHICLE_STATS_CSV_PATH = OUTPUT_DIR / "vehicle_stats.csv"
```

- [ ] **Step 2: 验证导入**

```bash
cd /Users/liujiahang/科研/交通流算法
python3 -c "from src.config.settings import SPEED_WINDOW_FRAMES, SPEED_MIN_DIST_M, SPEED_MIN_SAMPLES, VEHICLE_STATS_CSV_PATH; print(SPEED_WINDOW_FRAMES, SPEED_MIN_DIST_M, SPEED_MIN_SAMPLES, VEHICLE_STATS_CSV_PATH)"
```

期望输出：`15 0.1 5 /Users/liujiahang/科研/交通流算法/outputs/vehicle_stats.csv`

- [ ] **Step 3: 提交**

```bash
git add src/config/settings.py
git commit -m "feat(config): 新增测速常量（窗口长度、最小距离、采样阈值）"
```

---

### Task 2: 新增 SpeedEstimator 类

**Files:**
- Create: `src/cross_section/speed_estimator.py`

- [ ] **Step 1: 创建文件并写入完整实现**

创建 `/Users/liujiahang/科研/交通流算法/src/cross_section/speed_estimator.py`：

```python
"""
车辆测速模块（滑动窗口）。

A 模式：每帧返回当前 track 的瞬时速度（窗口首尾两点）
C 模式：track 消失时返回整段路径的平均/最大/最小速度

像素 → 世界坐标：
- 优先用单应矩阵 H（cv2.perspectiveTransform）
- H 不可用时退化到 PIXELS_PER_METER 兜底
"""
from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np


class SpeedEstimator:
    def __init__(
        self,
        homography: np.ndarray | None,
        fps: float,
        window: int = 15,
        pixels_per_meter: float = 85.0,
        min_dist_m: float = 0.1,
    ) -> None:
        self.H = homography
        self.fps = fps
        self.window = window
        self.ppm = pixels_per_meter
        self.min_dist_m = min_dist_m

        # tid -> deque[(frame_idx, wx, wy)]
        self._tracks: dict[int, deque] = {}
        # tid -> {'sum','count','max','min','first_frame','last_frame'}
        self._stats: dict[int, dict] = {}

    def _pix_to_world(self, cx: float, cy: float) -> tuple[float, float]:
        if self.H is not None:
            pt = np.float32([[[cx, cy]]])
            wp = cv2.perspectiveTransform(pt, self.H)[0][0]
            return float(wp[0]), float(wp[1])
        return cx / self.ppm, cy / self.ppm

    def update(self, frame_idx: int, tid: int, cx: float, cy: float) -> None:
        wx, wy = self._pix_to_world(cx, cy)
        hist = self._tracks.setdefault(tid, deque(maxlen=self.window))
        hist.append((frame_idx, wx, wy))

        # 累计统计
        v = self.instant_speed(tid)
        if v is None or v <= 0:
            return
        s = self._stats.setdefault(tid, {
            'sum': 0.0, 'count': 0,
            'max': 0.0, 'min': float('inf'),
            'first_frame': frame_idx,
            'last_frame': frame_idx,
        })
        s['sum'] += v
        s['count'] += 1
        s['max'] = max(s['max'], v)
        s['min'] = min(s['min'], v)
        s['last_frame'] = frame_idx

    def instant_speed(self, tid: int) -> float | None:
        hist = self._tracks.get(tid)
        if not hist or len(hist) < 2:
            return None
        f0, wx0, wy0 = hist[0]
        f1, wx1, wy1 = hist[-1]
        dist = math.hypot(wx1 - wx0, wy1 - wy0)
        if dist < self.min_dist_m:
            return 0.0
        dt = (f1 - f0) / self.fps
        if dt <= 0:
            return None
        return dist / dt * 3.6

    def get_stats(self, tid: int) -> dict | None:
        s = self._stats.get(tid)
        if not s or s['count'] == 0:
            return None
        return {
            'avg_kmh': s['sum'] / s['count'],
            'max_kmh': s['max'],
            'min_kmh': s['min'],
            'n_samples': s['count'],
            'first_frame': s['first_frame'],
            'last_frame': s['last_frame'],
        }

    def finalize(self, tid: int) -> dict | None:
        stats = self.get_stats(tid)
        self._tracks.pop(tid, None)
        self._stats.pop(tid, None)
        return stats
```

- [ ] **Step 2: 写入冒烟测试脚本**

创建临时测试 `/tmp/test_speed.py`：

```python
import sys; sys.path.insert(0, '/Users/liujiahang/科研/交通流算法')
import numpy as np
from src.cross_section.speed_estimator import SpeedEstimator

# 无 H：用 PPM=85 兜底，10m/s = 36km/h
est = SpeedEstimator(homography=None, fps=25.0, window=15, pixels_per_meter=85.0)

# 模拟一辆车 25 帧内移动 250 像素（≈2.94m）
for frame_idx in range(25):
    cx, cy = 1000 + frame_idx * 10, 1000   # 每帧前进 10px
    est.update(frame_idx, tid=1, cx=cx, cy=cy)

v = est.instant_speed(1)
assert v is not None and v > 0, f"瞬时速度异常: {v}"
print(f"瞬时速度: {v:.1f} km/h")

# 14 帧间隔（窗口首尾差 14 帧 = 0.56s），位移 14*10/85 ≈ 1.65m
# v ≈ 1.65 / 0.56 * 3.6 ≈ 10.6 km/h
assert 8 < v < 13, f"速度不在合理范围: {v}"

stats = est.finalize(1)
assert stats is not None
assert stats['n_samples'] > 0
print(f"统计: avg={stats['avg_kmh']:.1f} max={stats['max_kmh']:.1f} min={stats['min_kmh']:.1f} n={stats['n_samples']}")

# 静止情况：min_dist_m=0.1 → 应返回 0
est2 = SpeedEstimator(None, 25.0)
for frame_idx in range(20):
    est2.update(frame_idx, tid=2, cx=1000, cy=1000)
v_static = est2.instant_speed(2)
assert v_static == 0.0, f"静止应返回 0: {v_static}"
print(f"静止: {v_static} km/h ✓")

print("\nSpeedEstimator 测试通过")
```

- [ ] **Step 3: 跑测试**

```bash
python3 /tmp/test_speed.py
```

期望输出：
```
瞬时速度: 10.6 km/h
统计: avg=... max=... min=... n=...
静止: 0.0 km/h ✓

SpeedEstimator 测试通过
```

- [ ] **Step 4: 提交**

```bash
git add src/cross_section/speed_estimator.py
git commit -m "feat(speed): 新增 SpeedEstimator 滑动窗口测速器"
```

- [ ] **Step 5: 清理临时测试**

```bash
rm /tmp/test_speed.py
```

---

### Task 3: 在 lane_annotator 加备用 H 标定子流程

**Files:**
- Modify: `src/cross_section/lane_annotator.py`

- [ ] **Step 1: 阅读现有 ZoomPanAnnotator 的鼠标事件结构**

```bash
grep -n "_mouse_cb\|_render\|EVENT_LBUTTON\|self.lanes" /Users/liujiahang/科研/交通流算法/src/cross_section/lane_annotator.py | head -20
```

确认现有标注工具是 `ZoomPanAnnotator` 类，鼠标回调是 `_mouse_cb`。

- [ ] **Step 2: 在 lane_annotator.py 末尾追加备用 H 标定函数**

在 `/Users/liujiahang/科研/交通流算法/src/cross_section/lane_annotator.py` 末尾追加（在 `if __name__ == "__main__":` 块之前）：

```python


def annotate_homography(
    image: np.ndarray,
    win_w: int = 1280,
    win_h: int = 800,
) -> dict | None:
    """备用单应矩阵标定：用户点 4 个角点 + 输入物理长宽。

    复用 ZoomPanAnnotator 的视口缩放/平移，只把车道线标注改成 4 点矩形标注。

    Returns:
        {'src_pts': [(x,y)*4], 'world_w': float, 'world_h': float, 'H': 3x3 ndarray}
        或 None（用户取消）
    """
    # 用 1 条 "车道" 复用现有标注器：用户点 4 个点 = 矩形 4 角
    print("\n[H 标定] 请在弹出窗口中按顺序点 4 个角点（左上 → 右上 → 右下 → 左下）")
    print("           可以是斑马线一条条纹、停止线一段、或任何已知尺寸的矩形")
    result = annotate(image, n_lanes=1)
    if result is None or 1 not in result or len(result[1]) != 4:
        print("[H 标定] 未获得 4 个角点，取消")
        return None

    src_pts = np.float32(result[1])
    try:
        world_w = float(input("\n请输入这个矩形的宽度（米，沿水平方向）: "))
        world_h = float(input("请输入这个矩形的高度（米，沿垂直方向）: "))
    except ValueError:
        print("[H 标定] 输入无效，取消")
        return None

    dst_pts = np.float32([
        [0,       0      ],
        [world_w, 0      ],
        [world_w, world_h],
        [0,       world_h],
    ])
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("[H 标定] 单应矩阵计算失败")
        return None

    return {
        'src_pts': [tuple(p) for p in result[1]],
        'world_w': world_w,
        'world_h': world_h,
        'H': H,
    }
```

- [ ] **Step 3: 验证 import 不破坏**

```bash
cd /Users/liujiahang/科研/交通流算法
python3 -c "from src.cross_section.lane_annotator import annotate, annotate_homography; print('OK')"
```

期望输出：`OK`

- [ ] **Step 4: 提交**

```bash
git add src/cross_section/lane_annotator.py
git commit -m "feat(annotator): 新增 annotate_homography 备用单应矩阵标定"
```

---

### Task 4: 在 lane_calibration 加自动 H 计算 + 持久化

**Files:**
- Modify: `src/cross_section/lane_calibration.py`

- [ ] **Step 1: 在文件顶部 import 处加 ZebraDetector 和 annotate_homography**

打开 `/Users/liujiahang/科研/交通流算法/src/cross_section/lane_calibration.py`，找到现有的 import 区域。当前已有：

```python
from src.cross_section.lane_annotator import annotate as run_annotator
from src.cross_section.lane_detector import build_background
from src.cross_section.lane_lighting import decide_preset
```

改为：

```python
from src.cross_section.lane_annotator import (
    annotate as run_annotator,
    annotate_homography,
)
from src.cross_section.lane_detector import build_background
from src.cross_section.lane_lighting import decide_preset
from src.cross_section.zebra_detector import ZebraDetector
```

- [ ] **Step 2: 在 CalibrationData dataclass 加 homography 字段**

找到 `@dataclass class CalibrationData:` 定义，在最后字段后追加：

```python
    homography: np.ndarray | None = None       # 3x3 像素→世界坐标
    homography_method: str = "fallback_ppm"    # auto_zebra / manual / fallback_ppm
```

整个 dataclass 完整长这样：

```python
@dataclass
class CalibrationData:
    """运行时持有的标定信息。"""
    entrance: str
    lanes: dict[int, list[tuple[int, int]]]
    ref_image: np.ndarray
    metadata: dict
    start_time: datetime | None = None
    end_time:   datetime | None = None
    lighting_preset: str = 'off_peak'
    homography: np.ndarray | None = None
    homography_method: str = "fallback_ppm"
```

- [ ] **Step 3: 加 H 自动/备用标定函数**

在 `save_calibration` 函数之前插入（也就是 `find_calibration` 之后）：

```python
def _compute_homography(bg_image: np.ndarray, frame: np.ndarray) -> tuple[np.ndarray | None, str]:
    """优先用 ZebraDetector 自动算 H，失败则进入备用标注流程。

    Returns:
        (H, method)；method ∈ {"auto_zebra", "manual", "fallback_ppm"}
    """
    print("[H] 尝试自动检测斑马线...")
    zresult = ZebraDetector().detect(bg_image)
    if zresult is not None:
        H, n_stripes, _rects = zresult
        print(f"[H] ✓ 自动检测成功（{n_stripes} 条条纹）")
        return H, "auto_zebra"

    print("[H] ⚠ 自动检测失败，进入备用人工标定")
    manual = annotate_homography(bg_image)
    if manual is None:
        print("[H] ✗ 备用标定也取消，将使用 PIXELS_PER_METER 兜底")
        return None, "fallback_ppm"
    return manual['H'], "manual"


def _save_homography(cal_dir: Path, H: np.ndarray, method: str, n_stripes: int = 0):
    """保存 H 矩阵到 homography.json。"""
    data = {
        'H': H.tolist(),
        'method': method,
        'n_stripes': n_stripes,
        'calibration_date': datetime.now().isoformat(),
    }
    (cal_dir / 'homography.json').write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )


def _load_homography(cal_dir: Path) -> tuple[np.ndarray | None, str]:
    """从 homography.json 读取 H。失败返回 (None, 'fallback_ppm')。"""
    path = cal_dir / 'homography.json'
    if not path.exists():
        return None, 'fallback_ppm'
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return np.array(data['H'], dtype=np.float64), data.get('method', 'unknown')
    except Exception:
        return None, 'fallback_ppm'
```

- [ ] **Step 4: 修改 find_calibration 让它也读 homography.json**

找到 `find_calibration` 函数末尾的 `return {'lanes': lanes, 'ref_image': ref_image, 'metadata': metadata}`，改为：

```python
    H, method = _load_homography(cal_dir)
    return {
        'lanes': lanes,
        'ref_image': ref_image,
        'metadata': metadata,
        'homography': H,
        'homography_method': method,
    }
```

- [ ] **Step 5: 修改 get_calibration 让它在两种路径下都填充 H**

找到 `get_calibration` 函数。**先处理"已有标定"分支**：

找到这段：
```python
    existing = find_calibration(entrance)
    if existing is not None:
        print(f"✓ 已加载 {entrance} 的标定（{len(existing['lanes'])} 条车道线）")
        # 顺便判断光照
        first_frame, _ = build_background(str(video_path), target_frame=100)
        preset = decide_preset(start_time, first_frame)
        return CalibrationData(
            entrance=entrance,
            lanes=existing['lanes'],
            ref_image=existing['ref_image'],
            metadata=existing['metadata'],
            start_time=start_time,
            end_time=end_time,
            lighting_preset=preset,
        )
```

替换为：

```python
    existing = find_calibration(entrance)
    if existing is not None:
        print(f"✓ 已加载 {entrance} 的标定（{len(existing['lanes'])} 条车道线）")
        first_frame, _ = build_background(str(video_path), target_frame=100)
        preset = decide_preset(start_time, first_frame)
        H = existing.get('homography')
        method = existing.get('homography_method', 'fallback_ppm')
        if H is None:
            print(f"⚠ {entrance} 暂无 H 矩阵，使用 PIXELS_PER_METER 兜底")
        else:
            print(f"✓ H 矩阵已加载（method={method}）")
        return CalibrationData(
            entrance=entrance,
            lanes=existing['lanes'],
            ref_image=existing['ref_image'],
            metadata=existing['metadata'],
            start_time=start_time,
            end_time=end_time,
            lighting_preset=preset,
            homography=H,
            homography_method=method,
        )
```

**然后处理"新建标定"分支**：

找到这段：
```python
    metadata = {
        'entrance':         entrance,
        'calibration_date': datetime.now().isoformat(),
        'source_video':     video_path.name,
        'source_frame':     100,
        'lane_count':       sum(1 for v in lanes.values() if v),
    }
    cal_dir = save_calibration(entrance, lanes, bg_bgr, metadata)
    print(f"✓ 已保存到 {cal_dir}")

    preset = decide_preset(start_time, first_frame)
    return CalibrationData(
        entrance=entrance,
        lanes=lanes,
        ref_image=bg_bgr,
        metadata=metadata,
        start_time=start_time,
        end_time=end_time,
        lighting_preset=preset,
    )
```

替换为：

```python
    metadata = {
        'entrance':         entrance,
        'calibration_date': datetime.now().isoformat(),
        'source_video':     video_path.name,
        'source_frame':     100,
        'lane_count':       sum(1 for v in lanes.values() if v),
    }
    cal_dir = save_calibration(entrance, lanes, bg_bgr, metadata)
    print(f"✓ 车道线已保存到 {cal_dir}")

    # 标定单应矩阵
    H, method = _compute_homography(bg_bgr, first_frame)
    if H is not None:
        _save_homography(cal_dir, H, method)
        print(f"✓ H 矩阵已保存（method={method}）")

    preset = decide_preset(start_time, first_frame)
    return CalibrationData(
        entrance=entrance,
        lanes=lanes,
        ref_image=bg_bgr,
        metadata=metadata,
        start_time=start_time,
        end_time=end_time,
        lighting_preset=preset,
        homography=H,
        homography_method=method,
    )
```

- [ ] **Step 6: 更新 main CLI 输出 H 信息**

找到 `main()` 函数的最后那段打印：

```python
    print(f"\n=== 标定信息 ===")
    print(f"进口:       {cal.entrance}")
    print(f"时段:       {cal.start_time} → {cal.end_time}")
    print(f"光照预设:    {cal.lighting_preset}")
    print(f"车道线数:    {len(cal.lanes)}")
    for lid, pts in sorted(cal.lanes.items()):
        print(f"  线{lid}: {len(pts)} 点")
```

末尾追加：

```python
    print(f"H 矩阵:     {'已加载（' + cal.homography_method + '）' if cal.homography is not None else '无（兜底 PPM）'}")
```

- [ ] **Step 7: 验证（已有标定加载）**

```bash
cd /Users/liujiahang/科研/交通流算法
python3 -m src.cross_section.lane_calibration --video "南进口_20260420075959至20260420081500.mp4" --no-annotate
```

期望看到 "H 矩阵: 无（兜底 PPM）" 因为南进口还没存 H。

- [ ] **Step 8: 给南进口手动补一个 H（自动检测）做测试**

```bash
cd /Users/liujiahang/科研/交通流算法
python3 -c "
import sys; sys.path.insert(0, '.')
import cv2
from pathlib import Path
from src.cross_section.zebra_detector import ZebraDetector
from src.cross_section.lane_calibration import _save_homography

bg = cv2.imread('calibrations/南进口/ref.jpg')
result = ZebraDetector().detect(bg)
if result is None:
    print('斑马线自动检测失败')
else:
    H, n, _ = result
    print(f'检测到 {n} 条条纹')
    _save_homography(Path('calibrations/南进口'), H, 'auto_zebra', n)
    print('已保存 calibrations/南进口/homography.json')
"
```

- [ ] **Step 9: 再次跑加载，验证 H 正确读取**

```bash
python3 -m src.cross_section.lane_calibration --video "南进口_20260420075959至20260420081500.mp4" --no-annotate
```

期望看到 "H 矩阵: 已加载（auto_zebra）"。

- [ ] **Step 10: 提交**

```bash
git add src/cross_section/lane_calibration.py calibrations/南进口/homography.json
git commit -m "feat(calibration): 标定流程末尾自动算 H 并持久化"
```

---

### Task 5: 重构 counter.py 使用外部 SpeedEstimator

**Files:**
- Modify: `src/cross_section/counter.py`

- [ ] **Step 1: 在 CrossSectionDetector.__init__ 加入 speed_estimator 参数**

找到 `__init__` 方法：

```python
    def __init__(
        self,
        lines: list[tuple[str, int, int, int, int, str, str]],
        homography: np.ndarray | None,
        pixels_per_meter: float,
        video_fps: float,
    ) -> None:
        self._lines = lines
        self._H = homography
        self._ppm = pixels_per_meter
        self._fps = video_fps

        # track_id -> deque of (frame_idx, wx, wy)，保留最近15帧用于速度估算
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=15))

        # (track_id, line_name) -> 上一帧叉积符号 (-1/0/+1)
```

改为（新增 `speed_estimator` 可选参数，向后兼容）：

```python
    def __init__(
        self,
        lines: list[tuple[str, int, int, int, int, str, str]],
        homography: np.ndarray | None,
        pixels_per_meter: float,
        video_fps: float,
        speed_estimator: "SpeedEstimator | None" = None,
    ) -> None:
        self._lines = lines
        self._H = homography
        self._ppm = pixels_per_meter
        self._fps = video_fps
        self._speed_estimator = speed_estimator

        # track_id -> deque of (frame_idx, wx, wy)，仅在没有外部 SpeedEstimator 时使用
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=15))

        # (track_id, line_name) -> 上一帧叉积符号 (-1/0/+1)
```

- [ ] **Step 2: 在文件顶部加 TYPE_CHECKING import**

找到 `counter.py` 最顶部的 import 区，在 `import math` 之前加：

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.cross_section.speed_estimator import SpeedEstimator
```

- [ ] **Step 3: 改 _estimate_speed 优先用外部 SpeedEstimator**

找到 `_estimate_speed` 方法：

```python
    def _estimate_speed(self, tid: int) -> float:
        """根据历史世界坐标估算速度（km/h）。"""
        hist = self._history[tid]
        if len(hist) < 2:
            return 0.0
        f0, wx0, wy0 = hist[0]
        f1, wx1, wy1 = hist[-1]
        dist_m = math.hypot(wx1 - wx0, wy1 - wy0)
        time_s = (f1 - f0) / self._fps
        if time_s <= 0:
            return 0.0
        return round(dist_m / time_s * 3.6, 1)
```

替换为：

```python
    def _estimate_speed(self, tid: int) -> float:
        """估算瞬时速度（km/h）。

        优先用外部 SpeedEstimator（保持算法一致性）；
        否则用内部 _history（向后兼容）。
        """
        if self._speed_estimator is not None:
            v = self._speed_estimator.instant_speed(tid)
            return round(v, 1) if v is not None else 0.0

        hist = self._history[tid]
        if len(hist) < 2:
            return 0.0
        f0, wx0, wy0 = hist[0]
        f1, wx1, wy1 = hist[-1]
        dist_m = math.hypot(wx1 - wx0, wy1 - wy0)
        time_s = (f1 - f0) / self._fps
        if time_s <= 0:
            return 0.0
        return round(dist_m / time_s * 3.6, 1)
```

- [ ] **Step 4: 验证 import 不破坏**

```bash
cd /Users/liujiahang/科研/交通流算法
python3 -c "from src.cross_section.counter import CrossSectionDetector; print('OK')"
```

- [ ] **Step 5: 提交**

```bash
git add src/cross_section/counter.py
git commit -m "refactor(counter): 接收外部 SpeedEstimator 保持测速口径一致"
```

---

### Task 6: 集成进 tracker.py（核心）

**Files:**
- Modify: `src/trajectory/tracker.py`

- [ ] **Step 1: 在 tracker.py 顶部更新 imports**

找到 `from src.config.settings import (` 这块，把它扩展为：

```python
from src.config.settings import (
    VEHICLE_CLASSES,
    DEVICE,
    CONF_THRESH,
    VIDEO_PATH,
    OUTPUT_DIR,
    MODEL_DIR,
    MODEL_NAME,
    TRAJ_SAMPLE_FPS,
    TRAJ_CSV_PATH,
    SECTION_LINES,
    HOMOGRAPHY_MATRIX,
    PIXELS_PER_METER,
    CROSS_SECTION_CSV_PATH,
    SPEED_WINDOW_FRAMES,
    SPEED_MIN_DIST_M,
    SPEED_MIN_SAMPLES,
    VEHICLE_STATS_CSV_PATH,
)
```

然后在已有 `from src.cross_section.*` 导入区追加：

```python
from src.cross_section.speed_estimator import SpeedEstimator
from src.cross_section.lane_calibration import get_calibration
```

- [ ] **Step 2: 在 TrajectoryTracker 类里加车道线绘制辅助方法**

在 `class TrajectoryTracker` 类的方法定义区（在 `def run(` 之前），加入两个辅助方法：

```python
    @staticmethod
    def _build_lane_overlay(shape: tuple[int, int],
                            lanes: dict[int, list[tuple[int, int]]]) -> np.ndarray:
        """根据车道线标注点构建半透明 overlay（启动时一次性生成）。"""
        from scipy.interpolate import UnivariateSpline
        h, w = shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        lane_colors = {
            1: (0, 255,   0),
            2: (0, 200, 255),
            3: (255,  80, 200),
            4: (255, 200,  0),
        }
        for lid, pts in sorted(lanes.items()):
            if len(pts) < 4:
                continue
            arr = np.array(pts, dtype=np.float64)
            ys, xs = arr[:, 1], arr[:, 0]
            order = np.argsort(ys)
            ys_u, xs_u = ys[order], xs[order]
            try:
                sp = UnivariateSpline(ys_u, xs_u, k=3, s=200 * len(ys_u))
                y_lo, y_hi = int(ys_u[0]), int(ys_u[-1])
                ys_e = np.arange(y_lo, y_hi + 1)
                xs_e = np.clip(sp(ys_e), 0, w - 1).astype(np.int32)
                curve = np.stack([xs_e, ys_e], axis=1).reshape(-1, 1, 2)
                color = lane_colors.get(lid, (200, 200, 200))
                cv2.polylines(overlay, [curve], False, color, 4)
            except Exception:
                pass
        return overlay

    @staticmethod
    def _assign_lane(cx: float, cy: float,
                     lanes: dict[int, list[tuple[int, int]]]) -> int | None:
        """根据 bbox 底部中心点判断车辆属于哪条车道（在两条相邻线之间）。

        简化实现：用每条线的样条在 cy 处求 x，cx 落在哪两条线之间则属于哪个车道。
        """
        from scipy.interpolate import UnivariateSpline
        xs_at_cy = []
        for lid, pts in sorted(lanes.items()):
            if len(pts) < 4:
                continue
            arr = np.array(pts, dtype=np.float64)
            ys, xs = arr[:, 1], arr[:, 0]
            order = np.argsort(ys)
            ys_u, xs_u = ys[order], xs[order]
            try:
                sp = UnivariateSpline(ys_u, xs_u, k=3, s=200 * len(ys_u))
                if cy < ys_u[0] or cy > ys_u[-1]:
                    return None
                xs_at_cy.append((lid, float(sp(cy))))
            except Exception:
                return None
        if len(xs_at_cy) < 2:
            return None
        xs_at_cy.sort(key=lambda t: t[1])
        for i in range(len(xs_at_cy) - 1):
            if xs_at_cy[i][1] <= cx <= xs_at_cy[i + 1][1]:
                return i + 1
        return None
```

- [ ] **Step 3: 在 run() 方法开头加入标定加载**

找到 `run()` 方法里下面这段：

```python
        # ── 断面检测 + 车道线检测初始化（Position A）──────────────────────────
        # 优先用 settings 中预设的 H；否则尝试从第一帧自动检测斑马线
        _H = HOMOGRAPHY_MATRIX
        _zebra_rects: list[tuple[int, int, int, int]] = []  # 条纹矩形（可视化用）
        _lane_lines: list = []                               # 车道线段（可视化用）

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        _ret_ref, _ref_frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # rewind

        if _ret_ref:
            # 斑马线检测
            if _H is None:
                _zresult = ZebraDetector().detect(_ref_frame)
                if _zresult is not None:
                    _H, _n, _zebra_rects = _zresult
                    print(f"[断面] 自动检测斑马线成功，{_n}条纹，H已计算")
                else:
                    print("[断面] 斑马线自动检测失败，使用 PIXELS_PER_METER 兜底")
            # 车道线检测（一次性，零运行时开销）
            _lane_lines = LaneDetector().detect(_ref_frame)
            print(f"[车道] 检测到 {len(_lane_lines)} 条车道线段")

        section_det = CrossSectionDetector(SECTION_LINES, _H, PIXELS_PER_METER, video_fps)
```

替换为：

```python
        # ── 标定加载 + 测速器初始化 ──────────────────────────────────────────
        cal = get_calibration(video_path)
        if cal is None:
            print("[标定] ⚠ 加载失败，全部退化到 settings 默认值")
            _H = HOMOGRAPHY_MATRIX
            _lane_overlay = None
            _lanes = {}
        else:
            _H = cal.homography
            _lanes = cal.lanes
            print(f"[标定] ✓ {cal.entrance} | 车道线 {len(_lanes)} 条 | H={cal.homography_method} | 光照={cal.lighting_preset}")

            # 一次性构建车道线 overlay（与帧同尺寸）
            _lane_overlay = self._build_lane_overlay(
                (meta["height"], meta["width"]), _lanes
            )

        # 测速器
        speed_est = SpeedEstimator(
            homography=_H,
            fps=video_fps,
            window=SPEED_WINDOW_FRAMES,
            pixels_per_meter=PIXELS_PER_METER,
            min_dist_m=SPEED_MIN_DIST_M,
        )

        # 断面检测器（复用同一个 SpeedEstimator 保持口径一致）
        section_det = CrossSectionDetector(
            SECTION_LINES, _H, PIXELS_PER_METER, video_fps,
            speed_estimator=speed_est,
        )
```

注意：删除了原来的 `_zebra_rects = []` 和 `_lane_lines = []` 与对应的检测调用——它们被 cal.lanes / lane_overlay 取代。

- [ ] **Step 4: 在 run() 开始处打开 vehicle_stats.csv**

找到这段：

```python
        cross_path = Path(CROSS_SECTION_CSV_PATH)
        cross_path.parent.mkdir(parents=True, exist_ok=True)
```

紧跟着追加：

```python
        stats_path = Path(VEHICLE_STATS_CSV_PATH)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_fh = open(stats_path, 'w', newline='', encoding='utf-8')
        stats_writer = csv.writer(stats_fh)
        stats_writer.writerow([
            "track_id", "first_frame", "last_frame", "lane_id",
            "avg_speed_kmh", "max_speed_kmh", "min_speed_kmh", "n_samples",
        ])

        # 记录每个 track 最后已知的车道号（消失时写入 stats）
        last_known_lane: dict[int, int | None] = {}
        active_tids: set[int] = set()
```

- [ ] **Step 5: 在主循环里调用 SpeedEstimator.update + 绘制 overlay + 速度标签**

找到主循环里的 `for det in ...:` 区域（处理每个检测框的部分）。在每个 track 的处理点，找到形如：

```python
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
```

或类似的中心点计算（注意 `cy` 应该是 bbox 底部）。在每个 track 处理时增加：

```python
# 测速更新（bbox 底部中心）
cy_bottom = float(y2)
cx_center = float((x1 + x2) / 2)
speed_est.update(frame_idx, tid, cx_center, cy_bottom)

# 车道归属
lane_id = self._assign_lane(cx_center, cy_bottom, _lanes) if _lanes else None
if lane_id is not None:
    last_known_lane[tid] = lane_id

# 瞬时速度
v_now = speed_est.instant_speed(tid)
speed_text = f"{v_now:.0f}km/h" if v_now is not None else ""
lane_text = f"L{lane_id}" if lane_id is not None else ""
overlay_label = " ".join(x for x in [lane_text, speed_text] if x)
if overlay_label:
    put_text(annotated, overlay_label, (int(x1), int(y1) - 10), color=(0, 255, 255))

active_tids.add(tid)
```

注意：上面假设循环里已有 `x1, y1, x2, y2`、`tid`、`frame_idx`、`annotated`（已绘制 bbox 的帧）变量。如果命名不同需要调整。可以先 grep 看实际变量名：

```bash
grep -n "for det\|for tid\|annotated\|frame_idx\|track\.id\|result\.boxes" /Users/liujiahang/科研/交通流算法/src/trajectory/tracker.py | head -20
```

- [ ] **Step 6: 在主循环外（每帧结束前）叠加车道线 overlay**

在主循环的 `writer.write(annotated)` 之前，加入：

```python
        if _lane_overlay is not None:
            annotated = cv2.addWeighted(annotated, 1.0, _lane_overlay, 0.5, 0)
```

- [ ] **Step 7: 在主循环外（每帧后）检测消失的 track 并写 stats**

在主循环里，每帧结束的位置（紧靠着 `writer.write` 之前或之后，但要在下一帧 active_tids 被覆盖之前），加入：

```python
        # 检测消失的 track，写 stats
        prev_tids = set(speed_est._tracks.keys()) | set(last_known_lane.keys())
        expired = prev_tids - active_tids
        for tid in expired:
            stats = speed_est.finalize(tid)
            if stats and stats['n_samples'] >= SPEED_MIN_SAMPLES:
                stats_writer.writerow([
                    tid,
                    stats['first_frame'],
                    stats['last_frame'],
                    last_known_lane.get(tid),
                    round(stats['avg_kmh'], 1),
                    round(stats['max_kmh'], 1),
                    round(stats['min_kmh'], 1),
                    stats['n_samples'],
                ])
            last_known_lane.pop(tid, None)
        active_tids.clear()
```

- [ ] **Step 8: 主循环结束后 finalize 所有剩余 track + 关闭文件**

找到主循环结束的位置（通常在 `cap.release()` 之前），加入：

```python
        # finalize 所有还活着的 track
        for tid in list(speed_est._tracks.keys()):
            stats = speed_est.finalize(tid)
            if stats and stats['n_samples'] >= SPEED_MIN_SAMPLES:
                stats_writer.writerow([
                    tid,
                    stats['first_frame'],
                    stats['last_frame'],
                    last_known_lane.get(tid),
                    round(stats['avg_kmh'], 1),
                    round(stats['max_kmh'], 1),
                    round(stats['min_kmh'], 1),
                    stats['n_samples'],
                ])
        stats_fh.close()
        print(f"[Stats] vehicle_stats.csv 写出: {stats_path}")
```

- [ ] **Step 9: 调整 _START_FRAME 和 _END_FRAME 用于测试**

修改 tracker.py 顶部：

```python
_START_FRAME = 0
_END_FRAME   = 1000
_OUTPUT_VID  = OUTPUT_DIR / "trajectory.mp4"
```

并把默认视频改为北进口：

```python
# 测试用：默认视频
_TEST_VIDEO = Path("北进口_20260420075959至20260420081500.mp4")
```

然后在 `run()` 默认参数里把 `video_path=VIDEO_PATH` 改为 `video_path=_TEST_VIDEO`。

- [ ] **Step 10: 端到端跑 1000 帧**

```bash
cd /Users/liujiahang/科研/交通流算法
python3 -u src/trajectory/tracker.py 2>&1 | tail -50
```

期望：
- 打印 `[标定] ✓ 北进口 ...` 标定加载成功
- 运行 1000 帧无 Python 报错
- 生成 `outputs/trajectory.mp4`，画面上能看到：彩色车道线、车辆框 + L? + 速度
- 生成 `outputs/vehicle_stats.csv`，每辆出现至少 5 帧的车一行

- [ ] **Step 11: 抽查输出**

```bash
ls -la outputs/trajectory.mp4 outputs/vehicle_stats.csv
head -5 outputs/vehicle_stats.csv
wc -l outputs/vehicle_stats.csv
```

期望：trajectory.mp4 > 10MB，vehicle_stats.csv 包含若干行车辆数据。

- [ ] **Step 12: 提交**

```bash
git add src/trajectory/tracker.py
git commit -m "feat(tracker): 集成 SpeedEstimator + 车道线可视化 + vehicle_stats.csv 输出"
```

---

### Task 7: 视觉验证

**Files:**
- 无新增/修改

- [ ] **Step 1: 打开输出视频检查**

```bash
open /Users/liujiahang/科研/交通流算法/outputs/trajectory.mp4
```

检查：
- 车道线在画面上可见，颜色分明（4 条线）
- 车辆框上有车道号 + 速度（实时变化）
- 速度数值合理（早高峰场景预期 20-60 km/h 居多）
- 远端车辆没有出现明显错误车道归属

- [ ] **Step 2: 检查 vehicle_stats.csv**

```bash
cat outputs/vehicle_stats.csv | head -20
```

期望：
- 至少 10+ 条记录
- avg/max/min 速度数值合理
- 大多数 lane_id 非 null
- n_samples 都 >= 5

- [ ] **Step 3: 视觉验证通过则标记完成**

如果不通过，回到对应 Task 修复，重跑 Task 6 Step 10-11 验证。

---

## 执行顺序总结

```
Task 1 (settings)
   ↓
Task 2 (speed_estimator) ─┐
                          ├──→ Task 5 (counter refactor) ──→ Task 6 (tracker)
Task 3 (annotator H 备用) ─┤                                       ↓
                          ├──→ Task 4 (calibration H 整合) ─→     Task 7 (视觉验证)
ZebraDetector (已存在)    ─┘
```

每个 Task 完成后必须能独立验证，不会破坏 tracker.py（即使中间 Task 还没集成进 tracker.py，tracker.py 也能照常跑）。

---

## 自检

**Spec 覆盖**：
- ✓ SpeedEstimator 滑动窗口 → Task 2
- ✓ A 模式（瞬时速度）→ Task 6 Step 5
- ✓ C 模式（路径统计）→ Task 2 + Task 6 Step 7-8
- ✓ 单应矩阵自动标定 → Task 4
- ✓ 备用人工 H 标定 → Task 3
- ✓ homography.json 持久化 → Task 4 Step 3
- ✓ CalibrationData 扩展 → Task 4 Step 2
- ✓ tracker.py 集成 → Task 6
- ✓ vehicle_stats.csv 输出 → Task 6
- ✓ 车道线可视化 → Task 6 Step 2/6
- ✓ counter.py 复用 SpeedEstimator → Task 5
- ✓ 1000 帧北进口验证 → Task 6 Step 10 + Task 7

**类型一致性**：
- `SpeedEstimator.instant_speed` 在 Task 2、Task 5、Task 6 都用 `float | None` 返回类型 ✓
- `CalibrationData.homography` 在 Task 4 定义，Task 6 引用 ✓
- `lanes: dict[int, list[tuple[int, int]]]` 类型在 lane_annotator、lane_calibration、tracker 一致 ✓

**潜在风险**：
- tracker.py 现有变量命名（`tid`、`frame_idx`、`annotated` 等）在不同地方可能不同，Task 6 Step 5 已提示先 grep 确认
- ZebraDetector 在北进口可能识别失败（如果是黑白监控背景或斑马线被遮挡），会进入备用流程——这是预期行为
