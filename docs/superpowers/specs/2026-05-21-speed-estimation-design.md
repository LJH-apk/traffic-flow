# 车辆测速 + 单应矩阵自动标定 设计

**日期**：2026-05-21
**作者**：W + Claude
**目标**：实现 A 模式（每帧实时速度，画面叠加）+ C 模式（路径平均速度，CSV 输出），并整合单应矩阵 H 的自动/备用标定流程到 `lane_calibration`。

---

## 背景

项目已有的速度逻辑：
- `src/cross_section/counter.py` 的 `_estimate_speed`：用 `_history[tid]` 取首尾两点算瞬时速度，**仅在过断面线时触发**
- 单应矩阵 `HOMOGRAPHY_MATRIX` 当前为 `None`，依赖 `PIXELS_PER_METER=85` 兜底
- `ZebraDetector.detect()` 已实现斑马线自动检测 + 单应矩阵计算（GB 5768.3）

不足：
1. 实时显示无速度
2. 每辆车没有汇总性的"平均速度"数据
3. H 没有持久化（每次启动都要重算或用 fallback）

---

## 总体架构

```
src/cross_section/
  speed_estimator.py       ← 新增：滑动窗口测速器
  lane_calibration.py      ← 修改：标定流程末尾加自动 H 计算 + 持久化
  zebra_detector.py        ← 不变（直接复用）
  counter.py               ← 修改：CrossSectionDetector 改用 SpeedEstimator
  lane_annotator.py        ← 修改：增加备用 H 标定子流程（4 角点 + 物理尺寸输入）
```

---

## 单应矩阵 H 标定流程

### 触发时机

在 `lane_calibration.get_calibration()` 流程中，**车道线交互标注完成后**自动运行：

```
车道线标注完成
  ↓
ZebraDetector.detect(背景图)
  ↓
├─ 成功（≥2 条条纹）：H = 自动计算结果，记 method="auto_zebra"
└─ 失败：弹出提示，调用 lane_annotator 的备用 H 标定子流程
         （用户点 4 角点 + 输入矩形物理长宽），记 method="manual"
  ↓
保存到 calibrations/{进口}/homography.json
```

### 保存格式

`calibrations/{进口}/homography.json`：

```json
{
  "H": [[h11, h12, h13], [h21, h22, h23], [h31, h32, h33]],
  "method": "auto_zebra",
  "n_stripes": 5,
  "pixels_per_meter_fallback": 85.0,
  "calibration_date": "2026-05-21T13:31:09"
}
```

### CalibrationData 扩展

`lane_calibration.CalibrationData` 新增字段：

```python
homography: np.ndarray | None = None       # 3x3 矩阵
homography_method: str = "fallback_ppm"    # 自动/手动/兜底
```

---

## 速度估计模块

### 接口：`src/cross_section/speed_estimator.py`

```python
class SpeedEstimator:
    """滑动窗口测速器。

    A 模式：每帧返回当前 track 的瞬时速度（窗口首尾两点）。
    C 模式：track 消失时返回整段路径的平均/最大/最小速度。
    """

    def __init__(self,
                 homography: np.ndarray | None,
                 fps: float,
                 window: int = 15,
                 pixels_per_meter: float = 85.0,
                 min_dist_m: float = 0.1):
        ...

    def update(self, frame_idx: int, tid: int, cx: float, cy: float) -> None:
        """每帧每个 track 调用一次，记录当前像素位置。"""

    def instant_speed(self, tid: int) -> float | None:
        """A 模式：当前滑动窗口内的瞬时速度（km/h）。
        窗口未填满或距离过小时返回 None。"""

    def get_stats(self, tid: int) -> dict | None:
        """C 模式：返回 track 整段路径的统计。
        {'avg_kmh', 'max_kmh', 'min_kmh', 'n_samples', 'first_frame', 'last_frame'}
        """

    def finalize(self, tid: int) -> dict | None:
        """track 消失时调用，返回 get_stats(tid) 并清理内存。"""
```

### 内部状态

```python
self._tracks: dict[int, deque] = {}            # tid → 最近 window 帧的 (frame_idx, wx, wy)
self._stats: dict[int, dict] = {}              # tid → 累计统计（sum/count/max/min/first_frame）
```

### 像素 → 世界坐标

```python
def _pix_to_world(self, cx, cy) -> tuple[float, float]:
    if self.H is not None:
        pt = np.float32([[[cx, cy]]])
        wp = cv2.perspectiveTransform(pt, self.H)[0][0]
        return float(wp[0]), float(wp[1])
    return cx / self.ppm, cy / self.ppm
```

### 瞬时速度计算

```python
def instant_speed(self, tid):
    hist = self._tracks.get(tid)
    if not hist or len(hist) < 2:
        return None
    f0, wx0, wy0 = hist[0]
    f1, wx1, wy1 = hist[-1]
    dist = math.hypot(wx1 - wx0, wy1 - wy0)
    if dist < self.min_dist_m:
        return 0.0
    dt = (f1 - f0) / self.fps
    return dist / dt * 3.6 if dt > 0 else None
```

### 累计统计更新

在 `update()` 内部，每次新帧也累加 `_stats`：

```python
v = self.instant_speed(tid)  # 调用前已 append
if v is not None and v > 0:
    s = self._stats.setdefault(tid, {'sum':0, 'count':0,
                                      'max':0, 'min':1e9,
                                      'first_frame': frame_idx})
    s['sum'] += v
    s['count'] += 1
    s['max'] = max(s['max'], v)
    s['min'] = min(s['min'], v)
    s['last_frame'] = frame_idx
```

---

## tracker.py 集成

### 启动阶段

```python
cal = get_calibration(video_path)              # 加载/触发标定
H = cal.homography                              # 可能是 None
speed_est = SpeedEstimator(H, fps=cap.get(CAP_PROP_FPS),
                           window=15,
                           pixels_per_meter=PIXELS_PER_METER)
```

### 主循环

```python
for frame_idx, frame in enumerate(iter_frames(cap)):
    results = model.track(...)
    for det in results:
        tid = det.id
        x1, y1, x2, y2 = det.bbox
        cx, cy = (x1+x2)/2, y2                  # 底部中心
        speed_est.update(frame_idx, tid, cx, cy)
        v = speed_est.instant_speed(tid)
        lane_id = assign_lane(cx, cy, cal.lanes) # 已有

        # 画面叠加
        label = f"L{lane_id}" + (f" · {v:.0f} km/h" if v else "")
        draw_bbox(frame, det.bbox, label, color)

    # 车道线可视化（启动时一次绘制到 overlay，每帧直接叠加）
    frame = cv2.addWeighted(frame, 1.0, lane_overlay, 0.5, 0)
```

### track 消失时

```python
expired_tids = set(prev_tids) - set(current_tids)
for tid in expired_tids:
    stats = speed_est.finalize(tid)
    if stats and stats['n_samples'] >= 5:       # 太短的不写
        vehicle_stats_writer.writerow([
            tid, stats['first_frame'], stats['last_frame'],
            last_known_lane[tid],
            round(stats['avg_kmh'], 1),
            round(stats['max_kmh'], 1),
            round(stats['min_kmh'], 1),
        ])
```

---

## CSV 输出

新增 `outputs/vehicle_stats.csv`：

```csv
track_id,first_frame,last_frame,lane_id,avg_speed_kmh,max_speed_kmh,min_speed_kmh,n_samples
12,142,287,2,38.4,45.2,30.1,143
13,156,310,3,52.1,58.0,46.5,152
...
```

已有的 `cross_section.csv` 中的 `speed_kmh` 列改为读取 `SpeedEstimator.instant_speed(tid)`，去掉 `counter.py` 内部的 `_estimate_speed`。

---

## 异常处理

| 情况 | 处理 |
|------|------|
| H 不可用 | fallback 到 `PIXELS_PER_METER`，警告日志，`homography_method="fallback_ppm"` |
| 滑动窗口未填满 | `instant_speed` 返回 `None`，画面显示 `--` |
| 车辆静止/极慢（< min_dist_m） | 返回 0，避免噪声放大 |
| track 仅出现 1-4 帧 | `n_samples < 5`，不写入 vehicle_stats.csv（噪声过滤） |
| 自动 H 检测失败 | 弹出 lane_annotator 备用流程（点 4 角 + 输入尺寸） |

---

## 验证方案

**测试视频**：`北进口_20260420075959至20260420081500.mp4`
**测试帧数**：前 1000 帧
**验证点**：
1. 启动时车道线标定通过（已有 calibrations/北进口/）
2. 自动 H 计算运行，homography.json 生成
3. 每帧画面叠加：
   - 车道线半透明覆盖（彩色）
   - 每辆车 bbox + 车道号 + 瞬时速度
4. `outputs/vehicle_stats.csv` 生成，含至少 10+ 辆车的统计
5. 速度合理性：早高峰场景下 0-60 km/h 区间为主

---

## 文件改动汇总

| 文件 | 改动 |
|------|------|
| `src/cross_section/speed_estimator.py` | **新增**，~150 行 |
| `src/cross_section/lane_calibration.py` | 加 `_run_homography_calibration()` + 保存逻辑，~50 行修改 |
| `src/cross_section/lane_annotator.py` | 新增备用 H 标定子流程，~80 行修改 |
| `src/cross_section/counter.py` | `CrossSectionDetector` 接收外部 SpeedEstimator，去除内部 `_estimate_speed`，~30 行修改 |
| `src/trajectory/tracker.py` | 集成 SpeedEstimator + CSV 输出 + 车道线可视化，~60 行修改 |
| `src/config/settings.py` | 新增 `VEHICLE_STATS_CSV_PATH`、`SPEED_WINDOW_FRAMES`，~5 行修改 |

**总计**：新增 ~150 行，修改 ~225 行。

---

## 暂未涉及的内容

以下议题已识别但**不在本 spec 范围**，后续单独设计：

- **断面线增补**：根据学姐反馈需在停止线前加主断面 + 右转 + 掉头三条线，需先在 tracker.py 集成完成后单独议题处理
- **车头时距 headway / 车间距 spacing**：依赖断面线，断面线议题完成后再处理
- **CLI 启动入口**：等所有功能定型后整合
