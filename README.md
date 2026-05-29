# 交通流数据检测算法

第十届全国高校智能交通创新创业大赛·企业赛道  
**面向交通流的数据检测算法挑战赛**

对丁字路口（南北主路 + 东进口，共 10 车道）的 4K 监控视频进行实时车辆检测，输出 10 类结构化交通数据，目标帧率 ≥ 15 fps。

当前性能：**yolo26n + FP16 + 1080p 输出，平均 16.5 fps（M1 MPS），满足比赛要求。**

---

## 环境要求

- Python 3.13，macOS Apple Silicon（M1）或 NVIDIA GPU
- 推理设备：`mps`（M1）/ `cuda`（NVIDIA）/ `cpu`
- 核心依赖：`ultralytics`、`opencv-python`、`numpy`、`scipy`、`hyperlpr3`、`openpyxl`

```bash
pip install ultralytics opencv-python numpy scipy hyperlpr3 openpyxl
```

---

## 快速开始

### 1. 准备视频文件

**文件名必须包含进口名**，系统依靠文件名自动加载对应的断面线和车道标定数据：

```
北进口_20260420075959至20260420081500.mp4   ← 北进口
南进口_20260420080000至20260420081500.mp4   ← 南进口
东进口_20260420075958至20260420081459.mp4   ← 东进口
```

支持的进口关键词：`北进口` / `南进口` / `东进口`（或 `north` / `south` / `east`）。  
若文件名不含任何进口名，系统会退化为加载全部断面线，断面归属和标定数据将不可用。

将视频放入项目根目录或 `data/` 目录，然后修改 `src/trajectory/tracker.py` 顶部的路径变量：

```python
_TEST_VIDEO  = "北进口_20260420075959至20260420081500.mp4"  # 修改为实际文件名
_START_FRAME = 0      # 起始帧，0 = 从头开始
_END_FRAME   = 9000   # 终止帧，None = 跑到结尾
```

### 2. 运行检测与跟踪

```bash
# 逐帧检测（可选，输出 outputs/detection.mp4）
python3 -u src/detection/detector.py

# 轨迹跟踪 + 车牌识别 + 断面过车 + 轨迹分组（核心流程）
# 输出：outputs/trajectory.mp4
#        outputs/trajectory.csv        逐帧轨迹
#        outputs/cross_section.csv     断面过车事件
#        outputs/vehicle_stats.csv     车辆统计
#        outputs/trajectory_groups.csv 轨迹分组
#        outputs/traffic_report.xlsx   汇总报表
python3 -u src/trajectory/tracker.py
```

### 3. 查看数据仪表盘

```bash
python3 run_dashboard.py
# 浏览器自动打开 http://localhost:8765
```

---

## 输出数据

| 文件 | 内容 |
|------|------|
| `outputs/trajectory.csv` | 逐帧轨迹：`frame_id, timestamp_s, track_id, class_name, lane_id, lane_type, cx, cy, x1, y1, x2, y2, speed_kmh, plate` |
| `outputs/cross_section.csv` | 断面过车事件：`frame_id, timestamp_s, section, track_id, class_name, color, direction, speed_kmh, headway_s, spacing_m` |
| `outputs/vehicle_stats.csv` | 车辆统计：平均速度等聚合指标 |
| `outputs/trajectory_groups.csv` | 轨迹分组：时间窗口、进口、转向类型、车辆类型、track_id 列表 |
| `outputs/traffic_report.xlsx` | 多 Sheet 报表：断面过车 / 车辆轨迹 / 流量统计 / 空间占有率 / 排队长度 / **轨迹分组** |
| `outputs/trajectory.mp4` | 1080p 带轨迹叠加的可视化视频 |

---

## 项目结构

```
src/
  config/settings.py            # 唯一配置源：路径、模型、阈值、断面线、标定参数
  detection/detector.py         # 纯检测模块（无跟踪）
  trajectory/
    tracker.py                  # BoT-SORT 跟踪 + 车牌识别 + 测速 + 车道归属 + 断面统计
    traj_grouper.py             # 轨迹相似度分组（余弦+JS散度+欧氏距离，并查集聚类）
  cross_section/
    counter.py                  # 叉积过线检测 + 车头时距/间距计算
    zebra_detector.py           # 斑马线自动检测 → 单应矩阵 H
    speed_estimator.py          # 滑动窗口测速器
    lane_detector.py            # 车道线检测（Hough + 极坐标 NMS）
    lane_calibration.py         # 车道线标定数据加载与持久化
    calibrate.py                # 交互式单应矩阵手动标定
    annotate_lane.py            # 交互式车道线标注工具
  utils/
    video_io.py                 # 视频读写 + 异步写入器 AsyncWriter
    visualization.py            # 绘制 bbox、中文标签（PIL 渲染）
    visualize_trajectories.py   # 静态轨迹图（CSV → PNG）
    visualize_trajectories_video.py  # 视频轨迹渐隐叠加
  evaluation/
    eval_on_video.py            # 伪 GT 精度评测（yolo26x 作 GT）
    eval_coco.py                # COCO val2017 标准评测
tests/
  trajectory/
    test_traj_grouper.py        # TrajGrouper 单元测试（13 个）
    test_grouper_smoke.py       # 冒烟测试（读 trajectory.csv 模拟在线分组）
dashboard/
  index.html                    # 数据可视化前端
  server.py                     # 零依赖 HTTP 服务（内置 http.server）
calibrations/                   # 各进口车道线 + 单应矩阵标定数据
```

---

## 标定

系统运行时会自动检测斑马线并计算单应矩阵。若自动检测失败，可手动标定：

```bash
# 手动标定单应矩阵（斑马线 4 角点 → H 矩阵）
python3 src/cross_section/calibrate.py

# 交互式车道线标注（生成 calibrations/<进口>/ 标定文件）
python3 src/cross_section/annotate_lane.py
```

---

## 模型权重

| 文件 | 大小 | 用途 |
|------|------|------|
| `yolo26n.pt` | 5 MB | 速度最快（当前主力，16.5 fps） |
| `yolo26m.pt` | 20 MB | 精度/速度均衡 |
| `yolo26l.pt` | 50 MB | 高精度 |
| `yolo26x.pt` | 113 MB | 伪 GT 生成器（评测专用） |

在 `src/config/settings.py` 中修改 `MODEL_NAME` 切换模型。迁移至 NVIDIA 时将 `DEVICE` 改为 `"cuda"`。

---

## 性能优化

| 优化项 | 说明 |
|--------|------|
| FP16 推理 | `model.track(half=True)`，MPS 推理提速约 40% |
| 1080p 输出 | 检测/跟踪在 4K 进行，绘制降采样到 1080p，绘制耗时从 200ms 降至 50ms |
| 异步写入 | `AsyncWriter` 后台线程写视频，不阻塞主循环 |
| Grace period | track 消失后缓冲 10 帧再确认，碎片化率从 51% 降至 13% |

---

## 关键配置（src/config/settings.py）

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | `yolo26n.pt` | 主推理模型 |
| `DEVICE` | `mps` | 推理设备 |
| `CONF_THRESH` | `0.25` | 检测置信度阈值 |
| `TRAJ_SAMPLE_FPS` | `1` | 轨迹采样频率（次/秒） |
| `TRAJ_GROUP_INTERVAL_S` | `15.0` | 轨迹分组窗口间隔（秒） |
| `TRAJ_GROUP_COS_THRESH` | `0.70` | 余弦相似度阈值 |
| `TRAJ_GROUP_JSD_THRESH` | `0.70` | JS 散度相似度阈值 |
| `TRAJ_GROUP_EUC_THRESH` | `0.70` | 欧氏距离阈值（比较时除以 100） |
| `SECTION_LINES_MAP` | 北/南/东进口各断面 | 断面线坐标与方向标签 |
| `QUEUE_SPEED_THRESH_KMH` | `10.0` | 排队判定速度阈值（km/h） |
