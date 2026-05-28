# 交通流数据检测算法

第十届全国高校智能交通创新创业大赛·企业赛道  
**面向交通流的数据检测算法挑战赛**

对丁字路口（南北主路 + 东进口，共 10 车道）的 4K 监控视频进行实时车辆检测，输出 10 类结构化交通数据，目标帧率 ≥ 15 fps。

---

## 环境要求

- Python 3.13，macOS Apple Silicon（M1）或 NVIDIA GPU
- 推理设备：`mps`（M1）/ `cuda`（NVIDIA）/ `cpu`
- 核心依赖：`ultralytics`、`opencv-python`、`numpy`、`hyperlpr3`

```bash
pip install ultralytics opencv-python numpy hyperlpr3
```

---

## 快速开始

将比赛视频放入 `data/test_video.mp4`，然后按顺序运行：

```bash
# 1. 逐帧检测（输出 outputs/detection.mp4）
python3 -u src/detection/detector.py

# 2. 轨迹跟踪 + 车牌识别 + 断面过车统计
#    输出：outputs/trajectory.mp4 / trajectory.csv / cross_section.csv / vehicle_stats.csv
python3 -u src/trajectory/tracker.py

# 3. 可视化数据仪表盘（浏览器自动打开 http://localhost:8765）
python3 run_dashboard.py
```

---

## 输出数据

| 文件 | 内容 |
|------|------|
| `outputs/trajectory.csv` | 逐帧轨迹：`frame_id, timestamp_s, track_id, class_name, cx, cy, x1, y1, x2, y2, speed_kmh, plate` |
| `outputs/cross_section.csv` | 断面过车事件：`frame_id, timestamp_s, section, track_id, class_name, color, direction, speed_kmh, headway_s, spacing_m` |
| `outputs/vehicle_stats.csv` | 车辆统计：平均速度、排队长度等聚合指标 |
| `outputs/trajectory.mp4` | 带轨迹叠加的可视化视频 |

---

## 项目结构

```
src/
  config/settings.py            # 唯一配置源：路径、模型、阈值、断面线、标定参数
  detection/detector.py         # 纯检测模块（无跟踪）
  trajectory/tracker.py         # ByteTrack 跟踪 + 车牌识别 + 断面统计主流程
  cross_section/
    counter.py                  # 叉积过线检测 + 车速/车头时距/间距计算
    zebra_detector.py           # 斑马线自动检测 → 单应矩阵 H
    speed_estimator.py          # 滑动窗口测速器
    lane_detector.py            # 车道线检测（Hough + 极坐标 NMS）
    lane_calibration.py         # 车道线标定数据加载与持久化
    lane_annotator.py           # 交互式车道线标注工具
    calibrate.py                # 交互式单应矩阵手动标定
  utils/
    video_io.py                 # 视频读写工具
    visualization.py            # 绘制 bbox、中文标签（PIL 渲染）
    visualize_trajectories.py   # 静态轨迹图（CSV → PNG）
    visualize_trajectories_video.py  # 视频轨迹渐隐叠加
  evaluation/
    eval_on_video.py            # 伪 GT 精度评测（yolo26x 作 GT）
    eval_coco.py                # COCO val2017 标准评测
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
| `yolo26n.pt` | 5 MB | 速度最快 |
| `yolo26m.pt` | 20 MB | 精度/速度均衡（推荐） |
| `yolo26l.pt` | 50 MB | 高精度 |
| `yolo26x.pt` | 113 MB | 伪 GT 生成器（评测专用） |

在 `src/config/settings.py` 中修改 `MODEL_NAME` 切换模型。迁移至 NVIDIA 时将 `DEVICE` 改为 `"cuda"`。

---

## 关键配置（src/config/settings.py）

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | `yolo26n.pt` | 主推理模型 |
| `DEVICE` | `mps` | 推理设备 |
| `CONF_THRESH` | `0.25` | 检测置信度阈值 |
| `TRAJ_SAMPLE_FPS` | `1` | 轨迹采样频率（次/秒） |
| `SECTION_LINES_MAP` | 北/南/东进口各断面 | 断面线坐标与方向标签 |
| `QUEUE_SPEED_THRESH_KMH` | `10.0` | 排队判定速度阈值（km/h） |
