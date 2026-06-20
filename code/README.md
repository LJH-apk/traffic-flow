# 交通流数据检测算法

第十届全国高校智能交通创新创业大赛企业赛道参赛作品。

## 项目概述

本项目为交通路口 4K 监控视频的实时车辆检测与交通流参数提取系统。输入为固定机位拍摄的 15 分钟 4K（3840×2160, 25fps）视频，输出为 10 类结构化交通数据（断面流量、车头时距、速度、车型、车道、轨迹、车牌、排队长度、平均速度、密度）。

**技术路线：** YOLOv26m 逐帧检测 → ByteTrack 多目标跟踪 → 叉积法断面过车判定 → B 样条车道归属标定 → 单应矩阵速度估算 → HyperLPR3 车牌识别 → Excel 统计报表。

**覆盖范围：** 丁字路口三个进口（北进口、东进口、南进口），8 个断面，10 条车道。北进口经 142 辆人工标定数据验证，车型分类准确率 95.3%、车道判定准确率 93.8%。

**工程特性：** 模块化设计，配置集中管理（`config/settings.py`），支持 macOS MPS / NVIDIA CUDA 双平台推理，内置 HTTP 仪表盘实时监控，一键批量处理三进口视频。

- **检测模型**：YOLOv26m（可替换 n/s/l/x）
- **跟踪算法**：ByteTrack（botsort.yaml）
- **车牌识别**：HyperLPR3（多帧投票 + 正则降级匹配）
- **车道判定**：B 样条几何标定 + 轨迹线段断面交点 + 方向先验
- **速度估算**：单应矩阵世界坐标转换，滑动窗口平滑

## 快速启动

```bash
# 安装依赖
pip install -r requirements.txt

# 交互菜单
 python3 -m src.main

# 直接运行
python3 -m src.main detect      # 逐帧检测
python3 -m src.main track       # 轨迹跟踪 + 车牌 + 断面过车
python3 -m src.main run-all     # 三进口批量跟踪
python3 -m src.main dashboard   # 启动仪表盘
```

## 目录结构

```
code/
├── src/
│   ├── main.py                  # 主入口
│   ├── config/settings.py       # 全局配置
│   ├── detection/detector.py    # 纯检测模块
│   ├── trajectory/
│   │   ├── tracker.py           # ByteTrack 跟踪 + 断面过车
│   │   ├── lane_assignment.py   # 车道归属判定
│   │   ├── plate_recognizer.py  # 车牌识别（HyperLPR3）
│   │   ├── traffic_report.py    # Excel 报表导出
│   │   └── traj_grouper.py      # 轨迹分组
│   ├── cross_section/
│   │   ├── counter.py           # 断面过车检测（叉积法）
│   │   ├── speed_estimator.py   # 速度估算（单应矩阵+世界坐标）
│   │   ├── lane_calibration.py  # 车道标定管理
│   │   └── section_calibration.py # 断面线加载
│   ├── dashboard/
│   │   ├── server.py            # HTTP 服务端
│   │   ├── live.py              # 实时发布器
│   │   ├── build_data.py        # 离线数据构建
│   │   └── static/              # 前端静态资源
│   ├── evaluation/
│   │   └── manual_validation.py # 人工标注验证工具
│   └── utils/                   # 视频 I/O、可视化工具
├── docs/
│   └── frontend_api.html        # 前端接口文档
└── README.md
```

## 10 类结构化输出

| # | 数据类型 | 输出文件 | 说明 |
|:-:|------|------|------|
| 1 | 断面流量 | `cross_section.csv` | 断面级过车事件，含时间戳 |
| 2 | 车头时距 | `cross_section.csv` | `headway_s` 字段 |
| 3 | 速度 | `cross_section.csv` / `vehicle_stats.csv` | km/h |
| 4 | 车型 | `cross_section.csv` | car/truck/bus/motorcycle/bicycle |
| 5 | 车道 | `cross_section.csv` | 车道 1-4 + OPPOSITE + 右转/掉头 |
| 6 | 轨迹点 | `trajectory.csv` | 1FPS 采样，含坐标+速度 |
| 7 | 车牌 | `trajectory.csv` / `cross_section.csv` | `plate` 字段 |
| 8 | 排队长度 | `traffic_report.xlsx` | 速度 < 10km/h 排队判定 |
| 9 | 平均速度 | `vehicle_stats.csv` | 每车统计 |
| 10 | 密度/间距 | `cross_section.csv` | `spacing_m` 字段 |

## 精度指标（北进口人工标定验证，142 辆）

| 指标 | 精度 | 要求 |
|------|:-:|:-:|
| 车型分类 | 95.3% | ≥90% |
| 车道判定 | 93.8% | ≥90% |
| 方向判定 | 96.1% | — |
| 车头间距一致性 | 93.2% | — |
| 过线时间 MAE | 0.21s | — |

## 三进口覆盖

| 进口 | 断面 | 车道 | 事件数 |
|------|------|------|:-:|
| 北进口 | 主断面 / 右转 / 掉头 | 车道 1-4 + 对向 | 1029 |
| 东进口 | 主断面 / 右转 | 车道 1-3 + 右转 + 对向 | 988 |
| 南进口 | 主断面 / 右转 / 掉头 | 车道 1-3 + 对向 | 1576 |

## 推理性能

| 设备 | 分辨率 | 帧率 |
|------|------|:-:|
| Apple M2 (MPS) | 4K (3840×2160) | ~16 FPS |
| NVIDIA 4090 (CUDA) | 1080p | ~16 FPS |

注：4K 视频 CPU 预处理是主要瓶颈，GPU 利用率较低。

## 环境

- Python 3.12+
- 核心依赖：`ultralytics`、`opencv-python`、`numpy`、`scipy`、`matplotlib`、`openpyxl`
- 可选：`hyperlpr3`（车牌识别）

## 视频数据准备

检测视频须按以下命名规范放置，系统根据文件名自动识别进口方向（北进口/东进口/南进口）。

| 启动方式 | 视频存放路径 |
|----------|-------------|
| 源代码启动 | `src/assets/data/北进口_20260420075959至20260420081500.mp4` |
| 编译程序 (exe) 启动 | `_internal/src/assets/data/北进口_20260420075959至20260420081500.mp4` |

> 命名格式：`{进口方向}_{年月日}{时分秒}至{年月日}{时分秒}.mp4`，例如 `南进口_20260420080000至20260420081500.mp4`。
> 同时需要将对应模型权重（.pt）放入 `src/assets/models/`（源代码）或 `_internal/src/assets/models/`（编译版）。

## 许可证

本项目仅用于第十届全国高校智能交通创新创业大赛企业赛道参赛评审，不适用于其他用途。

- **视频数据**：`data/` 目录下的视频文件为大赛组委会提供，版权归大赛主办方所有，仅限本赛道参赛使用。
- **模型权重**：`src/assets/models/` 目录下的 YOLO 权重文件遵循 Ultralytics AGPL-3.0 许可证。
- **源代码**：`src/` 目录下的代码为参赛团队原创，仅供大赛评审使用。

第三方依赖许可证详见各 Python 包的 `LICENSE` 文件。
