# 交通流数据检测算法

第十届全国高校智能交通创新创业大赛企业赛道参赛作品。

## 项目概述

针对丁字路口（南北主路 + 东进口，共 10 车道）15 分钟 4K 监控视频，实现实时车辆检测、跟踪、车牌识别与 10 类结构化交通数据输出。

- **检测模型**：YOLOv26m（可替换 n/s/l/x）
- **跟踪算法**：ByteTrack（botsort.yaml）
- **车牌识别**：HyperLPR3（多帧投票机制）
- **车道判定**：B 样条几何标定 + 轨迹线段断面交点

## 快速启动

```bash
# 安装依赖
pip install -r src/requirements.txt

# 交互菜单
PYTHONPATH=. python3 -m src.main

# 直接运行
PYTHONPATH=. python3 -m src.main detect      # 逐帧检测
PYTHONPATH=. python3 -m src.main track       # 轨迹跟踪 + 车牌 + 断面过车
PYTHONPATH=. python3 -m src.main run-all     # 三进口批量跟踪
PYTHONPATH=. python3 -m src.main dashboard   # 启动仪表盘
PYTHONPATH=. python3 -m src.main eval-video  # 伪GT精度评测
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
| Apple M1 (MPS) | 4K (3840×2160) | ~11 FPS |
| NVIDIA 4090 (CUDA) | 4K | ~9 FPS |

注：4K 视频 CPU 预处理是主要瓶颈，GPU 利用率约 2%。

## 环境

- Python 3.12+
- 核心依赖：`ultralytics`、`opencv-python`、`numpy`、`scipy`、`matplotlib`、`openpyxl`
- 可选：`hyperlpr3`（车牌识别）

## 许可证

本作品仅用于第十届全国高校智能交通创新创业大赛。
