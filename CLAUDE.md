# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目背景

第十届全国高校智能交通创新创业大赛企业赛道：面向交通流的数据检测算法挑战赛。

目标：对丁字路口（南北主路 + 东进口，共10车道）15分钟4K监控视频进行实时车辆检测，输出10类结构化交通数据，帧率须 ≥15fps。

比赛视频：`test_video.mp4`（3840×2160，25fps，22568帧，H.264）
可正常播放版本：`test_video_fixed.mp4`（已修复 codec_tag）

## 环境

- Python 3.13，macOS Apple Silicon（M1）
- 推理设备：`mps`（Metal Performance Shaders）
- 核心依赖：`ultralytics 8.4.51`、`opencv-python`、`numpy`、`matplotlib`、`rapidocr-onnxruntime`

## 运行命令

```bash
# 逐帧检测（前1000帧，输出 outputs/detection.mp4）
python3 -u src/detection/detector.py

# 轨迹跟踪 + 车牌识别（输出 outputs/trajectory.mp4 + outputs/trajectory.csv）
python3 -u src/trajectory/tracker.py

# 在比赛视频上评测精度（伪GT方案，采样200帧）
python3 -u src/evaluation/eval_on_video.py

# 在COCO val2017上评测（需下载~1GB数据集，不能用coco128.yaml，会导致结果虚高）
python3 -u src/evaluation/eval_coco.py

# 轨迹可视化
python3 -u visualize_trajectories.py        # 静态图
python3 -u visualize_trajectories_video.py  # 视频叠加
```

## 代码架构

```
src/
  config/settings.py       # 唯一配置源：路径、模型名、设备、阈值、类别映射
  detection/detector.py    # VehicleDetector：纯检测，无跟踪，流式写出视频
  trajectory/tracker.py    # TrajectoryTracker + PlateRecognizer：ByteTrack跟踪 + OCR + CSV
  utils/video_io.py        # open_video / video_meta / make_writer / iter_frames
  utils/visualization.py   # draw_boxes / put_fps_text（含PIL中文渲染）
  evaluation/
    eval_on_video.py       # 伪GT精度评测（yolo26x作GT，对比待测模型）
    eval_coco.py           # ultralytics model.val() 在COCO val2017上评测
```

所有业务模块均从 `src/config/settings.py` 读取配置，禁止在业务代码中硬编码路径或超参。各模块顶部有 `sys.path.insert` 保证从任意目录直接运行。

## 关键配置（src/config/settings.py）

| 常量 | 当前值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | `yolo26m.pt` | 主推理模型，可换 n/s/m/l/x |
| `MODEL_NAME_GT` | `yolo26x.pt` | 伪GT生成器（eval专用） |
| `DEVICE` | `mps` | 迁移NVIDIA时改为 `cuda` |
| `CONF_THRESH` | `0.25` | 检测置信度阈值 |
| `TRAJ_SAMPLE_FPS` | `1` | 轨迹CSV采样频率（每秒1次） |
| `VEHICLE_CLASSES` | bicycle/car/motorcycle/bus/truck | COCO类别ID到名称映射（不含person） |

## 已下载模型权重

| 文件 | 大小 | 用途 |
|------|------|------|
| `yolo26n.pt` | 5.3MB | 速度最快，基准测试 |
| `yolo26m.pt` | ~20MB | 精度/速度均衡，当前主力 |
| `yolo26l.pt` | ~50MB | 评测对比 |
| `yolo26x.pt` | ~113MB | 伪GT生成器 |

## tracker.py 核心流程

`TrajectoryTracker.run()` 主循环：
1. 每帧调 `model.track(..., persist=True)` 运行 ByteTrack
2. 每帧从 `PlateRecognizer._cache` 读已确认车牌（零OCR开销）
3. 采样帧（每 `sample_interval` 帧）调 `PlateRecognizer.recognize()` 跑OCR并写CSV行
4. 对已识别车牌的车辆画黄色矩形（车辆框底部30%+向下延伸5%，宽度收窄10%）
5. 调 `draw_boxes(..., plates)` 在标签末尾追加车牌号

`PlateRecognizer` OCR引擎优先级：RapidOCR（ONNX）→ PaddleOCR → 禁用。首次识别成功后缓存，同一 `track_id` 不重复OCR。

## visualization.py 中文渲染

`draw_boxes` 调用 `_put_text`：含CJK字符时走 Pillow（`/System/Library/Fonts/STHeiti Light.ttc`，字号18），否则走 `cv2.putText`。PIL不可用时静默降级（中文显示为 `???`，功能不受影响）。

## 比赛评分要点

- **检测精度**（20分）：各类别准确率 ≥90% 满分，每降1%扣2分
- **帧率**（15分）：≥15fps 满分，每降1fps扣1分
- **抗干扰**（5分）：早高峰/平峰/晚高峰/雨天/夜间各1分
- **数据提取完整性**（10分）：10类数据（断面过车5类、轨迹2类、统计3类）每符合1种得1分
- **数据清洗合理性**（5分）：PPT中说明异常处理方式（逆行、ID跳变、低置信误检等），每条合理意见1分

## 精度评测注意事项

`eval_on_video.py` 分数反映"待测模型与 yolo26x 的相似度"，非真实精度。置信度阈值须统一（当前均为0.25），否则FP虚高。bus 类视频样本少，AP方差大，参考价值有限。
