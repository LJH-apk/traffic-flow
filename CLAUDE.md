# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 对话约束
- 使用中文回答，不要冒出奇怪的英文词汇
- 在每一轮的对话后面加一个 喵～



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

# 轨迹跟踪 + 车牌识别 + 断面过车（输出 trajectory.mp4 / trajectory.csv / cross_section.csv）
python3 -u src/trajectory/tracker.py

# 在比赛视频上评测精度（伪GT方案，采样200帧）
python3 -u src/evaluation/eval_on_video.py

# 在COCO val2017上评测（需下载~1GB数据集，不能用coco128.yaml，会导致结果虚高）
python3 -u src/evaluation/eval_coco.py

# 手动标定：斑马线→单应矩阵（自动检测失败时使用）
python3 src/cross_section/calibrate.py

# 轨迹可视化
python3 -u src/utils/visualize_trajectories.py        # 静态图
python3 -u src/utils/visualize_trajectories_video.py  # 视频叠加

# 车道线手动标注（生成 calibrations/ 标定数据）
python3 src/cross_section/annotate_lane.py

# AI 交通分析智能体（DeepSeek 桌面应用；需 export DEEPSEEK_API_KEY=sk-xxx）
python3 run_agent.py    # 等价 python3 -m src.main agent；无 pywebview 时自动退化为浏览器模式
```

## 代码架构

```
src/
  config/settings.py       # 唯一配置源：路径、模型名、设备、阈值、类别映射、断面线
  detection/detector.py    # VehicleDetector：纯检测，无跟踪，流式写出视频
  trajectory/tracker.py    # TrajectoryTracker + PlateRecognizer：ByteTrack跟踪 + OCR + CSV
  cross_section/
    zebra_detector.py      # ZebraDetector：自适应阈值→水平形态学→单应矩阵 H，返回(H, n, stripe_rects)
    counter.py             # CrossSectionDetector（叉积过线）+ detect_color（HSV颜色分类）
    calibrate.py           # 交互式手动标定（斑马线4角点→H矩阵）
    lane_detector.py       # LaneDetector：Hough+极坐标NMS（当前版本，效果有限）
  utils/video_io.py                    # open_video / video_meta / make_writer / iter_frames
  utils/visualization.py               # draw_boxes / put_fps_text（含PIL中文渲染）
  utils/visualize_trajectories.py      # 静态轨迹图（trajectory.csv → PNG）
  utils/visualize_trajectories_video.py # 视频轨迹叠加（渐隐效果）
  cross_section/annotate_lane.py       # 交互式车道线手动标注工具
  evaluation/
    eval_on_video.py       # 伪GT精度评测（yolo26x作GT，对比待测模型）
    eval_coco.py           # ultralytics model.val() 在COCO val2017上评测
  agent/                   # AI 交通分析智能体（DeepSeek 函数调用桌面应用）
    datastore.py           # TrafficDataStore：读三进口full CSV+dashboard JSON，聚合查询
                           # 含测速可信度交叉验证（到达/离去对称性）、车流波浪模式识别
    tools.py               # ~20个 function calling 工具：交通数据类 + 项目自省类
    project_tools.py       # 只读项目自省工具：读文件/检索/Python模块解析/CSV体检/列输出文件
                           # （沙箱限定项目目录内，不执行任意命令）
    knowledge.py           # 交通工程知识库（带出处）+ 关键词检索，抑制幻觉
    llm.py                 # DeepSeek 客户端（requests 实现 SSE 流式 + tool_calls 增量合并）
    analyst.py             # 系统提示词（防幻觉铁律）+ 工具循环 + 一键报告流水线
    server.py              # 零依赖 HTTP 服务（端口8766，SSE 推流）
    app.py                 # pywebview 桌面窗口入口（缺失时退化浏览器）
    static/                # 前端：聊天流+工具步骤卡+KPI侧栏+报告抽屉（深蓝科技风）
                           # 内置内部命令 /help /clear /report /summary /reload /tools /model
                           # （聊天框输入，客户端直接处理，不占用 LLM 调用）
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
| `SECTION_LINES` | 断面A/B | 格式：(name, lx1,ly1,lx2,ly2, dir_pos, dir_neg) |
| `HOMOGRAPHY_MATRIX` | `None` | 像素→路面世界坐标（米），None时用像素兜底 |
| `PIXELS_PER_METER` | `85.0` | H不可用时的兜底系数 |

## 已下载模型权重

| 文件 | 大小 | 用途 |
|------|------|------|
| `yolo26n.pt` | 5.3MB | 速度最快，基准测试 |
| `yolo26m.pt` | ~20MB | 精度/速度均衡，当前主力 |
| `yolo26l.pt` | ~50MB | 评测对比 |
| `yolo26x.pt` | ~113MB | 伪GT生成器 |

## tracker.py 核心流程

`TrajectoryTracker.run()` 启动时（主循环前）：
- 读第一帧，`ZebraDetector.detect()` 自动检测斑马线 → 计算单应矩阵 H（失败则用 `PIXELS_PER_METER` 兜底）
- `LaneDetector.detect()` 检测车道线段（一次性，结果静态复用）
- 打开 `cross_section.csv`，初始化 `CrossSectionDetector`

主循环每帧：
1. `model.track(..., persist=True)` 运行 ByteTrack
2. 从 `PlateRecognizer._cache` 读已确认车牌（零OCR开销）；采样帧调 `recognize()` 写轨迹CSV行
3. `CrossSectionDetector.update()` 检测叉积符号翻转 → 过线事件写 `cross_section.csv`
4. 绘制顺序（从底层到顶层）：车道线（半透明）→ 斑马线框 → 断面线 → 车牌框 → 车辆框

`PlateRecognizer` 使用 HyperLPR3，首次识别成功后缓存，同一 `track_id` 不重复OCR。HyperLPR3 不可用时静默禁用。

## cross_section 模块说明

**`ZebraDetector.detect(frame)`**：返回 `(H_3x3, n_stripes, stripe_rects) | None`。
stripe_rects 为各条纹 `(x,y,w,h)` 列表，直接用于视频叠加可视化。

**`CrossSectionDetector.update(frame_idx, ts, tid, cls, frame, x1,y1,x2,y2)`**：
返回本帧触发的过线事件列表（通常为空）。每个事件含：
`frame_id, timestamp_s, section, track_id, class_name, color, direction, speed_kmh, headway_s, spacing_m`

速度计算：维护最近15帧世界坐标历史（`_history[tid]`），距离/时间×3.6 得 km/h。
车头时距/间距：记录同断面同方向上一辆车的时间戳和速度（`_last_crossing`）。

**`detect_color(frame, x1,y1,x2,y2)`**：采样 bbox 中央60%×60% 区域，HSV中位数分类为黑/白/银/灰/红/黄/绿/蓝/其他。

## visualization.py 中文渲染

`put_text`：含CJK字符时走 Pillow（`/System/Library/Fonts/STHeiti Light.ttc`，字号18），否则走 `cv2.putText`。PIL不可用时静默降级。

## 比赛评分要点

- **检测精度**（20分）：各类别准确率 ≥90% 满分，每降1%扣2分
- **帧率**（15分）：≥15fps 满分，每降1fps扣1分
- **抗干扰**（5分）：早高峰/平峰/晚高峰/雨天/夜间各1分
- **数据提取完整性**（10分）：10类数据（断面过车5类、轨迹2类、统计3类）每符合1种得1分
- **数据清洗合理性**（5分）：PPT中说明异常处理方式（逆行、ID跳变、低置信误检等），每条合理意见1分

## 精度评测注意事项

`eval_on_video.py` 分数反映"待测模型与 yolo26x 的相似度"，非真实精度。置信度阈值须统一（当前均为0.25），否则FP虚高。bus 类视频样本少，AP方差大，参考价值有限。
