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
- 核心依赖：`ultralytics 8.4.51`、`opencv-python`、`numpy`、`matplotlib`

## 运行脚本

```bash
# 检测测试（前1000帧，MPS加速，流式写出视频）
python3 -u test_yolo26.py

# 在比赛视频上评测精度（伪GT方案，采样200帧）
python3 -u eval_on_video.py

# 在COCO val2017上评测（需下载~1GB数据集）
python3 -u eval_yolo26m.py
```

## 脚本说明

### `test_yolo26.py`
逐帧推理 + 流式写出，核心配置在文件顶部：
- `MODEL`：当前为 `yolo26x.pt`，可换 `yolo26n/s/m/l/x.pt` 或 `yolov8m.pt` 等
- `MAX_FRAMES`：限制处理帧数，`None` 为全量
- `DEVICE`：`"mps"` / `"cpu"`
- 输出：带检测框的视频 + 终端每30帧打印一次 FPS

### `eval_on_video.py`
伪GT精度评测，核心配置：
- `model_gt`：伪GT生成器，固定用 `yolo26x.pt`
- `model_pred`：待评测模型，当前为 `yolo26l.pt`
- `CONF_GT` / `CONF_PRED`：置信度阈值，当前均为 `0.25`
- `SAMPLE_EVERY=25`，`MAX_SAMPLES=200`（约200秒范围内均匀采样）
- 输出：`eval_on_video.json` + `eval_on_video.png`

### `eval_yolo26m.py`
调用 ultralytics `model.val()` 在标准数据集上评测，当前数据集为 `coco.yaml`（val2017）。
注意：不能用 `coco128.yaml`，那是训练集子集，会导致结果虚高。

## 已下载模型权重

| 文件 | 大小 | 用途 |
|------|------|------|
| `yolo26n.pt` | 5.3MB | 速度最快，基准测试 |
| `yolo26m.pt` | ~20MB | 精度/速度均衡 |
| `yolo26l.pt` | ~50MB | 当前评测主力 |
| `yolo26x.pt` | ~113MB | 伪GT生成器 |

权重保存在脚本执行目录（`/Users/liujiahang/科研/交通流算法/`）。

## 交通类别映射（COCO id）

```python
TRAFFIC_CLASSES = {0: "person", 1: "bicycle", 2: "car",
                   3: "motorcycle", 5: "bus", 7: "truck"}
```

## 比赛评分要点

- **检测精度**（20分）：各类别准确率 ≥90% 满分，每降1%扣2分
- **帧率**（15分）：≥15fps 满分，每降1fps扣1分
- **抗干扰**（5分）：早高峰/平峰/晚高峰/雨天/夜间各1分
- 需输出10类数据：截面通过数据5类、轨迹数据2类、统计数据3类（占有率/流量/排队长度）

## 精度评测注意事项

`eval_on_video.py` 的伪GT方案局限性：
- 分数反映的是"待评测模型与 yolo26x 的相似度"，非真实精度
- 置信度阈值不一致会导致FP虚高（已统一为0.25）
- bus 类因视频中样本少，AP 方差较大，参考价值有限
