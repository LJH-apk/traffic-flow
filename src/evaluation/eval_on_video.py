"""
在比赛视频上评测模型精度。
用 yolo26x 推理结果作为伪标注（Pseudo-GT），计算 mAP50 / mAP50-95。

运行::

    python3 -u src/evaluation/eval_on_video.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parents[2]))

import json

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from src.config.settings import (
    CONF_GT,
    CONF_PRED,
    DEVICE,
    MAX_SAMPLES,
    MODEL_DIR,
    MODEL_NAME,
    MODEL_NAME_GT,
    OUTPUT_DIR,
    SAMPLE_EVERY,
    VEHICLE_CLASSES,
    VIDEO_PATH,
)

matplotlib.rcParams["font.family"] = "Arial Unicode MS"

TRAFFIC_CLASSES = VEHICLE_CLASSES  # {cid: name}，评测用
IOU_THRESHOLDS  = np.linspace(0.5, 0.95, 10)


# ── IoU 计算 ──────────────────────────────────────────────────────────────────
def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """计算单个框与一组框的 IoU，格式均为 [x1,y1,x2,y2]。"""
    xi1 = np.maximum(box[0], boxes[:, 0])
    yi1 = np.maximum(box[1], boxes[:, 1])
    xi2 = np.minimum(box[2], boxes[:, 2])
    yi2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(xi2 - xi1, 0) * np.maximum(yi2 - yi1, 0)
    area_box   = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return inter / np.maximum(union, 1e-6)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """单类别 AP 计算（11 点插值）。"""
    recalls    = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]))


def compute_ap_at_iou(
    gt_list: list[np.ndarray],
    pred_list: list[np.ndarray],
    iou_thresh: float,
) -> float:
    """计算指定 IoU 阈值下的 AP。

    Args:
        gt_list:   每张图的 GT 框列表，每项 shape (N, 4)。
        pred_list: 每张图的预测框列表，每项 shape (M, 5)，列=[x1,y1,x2,y2,conf]。
        iou_thresh: IoU 阈值。

    Returns:
        AP 值（0~1）。
    """
    all_confs  = []
    all_tp     = []
    n_gt_total = 0

    for gt_boxes, pred_boxes in zip(gt_list, pred_list):
        n_gt_total += len(gt_boxes)
        if len(pred_boxes) == 0:
            continue
        order      = np.argsort(-pred_boxes[:, 4])
        pred_boxes = pred_boxes[order]
        matched    = np.zeros(len(gt_boxes), dtype=bool)

        for pb in pred_boxes:
            all_confs.append(pb[4])
            if len(gt_boxes) == 0:
                all_tp.append(0)
                continue
            ious = box_iou(pb[:4], gt_boxes)
            best = int(np.argmax(ious))
            if ious[best] >= iou_thresh and not matched[best]:
                matched[best] = True
                all_tp.append(1)
            else:
                all_tp.append(0)

    if len(all_confs) == 0 or n_gt_total == 0:
        return 0.0

    order  = np.argsort(-np.array(all_confs))
    tp_cum = np.cumsum(np.array(all_tp)[order])
    fp_cum = np.cumsum(1 - np.array(all_tp)[order])
    recalls    = tp_cum / n_gt_total
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
    return compute_ap(recalls, precisions)


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main() -> None:
    """评测主流程。"""
    print("加载模型...")
    model_gt   = YOLO(str(MODEL_DIR / MODEL_NAME_GT))
    model_pred = YOLO(str(MODEL_DIR / MODEL_NAME))

    cap    = cv2.VideoCapture(str(VIDEO_PATH))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_ids = list(range(0, min(total, SAMPLE_EVERY * MAX_SAMPLES), SAMPLE_EVERY))
    print(f"视频共 {total} 帧，采样 {len(sample_ids)} 帧（每{SAMPLE_EVERY}帧取1帧）\n")

    frames = []
    for fid in sample_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    print(f"实际读取 {len(frames)} 帧")

    class_gt   = {cid: [] for cid in TRAFFIC_CLASSES}
    class_pred = {cid: [] for cid in TRAFFIC_CLASSES}

    print("\n生成伪GT（yolo26x）并推理待评测模型...")
    for i, frame in enumerate(frames):
        res_gt   = model_gt  (frame, device=DEVICE, conf=CONF_GT,   verbose=False)[0]
        res_pred = model_pred(frame, device=DEVICE, conf=CONF_PRED, verbose=False)[0]

        for cid in TRAFFIC_CLASSES:
            gt_boxes = res_gt.boxes
            gt_cid   = (gt_boxes.xyxy[gt_boxes.cls.int() == cid].cpu().numpy()
                        if len(gt_boxes) else np.zeros((0, 4)))
            class_gt[cid].append(gt_cid)

            pd_boxes = res_pred.boxes
            mask = (pd_boxes.cls.int() == cid)
            if mask.sum() > 0:
                xy = pd_boxes.xyxy[mask].cpu().numpy()
                cf = pd_boxes.conf[mask].cpu().numpy().reshape(-1, 1)
                class_pred[cid].append(np.hstack([xy, cf]))
            else:
                class_pred[cid].append(np.zeros((0, 5)))

        if (i + 1) % 20 == 0:
            print(f"  已处理 {i+1}/{len(frames)} 帧")

    print("\n计算 AP...")
    results = {}
    for cid, cname in TRAFFIC_CLASSES.items():
        ap50   = compute_ap_at_iou(class_gt[cid], class_pred[cid], 0.5)
        ap5095 = float(np.mean([
            compute_ap_at_iou(class_gt[cid], class_pred[cid], t)
            for t in IOU_THRESHOLDS
        ]))
        results[cname] = {"AP50": round(ap50, 4), "AP50_95": round(ap5095, 4)}

    map50   = float(np.mean([v["AP50"]    for v in results.values()]))
    map5095 = float(np.mean([v["AP50_95"] for v in results.values()]))

    print(f"\n{'='*50}")
    print(f"  {MODEL_NAME} vs {MODEL_NAME_GT}（伪GT）评测结果")
    print(f"  采样帧数: {len(frames)}  阈值GT={CONF_GT} / Pred={CONF_PRED}")
    print(f"{'='*50}")
    print(f"  {'类别':<14}  AP@0.5    AP@0.5:0.95")
    print(f"  {'-'*44}")
    for cname, v in results.items():
        print(f"  {cname:<14}  {v['AP50']:.4f}    {v['AP50_95']:.4f}")
    print(f"  {'-'*44}")
    print(f"  {'mAP (均值)':<14}  {map50:.4f}    {map5095:.4f}")

    output = {
        "model": MODEL_NAME, "pseudo_gt": MODEL_NAME_GT,
        "samples": len(frames),
        "mAP50": round(map50, 4), "mAP50_95": round(map5095, 4),
        "per_class": results,
    }
    json_path = OUTPUT_DIR / "eval_on_video.json"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n详细数据已保存: {json_path}")

    _plot(results, map50, map5095)
    print("图表已保存: eval_on_video.png")


def _plot(results: dict, map50: float, map5095: float) -> None:
    """绘制 AP 柱状图并保存。"""
    classes   = list(results.keys())
    ap50_vals = [results[c]["AP50"]    for c in classes]
    ap95_vals = [results[c]["AP50_95"] for c in classes]

    x     = np.arange(len(classes))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    b1 = ax.bar(x - width/2, ap50_vals, width, label="AP@0.5",      color="#4C9BE8", zorder=3)
    b2 = ax.bar(x + width/2, ap95_vals, width, label="AP@0.5:0.95", color="#E8834C", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylim(0, 1.05); ax.set_ylabel("AP")
    ax.set_title(f"{MODEL_NAME} 逐类别 AP（伪GT by {MODEL_NAME_GT}）")
    ax.legend(); ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0); ax.set_axisbelow(True)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax2 = axes[1]
    bars = ax2.bar(["mAP@0.5", "mAP@0.5:0.95"], [map50, map5095],
                   color=["#4C9BE8", "#E8834C"], width=0.4, zorder=3)
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("mAP")
    ax2.set_title(f"{MODEL_NAME} 整体 mAP")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0); ax2.set_axisbelow(True)
    for bar, val in zip(bars, [map50, map5095]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.suptitle(f"评测数据集：比赛测试视频（{len(classes)} 类）", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eval_on_video.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
