"""
在 COCO val2017 标准数据集上评测模型精度。

注意：需提前下载 COCO val2017（约 1GB），不能用 coco128.yaml（训练集子集）。

运行::

    python3 -u src/evaluation/eval_coco.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parents[2]))

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from src.config.settings import DEVICE, MODEL_DIR, MODEL_NAME, OUTPUT_DIR

matplotlib.rcParams["font.family"] = "Arial Unicode MS"

DATA            = "coco.yaml"   # COCO val2017 真正的验证集
TRAFFIC_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]


def run_val() -> None:
    """在 COCO val2017 上运行 model.val() 并输出评测结果。"""
    model = YOLO(str(MODEL_DIR / MODEL_NAME))
    print(f"模型: {MODEL_NAME}  数据集: {DATA}  设备: {DEVICE}\n")

    metrics = model.val(data=DATA, device=DEVICE, verbose=True)

    map50   = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    map75   = float(metrics.box.map75)
    prec    = float(metrics.box.mp)
    recall  = float(metrics.box.mr)

    print(f"\n{'='*45}")
    print(f"  整体指标")
    print(f"{'='*45}")
    print(f"  mAP@0.5        : {map50:.4f}")
    print(f"  mAP@0.5:0.95   : {map5095:.4f}")
    print(f"  mAP@0.75       : {map75:.4f}")
    print(f"  Precision (mean): {prec:.4f}")
    print(f"  Recall    (mean): {recall:.4f}")

    names      = model.names
    ap50_per   = metrics.box.ap50.tolist()
    ap5095_per = metrics.box.ap.tolist()

    per_class = [
        {"class": names.get(i, str(i)), "AP50": round(a50, 4), "AP50_95": round(a5095, 4)}
        for i, (a50, a5095) in enumerate(zip(ap50_per, ap5095_per))
    ]
    traffic = [d for d in per_class if d["class"] in TRAFFIC_CLASSES]

    print(f"\n{'='*45}")
    print(f"  交通类别逐类 AP")
    print(f"{'='*45}")
    print(f"  {'类别':<14}  AP@0.5    AP@0.5:0.95")
    print(f"  {'-'*40}")
    for d in traffic:
        print(f"  {d['class']:<14}  {d['AP50']:.4f}    {d['AP50_95']:.4f}")

    result = {
        "model": MODEL_NAME, "data": DATA,
        "overall": {
            "mAP50": map50, "mAP50_95": map5095, "mAP75": map75,
            "precision": prec, "recall": recall,
        },
        "per_class_all":     per_class,
        "per_class_traffic": traffic,
    }
    json_path = OUTPUT_DIR / "eval_coco.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n详细数据已保存: {json_path}")

    _plot_overall(map50, map5095, map75, prec, recall)
    _plot_per_class(traffic)
    print("图表已保存到 outputs/")


def _plot_overall(
    map50: float,
    map5095: float,
    map75: float,
    prec: float,
    recall: float,
) -> None:
    """绘制整体指标柱状图。"""
    labels = ["mAP@0.5", "mAP@0.5:0.95", "mAP@0.75", "Precision", "Recall"]
    values = [map50, map5095, map75, prec, recall]
    colors = ["#4C9BE8", "#E8834C", "#4CE87A", "#C84CE8", "#E8C84C"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title(f"{MODEL_NAME} 整体精度指标（{DATA}）")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0); ax.set_axisbelow(True)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eval_coco_overall.png", dpi=150)
    plt.close()


def _plot_per_class(traffic: list[dict]) -> None:
    """绘制逐类别 AP 柱状图。"""
    if not traffic:
        return
    classes   = [d["class"] for d in traffic]
    ap50_vals = [d["AP50"]    for d in traffic]
    ap95_vals = [d["AP50_95"] for d in traffic]

    x     = np.arange(len(classes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width/2, ap50_vals, width, label="AP@0.5",      color="#4C9BE8", zorder=3)
    b2 = ax.bar(x + width/2, ap95_vals, width, label="AP@0.5:0.95", color="#E8834C", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(0, 1.05); ax.set_ylabel("AP Score")
    ax.set_title(f"{MODEL_NAME} 交通类别逐类 AP（COCO val2017）")
    ax.legend(); ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0); ax.set_axisbelow(True)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eval_coco_per_class.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    run_val()
