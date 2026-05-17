"""
yolo26m 精度评测脚本
使用 COCO128 验证集计算 mAP50 / mAP50-95，输出详细数据并绘图
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from ultralytics import YOLO

matplotlib.rcParams["font.family"] = "Arial Unicode MS"   # macOS 中文字体

MODEL    = "yolo26m.pt"
DATA     = "coco.yaml"      # COCO val2017，真正的验证集（需下载 ~1GB）
DEVICE   = "mps"
OUTPUT   = Path("/Users/liujiahang/科研/交通流算法")

# 只关注交通相关类别（COCO 类名）
TRAFFIC_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]


def run_val():
    model = YOLO(MODEL)
    print(f"模型: {MODEL}  数据集: {DATA}  设备: {DEVICE}\n")

    metrics = model.val(data=DATA, device=DEVICE, verbose=True)

    # ---------- 提取整体指标 ----------
    map50    = float(metrics.box.map50)
    map5095  = float(metrics.box.map)
    map75    = float(metrics.box.map75)
    precision = float(metrics.box.mp)
    recall    = float(metrics.box.mr)

    print(f"\n{'='*45}")
    print(f"  整体指标")
    print(f"{'='*45}")
    print(f"  mAP@0.5        : {map50:.4f}")
    print(f"  mAP@0.5:0.95   : {map5095:.4f}")
    print(f"  mAP@0.75       : {map75:.4f}")
    print(f"  Precision (mean): {precision:.4f}")
    print(f"  Recall    (mean): {recall:.4f}")

    # ---------- 提取逐类别指标 ----------
    names       = model.names                          # {id: name}
    ap50_per    = metrics.box.ap50.tolist()            # per-class AP@0.5
    ap5095_per  = metrics.box.ap.tolist()              # per-class AP@0.5:0.95

    per_class = []
    for i, (a50, a5095) in enumerate(zip(ap50_per, ap5095_per)):
        name = names.get(i, str(i))
        per_class.append({"class": name, "AP50": round(a50, 4), "AP50_95": round(a5095, 4)})

    # 只保留交通相关类别
    traffic = [d for d in per_class if d["class"] in TRAFFIC_CLASSES]

    print(f"\n{'='*45}")
    print(f"  交通类别逐类 AP")
    print(f"{'='*45}")
    print(f"  {'类别':<14}  AP@0.5    AP@0.5:0.95")
    print(f"  {'-'*40}")
    for d in traffic:
        print(f"  {d['class']:<14}  {d['AP50']:.4f}    {d['AP50_95']:.4f}")

    # ---------- 保存 JSON ----------
    result = {
        "model": MODEL, "data": DATA,
        "overall": {
            "mAP50": map50, "mAP50_95": map5095, "mAP75": map75,
            "precision": precision, "recall": recall,
        },
        "per_class_all": per_class,
        "per_class_traffic": traffic,
    }
    json_path = OUTPUT / "eval_yolo26m.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n详细数据已保存: {json_path}")

    # ---------- 绘图 ----------
    _plot_overall(map50, map5095, map75, precision, recall)
    _plot_per_class(traffic)
    print("图表已保存到项目目录")


def _plot_overall(map50, map5095, map75, precision, recall):
    labels  = ["mAP@0.5", "mAP@0.5:0.95", "mAP@0.75", "Precision", "Recall"]
    values  = [map50, map5095, map75, precision, recall]
    colors  = ["#4C9BE8", "#E8834C", "#4CE87A", "#C84CE8", "#E8C84C"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"yolo26m 整体精度指标（{DATA}）")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT / "eval_yolo26m_overall.png", dpi=150)
    plt.close()


def _plot_per_class(traffic):
    if not traffic:
        return
    classes   = [d["class"] for d in traffic]
    ap50_vals = [d["AP50"]    for d in traffic]
    ap95_vals = [d["AP50_95"] for d in traffic]

    x     = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width/2, ap50_vals, width, label="AP@0.5",       color="#4C9BE8", zorder=3)
    b2 = ax.bar(x + width/2, ap95_vals, width, label="AP@0.5:0.95",  color="#E8834C", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AP Score")
    ax.set_title("yolo26m 交通类别逐类 AP（COCO128）")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT / "eval_yolo26m_per_class.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    run_val()
