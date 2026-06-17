"""
箭头 + 斑马线检测模型训练脚本。
上传到云服务器后直接运行：python3 -u train_arrow_zebra.py
"""
import csv, math, time
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ── 配置 ──────────────────────────────────────────────────────────────────────
DATA_YAML  = "data/datasets/arrow_zebra/data.yaml"
PRETRAIN   = "yolo26n.pt"
RUN_NAME   = datetime.now().strftime("%Y%m%d_%H%M")

EPOCHS   = 100
BATCH    = 64       # M1 本地改 8，workers 改 0
IMGSZ    = 640
DEVICE   = 0        # M1 本地改 "mps"
WORKERS  = 8
PATIENCE = 20       # 早停：val mAP 连续 N epoch 无提升则停

LR0      = 0.01     # 初始 LR
LRF      = 0.5      # 最终 LR = LR0 * LRF（余弦退火终点）


# ── 进度监控 Callback ─────────────────────────────────────────────────────────
class Monitor:
    HEADER = f"{'Ep':>6}  {'LR':>8}  {'box':>7}  {'cls':>7}  {'dfl':>7}  {'mAP50':>7}  {'mAP95':>7}  {'t/s':>5}"

    def __init__(self, total):
        self.total = total
        self.t0    = 0.0
        self.rows  = []
        self.csv   = Path(f"runs/arrow_zebra/{RUN_NAME}/log.csv")
        self.csv.parent.mkdir(parents=True, exist_ok=True)
        with self.csv.open("w", newline="") as f:
            csv.writer(f).writerow(["epoch","lr","box","cls","dfl","map50","map95","time_s"])
        print(self.HEADER)
        print("-" * len(self.HEADER))

    def epoch_start(self, trainer):
        self.t0 = time.perf_counter()

    def epoch_end(self, trainer):
        ep   = trainer.epoch + 1
        secs = time.perf_counter() - self.t0
        m    = trainer.metrics or {}
        tl   = trainer.tloss

        box = float(tl[0]) if tl is not None and len(tl) > 2 else float("nan")
        cls = float(tl[1]) if tl is not None and len(tl) > 2 else float("nan")
        dfl = float(tl[2]) if tl is not None and len(tl) > 2 else float("nan")
        m50 = float(m.get("metrics/mAP50(B)",    float("nan")))
        m95 = float(m.get("metrics/mAP50-95(B)", float("nan")))
        lr  = trainer.optimizer.param_groups[0]["lr"] if trainer.optimizer else float("nan")

        def f(v): return f"{v:.4f}" if not math.isnan(v) else "  ——  "
        print(f"{ep:>5}/{self.total}  {lr:>8.6f}  {f(box):>7}  {f(cls):>7}  {f(dfl):>7}  {f(m50):>7}  {f(m95):>7}  {secs:>4.0f}s", flush=True)

        with self.csv.open("a", newline="") as f_:
            csv.writer(f_).writerow([ep, lr, box, cls, dfl, m50, m95, round(secs, 1)])


# ── 主训练 ────────────────────────────────────────────────────────────────────
model   = YOLO(PRETRAIN)
monitor = Monitor(EPOCHS)
model.add_callback("on_train_epoch_start", monitor.epoch_start)
model.add_callback("on_fit_epoch_end",     monitor.epoch_end)

results = model.train(
    data         = DATA_YAML,
    epochs       = EPOCHS,
    batch        = BATCH,
    imgsz        = IMGSZ,
    device       = DEVICE,
    workers      = WORKERS,
    patience     = PATIENCE,
    project      = "runs/arrow_zebra",
    name         = RUN_NAME,
    lr0          = LR0,
    lrf          = LRF,
    cos_lr       = True,
    warmup_epochs= 3,
    save_period  = 10,
    verbose      = False,
)

print(f"\n最优权重: {Path(results.save_dir) / 'weights' / 'best.pt'}")
