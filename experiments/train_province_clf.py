"""
训练省份字符分类器（31类）。

合成训练数据（仿宋_GB2312 + 宋体 + 黑体），训练轻量 CNN。
输出：src/utils/province_clf.pt

用法：
    python3 train_province_clf.py
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.province_clf import ProvinceCLF, PROVINCE_CHARS, IMG_SIZE

# ── 配置 ─────────────────────────────────────────────────────────────────────
_FONT_CANDIDATES = [
    "/Library/Fonts/仿宋_GB2312.ttf",
    "/Library/Fonts/小标宋FZXBSJW.TTF",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
]
N_PER_CLASS = 2000
BATCH_SIZE  = 256
EPOCHS      = 35
LR          = 1e-3
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_OUT   = Path("src/utils/province_clf.pt")

# ── 数据合成 ──────────────────────────────────────────────────────────────────
_CANVAS = IMG_SIZE * 2   # 先在 2× 分辨率渲染，再下采样


def _perspective(arr: np.ndarray, max_shift: int) -> np.ndarray:
    s = max_shift
    h, w = arr.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.randint(0, s), random.randint(0, s)],
        [w - random.randint(0, s), random.randint(0, s)],
        [w - random.randint(0, s), h - random.randint(0, s)],
        [random.randint(0, s), h - random.randint(0, s)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    bg = float(arr[0, 0])
    return cv2.warpPerspective(arr, M, (w, h), borderValue=bg)


def make_sample(char: str, fonts: list) -> np.ndarray:
    """合成一张省份字符灰度图（带随机增强），返回 float32 [0,1] (IMG_SIZE, IMG_SIZE)。"""
    c = _CANVAS
    font = random.choice(fonts)

    # 背景色 & 前景色（模拟蓝牌/绿牌/黄牌灰度化后的效果）
    bg = random.randint(160, 255)
    fg = random.randint(0, 60)

    img  = Image.new("L", (c, c), bg)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]
    ox   = (c - tw) // 2 - bbox[0] + random.randint(-6, 6)
    oy   = (c - th) // 2 - bbox[1] + random.randint(-6, 6)
    draw.text((ox, oy), char, font=font, fill=fg)
    arr = np.array(img, dtype=np.float32)

    # 透视扭曲
    if random.random() < 0.6:
        arr = _perspective(arr, int(c * 0.10))

    # 小角度旋转
    if random.random() < 0.5:
        angle = random.uniform(-12, 12)
        M = cv2.getRotationMatrix2D((c / 2, c / 2), angle, 1.0)
        arr = cv2.warpAffine(arr, M, (c, c), borderValue=bg)

    # 缩放到目标尺寸（模拟低分辨率摄像头）
    out_size = random.choice([IMG_SIZE, IMG_SIZE, IMG_SIZE, IMG_SIZE // 2])
    arr = cv2.resize(arr, (out_size, out_size), interpolation=cv2.INTER_AREA)
    arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # 高斯模糊（模拟失焦）
    if random.random() < 0.7:
        k  = random.choice([3, 5])
        sg = random.uniform(0.5, 2.5)
        arr = cv2.GaussianBlur(arr, (k, k), sg)

    # 高斯噪声
    noise = np.random.normal(0, random.uniform(0, 18), arr.shape).astype(np.float32)
    arr   = np.clip(arr + noise, 0, 255)

    # 亮度 / 对比度抖动
    alpha = random.uniform(0.7, 1.3)
    beta  = random.uniform(-25, 25)
    arr   = np.clip(arr * alpha + beta, 0, 255)

    return arr / 255.0


def generate_dataset(fonts: list) -> tuple[np.ndarray, np.ndarray]:
    labels = list(PROVINCE_CHARS)
    n      = len(labels)
    total  = n * N_PER_CLASS
    X = np.zeros((total, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    Y = np.zeros(total, dtype=np.int64)
    for i, char in enumerate(labels):
        print(f"  [{i+1:2d}/{n}] {char} ... ", end="\r")
        for j in range(N_PER_CLASS):
            X[i * N_PER_CLASS + j, 0] = make_sample(char, fonts)
            Y[i * N_PER_CLASS + j]    = i
    print(f"  数据集生成完毕：{total} 张 ({n} 类 × {N_PER_CLASS})         ")
    return X, Y


# ── 训练 ─────────────────────────────────────────────────────────────────────
def train(X: np.ndarray, Y: np.ndarray) -> None:
    idx  = np.random.permutation(len(Y))
    X, Y = X[idx], Y[idx]
    split = int(len(Y) * 0.9)
    tr_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(Y[:split]))
    va_ds = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(Y[split:]))
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model    = ProvinceCLF(n_classes=len(PROVINCE_CHARS)).to(DEVICE)
    opt      = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step()

        # 验证
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                correct += (model(xb).argmax(1) == yb).sum().item()
                total   += len(yb)
        acc = correct / total

        if acc > best_acc:
            best_acc = acc
            MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)

        print(f"Epoch {ep:3d}/{EPOCHS}  loss={total_loss/len(tr_dl):.4f}"
              f"  val_acc={acc:.4f}  best={best_acc:.4f}")

    print(f"\n✓ 最佳验证精度: {best_acc:.4f}  模型已保存至: {MODEL_OUT}")


# ── 主入口 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 加载字体
    fonts = []
    font_size = int(_CANVAS * 0.72)
    for fp in _FONT_CANDIDATES:
        p = Path(fp)
        if not p.exists():
            continue
        for idx in range(4):   # TTC 集合最多尝试4个子字体
            try:
                f = ImageFont.truetype(str(p), font_size, index=idx)
                # 简单检验：能渲染"京"字才算有效
                tmp = Image.new("L", (80, 80), 255)
                ImageDraw.Draw(tmp).text((5, 5), "京", font=f, fill=0)
                arr = np.array(tmp)
                if arr.min() < 200:   # 确实画出了笔画
                    fonts.append(f)
                    print(f"  ✓ {p.name}  index={idx}  size={font_size}")
                    break
            except Exception:
                break

    if not fonts:
        print("未找到有效中文字体，退出")
        sys.exit(1)

    print(f"\n使用 {len(fonts)} 种字体  设备: {DEVICE}")
    print(f"生成训练数据（{len(PROVINCE_CHARS)} 类 × {N_PER_CLASS} 张）...")
    X, Y = generate_dataset(fonts)
    print("开始训练...")
    train(X, Y)
