"""
省份字符余弦相似度识别 — 离线验证脚本

流程：
  1. 用 PIL 生成 33 个省份字符参考图（蓝底白字，模拟车牌）
  2. 用 ResNet-18 pretrained 提取特征向量（离线一次性）
  3. 从视频帧采样真实车牌，裁出省份字符区域
  4. 计算余弦相似度，打印 Top-3 候选 + 与 HyperLPR3 结果对比

运行：python3 test_province_cosine.py
"""

import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

PROVINCES   = list("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁")
VIDEO_PATH  = "test_video_fixed.mp4"
FONT_PATH   = "/System/Library/Fonts/STHeiti Medium.ttc"
IMG_SIZE    = 64          # 参考图与 crop 统一缩放到此尺寸
SAMPLE_FRAMES = 40        # 从视频采样的帧数
CONF_THRESH = 0.5         # HyperLPR3 置信度下限（宽松，收集更多样本）
UPSCALE     = 4           # 省份字符放大倍数（送入模型前）


# ── 1. 特征提取器 ──────────────────────────────────────────────────────────────
def build_extractor() -> tuple:
    """返回 (model, transform)，model 输出 512-dim 特征向量。"""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()   # 去掉分类头，保留 512-dim avgpool 输出
    model.eval()
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


def extract(model, transform, img: Image.Image) -> torch.Tensor:
    """PIL Image → L2 归一化特征向量 (512,)。"""
    x = transform(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        feat = model(x).squeeze(0)
    return F.normalize(feat, dim=0)


# ── 2. 生成省份字符参考图 ──────────────────────────────────────────────────────
def make_reference_images(font_path: str) -> dict[str, Image.Image]:
    """为每个省份字符生成蓝底白字参考图。"""
    refs = {}
    try:
        font = ImageFont.truetype(font_path, 48)
    except Exception:
        font = ImageFont.load_default()

    for ch in PROVINCES:
        img = Image.new("RGB", (64, 64), color=(0, 68, 153))  # 车牌蓝
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), ch, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((64 - tw) / 2, (64 - th) / 2), ch, fill=(255, 255, 255), font=font)
        refs[ch] = img
    return refs


# ── 3. 构建参考特征矩阵 ────────────────────────────────────────────────────────
def build_reference_matrix(
    model, transform, refs: dict[str, Image.Image]
) -> tuple[torch.Tensor, list[str]]:
    """返回 (矩阵 33×512, 省份列表)，矩阵已 L2 归一化。"""
    vecs, labels = [], []
    for ch, img in refs.items():
        vecs.append(extract(model, transform, img))
        labels.append(ch)
    return torch.stack(vecs), labels   # (33, 512)


# ── 4. 从帧裁出省份字符 crop ───────────────────────────────────────────────────
def crop_province(frame: np.ndarray, plate_box: list) -> Image.Image | None:
    """从完整帧裁出车牌省份字符区域并放大，返回 PIL Image。"""
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in plate_box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    plate = frame[y1:y2, x1:x2]
    h, w = plate.shape[:2]
    char_w = max(1, w // 7)
    prov = plate[:, :char_w]
    upscaled = cv2.resize(prov, (char_w * UPSCALE, h * UPSCALE), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))


# ── 5. Top-K 查询 ──────────────────────────────────────────────────────────────
def topk_province(
    feat: torch.Tensor,
    ref_matrix: torch.Tensor,
    ref_labels: list[str],
    k: int = 3,
) -> list[tuple[str, float]]:
    """返回余弦相似度 Top-K (省份字符, 相似度) 列表。"""
    sims = (ref_matrix @ feat).tolist()
    ranked = sorted(zip(ref_labels, sims), key=lambda x: -x[1])
    return ranked[:k]


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("加载 ResNet-18 特征提取器...")
    model, transform = build_extractor()

    print("生成省份字符参考图并提取特征向量...")
    refs = make_reference_images(FONT_PATH)
    ref_matrix, ref_labels = build_reference_matrix(model, transform, refs)
    print(f"  参考矩阵: {ref_matrix.shape}  (33 × 512)")

    try:
        import hyperlpr3 as lpr3
        catcher = lpr3.LicensePlateCatcher()
        print("HyperLPR3 加载成功\n")
    except ImportError:
        print("HyperLPR3 不可用，退出")
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = [int(total * i / SAMPLE_FRAMES) for i in range(SAMPLE_FRAMES)]

    records = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        for item in (catcher(frame) or []):
            text, conf = item[0], float(item[1])
            if conf < CONF_THRESH or not text:
                continue

            prov_img = crop_province(frame, item[3])
            if prov_img is None:
                continue

            feat = extract(model, transform, prov_img)
            topk = topk_province(feat, ref_matrix, ref_labels, k=3)

            hyper_prov = text[0] if text[0] in PROVINCES else "?"
            cosine_top1 = topk[0][0]
            records.append({
                "frame":       idx,
                "hyper_full":  text,
                "hyper_prov":  hyper_prov,
                "top1":        topk[0][0],
                "top1_sim":    topk[0][1],
                "top2":        topk[1][0],
                "top2_sim":    topk[1][1],
                "top3":        topk[2][0],
                "top3_sim":    topk[2][1],
                "agree":       hyper_prov == cosine_top1,
            })
    cap.release()

    # ── 打印报告 ──────────────────────────────────────────────────────────────
    print(f"{'帧':>6}  {'HyperLPR3':^14}  {'HY省':^4}  "
          f"{'Top1':^4} {'Sim':>5}  {'Top2':^4} {'Sim':>5}  {'Top3':^4} {'Sim':>5}  {'一致':^4}")
    print("-" * 78)
    for r in records:
        agree = "✓" if r["agree"] else "✗"
        print(
            f"{r['frame']:>6}  {r['hyper_full']:^14}  {r['hyper_prov']:^4}  "
            f"{r['top1']:^4} {r['top1_sim']:>5.3f}  "
            f"{r['top2']:^4} {r['top2_sim']:>5.3f}  "
            f"{r['top3']:^4} {r['top3_sim']:>5.3f}  {agree:^4}"
        )

    total_rec = len(records)
    agree_cnt = sum(r["agree"] for r in records)
    # 川板：检查余弦能否正确识别（当 HyperLPR3 认为是川时）
    chuan_by_hyper = [r for r in records if r["hyper_prov"] == "川"]
    chuan_correct  = [r for r in chuan_by_hyper if r["top1"] == "川"]
    print(f"\n共采样车牌: {total_rec}")
    print(f"HyperLPR3 省份 = 余弦 Top1 一致: {agree_cnt}/{total_rec} ({100*agree_cnt/total_rec:.1f}%)" if total_rec else "")
    if chuan_by_hyper:
        print(f"HyperLPR3 认为是川的样本: {len(chuan_by_hyper)}  "
              f"余弦 Top1 也是川: {len(chuan_correct)} ({100*len(chuan_correct)/len(chuan_by_hyper):.1f}%)")

    # 关键：当 HyperLPR3 认为是辽/沪时，余弦 Top1 是什么？
    confused = [r for r in records if r["hyper_prov"] in ("辽", "沪")]
    if confused:
        print(f"\nHyperLPR3 误识为辽/沪的样本 ({len(confused)} 个):")
        for r in confused:
            print(f"  帧{r['frame']:>6} {r['hyper_full']:^14}  "
                  f"余弦Top1={r['top1']}({r['top1_sim']:.3f})  "
                  f"Top2={r['top2']}({r['top2_sim']:.3f})  "
                  f"Top3={r['top3']}({r['top3_sim']:.3f})")


if __name__ == "__main__":
    main()
