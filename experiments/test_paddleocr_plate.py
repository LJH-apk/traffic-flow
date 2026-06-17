"""
PaddleOCR vs HyperLPR3 车牌省份识别对比测试

流程：
  采样视频帧 → HyperLPR3 定位车牌框
  → 对每个 crop 同时用 HyperLPR3 和 PaddleOCR 识别
  → 对比省份字符识别结果

运行：python3 test_paddleocr_plate.py
"""

import sys
import re
import cv2
import numpy as np

PROVINCES     = set("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁")
VIDEO_PATH    = "test_video_fixed.mp4"
SAMPLE_FRAMES = 40
CONF_THRESH   = 0.5     # HyperLPR3 置信度下限（宽松，多收集样本）
UPSCALE       = 3       # 送 PaddleOCR 前放大倍数

_PLATE_FULL = re.compile(
    r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁][A-Z][A-Z0-9]{5}"
)
_PLATE_BODY = re.compile(r"[A-Z][A-Z0-9]{5}$")


def match_plate(text: str) -> str:
    clean = text.replace(" ", "").upper()
    m = _PLATE_FULL.search(clean)
    if m:
        return m.group()
    m = _PLATE_BODY.search(clean)
    return m.group() if m else ""


def main() -> None:
    # ── 加载 HyperLPR3 ──────────────────────────────────────────────────────
    try:
        import hyperlpr3 as lpr3
        catcher = lpr3.LicensePlateCatcher()
        print("[HyperLPR3] 加载成功")
    except ImportError:
        print("[HyperLPR3] 未安装，退出")
        sys.exit(1)

    # ── 加载 PaddleOCR ──────────────────────────────────────────────────────
    import warnings
    warnings.filterwarnings("ignore")
    from paddleocr import PaddleOCR
    # use_angle_cls=True 处理旋转车牌；lang='ch' 支持中文
    ocr = PaddleOCR(lang="ch")
    print("[PaddleOCR]  加载成功\n")

    cap = cv2.VideoCapture(VIDEO_PATH)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(total * i / SAMPLE_FRAMES) for i in range(SAMPLE_FRAMES)]

    records = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        for item in (catcher(frame) or []):
            text_h, conf = item[0], float(item[1])
            if conf < CONF_THRESH or not text_h:
                continue

            # 裁出车牌 crop
            fh, fw = frame.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in item[3]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            h, w = crop.shape[:2]

            # 放大后送 PaddleOCR
            big = cv2.resize(crop, (w * UPSCALE, h * UPSCALE),
                             interpolation=cv2.INTER_CUBIC)
            try:
                paddle_result = ocr.predict(big)
                paddle_texts = []
                for r in (paddle_result or []):
                    for t in r.get("rec_texts", []):
                        paddle_texts.append(t)
                paddle_raw = " | ".join(paddle_texts)
                paddle_plate = ""
                for t in paddle_texts:
                    p = match_plate(t)
                    if p:
                        paddle_plate = p
                        break
            except Exception as e:
                paddle_raw, paddle_plate = f"ERR:{e}", ""

            hyper_plate  = match_plate(text_h)
            hyper_prov   = hyper_plate[0]  if hyper_plate and hyper_plate[0] in PROVINCES else "?"
            paddle_prov  = paddle_plate[0] if paddle_plate and paddle_plate[0] in PROVINCES else "?"

            records.append({
                "frame":        idx,
                "hyper_plate":  hyper_plate,
                "hyper_prov":   hyper_prov,
                "hyper_conf":   conf,
                "paddle_raw":   paddle_raw,
                "paddle_plate": paddle_plate,
                "paddle_prov":  paddle_prov,
                "agree":        hyper_prov == paddle_prov and hyper_prov != "?",
                "size":         f"{w}x{h}",
            })

    cap.release()

    # ── 打印报告 ──────────────────────────────────────────────────────────────
    print(f"{'帧':>6}  {'HyperLPR3':^14} {'conf':>5}  {'Paddle识别':^20}  "
          f"{'Paddle车牌':^12}  {'HY省':^4} {'PD省':^4}  {'一致':^4}  {'尺寸':>7}")
    print("-" * 90)
    for r in records:
        agree = "✓" if r["agree"] else ("~" if r["paddle_prov"] != "?" else "✗")
        print(
            f"{r['frame']:>6}  {r['hyper_plate']:^14} {r['hyper_conf']:>5.3f}  "
            f"{r['paddle_raw'][:20]:^20}  "
            f"{r['paddle_plate']:^12}  "
            f"{r['hyper_prov']:^4} {r['paddle_prov']:^4}  {agree:^4}  {r['size']:>7}"
        )

    total_rec = len(records)
    agree_cnt = sum(r["agree"] for r in records)
    paddle_hit = sum(bool(r["paddle_plate"]) for r in records)
    paddle_prov_valid = sum(r["paddle_prov"] != "?" for r in records)

    print("=" * 90)
    print(f"共采样车牌       : {total_rec}")
    print(f"Paddle 识别出车牌: {paddle_hit}/{total_rec} ({100*paddle_hit/total_rec:.1f}%)" if total_rec else "")
    print(f"Paddle 省份合法  : {paddle_prov_valid}/{total_rec}")
    print(f"两者省份一致     : {agree_cnt}/{total_rec} ({100*agree_cnt/total_rec:.1f}%)" if total_rec else "")

    # 重点：HyperLPR3 认为是辽/沪时，Paddle 说什么
    confused = [r for r in records if r["hyper_prov"] in ("辽", "沪")]
    if confused:
        print(f"\nHyperLPR3 认为是辽/沪的样本 ({len(confused)} 个):")
        for r in confused:
            print(f"  帧{r['frame']:>6}  HY={r['hyper_plate']:<12} "
                  f"PD={r['paddle_plate'] or '—':<12}  "
                  f"PaddleRaw={r['paddle_raw'][:30]}")


if __name__ == "__main__":
    main()
