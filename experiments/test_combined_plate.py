"""
组合策略测试：HyperLPR3 识别主体 + PaddleOCR 纠正省份字符

流程：
  HyperLPR3 识别完整车牌（定位框 + 城市码+5位）
  → 裁出省份字符区域（车牌宽 1/7）放大送 PaddleOCR
  → PaddleOCR 返回文字中提取第一个合法省份字符
  → 用 PaddleOCR 省份替换 HyperLPR3 省份，得到最终车牌

运行：python3 test_combined_plate.py
"""

import sys
import re
import cv2
import numpy as np

PROVINCES     = set("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁")
VIDEO_PATH    = "test_video_fixed.mp4"
SAMPLE_FRAMES = 40
CONF_THRESH   = 0.5
PROV_UPSCALE  = 6      # 省份字符放大倍数（单独送 PaddleOCR）

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


def extract_province_from_texts(texts: list[str]) -> str:
    """从 PaddleOCR 返回的文字列表里提取第一个合法省份字符。"""
    for t in texts:
        for ch in t:
            if ch in PROVINCES:
                return ch
    return ""


def main() -> None:
    try:
        import hyperlpr3 as lpr3
        catcher = lpr3.LicensePlateCatcher()
        print("[HyperLPR3] 加载成功")
    except ImportError:
        print("[HyperLPR3] 未安装，退出")
        sys.exit(1)

    import warnings
    warnings.filterwarnings("ignore")
    from paddleocr import PaddleOCR
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

            fh, fw = frame.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in item[3]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            plate_crop = frame[y1:y2, x1:x2]
            h, w = plate_crop.shape[:2]

            # 整张车牌放大后送 PaddleOCR，从结果里提取省份字符
            plate_big = cv2.resize(
                plate_crop,
                (w * PROV_UPSCALE, h * PROV_UPSCALE),
                interpolation=cv2.INTER_CUBIC,
            )

            # PaddleOCR 识别省份字符
            try:
                pd_result = ocr.predict(plate_big)
                pd_texts = []
                for r in (pd_result or []):
                    pd_texts.extend(r.get("rec_texts", []))
                pd_prov = extract_province_from_texts(pd_texts)
                pd_raw  = " | ".join(pd_texts)
            except Exception as e:
                pd_prov, pd_raw = "", f"ERR:{e}"

            # HyperLPR3 结果
            hy_plate = match_plate(text_h)
            hy_prov  = hy_plate[0] if hy_plate and hy_plate[0] in PROVINCES else ""
            hy_body  = hy_plate[1:] if len(hy_plate) > 1 else hy_plate

            # 组合结果：PaddleOCR 省份 + HyperLPR3 主体
            if pd_prov and hy_body:
                combined = pd_prov + hy_body
            elif hy_plate:
                combined = hy_plate
            else:
                combined = ""

            records.append({
                "frame":    idx,
                "hy_plate": hy_plate,
                "hy_prov":  hy_prov,
                "hy_conf":  conf,
                "pd_raw":   pd_raw,
                "pd_prov":  pd_prov,
                "combined": combined,
                "changed":  pd_prov != "" and pd_prov != hy_prov,
                "size":     f"{w}x{h}",
            })

    cap.release()

    # ── 打印报告 ──────────────────────────────────────────────────────────────
    print(f"{'帧':>6}  {'HyperLPR3':^12} {'conf':>5}  {'PD省份raw':^16}  "
          f"{'PD省':^4}  {'最终车牌':^12}  {'省份变更':^6}  {'尺寸':>7}")
    print("-" * 85)
    for r in records:
        changed = "← 已纠正" if r["changed"] else ""
        print(
            f"{r['frame']:>6}  {r['hy_plate']:^12} {r['hy_conf']:>5.3f}  "
            f"{r['pd_raw'][:16]:^16}  "
            f"{r['pd_prov'] or '—':^4}  "
            f"{r['combined']:^12}  "
            f"{changed:<8}  {r['size']:>7}"
        )

    print("=" * 85)
    total_rec  = len(records)
    changed_cnt = sum(r["changed"] for r in records)
    combined_valid = sum(bool(r["combined"]) for r in records)
    pd_prov_hit = sum(bool(r["pd_prov"]) for r in records)

    print(f"共采样车牌          : {total_rec}")
    print(f"PaddleOCR 识别出省份: {pd_prov_hit}/{total_rec} ({100*pd_prov_hit/total_rec:.1f}%)")
    print(f"省份被纠正          : {changed_cnt}/{total_rec} ({100*changed_cnt/total_rec:.1f}%)")
    print(f"最终有效车牌        : {combined_valid}/{total_rec} ({100*combined_valid/total_rec:.1f}%)")

    if changed_cnt:
        print(f"\n省份被纠正的样本：")
        for r in records:
            if r["changed"]:
                print(f"  帧{r['frame']:>6}  {r['hy_plate']:^12}  →  {r['combined']:^12}  "
                      f"(PD raw: {r['pd_raw'][:30]})")


if __name__ == "__main__":
    main()
