"""
RapidOCR 省份字符二次确认 — 独立测试脚本

流程：
  采样视频帧 → HyperLPR3 识别全牌 + 定位车牌框
  → 裁出省份字符区域（第1字符，约占车牌宽 1/7）
  → 放大 4x 送 RapidOCR
  → 编辑距离纠错到最近合法省份
  → 打印对比报告

运行：python3 test_province_ocr.py
"""

import sys
import unicodedata
import cv2
import numpy as np

PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁"
VIDEO_PATH = "test_video_fixed.mp4"
SAMPLE_COUNT = 40        # 采样帧数
CONF_THRESH = 0.5        # HyperLPR3 置信度下限
UPSCALE = 4              # 省份字符放大倍数


def edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j - 1] if a[i-1] == b[j-1] else 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n]


def nearest_province(char: str) -> str:
    """编辑距离最近的合法省份字符。"""
    if char in PROVINCES:
        return char
    best, best_d = char, 999
    for p in PROVINCES:
        d = edit_distance(char, p)
        if d < best_d:
            best, best_d = p, d
    return best


def crop_province_char(
    plate_crop: np.ndarray,
) -> np.ndarray:
    """从车牌 crop 裁出第一个字符区域（省份位）并放大。"""
    h, w = plate_crop.shape[:2]
    char_w = max(1, w // 7)
    char_region = plate_crop[:, :char_w]
    upscaled = cv2.resize(
        char_region,
        (char_w * UPSCALE, h * UPSCALE),
        interpolation=cv2.INTER_CUBIC,
    )
    return upscaled


def run_rapid_ocr(ocr, img: np.ndarray) -> str:
    """调用 RapidOCR，返回识别文字（去空格，取第一段）。"""
    try:
        result, _ = ocr(img)
        if result:
            return result[0][1].replace(" ", "").strip()
    except Exception as e:
        print(f"  [RapidOCR 异常] {e}")
    return ""


def main() -> None:
    try:
        import hyperlpr3 as lpr3
        catcher = lpr3.LicensePlateCatcher()
        print("[HyperLPR3] 加载成功")
    except ImportError:
        print("[HyperLPR3] 未安装，退出")
        sys.exit(1)

    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    print("[RapidOCR]  加载成功")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"无法打开视频: {VIDEO_PATH}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = [int(total * i / SAMPLE_COUNT) for i in range(SAMPLE_COUNT)]

    records = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        results = catcher(frame) or []
        for item in results:
            text, conf = item[0], float(item[1])
            if conf < CONF_THRESH or not text:
                continue

            box = item[3]
            x1, y1, x2, y2 = [int(v) for v in box]
            fh, fw = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            plate_crop = frame[y1:y2, x1:x2]
            province_img = crop_province_char(plate_crop)

            rapid_raw = run_rapid_ocr(ocr, province_img)
            # 只取第一个字符做省份匹配
            rapid_char = rapid_raw[0] if rapid_raw else ""
            corrected = nearest_province(rapid_char) if rapid_char else ""

            hyper_province = text[0] if text else ""
            hyper_valid = hyper_province in PROVINCES

            records.append({
                "frame":           idx,
                "hyper_full":      text,
                "hyper_conf":      conf,
                "hyper_province":  hyper_province,
                "hyper_valid":     hyper_valid,
                "rapid_raw":       rapid_raw,
                "rapid_char":      rapid_char,
                "corrected":       corrected,
                "agree":           hyper_province == corrected,
                "plate_size":      f"{x2-x1}x{y2-y1}",
            })

    cap.release()

    # ── 打印报告 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'帧':>6}  {'HyperLPR3':^16}  {'conf':>5}  {'省份合法':^6}  "
          f"{'RapidOCR原始':^12}  {'纠错后':^4}  {'一致':^4}  {'车牌尺寸':>8}")
    print("-" * 72)
    for r in records:
        agree_mark = "✓" if r["agree"] else "✗"
        valid_mark = "✓" if r["hyper_valid"] else "✗"
        print(
            f"{r['frame']:>6}  {r['hyper_full']:^16}  {r['hyper_conf']:>5.3f}  "
            f"{valid_mark:^6}  {r['rapid_raw'] or '—':^12}  "
            f"{r['corrected'] or '—':^4}  {agree_mark:^4}  {r['plate_size']:>8}"
        )

    print("=" * 72)
    total_rec = len(records)
    hyper_valid_cnt = sum(r["hyper_valid"] for r in records)
    agree_cnt = sum(r["agree"] for r in records)
    rapid_hit_cnt = sum(bool(r["rapid_char"]) for r in records)
    print(f"\n共识别车牌: {total_rec}")
    print(f"HyperLPR3 省份合法率: {hyper_valid_cnt}/{total_rec} "
          f"({100*hyper_valid_cnt/total_rec:.1f}%)" if total_rec else "")
    print(f"RapidOCR 有返回结果:  {rapid_hit_cnt}/{total_rec}")
    print(f"两者省份一致:         {agree_cnt}/{total_rec} "
          f"({100*agree_cnt/total_rec:.1f}%)" if total_rec else "")
    if not records:
        print("未识别到任何车牌，请检查视频路径或 HyperLPR3 配置。")


if __name__ == "__main__":
    main()
