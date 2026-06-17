"""
用真实视频帧测试省份字符分类器。

从视频中随机抽帧，HyperLPR3 检测车牌，截取省份字符 crop，
分别打印 HyperLPR3 结果 和 CNN 分类器结果，方便对比。

用法：
    python3 test_province_clf.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.province_clf import predict, is_available, PROVINCE_CHARS, IMG_SIZE

VIDEO   = "北进口_20260420075959至20260420081500.mp4"
N_FRAME = 500   # 随机采样帧数范围
STEP    = 10    # 每隔多少帧检测一次

def crop_province(frame, box_xyxy, rel_box):
    """从车辆 bbox + 车牌相对坐标中裁出省份字符灰度图。"""
    x1, y1, x2, y2 = box_xyxy
    bw, bh = x2 - x1, y2 - y1
    px1 = max(0, int(x1 + rel_box[0] * bw))
    py1 = max(0, int(y1 + rel_box[1] * bh))
    px2 = min(frame.shape[1], int(x1 + rel_box[2] * bw))
    py2 = min(frame.shape[0], int(y1 + rel_box[3] * bh))
    plate = frame[py1:py2, px1:px2]
    if plate.size == 0:
        return None
    pw = plate.shape[1]
    # 省份字符约占车牌宽度前 20%
    prov = plate[:, : max(1, pw * 2 // 10)]
    return cv2.cvtColor(prov, cv2.COLOR_BGR2GRAY)

def main():
    if not is_available():
        print("province_clf.pt 未找到，请先运行 train_province_clf.py")
        return

    try:
        import hyperlpr3 as lpr3
        catcher = lpr3.LicensePlateCatcher()
    except ImportError:
        print("hyperlpr3 未安装")
        return

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"无法打开视频: {VIDEO}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频共 {total} 帧，每 {STEP} 帧检测一次\n")

    found = 0
    for fi in range(0, min(N_FRAME, total), STEP):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        results = catcher(frame)
        for item in (results or []):
            text, conf = item[0], float(item[1])
            if conf < 0.6:
                continue

            # 模拟 rel_box（这里用车牌在整帧中的绝对坐标，转换为"整帧"rel）
            px1, py1, px2, py2 = item[3]
            fh, fw = frame.shape[:2]
            box_xyxy = (0, 0, fw, fh)
            rel_box  = (px1/fw, py1/fh, px2/fw, py2/fh)

            gray = crop_province(frame, box_xyxy, rel_box)
            if gray is None:
                continue

            cnn_char, cnn_conf = predict(gray, device="cpu")

            lpr_prov = text[0] if text and text[0] in PROVINCE_CHARS else "?"
            match = "✓" if cnn_char == lpr_prov else "✗"

            print(f"帧{fi:5d}  HyperLPR3: {text:8s} conf={conf:.2f}"
                  f"  省份={lpr_prov}"
                  f"  CNN: {cnn_char} conf={cnn_conf:.2f}  {match}")

            # 保存省份 crop 便于目视检查
            save = cv2.resize(gray, (IMG_SIZE*4, IMG_SIZE*4), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"outputs/prov_f{fi}_{text[:3]}.png", save)
            found += 1

    cap.release()
    print(f"\n共找到 {found} 个车牌，crop 已保存至 outputs/prov_*.png")

if __name__ == "__main__":
    main()
