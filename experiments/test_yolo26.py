"""
YOLO26 交通视频目标检测测试脚本（MPS 加速）
"""
import cv2
import time
import sys
from ultralytics import YOLO

VIDEO_PATH  = "/Users/liujiahang/科研/交通流算法/test_video.mp4"
OUTPUT_PATH = "/Users/liujiahang/科研/交通流算法/output_test_yolo26x.mp4"
DEVICE      = "mps"   # Apple Silicon GPU，改为 "cpu" 可回退
MAX_FRAMES  = 1000    # 只处理前N帧，设为 None 则处理全部

# 只保留交通相关类别（COCO id）
TRAFFIC_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

CLASS_COLORS = {
    "car":        (0,   255,   0),
    "truck":      (0,   128, 255),
    "bus":        (255, 128,   0),
    "motorcycle": (255,   0, 255),
    "bicycle":    (0,   255, 255),
    "person":     (255, 255,   0),
}


def run_detection():
    model = YOLO("yolo26x.pt")
    model.to(DEVICE)
    print(f"推理设备: {DEVICE}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"无法打开视频: {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if MAX_FRAMES is not None:
        total_frames = min(total_frames, MAX_FRAMES)
    fps_video    = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频信息: {width}x{height}  {fps_video:.1f}fps  共{total_frames}帧")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_video, (width, height))

    frame_count    = 0
    fps_list       = []
    vehicle_counts = {cls: 0 for cls in TRAFFIC_CLASSES.values()}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0      = time.perf_counter()
        results = model(frame, classes=list(TRAFFIC_CLASSES.keys()),
                        device=DEVICE, verbose=False)[0]
        elapsed = time.perf_counter() - t0
        cur_fps = 1.0 / elapsed
        fps_list.append(cur_fps)

        # 绘制检测框
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in TRAFFIC_CLASSES:
                continue
            label  = TRAFFIC_CLASSES[cls_id]
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color  = CLASS_COLORS.get(label, (255, 255, 255))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            vehicle_counts[label] += 1

        # 左上角显示实时 FPS 和检测数量
        n_det = len(results.boxes)
        cv2.putText(frame, f"FPS: {cur_fps:.1f}  Det: {n_det}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        writer.write(frame)   # 逐帧流式写出，不缓存在内存
        frame_count += 1
        if MAX_FRAMES is not None and frame_count >= MAX_FRAMES:
            break

        if frame_count % 30 == 0:
            avg30 = sum(fps_list[-30:]) / 30
            pct = frame_count / total_frames * 100
            print(f"[{pct:5.1f}%] 帧 {frame_count:4d}/{total_frames}  "
                  f"近30帧均推理FPS: {avg30:.1f}", flush=True)

    cap.release()
    writer.release()

    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    min_fps = min(fps_list) if fps_list else 0
    max_fps = max(fps_list) if fps_list else 0

    print(f"\n=== 检测完成 ===")
    print(f"处理帧数    : {frame_count}")
    print(f"平均推理FPS : {avg_fps:.1f}  (min {min_fps:.1f} / max {max_fps:.1f})")
    print(f"比赛要求    : {'✓ 达标(≥15fps)' if avg_fps >= 15 else '✗ 未达标(<15fps)'}")
    print(f"\n各类别累计检测数:")
    for cls, cnt in vehicle_counts.items():
        if cnt > 0:
            print(f"  {cls:12s}: {cnt}")
    print(f"\n输出视频: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_detection()
