"""
自动斑马线检测。

基于经典 CV 提取白色水平条纹，以 GB 5768.3（条纹宽=间距=0.40m）为标尺，
计算像素坐标 → 路面世界坐标（米）的单应矩阵 H。

当斑马线被遮挡或特征不显著时，返回 None，调用方应退化到 PIXELS_PER_METER 兜底。
"""
import cv2
import numpy as np


class ZebraDetector:
    """自动斑马线检测，输出像素→路面坐标单应矩阵。

    Args:
        min_area:          条纹最小轮廓面积（过滤噪声小块），单位像素²。
        aspect_ratio:      条纹最小宽高比（≥5 才视为横向条纹）。
        regularity_thresh: 条纹间距变异系数阈值，超过此值尝试过滤后重检。
        min_stripes:       最少条纹数，不足则返回 None。
    """

    def __init__(
        self,
        min_area: int = 3000,
        aspect_ratio: float = 5.0,
        regularity_thresh: float = 0.3,
        min_stripes: int = 2,
    ) -> None:
        self.min_area = min_area
        self.aspect_ratio = aspect_ratio
        self.regularity_thresh = regularity_thresh
        self.min_stripes = min_stripes

    def detect(
        self,
        frame: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
    ) -> tuple[np.ndarray, int, list[tuple[int, int, int, int]]] | None:
        """检测斑马线，返回 (H矩阵, 条纹数, 条纹矩形列表) 或 None（检测失败）。

        Args:
            frame: BGR 帧。
            roi:   可选 (x1,y1,x2,y2) 感兴趣区域，限制搜索范围。
                   None 时搜索全帧（通用模式）。

        Returns:
            (H, n_stripes, stripe_rects) 或 None。
            H 是 3×3 单应矩阵，将像素坐标映射到路面米制坐标。
            stripe_rects 是每条条纹的 (x, y, w, h) 列表（原始帧坐标）。
        """
        if roi is not None:
            x1, y1, x2, y2 = roi
            img = frame[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            img = frame
            offset_x, offset_y = 0, 0

        # Step 1: 灰度 + 高斯模糊
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Step 2: 自适应阈值 → 白色区域掩膜（应对不同光照/阴影）
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, -10,
        )

        # Step 3: 水平形态学闭运算 → 强化水平条纹，抑制竖向噪声
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
        h_processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)

        # Step 4: 轮廓检测
        contours, _ = cv2.findContours(
            h_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Step 5: 过滤候选条纹（面积 + 宽高比）
        stripes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if cv2.contourArea(cnt) < self.min_area:
                continue
            if h == 0 or w / h < self.aspect_ratio:
                continue
            stripes.append({
                "x":  x + offset_x,
                "y":  y + offset_y,
                "w":  w,
                "h":  h,
                "cy": y + h / 2.0 + offset_y,
            })

        if len(stripes) < self.min_stripes:
            return None

        # 按 cy 排序
        stripes.sort(key=lambda s: s["cy"])

        # Step 6: 等间距检验，不规则时尝试过滤
        if len(stripes) >= 2:
            cys = [s["cy"] for s in stripes]
            gaps = [cys[i + 1] - cys[i] for i in range(len(cys) - 1)]
            mean_gap = float(np.mean(gaps))
            if mean_gap > 0:
                regularity = float(np.std(gaps)) / mean_gap
                if regularity > self.regularity_thresh and len(stripes) > 3:
                    stripes = self._filter_regular(stripes)
                    if len(stripes) < self.min_stripes:
                        return None

        n_stripes = len(stripes)
        mean_stripe_h = float(np.mean([s["h"] for s in stripes]))

        # Step 7: 提取4角点（像素坐标）
        top_stripe = stripes[0]
        bot_stripe = stripes[-1]

        # X 范围取各条纹交集，保证矩形有效
        min_x = max(s["x"] for s in stripes)
        max_x = min(s["x"] + s["w"] for s in stripes)
        if max_x <= min_x:
            min_x = min(s["x"] for s in stripes)
            max_x = max(s["x"] + s["w"] for s in stripes)

        src_pts = np.float32([
            [min_x, top_stripe["y"]],
            [max_x, top_stripe["y"]],
            [max_x, bot_stripe["y"] + bot_stripe["h"]],
            [min_x, bot_stripe["y"] + bot_stripe["h"]],
        ])

        # Step 8: 世界坐标（米，GB 5768.3：条纹宽=间距=0.40m）
        world_height = (2 * n_stripes - 1) * 0.40
        pixel_width = float(max_x - min_x)
        if mean_stripe_h > 0:
            world_width = pixel_width * (0.40 / mean_stripe_h)
        else:
            world_width = pixel_width / 85.0  # 兜底估算

        dst_pts = np.float32([
            [0,           0           ],
            [world_width, 0           ],
            [world_width, world_height],
            [0,           world_height],
        ])

        # Step 9: 计算单应矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None

        stripe_rects = [(s["x"], s["y"], s["w"], s["h"]) for s in stripes]
        return H, n_stripes, stripe_rects

    def _filter_regular(self, stripes: list[dict]) -> list[dict]:
        """去除破坏等间距的条纹，保留最长的规则连续子序列。"""
        if len(stripes) <= 2:
            return stripes
        cys = [s["cy"] for s in stripes]
        gaps = [cys[i + 1] - cys[i] for i in range(len(cys) - 1)]
        target_gap = float(np.median(gaps))
        best_start, best_len = 0, 1
        cur_start, cur_len = 0, 1
        for i, gap in enumerate(gaps):
            if target_gap > 0 and abs(gap - target_gap) / target_gap <= 0.4:
                cur_len += 1
                if cur_len > best_len:
                    best_start, best_len = cur_start, cur_len
            else:
                cur_start = i + 1
                cur_len = 1
        return stripes[best_start: best_start + best_len]

    def visualize(self, frame: np.ndarray, result) -> np.ndarray:
        """在帧上叠加检测结果（调试用）。"""
        vis = frame.copy()
        if result is None:
            cv2.putText(vis, "Zebra: NOT FOUND", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            H, n, stripe_rects = result
            for x, y, w, h in stripe_rects:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 200), 2)
            cv2.putText(vis, f"Zebra: {n} stripes, H computed", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return vis
