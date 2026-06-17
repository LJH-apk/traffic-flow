"""车牌识别与多帧投票。"""
from __future__ import annotations

import re

import numpy as np

PROVINCE_CHARS = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁"
PLATE_FULL_PATTERN = re.compile(rf"[{PROVINCE_CHARS}][A-Z][A-Z0-9]{{5}}")
PLATE_BODY_PATTERN = re.compile(r"[A-Z][A-Z0-9]{5}$")

PLATE_CONF_THRESH = 0.85
VOTE_MIN = 3
VOTE_MAX = 7
PLATE_BODY_MIN_VOTES = 3
PLATE_BODY_MIN_RATIO = 0.50
PLATE_PROVINCE_MIN_VOTES = 2
PLATE_PROVINCE_MIN_RATIO = 0.60
PLATE_MIN_BOTTOM_RATIO = 0.55
PLATE_MIN_BOX_W = 80
PLATE_MIN_BOX_H = 80
PLATE_ROI_TOP_RATIO = 0.45
PLATE_VEHICLE_CLASSES = {"car", "bus", "truck"}

PlateBox = tuple[float, float, float, float]
VoteEntry = tuple[str, float, PlateBox | None]


def rel_to_abs(
    rel_box: PlateBox,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[int, int, int, int]:
    rx1, ry1, rx2, ry2 = rel_box
    vw, vh = x2 - x1, y2 - y1
    return (
        x1 + int(rx1 * vw),
        y1 + int(ry1 * vh),
        x1 + int(rx2 * vw),
        y1 + int(ry2 * vh),
    )


class PlateRecognizer:
    """使用 HyperLPR3 识别车牌，并通过多帧投票锁定结果。"""

    def __init__(self) -> None:
        self._cache: dict[int, tuple[str, PlateBox | None]] = {}
        self._pending: dict[int, list[VoteEntry]] = {}
        self._catcher = None
        self._enabled = False

        try:
            import hyperlpr3 as lpr3  # type: ignore
            self._catcher = lpr3.LicensePlateCatcher()
            self._enabled = True
            print("[PlateRecognizer] HyperLPR3 加载成功")
        except ImportError:
            print("[PlateRecognizer] 未找到 hyperlpr3，车牌识别已禁用")

    @property
    def cache(self) -> dict[int, tuple[str, PlateBox | None]]:
        return self._cache

    def recognize(
        self,
        frame: np.ndarray,
        track_id: int,
        box_xyxy: tuple[int, int, int, int],
    ) -> tuple[str, PlateBox | None]:
        if track_id in self._cache:
            return self._cache[track_id]
        if not self._enabled:
            return "", None

        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = box_xyxy
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(fw, x2), min(fh, y2)
        crop_w, crop_h = cx2 - cx1, cy2 - cy1
        if crop_w <= 0 or crop_h <= 0:
            return "", None
        if cy2 / fh < PLATE_MIN_BOTTOM_RATIO:
            return "", None
        if crop_w < PLATE_MIN_BOX_W or crop_h < PLATE_MIN_BOX_H:
            return "", None

        roi_y1 = cy1 + int(crop_h * PLATE_ROI_TOP_RATIO)
        crop = frame[roi_y1:cy2, cx1:cx2]
        roi_h = cy2 - roi_y1
        if roi_h <= 0:
            return "", None

        try:
            results = self._catcher(crop)
        except Exception:
            return "", None

        for item in results or []:
            text, conf = item[0], float(item[1])
            if conf < PLATE_CONF_THRESH:
                continue
            plate = self._match_plate(text)
            if not plate:
                continue

            px1, py1, px2, py2 = item[3]
            rel_box = (
                px1 / crop_w,
                (roi_y1 - cy1 + py1) / crop_h,
                px2 / crop_w,
                (roi_y1 - cy1 + py2) / crop_h,
            )

            votes = self._pending.setdefault(track_id, [])
            votes.append((plate, conf, rel_box))
            if len(votes) > VOTE_MAX:
                votes.pop(0)

        votes = self._pending.get(track_id, [])
        if len(votes) >= VOTE_MIN:
            result, stable = self._tally(votes)
            if stable or len(votes) >= VOTE_MAX:
                self._cache[track_id] = result
                del self._pending[track_id]
                return result

        return "", None

    @staticmethod
    def _tally(votes: list[VoteEntry]) -> tuple[tuple[str, PlateBox | None], bool]:
        from collections import Counter

        provinces: list[str] = []
        bodies: list[str] = []
        for plate, _, _ in votes:
            province, body = PlateRecognizer._split_plate(plate)
            if province:
                provinces.append(province)
            if body:
                bodies.append(body)

        body = ""
        body_stable = False
        if bodies:
            body, body_votes = Counter(bodies).most_common(1)[0]
            body_stable = (
                body_votes >= PLATE_BODY_MIN_VOTES
                and body_votes / len(bodies) >= PLATE_BODY_MIN_RATIO
            )

        province = ""
        province_stable = False
        if provinces:
            province, province_votes = Counter(provinces).most_common(1)[0]
            province_stable = (
                province_votes >= PLATE_PROVINCE_MIN_VOTES
                and province_votes / len(provinces) >= PLATE_PROVINCE_MIN_RATIO
            )

        plate = (province + body) if (province_stable and body) else body
        rel_box = max(votes, key=lambda v: v[1])[2]
        return (plate, rel_box), (body_stable and province_stable)

    @staticmethod
    def _split_plate(plate: str) -> tuple[str, str]:
        if len(plate) >= 7 and plate[0] in PROVINCE_CHARS:
            return plate[0], plate[1:7]
        if len(plate) >= 6:
            return "", plate[-6:]
        return "", ""

    @staticmethod
    def _match_plate(text: str) -> str:
        clean = text.replace(" ", "").upper()
        full_match = PLATE_FULL_PATTERN.search(clean)
        if full_match:
            return full_match.group()
        body_match = PLATE_BODY_PATTERN.search(clean)
        if body_match:
            return body_match.group()
        return ""
