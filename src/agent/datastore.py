"""
交通数据接入层：为 AI 分析智能体提供统一的查询与聚合接口。

数据来源（与 build_data.py 保持同一优先级）：
  - outputs/cross_section*_full.csv   过线事件（流量/速度/车头时距的事实表）
  - outputs/vehicle_stats*.csv        轨迹级速度统计
  - outputs/dashboard/*.json          预聚合结果（overview / charts / meta）

所有查询方法返回可 JSON 序列化的 dict，供 function calling 直接回喂给 LLM。
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.config.settings import (
    HEADWAY_DANGEROUS_S,
    OUTPUT_DIR,
    PCU_FACTORS,
    QUEUE_GAP_M,
    QUEUE_SPEED_THRESH_KMH,
    SATURATION_FLOW_PCU_H,
    SPEED_VALID_MAX_KMH,
    VEHICLE_LENGTHS_M,
    CLASS_NAMES_ZH,
)
from src.dashboard.build_data import (
    CROSS_SECTION_FALLBACKS,
    CROSS_SECTION_SOURCES,
    VEHICLE_STATS_SOURCES,
    _load_all_from_sources,
)

ENTRANCES = ("北进口", "南进口", "东进口")
ENTRANCE_TO_SCOPE = {"北进口": "north", "南进口": "south", "东进口": "east"}
DASHBOARD_DIR = OUTPUT_DIR / "dashboard"

# 速度直方图分箱（km/h）
SPEED_BINS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, 120)]


def _f(value, default=None):
    """安全转 float，空串/异常返回 default。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v


def _r(value, digits=1):
    return None if value is None else round(value, digits)


def _percentile(sorted_values: list[float], p: float):
    """线性插值分位数，sorted_values 需已排序且非空。"""
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def _mean(values):
    return sum(values) / len(values) if values else None


def _median(values_sorted):
    return _percentile(values_sorted, 0.5)


def entrance_of(section: str) -> str:
    for name in ENTRANCES:
        if section.startswith(name):
            return name
    return "未知"


class TrafficDataStore:
    """加载一次、内存聚合。所有查询容忍数据缺失（返回带 error 说明的 dict）。"""

    def __init__(self, outputs_dir: Path | None = None):
        self.outputs_dir = Path(outputs_dir) if outputs_dir else OUTPUT_DIR
        self._loaded = False
        self._events: list[dict] = []          # 过线事件（已解析数值）
        self._vehicle_stats: list[dict] = []   # 轨迹级统计
        self._dashboard_cache: dict[str, dict | None] = {}

    # ── 加载 ─────────────────────────────────────────────────────────────

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.reload()

    def reload(self) -> None:
        self._dashboard_cache.clear()
        self._events = [self._parse_event(r) for r in
                        _load_all_from_sources(CROSS_SECTION_SOURCES, CROSS_SECTION_FALLBACKS)]
        self._events = [e for e in self._events if e is not None]
        self._events.sort(key=lambda e: e["ts"])
        self._vehicle_stats = _load_all_from_sources([], VEHICLE_STATS_SOURCES)
        self._loaded = True

    @staticmethod
    def _parse_event(row: dict) -> dict | None:
        ts = _f(row.get("timestamp_s"))
        section = (row.get("section") or "").strip()
        if ts is None or not section:
            return None
        return {
            "ts": ts,
            "section": section,
            "entrance": entrance_of(section),
            "track_id": row.get("track_id", ""),
            "class_name": (row.get("class_name") or "").strip(),
            "lane_id": (row.get("lane_id") or "").strip(),
            "vehicle_category": (row.get("vehicle_category") or "").strip(),
            "color": (row.get("color") or "").strip(),
            "direction": (row.get("direction") or "").strip(),
            "speed": _f(row.get("speed_kmh")),
            "headway": _f(row.get("headway_s")),
            "spacing": _f(row.get("spacing_m")),
            "plate": (row.get("plate") or "").strip(),
        }

    def dashboard_json(self, name: str, entrance: str | None = None) -> dict | None:
        """读 outputs/dashboard/{name}[_scope].json，缺失返回 None。"""
        scope = ENTRANCE_TO_SCOPE.get(entrance or "")
        key = f"{name}_{scope}" if scope else name
        if key not in self._dashboard_cache:
            path = DASHBOARD_DIR / f"{key}.json"
            try:
                self._dashboard_cache[key] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._dashboard_cache[key] = None
        return self._dashboard_cache[key]

    # ── 基础属性 ──────────────────────────────────────────────────────────

    @property
    def data_ready(self) -> bool:
        self.ensure_loaded()
        return bool(self._events)

    def _filtered(self, entrance=None, section=None, class_name=None,
                  direction=None, only_main=False) -> list[dict]:
        self.ensure_loaded()
        out = []
        for e in self._events:
            if entrance and e["entrance"] != entrance:
                continue
            if section and section not in e["section"]:
                continue
            if class_name and e["class_name"] != class_name:
                continue
            if direction and e["direction"] != direction:
                continue
            if only_main and "主断面" not in e["section"]:
                continue
            out.append(e)
        return out

    def _duration_s(self) -> float:
        self.ensure_loaded()
        if not self._events:
            return 0.0
        return max(1.0, self._events[-1]["ts"] - self._events[0]["ts"])

    # ── 查询：总体概况 ────────────────────────────────────────────────────

    def overview(self) -> dict:
        self.ensure_loaded()
        if not self._events:
            return {"error": "暂无过线事件数据，请先运行检测（run-all）生成 outputs/ 数据"}
        ov = self.dashboard_json("overview") or {}
        by_class = Counter(e["class_name"] for e in self._events if e["class_name"])
        by_entrance = Counter(e["entrance"] for e in self._events)
        by_direction = Counter(e["direction"] for e in self._events if e["direction"])
        duration = self._duration_s()
        return {
            "说明": "数据来自丁字路口三进口（北/南/东）早高峰约15分钟监控视频的过线事件",
            "视频时长_s": _r(duration),
            "轨迹总数": ov.get("unique_tracks"),
            "过线事件总数": len(self._events),
            "小时流量估算_辆每小时": _r(len(self._filtered(only_main=True, direction="到达")) * 3600 / duration, 0),
            "分车型事件数": {CLASS_NAMES_ZH.get(k, k): v for k, v in by_class.most_common()},
            "分进口事件数": dict(by_entrance.most_common()),
            "分方向事件数": dict(by_direction.most_common()),
            "分断面事件数": dict(Counter(e["section"] for e in self._events).most_common()),
        }

    # ── 查询：流量时间序列 ────────────────────────────────────────────────

    def flow_series(self, entrance=None, section=None, class_name=None,
                    direction="到达", bucket_s=60, only_main=True) -> dict:
        bucket_s = max(10, int(bucket_s or 60))
        events = self._filtered(entrance, section, class_name, direction, only_main)
        if not events:
            return {"error": "该筛选条件下没有过线事件", "筛选": {"进口": entrance, "断面": section,
                    "车型": class_name, "方向": direction}}
        buckets: dict[int, int] = defaultdict(int)
        for e in events:
            buckets[int(e["ts"] // bucket_s)] += 1
        last = max(buckets)
        counts = [buckets.get(i, 0) for i in range(last + 1)]
        peak_i = max(range(len(counts)), key=counts.__getitem__)
        total = len(events)
        duration = self._duration_s()
        return {
            "筛选": {"进口": entrance or "全部", "方向": direction or "全部",
                    "车型": class_name or "全部", "仅主断面": only_main},
            "时间粒度_s": bucket_s,
            "各时段过车数": counts,
            "总过车数": total,
            "平均流率_辆每小时": _r(total * 3600 / duration, 0),
            "高峰时段": {"起点_s": peak_i * bucket_s, "终点_s": (peak_i + 1) * bucket_s,
                       "过车数": counts[peak_i],
                       "折算小时流率_辆每小时": _r(counts[peak_i] * 3600 / bucket_s, 0)},
        }

    # ── 查询：速度统计 ────────────────────────────────────────────────────

    def speed_stats(self, entrance=None, section=None, class_name=None) -> dict:
        events = self._filtered(entrance, section, class_name)
        raw = [e["speed"] for e in events if e["speed"] is not None]
        valid = sorted(v for v in raw if 0 < v <= SPEED_VALID_MAX_KMH)
        if not valid:
            return {"error": "该筛选条件下没有有效速度样本"}
        hist = {f"{lo}-{hi}": sum(1 for v in valid if lo <= v < hi) for lo, hi in SPEED_BINS}
        return {
            "筛选": {"进口": entrance or "全部", "车型": class_name or "全部"},
            "有效样本数": len(valid),
            "平均速度_kmh": _r(_mean(valid)),
            "中位速度_kmh": _r(_median(valid)),
            "85分位速度_kmh": _r(_percentile(valid, 0.85)),
            "15分位速度_kmh": _r(_percentile(valid, 0.15)),
            "最高有效速度_kmh": _r(valid[-1]),
            "速度直方图": hist,
            "低速占比_低于10kmh": _r(sum(1 for v in valid if v < 10) / len(valid), 3),
            "异常值_超过120kmh_条数": sum(1 for v in raw if v > SPEED_VALID_MAX_KMH),
            "过线瞬时速度为0_条数": sum(1 for v in raw if v == 0),
        }

    # ── 查询：车头时距 / 车头间距 ─────────────────────────────────────────

    def headway_stats(self, entrance=None, section=None) -> dict:
        events = self._filtered(entrance, section)
        hw_events = [e for e in events
                     if e["headway"] is not None and 0 < e["headway"] <= 120]
        hw = sorted(e["headway"] for e in hw_events)
        sp = sorted(e["spacing"] for e in events
                    if e["spacing"] is not None and 0 < e["spacing"] <= 200)
        if not hw:
            return {"error": "该筛选条件下没有有效车头时距样本"}

        short = [e for e in hw_events if e["headway"] < HEADWAY_DANGEROUS_S]
        short_with_speed = [e for e in short if e["speed"] is not None and e["speed"] > 0]
        true_danger = [e for e in short_with_speed if e["speed"] < QUEUE_SPEED_THRESH_KMH]
        platoon = [e for e in short_with_speed if e["speed"] >= QUEUE_SPEED_THRESH_KMH]

        return {
            "筛选": {"进口": entrance or "全部", "断面": section or "全部"},
            "车头时距": {
                "样本数": len(hw),
                "平均_s": _r(_mean(hw), 2),
                "中位_s": _r(_median(hw), 2),
                "最小_s": _r(hw[0], 2),
                f"短车头时距占比_小于{HEADWAY_DANGEROUS_S}s": _r(len(short) / len(hw), 3),
            },
            "短车头时距细分": {
                "说明": ("短车头时距须结合速度区分：低速且近距离才是真实危险跟驰（拥堵跟车）；"
                        "正常车速下的近距离通常是信号放行车队集中通过，是高效放行而非危险，"
                        "不应笼统计为'危险跟驰'"),
                "真实危险跟驰_低速且近距离_占比": (
                    _r(len(true_danger) / len(short_with_speed), 3) if short_with_speed else None),
                "放行车队_正常车速且近距离_占比": (
                    _r(len(platoon) / len(short_with_speed), 3) if short_with_speed else None),
            },
            "车头间距": {
                "样本数": len(sp),
                "平均_m": _r(_mean(sp)),
                "中位_m": _r(_median(sp)),
            } if sp else {"样本数": 0},
        }

    # ── 查询：车流时间模式（波浪放行 vs 持续拥堵）────────────────────────────

    def flow_pattern(self, entrance=None, section=None, bucket_s=10) -> dict:
        events = self._filtered(entrance=entrance, section=section,
                                direction="到达", only_main=not section)
        if not events:
            return {"error": "该筛选条件下没有到达事件"}
        bucket_s = max(5, int(bucket_s or 10))
        buckets: dict[int, int] = defaultdict(int)
        for e in events:
            buckets[int(e["ts"] // bucket_s)] += 1
        last = max(buckets)
        counts = [buckets.get(i, 0) for i in range(last + 1)]
        mean = _mean(counts) or 0.0
        std = (sum((c - mean) ** 2 for c in counts) / len(counts)) ** 0.5
        cv = _r(std / mean, 2) if mean else None
        empty_ratio = sum(1 for c in counts if c == 0) / len(counts)

        if cv is not None and cv >= 0.5 and empty_ratio >= 0.1:
            pattern = "波浪式放行（信号周期性释放特征明显，符合信号交叉口正常规律）"
        elif cv is not None and cv < 0.35:
            pattern = "持续均匀车流"
        else:
            pattern = "介于波浪与持续之间"
        return {
            "筛选": {"进口": entrance or "全部", "断面": section or "全部", "分桶粒度_s": bucket_s},
            "分桶过车数": counts,
            "变异系数CV": cv,
            "空档桶占比": _r(empty_ratio, 2),
            "模式判断": pattern,
            "说明": ("波浪式放行（车流忽多忽少、有明显空档）是信号交叉口绿灯集中放行、红灯断流的正常特征，"
                    "不等同于拥堵；判断是否拥堵应结合排队通过占比（estimate_queue）与低速占比，"
                    "而非仅凭车流本身的波动性或断面速度偏低（断面临近停止线，车辆本就会减速）"),
        }

    # ── 查询：车道利用 ────────────────────────────────────────────────────

    def lane_usage(self, entrance=None) -> dict:
        events = self._filtered(entrance)
        if not events:
            return {"error": "该筛选条件下没有事件"}
        counter = Counter(e["lane_id"] for e in events if e["lane_id"])
        known = {k: v for k, v in counter.items() if k.isdigit()}
        result = {
            "筛选": {"进口": entrance or "全部"},
            "各车道过车数": dict(sorted(known.items())),
            "对向车道_OPPOSITE": counter.get("OPPOSITE", 0),
            "未识别车道_UNKNOWN": counter.get("UNKNOWN", 0),
        }
        if len(known) >= 2:
            busiest = max(known, key=known.get)
            idlest = min(known, key=known.get)
            result["最忙车道"] = {"车道": busiest, "过车数": known[busiest]}
            result["最闲车道"] = {"车道": idlest, "过车数": known[idlest]}
            result["车道不均衡系数_最忙除以最闲"] = _r(known[busiest] / max(1, known[idlest]))
        return result

    # ── 查询：转向构成 ────────────────────────────────────────────────────

    def turn_movements(self, entrance=None) -> dict:
        charts = self.dashboard_json("charts", entrance)
        result: dict = {"筛选": {"进口": entrance or "全部"}}
        if charts and charts.get("turn_groups"):
            tg = charts["turn_groups"]
            zh = {"left_turn": "左转", "right_turn": "右转", "straight": "直行",
                  "non_motor": "非机动车", "wrong_way": "疑似逆行", "unknown": "未识别"}
            result["轨迹分组转向构成"] = {
                zh.get(c, c): n for c, n in zip(tg["categories"], tg["counts"])}
            result["备注"] = ("疑似逆行多为对向车道正常来车（视角原因）与跟踪ID跳变所致，"
                            "属数据清洗对象，不代表真实违法量")
        turn_sections = Counter()
        for e in self._filtered(entrance):
            for kw in ("右转", "掉头"):
                if kw in e["section"] and e["direction"] == kw:
                    turn_sections[e["section"]] += 1
        if turn_sections:
            result["专用断面转向过车数"] = dict(turn_sections)
        if len(result) == 1:
            result["error"] = "无转向数据（dashboard/charts.json 未构建）"
        return result

    # ── 查询：事件明细采样 ────────────────────────────────────────────────

    def sample_events(self, entrance=None, section=None, class_name=None,
                      min_speed=None, max_speed=None, limit=10) -> dict:
        limit = min(int(limit or 10), 30)
        rows = []
        for e in self._filtered(entrance, section, class_name):
            if min_speed is not None and (e["speed"] is None or e["speed"] < min_speed):
                continue
            if max_speed is not None and (e["speed"] is None or e["speed"] > max_speed):
                continue
            rows.append({
                "时刻_s": _r(e["ts"], 2), "断面": e["section"], "track_id": e["track_id"],
                "车型": CLASS_NAMES_ZH.get(e["class_name"], e["class_name"]),
                "车道": e["lane_id"], "颜色": e["color"], "方向": e["direction"],
                "速度_kmh": e["speed"], "车头时距_s": e["headway"], "车牌": e["plate"] or None,
            })
        return {"匹配总数": len(rows), "样本": rows[:limit]}

    # ── 查询：测速可信度交叉验证（到达/离去方向对称性）──────────────────────

    def speed_reliability_check(self) -> dict:
        """
        真实拥堵应表现为'到达慢、离去快'（车辆通过瓶颈后恢复速度）；
        若同一断面到达与离去速度同样极低且接近，通常是该断面单应矩阵标定误差
        导致的测速失真，而非真实拥堵——用于防止把测速伪影误判为交通现象。
        """
        self.ensure_loaded()
        if not self._events:
            return {"error": "暂无数据"}
        result: dict = {
            "说明": ("诊断逻辑：真实拥堵下到达方向慢、离去方向应明显更快（车辆通过瓶颈后加速离开）；"
                    "若到达与离去速度同样极低（<8km/h）且相差不大，是断面标定误差的典型信号，"
                    "该断面的'拥堵'结论不可信，应以精确标定的断面或人工视频复核为准，"
                    "不应笼统据此判定路口拥堵。"),
            "各进口主断面": [],
        }
        sections = sorted({e["section"] for e in self._events if "主断面" in e["section"]})
        for sec in sections:
            arr = [e["speed"] for e in self._events
                   if e["section"] == sec and e["direction"] == "到达"
                   and e["speed"] is not None and 0 <= e["speed"] <= SPEED_VALID_MAX_KMH]
            dep = [e["speed"] for e in self._events
                   if e["section"] == sec and e["direction"] == "离去"
                   and e["speed"] is not None and 0 <= e["speed"] <= SPEED_VALID_MAX_KMH]
            if not arr or not dep:
                continue
            arr_mean, dep_mean = _mean(arr), _mean(dep)
            suspicious = arr_mean < 8 and dep_mean < 8 and abs(arr_mean - dep_mean) < 5
            result["各进口主断面"].append({
                "进口": entrance_of(sec), "断面": sec,
                "到达均速_kmh": _r(arr_mean), "离去均速_kmh": _r(dep_mean),
                "到达样本数": len(arr), "离去样本数": len(dep),
                "可信度判断": ("测速可能失真：到达/离去速度同样极低且接近，疑似标定误差而非真实拥堵"
                             if suspicious else "到达明显慢于离去，符合真实拥堵/正常通行的物理规律，测速可信"),
            })
        return result

    # ── 查询：数据质量 / 异常检测 ─────────────────────────────────────────

    def anomalies(self) -> dict:
        self.ensure_loaded()
        if not self._events:
            return {"error": "暂无数据"}
        n = len(self._events)
        speeds = [e["speed"] for e in self._events if e["speed"] is not None]
        over = sorted((v for v in speeds if v > SPEED_VALID_MAX_KMH), reverse=True)
        zero = sum(1 for v in speeds if v == 0)
        lanes = Counter(e["lane_id"] for e in self._events)
        plates = sum(1 for e in self._events if e["plate"])
        short_tracks = 0
        for r in self._vehicle_stats:
            n_samples = _f(r.get("n_samples"), 0) or 0
            if 0 < n_samples < 5:
                short_tracks += 1
        charts = self.dashboard_json("charts") or {}
        wrong_way = 0
        if charts.get("turn_groups"):
            tg = charts["turn_groups"]
            wrong_way = dict(zip(tg["categories"], tg["counts"])).get("wrong_way", 0)
        return {
            "说明": "以下为检测/跟踪层面的数据质量问题及建议的清洗规则，供分析时剔除干扰",
            "超速异常值": {
                "条数": len(over), "最大值_kmh": _r(over[0]) if over else None,
                "成因": "单应矩阵边缘畸变或跟踪ID跳变导致位移突变",
                "清洗规则": f"速度>{SPEED_VALID_MAX_KMH:.0f}km/h 的事件剔除，不参与速度统计",
            },
            "过线速度为0": {
                "条数": zero, "占比": _r(zero / max(1, len(speeds)), 3),
                "成因": "排队起步缓行或速度窗口样本不足",
                "清洗规则": "速度统计仅取 >0 样本；流量统计保留",
            },
            "车道未识别": {
                "UNKNOWN条数": lanes.get("UNKNOWN", 0),
                "OPPOSITE条数": lanes.get("OPPOSITE", 0),
                "成因": "车辆越线行驶或位于标定车道多边形之外（含对向车道）",
                "清洗规则": "车道级统计仅计入编号车道；OPPOSITE 单列为对向流量",
            },
            "疑似逆行轨迹组": {
                "条数": wrong_way,
                "成因": "对向来车视角误判与ID跳变为主，需人工复核后才可判定违法",
                "清洗规则": "逆行判定叠加位移阈值与持续帧数过滤，短片段丢弃",
            },
            "短命轨迹": {
                "条数": short_tracks,
                "成因": "遮挡、密集小目标导致跟踪断裂（ID跳变）",
                "清洗规则": "速度样本数<5 的轨迹不参与轨迹级速度统计",
            },
            "车牌识别覆盖": {
                "有车牌事件数": plates, "覆盖率": _r(plates / n, 3),
                "成因": "4K远景像素不足，仅近端大目标可识别",
                "清洗规则": "车牌仅作个案追踪，不作为统计口径",
            },
        }

    # ── 查询：信号配时估算（Webster 粗估） ────────────────────────────────

    def signal_timing_estimate(self) -> dict:
        self.ensure_loaded()
        if not self._events:
            return {"error": "暂无数据"}
        duration = self._duration_s()
        approaches = []
        for ent in ENTRANCES:
            arrivals = self._filtered(entrance=ent, direction="到达", only_main=True)
            if not arrivals:
                continue
            pcu = sum(PCU_FACTORS.get(e["class_name"], 1.0) for e in arrivals)
            pcu_h = pcu * 3600 / duration
            lanes = {e["lane_id"] for e in arrivals if e["lane_id"].isdigit()}
            lane_n = max(1, len(lanes))
            y = pcu_h / (lane_n * SATURATION_FLOW_PCU_H)
            approaches.append({"进口": ent, "高峰小时流量_pcu每小时": _r(pcu_h, 0),
                               "有效车道数": lane_n, "流量比y": _r(y, 3)})
        if not approaches:
            return {"error": "主断面到达事件不足，无法估算"}
        big_y = sum(a["流量比y"] for a in approaches)
        lost = 4 * len(approaches)  # 每相位损失约4s
        saturated = big_y >= 0.85
        cycle = None if saturated else max(40.0, min(160.0, (1.5 * lost + 5) / (1 - big_y)))
        if cycle:
            green_total = cycle - lost
            for a in approaches:
                a["建议绿灯_s"] = _r(green_total * a["流量比y"] / big_y, 0)
        return {
            "方法": "Webster 最优周期公式（按进口各设一相位的简化 T 形路口模型）",
            "假设": [f"单车道饱和流率 {SATURATION_FLOW_PCU_H:.0f} pcu/h",
                     "以15分钟观测流量直接折算高峰小时流量",
                     f"每相位损失时间 4s，共 {lost}s",
                     "PCU换算: " + ", ".join(f"{CLASS_NAMES_ZH[k]}={v}" for k, v in PCU_FACTORS.items())],
            "各进口": approaches,
            "总流量比Y": _r(big_y, 3),
            "建议周期_s": _r(cycle, 0) if cycle else None,
            "饱和提示": "Y≥0.85，路口接近或达到饱和，建议渠化改造/需求管理而非单纯调周期" if saturated else None,
        }

    # ── 查询：排队估算 ────────────────────────────────────────────────────

    def queue_estimate(self, entrance=None) -> dict:
        """排队粗估：主断面到达事件中低速过线视为排队状态，按分钟桶取峰值。"""
        ents = [entrance] if entrance else list(ENTRANCES)
        result: dict = {
            "方法": (f"过线速度<{QUEUE_SPEED_THRESH_KMH:.0f}km/h 判为排队通过，"
                    f"排队长度=峰值分钟排队车数×(平均车长+{QUEUE_GAP_M}m)，属粗估"),
            "各进口": [],
        }
        for ent in ents:
            arrivals = self._filtered(entrance=ent, direction="到达", only_main=True)
            known = [e for e in arrivals
                     if e["speed"] is not None and 0 <= e["speed"] <= SPEED_VALID_MAX_KMH]
            if not known:
                continue
            queued = [e for e in known if e["speed"] < QUEUE_SPEED_THRESH_KMH]
            per_min = Counter(int(e["ts"] // 60) for e in queued)
            peak_min, peak_cnt = (per_min.most_common(1)[0] if per_min else (None, 0))
            avg_len = _mean([VEHICLE_LENGTHS_M.get(e["class_name"], 4.5)
                             for e in queued]) or 4.5
            result["各进口"].append({
                "进口": ent,
                "排队通过占比": _r(len(queued) / len(known), 3),
                "峰值分钟": peak_min,
                "峰值分钟排队车数": peak_cnt,
                "估算最大排队长度_m": _r(peak_cnt * (avg_len + QUEUE_GAP_M), 0),
            })
        if not result["各进口"]:
            return {"error": "该筛选条件下没有有效速度样本"}
        return result

    # ── 查询：三进口横向对比 ─────────────────────────────────────────────

    def compare_entrances(self) -> dict:
        self.ensure_loaded()
        if not self._events:
            return {"error": "暂无数据"}
        duration = self._duration_s()
        rows = []
        for ent in ENTRANCES:
            arrivals = self._filtered(entrance=ent, direction="到达", only_main=True)
            ev = self._filtered(entrance=ent)
            if not ev:
                continue
            speeds = sorted(e["speed"] for e in ev
                            if e["speed"] is not None and 0 < e["speed"] <= SPEED_VALID_MAX_KMH)
            hw = [e["headway"] for e in ev
                  if e["headway"] is not None and 0 < e["headway"] <= 120]
            classes = Counter(e["class_name"] for e in ev if e["class_name"])
            top = classes.most_common(1)[0][0] if classes else None
            rows.append({
                "进口": ent,
                "主断面到达_辆": len(arrivals),
                "折算流率_辆每小时": _r(len(arrivals) * 3600 / duration, 0),
                "平均速度_kmh": _r(_mean(speeds)),
                "85分位速度_kmh": _r(_percentile(speeds, 0.85)),
                "危险跟驰占比": _r(sum(1 for v in hw if v < HEADWAY_DANGEROUS_S) / len(hw), 3)
                              if hw else None,
                "最多车型": CLASS_NAMES_ZH.get(top, top),
            })
        ranked = sorted(rows, key=lambda r: -(r["主断面到达_辆"] or 0))
        return {"对比表": rows, "压力排序": [r["进口"] for r in ranked]}

    # ── 前端侧栏 KPI ─────────────────────────────────────────────────────

    def summary_for_ui(self) -> dict:
        self.ensure_loaded()
        ov = self.dashboard_json("overview") or {}
        if not self._events:
            return {"data_ready": False}
        duration = self._duration_s()
        arrivals = self._filtered(direction="到达", only_main=True)
        buckets = Counter(int(e["ts"] // 60) for e in arrivals)
        peak_min, peak_cnt = (buckets.most_common(1)[0] if buckets else (0, 0))
        speeds = sorted(e["speed"] for e in self._events
                        if e["speed"] is not None and 0 < e["speed"] <= SPEED_VALID_MAX_KMH)
        class_counts = Counter(e["class_name"] for e in self._events if e["class_name"])
        return {
            "data_ready": True,
            "unique_tracks": ov.get("unique_tracks"),
            "total_events": len(self._events),
            "duration_s": _r(duration, 0),
            "avg_speed_kmh": _r(_mean(speeds)),
            "p85_speed_kmh": _r(_percentile(speeds, 0.85)),
            "peak_minute": {"minute": peak_min, "count": peak_cnt},
            "hourly_flow": _r(len(arrivals) * 3600 / duration, 0),
            "class_counts": {CLASS_NAMES_ZH.get(k, k): v for k, v in class_counts.most_common()},
            "entrance_counts": dict(Counter(e["entrance"] for e in self._events).most_common()),
            "qc": {
                "speed_outliers": sum(1 for e in self._events
                                      if e["speed"] is not None and e["speed"] > SPEED_VALID_MAX_KMH),
                "zero_speed": sum(1 for e in self._events if e["speed"] == 0),
            },
        }

    # ── 报告用全量摘要 ────────────────────────────────────────────────────

    def full_digest(self) -> dict:
        """一次性汇总所有关键聚合，注入报告生成提示词。"""
        self.ensure_loaded()
        digest = {
            "总体概况": self.overview(),
            "三进口对比": self.compare_entrances(),
            "测速可信度交叉验证": self.speed_reliability_check(),
            "数据质量与异常": self.anomalies(),
            "信号配时估算": self.signal_timing_estimate(),
            "排队估算": self.queue_estimate(),
            "全路口车头时距": self.headway_stats(),
            "分进口": {},
        }
        for ent in ENTRANCES:
            digest["分进口"][ent] = {
                "流量": self.flow_series(entrance=ent),
                "车流模式_波浪或拥堵": self.flow_pattern(entrance=ent),
                "速度": self.speed_stats(entrance=ent),
                "车头时距": self.headway_stats(entrance=ent),
                "车道利用": self.lane_usage(entrance=ent),
                "转向构成": self.turn_movements(entrance=ent),
            }
        return digest
