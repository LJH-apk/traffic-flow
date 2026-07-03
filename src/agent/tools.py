"""
智能体工具集：OpenAI function calling 格式的 schema 定义 + 执行调度。

每个工具对应 TrafficDataStore 的一个查询方法，返回 JSON 字符串（回喂 LLM）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.agent.datastore import TrafficDataStore
from src.agent.knowledge import search_knowledge
from src.agent.project_tools import (
    analyze_csv_file,
    inspect_project_config,
    list_output_files,
    read_project_file,
    search_project_files,
    summarize_python_module,
)

_ENTRANCE_ENUM = ["北进口", "南进口", "东进口"]
_CLASS_ENUM = ["car", "truck", "bus", "motorcycle", "bicycle"]

# 工具中文名（前端步骤卡片展示用）
TOOL_LABELS_ZH: dict[str, str] = {
    "get_overview": "查询总体概况",
    "query_flow": "查询流量时间序列",
    "query_speed": "查询速度统计",
    "query_headway": "查询车头时距",
    "query_lane_usage": "查询车道利用",
    "query_turn_movements": "查询转向构成",
    "query_events": "抽样过线事件明细",
    "detect_anomalies": "检测数据异常",
    "estimate_signal_timing": "估算信号配时",
    "estimate_queue": "估算排队状态",
    "compare_entrances": "三进口横向对比",
    "query_knowledge": "查询领域知识库",
    "analyze_flow_pattern": "分析车流波动模式",
    "check_speed_reliability": "交叉验证测速可信度",
    "inspect_project_config": "检查项目关键配置",
    "list_output_files": "列出实验输出文件",
    "search_project_files": "检索项目文件",
    "read_project_file": "读取项目文件",
    "summarize_python_module": "解析 Python 模块",
    "analyze_csv_file": "分析 CSV 文件",
}


def _tool(name: str, description: str, properties: dict | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": [],
            },
        },
    }


_P_ENTRANCE = {"type": "string", "enum": _ENTRANCE_ENUM,
               "description": "进口名，不传则统计全路口"}
_P_CLASS = {"type": "string", "enum": _CLASS_ENUM,
            "description": "车型（COCO类别名），不传则全部车型"}
_P_SECTION = {"type": "string",
              "description": "断面名关键词（如 主断面/右转/掉头），不传则全部断面"}

TOOL_DEFINITIONS: list[dict] = [
    _tool("get_overview",
          "获取路口总体概况：视频时长、轨迹总数、过线事件总数、小时流量估算、分车型/分进口/分方向/分断面事件数。回答任何宏观问题前先调用。"),
    _tool("query_flow",
          "查询流量时间序列（按时间粒度分桶的过车数）、平均流率与高峰时段。默认只统计主断面'到达'方向（即进口到达流量）。",
          {"entrance": _P_ENTRANCE, "class_name": _P_CLASS,
           "direction": {"type": "string", "enum": ["到达", "离去"],
                         "description": "过线方向，默认'到达'"},
           "bucket_s": {"type": "integer", "description": "分桶粒度秒数，默认60"},
           "only_main": {"type": "boolean", "description": "是否只统计主断面，默认true"}}),
    _tool("query_speed",
          "查询断面速度统计：均值/中位/85分位/15分位、速度直方图、低速占比、异常值条数。",
          {"entrance": _P_ENTRANCE, "section": _P_SECTION, "class_name": _P_CLASS}),
    _tool("query_headway",
          "查询车头时距与车头间距统计：均值/中位/最小、危险跟驰（<1.5s）占比。",
          {"entrance": _P_ENTRANCE, "section": _P_SECTION}),
    _tool("query_lane_usage",
          "查询车道利用：各车道过车数、最忙/最闲车道、车道不均衡系数、未识别与对向车道数量。",
          {"entrance": _P_ENTRANCE}),
    _tool("query_turn_movements",
          "查询转向构成：直行/左转/右转/掉头/疑似逆行的轨迹组数量，以及右转、掉头专用断面过车数。",
          {"entrance": _P_ENTRANCE}),
    _tool("query_events",
          "抽样查询原始过线事件明细（时刻/断面/车型/车道/颜色/速度/车头时距/车牌），用于举例或核查个案。",
          {"entrance": _P_ENTRANCE, "section": _P_SECTION, "class_name": _P_CLASS,
           "min_speed": {"type": "number", "description": "最低速度过滤（km/h）"},
           "max_speed": {"type": "number", "description": "最高速度过滤（km/h）"},
           "limit": {"type": "integer", "description": "返回条数，默认10，最多30"}}),
    _tool("detect_anomalies",
          "检测数据质量问题与异常：超速离群值、零速事件、未识别车道、疑似逆行、短命轨迹（ID跳变）、车牌覆盖率，并给出每项的成因与清洗规则。讨论数据可靠性或数据清洗时调用。"),
    _tool("estimate_signal_timing",
          "基于 Webster 公式估算信号配时：各进口高峰小时流量（pcu/h）、流量比、建议周期与绿灯时长。给出配时优化建议前调用。"),
    _tool("estimate_queue",
          "估算排队状态：各进口排队通过占比、峰值分钟排队车数与最大排队长度粗估（低速过线判定法）。讨论拥堵、排队溢出风险时调用。",
          {"entrance": _P_ENTRANCE}),
    _tool("compare_entrances",
          "三进口横向对比：到达流量、折算流率、平均/85分位速度、危险跟驰占比、主要车型与压力排序。比较各进口或找瓶颈时优先调用。"),
    _tool("analyze_flow_pattern",
          "分析到达车流的时间波动模式：区分『波浪式放行』（信号周期性集中放行+断流，正常现象）与『持续车流』，"
          "返回分桶过车数、变异系数与模式判断。当需要判断某进口是否真实拥堵、"
          "或用户反映实际观察（如'车是一波一波的但不堵'）与统计结论不符时，必须调用本工具核实，"
          "不要仅凭车头时距短或断面速度低就下'拥堵'结论。",
          {"entrance": _P_ENTRANCE, "section": _P_SECTION,
           "bucket_s": {"type": "integer", "description": "分桶粒度秒数，默认10"}}),
    _tool("check_speed_reliability",
          "交叉验证各主断面测速是否可信：比对同一断面『到达』与『离去』方向的平均速度。"
          "真实拥堵应是到达慢、离去明显更快；若两个方向同样极低且接近，"
          "是单应矩阵标定误差导致的测速失真信号，不是真实拥堵。"
          "任何时候要基于速度断言某进口'拥堵'/'缓行'之前，必须先调用本工具核实该断面测速是否可信，"
          "不可信的断面不得作为拥堵证据。"),
    _tool("query_knowledge",
          "查询交通工程领域知识库：Webster配时、服务水平LOS分级、PCU换算、饱和流率、车头时距安全阈值、排队估算方法、机非混行治理、行人最小绿灯、数据清洗规范等，返回权威要点与出处。凡引用公式、标准数值、分级阈值前必须先调用本工具，并在回答中注明来源。",
          {"query": {"type": "string",
                     "description": "检索关键词，如'Webster 配时周期'、'服务水平分级'、'排队长度'"},
           "limit": {"type": "integer", "description": "返回条目数，默认3"}}),
    _tool("inspect_project_config",
          "读取本项目关键配置：模型、设备、阈值、输入输出路径、测速/排队参数、断面线。回答项目配置、当前算法参数、路径相关问题时调用。"),
    _tool("list_output_files",
          "列出 outputs 目录下已有实验产物及文件大小。判断当前有哪些结果文件、报告、视频、CSV 可用时调用。",
          {"limit": {"type": "integer", "description": "最多返回文件数，默认80"}}),
    _tool("search_project_files",
          "在项目文本文件中按关键词检索，返回文件、行号和命中片段。回答代码位置、实现细节、某个常量/函数在哪里定义时调用。",
          {"query": {"type": "string", "description": "检索关键词"},
           "path": {"type": "string", "description": "可选：限定检索目录或文件，如 src/trajectory"},
           "limit": {"type": "integer", "description": "最多返回命中数，默认20"}}),
    _tool("read_project_file",
          "读取项目内文本文件的指定行段。需要解释代码、配置或文档原文时调用。",
          {"path": {"type": "string", "description": "项目相对路径，如 src/agent/analyst.py"},
           "start_line": {"type": "integer", "description": "起始行号，默认1"},
           "max_lines": {"type": "integer", "description": "最多读取行数，默认160，最大300"}}),
    _tool("summarize_python_module",
          "静态解析项目内 Python 文件，返回类、顶层函数、方法和导入概览。理解模块职责或回答代码架构问题时调用。",
          {"path": {"type": "string", "description": "项目相对路径，如 src/trajectory/tracker.py"}}),
    _tool("analyze_csv_file",
          "通用 CSV 体检：行列数、字段、缺失、数值范围、类别分布与样例。分析非固定交通结果表或临时实验 CSV 时调用。",
          {"path": {"type": "string", "description": "项目相对 CSV 路径，如 outputs/trajectory.csv"},
           "limit_columns": {"type": "integer", "description": "最多分析字段数，默认30"}}),
]


class ToolExecutor:
    """按名称调度工具执行，返回 JSON 字符串；任何异常都转为 error 字段。"""

    def __init__(self, store: TrafficDataStore | None = None):
        self.store = store or TrafficDataStore()

    def execute(self, name: str, arguments: str | dict | None) -> str:
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments) if arguments.strip() else {}
            except ValueError:
                args = {}
        else:
            args = dict(arguments or {})
        try:
            result = self._dispatch(name, args)
        except Exception as exc:  # noqa: BLE001 — 工具错误必须回喂模型而非中断会话
            result = {"error": f"工具执行失败: {exc}"}
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))

    def _dispatch(self, name: str, a: dict) -> dict:
        s = self.store
        if name == "get_overview":
            return s.overview()
        if name == "query_flow":
            return s.flow_series(
                entrance=a.get("entrance"), class_name=a.get("class_name"),
                direction=a.get("direction", "到达"),
                bucket_s=a.get("bucket_s", 60),
                only_main=a.get("only_main", True))
        if name == "query_speed":
            return s.speed_stats(entrance=a.get("entrance"), section=a.get("section"),
                                 class_name=a.get("class_name"))
        if name == "query_headway":
            return s.headway_stats(entrance=a.get("entrance"), section=a.get("section"))
        if name == "query_lane_usage":
            return s.lane_usage(entrance=a.get("entrance"))
        if name == "query_turn_movements":
            return s.turn_movements(entrance=a.get("entrance"))
        if name == "query_events":
            return s.sample_events(
                entrance=a.get("entrance"), section=a.get("section"),
                class_name=a.get("class_name"), min_speed=a.get("min_speed"),
                max_speed=a.get("max_speed"), limit=a.get("limit", 10))
        if name == "detect_anomalies":
            return s.anomalies()
        if name == "estimate_signal_timing":
            return s.signal_timing_estimate()
        if name == "estimate_queue":
            return s.queue_estimate(entrance=a.get("entrance"))
        if name == "compare_entrances":
            return s.compare_entrances()
        if name == "analyze_flow_pattern":
            return s.flow_pattern(entrance=a.get("entrance"), section=a.get("section"),
                                  bucket_s=a.get("bucket_s", 10))
        if name == "check_speed_reliability":
            return s.speed_reliability_check()
        if name == "query_knowledge":
            return search_knowledge(a.get("query", ""), a.get("limit", 3))
        if name == "inspect_project_config":
            return inspect_project_config()
        if name == "list_output_files":
            return list_output_files(limit=a.get("limit", 80))
        if name == "search_project_files":
            return search_project_files(
                query=a.get("query", ""), path=a.get("path"), limit=a.get("limit", 20))
        if name == "read_project_file":
            return read_project_file(
                path=a.get("path", ""), start_line=a.get("start_line"),
                max_lines=a.get("max_lines", 160))
        if name == "summarize_python_module":
            return summarize_python_module(path=a.get("path", ""))
        if name == "analyze_csv_file":
            return analyze_csv_file(path=a.get("path", ""),
                                    limit_columns=a.get("limit_columns", 30))
        return {"error": f"未知工具: {name}"}
