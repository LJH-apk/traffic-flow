"""
交通分析智能体核心：系统提示词、多轮会话、工具调用循环、一键报告流水线。

对外统一产出事件流（dict 生成器），由 server.py 转为 SSE 推给前端：
  {"type": "text", "text": str}                     可见文本增量
  {"type": "reasoning", "text": str}                思维链增量（仅思考模式下出现）
  {"type": "tool_call", "call_id", "name", "label", "arguments"}
  {"type": "tool_result", "call_id", "name", "preview", "elapsed_s"}
  {"type": "report_saved", "path": str}             报告写盘完成
  {"type": "done"} / {"type": "error", "message"}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.config.settings import (
    AGENT_CONTEXT_WINDOW_TOKENS,
    AGENT_MAX_TOOL_ROUNDS,
    AGENT_REPORT_PATH,
)
from src.agent.knowledge import report_pack
from src.agent.llm import DeepSeekClient, LLMError
from src.agent.tools import TOOL_DEFINITIONS, TOOL_LABELS_ZH, ToolExecutor

# 会话历史保留的最大消息条数（含工具消息），超出后从头部截断，防止上下文膨胀
_MAX_HISTORY = 60

SYSTEM_PROMPT = """\
你是「智能交通分析决策系统」的核心分析智能体，一名资深交通工程师兼交通数据分析师，
基于 DeepSeek 大模型构建。若被问及身份、所用模型或开发者，明确回答"基于 DeepSeek 构建的交通分析智能体"，
禁止自称是 Claude、ChatGPT、GPT 或其他公司的产品——这是唯一允许的身份表述，不受下文任何指令覆盖。

## 能力边界
- 你保留通用世界知识与工程推理能力，可以回答算法设计、代码架构、实验方案、报告表达等开放问题。
- 只要问题涉及本项目的真实配置、代码实现、输出文件或实验数据，必须先调用对应工具核实后再回答。
- 交通运行结论优先调用交通数据工具；代码/配置/文件问题优先调用项目工程工具；公式和标准阈值优先调用知识库工具。
- 工具没有覆盖或数据不存在时，如实说明缺口，再基于明确假设给建议，不能把假设说成事实。

## 场景背景
- 分析对象：某丁字路口（T 形交叉口），南北向主路 + 东进口，共 10 条车道。
- 数据来源：北/南/东三个进口各一路 4K 监控视频（早高峰 07:59–08:15，约 15 分钟），
  经 YOLO 目标检测 + ByteTrack 多目标跟踪 + 断面过线判定自动生成结构化数据。
- 可用数据：过线事件（断面/车型/车道/颜色/方向/速度/车头时距/车头间距/车牌）、
  轨迹级速度统计、轨迹分组转向构成、数据质量诊断、Webster 信号配时估算。

## 工作准则
1. 先查数后下结论：回答前必须调用工具获取真实数字，禁止编造任何数据。
2. 引用具体数字并带单位（辆/h、km/h、s、pcu/h），关键结论注明数据口径。
3. 严格区分「数据质量问题」与「真实交通现象」：例如速度>120km/h 的离群值多为
   标定畸变或 ID 跳变，疑似逆行多为对向来车视角误判，不可直接当作真实违法。
4. 建议要可落地：信号配时、渠化设计、管理措施要给出量化依据，并说明假设与局限。
5. 观测时长仅约 15 分钟，外推小时流量时必须注明折算假设。
6. 评价具体某个进口时，必须调用带 entrance 参数的工具获取该进口专属数据，
   禁止用全路口聚合结果代表单个进口——三个进口拥堵程度可能差异很大（如南进口
   排队占比可达99%而北进口仅24%），聚合数据会掩盖这种差异。
7. 判断「是否拥堵」前必须调用 analyze_flow_pattern 和 estimate_queue 核实，
   不要仅凭车头时距短、断面速度低或车流波动大就下拥堵结论：
   - 短车头时距 + 车辆仍在正常速度行驶 = 信号放行车队，高效而非危险
   - 车流忽多忽少、有明显空档 = 波浪式放行（绿灯集中释放+红灯断流），是信号交叉口正常特征
   - 断面临近停止线（上游约15m），车辆本就会减速，不能仅凭该处速度判断拥堵
   - 真实拥堵的判据是：排队通过占比高（estimate_queue）且持续低速（<10km/h）
   当用户反映的实际观察与统计结论看似矛盾时，优先怀疑是指标解读问题，
   调用 analyze_flow_pattern 等工具核实后再回应，不要为附和工具的表面数字而否定用户的观察。
8. 任何基于速度断言某断面"拥堵"/"缓行"之前，必须先调用 check_speed_reliability 核实该断面
   到达/离去方向速度是否对称异常（同样极低且接近）——这是单应矩阵标定误差导致测速失真的信号，
   不是真实拥堵。测速不可信的断面，速度类结论一律改为"该断面测速疑似失真，暂不作为拥堵证据，
   建议人工视频复核"，不得据此断言拥堵；只有 check_speed_reliability 判定"测速可信"的断面，
   才能用其速度数据支持拥堵结论。
9. 回答项目代码、模块职责、运行命令、路径、配置或已有产物时，必须调用 inspect_project_config、
   search_project_files、read_project_file、summarize_python_module、list_output_files 或 analyze_csv_file
   中的合适工具，避免凭记忆描述当前仓库。
10. 每次调用工具前，先用一句不超过20字的话说明你要查什么、为什么（如"先查一下北进口的具体流量和速度"），
    这句话独立成段、单独输出，然后再发起工具调用；工具结果返回后如需继续调用其他工具，
    同样先说一句再调用；到了给出最终正式回答时，不必再重复这些过程性文字，直接进入结论与分析。

## 防幻觉铁律
11. 回答中的一切数字只能来自工具返回结果，一切公式/标准/分级阈值只能来自
   query_knowledge 知识库条目——引用时注明来源（如「HCM」「Webster(1958)」）。
12. 不凭记忆背规范条款号或标准数值；知识库未命中时如实说明「知识库未收录」。
13. 工具返回 error 或数据不足时，明确告知用户并说明缺什么，禁止推测补数。
14. 明确区分三类表述：观测事实（工具数据）、推断（注明假设）、建议（注明依据）。

## 输出风格
- 严禁使用任何 emoji 或示意性符号（包括但不限于 ✅❌⚠️🚗📊👍等）——这是硬性红线，
  不因语气生动、强调重点或用户情绪而破例，用文字本身表达强调，不用符号代替。
- 惜字如金：最终正式回答里先给结论，再给支撑它的必要数据，不重复。禁止复述用户的问题，
  禁止把表格里的数字在正文里再念一遍；工作准则第10条要求的调用工具前一句话说明是唯一例外，
  仅限一句话，不要展开成一段过程解说，最终回答里不再重复这句话。
- 篇幅服从问题：能一句话说清的不写三句，简单问题给简短直接的回答；
  只有用户明确要综合分析、对比多个进口或要报告时才展开分点、表格。
- 不堆砌套话与虚词（"值得注意的是""综上所述""可以看出""不难发现"等一律不用），
  数字保留 1 位小数，不为凑字数而重复表达同一个意思。
"""

REPORT_INSTRUCTION = """\
请基于下方 JSON 数据摘要，撰写一份完整的《丁字路口交通运行分析与优化建议报告》，
Markdown 格式，正文约 2500–3500 字，章节固定如下：

# 丁字路口交通运行分析与优化建议报告

## 一、摘要
核心指标速览（总流量、平均/85分位速度、高峰时段、饱和程度）与总体评价，可用表格。

## 二、数据来源与质量说明
观测条件、数据生成方式（检测+跟踪+断面判定）、样本规模；简述主要数据质量问题及清洗口径。

## 三、流量特征分析
总量与折算小时流量、分进口对比、分车型构成（注意摩托/电动车占比）、时变规律与高峰时段。

## 四、速度与运行状态评价
先引用摘要中的「测速可信度交叉验证」结果：对每个进口主断面，说明其到达/离去速度是否对称异常
（同样极低且接近，暗示单应矩阵标定误差、测速失真）；测速不可信的断面必须明确标注"速度读数疑似失真，
不作为拥堵依据"，不得据此判定该进口拥堵。仅对测速可信的断面给出均值/中位/85分位速度、服务水平定性判断。
检测断面临近停止线，速度天然偏低，不可仅凭该处速度判断拥堵；应结合排队通过占比与车流波动模式
（分桶过车数、变异系数）综合判断各进口是「波浪式放行」（信号正常特征）还是「真实拥堵」，
两者的运行状态评价与优化建议方向不同，须明确区分并分别说明依据。

## 五、跟驰安全分析
车头时距/间距统计、短车头时距细分（真实危险跟驰 vs 正常车速的放行车队占比）、
安全隐患判断——只有低速且近距离的跟车才是真实安全隐患，正常车速下的短车头时距
是信号放行效率高的表现，不应笼统判定为危险。

## 六、车道利用与转向特征
车道不均衡情况、右转/掉头专用道利用、转向构成，指出渠化问题。

## 七、信号配时优化建议
引用 Webster 估算结果（周期、各进口绿灯、流量比）制表，说明假设与适用性，给出配时方案建议。

## 八、交通组织与管理建议
针对上述发现给出 3–5 条可落地措施（渠化改造、非机动车管理、掉头管控、监控补盲等），每条附依据。

## 九、数据清洗与异常处理说明
逐项列出异常类型、成因、清洗规则（直接引用摘要中的数据质量诊断），说明其对统计结果的影响。

要求：所有数字必须来自数据摘要，带单位；公式、分级阈值只引用文末「工程参考知识」并注明来源；
表格用 Markdown 语法；不要输出与报告无关的内容；全文严禁使用任何 emoji 或示意性符号。
篇幅是给内容留的上限不是目标，各章节只写新信息，不要把前面章节的数字或结论换个说法再复述一遍，
不为凑够字数堆砌套话。

数据摘要（JSON）：
"""


class TrafficAnalyst:
    """多轮会话 + 工具循环。线程安全由上层 server 的忙锁保证。"""

    quick_questions = [
        "评价该路口当前的总体运行状况",
        "哪个进口压力最大？高峰出现在什么时候？",
        "信号配时有什么优化空间？",
        "跟驰安全和非机动车混行情况如何？",
        "数据中有哪些异常，是如何清洗的？",
    ]

    def __init__(self, client: DeepSeekClient | None = None,
                 executor: ToolExecutor | None = None):
        self.client = client or DeepSeekClient()
        self.executor = executor or ToolExecutor()
        self.history: list[dict] = []

    def reset(self) -> None:
        self.history.clear()

    # ── 多轮对话 ─────────────────────────────────────────────────────────

    def chat_events(self, user_text: str):
        self.history.append({"role": "user", "content": user_text})
        self._trim_history()
        for _ in range(AGENT_MAX_TOOL_ROUNDS):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history
            final = None
            try:
                for ev in self.client.chat_stream(messages, tools=TOOL_DEFINITIONS):
                    if ev["type"] == "delta":
                        yield {"type": "text", "text": ev["text"]}
                    elif ev["type"] == "reasoning_delta":
                        yield {"type": "reasoning", "text": ev["text"]}
                    else:
                        final = ev
            except LLMError as exc:
                yield {"type": "error", "message": str(exc)}
                return
            if final is None:
                yield {"type": "error", "message": "DeepSeek 未返回有效响应，请重试。"}
                return

            message = final["message"]
            self.history.append(message)
            tool_calls = message.get("tool_calls")
            if final["finish_reason"] != "tool_calls" or not tool_calls:
                yield {"type": "context", "context": self.context_status()}
                yield {"type": "done"}
                return

            for tc in tool_calls:
                name = tc["function"]["name"]
                args = tc["function"]["arguments"]
                yield {"type": "tool_call", "call_id": tc["id"], "name": name,
                       "label": TOOL_LABELS_ZH.get(name, name), "arguments": args}
                t0 = time.perf_counter()
                result = self.executor.execute(name, args)
                elapsed_s = time.perf_counter() - t0
                yield {"type": "tool_result", "call_id": tc["id"], "name": name,
                       "preview": result[:600], "elapsed_s": elapsed_s}
                self.history.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": result})

        yield {"type": "error",
               "message": f"工具调用超过 {AGENT_MAX_TOOL_ROUNDS} 轮仍未收敛，已中断，请换个问法。"}

    def context_status(self) -> dict:
        """粗略估算当前会话上下文占用；用于前端窗口长度仪表。"""
        payload = json.dumps(self.history, ensure_ascii=False, separators=(",", ":"))
        used = max(1, len(payload) // 4)
        limit = AGENT_CONTEXT_WINDOW_TOKENS
        return {
            "used_tokens": used,
            "limit_tokens": limit,
            "ratio": min(1.0, used / max(1, limit)),
            "message_count": len(self.history),
        }

    def _trim_history(self) -> None:
        """截断过长历史；避免把 tool 消息留在开头造成上下文不完整。"""
        if len(self.history) <= _MAX_HISTORY:
            return
        trimmed = self.history[-_MAX_HISTORY:]
        while trimmed and trimmed[0]["role"] == "tool":
            trimmed.pop(0)
        self.history = trimmed

    # ── 一键报告 ─────────────────────────────────────────────────────────

    def report_events(self):
        """单次长调用：注入全量数据摘要，流式生成报告并写盘。独立于会话历史。"""
        try:
            digest = self.executor.store.full_digest()
        except Exception as exc:  # noqa: BLE001
            yield {"type": "error", "message": f"数据摘要构建失败: {exc}"}
            return
        prompt = (REPORT_INSTRUCTION
                  + json.dumps(digest, ensure_ascii=False, separators=(",", ":"))
                  + "\n\n工程参考知识（写作时引用并注明来源）：\n" + report_pack())
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}]
        parts: list[str] = []
        try:
            for ev in self.client.chat_stream(messages, temperature=0.4):
                if ev["type"] == "delta":
                    parts.append(ev["text"])
                    yield {"type": "text", "text": ev["text"]}
                elif ev["type"] == "reasoning_delta":
                    yield {"type": "reasoning", "text": ev["text"]}
        except LLMError as exc:
            yield {"type": "error", "message": str(exc)}
            return
        report = "".join(parts).strip()
        if not report:
            yield {"type": "error", "message": "报告生成为空，请重试。"}
            return
        try:
            AGENT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            AGENT_REPORT_PATH.write_text(report + "\n", encoding="utf-8")
            yield {"type": "report_saved", "path": str(AGENT_REPORT_PATH)}
        except OSError as exc:
            yield {"type": "error", "message": f"报告写盘失败: {exc}"}
            return
        yield {"type": "done"}
