"""AI 交通分析智能体：数据层 / 工具 / 会话循环 / 服务端点 / 前端接线 测试。"""
import json
import urllib.request
from pathlib import Path

import pytest

from src.agent.analyst import TrafficAnalyst
from src.agent.datastore import TrafficDataStore, _percentile, entrance_of
from src.agent.knowledge import search_knowledge
from src.agent.llm import LLMError
from src.agent.tools import TOOL_DEFINITIONS, TOOL_LABELS_ZH, ToolExecutor

INDEX = Path("src/agent/static/index.html")
AGENT_JS = Path("src/agent/static/assets/agent.js")
AGENT_CSS = Path("src/agent/static/assets/agent.css")


# ── 假 LLM 客户端 ─────────────────────────────────────────────────────────


class FakeClient:
    """第一轮返回 get_overview 工具调用，第二轮流式返回文本答案。"""

    model = "fake-model"
    available = True
    thinking_strength = "medium"

    def __init__(self):
        self.calls = 0

    def chat_stream(self, messages, tools=None, temperature=0.6):
        self.calls += 1
        if self.calls == 1:
            yield {"type": "message", "finish_reason": "tool_calls",
                   "message": {"role": "assistant", "content": "",
                               "tool_calls": [{"id": "c1", "type": "function",
                                               "function": {"name": "get_overview",
                                                            "arguments": "{}"}}]}}
        else:
            yield {"type": "delta", "text": "分析完成。"}
            yield {"type": "message", "finish_reason": "stop",
                   "message": {"role": "assistant", "content": "分析完成。"}}


class BrokenClient:
    model = "fake-model"
    available = True
    thinking_strength = "medium"

    def chat_stream(self, messages, tools=None, temperature=0.6):
        raise LLMError("模拟网络故障")
        yield  # pragma: no cover


# ── 知识库 ────────────────────────────────────────────────────────────────


def test_knowledge_hit_webster():
    result = search_knowledge("Webster 配时 周期")
    assert "命中条目" in result
    top = result["命中条目"][0]
    assert top["主题"] == "Webster 信号配时"
    assert "C0=(1.5L+5)/(1-Y)" in top["要点"]
    assert top["来源"]


def test_knowledge_miss_returns_topics():
    result = search_knowledge("量子力学")
    assert "error" in result
    assert "可用主题" in result and len(result["可用主题"]) >= 10


# ── 数据层纯函数 ──────────────────────────────────────────────────────────


def test_percentile_and_entrance():
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert _percentile([5.0], 0.85) == 5.0
    assert entrance_of("北进口主断面") == "北进口"
    assert entrance_of("东进口右转专用道") == "东进口"
    assert entrance_of("神秘断面") == "未知"


def test_parse_event_tolerates_bad_rows():
    assert TrafficDataStore._parse_event({"timestamp_s": "", "section": "x"}) is None
    ev = TrafficDataStore._parse_event({
        "timestamp_s": "1.5", "section": "南进口主断面", "speed_kmh": "",
        "headway_s": "2.0", "class_name": "car"})
    assert ev["speed"] is None and ev["headway"] == 2.0
    assert ev["entrance"] == "南进口"


# ── 真实数据聚合（outputs/ 缺失时跳过）─────────────────────────────────────


@pytest.fixture(scope="module")
def store():
    s = TrafficDataStore()
    if not s.data_ready:
        pytest.skip("outputs/ 无过线数据")
    return s


def test_overview_and_flow(store):
    ov = store.overview()
    assert ov["过线事件总数"] > 0
    flow = store.flow_series(entrance="北进口")
    assert flow["总过车数"] == sum(flow["各时段过车数"])
    assert flow["高峰时段"]["过车数"] == max(flow["各时段过车数"])


def test_speed_stats_filters_outliers(store):
    st = store.speed_stats()
    assert 0 < st["平均速度_kmh"] <= 120
    assert st["85分位速度_kmh"] >= st["中位速度_kmh"]


def test_headway_breakdown_distinguishes_platoon_from_danger(store):
    """回归锁定：短车头时距须按速度拆分，不能把放行车队误判为危险跟驰。"""
    north = store.headway_stats(entrance="北进口")
    south = store.headway_stats(entrance="南进口")
    if "error" in north or "error" in south:
        pytest.skip("缺少北/南进口车头时距样本")
    n_breakdown = north["短车头时距细分"]
    s_breakdown = south["短车头时距细分"]
    assert n_breakdown["真实危险跟驰_低速且近距离_占比"] is not None
    # 北进口应以正常车速的放行车队为主，南进口应以真实拥堵跟车为主
    assert (n_breakdown["放行车队_正常车速且近距离_占比"]
            > n_breakdown["真实危险跟驰_低速且近距离_占比"])
    assert (s_breakdown["真实危险跟驰_低速且近距离_占比"]
            > n_breakdown["真实危险跟驰_低速且近距离_占比"])


def test_flow_pattern_reports_burstiness(store):
    pattern = store.flow_pattern(entrance="北进口")
    if "error" in pattern:
        pytest.skip(pattern["error"])
    assert pattern["变异系数CV"] is not None
    assert len(pattern["分桶过车数"]) > 0
    assert isinstance(pattern["模式判断"], str) and pattern["模式判断"]


def test_signal_timing_reasonable(store):
    sig = store.signal_timing_estimate()
    if "error" in sig:
        pytest.skip(sig["error"])
    assert sig["建议周期_s"] is None or 40 <= sig["建议周期_s"] <= 160


def test_full_digest_serializable(store):
    text = json.dumps(store.full_digest(), ensure_ascii=False)
    assert len(text) < 20000  # 摘要必须足够小，能塞进报告提示词


# ── 工具调度 ──────────────────────────────────────────────────────────────


def test_tool_definitions_complete():
    names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    assert names == set(TOOL_LABELS_ZH)
    assert "query_knowledge" in names and "estimate_signal_timing" in names


def test_executor_handles_unknown_and_bad_args():
    ex = ToolExecutor()
    assert "未知工具" in ex.execute("no_such_tool", "{}")
    result = json.loads(ex.execute("query_knowledge", "not-a-json"))
    assert "error" in result or "命中条目" in result  # 坏参数不崩溃


# ── 会话循环 ──────────────────────────────────────────────────────────────


def test_analyst_tool_loop():
    analyst = TrafficAnalyst(client=FakeClient())
    events = list(analyst.chat_events("总体情况如何"))
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "text", "context", "done"]
    assert [m["role"] for m in analyst.history] == ["user", "assistant", "tool", "assistant"]
    assert events[0]["label"] == "查询总体概况"
    assert "elapsed_s" in events[1]
    assert events[3]["context"]["limit_tokens"] > 0


def test_analyst_llm_error_becomes_event():
    analyst = TrafficAnalyst(client=BrokenClient())
    events = list(analyst.chat_events("你好"))
    assert events[-1]["type"] == "error"
    assert "模拟网络故障" in events[-1]["message"]


def test_report_pipeline_saves_file(tmp_path, monkeypatch):
    import src.agent.analyst as analyst_mod

    monkeypatch.setattr(analyst_mod, "AGENT_REPORT_PATH", tmp_path / "report.md")
    analyst = TrafficAnalyst(client=FakeClient())
    analyst.client.calls = 1  # 跳到文本轮
    events = list(analyst.report_events())
    types = [e["type"] for e in events]
    assert "report_saved" in types and types[-1] == "done"
    assert (tmp_path / "report.md").read_text(encoding="utf-8").strip() == "分析完成。"


# ── HTTP 服务端点 ─────────────────────────────────────────────────────────


@pytest.fixture()
def agent_server():
    import threading

    from src.agent.server import AgentApp, Handler, create_server

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 绕开系统代理
    urllib.request.install_opener(opener)

    app = AgentApp(analyst=TrafficAnalyst(client=FakeClient()))
    old_app = Handler.app
    Handler.app = app
    server = create_server(port=8790)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    yield f"http://{host}:{port}"
    server.shutdown()
    Handler.app = old_app


def test_server_endpoints(agent_server):
    with urllib.request.urlopen(agent_server + "/api/agent/health", timeout=5) as r:
        health = json.loads(r.read())
    assert health["model"] == "fake-model"

    for path in ("/", "/assets/agent.css", "/assets/agent.js"):
        with urllib.request.urlopen(agent_server + path, timeout=5) as r:
            assert r.status == 200

    req = urllib.request.Request(
        agent_server + "/api/agent/chat", method="POST",
        data=json.dumps({"message": "hi"}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        assert "text/event-stream" in r.headers["Content-Type"]
        events = [json.loads(line[5:]) for line in r.read().decode().splitlines()
                  if line.startswith("data:")]
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "text", "context", "done"]


def test_server_tools_endpoint(agent_server):
    with urllib.request.urlopen(agent_server + "/api/agent/tools", timeout=5) as r:
        tools = json.loads(r.read())
    names = {t["name"] for t in tools}
    assert names == set(TOOL_LABELS_ZH)
    assert all({"name", "label", "description"} <= t.keys() for t in tools)


def test_server_rejects_empty_message(agent_server):
    req = urllib.request.Request(
        agent_server + "/api/agent/chat", method="POST",
        data=b"{}", headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 400


# ── 前端接线 ──────────────────────────────────────────────────────────────


def test_frontend_wiring():
    html = INDEX.read_text(encoding="utf-8")
    js = AGENT_JS.read_text(encoding="utf-8")
    css = AGENT_CSS.read_text(encoding="utf-8")

    assert "/assets/agent.css" in html and "/assets/agent.js" in html
    for api in ("/api/agent/chat", "/api/agent/report", "/api/agent/summary",
                "/api/agent/health", "/api/agent/reset", "/api/agent/report/download",
                "/api/agent/tools", "/api/agent/reload", "/api/agent/settings"):
        assert api in js
    assert "text/event-stream" in js       # SSE 响应校验
    assert "think-text" in css             # 思考流光样式
    assert "context-popover" in css and "contextMeter" in html
    assert "</style>" not in css           # 纯 CSS 文件不应残留 HTML 标签

    # 内部命令：每个 SLASH_COMMANDS 键都要有对应的 dispatch 分支
    for cmd in ("help", "clear", "report", "summary", "reload", "tools", "model"):
        assert f"{cmd}:" in js or f"'{cmd}'" in js or f'"{cmd}"' in js
    assert "dispatchSlashCommand" in js and "/help" in js
