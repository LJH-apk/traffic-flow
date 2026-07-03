"""
DeepSeek 客户端（OpenAI 兼容 /chat/completions，requests 实现）。

支持：非流式调用、SSE 流式（含 tool_calls 增量合并）、错误归一化为中文提示。
不引入 openai SDK，保持与项目零依赖打包链路一致（requests 随 ultralytics 自带）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.config.settings import (
    AGENT_LLM_TIMEOUT_S,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_THINKING_STRENGTH,
)


class LLMError(RuntimeError):
    """带用户可读中文消息的 LLM 调用错误。"""


class DeepSeekClient:
    THINKING_STRENGTHS = ("low", "medium", "high", "ultra")
    REASONING_EFFORT_MAP = {"low": "low", "medium": "medium", "high": "high", "ultra": "high"}

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, timeout: int | None = None,
                 thinking_strength: str | None = None):
        self.api_key = api_key if api_key is not None else DEEPSEEK_API_KEY
        self.base_url = (base_url or DEEPSEEK_BASE_URL).rstrip("/")
        self.model = model or DEEPSEEK_MODEL
        self.timeout = timeout or AGENT_LLM_TIMEOUT_S
        self.thinking_strength = self._normalize_thinking_strength(
            thinking_strength or DEEPSEEK_THINKING_STRENGTH
        )

    @classmethod
    def _normalize_thinking_strength(cls, value: str | None) -> str:
        v = (value or "medium").strip().lower()
        return v if v in cls.THINKING_STRENGTHS else "medium"

    def configure(self, model: str | None = None, thinking_strength: str | None = None) -> None:
        if model:
            self.model = model
        if thinking_strength:
            self.thinking_strength = self._normalize_thinking_strength(thinking_strength)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    # ── 请求 ─────────────────────────────────────────────────────────────

    def _request(self, payload: dict, stream: bool) -> requests.Response:
        if not self.api_key:
            raise LLMError("未配置 DEEPSEEK_API_KEY 环境变量。"
                           "请在终端执行 export DEEPSEEK_API_KEY=sk-xxx 后重启应用。")
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json=payload,
                stream=stream,
                timeout=(10, self.timeout),
            )
        except requests.exceptions.Timeout as exc:
            raise LLMError("DeepSeek API 请求超时，请检查网络后重试。") from exc
        except requests.exceptions.RequestException as exc:
            raise LLMError(f"无法连接 DeepSeek API（{exc.__class__.__name__}），"
                           "请检查网络连接。") from exc
        if resp.status_code != 200:
            raise LLMError(self._status_message(resp))
        return resp

    @staticmethod
    def _status_message(resp: requests.Response) -> str:
        detail = ""
        try:
            detail = resp.json().get("error", {}).get("message", "")
        except ValueError:
            pass
        mapping = {
            401: "API Key 无效或已过期，请检查 DEEPSEEK_API_KEY。",
            402: "DeepSeek 账户余额不足，请前往 platform.deepseek.com 充值。",
            429: "请求过于频繁，已被限流，请稍后重试。",
        }
        base = mapping.get(resp.status_code, f"DeepSeek API 返回错误（HTTP {resp.status_code}）。")
        return f"{base} {detail}".strip()

    def _payload(self, messages, tools, temperature, stream) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            "reasoning_effort": self.REASONING_EFFORT_MAP[self.thinking_strength],
        }
        if tools:
            payload["tools"] = tools
        return payload

    # ── 非流式 ───────────────────────────────────────────────────────────

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float = 0.6) -> dict:
        """返回 assistant message dict：{role, content, tool_calls?}。"""
        resp = self._request(self._payload(messages, tools, temperature, False), stream=False)
        try:
            return resp.json()["choices"][0]["message"]
        except (ValueError, KeyError, IndexError) as exc:
            raise LLMError("DeepSeek API 返回格式异常。") from exc

    # ── 流式 ─────────────────────────────────────────────────────────────

    def chat_stream(self, messages: list[dict], tools: list[dict] | None = None,
                    temperature: float = 0.6):
        """
        SSE 流式生成器，依次产出：
          {"type": "delta", "text": str}            —— 可见文本增量
          {"type": "message", "message": {...},     —— 结束时的完整 assistant 消息
           "finish_reason": "stop" | "tool_calls"}
        """
        resp = self._request(self._payload(messages, tools, temperature, True), stream=True)
        content_parts: list[str] = []
        tool_calls: dict[int, dict] = {}
        finish_reason = None
        try:
            for raw in resp.iter_lines():
                if not raw or not raw.startswith(b"data:"):
                    continue
                data = raw[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                text = delta.get("content")
                if text:
                    content_parts.append(text)
                    yield {"type": "delta", "text": text}
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tool_calls.setdefault(idx, {
                        "id": "", "type": "function",
                        "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
        finally:
            resp.close()

        message: dict = {"role": "assistant", "content": "".join(content_parts)}
        if tool_calls:
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
            finish_reason = finish_reason or "tool_calls"
        yield {"type": "message", "message": message,
               "finish_reason": finish_reason or "stop"}
