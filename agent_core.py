from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

import httpx

logger = logging.getLogger("agent")


@dataclass
class ModelConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    timeout: float = 120.0


class ToolRegistry:
    """
    极简工具注册器：
    - 工具就是普通 Python 函数
    - 用 JSON Schema 描述参数
    - Agent 自动把函数暴露给支持 tools/function calling 的模型
    """

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    def register(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> Callable[..., Any]:
        tool_name = name or func.__name__
        doc = description or inspect.getdoc(func) or f"Call tool {tool_name}."

        if parameters is None:
            parameters = {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            }

        self._tools[tool_name] = {
            "func": func,
            "spec": {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": doc,
                    "parameters": parameters,
                },
            },
        }
        return func

    def schemas(self) -> list[dict[str, Any]]:
        return [item["spec"] for item in self._tools.values()]

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]["func"](**arguments)


class Agent:
    """
    一个最小可用 Agent Loop：

    用户消息
      -> 模型
      -> 如果模型调用工具：执行工具
      -> 把工具结果喂回模型
      -> 直到模型输出最终答案
    """

    def __init__(
        self,
        config: ModelConfig,
        system_prompt: str,
        tools: ToolRegistry | None = None,
        max_steps: int = 8,
        verbose: bool = True,
    ) -> None:
        self.config = config
        self.system_prompt = system_prompt
        self.tools = tools or ToolRegistry()
        self.max_steps = max_steps
        self.verbose = verbose

    def _chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        url = self.config.base_url.rstrip("/") + "/chat/completions"

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }

        schemas = self.tools.schemas()
        if schemas:
            payload["tools"] = schemas
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.config.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def _normalize_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
        clean = {
            "role": "assistant",
            "content": message.get("content"),
        }
        if message.get("tool_calls"):
            clean["tool_calls"] = message["tool_calls"]
        return clean

    def run(
        self,
        user_input: str,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        trace: list[dict[str, Any]] = []

        if self.verbose:
            logger.info("=" * 50)
            logger.info("Agent 启动 | 模型: %s | 最大步数: %d", self.config.model, self.max_steps)
            logger.info("可用工具: %s", [t["function"]["name"] for t in self.tools.schemas()])

        for step in range(1, self.max_steps + 1):
            if self.verbose:
                logger.info("--- Step %d/%d ---", step, self.max_steps)

            raw = self._chat(messages)
            message = raw["choices"][0]["message"]
            assistant_message = self._normalize_assistant_message(message)
            messages.append(assistant_message)

            tool_calls = message.get("tool_calls") or []

            trace.append(
                {
                    "step": step,
                    "assistant": message.get("content"),
                    "tool_calls": tool_calls,
                }
            )

            if not tool_calls:
                if self.verbose:
                    logger.info("模型未请求工具 → 返回最终答案")
                return {
                    "answer": message.get("content") or "",
                    "messages": messages,
                    "trace": trace,
                    "raw_response": raw,
                }

            for tool_call in tool_calls:
                fn = tool_call["function"]
                tool_name = fn["name"]

                raw_args = fn.get("arguments") or "{}"
                if self.verbose:
                    logger.info("调用工具: %s(%s)", tool_name, raw_args)

                try:
                    arguments = json.loads(raw_args)
                except json.JSONDecodeError as e:
                    result = {
                        "ok": False,
                        "error": f"Invalid tool arguments JSON: {e}",
                        "raw_arguments": raw_args,
                    }
                    if self.verbose:
                        logger.error("工具参数解析失败: %s", e)
                else:
                    try:
                        output = self.tools.call(tool_name, arguments)
                        result = {"ok": True, "result": output}
                        if self.verbose:
                            logger.info("工具返回: %s", json.dumps(output, ensure_ascii=False, default=str)[:200])
                    except Exception as e:
                        result = {
                            "ok": False,
                            "error": f"{type(e).__name__}: {e}",
                        }
                        if self.verbose:
                            logger.error("工具执行失败: %s", e)

                trace.append(
                    {
                        "step": step,
                        "tool": tool_name,
                        "arguments": raw_args,
                        "result": result,
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(
                            result,
                            ensure_ascii=False,
                            default=str,
                        ),
                    }
                )

        return {
            "answer": "Agent reached max_steps before producing a final answer.",
            "messages": messages,
            "trace": trace,
            "raw_response": None,
        }
