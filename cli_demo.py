"""最小可用 Agent 演示：直接命令行测试乘法工具。"""
from __future__ import annotations

import logging
import os
import sys

# 修复 Windows 控制台中文乱码
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv

from agent_core import Agent, ModelConfig
from tools import registry

# 设置日志（INFO 级别，显示时间、级别、消息）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

load_dotenv()

config = ModelConfig(
    base_url=os.getenv("MODEL_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.getenv("MODEL_API_KEY", ""),
    model=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
    temperature=0.2,
)

# 可用的工具
print("可用工具:", [t["function"]["name"] for t in registry.schemas()])

# 创建 Agent
agent = Agent(
    config=config,
    system_prompt="你是一个简洁的数学助手。用户问乘法时，调用 multiply 工具计算，然后告诉用户结果。回答使用中文。",
    tools=registry,
    max_steps=5,
)

# 测试问题
questions = [
    "123 乘 456 等于多少？",
    "100.5 乘以 3.2 等于多少？",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"用户: {q}")
    result = agent.run(user_input=q)
    print(f"Agent: {result['answer']}")
    print(f"\nTrace (步骤数: {len(result['trace'])}):")
    for t in result["trace"]:
        if "tool_calls" in t and t["tool_calls"]:
            for tc in t["tool_calls"]:
                fn = tc["function"]
                print(f"  → 调用工具: {fn['name']}({fn['arguments']})")