from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from agent_core import ToolRegistry


registry = ToolRegistry()


def get_current_time() -> str:
    """返回服务器当前时间。"""
    return datetime.now().isoformat(timespec="seconds")


registry.register(
    get_current_time,
    name="get_current_time",
    description="获取服务器当前时间。",
    parameters={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
)


def calculator(expression: str) -> Any:
    """
    一个演示用计算器。
    为了保持简单，仅允许数字与基础算术字符。
    """
    allowed = set("0123456789+-*/(). %")
    if not expression or any(ch not in allowed for ch in expression):
        raise ValueError("Expression contains unsupported characters.")
    return eval(expression, {"__builtins__": {}}, {})


registry.register(
    calculator,
    description="进行基础数学计算，例如 '(100+20)*0.8'。",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "只包含数字、括号和基础算术运算符的表达式。",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
)


def search_excel_rows(
    query: str,
    excel_path: str,
    sheet_name: str = "结果汇总",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    一个非常简单的 Excel 检索工具示例。
    这还不是正式 RAG，只是为了演示：
    以后你可以直接把这里替换成向量库检索。
    """
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df.fillna("")

    text = df.astype(str).agg(" | ".join, axis=1)

    tokens = [x.strip().lower() for x in query.replace("，", " ").split() if x.strip()]

    def score_row(row_text: str) -> int:
        t = row_text.lower()
        return sum(t.count(token) for token in tokens)

    scores = text.map(score_row)
    ranked = df.assign(_score=scores).sort_values("_score", ascending=False)

    if (ranked["_score"] > 0).any():
        ranked = ranked[ranked["_score"] > 0]

    ranked = ranked.head(max(1, min(int(top_k), 20)))

    result = []
    for idx, row in ranked.iterrows():
        item = {"row_index": int(idx), "score": int(row["_score"])}
        for col in df.columns:
            item[str(col)] = row[col]
        result.append(item)

    return result


registry.register(
    search_excel_rows,
    description=(
        "在 Excel 工作表中做简单关键词检索，返回最相关的若干行。"
        "这是示例工具，后续可以替换成真正的向量 RAG。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "用户想检索的诊断、项目或临床问题。",
            },
            "excel_path": {
                "type": "string",
                "description": "Excel 文件路径。",
            },
            "sheet_name": {
                "type": "string",
                "description": "工作表名称，默认：结果汇总。",
                "default": "结果汇总",
            },
            "top_k": {
                "type": "integer",
                "description": "最多返回多少条记录。",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["query", "excel_path"],
        "additionalProperties": False,
    },
)
