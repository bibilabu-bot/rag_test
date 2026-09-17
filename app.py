from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from agent_core import Agent, ModelConfig
from tools import registry


load_dotenv()

st.set_page_config(
    page_title="Simple Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("Simple Agent Framework")
st.caption("可填写提示词、模型 API；工具直接写在 tools.py。")

with st.sidebar:
    st.header("模型配置")

    base_url = st.text_input(
        "Base URL",
        value=os.getenv("MODEL_BASE_URL", "https://api.openai.com/v1"),
        help="要求兼容 OpenAI Chat Completions 接口。",
    )

    api_key = st.text_input(
        "API Key",
        value=os.getenv("MODEL_API_KEY", ""),
        type="password",
    )

    model = st.text_input(
        "Model",
        value=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.2,
        step=0.1,
    )

    max_steps = st.slider(
        "Agent 最大循环次数",
        min_value=1,
        max_value=20,
        value=8,
    )

st.subheader("系统提示词")

default_prompt = """你是一个专业、简洁的智能体。

规则：
1. 优先理解用户真实需求。
2. 有可用工具且工具能提高准确性时，可以调用工具。
3. 工具返回的数据只是事实材料，你需要整理后再回答。
4. 不要编造工具没有返回的信息。
5. 回答默认使用中文。
"""

system_prompt = st.text_area(
    "System Prompt",
    value=default_prompt,
    height=220,
    label_visibility="collapsed",
)

st.subheader("可用工具")
tool_names = [x["function"]["name"] for x in registry.schemas()]
st.write(" / ".join(tool_names) if tool_names else "当前没有工具")

if "history" not in st.session_state:
    st.session_state.history = []

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("输入你的问题……")

if user_input:
    st.session_state.display_messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    if not api_key:
        with st.chat_message("assistant"):
            st.error("请先填写 API Key。")
    else:
        config = ModelConfig(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
        )

        agent = Agent(
            config=config,
            system_prompt=system_prompt,
            tools=registry,
            max_steps=max_steps,
        )

        with st.chat_message("assistant"):
            with st.spinner("运行中..."):
                try:
                    result = agent.run(
                        user_input=user_input,
                        history=st.session_state.history,
                    )
                    answer = result["answer"]
                    st.markdown(answer)

                    with st.expander("查看 Agent Trace"):
                        st.json(result["trace"])

                except Exception as e:
                    answer = f"调用失败：{type(e).__name__}: {e}"
                    st.error(answer)

        st.session_state.display_messages.append(
            {"role": "assistant", "content": answer}
        )

        # 只把最终对话写入 history，不把 trace 全塞回去。
        st.session_state.history.extend(
            [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": answer},
            ]
        )

if st.button("清空对话"):
    st.session_state.history = []
    st.session_state.display_messages = []
    st.rerun()
