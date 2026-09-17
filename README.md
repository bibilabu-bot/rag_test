# Simple Agent Framework

这是一个尽量简单、方便自己继续改的 Python Agent 框架。

当前只做四件事：

1. 自己填写 System Prompt
2. 自己填写模型 API
3. 自己写 Tool
4. 模型自动决定是否调用 Tool，并循环直到输出最终答案

没有引入 LangChain / LangGraph，方便你先把 Agent 原理和自己的业务流程跑通。

---

## 目录

```text
simple_agent_framework/
├─ app.py
├─ agent_core.py
├─ tools.py
├─ requirements.txt
├─ .env.example
└─ README.md
```

---

## 1. 安装

建议 Python 3.11+。

```bash
pip install -r requirements.txt
```

---

## 2. 启动

```bash
streamlit run app.py
```

浏览器会出现一个简单界面。

左侧填写：

- Base URL
- API Key
- Model
- Temperature

主界面填写：

- System Prompt
- 用户消息

---

## 3. 模型 API 要求

当前框架默认调用：

```text
POST {BASE_URL}/chat/completions
```

即 OpenAI Chat Completions 兼容格式。

例如：

```text
https://api.openai.com/v1
```

也可以换成你自己的中转站、国产模型平台、本地模型网关，只要兼容这个协议。

配置示例：

```env
MODEL_BASE_URL=https://api.openai.com/v1
MODEL_API_KEY=your_api_key
MODEL_NAME=gpt-4.1-mini
```

复制：

```bash
copy .env.example .env
```

Linux / macOS：

```bash
cp .env.example .env
```

也可以完全不写 `.env`，直接在网页侧边栏填。

---

## 4. Tool 怎么写

打开 `tools.py`。

普通 Python 函数就是 Tool。

例如：

```python
def get_weather(city: str):
    return {
        "city": city,
        "temperature": 25,
        "weather": "晴",
    }
```

然后注册：

```python
registry.register(
    get_weather,
    description="查询城市天气。",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称"
            }
        },
        "required": ["city"],
        "additionalProperties": False,
    },
)
```

模型看到的就是：

```text
get_weather(city)
```

当模型认为有必要时，它会自己调用。

---

## 5. Agent 是怎么工作的

核心逻辑只有：

```text
用户问题
   ↓
LLM
   ↓
有没有 tool_call？
   ├─ 没有 → 最终回答
   │
   └─ 有
       ↓
     Python Tool
       ↓
     Tool Result
       ↓
      LLM
       ↓
   再判断是否需要工具
```

代码在：

```text
agent_core.py
```

以后你做 Workflow，本质上就是在这个循环外面增加节点和判断。

---

## 6. Excel 示例工具

`tools.py` 已经放了一个：

```python
search_excel_rows()
```

它只是一个演示版关键词检索，不是真正的向量 RAG。

参数：

```text
query
excel_path
sheet_name
top_k
```

例如你把 Excel 放到：

```text
D:/rag/data.xlsx
```

可以在 System Prompt 中告诉 Agent：

```text
当用户询问诊断与医疗项目关系时，
优先使用 search_excel_rows 工具检索数据。

Excel 路径：
D:/rag/data.xlsx

工作表：
结果汇总
```

这样模型就可以自己去调用。

---

## 7. 后面怎么升级成 RAG

你以后只需要把：

```python
search_excel_rows()
```

换成：

```python
search_knowledge_base()
```

函数内部改为：

```text
query
 ↓
Embedding
 ↓
Vector DB
 ↓
Top K Chunks
 ↓
返回 Agent
```

Agent Core 一行都不用改。

推荐以后按这个顺序升级：

```text
现在
keyword search

↓

embedding + numpy

↓

FAISS / Chroma

↓

hybrid search

↓

reranker

↓

完整 RAG workflow
```

---

## 8. 后面怎么升级成工作流

目前：

```text
Agent
```

以后可以扩成：

```text
用户问题
   ↓
Intent Router
   ↓
┌─────────────┬──────────────┐
RAG Agent     SQL Agent
   ↓              ↓
Retrieval       Database
   └──────┬───────┘
          ↓
      Answer Agent
```

这时可以继续自己写，也可以再引入 LangGraph。

建议现在不要急着上 LangGraph。

先把：

```text
Prompt
Tool
Agent Loop
RAG
```

这四层搞明白，后面的 Workflow 会非常自然。

---

## 9. 一个重要说明

不是所有所谓“OpenAI Compatible API”都完整支持：

```text
tools
tool_choice
tool_calls
```

如果你的模型 API 只支持普通聊天、不支持 Function Calling，那么普通对话能运行，但 Agent Tool Loop 无法正常工作。

选择模型时需要确认它支持 Tool / Function Calling。
