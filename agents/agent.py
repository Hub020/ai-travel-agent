"""LangGraph 编排层：负责在 LLM 与工具之间循环决策，并在末尾执行发邮件。

核心节点：
- call_tools_llm: 让模型决定下一步是否需要调工具；
- invoke_tools: 执行模型请求的工具调用；
- email_sender: 将最终结果转 HTML 并发邮件。
"""

# pylint: disable = http-used,print-used,no-self-use

import datetime
import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_community.chat_models import ChatTongyi   # 使用通义千问（免费）
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# 导入已有工具
from agents.tools.flights_finder import flights_finder
from agents.tools.hotels_finder import hotels_finder

# 导入新增工具（如果尚未实现，可先注释，后续补充）
try:
    from agents.tools.cheap_dates_finder import find_cheap_flight_dates
    from agents.tools.weather import get_weather_forecast
    NEW_TOOLS = [find_cheap_flight_dates, get_weather_forecast]
except ImportError:
    print("Warning: New tools not found. Proceeding with basic tools only.")
    NEW_TOOLS = []

try:
    from agents.knowledge.retriever import retrieve_travel_knowledge
    RAG_AVAILABLE = True
except ImportError:
    print("Warning: RAG knowledge base not found. Proceeding without RAG.")
    RAG_AVAILABLE = False

_ = load_dotenv()

CURRENT_YEAR = datetime.datetime.now().year


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


# 增强的系统提示词，支持多步骤规划（中文输出）
TOOLS_SYSTEM_PROMPT = f"""你是一个智能旅行规划助手。用户会提出一个包含目的地、时间、偏好等信息的请求。
你需要按以下步骤完成任务：

1. **确定实惠日期**：如果用户要求"本月内最实惠的日期"，你需要调用 `find_cheap_flight_dates` 工具。该工具会返回推荐日期。如果用户没有要求实惠日期，则使用用户指定的日期。
2. **查询航班和酒店**：基于步骤1确定的日期，调用 `flights_finder` 和 `hotels_finder` 工具获取具体信息。
3. **查询天气**：调用 `get_weather_forecast` 获取该日期的天气，并给出衣物建议。
4. **规划路线**：根据用户要求的"轻松休闲、不要网红打卡"，推荐2-3个当地经典、人少的景点，并给出行程安排。
5. **推荐美食**：根据用户的口味（例如不辣、当地正宗），推荐2-3道特色菜和餐厅。

重要：当用户询问景点、美食、当地推荐等问题时，必须调用 `knowledge_retriever_tool` 工具从知识库检索信息，
而不是依赖模型内置知识。知识库检索能提供更准确、更权威的推荐。

知识库工具使用说明：
- 查询景点：knowledge_type="attractions"，传入城市名和查询关键词
- 查询美食：knowledge_type="foods"，传入城市名和查询关键词
- 同时查询：knowledge_type="both"

你必须按顺序执行以上步骤，并在最终回答中汇总所有信息，以清晰的中文输出。

注意：
- 所有工具调用必须遵循工具的参数要求。
- 当用户提供城市名时，请自动转换为 IATA 机场代码（如北京 -> PEK，上海 -> SHA）。
- 价格货币统一使用人民币（CNY）。
- 最终回答必须全部使用简体中文，包括标签、描述等。

你可以同时调用多个工具，例如一次性调用 flights_finder 和 hotels_finder，减少交互轮次。

重要：在输出航班、酒店、景点、美食时，必须附上相关链接：
- 航班：使用返回的 `booking_link` 字段，显示为 [预订链接]
- 酒店：使用返回的 `website_link` 字段，显示为 [酒店官网]
- 景点：提供百度百科或百度地图搜索链接 `https://map.baidu.com/search/景点名称`（例如：`https://map.baidu.com/search/外滩`
- 美食：提供大众点评搜索链接（例如：https://www.dianping.com/search/keyword/城市/餐厅名）

所有链接必须使用 Markdown 格式 [文本](URL)。如果某个项目没有具体链接，可以提供搜索建议。

当前年份：{CURRENT_YEAR}
"""

EMAILS_SYSTEM_PROMPT = """你的任务是将结构化的类Markdown文本转换为有效的HTML邮件正文。

- 不要在你的回复中包含 ```html 前置声明。
- 输出应为正确的HTML格式，可直接用作邮件正文。
- 使用简体中文。

示例：
输入：
我想从北京到上海，10月1-7日，帮我找航班和酒店。

输出：
<!DOCTYPE html>
<html>
<head>
    <title>航班和酒店选项</title>
</head>
<body>
    <h2>北京到上海的航班</h2>
    ...
</body>
</html>
"""

# 合并所有工具
from agents.tools.knowledge_base import knowledge_retriever_tool
TOOLS = [flights_finder, hotels_finder, knowledge_retriever_tool] + NEW_TOOLS


class Agent:
    def __init__(self):
        # name -> tool 实例映射，便于按 tool_call 名称分发调用。
        self._tools = {t.name: t for t in TOOLS}
        # 使用阿里通义千问（免费模型，需设置 DASHSCOPE_API_KEY）
        self._tools_llm = ChatTongyi(
            model="qwen-turbo",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            temperature=0.7,
        ).bind_tools(TOOLS)
        # 状态图构建：call_tools_llm <-> invoke_tools 循环，直到无工具调用再进入 email_sender。
        builder = StateGraph(AgentState)
        builder.add_node('call_tools_llm', self.call_tools_llm)
        builder.add_node('invoke_tools', self.invoke_tools)
        builder.add_node('email_sender', self.email_sender)
        builder.set_entry_point('call_tools_llm')

        builder.add_conditional_edges(
            'call_tools_llm',
            Agent.exists_action,
            {'more_tools': 'invoke_tools', 'email_sender': 'email_sender'}
        )
        builder.add_edge('invoke_tools', 'call_tools_llm')
        builder.add_edge('email_sender', END)
        # MemorySaver 让同一 thread_id 能在多次 invoke 之间共享上下文。
        memory = MemorySaver()
        # 在邮件节点前中断，给前端一个“是否发送邮件”的人工确认机会。
        self.graph = builder.compile(checkpointer=memory, interrupt_before=['email_sender'])

        print(self.graph.get_graph().draw_mermaid())

    @staticmethod
    def exists_action(state: AgentState):
        """检查模型是否请求工具调用，决定图的下一跳。"""
        result = state['messages'][-1]  # 获取最后一条 AI 消息
        if len(result.tool_calls) == 0:
            return 'email_sender'
        return 'more_tools'

    def email_sender(self, state: AgentState):
        """将最终文本转成 HTML，并通过 SendGrid 发出邮件。"""
        print('Sending email')
        email_llm = ChatTongyi(
            model="qwen-turbo",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            temperature=0.1,
        )
        email_message = [
            SystemMessage(content=EMAILS_SYSTEM_PROMPT),
            HumanMessage(content=state['messages'][-1].content)
        ]
        email_response = email_llm.invoke(email_message)
        print('Email content:', email_response.content)

        message = Mail(
            from_email=os.environ['FROM_EMAIL'],
            to_emails=os.environ['TO_EMAIL'],
            subject=os.environ['EMAIL_SUBJECT'],
            html_content=email_response.content
        )
        try:
            sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
            response = sg.send(message)
            print(response.status_code)
            print(response.body)
            print(response.headers)
        except Exception as e:
            print(str(e))

    def call_tools_llm(self, state: AgentState):
        """给工具型 LLM 注入系统提示词，让其规划并产出 tool_calls。"""
        messages = state['messages']
        messages = [SystemMessage(content=TOOLS_SYSTEM_PROMPT)] + messages
        message = self._tools_llm.invoke(messages)
        return {'messages': [message]}

    def invoke_tools(self, state: AgentState):
        """执行 LLM 请求的工具，并把结果包装为 ToolMessage 回填对话。"""
        tool_calls = state['messages'][-1].tool_calls  # 获取 LLM 请求调用的工具列表
        results = []
        for t in tool_calls:
            print(f'Calling: {t}')
            if not t['name'] in self._tools:
                print('\n ....bad tool name....')
                result = 'bad tool name, retry'
            else:
                result = self._tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print('Back to the model!')
        return {'messages': results}