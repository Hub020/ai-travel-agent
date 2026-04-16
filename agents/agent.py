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
    # 景点和美食推荐可用 LLM 内置知识，无需专门工具
    NEW_TOOLS = [find_cheap_flight_dates, get_weather_forecast]
except ImportError:
    print("Warning: New tools not found. Proceeding with basic tools only.")
    NEW_TOOLS = []

_ = load_dotenv()

CURRENT_YEAR = datetime.datetime.now().year


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


# 增强的系统提示词，支持多步骤规划（中文输出）
TOOLS_SYSTEM_PROMPT = f"""你是一个智能旅行规划助手。用户会提出一个包含目的地、时间、偏好等信息的请求。
你需要按以下步骤完成任务：

1. **确定实惠日期**：如果用户要求“本月内最实惠的日期”，你需要调用 `find_cheap_flight_dates` 工具。该工具会返回推荐日期。如果用户没有要求实惠日期，则使用用户指定的日期。
2. **查询航班和酒店**：基于步骤1确定的日期，调用 `flights_finder` 和 `hotels_finder` 工具获取具体信息。
3. **查询天气**：调用 `get_weather_forecast` 获取该日期的天气，并给出衣物建议。
4. **规划路线**：根据用户要求的“轻松休闲、不要网红打卡”，推荐2-3个当地经典、人少的景点，并给出行程安排。
5. **推荐美食**：根据用户的口味（例如不辣、当地正宗），推荐2-3道特色菜和餐厅。

你必须按顺序执行以上步骤，并在最终回答中汇总所有信息，以清晰的中文输出。

注意：
- 所有工具调用必须遵循工具的参数要求。如果某个工具不可用，请使用你的内置知识提供合理建议。
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
TOOLS = [flights_finder, hotels_finder] + NEW_TOOLS


class Agent:
    def __init__(self):
        self._tools = {t.name: t for t in TOOLS}
        # 使用阿里通义千问（免费模型，需设置 DASHSCOPE_API_KEY）
        self._tools_llm = ChatTongyi(
            model="qwen-turbo",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            temperature=0.7,
        ).bind_tools(TOOLS)

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
        memory = MemorySaver()
        self.graph = builder.compile(checkpointer=memory, interrupt_before=['email_sender'])

        print(self.graph.get_graph().draw_mermaid())

    @staticmethod
    def exists_action(state: AgentState):
        result = state['messages'][-1]
        if len(result.tool_calls) == 0:
            return 'email_sender'
        return 'more_tools'

    def email_sender(self, state: AgentState):
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
        messages = state['messages']
        messages = [SystemMessage(content=TOOLS_SYSTEM_PROMPT)] + messages
        message = self._tools_llm.invoke(messages)
        return {'messages': [message]}

    def invoke_tools(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
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