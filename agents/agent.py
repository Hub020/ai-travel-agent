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

MULTI_ROUND_PLANNER_PROMPT = """你是“旅行规划决策者（Planner）”。
你的目标是基于当前对话，产出一个可执行的“下一步决策方案”，用于指导工具调用和结果整合。

要求：
1. 明确优先级（先查什么、后查什么）。
2. 明确需要调用的工具、每个工具的关键参数。
3. 给出最终输出结构（航班/酒店/天气/景点/美食）。
4. 用简体中文输出，简洁清晰。
"""

MULTI_ROUND_CRITIC_PROMPT = """你是“旅行规划评审者（Critic）”。
你的任务是评审 Planner 提供的方案是否完整、可执行、符合用户需求。

请严格按以下格式输出：
DECISION: APPROVE 或 REVISE
FEEDBACK: <具体改进意见；若通过可写“方案完整，可执行”>

评审标准：
- 是否覆盖用户关键需求（日期、预算倾向、偏好、节奏等）
- 是否明确工具调用步骤和参数
- 是否有明显逻辑遗漏或顺序问题
- 是否可落地执行
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
        self._enable_multi_round_decision = False
        self._max_decision_rounds = 2
        self._last_decision_trace = []
        # 使用阿里通义千问（免费模型，需设置 DASHSCOPE_API_KEY）
        self._tools_llm = ChatTongyi(
            model="qwen-turbo",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            temperature=0.7,
        ).bind_tools(TOOLS)
        # 借鉴 AutoGen 的“Planner + Critic”双角色协商。
        self._planner_llm = ChatTongyi(
            model="qwen-turbo",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            temperature=0.5,
        )
        self._critic_llm = ChatTongyi(
            model="qwen-turbo",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            temperature=0.2,
        )
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

    def configure_multi_round_decision(self, enabled: bool, max_rounds: int = 2):
        """运行时配置多轮决策功能。"""
        self._enable_multi_round_decision = enabled
        self._max_decision_rounds = max(1, min(max_rounds, 5))

    def get_last_decision_trace(self) -> list[dict]:
        """返回最近一次多轮决策对话记录。"""
        return self._last_decision_trace

    @staticmethod
    def _is_plan_approved(critic_text: str) -> bool:
        first_line = critic_text.strip().splitlines()[0].upper() if critic_text.strip() else ""
        return "APPROVE" in first_line and "REVISE" not in first_line

    def _run_multi_round_decision(self, messages: list[AnyMessage]) -> str:
        """通过 Planner/Critic 多轮博弈，得到更稳健的执行策略。"""
        self._last_decision_trace = []
        if not self._enable_multi_round_decision:
            return ""

        latest_plan = ""
        latest_feedback = ""
        approved = False

        for round_idx in range(self._max_decision_rounds):
            planner_messages = [SystemMessage(content=MULTI_ROUND_PLANNER_PROMPT)] + messages
            if latest_feedback:
                planner_messages.append(
                    HumanMessage(
                        content=(
                            "请根据以下评审意见修订你的决策方案，并给出更可执行的版本：\n"
                            f"{latest_feedback}"
                        )
                    )
                )
            planner_reply = self._planner_llm.invoke(planner_messages)
            latest_plan = planner_reply.content

            critic_messages = [
                SystemMessage(content=MULTI_ROUND_CRITIC_PROMPT),
                HumanMessage(
                    content=(
                        f"这是第 {round_idx + 1} 轮评审。\n"
                        "请评审以下 Planner 方案：\n\n"
                        f"{latest_plan}"
                    )
                ),
            ]
            critic_reply = self._critic_llm.invoke(critic_messages)
            latest_feedback = critic_reply.content
            self._last_decision_trace.append(
                {
                    "round": round_idx + 1,
                    "planner": latest_plan,
                    "critic": latest_feedback,
                    "approved": self._is_plan_approved(latest_feedback),
                }
            )
            if self._is_plan_approved(latest_feedback):
                approved = True
                break

        status = "已通过评审" if approved else "达到最大轮次，采用当前最优方案"
        return (
            "【多轮决策结论（AutoGen风格）】\n"
            f"状态：{status}\n"
            f"策略方案：\n{latest_plan}\n\n"
            f"最后一轮评审意见：\n{latest_feedback}\n\n"
            "请你严格依据上述策略决定是否调用工具，并产出最终旅行方案。"
        )

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
        strategy_context = self._run_multi_round_decision(messages)
        if strategy_context:
            messages = messages + [HumanMessage(content=strategy_context)]
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