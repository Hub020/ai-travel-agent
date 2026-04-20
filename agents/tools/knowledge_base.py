"""知识库工具：把 RAG 检索结果转成 LLM 更容易消费的文本块。"""

from typing import Optional

from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool


class KnowledgeInput(BaseModel):
    query: str = Field(description='Search query for attractions or food')
    city: Optional[str] = Field(None, description='City name to filter results')
    knowledge_type: str = Field('both', description='Type: "attractions", "foods", or "both"')
    limit: int = Field(3, description='Maximum number of results to return')


@tool(args_schema=KnowledgeInput)
def knowledge_retriever_tool(
    query: str,
    city: Optional[str] = None,
    knowledge_type: str = 'both',
    limit: int = 3
) -> str:
    """Retrieve travel knowledge including attractions and local foods from the knowledge base.

    Use this tool when users ask about:
    - Tourist attractions, sightseeing spots, scenic places
    - Local food recommendations, restaurants, cuisine
    - Things to do in a specific city
    - Off-the-beaten-path recommendations

    The knowledge base contains curated information about attractions and foods in major Chinese cities.
    """
    from agents.knowledge.retriever import retrieve_travel_knowledge

    # 委托给底层 retriever 执行语义检索与过滤。
    result = retrieve_travel_knowledge(
        query=query,
        city=city,
        knowledge_type=knowledge_type,
        limit=limit
    )

    # 将结构化检索结果渲染为分段文本，便于模型直接引用。
    formatted_output = []

    if 'attractions' in result and result['attractions']:
        formatted_output.append("【景点推荐】")
        for attr in result['attractions']:
            metadata = attr['metadata']
            content = attr['content']
            formatted_output.append(f"\n📍 {metadata['name']}（{metadata['city']}）")
            formatted_output.append(f"类型：{metadata['category']}")
            formatted_output.append(content)
            formatted_output.append("")

    if 'foods' in result and result['foods']:
        formatted_output.append("\n【美食推荐】")
        for food in result['foods']:
            metadata = food['metadata']
            content = food['content']
            formatted_output.append(f"\n🍜 {metadata['name']}（{metadata['city']}）")
            formatted_output.append(f"类型：{metadata['category']}")
            formatted_output.append(content)
            formatted_output.append("")

    if not formatted_output:
        return "未找到相关知识库信息，请尝试其他查询词。"

    return "\n".join(formatted_output)