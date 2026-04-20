"""
旅行知识检索（RAG 检索层）

【当前生效】Embedding + FAISS
  - Embedding：DashScope（通义）文本向量，通过 `DashScopeEmbeddings` 调用，需环境变量 `DASHSCOPE_API_KEY`。
  - Vector DB：FAISS（Facebook AI Similarity Search），进程内内存索引；此处「Vector DB」特指 FAISS 实现，
    与下方注释块中「其它通用向量库（Chroma / Pinecone / PGVector 等）」方案区分。

【未采用 / 仅作对照】泛化「Embedding + 其它 Vector DB」写法见文件中部注释块。

【已废弃】纯关键词匹配检索整段源码保留在文件末尾（逐行 `#` 注释），便于与语义检索对照。
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------------------------
# 【未启用】泛称的「Embedding + Vector DB」其它对接方式（与当前 FAISS 区分）
# 若改用 Chroma / Pinecone / PGVector 等，通常替换为对应 LangChain VectorStore，
# 并配置持久化路径或服务端点；当前仓库未安装这些依赖，故整段保持注释。
# ---------------------------------------------------------------------------
# from langchain_community.vectorstores import Chroma
# # 示例：Chroma 持久化（需 `pip install chromadb`）
# # store = Chroma.from_documents(documents, embedding=embeddings, persist_directory="./chroma_db")
# from langchain_community.vectorstores import Pinecone
# # 示例：Pinecone 云服务（需 API key 与索引名）
# # Pinecone.from_documents(documents, embedding, index_name="travel-kb")


class KnowledgeRetriever:
    """Embedding（DashScope）+ 向量库 FAISS：语义相似度检索。"""

    def __init__(self) -> None:
        # 每类知识建立一个独立向量索引，避免不同领域内容相互干扰。
        self._base_path = Path(__file__).resolve().parent
        self._embeddings = DashScopeEmbeddings(
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        self._stores = {
            "attractions": self._build_vector_store("attractions"),
            "foods": self._build_vector_store("foods"),
        }

    def _load_knowledge_data(self, knowledge_type: str) -> List[Dict[str, Any]]:
        """优先从 JSON 读取，缺失时回退到 Python 常量数据。"""
        json_path = self._base_path / f"{knowledge_type}.json"
        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as file:
                return json.load(file)

        if knowledge_type == "attractions":
            from agents.knowledge.attractions import ATTRACTIONS_DATA

            return ATTRACTIONS_DATA

        from agents.knowledge.foods import FOODS_DATA

        return FOODS_DATA

    def _create_attraction_document(self, attraction: dict) -> Document:
        """把景点结构化数据拼成更适合语义检索的自然语言文档。"""
        highlights = "、".join(attraction.get("highlights", []))
        tips = attraction.get("tips", "")
        tags = attraction.get("tags", [])
        text = f"""{attraction["city"]} {attraction["name"]}（{attraction["category"]}）

景点简介：{attraction["description"]}

主要亮点：{highlights}

游览建议：{tips}

相关标签：{"、".join(tags)}"""
        return Document(
            page_content=text,
            metadata={
                "id": attraction.get("id", ""),
                "city": attraction["city"],
                "name": attraction["name"],
                "category": attraction["category"],
                "tags": tags,
                "knowledge_type": "attractions",
            },
        )

    def _create_food_document(self, food: dict) -> Document:
        """把美食结构化数据拼成可检索文档，并保留关键 metadata。"""
        restaurants = food.get("recommended_restaurants", [])
        restaurant_lines = [
            f"- {item['name']}（{item['location']}，价格档次：{item['price_level']}，推荐理由：{item['note']}）"
            for item in restaurants
        ]
        tips = food.get("tips", "")
        tags = food.get("tags", [])
        restaurant_block = "\n".join(restaurant_lines)
        text = f"""{food["city"]} {food["name"]}（{food["category"]}）

美食简介：{food["description"]}

推荐餐厅：
{restaurant_block}

食用建议：{tips}

相关标签：{"、".join(tags)}"""
        return Document(
            page_content=text,
            metadata={
                "id": food.get("id", ""),
                "city": food["city"],
                "name": food["name"],
                "category": food["category"],
                "tags": tags,
                "knowledge_type": "foods",
            },
        )

    def _build_vector_store(self, knowledge_type: str) -> FAISS:
        """根据知识类型构建 FAISS 索引（进程内内存索引）。"""
        data = self._load_knowledge_data(knowledge_type)
        if knowledge_type == "attractions":
            documents = [self._create_attraction_document(item) for item in data]
        else:
            documents = [self._create_food_document(item) for item in data]

        try:
            return FAISS.from_documents(documents, self._embeddings)
        except Exception as exc:
            raise RuntimeError(
                "FAISS 向量索引构建失败：请安装 faiss-cpu，并确认已设置 DASHSCOPE_API_KEY。"
            ) from exc

    def _semantic_search(
        self,
        knowledge_type: str,
        query: str,
        city: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """语义召回后再做业务过滤（城市/标签）并返回 top-k。"""
        vector_store = self._stores[knowledge_type]
        # 先多召回一些候选，再过滤，减少过滤后结果不足的问题。
        candidate_k = max(limit * 4, 10)
        docs_and_scores = vector_store.similarity_search_with_score(query, k=candidate_k)

        filtered_results: List[Dict[str, Any]] = []
        for document, score in docs_and_scores:
            metadata = document.metadata
            doc_tags = metadata.get("tags", [])

            if city and metadata.get("city") != city:
                continue
            if tags and not any(tag in doc_tags for tag in tags):
                continue

            filtered_results.append(
                {
                    "content": document.page_content,
                    "metadata": {
                        "city": metadata.get("city", ""),
                        "name": metadata.get("name", ""),
                        "category": metadata.get("category", ""),
                        "tags": ",".join(doc_tags),
                        # FAISS 返回距离，映射到 (0,1] 便于前端理解为“相似度”。
                        "similarity_score": round(1 / (1 + float(score)), 4),
                    },
                }
            )

            if len(filtered_results) >= limit:
                break

        return filtered_results

    def search_attractions(
        self,
        query: str,
        city: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        return self._semantic_search("attractions", query, city, tags, limit)

    def search_foods(
        self,
        query: str,
        city: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        return self._semantic_search("foods", query, city, tags, limit)

    def search_both(
        self,
        query: str,
        city: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 3,
    ) -> Dict[str, List[Dict[str, Any]]]:
        attractions = self.search_attractions(query, city, tags, limit)
        foods = self.search_foods(query, city, tags, limit)
        return {"attractions": attractions, "foods": foods}


_retriever_instance: Optional[KnowledgeRetriever] = None


def get_retriever() -> KnowledgeRetriever:
    """懒加载单例：避免每次工具调用都重复构建 FAISS 索引。"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = KnowledgeRetriever()
    return _retriever_instance


def retrieve_travel_knowledge(
    query: str,
    city: Optional[str] = None,
    knowledge_type: str = "both",
    limit: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    """统一检索入口：按类型路由到景点/美食/混合检索。"""
    retriever = get_retriever()

    if knowledge_type == "attractions":
        return {"attractions": retriever.search_attractions(query, city, limit=limit)}
    if knowledge_type == "foods":
        return {"foods": retriever.search_foods(query, city, limit=limit)}
    return retriever.search_both(query, city, limit=limit)


if __name__ == "__main__":
    result = retrieve_travel_knowledge("有什么好吃的", city="北京", limit=2)
    print("=== 北京美食搜索结果 ===")
    for food in result.get("foods", []):
        print(f"\n--- {food['metadata']['name']} ---")
        print(food["content"][:200])

    result = retrieve_travel_knowledge("有哪些景点", city="杭州", limit=2)
    print("\n=== 杭州景点搜索结果 ===")
    for attr in result.get("attractions", []):
        print(f"\n--- {attr['metadata']['name']} ---")
        print(attr["content"][:200])


# ==============================================================================
# 【已废弃 / 逐行注释保留】关键词检索版 —— 无 Embedding、无 FAISS、无语义向量
# 与上方「Embedding + FAISS」对照；勿取消注释与当前类混用。
# ==============================================================================
# from typing import Optional, List, Dict, Any
#
# from agents.knowledge.attractions import ATTRACTIONS_DATA
# from agents.knowledge.foods import FOODS_DATA
#
#
# class KnowledgeRetriever:
#     def __init__(self):
#         self.attractions = ATTRACTIONS_DATA
#         self.foods = FOODS_DATA
#
#     def _create_attraction_document(self, attraction: dict) -> str:
#         highlights = "、".join(attraction.get("highlights", []))
#         tips = attraction.get("tips", "")
#         tags = "、".join(attraction.get("tags", []))
#         return f"""{attraction["city"]} {attraction["name"]}（{attraction["category"]}）
#
# 景点简介：{attraction["description"]}
#
# 主要亮点：{highlights}
#
# 游览建议：{tips}
#
# 相关标签：{tags}"""
#
#     def _create_food_document(self, food: dict) -> str:
#         desc = food["description"]
#         restaurants = food.get("recommended_restaurants", [])
#         restaurant_info = ""
#         for r in restaurants:
#             restaurant_info += f"- {r['name']}（{r['location']}，价格档次：{r['price_level']}，推荐理由：{r['note']}）\n"
#         tips = food.get("tips", "")
#         tags = "、".join(food.get("tags", []))
#         return f"""{food["city"]} {food["name"]}（{food["category"]}）
#
# 美食简介：{desc}
#
# 推荐餐厅：
# {restaurant_info}
#
# 食用建议：{tips}
#
# 相关标签：{tags}"""
#
#     def _keyword_score(self, text: str, query: str) -> int:
#         score = 0
#         query_tokens = query.lower().split()
#         text_lower = text.lower()
#         for token in query_tokens:
#             if token in text_lower:
#                 score += 1
#         return score
#
#     def search_attractions(
#         self,
#         query: str,
#         city: Optional[str] = None,
#         tags: Optional[list[str]] = None,
#         limit: int = 3
#     ) -> List[Dict[str, Any]]:
#         results = []
#
#         for attr in self.attractions:
#             if city and attr["city"] != city:
#                 continue
#             if tags and not any(tag in attr.get("tags", []) for tag in tags):
#                 continue
#
#             text = self._create_attraction_document(attr)
#             score = self._keyword_score(text, query)
#             if score > 0:
#                 results.append((attr, score))
#
#         results.sort(key=lambda x: x[1], reverse=True)
#         top_results = results[:limit]
#
#         formatted_results = []
#         for attr, score in top_results:
#             formatted_results.append({
#                 "content": self._create_attraction_document(attr),
#                 "metadata": {
#                     "city": attr["city"],
#                     "name": attr["name"],
#                     "category": attr["category"],
#                     "tags": ",".join(attr.get("tags", []))
#                 }
#             })
#         return formatted_results
#
#     def search_foods(
#         self,
#         query: str,
#         city: Optional[str] = None,
#         tags: Optional[list[str]] = None,
#         limit: int = 3
#     ) -> List[Dict[str, Any]]:
#         results = []
#
#         for food in self.foods:
#             if city and food["city"] != city:
#                 continue
#             if tags and not any(tag in food.get("tags", []) for tag in tags):
#                 continue
#
#             text = self._create_food_document(food)
#             score = self._keyword_score(text, query)
#             if score > 0:
#                 results.append((food, score))
#
#         results.sort(key=lambda x: x[1], reverse=True)
#         top_results = results[:limit]
#
#         formatted_results = []
#         for food, score in top_results:
#             formatted_results.append({
#                 "content": self._create_food_document(food),
#                 "metadata": {
#                     "city": food["city"],
#                     "name": food["name"],
#                     "category": food["category"],
#                     "tags": ",".join(food.get("tags", []))
#                 }
#             })
#         return formatted_results
#
#     def search_both(
#         self,
#         query: str,
#         city: Optional[str] = None,
#         tags: Optional[list[str]] = None,
#         limit: int = 3
#     ) -> Dict[str, List[Dict[str, Any]]]:
#         attractions = self.search_attractions(query, city, tags, limit)
#         foods = self.search_foods(query, city, tags, limit)
#         return {
#             "attractions": attractions,
#             "foods": foods
#         }
