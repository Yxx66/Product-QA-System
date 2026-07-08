"""查询流程状态类型定义

每个检索节点写入独立字段，避免多路结果冲突。
"""
from typing import TypedDict, List
import copy


class QueryGraphState(TypedDict):
    """查询流程图状态

    字段分组：
    - 会话信息: session_id, message_id
    - 输入数据: original_query, item_names, history, rewritten_query
    - 检索结果: embedding_chunks, hyde_embedding_chunks, kg_chunks, web_search_docs
    - 融合结果: rrf_chunks, reranked_docs
    - 输出数据: prompt, answer
    - 控制标志: is_stream
    """
    session_id: str
    message_id: str
    original_query: str
    embedding_chunks: list
    hyde_embedding_chunks: list
    rrf_chunks: list
    web_search_docs: list
    reranked_docs: list
    prompt: str
    answer: str
    item_names: List[str]
    rewritten_query: str
    history: list
    is_stream: bool
    kg_chunks: list
    kg_triples: list


# ==================== 默认状态 ====================
DEFAULT_STATE: QueryGraphState = {
    "session_id": "",
    "message_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "rrf_chunks": [],
    "web_search_docs": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
    "kg_chunks": [],
    "kg_triples": [],
}


def create_default_state(**overrides) -> QueryGraphState:
    """创建默认状态，支持字段覆盖"""
    state = copy.deepcopy(DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> QueryGraphState:
    """获取默认状态副本"""
    return copy.deepcopy(DEFAULT_STATE)


# 兼容旧版
graph_default_state = DEFAULT_STATE
