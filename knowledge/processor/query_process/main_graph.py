"""查询流程主图

使用 LangGraph 构建知识库查询工作流。

流程结构：
    item_name_confirm
          │
          ├── (有答案) ──────────> answer_output
          │                              │
          └── (无答案) ──> multi_search ─┬─>│
                             │          │  │
                   ┌─────────┼──────────┼──┤
                   v         v          v  v
              embedding   hyde       kg   web
                   │         │          │  │
                   └─────────┴──────────┴──┘
                                 │
                                join
                                 │
                                rrf
                                 │
                              rerank
                                 │
                           answer_output
                                 │
                                END
"""
from pathlib import Path

from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from knowledge.processor.query_process.state import (
    QueryGraphState,
    create_default_state,
)

# 查询流程节点
from knowledge.processor.query_process.nodes.item_name_confirm import ItemNameConfirm
from knowledge.processor.query_process.nodes.SearchEmbeddingNode import SearchEmbeddingNode
from knowledge.processor.query_process.nodes.search_embedding_hyde import search_embedding_hyde
from knowledge.processor.query_process.nodes.kg_query_node import KGQueryNode
from knowledge.processor.query_process.nodes.network_search_node import network_search_node
from knowledge.processor.query_process.nodes.RRF_node import RRF_node
from knowledge.processor.query_process.nodes.Rerank_node import Rerank_node
from knowledge.processor.query_process.nodes.answer_output_node import AnswerOutputNode

# 加载环境变量
load_dotenv()


# ==================== 路由函数 ====================

def route_after_item_confirm(state: QueryGraphState) -> bool:
    """商品名称确认后的路由

    如果已有答案（如闲聊回复），跳过检索直接输出。
    """
    if state.get("answer"):
        return True
    return False


# ==================== 虚节点 ====================

def multi_search_node(state: dict) -> dict:
    """多路搜索分发节点（虚节点，不做任何处理）"""
    return state


def join_node(state: dict) -> dict:
    """多路搜索汇合节点（虚节点，不做任何处理）"""
    return {}


# ==================== 图构建 ====================

def create_query_graph() -> StateGraph:
    """创建查询流程图

    Returns:
        编译后的 StateGraph 实例
    """
    workflow = StateGraph(QueryGraphState)

    nodes = {
        "item_name_confirm": ItemNameConfirm(),
        "multi_search": multi_search_node,
        "search_embedding": SearchEmbeddingNode(),
        "search_embedding_hyde": search_embedding_hyde(),
        "query_kg": KGQueryNode(),
        "web_search_mcp": network_search_node(),
        "join": join_node,
        "rrf": RRF_node(),
        "rerank": Rerank_node(),
        "answer_output": AnswerOutputNode(),
    }

    for name, node in nodes.items():
        workflow.add_node(name, node)

    # 入口
    workflow.set_entry_point("item_name_confirm")

    # 条件路由：有答案跳过检索
    workflow.add_conditional_edges(
        "item_name_confirm",
        route_after_item_confirm,
        {False: "multi_search", True: "answer_output"},
    )

    # 多路分发（并行）
    workflow.add_edge("multi_search", "search_embedding")
    workflow.add_edge("multi_search", "search_embedding_hyde")
    workflow.add_edge("multi_search", "query_kg")
    workflow.add_edge("multi_search", "web_search_mcp")

    # 多路汇合
    workflow.add_edge("search_embedding", "join")
    workflow.add_edge("search_embedding_hyde", "join")
    workflow.add_edge("query_kg", "join")
    workflow.add_edge("web_search_mcp", "join")

    # 顺序流
    workflow.add_edge("join", "rrf")
    workflow.add_edge("rrf", "rerank")
    workflow.add_edge("rerank", "answer_output")
    workflow.add_edge("answer_output", END)

    return workflow.compile()


# 全局图实例
query_app = create_query_graph()


def run_query(query: str, session_id: str = "", item_names: list = None, is_stream: bool = False) -> dict:
    """便捷函数：运行查询流程"""
    initial_state = create_default_state(
        session_id=session_id or "default",
        original_query=query,
        item_names=item_names or [],
        is_stream=is_stream,
    )

    final_state = None
    for event in query_app.stream(initial_state):
        for key, value in event.items():
            print(f"节点: {key}")
            final_state = value

    return final_state or initial_state


# ==================== 命令行入口 ====================

if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging
    setup_logging()

    print("=" * 60)
    print("知识库查询流程测试")
    print("=" * 60)

    test_query = "RS PRO RS-12数字万用表面板结构与端口示意图是什么"
    print(f"查询: {test_query}")
    print("-" * 60)

    try:
        result = run_query(
            query=test_query,
            session_id="test_001",
            item_names=[],
            is_stream=False,
        )

        print("-" * 60)
        print("流程完成!")
        print(f"识别商品: {result.get('item_names', [])}")
        print(f"检索切片数: {len(result.get('embedding_chunks', []))}")
        print(f"RRF 融合数: {len(result.get('rrf_chunks', []))}")
        print(f"Rerank 结果数: {len(result.get('reranked_docs', []))}")

        answer = result.get("answer", "N/A")
        print(f"答案: {answer[:500]}{'...' if len(answer) > 500 else ''}")

    except Exception as e:
        print(f"流程执行失败: {e}")
        import traceback
        traceback.print_exc()

    print("-" * 60)
    print("图结构:")
    query_app.get_graph().print_ascii()
