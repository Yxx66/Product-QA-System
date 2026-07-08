"""
Rerank 重排序节点

使用 BGE Reranker 对多路检索结果进行深度语义重排序。
采用分层设计，每层职责清晰，便于替换 reranker 或扩展开源策略。

架构：
    process(state)                   # 编排器
    ├── _collect_chunks(state)       # 合并多来源 chunks（按 chunk_id 去重）
    ├── _prepare_documents(chunks)   # 提取 content 文本列表
    ├── _rerank(query, documents)    # 调用 BGE Reranker（核心）
    └── _build_reranked_chunks(chunks, results)  # 分数映射回原始 chunks

扩展点：
    - 替换 reranker：重写 _rerank 方法或抽成独立策略类
    - 来源加权：_collect_chunks 已标记 _source 字段
    - Top-K 截断：通过 _top_k_rerank 断崖式截断
"""

from typing import List, Dict, Any, Tuple

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.config import get_config


class Rerank_node(BaseNode):
    """Rerank 重排序节点。

    流程: 收集四路检索结果 → 提取 content → BGE Reranker → 重排序输出
    """

    name = "rerank"

    # ================================================================== #
    #                           主流程                                     #
    # ================================================================== #

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """Rerank 编排器。"""
        config = get_config()
        min_top_k = config.rerank_min_topk
        max_top_k = config.rerank_max_topk
        gap_abs = config.rerank_gap_abs
        gap_ratio = config.rerank_gap_ratio

        # 1. 获取 query
        query = state.get("rewritten_query") or state.get("original_query", "")
        if not query:
            self.logger.warning("query 为空，跳过 rerank")
            state["reranked_docs"] = []
            return state

        # 2. 收集并去重多路 chunks
        chunks = self._collect_chunks(state)
        if not chunks:
            self.logger.warning("无可用的检索结果，跳过 rerank")
            state["reranked_docs"] = []
            return state

        self.logger.info(f"Rerank 输入: {len(chunks)} 条待重排序")

        # 3. 提取 content 文本列表
        documents = self._prepare_documents(chunks)

        # 4. 调用 Reranker
        try:
            results = self._rerank(query, documents)
        except Exception as e:
            self.logger.error(f"Reranker 调用失败: {e}，降级使用原始顺序")
            state["reranked_docs"] = chunks
            return state

        if not results:
            self.logger.warning("Reranker 返回空结果，降级使用原始顺序")
            state["reranked_docs"] = chunks
            return state

        # 5. 将 rerank 分数映射回原始 chunks 并排序
        reranked_chunks = self._build_reranked_chunks(chunks, results)
        self.logger.info(f"Rerank 完成，输出 {len(reranked_chunks)} 条")

        if reranked_chunks:
            scores = [c.get("rerank_score", 0) for c in reranked_chunks]
            self.logger.info(f"Rerank 分数范围: [{min(scores):.6f}, {max(scores):.6f}]")

        # 6. 断崖式 Top-K 截断
        reranked_chunks = self._top_k_rerank(
            reranked_chunks,
            max_top_k=max_top_k,
            min_top_k=min_top_k,
            gap_abs=gap_abs,
            gap_ratio=gap_ratio,
        )
        self.logger.info(f"Top-K 截断后: {len(reranked_chunks)} 条")

        state["reranked_docs"] = reranked_chunks
        return state

    # ================================================================== #
    #                      收集 & 去重 chunks                              #
    # ================================================================== #

    def _collect_chunks(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """合并多路检索结果，按 chunk_id 去重，保留首次出现的版本。

        来源字段映射：
            embedding/hyde: pk → chunk_id
            kg: source_id → chunk_id
            web_search: 无 chunk_id，以 content 前 100 字符为 key 去重

        Returns:
            去重后的 chunk 列表，每项含 chunk_id, content, item_name, title,
            file_title, parent_title, score, _source
        """
        seen: set = set()
        merged: List[Dict[str, Any]] = []

        # embedding & hyde：格式一致，pk → chunk_id
        for source_name, source_key in [("embedding", "embedding_chunks"),
                                        ("hyde", "hyde_embedding_chunks")]:
            for doc in (state.get(source_key) or []):
                entity = self._normalize_chunk(doc, source_name)
                chunk_id = entity.get("chunk_id", "")
                if chunk_id and chunk_id not in seen:
                    seen.add(chunk_id)
                    merged.append(entity)

        # kg：source_id → chunk_id
        for doc in (state.get("kg_chunks") or []):
            entity = self._normalize_chunk(doc, "kg")
            chunk_id = entity.get("chunk_id", "")
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                merged.append(entity)

        # web_search：无标准 chunk_id，用 content 前 100 字符去重
        for doc in (state.get("web_search_docs") or []):
            entity = self._normalize_chunk(doc, "web_search")
            content = entity.get("content", "")
            dedup_key = content[:100]
            if dedup_key and dedup_key not in seen:
                seen.add(dedup_key)
                merged.append(entity)

        return merged

    # ================================================================== #
    #                      chunk 归一化                                    #
    # ================================================================== #

    def _normalize_chunk(self, doc: Dict, source: str) -> Dict[str, Any]:
        """将各路文档统一归一化为标准 entity 格式。

        Args:
            doc: 原始文档 dict
            source: 来源标记 "embedding" | "hyde" | "kg" | "web_search"

        Returns:
            统一 entity dict
        """
        # 兼容 {"entity": {...}, "distance": ...} 包装
        raw = doc.get("entity") or doc

        # 统一 chunk_id 字段
        chunk_id = (
            raw.get("chunk_id")
            or raw.get("pk")
            or raw.get("source_id")
            or raw.get("id", "")
        )

        entity = {
            "chunk_id": chunk_id,
            "content": raw.get("content", ""),
            "item_name": raw.get("item_name", ""),
            "title": raw.get("title", ""),
            "file_title": raw.get("file_title", ""),
            "parent_title": raw.get("parent_title", ""),
            "score": raw.get("score", 0.0),
            "_source": source,
        }
        return entity

    # ================================================================== #
    #                    提取 content 文本列表                              #
    # ================================================================== #

    def _prepare_documents(self, chunks: List[Dict]) -> List[str]:
        """从 chunks 中提取 content 文本列表，用于传给 reranker。

        Args:
            chunks: 归一化后的 chunk 列表（含 content 字段）

        Returns:
            content 文本列表，顺序与 chunks 一一对应
        """
        return [c.get("content", "") for c in chunks]

    # ================================================================== #
    #                    BGE Reranker 核心调用                              #
    # ================================================================== #

    def _rerank(self, query: str, documents: List[str]) -> List[Any]:
        """调用 BGE Reranker 对 query 和 documents 进行语义重排序。

        Args:
            query: 用户查询文本
            documents: 待重排序的文档内容列表

        Returns:
            RerankResultItem 列表，按相关性得分降序排列。
            每个 item 包含 .score 和 .text 属性。

        Raises:
            各种异常由调用方捕获处理
        """
        import os
        from pymilvus.model.reranker import BGERerankFunction

        model_path = os.getenv("RERANK_MODEL_PATH", "BAAI/bge-reranker-v2-m3")
        device = os.getenv("RERANK_DEVICE", "cpu")
        bge_rf = BGERerankFunction(
            model_name=model_path,
            device=device,
        )

        results = bge_rf(query, documents)
        return results

    # ================================================================== #
    #                  分数映射 & 重排序                                   #
    # ================================================================== #

    def _build_reranked_chunks(
        self,
        chunks: List[Dict],
        rerank_results: List[Any],
    ) -> List[Dict]:
        """将 reranker 返回的分数映射回原始 chunks，并按得分降序排序。

        Args:
            chunks: 原始 chunk 列表（与传入 _rerank 的顺序一致）
            rerank_results: _rerank 返回的结果列表。
                每个 item 有 .text（文档内容）和 .score（重排序得分）。
                注意：bge_rf 返回的结果顺序就是按得分降序排列的，
                但结果数量可能少于输入（某些 reranker 可能截断），
                且 text 内容与 chunk content 完全对应。

        Returns:
            按 rerank_score 降序排列的 chunk 列表。
            每个 chunk 添加 "rerank_score" 字段。
        """
        # 构建 text → score 映射（reranker 可能返回部分结果）
        text_to_score: Dict[str, float] = {}
        for item in rerank_results:
            text_to_score[item.text] = item.score

        # 为每个 chunk 注入 rerank_score
        scored_chunks: List[Dict] = []
        for chunk in chunks:
            content = chunk.get("content", "")
            score = text_to_score.get(content, 0.0)
            chunk_with_score = {**chunk, "rerank_score": score}
            scored_chunks.append(chunk_with_score)

        # 按 rerank_score 降序排序，分数相同则保持原始顺序
        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored_chunks


    # ================================================================== #
    #                   断崖式 Top-K 截断                                  #
    # ================================================================== #

    def _top_k_rerank(
        self,
        chunks: List[Dict],
        max_top_k: int,
        min_top_k: int,
        gap_abs: float,
        gap_ratio: float,
    ) -> List[Dict]:
        """断崖式 Top-K 截断。

        在 [min_top_k, max_top_k] 区间内寻找得分"断崖"——
        当相邻两个 chunk 的 rerank_score 同时满足：
          - 绝对差距 > gap_abs
          - 相对降幅 > gap_ratio
        时认为出现断崖，在此处截断返回。

        若无断崖，则返回前 max_top_k 条。

        Args:
            chunks: 已按 rerank_score 降序排列的 chunk 列表
            max_top_k: 最大返回条数
            min_top_k: 最小返回条数（断崖搜索起点）
            gap_abs: 绝对分差阈值
            gap_ratio: 相对降幅阈值

        Returns:
            截断后的 chunk 列表
        """
        if not chunks:
            return []

        score_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        total = len(score_chunks)

        # 边界保护：如果总条数 <= min_top_k，直接返回前 max_top_k
        if total <= min_top_k:
            return score_chunks[:max_top_k]

        # 从 min_top_k 开始检查相邻分差（0-indexed）
        end = min(max_top_k - 1, total - 1)
        for i in range(min_top_k - 1, end):
            current_score = score_chunks[i]["rerank_score"]
            next_score = score_chunks[i + 1]["rerank_score"]

            # 分数相等不算断崖
            if current_score == next_score:
                continue

            abs_diff = current_score - next_score
            ratio_diff = abs_diff / current_score

            if abs_diff > gap_abs and ratio_diff > gap_ratio:
                return score_chunks[:i + 1]

        # 没找到断崖 → 取满 max_top_k
        return score_chunks[:max_top_k]


# ================================================================== #
#                        兼容 & 测试                                   #
# ================================================================== #

_node_instance = Rerank_node()


def node_rerank(state: QueryGraphState) -> QueryGraphState:
    """兼容原有调用方式的入口函数。"""
    return _node_instance(state)


if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging
    from knowledge.processor.query_process.state import create_default_state

    setup_logging()

    print("=" * 60)
    print("开始测试: Rerank 重排序节点 (Rerank_node)")
    print("=" * 60)

    # 模拟四路检索结果（匹配真实上游输出格式）
    mock_state = create_default_state()
    mock_state["rewritten_query"] = "万用表怎么测电压"

    mock_state["embedding_chunks"] = [
        {"pk": "chunk_1", "content": "向量搜索结果#1：万用表测电压需要将红黑表笔并联...", "title": "## 标题1", "file_title": "测试文档", "parent_title": "## 父标题", "item_name": "测试商品", "score": 0.95},
        {"pk": "chunk_2", "content": "向量搜索结果#2：电压测量是万用表最常用的功能之一", "title": "## 标题2", "file_title": "测试文档", "parent_title": "## 父标题", "item_name": "测试商品", "score": 0.90},
        {"pk": "chunk_3", "content": "向量搜索结果#3：电阻测量需要断开电源", "title": "## 标题3", "file_title": "测试文档", "parent_title": "## 父标题", "item_name": "测试商品", "score": 0.85},
    ]
    mock_state["hyde_embedding_chunks"] = [
        {"pk": "chunk_2", "content": "HyDE搜索结果#1：电压测量步骤详解", "title": "## HyDE标题", "file_title": "测试文档", "parent_title": "## 父标题", "item_name": "测试商品", "score": 0.88},
        {"pk": "chunk_4", "content": "HyDE搜索结果#2：万用表使用注意事项", "title": "## HyDE标题", "file_title": "测试文档", "parent_title": "## 父标题", "item_name": "测试商品", "score": 0.75},
    ]
    mock_state["kg_chunks"] = [
        {"source_id": "chunk_5", "content": "知识图谱结果#1：测电压 - 红黑表笔接法", "item_name": "测试商品", "score": 4.0, "cnt": 3},
        {"source_id": "chunk_1", "content": "知识图谱结果#2：测电压 - 档位选择", "item_name": "测试商品", "score": 2.0, "cnt": 1},
    ]
    mock_state["web_search_docs"] = [
        {"title": "万用表测电压教程", "content": "网络搜索结果#1：将万用表调至电压档...", "score": 0.7},
        {"title": "数字万用表使用方法", "content": "网络搜索结果#2：测电压时注意量程...", "score": 0.6},
    ]

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print(f"  kg_chunks: {len(mock_state['kg_chunks'])} 条")
    print(f"  web_search_docs: {len(mock_state['web_search_docs'])} 条")
    print("-" * 60)

    # 执行 rerank
    result = node_rerank(mock_state)

    # 打印结果
    print("\n【Rerank 结果】:")
    for i, chunk in enumerate(result.get("reranked_docs", []), 1):
        score_val = chunk.get('rerank_score', 'N/A')
        score_str = f"{score_val:.6f}" if isinstance(score_val, (int, float)) else str(score_val)
        print(f"[{i}] score={score_str} | "
              f"source={chunk.get('_source', '?')} | "
              f"content={chunk.get('content', '')}")