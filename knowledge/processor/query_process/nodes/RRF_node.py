"""
RRF 融合排序节点

使用 Reciprocal Rank Fusion 算法融合多路检索结果。
（骨架代码，后续完善）
"""

from typing import List, Dict, Any, Tuple

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.config import get_config
from knowledge.processor.query_process.exceptions import ValidationError


class RRF_node(BaseNode):
    """RRF 融合排序节点。

    流程: 收集三路检索结果 → 带权重 RRF 融合 → 按得分降序返回
    """

    name = "rrf"

    # ================================================================== #
    #                           主流程                                     #
    # ================================================================== #

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # RRF 配置参数（可通过 ImportConfig 扩展）
        kg_weight = 0.7      # 知识图谱权重，默认 0.7
        k = 60               # RRF 平滑常数 k
        max_results = None   # 最大返回结果数，None 表示全部

        # Step 1: 收集三路检索结果
        sources = {
            "embedding": (
                self._extract_entities(state.get("embedding_chunks"), source="embedding"),
                1.0,
            ),
            "hyde": (
                self._extract_entities(state.get("hyde_embedding_chunks"), source="hyde"),
                1.0,
            ),
            "kg": (
                self._extract_entities(state.get("kg_chunks"), source="kg"),
                kg_weight,
            ),
        }

        self.logger.info(
            f"RRF 输入: {', '.join(f'{k}={len(v[0])}' for k, v in sources.items())}"
        )

        # Step 2-5: 执行 RRF 融合
        source_weights = list(sources.values())
        rrf_results = self._reciprocal_rank_fusion(
            source_weights,
            k=k,
            max_results=max_results,
        )

        # Step 6: 输出结果
        rrf_chunks = [doc for doc, _ in rrf_results]
        self.logger.info(f"RRF 融合完成，返回 {len(rrf_chunks)} 条结果")

        if rrf_results:
            scores = [s for _, s in rrf_results]
            self.logger.info(f"分数范围: [{min(scores):.6f}, {max(scores):.6f}]")

        state["rrf_chunks"] = rrf_chunks
        return state

    # ================================================================== #
    #                      RRF 算法实现                                    #
    # ================================================================== #

    @staticmethod
    def _reciprocal_rank_fusion(
        source_weights: List[Tuple[List[Dict], float]],
        k: int = 60,
        max_results: int = None,
    ) -> List[Tuple[Dict, float]]:
        """带权重的 RRF 融合。

        公式: score(d) = Σ weight_i / (k + rank_i(d))

        Args:
            source_weights: [(文档列表, 权重), ...]
            k: RRF 常数，值越大则排名差异对得分的影响越平滑。
            max_results: 返回前 N 个，None 表示全部。

        Returns:
            [(文档, 得分), ...] 按得分降序。
        """
        # Step 3: 构建得分映射表
        score_map: Dict[str, float] = {}
        # Step 4: 构建文档映射表
        chunk_map: Dict[str, Dict] = {}

        for rank_list, weight in source_weights:
            for pos, item in enumerate(rank_list, start=1):
                chunk_id = item.get("chunk_id")
                if not chunk_id:
                    continue
                # RRF 公式
                score_map[chunk_id] = score_map.get(chunk_id, 0.0) + weight / (k + pos)
                chunk_map.setdefault(chunk_id, item)

        # Step 5: 排序与截断
        merged = sorted(
            [(chunk_map[cid], score) for cid, score in score_map.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        return merged[:max_results] if max_results else merged

    # ================================================================== #
    #                      工具方法                                        #
    # ================================================================== #

    @staticmethod
    def _extract_entities(state_list, source: str = "") -> List[Dict[str, Any]]:
        """将上游节点输出统一规整为 entity 字典列表。

        三路数据字段映射：
        - embedding/hyde: pk → chunk_id
        - kg: source_id → chunk_id

        Args:
            state_list: 上游节点输出的文档列表
            source: 来源标记 "embedding" | "hyde" | "kg"

        Returns:
            统一 entity 列表，每项包含 chunk_id, content, item_name, title,
            file_title, parent_title, score, _source
        """
        out: List[Dict[str, Any]] = []
        for doc in (state_list or []):
            if not doc or not hasattr(doc, "get"):
                continue

            # 兼容 {"entity": {...}, "distance": ...} 包装
            raw = doc.get("entity") or doc

            # 统一 chunk_id 字段
            chunk_id = raw.get("chunk_id") or raw.get("pk") or raw.get("source_id") or ""

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
            out.append(entity)
        return out


# ================================================================== #
#                        兼容 & 测试                                   #
# ================================================================== #

_node_instance = RRF_node()


def node_rrf(state:QueryGraphState) -> QueryGraphState:
    """兼容原有调用方式的入口函数。"""
    return _node_instance(state)


if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging
    from knowledge.processor.query_process.state import create_default_state

    setup_logging()

    print("=" * 60)
    print("开始测试: RRF 融合节点 (RRF_node)")
    print("=" * 60)

    # 模拟三路检索结果（匹配真实上游输出格式）
    # embedding/hyde: pk 作为 chunk_id，无 entity 包装
    mock_state = create_default_state()
    mock_state["embedding_chunks"] = [
        {"pk": "chunk_1", "content": "向量搜索结果#1", "title": "## 标题1", "file_title": "测试文档", "parent_title": "## 父标题", "item_name": "测试商品", "score": 0.95},
        {"pk": "chunk_2", "content": "向量搜索结果#2", "title": "## 标题2", "file_title": "测试文档", "parent_title": "## 父标题", "item_name": "测试商品", "score": 0.90},
        {"pk": "chunk_3", "content": "向量搜索结果#3", "title": "## 标题3", "file_title": "测试文档", "parent_title": "## 父标题", "item_name": "测试商品", "score": 0.85},
    ]
    mock_state["hyde_embedding_chunks"] = [
        {"pk": "chunk_2", "content": "HyDE搜索结果#1", "title": "## HyDE标题", "file_title": "测试文档", "parent_title": "## 父标题", "item_name": "测试商品", "score": 0.88},
        {"pk": "chunk_1", "content": "HyDE搜索结果#2", "title": "## HyDE标题", "file_title": "测试文档", "parent_title": "## 父标题", "item_name": "测试商品", "score": 0.82},
        {"pk": "chunk_4", "content": "HyDE搜索结果#3", "title": "## HyDE标题", "file_title": "测试文档", "parent_title": "## 父标题", "item_name": "测试商品", "score": 0.75},
    ]
    # kg: source_id 作为 chunk_id，无 title/file_title/parent_title
    mock_state["kg_chunks"] = [
        {"source_id": "chunk_5", "content": "知识图谱结果#1", "item_name": "测试商品", "score": 4.0, "cnt": 3},
        {"source_id": "chunk_1", "content": "知识图谱结果#2", "item_name": "测试商品", "score": 2.0, "cnt": 1},
    ]

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print(f"  kg_chunks: {len(mock_state['kg_chunks'])} 条")
    print("-" * 60)

    # 执行 RRF 融合
    result = node_rrf(mock_state)

    # 打印结果
    print("\n【融合结果】:")
    for i, chunk in enumerate(result.get("rrf_chunks", []), 1):
        print(f"[{i}] {chunk.get('chunk_id')} - {chunk.get('content')}")

    print("-" * 60)
    print("测试完成")