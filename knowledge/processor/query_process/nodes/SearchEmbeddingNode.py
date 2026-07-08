from typing import List
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from pymilvus import AnnSearchRequest, Function, FunctionType


class SearchEmbeddingNode(BaseNode):
    """向量检索节点：对 query 做 dense + sparse 混合检索"""
    name = "search_embedding"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        self.log_step("向量检索", "开始")

        # 1. 获取查询文本（优先 rewritten_query，兜底 original_query）
        query = state.get("rewritten_query") or state.get("original_query", "")
        if not query:
            self.logger.warning("query 为空，跳过检索")
            return {"embedding_chunks": []}

        # 2. 混合检索
        results = self._hybrid_search(query)

        # 3. 只返回本节点的 chunk 字段，避免并行冲突
        self.log_step("向量检索完成", f"命中 {len(results)} 条")
        return {"embedding_chunks": results}

    def _hybrid_search(self, query: str, top_k: int = 10) -> List[dict]:
        """dense + sparse + RRF 混合检索"""
        from knowledge.tools.BGE3_client import get_bgem3_client
        from knowledge.tools.milvus_client import get_milvus_client

        bge3 = get_bgem3_client()
        milvus_client = get_milvus_client()
        if not bge3 or not milvus_client:
            return []

        # 1. 编码 query
        embeddings = bge3.encode_documents([query])
        dense_vector = embeddings["dense"][0].tolist()

        sparse_matrix = embeddings["sparse"]
        start = sparse_matrix.indptr[0]
        end = sparse_matrix.indptr[1]
        sparse_indices = sparse_matrix.indices[start:end].tolist()
        sparse_values = sparse_matrix.data[start:end].tolist()
        sparse_vector = dict(zip(sparse_indices, sparse_values))

        # 2. 构建请求
        request_dense = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=top_k,
        )
        request_sparse = AnnSearchRequest(
            data=[sparse_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=top_k,
        )

        # 3. RRF 融合
        ranker = Function(
            name="rrf",
            input_field_names=[],
            function_type=FunctionType.RERANK,
            params={"reranker": "rrf", "k": 60},
        )

        # 4. 执行检索
        results = milvus_client.hybrid_search(
            collection_name="knowledge_chunks",
            reqs=[request_dense, request_sparse],
            ranker=ranker,
            limit=top_k,
            output_fields=["pk", "title", "content", "file_title", "parent_title", "item_name"],
        )

        # 5. 格式化结果
        output = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                output.append({
                    "pk": entity.get("pk", ""),
                    "title": entity.get("title", ""),
                    "content": entity.get("content", ""),
                    "file_title": entity.get("file_title", ""),
                    "parent_title": entity.get("parent_title", ""),
                    "item_name": entity.get("item_name", ""),
                    "score": round(hit.get("distance", 0.0), 4),
                })

        return output


if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging
    from knowledge.processor.query_process.state import create_default_state
    setup_logging()

    node = SearchEmbeddingNode()

    test_state = create_default_state()
    test_state["original_query"] = "万用表怎么测电压"
    test_state["rewritten_query"] = ""

    result = node.process(test_state)
    # print(result["embedding_chunks"][0].get("content"))
    print(test_state.get("embedding_chunks", []))
    # print(f"\n检索结果: {len(result.get('embedding_chunks', []))} 条")
    # for i, r in enumerate(result.get("embedding_chunks", [])[:5]):
    #     print(f"  {i+1}. [{r['score']}] {r['title']} - {r['content']}...")
