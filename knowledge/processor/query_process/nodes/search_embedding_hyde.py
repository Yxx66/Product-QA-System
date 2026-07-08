from typing import List
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from pymilvus import AnnSearchRequest, Function, FunctionType


class search_embedding_hyde(BaseNode):
    """
    假设性查询节点（HyDE）：
    1. 调用 LLM 生成假设性回答
    2. 对假设性回答做向量检索
    3. 将检索结果写入 state
    """
    name = "search_embedding_hyde"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        self.log_step("HyDE检索", "开始")

        # 1. 参数校验
        item_name = state.get("item_names", [])
        rewritten_query = state.get("rewritten_query", "") or state.get("original_query", "")
        item_name, rewritten_query = self._validate(item_name, rewritten_query)
        if not item_name or not rewritten_query:
            return {"hyde_embedding_chunks": []}

        # 2. 调用 LLM 生成假设性回答
        hyde_text = self._generate_query(rewritten_query)
        if not hyde_text:
            return {"hyde_embedding_chunks": []}

        # 3. 对假设性回答做向量检索
        search_results = self._search_embedding(hyde_text)

        # 4. 只返回本节点的 chunk 字段
        self.log_step("HyDE检索完成", f"命中 {len(search_results)} 条")
        return {"hyde_embedding_chunks": search_results}

    def _validate(self, item_name, rewritten_query):
        """参数校验"""
        if not item_name:
            self.logger.warning("item_name 为空")
        if not rewritten_query:
            self.logger.warning("rewritten_query 为空")
        return item_name, rewritten_query

    def _generate_query(self, query: str) -> str:
        """调用 LLM 生成假设性回答"""
        from knowledge.tools.llm_client import get_llm_client

        llm = get_llm_client()
        if not llm:
            return ""

        system_prompt = "你是一个技术文档助手，请根据问题生成一个假设性的技术回答。"
        prompt = f"""请根据以下问题，生成一个假设性的技术回答（不需要准确，只需要在语义上与相关文档相似）。

问题：{query}

要求：
1. 用技术文档的风格写
2. 包含相关的技术术语
3. 长度控制在 200 字以内
4. 只返回回答内容，不要添加解释"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            response = llm.chat.completions.create(
                model="qwen-plus",
                messages=messages,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"LLM请求失败: {e}")
            return ""

    def _search_embedding(self, query: str, top_k: int = 5) -> List[dict]:
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

    node = search_embedding_hyde()

    print("=" * 50)
    print("测试1: HyDE 生成 + 检索")
    print("=" * 50)

    test_state = create_default_state()
    test_state["original_query"] = "万用表怎么测电压"
    test_state["rewritten_query"] = ""
    test_state["item_names"] = ["万用表"]

    result = node.process(test_state)
    hyde_results = result.get("hyde_embedding_chunks", [])
    print(f"检索结果: {len(hyde_results)} 条")
    for i, r in enumerate(hyde_results[:5]):
        print(f"  {i+1}. [{r['score']}] {r['title']} - {r['content'][:60]}...")

    print("\n" + "=" * 50)
    print("测试2: 空 item_name 跳过")
    print("=" * 50)
    test_state2 = create_default_state()
    test_state2["original_query"] = "测试"
    test_state2["item_names"] = []
    result2 = node.process(test_state2)
    print(f"结果: {result2.get('hyde_embedding_chunks')}")
    print(test_state.get("hyde_embedding_chunks"))
