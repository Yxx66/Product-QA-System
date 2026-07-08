import os
import logging
from typing import List, Dict, Any
from pymilvus import MilvusClient, AnnSearchRequest, Function, FunctionType

logger = logging.getLogger(__name__)

COLLECTION_NAME = os.getenv(
    "ITEM_NAME_COLLECTION", "knowledge_item_names"
)


def get_milvus_client():
    milvus_url = os.getenv("MILVUS_URL", "http://localhost:19530")
    client = MilvusClient(uri=milvus_url)
    return client


def _create_collection(client: MilvusClient, collection_name: str, dimension: int = 1024):
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        dimension=dimension,
    )


def _encode_query(query: str) -> tuple:
    """将查询文本编码为 dense 和 sparse 向量。"""
    from knowledge.tools.BGE3_client import get_bgem3_client

    bge_ef = get_bgem3_client()
    query_vectors = bge_ef.encode_documents([query])

    # 提取稠密向量
    dense_vector = query_vectors["dense"][0].tolist()

    # 提取稀疏向量
    sparse_matrix = query_vectors["sparse"]
    start = sparse_matrix.indptr[0]
    end = sparse_matrix.indptr[1]
    sparse_indices = sparse_matrix.indices[start:end].tolist()
    sparse_values = sparse_matrix.data[start:end].tolist()
    sparse_vector = dict(zip(sparse_indices, sparse_values))

    return dense_vector, sparse_vector


def search_item_name(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    通过混合检索（dense + sparse + RRF）搜索最相关的 item_name。

    Args:
        query: 查询文本
        top_k: 返回结果数量

    Returns:
        包含 item_name, file_title, distance 的结果列表
    """
    client = get_milvus_client()

    # 1. 编码查询
    dense_vector, sparse_vector = _encode_query(query)

    # 2. 构建搜索请求
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

    # 3. 构建 RRF 排序器
    ranker = Function(
        name="rrf",
        input_field_names=[],
        function_type=FunctionType.RERANK,
        params={"reranker": "rrf", "k": 60},
    )

    # 4. 执行混合检索
    results = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[request_dense, request_sparse],
        ranker=ranker,
        limit=top_k,
        output_fields=["file_title", "item_name"],
    )

    # 5. 格式化结果
    output = []
    for hits in results:
        for hit in hits:
            entity = hit.get("entity", {})
            output.append({
                "item_name": entity.get("item_name", ""),
                "file_title": entity.get("file_title", ""),
                "distance": hit.get("distance", 0.0),
            })

    return output


if __name__ == '__main__':
    from knowledge.processor import setup_logging
    setup_logging()

    # 测试连接
    milvus_client = get_milvus_client()
    print(f"已有集合: {milvus_client.list_collections()}")

    # 测试检索
    print("\n" + "=" * 50)
    print("测试: 混合检索 item_name")
    print("=" * 50)
    results = search_item_name("苏伯尔电磁炉")
    for r in results:
        print(f"  item_name: {r['item_name']}, distance: {r['distance']:.4f}")
