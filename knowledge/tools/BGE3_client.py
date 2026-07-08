import os
import threading
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

_bge_ef = None
_lock = threading.Lock()


def get_bgem3_client():
    """获取 BGE-M3 客户端（线程安全，模块级缓存）。

    模型路径优先级：环境变量 BGE_MODEL_PATH > 默认 "BAAI/bge-m3"
    """
    global _bge_ef
    if _bge_ef is not None:
        return _bge_ef
    with _lock:
        if _bge_ef is not None:
            return _bge_ef
        model_path = os.getenv("BGE_MODEL_PATH", "BAAI/bge-m3")
        device = os.getenv("BGE_DEVICE", "cuda:0")
        _bge_ef = BGEM3EmbeddingFunction(
            model_name=model_path,
            device=device,
        )
        return _bge_ef


if __name__ == '__main__':
    bge_ef = get_bgem3_client()
    query = "如何使用Milvus"
    documents = ["Milvus是一个开源的向量数据库系统", "Milvus是一个开源的向量数据库系统", "Milvus是一个开源的向量数据库系统"]

    results = bge_ef.encode_documents(documents)
    print(results["dense"][0].tolist())
