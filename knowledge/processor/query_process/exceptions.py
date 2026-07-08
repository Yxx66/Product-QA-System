"""查询流程自定义异常类

异常层级：
QueryProcessError (基础异常)
├── ConfigurationError      # 配置错误
├── SearchError             # 搜索错误
├── EmbeddingError          # 向量化错误
├── LLMError                # LLM 调用错误
├── StorageError            # 存储错误
│   ├── MilvusError
│   ├── Neo4jError
│   └── MongoDBError
├── ValidationError         # 数据验证错误
├── EntityAlignmentError    # 实体对齐错误
├── RerankError             # 重排序错误
└── ItemNameConfirmError    # 商品名确认错误
"""


class QueryProcessError(Exception):
    """查询流程基础异常"""

    def __init__(self, message: str, node_name: str = "", cause: Exception = None):
        self.node_name = node_name
        self.cause = cause
        super().__init__(message)

    def __str__(self):
        parts = []
        if self.node_name:
            parts.append(f"[{self.node_name}]")
        parts.append(super().__str__())
        if self.cause:
            parts.append(f"(原因: {self.cause})")
        return " ".join(parts)


class ConfigurationError(QueryProcessError):
    """配置错误"""
    pass


class SearchError(QueryProcessError):
    """搜索错误"""
    pass


class EmbeddingError(QueryProcessError):
    """向量化错误"""
    pass


class LLMError(QueryProcessError):
    """LLM 调用错误"""
    pass


class StorageError(QueryProcessError):
    """存储错误"""
    pass


class MilvusError(StorageError):
    """Milvus 存储错误"""
    pass


class Neo4jError(StorageError):
    """Neo4j 存储错误"""
    pass


class MongoDBError(StorageError):
    """MongoDB 存储错误"""
    pass


class ValidationError(QueryProcessError):
    """数据验证错误"""
    pass


class EntityAlignmentError(QueryProcessError):
    """实体对齐错误"""
    pass


class RerankError(QueryProcessError):
    """重排序错误"""
    pass


class ItemNameConfirmError(QueryProcessError):
    """商品名称确认错误"""
    pass
