"""查询流程配置管理模块

集中管理所有配置项，支持环境变量覆盖。
"""
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class QueryConfig:
    """查询流程配置"""

    # ==================== 文本处理配置 ====================
    max_context_chars: int = 12000

    # ==================== Rerank 配置 ====================
    rerank_max_topk: int = 10
    rerank_min_topk: int = 3
    rerank_gap_ratio: float = 0.25
    rerank_gap_abs: float = 0.5

    # ==================== RRF 配置 ====================
    rrf_k: int = 60
    rrf_kg_weight: float = field(
        default_factory=lambda: float(os.getenv("RRF_KG_WEIGHT", "0.7"))
    )
    rrf_max_results: int = 10

    # ==================== 检索配置 ====================
    embedding_search_limit: int = 10
    hyde_search_limit: int = 5

    # ==================== 知识图谱配置 ====================
    kg_entity_align_min_score: Optional[float] = field(
        default_factory=lambda: (
            float(os.getenv("KG_ENTITY_ALIGN_MIN_SCORE"))
            if os.getenv("KG_ENTITY_ALIGN_MIN_SCORE")
            else None
        )
    )
    kg_max_seed_candidates: int = 3
    kg_max_total_seeds: int = 30
    kg_max_triples_per_seed: int = 50
    kg_max_total_triples: int = 200
    kg_max_total_chunks: int = 200

    # ==================== LLM 配置 ====================
    openai_api_base: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("MODEL", "")
    )
    item_model: str = field(
        default_factory=lambda: os.getenv("ITEM_MODEL", "")
    )

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )
    chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
    )
    item_name_collection: str = "item_name_collection_test"
    entity_name_collection: str = field(
        default_factory=lambda: os.getenv("ENTITY_NAME_COLLECTION", "")
    )

    # ==================== Neo4j 配置 ====================
    neo4j_uri: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "")
    )
    neo4j_username: str = field(
        default_factory=lambda: os.getenv("NEO4J_USERNAME", "")
    )
    neo4j_password: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "")
    )
    neo4j_database: str = field(
        default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j")
    )

    # ==================== MCP 配置 ====================
    mcp_dashscope_base_url: str = field(
        default_factory=lambda: os.getenv("MCP_DASHSCOPE_BASE_URL", "")
    )
    mcp_dashscope_api_key: str = field(
        default_factory=lambda: os.getenv("MCP_DASHSCOPE_API_KEY", "")
    )

    @classmethod
    def from_env(cls) -> "QueryConfig":
        """从环境变量加载配置"""
        return cls()

    def validate(self, strict: bool = False) -> None:
        """验证配置是否完整"""
        required_fields = ["milvus_url", "chunks_collection"]
        missing = [k for k in required_fields if not getattr(self, k)]
        if missing:
            msg = f"缺少必要配置: {missing}"
            if strict:
                raise ValueError(msg)
            else:
                print(f"警告: {msg}")


# ==================== 全局单例 ====================
_config: Optional[QueryConfig] = None


def get_config() -> QueryConfig:
    """获取配置单例"""
    global _config
    if _config is None:
        _config = QueryConfig.from_env()
    return _config


def reset_config() -> None:
    """重置配置（用于测试）"""
    global _config
    _config = None
