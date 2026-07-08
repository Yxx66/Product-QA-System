import os
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

_llm_client = None


def get_llm_client():
    """获取 LLM 客户端（模块级缓存，整个进程只创建一次）。

    优先使用环境变量 OPENAI_API_KEY 和 OPENAI_API_BASE，
    兼容旧环境变量 DASHSCOPE_API_KEY。
    """
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv(
        "OPENAI_API_BASE",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    if not api_key:
        raise ValueError(
            "未设置 LLM API Key，请在 .env 中配置 OPENAI_API_KEY"
        )

    try:
        _llm_client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"LLM 客户端初始化成功 (base_url={base_url})")
        return _llm_client
    except Exception as e:
        raise Exception(f"获取 LLM 客户端失败: {e}")