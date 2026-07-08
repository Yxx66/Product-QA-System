from typing import List
import asyncio
from openai import AsyncOpenAI

from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp
from agents.model_settings import ModelSettings
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class network_search_node(BaseNode):
    name = "network_search"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        self.log_step("网络检索", "开始")

        # 1. 参数校验
        item_name = state.get("item_names", [])
        rewritten_query = state.get("rewritten_query", "") or state.get("original_query", "")
        item_name, rewritten_query = self._validate(item_name, rewritten_query)
        if not item_name or not rewritten_query:
            self.logger.warning("item_name 或 rewritten_query 为空，跳过网络搜索")
            return {"web_search_docs": []}

        # 2. 网络搜索
        network_search_results = self._network_search(item_name, rewritten_query)
        self.log_step("网络检索完成", f"命中 {len(network_search_results)} 条")
        return {"web_search_docs": network_search_results}

    def _validate(self, item_name, rewritten_query):
        """参数校验"""
        if not item_name:
            self.logger.warning("item_name 为空")
        if not rewritten_query:
            self.logger.warning("rewritten_query 为空")
        return item_name, rewritten_query

    def _network_search(self, item_name, rewritten_query):
        """网络搜索（同步入口，内部包装异步 MCP 调用）"""
        try:
            return asyncio.run(self._async_network_search(item_name, rewritten_query))
        except Exception as e:
            self.logger.error(f"网络搜索失败: {e}")
            return []

    async def _async_network_search(self, item_name, rewritten_query):
        """异步 MCP 网络搜索核心逻辑"""
        # 拼接查询
        names = item_name if isinstance(item_name, list) else [item_name]
        query = f"关于 {'、'.join(names)} 的问题: {rewritten_query}"

        token = self.config.openai_api_key
        base_url = self.config.mcp_dashscope_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model_name = self.config.default_model or "qwen-plus"

        if not token:
            self.logger.error("OPENAI_API_KEY 未配置，跳过网络搜索")
            return []

        external_client = AsyncOpenAI(
            api_key=token,
            base_url=base_url,
        )

        async with MCPServerStreamableHttp(
            name="DashScope WebSearch",
            params={
                "url": "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp",
                "headers": {"Authorization": f"Bearer {token}"},
                "timeout": 10,
            },
            cache_tools_list=True,
            max_retry_attempts=3,
        ) as server:
            model = OpenAIChatCompletionsModel(
                model=model_name,
                openai_client=external_client,
            )

            agent = Agent(
                name="Assistant",
                model=model,
                instructions="Use the MCP tools to answer the questions. When calling search, pass count=5.",
                mcp_servers=[server],
                model_settings=ModelSettings(tool_choice="required"),
            )

            result = await Runner.run(agent, query)
            self.logger.info(f"网络搜索完成，结果长度: {len(result.final_output)}")

            return [{"content": result.final_output}]


# ==================== 测试入口 ====================
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from knowledge.processor.query_process.config import get_config, reset_config
    from knowledge.processor.query_process.state import create_default_state

    reset_config()
    config = get_config()

    # 构造测试 state
    state = create_default_state(
        item_names=["万用表"],
        original_query="万用表怎么测电压",
        rewritten_query="万用表怎么测电压",
    )

    node = network_search_node(config=config)
    result_state = node(state)

    print("\n========== 网络搜索结果 ==========")
    for i, chunk in enumerate(result_state.get("web_search_docs", []), 1):
        content = chunk.get("content", "")
        print(f"\n--- 结果 {i} ---")
        print(content)
    print(f"\n共命中 {len(result_state.get('web_search_docs', []))} 条")