"""
答案生成输出节点

作为查询流程的最终节点，根据检索融合/重排序后的结果（reranked_docs）
和用户查询，调用 LLM 生成最终答案。

两种入口路径：
1. item_name_confirm 无法确认商品 → 直接 preset answer → 跳过本节点的 LLM 调用
2. 完整检索流水线 → rrf → rerank → 本节点 LLM 生成答案
"""

from typing import List, Dict

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.config import get_config

# ★ 正式导入：SSE 推送 + 任务结果 + MongoDB 历史写入
from knowledge.tools.sse_utils import push_to_session, SSEEvent
from knowledge.tools.task_utils import set_task_result
from knowledge.utils.mongo_history_utils import save_chat_message



# ==================== Prompt 模板 ====================

ANSWER_SYSTEM_PROMPT = """你是一个专业的知识库助手，请根据以下信息回答用户问题。

要求：
1. 严格基于【参考内容】作答，不要编造不存在的事实
2. 如果参考内容不足，明确说明"根据现有资料，未找到相关信息"
3. 回答简洁清晰，使用自然的口语化表达，不要用固定的列表格式
4. 如果用户的问题涉及图片且资料中有图片URL，可以引用

【参考内容】
{context}

【历史对话】
{history}

【相关商品/实体】
{item_names}

【图谱关系描述】
{graph_relation_description}

【用户问题】
{question}

请回答："""


# ==================== 节点类 ====================

class AnswerOutputNode(BaseNode):
    """答案生成输出节点。

    流程：
        1. 检查 state["answer"] 是否已由上游预置
        2. 收集检索结果（reranked_docs / kg_triples）
        3. 构建 LLM prompt（含文档、历史、图谱 + 字符预算控制）
        4. LLM 生成答案（流式 / 非流式）
        5. 写入历史记录 + 推送结束事件
        6. 返回 state
    """

    name = "answer_output"

    # ================================================================== #
    #                           主流程                                    #
    # ================================================================== #

    def process(self, state: QueryGraphState) -> QueryGraphState:
        session_id = state.get("session_id")
        is_stream = state.get("is_stream")

        # Step 1: 已有答案（如商品确认提示）→ 直接推送
        if state.get("answer"):
            self._push_existing_answer(state)
        else:
            # Step 2-4: 构建提示词 → 生成答案
            prompt = self._build_prompt(state)
            state["prompt"] = prompt
            self._generate(state, prompt)

        # Step 5: 写入历史
        if state.get("answer"):
            self._write_history(state)

        # Step 6: 流式模式发送结束事件
        if is_stream:
            push_to_session(
                session_id,
                SSEEvent.FINAL,
                {"answer": state.get("answer", ""), "status": "completed"},
            )

        return state

    # ================================================================== #
    #                    已有答案推送                                       #
    # ================================================================== #

    def _push_existing_answer(self, state: QueryGraphState):
        """将已有答案推送到流或任务结果。"""
        answer = state["answer"]
        if state.get("is_stream"):
            push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})
        else:
            set_task_result(state["session_id"], "answer", answer)

    # ================================================================== #
    #                    提示词构建                                         #
    # ================================================================== #

    def _build_prompt(self, state: QueryGraphState) -> str:
        """根据检索结果、历史对话、图谱关系组装 LLM 提示词。

        每块信息源都有字符预算控制，防止超出 LLM 上下文窗口。
        """
        config = get_config()
        budget = config.max_context_chars

        question = state.get("rewritten_query") or state.get("original_query", "")
        item_names = state.get("item_names") or []

        # Step 3: 格式化检索文档
        context_str, budget = self._format_docs(
            state.get("reranked_docs") or [], budget,
        )

        # Step 4: 格式化历史对话
        history_str, budget = self._format_history(
            state.get("history") or [], budget,
        )

        # Step 5: 格式化图谱关系
        graph_str, budget = self._format_triples(
            state.get("kg_triples") or [], budget,
        )

        # Step 6: 组装完整提示词
        return ANSWER_SYSTEM_PROMPT.format(
            context=context_str or "无参考内容",
            history=history_str or "无历史对话",
            item_names=", ".join(item_names) if item_names else "无指定商品",
            graph_relation_description=graph_str or "无图谱关系",
            question=question,
        )

    # ================================================================== #
    #                    格式化工具 (字符预算控制)                           #
    # ================================================================== #

    def _format_docs(self, docs: List[Dict], budget: int) -> tuple:
        """格式化重排序文档，带字符预算控制。"""
        lines = []
        used = 0

        for i, doc in enumerate(docs, 1):
            text = (doc.get("content") or "").strip()
            if not text:
                continue

            meta = [f"[{i}]"]
            for key, fmt in [
                ("_source", "[source={}]"),
                ("chunk_id", "[chunk_id={}]"),
                ("title", "[title={}]"),
            ]:
                val = str(doc.get(key) or "").strip()
                if val:
                    meta.append(fmt.format(val))

            score = doc.get("score")
            if score is not None:
                meta.append(f"[score={float(score):.4f}]")

            doc_str = " ".join(meta) + "\n" + text
            if used + len(doc_str) > budget:
                break
            lines.append(doc_str)
            used += len(doc_str) + 2

        return "\n\n".join(lines), budget - used

    @staticmethod
    def _format_history(history: List[Dict], budget: int) -> tuple:
        """格式化历史对话。"""
        lines = []
        used = 0

        for msg in history:
            for role, key in [("用户", "user"), ("助手", "assistant")]:
                text = msg.get(key)
                if not text:
                    continue
                line = f"{role}: {text}"
                used += len(line) + 1
                if used > budget:
                    return "\n".join(lines), budget - used
                lines.append(line)

        return "\n".join(lines), budget - used

    @staticmethod
    def _format_triples(triples: List, budget: int) -> tuple:
        """格式化图谱三元组。"""
        lines = []
        used = 0

        for tr in triples:
            line = (str(tr) if tr is not None else "").strip()
            if not line:
                continue
            if used + len(line) > budget:
                break
            lines.append(line)
            used += len(line) + 1

        return "\n".join(lines), budget - used

    # ================================================================== #
    #                    LLM 生成                                          #
    # ================================================================== #

    def _generate(self, state: QueryGraphState, prompt: str):
        """调用 LLM 生成答案（流式/非流式）。"""
        self.log_step("generate", "生成答案")
        from knowledge.tools.llm_client import get_llm_client

        llm = get_llm_client()
        session_id = state.get("session_id")

        if state.get("is_stream"):
            state["answer"] = self._stream_generate(llm, prompt, session_id)
        else:
            state["answer"] = self._invoke_generate(llm, prompt, session_id)

    def _stream_generate(self, llm, prompt: str, session_id: str) -> str:
        """流式生成，逐 chunk 推送。"""
        result = ""
        try:
            response = llm.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.1,
                stream=True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    result += delta
                    push_to_session(session_id, "delta", {"delta": delta})
        except Exception as e:
            self.logger.error(f"流式生成出错: {e}")
        return result

    def _invoke_generate(self, llm, prompt: str, session_id: str) -> str:
        """非流式生成。"""
        try:
            response = llm.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.1,
            )
            answer = response.choices[0].message.content
            set_task_result(session_id, "answer", answer)
            return answer
        except Exception as e:
            self.logger.error(f"生成回答出错: {e}")
            return "抱歉，生成回答时出现错误。"

    # ================================================================== #
    #                    历史记录                                           #
    # ================================================================== #

    def _write_history(self, state: QueryGraphState):
        """将用户问题 + 助手回答一并写入 MongoDB 历史记录。

        按照文档设计的职责分离原则：
        - item_name_confirm_node 只读 MongoDB（查历史 + 回填）
        - answer_output 作为 pipeline 最终出口，统一写入新记录
        """
        session_id = state.get("session_id", "default")
        rewritten_query = state.get("rewritten_query", "") or state.get("original_query", "")
        item_names = state.get("item_names") or []

        try:
            # 写入用户问题
            user_id = save_chat_message(
                session_id=session_id,
                role="user",
                text=state.get("original_query", ""),
                rewritten_query=rewritten_query,
                item_names=item_names,
            )
            self.logger.info(f"[历史写入] 用户消息 → _id={user_id}")

            # 写入助手回答
            answer = (state.get("answer") or "").strip()
            if answer:
                asst_id = save_chat_message(
                    session_id=session_id,
                    role="assistant",
                    text=answer,
                    rewritten_query=rewritten_query,
                    item_names=item_names,
                )
                self.logger.info(f"[历史写入] 助手回答 → _id={asst_id}")
        except Exception as e:
            self.logger.warning(f"写入历史记录失败: {e}")


# ==================== 模块级便捷函数 ====================

_node_instance = AnswerOutputNode()


def node_answer_output(state: QueryGraphState) -> QueryGraphState:
    """兼容 LangGraph 直接注册调用的入口函数。"""
    return _node_instance(state)
