"""查询业务服务

封装查询流程的完整生命周期：
- submit_query:  提交查询（建状态 + 建 SSE 队列）
- run_query_graph: 在后台线程执行 LangGraph 流水线
- get_answer:    获取非流式模式的结果
- get_history / clear_history: 历史记录管理
"""
import uuid
import logging
from typing import List, Dict, Any

from knowledge.processor.query_process.main_graph import query_app
from knowledge.tools.task_utils import (
    update_task_status, get_task_result, init_task,
    TASK_STATUS_COMPLETED,
)
from knowledge.tools.sse_utils import create_sse_queue, push_to_session

logger = logging.getLogger(__name__)


class QueryService:
    """查询业务服务 —— 与导入侧 FileImportService 对称"""

    # ==================== ID 生成 ====================

    @staticmethod
    def generate_session_id() -> str:
        """生成会话 ID"""
        return str(uuid.uuid4())

    # ==================== 提交 + 执行 ====================

    def submit_query(self, session_id: str, is_stream: bool):
        """提交查询任务：重置状态 + 流式模式创建 SSE 队列。

        队列必须在 run_query_graph（后台线程）执行前建好，
        否则后台线程 push 时队列还不存在，消息会丢失。
        """
        init_task(session_id)  # 重置 done/running 列表，避免多次 query 累积
        if is_stream:
            create_sse_queue(session_id)

    def run_query_graph(
        self,
        session_id: str,
        user_query: str,
        is_stream: bool,
    ):
        """执行 LangGraph 查询流程（在后台线程中运行）。"""
        try:
            default_state = {
                "original_query": user_query,
                "session_id": session_id,
                "is_stream": is_stream,
            }
            query_app.invoke(default_state)
        except Exception as e:
            logger.error(f"查询流程执行失败: {e}", exc_info=True)
        finally:
            update_task_status(session_id, TASK_STATUS_COMPLETED)
            # ★ 兜底机制：即使 FINAL 事件丢了，progress(completed) 也能让前端恢复
            if is_stream:
                try:
                    from knowledge.tools.task_utils import task_push_queue
                    task_push_queue(session_id)
                except Exception:
                    pass

    # ==================== 结果获取 ====================

    @staticmethod
    def get_answer(session_id: str) -> str:
        """获取非流式模式的答案"""
        return get_task_result(session_id, "answer", "")

    # ==================== 历史记录 ====================

    @staticmethod
    def get_history(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """获取指定会话的历史记录"""
        from knowledge.utils.mongo_history_utils import get_recent_messages

        records = get_recent_messages(session_id, limit=limit)
        return [
            {
                "_id": str(r.get("_id", "")),
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts"),
            }
            for r in records
        ]

    @staticmethod
    def clear_history(session_id: str) -> int:
        """清空指定会话的历史记录"""
        from knowledge.utils.mongo_history_utils import clear_history
        return clear_history(session_id)

    @staticmethod
    def list_sessions(limit: int = 100) -> List[Dict[str, Any]]:
        """列出所有历史会话"""
        from knowledge.utils.mongo_history_utils import list_sessions
        return list_sessions(limit)
