"""查询流程节点基类

与导入流程的 BaseNode 类似，但：
- 日志命名空间是 query.{name}
- 异常用 QueryProcessError
- 任务追踪用 session_id + is_stream
- ★ 流式模式自动推送 SSE progress 事件给前端
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Optional
import logging

from knowledge.processor.query_process.config import QueryConfig, get_config
from knowledge.processor.query_process.exceptions import QueryProcessError

T = TypeVar("T")


class BaseNode(ABC):
    """查询流程节点基类

    所有查询节点继承此基类，实现 process 方法。
    基类统一处理日志、任务追踪、SSE 进度推送和异常。

    Example:
        class MyNode(BaseNode):
            name = "my_node"

            def process(self, state):
                # 业务逻辑
                return state
    """

    name: str = "base_node"

    def __init__(self, config: Optional[QueryConfig] = None):
        self.config = config or get_config()
        self.logger = logging.getLogger(f"query.{self.name}")

    def __call__(self, state: T) -> T:
        """节点执行入口（LangGraph 调用）。

        流程：标记 running → 推进度 → 执行业务 → 标记 done → 推进度
        """
        self.logger.info(f"--- {self.name} 开始 ---")

        session_id = state.get("session_id", "") if isinstance(state, dict) else ""
        is_stream = state.get("is_stream", False) if isinstance(state, dict) else False

        # ★ ① 节点开始 → 标记 running + 推送进度
        if session_id:
            try:
                from knowledge.tools.task_utils import add_running_task
                add_running_task(session_id, self.name)
                if is_stream:
                    self._push_progress(session_id)
            except Exception as e:
                self.logger.warning(f"任务追踪注册失败: {e}")

        try:
            result = self.process(state)
            self.logger.info(f"--- {self.name} 完成 ---")

            # ★ ② 节点完成 → 标记 done + 推送进度
            if session_id:
                try:
                    from knowledge.tools.task_utils import add_done_task
                    add_done_task(session_id, self.name)
                    if is_stream:
                        self._push_progress(session_id)
                except Exception as e:
                    self.logger.warning(f"任务完成标记失败: {e}")

            return result
        except QueryProcessError:
            raise
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}")
            raise QueryProcessError(
                message=str(e),
                node_name=self.name,
                cause=e
            )

    @abstractmethod
    def process(self, state: T) -> T:
        """节点核心处理逻辑（子类必须实现）"""
        pass

    def log_step(self, step_name: str, message: str = ""):
        """记录步骤日志"""
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)

    # ★ 新增：SSE 进度推送
    @staticmethod
    def _push_progress(session_id: str):
        """推送当前任务进度到 SSE 流。

        前端收到后渲染进度步骤列表（done_list + running_list）。
        每次推送全量快照，即使中间某条丢了，下一条也能正确覆盖。
        """
        try:
            from knowledge.tools.task_utils import task_push_queue
            task_push_queue(session_id)
        except Exception:
            pass


def setup_logging(level: int = logging.INFO):
    """配置查询流程日志"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
