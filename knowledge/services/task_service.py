"""
任务状态服务

封装 task_utils 的操作，提供给路由层调用。
"""
from knowledge.tools.task_utils import get_task_info, update_task_status, init_task


class TaskService:
    def get_task_info(self, task_id: str) -> dict:
        """获取任务状态（返回给前端）"""
        return get_task_info(task_id)

    def init_task(self, task_id: str):
        """初始化任务"""
        init_task(task_id)

    def complete_task(self, task_id: str):
        """标记任务完成"""
        update_task_status(task_id, "completed")

    def fail_task(self, task_id: str):
        """标记任务失败"""
        update_task_status(task_id, "failed")
