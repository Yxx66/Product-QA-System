"""任务状态接口的响应模型"""
from pydantic import BaseModel
from typing import List


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str
    status: str  # processing / completed / failed
    done_list: List[str]
    running_list: List[str]
