"""上传接口的响应模型"""
from pydantic import BaseModel


class UploadResponse(BaseModel):
    """上传成功后返回 task_id"""
    message: str
    task_id: str
