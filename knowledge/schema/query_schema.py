"""查询接口的请求/响应模型"""
from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """查询请求——前端发什么字段过来"""
    query: str = Field(..., description="用户查询内容")
    session_id: Optional[str] = Field(None, description="会话ID，不传则自动生成")
    is_stream: bool = Field(False, description="是否流式返回")


class QueryResponse(BaseModel):
    """非流式响应——POST /query 直接返回答案"""
    message: str = Field(..., description="状态消息")
    session_id: str = Field(..., description="会话ID")
    answer: str = Field("", description="生成的答案")


class StreamSubmitResponse(BaseModel):
    """流式提交响应——返回 session_id 让前端建 SSE 连接"""
    message: str = Field(..., description="状态消息")
    session_id: str = Field(..., description="会话ID，前端用此 ID 建立 SSE 连接")
