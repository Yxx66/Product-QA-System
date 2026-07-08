"""知识库查询 API 路由

端点一览：
  POST   /query              — 提交查询（非流式同步返回 / 流式后台执行）
  GET    /stream/{session_id} — SSE 流式实时推送
  GET    /history/{session_id} — 获取历史对话
  DELETE /history/{session_id} — 清空历史对话
  GET    /sessions           — 列出所有历史会话
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Depends

from fastapi.responses import StreamingResponse

from knowledge.schema.query_schema import QueryRequest, QueryResponse, StreamSubmitResponse
from knowledge.services.query_service import QueryService
from knowledge.core.deps import get_query_service
from knowledge.tools.sse_utils import sse_generator

router = APIRouter()


# ==================== 查询入口 ====================

@router.post("/query")
async def query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    service: QueryService = Depends(get_query_service),
):
    """处理查询请求。

    两种模式：
    - 非流式 (is_stream=False)：同步执行，等待完成，直接返回答案
    - 流式   (is_stream=True)： 后台异步执行，立即返回 session_id，
                                前端再调用 GET /stream/{session_id} 建立 SSE
    """
    session_id = request.session_id or service.generate_session_id()
    service.submit_query(session_id, request.is_stream)

    if request.is_stream:
        # 流式：放到后台线程执行，不阻塞请求
        background_tasks.add_task(
            service.run_query_graph, session_id, request.query, True,
        )
        return StreamSubmitResponse(
            message="Query submitted",
            session_id=session_id,
        )
    else:
        # 非流式：同步等待
        service.run_query_graph(session_id, request.query, False)
        answer = service.get_answer(session_id)
        return QueryResponse(
            message="处理完成",
            session_id=session_id,
            answer=answer,
        )


# ==================== SSE 流式端点 ====================

@router.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    """SSE 实时流式推送。

    前端建立 EventSource 后，后端持续推送：
    - ready:     连接建立
    - progress:  节点进度更新
    - delta:     LLM 逐字输出
    - final:     最终完整答案
    """
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
    )


# ==================== 历史记录端点 ====================

@router.get("/history/{session_id}")
async def get_history(
    session_id: str,
    limit: int = 50,
    service: QueryService = Depends(get_query_service),
):
    """获取指定会话的历史对话记录"""
    try:
        items = service.get_history(session_id, limit)
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


@router.delete("/history/{session_id}")
async def clear_chat_history(
    session_id: str,
    service: QueryService = Depends(get_query_service),
):
    """清空指定会话的全部历史记录"""
    count = service.clear_history(session_id)
    return {"message": "History cleared", "deleted_count": count}


# ==================== 会话列表端点 ====================

@router.get("/sessions")
async def list_sessions(
    limit: int = 100,
    service: QueryService = Depends(get_query_service),
):
    """列出所有历史会话（按最近活跃时间倒序）。

    返回每个会话的 ID、标题预览、消息条数、时间等。
    """
    try:
        sessions = service.list_sessions(limit)
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"list sessions error: {e}")
