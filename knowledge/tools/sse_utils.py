"""
SSE（Server-Sent Events）工具模块

核心机制：生产者-消费者模型
- 后台线程 push → queue.Queue ← sse_generator pull → yield 给前端

使用方式：
    # 生产者（LangGraph 节点内）
    push_to_session(session_id, "delta", {"delta": "内容"})

    # 消费者（FastAPI 路由）
    return StreamingResponse(sse_generator(session_id, request), media_type="text/event-stream")
"""

import json
import queue
import asyncio
from typing import Dict, Any, AsyncGenerator

from fastapi import Request


class SSEEvent:
    """SSE 事件类型常量"""
    READY = "ready"           # 连接建立
    DELTA = "delta"           # LLM 流式输出增量
    PROGRESS = "progress"     # 任务进度更新
    FINAL = "final"           # 最终完整答案
    ERROR = "error"           # 错误信息
    CLOSE = "__close__"       # 关闭连接（内部信号）


# ==================== 全局会话队列存储 ====================

_session_stream: Dict[str, queue.Queue] = {}


def create_sse_queue(session_id: str) -> queue.Queue:
    """创建并注册一个新的 SSE 队列（后台任务开始时调用）"""
    q = queue.Queue()
    _session_stream[session_id] = q
    return q


def get_sse_queue(session_id: str):
    """获取指定 session 的队列"""
    return _session_stream.get(session_id)


def remove_sse_queue(session_id: str):
    """移除指定 session 的队列（连接断开或任务完成时调用）"""
    _session_stream.pop(session_id, None)


def push_to_session(session_id: str, event: str, data: Dict[str, Any]):
    """向指定会话推送事件（LangGraph 节点内调用）

    Args:
        session_id: 会话 ID
        event: 事件类型，如 "delta"、"final"、"progress"
        data: 事件数据字典
    """
    q = _session_stream.get(session_id)
    if q:
        q.put({"event": event, "data": data})


# ==================== SSE 消息打包 ====================

def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式

    SSE 协议格式：
        event: <事件类型>\n
        data: <JSON数据>\n\n
    """
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


# ==================== SSE 生成器（核心） ====================

async def sse_generator(
    session_id: str, request: Request
) -> AsyncGenerator[str, None]:
    """SSE 异步生成器，用于 FastAPI StreamingResponse

    工作流程：
        1. 从全局存储中取出对应 session 的队列
        2. 先发 ready 事件告知前端连接已建立
        3. 循环从队列取消息，打包成 SSE 格式 yield 给前端
        4. 收到 __close__ 或前端断开时退出
        5. finally 中清理队列资源

    Args:
        session_id: 会话 ID
        request: FastAPI Request 对象（用于检测客户端断开）
    """
    q = get_sse_queue(session_id)
    if q is None:
        return

    loop = asyncio.get_running_loop()

    try:
        # 1) 发送连接建立信号
        yield _sse_pack(SSEEvent.READY, {})

        # 2) 循环消费队列
        while True:
            # 检测客户端是否断开连接
            if await request.is_disconnected():
                break

            try:
                # 从队列取消息，最多等待 1 秒
                # run_in_executor 避免阻塞 async 事件循环
                msg = await loop.run_in_executor(None, q.get, True, 1.0)
            except queue.Empty:
                continue  # 超时没消息，继续等

            event = msg.get("event")
            data = msg.get("data")

            if event == SSEEvent.CLOSE:
                break

            yield _sse_pack(event, data)

    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        # 生成器被取消或对端断开：静默退出
        pass
    finally:
        # 3) 清理资源
        remove_sse_queue(session_id)
