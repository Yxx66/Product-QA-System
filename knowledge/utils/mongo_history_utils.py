"""
MongoDB 历史对话工具

提供历史会话的完整 CRUD 操作：
- get_recent_messages: 读取最近 N 条历史消息
- save_chat_message:   写入一条对话消息
- update_message_item_names: 回填消息的 item_names
- clear_history:       清空指定会话的全部历史
"""


import time
import logging
from typing import List, Dict, Any, Optional

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)

# ==================== MongoDB 连接 ====================

_client: Optional[MongoClient] = None


def _get_client() -> MongoClient:
    """获取 MongoDB 客户端单例"""
    global _client
    if _client is None:
        import os
        mongo_url = os.getenv("MONGO_URL", "mongodb://localhost:27017")
        _client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
    return _client


def _get_collection():
    """获取 chat_history 集合"""
    import os
    db_name = os.getenv("MONGO_DB_NAME", "kb001")
    collection_name = os.getenv("MONGO_HISTORY_COLLECTION", "chat_history")
    client = _get_client()
    db = client[db_name]
    return db[collection_name]


# ==================== 读写操作 ====================

def get_recent_messages(
    session_id: str,
    limit: int = 10,
    include_ids: bool = True,
) -> List[Dict[str, Any]]:
    """获取指定会话的最近 N 条历史消息（按时间正序）。

    Args:
        session_id: 会话 ID
        limit:     获取条数上限
        include_ids: 是否在返回结果中包含 _id（用于后续回填）

    Returns:
        消息列表，每条消息包含 session_id / role / text / rewritten_query / item_names / ts / _id
    """
    try:
        coll = _get_collection()
        projection = None if include_ids else {"_id": 0}
        cursor = (
            coll.find({"session_id": session_id}, projection)
            .sort("ts", DESCENDING)
            .limit(limit)
        )
        messages = list(cursor)
        messages.reverse()  # 恢复为正序（最早在前，最新在后）
        return messages
    except PyMongoError as e:
        logger.error(f"查询历史消息失败 (session={session_id}): {e}")
        return []


def save_chat_message(
    session_id: str,
    role: str,
    text: str,
    rewritten_query: str = "",
    item_names: Optional[List[str]] = None,
) -> Optional[str]:
    """保存一条对话消息到 MongoDB。

    Args:
        session_id:     会话 ID
        role:           "user" 或 "assistant"
        text:           消息文本
        rewritten_query: 重写后的查询（用户消息用）
        item_names:     关联的商品名列表

    Returns:
        插入记录的 _id（字符串），失败返回 None
    """
    if item_names is None:
        item_names = []

    doc = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query,
        "item_names": item_names,
        "ts": time.time(),
    }

    try:
        coll = _get_collection()
        result = coll.insert_one(doc)
        return str(result.inserted_id)
    except PyMongoError as e:
        logger.error(f"保存聊天消息失败 (session={session_id}, role={role}): {e}")
        return None


def update_message_item_names(
    message_ids: List[str],
    item_names: List[str],
) -> int:
    """回填指定消息的 item_names 字段。

    只更新 item_names 为空的记录（字段不存在 / 空列表 / null），
    通过 $or 在数据库层面做二次防御，防止并发场景下误覆盖。

    Args:
        message_ids: 需要回填的消息 _id 列表（字符串）
        item_names:  要写入的商品名列表

    Returns:
        实际修改的文档数量
    """
    if not message_ids or not item_names:
        return 0

    try:
        from bson import ObjectId
        coll = _get_collection()
        obj_ids = [ObjectId(mid) for mid in message_ids]

        result = coll.update_many(
            {
                "_id": {"$in": obj_ids},
                "$or": [
                    {"item_names": {"$exists": False}},
                    {"item_names": []},
                    {"item_names": None},
                ],
            },
            {"$set": {"item_names": item_names}},
        )
        return result.modified_count
    except PyMongoError as e:
        logger.error(f"回填 item_names 失败: {e}")
        return 0


def clear_history(session_id: str) -> int:
    """清空指定会话的全部历史记录。

    Args:
        session_id: 会话 ID

    Returns:
        删除的文档数量
    """
    try:
        coll = _get_collection()
        result = coll.delete_many({"session_id": session_id})
        return result.deleted_count
    except PyMongoError as e:
        logger.error(f"清空历史记录失败 (session={session_id}): {e}")
        return 0


def list_sessions(limit: int = 100) -> List[Dict[str, Any]]:
    """列出所有历史会话，按最近活动时间倒序。

    对 chat_history 做聚合查询，按 session_id 分组，
    取每条对话的第一条 user 消息作为标题预览。

    Args:
        limit: 返回会话数量上限

    Returns:
        会话列表，每条包含:
        - session_id: 会话 ID
        - title: 第一条用户消息前 30 字（作标题）
        - first_ts: 第一条消息时间戳
        - last_ts: 最新消息时间戳
        - message_count: 总消息条数
    """
    try:
        coll = _get_collection()
        pipeline = [
            {"$sort": {"ts": 1}},  # 按时间升序，确保 $first 拿到最早消息
            {
                "$group": {
                    "_id": "$session_id",
                    "first_message": {"$first": "$text"},
                    "first_role": {"$first": "$role"},
                    "first_ts": {"$first": "$ts"},
                    "last_ts": {"$last": "$ts"},
                    "message_count": {"$sum": 1},
                }
            },
            {"$sort": {"last_ts": -1}},  # 最近活跃的排在前面
            {"$limit": limit},
        ]
        sessions = list(coll.aggregate(pipeline))

        result = []
        for s in sessions:
            title = ""
            if s.get("first_message"):
                # 取前 30 个字符作标题，去掉换行
                text = s["first_message"].replace("\n", " ").strip()
                title = text[:30] + ("..." if len(text) > 30 else "")

            result.append({
                "session_id": s["_id"],
                "title": title or "新对话",
                "first_ts": s.get("first_ts"),
                "last_ts": s.get("last_ts"),
                "message_count": s.get("message_count", 0),
            })

        return result
    except PyMongoError as e:
        logger.error(f"列出会话列表失败: {e}")
        return []


def ensure_indexes():
    """创建历史记录所需的索引（在应用启动时调用）。

    索引策略：
    - session_id + ts 复合索引：支持按会话按时间查询
    - session_id + role 复合索引：支持按会话按角色查询
    """
    try:
        coll = _get_collection()
        coll.create_index(
            [("session_id", ASCENDING), ("ts", DESCENDING)],
            name="idx_session_ts",
        )
        logger.info("MongoDB 历史记录索引创建完成")
    except PyMongoError as e:
        logger.warning(f"创建索引失败: {e}")
