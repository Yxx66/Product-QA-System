"""
任务追踪工具

在内存中记录每个任务的状态：
- running_list: 正在运行的节点名
- done_list: 已完成的节点名
- status: 任务总体状态 (processing / completed / failed)
- result: 任务结果（如 answer 文本）

前端通过轮询 GET /status/{task_id} 或 SSE 获取这些信息。
"""
from typing import Dict, List
from collections import defaultdict


# ==================== 任务状态常量 ====================

TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"


# ==================== 内存表 ====================

_tasks_running_list: Dict[str, List[str]] = defaultdict(list)
_tasks_done_list: Dict[str, List[str]] = defaultdict(list)
_tasks_status: Dict[str, str] = {}
_tasks_result: Dict[str, Dict[str, str]] = {}


# ==================== 节点名 → 中文名映射 ====================

_NODE_NAME_TO_CN: Dict[str, str] = {
    # 导入流程
    "entry": "检查文件",
    "pdf_to_md": "PDF转Markdown",
    "md_image": "图片处理",
    "split_documents": "文档切分",
    "item_name_recognition": "商品名识别",
    "bge_embedding": "向量生成",
    "import_milvus": "导入向量数据库",
    "kg_graph": "导入知识图谱",
    # 查询流程
    "item_name_confirm": "确认问题产品",
    "search_embedding": "向量检索",
    "search_embedding_hyde": "HyDE检索",
    "kg_query": "知识图谱查询",
    "network_search": "网络搜索",
    "rrf": "RRF融合排序",
    "rerank": "Rerank重排序",
    "answer_output": "生成答案",
}


def _to_cn(node_name: str) -> str:
    """节点名转中文"""
    return _NODE_NAME_TO_CN.get(node_name, node_name)


# ==================== 任务生命周期管理 ====================

def init_task(task_id: str):
    """初始化一个新任务 — 重置所有状态（每次 query 调用前执行）"""
    _tasks_status[task_id] = TASK_STATUS_PROCESSING
    _tasks_running_list[task_id] = []
    _tasks_done_list[task_id] = []
    _tasks_result.pop(task_id, None)


def add_running_task(task_id: str, node_name: str):
    """节点开始执行 → 加入 running_list"""
    _tasks_running_list[task_id].append(node_name)


def add_done_task(task_id: str, node_name: str):
    """节点执行完成 → 从 running_list 移除，加入 done_list"""
    if node_name in _tasks_running_list[task_id]:
        _tasks_running_list[task_id].remove(node_name)
    _tasks_done_list[task_id].append(node_name)


def update_task_status(task_id: str, status: str):
    """更新任务总体状态"""
    _tasks_status[task_id] = status


# ==================== 任务结果存取 ====================

def set_task_result(task_id: str, key: str, value: str):
    """设置任务结果（如 answer）"""
    if task_id not in _tasks_result:
        _tasks_result[task_id] = {}
    _tasks_result[task_id][key] = value


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    """获取任务结果"""
    return _tasks_result.get(task_id, {}).get(key, default)


# ==================== 任务信息查询 ====================

def get_task_info(task_id: str) -> dict:
    """获取任务完整信息（返回给前端）"""
    done_list = _tasks_done_list.get(task_id, [])
    running_list = _tasks_running_list.get(task_id, [])
    status = _tasks_status.get(task_id, "unknown")

    return {
        "task_id": task_id,
        "status": status,
        "done_list": [f"[{i+1}] {_to_cn(n)}" for i, n in enumerate(done_list)],
        "running_list": [f"[{len(done_list)+i+1}] {_to_cn(n)}" for i, n in enumerate(running_list)],
    }


# ==================== SSE 进度推送 ====================

def task_push_queue(task_id: str):
    """推送任务进度到 SSE 流（流式模式下使用）"""
    from knowledge.tools.sse_utils import push_to_session

    push_to_session(task_id, "progress", {
        "status": _tasks_status.get(task_id, TASK_STATUS_PROCESSING),
        "done_list": _tasks_done_list.get(task_id, []),
        "running_list": _tasks_running_list.get(task_id, []),
    })


# ==================== 清理 ====================

def clear_task(task_id: str):
    """清理指定任务的所有数据"""
    _tasks_running_list.pop(task_id, None)
    _tasks_done_list.pop(task_id, None)
    _tasks_status.pop(task_id, None)
    _tasks_result.pop(task_id, None)
