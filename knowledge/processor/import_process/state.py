"""
导入流程状态类型定义

定义完整的状态结构和辅助函数
"""

from typing import TypedDict, List
import copy


class Chunk(TypedDict):
    pk: str
    chunk_id: str
    title: str
    content: str
    file_title: str
    parent_title: str


class ImportGraphState(TypedDict, total=False):
    """
    导入流程图状态

    包含整个导入流程中传递的所有数据。
    使用 total=False 表示所有字段都是可选的。
    """

    # ==================== 任务标识 ====================
    task_id: str

    # ==================== 控制标志 ====================
    is_md_read_enabled: bool
    is_pdf_read_enabled: bool

    # ==================== 路径信息 ====================
    import_file_path: str # 导入文件路径
    file_dir: str # 文件目录
    pdf_path: str # pdf文件路径
    md_path: str # md文件路径

    # ==================== 文件信息 ====================
    file_title: str
    item_name: str

    # ==================== 处理中间数据 ====================
    md_content: str
    chunks: List[Chunk]
    image_contexts: List


# ==================== 默认状态模板 ====================
GRAPH_DEFAULT_STATE: ImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "file_dir": "",
    "import_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "md_content": "",
    "chunks": [],
    "image_contexts": [],
    "item_name": "",
}


def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持覆盖

    Args:
        **overrides: 要覆盖的字段

    Returns:
        新的状态实例

    Examples:
        >>> state = create_default_state(
        ...     task_id="task_001",
        ...     import_file_path="doc.pdf"
        ... )
    """
    state = copy.deepcopy(GRAPH_DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """
    获取默认状态副本

    Returns:
        状态副本（避免全局污染）
    """
    return copy.deepcopy(GRAPH_DEFAULT_STATE)
