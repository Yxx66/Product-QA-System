"""
FastAPI 依赖注入

用 @lru_cache 保证服务全局只创建一次（单例模式）。
"""
from functools import lru_cache

from pathlib import Path
from knowledge.services.task_service import TaskService
from knowledge.services.file_import_service import FileImportService
from knowledge.services.query_service import QueryService


@lru_cache
def get_task_service() -> TaskService:
    return TaskService()


@lru_cache
def get_file_import_service() -> FileImportService:
    base_dir = str(Path(__file__).resolve().parents[2])
    return FileImportService(base_dir=base_dir, task_service=get_task_service())


@lru_cache
def get_query_service() -> QueryService:
    """查询服务单例"""
    return QueryService()
