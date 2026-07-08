"""
文件导入路由

POST /upload    — 上传文件，启动后台任务
GET  /status/{task_id} — 查询任务状态
"""
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends

from knowledge.schema.upload_schema import UploadResponse
from knowledge.schema.task_schema import TaskStatusResponse
from knowledge.services.file_import_service import FileImportService
from knowledge.services.task_service import TaskService
from knowledge.core.deps import get_file_import_service, get_task_service

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        service: FileImportService = Depends(get_file_import_service),
) -> UploadResponse:
    """
    上传文件，启动后台流水线。

    1. 同步：保存文件，生成 task_id
    2. 异步：把流水线放到后台线程执行
    3. 立刻返回 task_id 给前端
    """
    task_id, file_dir, import_file_path = service.process_file_upload(file)

    # 放到后台线程执行，不阻塞请求
    background_tasks.add_task(
        service.run_upload_file_task, task_id, file_dir, import_file_path
    )

    return UploadResponse(message="上传成功", task_id=task_id)


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_status(
        task_id: str,
        task_service: TaskService = Depends(get_task_service),
) -> TaskStatusResponse:
    """
    查询任务状态，前端每 1.5 秒轮询一次。
    """
    info = task_service.get_task_info(task_id)
    return TaskStatusResponse(**info)
