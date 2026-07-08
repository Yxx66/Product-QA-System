"""
文件导入服务

处理文件上传，启动后台流水线任务。
"""
import uuid
import logging
from pathlib import Path
from knowledge.services.task_service import TaskService
from knowledge.processor.import_process.state import create_default_state

logger = logging.getLogger(__name__)


class FileImportService:
    def __init__(self, base_dir: str, task_service: TaskService):
        self.base_dir = Path(base_dir)
        self.task_service = task_service

    def process_file_upload(self, file) -> tuple:
        """
        同步处理：保存文件，返回 (task_id, file_dir, import_file_path)

        Args:
            file: FastAPI UploadFile 对象
        """
        # 1. 生成 task_id
        task_id = str(uuid.uuid4())[:8]

        # 2. 创建任务目录
        file_dir = self.base_dir / "import_temp_dir" / task_id
        file_dir.mkdir(parents=True, exist_ok=True)

        # 3. 保存文件
        file_path = file_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # 4. 初始化任务状态
        self.task_service.init_task(task_id)

        logger.info(f"文件上传成功: {file.filename}, task_id={task_id}")
        return task_id, str(file_dir), str(file_path)

    def run_upload_file_task(self, task_id: str, file_dir: str, import_file_path: str):
        """
        异步处理：在后台线程中执行 LangGraph 流水线

        这个方法由 FastAPI BackgroundTasks 调用。
        """
        try:
            # 1. 构建初始状态
            state = create_default_state(
                task_id=task_id,
                import_file_path=import_file_path,
                file_dir=file_dir,
            )

            # 2. 根据文件类型设置标志
            if import_file_path.endswith(".pdf"):
                state["is_pdf_read_enabled"] = True
            elif import_file_path.endswith(".md"):
                state["is_md_read_enabled"] = True
                state["md_path"] = import_file_path

            # 3. 执行流水线
            from knowledge.processor.import_process.main_graph import build_graph
            app = build_graph()
            result = app.invoke(state)

            # 4. 标记完成
            self.task_service.complete_task(task_id)
            logger.info(f"[{task_id}] 流水线完成")

        except Exception as e:
            logger.error(f"[{task_id}] 流水线失败: {e}")
            self.task_service.fail_task(task_id)
