from pathlib import Path
from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState, create_default_state
from knowledge.processor.import_process.exceptions import FileProcessingError, ValidationError


class EntryNode(BaseNode):
    """入口节点：初始化任务"""
    name = "entry"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("初始化", f"任务ID: {state.get('task_id', '')}")
        print(state)
        file_path = Path(state["import_file_path"])

        if file_path.exists():
            self.logger.info(f"文件存在: {file_path}")
        else:
            raise FileProcessingError(f"文件不存在: {file_path}")
        
        state["import_file_path"] = str(file_path)
        state["file_dir"] = str(file_path.parent)
        
        if file_path.suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = str(file_path)
        elif file_path.suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = str(file_path)
        else:
            raise ValidationError(f"文件格式不支持: {file_path}")

        state["file_title"] = file_path.stem
        self.logger.info(state)
        return state
