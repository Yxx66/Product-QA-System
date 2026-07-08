from pathlib import Path
from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState, create_default_state
from knowledge.processor.import_process.exceptions import FileProcessingError, ValidationError, PdfConversionError
import subprocess
import sys


class PdfToMdNode(BaseNode):
    """PDF转Markdown节点"""
    name = "pdf_to_md"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("PDF转换", "开始转换")
        self.validate_path(state)
        self.validate_file_dir(state)
        self.convert_pdf_to_md(state)
        state["md_path"] = self.get_output_path(state)
        return state

    def validate_path(self, state: ImportGraphState) -> bool:
        if state.get("pdf_path", None) is None:
            raise ValidationError("pdf_path is None")
        if not Path(state["pdf_path"]).exists():
            raise ValidationError(f"pdf_path not exists: {state['pdf_path']}")
        return True

    def validate_file_dir(self, state: ImportGraphState) -> bool:
        if state.get("file_dir", None) is None:
            raise ValidationError("file_dir is None")
        file_dir = Path(state["file_dir"])
        if not file_dir.exists():
            raise ValidationError(f"file_dir not exists: {state['file_dir']}")
        return True

    def convert_pdf_to_md(self, state: ImportGraphState) -> bool:
        pdf_path = Path(state["pdf_path"])
        file_dir = Path(state["file_dir"])

        output_dir = file_dir / "mineru_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "mineru",
            "-p", str(pdf_path),
            "-o", str(output_dir),
            "--backend", "pipeline",
            "--device", "cpu"
        ]

        self.logger.info(f"执行命令: {' '.join(cmd)}")

        log_file = output_dir / "mineru.log"
        try:
            with open(log_file, "w", encoding="utf-8") as log_f:
                result = subprocess.run(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    timeout=900,  # 15分钟超时
                )

            if result.returncode != 0:
                self.logger.error(f"PDF转换失败，返回码: {result.returncode}")
                self.logger.error(f"日志: {log_file}")
                return False

            self.logger.info("PDF转换成功")
            return True

        except subprocess.TimeoutExpired:
            self.logger.error(f"PDF转换超时(900秒)")
            return False
        except Exception as e:
            self.logger.error(f"PDF转换异常: {e}")
            return False
#获取最终输出路径
    def get_output_path(self, state: ImportGraphState) -> str:
        file_dir = Path(state["file_dir"])
        file_title = state["file_title"]

        # MinerU 输出在 auto 子目录
        possible_paths = [
            file_dir / "mineru_output" / file_title / "auto" / f"{file_title}.md",
            file_dir / "mineru_output" / file_title / f"{file_title}.md",
            file_dir / "mineru_output" / f"{file_title}.md",
        ]

        for path in possible_paths:
            if path.exists():
                self.logger.info(f"找到MD文件: {path}")
                return str(path)

        default_path = file_dir / "mineru_output" / file_title / "auto" / f"{file_title}.md"
        self.logger.warning(f"未找到MD文件，返回默认路径: {default_path}")
        return str(default_path)
