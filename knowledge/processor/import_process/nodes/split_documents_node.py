from pathlib import Path
import re
from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState, create_default_state
from knowledge.processor.import_process.exceptions import FileProcessingError, ValidationError


class SplitDocumentsNode(BaseNode):
    name = "split_documents"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("文档拆分", "开始拆分文档")
        #step1:将md标准化
        self.md_standardization(state)
        #step2:语义切分文档
        self.split_documents_heading(state)
        self.split_documents_content(state)
        return state

    def md_standardization(self, state: ImportGraphState) -> ImportGraphState:
        #判断路径是否存在
        if not state.get("md_path", ""):
            raise FileProcessingError("md_path 为空")
        #读取md文件内容
        self.log_step("标准化", f"开始: {state['md_path']}")
        md_path = Path(state["md_path"])
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(content)
        state["md_content"] = content
        self.log_step("标准化", f"完成: {md_path}")
        return state
    def split_documents_heading(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("语义切分", "按标题切分文档")
        content = state.get("md_content", "")
        file_title = state.get("file_title", "")

        # 正则匹配标题
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")
        lines = content.split("\n")

        sections = []
        current_title = ""
        current_level = 0
        body_lines = []
        has_title = False
        in_fence = False

        # 层级追踪：记录 1-6 级标题的最新足迹
        hierarchy = [""] * 7

        def _flush():
            """结算当前积累的内容为一个 section"""
            body = "\n".join(body_lines).strip()
            if current_title or body:
                # 向上找父标题
                parent_title = ""
                for lvl in range(current_level - 1, 0, -1):
                    if hierarchy[lvl]:
                        parent_title = hierarchy[lvl]
                        break
                # 找不到父标题，自己当家长
                if not parent_title:
                    parent_title = current_title if current_title else file_title

                sections.append({
                    "file_title": file_title,
                    "title": current_title,
                    "parent_title": parent_title,
                    "body": body,
                })

        for line in lines:
            # 检测代码围栏
            if line.strip().startswith("```") or line.strip().startswith("~~~"):
                in_fence = not in_fence

            match = heading_re.match(line) if not in_fence else None

            if match:
                has_title = True
                _flush()

                level = len(match.group(1))
                current_level = level
                current_title = line.strip()
                hierarchy[level] = current_title

                # 清空下级标题足迹
                for i in range(level + 1, 7):
                    hierarchy[i] = ""

                body_lines = []
            else:
                body_lines.append(line)

        # 结算最后一段
        _flush()

        # 无标题兜底
        if not has_title:
            sections = [{
                "file_title": file_title,
                "title": "无标题",
                "parent_title": file_title,
                "body": content,
            }]
            self.logger.info("全文无标题，作为单个 chunk 处理")

        state["chunks"] = sections
        self.logger.info(f"切分完成，共 {len(sections)} 个 chunk")
        return state
    def _extract_tables(self, text):
        """提取 markdown 表格，用占位符替换，返回替换后文本和表格列表"""
        lines = text.split("\n")
        tables = []
        result_lines = []
        i = 0
        while i < len(lines):
            if (lines[i].strip().startswith("|") and
                    i + 1 < len(lines) and re.match(r"\s*\|[\s:]*-+[\s:]*\|", lines[i + 1])):
                table_lines = [lines[i]]
                i += 1
                while i < len(lines) and lines[i].strip().startswith("|"):
                    table_lines.append(lines[i])
                    i += 1
                tables.append("\n".join(table_lines))
                result_lines.append(f"__TABLE_PLACEHOLDER_{len(tables) - 1}__")
            else:
                result_lines.append(lines[i])
                i += 1
        return "\n".join(result_lines), tables

    def _restore_tables(self, text, tables):
        """还原表格占位符"""
        for idx, table in enumerate(tables):
            text = text.replace(f"__TABLE_PLACEHOLDER_{idx}__", table)
        return text

    def _assemble_content(self, chunks):
        """组装最终 content = title + body，清理内部字段"""
        result = []
        for chunk in chunks:
            title = chunk.get("title", "")
            body = chunk.get("body", "")
            if title and body:
                content = f"{title}\n\n{body}"
            else:
                content = title or body
            result.append({
                "title": title,
                "content": content.strip(),
                "file_title": chunk.get("file_title", ""),
                "parent_title": chunk.get("parent_title", ""),
            })
        return result

    #二次切分
    def split_documents_content(self, state: ImportGraphState, max_content_length: int = 200, min_content_length: int = 500) -> ImportGraphState:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.log_step("二次切分", "判断chunks是否大于max_content_length")
        chunks = state.get("chunks", [])
        split_chunks = []

        # 第一步：切分超长 chunk
        for chunk in chunks:
            full_text = chunk["title"] + "\n" + chunk["body"]
            if len(full_text) <= max_content_length:
                split_chunks.append(chunk)
            else:
                # 预留 title 前缀空间
                title_prefix = chunk["title"] + "\n"
                available = max_content_length - len(title_prefix)
                if available <= 0:
                    split_chunks.append(chunk)
                    continue

                self.log_step("二次切分", f"对 '{chunk['title']}' 进行切分 (长度 {len(full_text)})")

                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
                    chunk_size=available,
                    chunk_overlap=0,
                )
                protected_text, tables = self._extract_tables(chunk["body"])
                texts = text_splitter.split_text(protected_text)
                texts = [self._restore_tables(t, tables) for t in texts]
                for part_idx, text in enumerate(texts):
                    split_chunks.append({
                        "file_title": chunk["file_title"],
                        "title": f"{chunk['title']}-{part_idx + 1}",
                        "parent_title": chunk["parent_title"],
                        "body": text,
                    })

        # 第二步：合并短 chunk（只有 < min_content_length 的才合并）
        merged_chunks = [split_chunks[0]] if split_chunks else []
        for chunk in split_chunks[1:]:
            prev = merged_chunks[-1]
            # 前一个 chunk 太短 + 同一个父标题 + 合并后不超过 max_content_length
            if (len(prev["body"]) < min_content_length and
                    chunk.get("parent_title") == prev.get("parent_title") and
                    len(prev["body"]) + len(chunk["body"]) <= max_content_length):
                prev["body"] = prev["body"] + "\n" + chunk["body"]
            else:
                merged_chunks.append(chunk)

        state["chunks"] = self._assemble_content(merged_chunks)
        self.log_step("二次切分", f"完成，共 {len(merged_chunks)} 个 chunk")
        return state
    #




if __name__ == "__main__":
    state = create_default_state()
    state["md_path"] = r"D:\python\tcl\project\Mananger_DataBase\mineru_output\表格测试.md"
    state["file_title"] = "表格测试"
    node = SplitDocumentsNode()
    node.process(state)

    print(f"\n{'='*60}")
    print(f"共 {len(state['chunks'])} 个 chunk")
    print(f"{'='*60}")
    for i, chunk in enumerate(state["chunks"]):
        print(f"\n--- Chunk {i+1} (长度 {len(chunk['content'])}) ---")
        print(f"标题: {chunk['title']}")
        print(f"内容:\n{chunk['content']}")
        print()


