import re
import os
import json
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState, create_default_state
from knowledge.processor.import_process.exceptions import ImageProcessingError
from knowledge.processor.import_process.config import get_config


@dataclass
class ImageContext:
    """图片上下文信息"""
    image_filename: str       # 图片文件名
    image_full_path: str      # 图片完整路径
    section_heading: str      # 最近的标题行
    pre_text: str             # 图片上方上下文
    post_text: str            # 图片下方上下文
    summary: str = ""         # VLM 生成的摘要
    minio_url: str = ""       # MinIO 公开 URL


class MD_IMAGE(BaseNode):
    name = "md_image"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("MD图片处理", "开始处理")
        image_dir = self.get_image_path(state)
        self.scan_Image(state)
        self.produce_Image_content(state)
        self.write_new_md(state)
        return state

    def get_image_path(self, state: ImportGraphState) -> str:
        md_path = Path(state["md_path"])
        image_dir = md_path.parent / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"图片目录: {image_dir}")
        return str(image_dir)

    def scan_Image(self, state: ImportGraphState) -> bool:
        self.log_step("图片扫描", "扫描并提取图片上下文")
        md_content, images_dir = self._read_md_content(state)
        image_contexts = self._find_image_contexts(md_content, images_dir, max_chars=200)
        state["image_contexts"] = image_contexts
        self.logger.info(f"找到 {len(image_contexts)} 张需要处理的图片")
        return len(image_contexts) > 0

    def _read_md_content(self, state: ImportGraphState) -> Tuple[str, Path]:
        """读取 md 文件内容，返回 (md_content, images_dir)"""
        md_path_str = state.get("md_path", "")
        if not md_path_str:
            raise ImageProcessingError("状态中 md_path 为空")
        md_path = Path(md_path_str)
        if not md_path.exists():
            raise ImageProcessingError(f"MD文件不存在: {md_path}")
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        state["md_content"] = md_content
        images_dir = md_path.parent / "images"
        return md_content, images_dir

    def _find_image_contexts(
        self,
        md_content: str,
        images_dir: Path,
        max_chars: int = 200
    ) -> List[ImageContext]:

        """扫描 md 中的图片引用，为每张图片提取上下文"""
        image_pattern = re.compile(r"!\[.*?\]\((.*?)\)")
        allowed_extensions = get_config().image_extensions
        lines = md_content.split("\n")
        results = []

        # 记录所有标题行的位置
        heading_lines = {}
        for idx, line in enumerate(lines):
            if re.match(r"^#{1,6}\s+", line):
                heading_lines[idx] = line.strip()

        # 记录所有图片行的信息
        image_lines = []
        for line_idx, line in enumerate(lines):
            match = image_pattern.search(line)
            if match:
                img_ref = match.group(1)
                # 提取文件名（去掉路径前缀和查询参数）
                img_filename = Path(img_ref).name.split("?")[0]
                # 过滤非图片后缀
                if Path(img_filename).suffix.lower() not in allowed_extensions:
                    continue
                image_lines.append((line_idx, img_filename, img_ref))

        for line_idx, img_filename, img_ref in image_lines:
            # 构建图片完整路径
            image_full_path = str(images_dir / img_filename)
            if not os.path.exists(image_full_path):
                self.logger.debug(f"图片文件不存在，跳过: {image_full_path}")
                continue

            # 向上找最近的标题
            section_heading = ""
            heading_line_idx = -1
            for i in range(line_idx - 1, -1, -1):
                if i in heading_lines:
                    section_heading = heading_lines[i]
                    heading_line_idx = i
                    break

            # 提取上文（标题到图片之间的段落）
            pre_start = heading_line_idx + 1 if heading_line_idx >= 0 else 0
            pre_lines = lines[pre_start:line_idx]
            pre_text = self._extract_paragraphs_with_limit(
                pre_lines, max_chars, direction="backward"
            )

            # 向下找下一个标题作为边界
            next_heading_idx = len(lines)
            for i in range(line_idx + 1, len(lines)):
                if i in heading_lines:
                    next_heading_idx = i
                    break

            # 提取下文（图片到下一个标题之间的段落）
            post_lines = lines[line_idx + 1:next_heading_idx]
            post_text = self._extract_paragraphs_with_limit(
                post_lines, max_chars, direction="forward"
            )

            results.append(ImageContext(
                image_filename=img_filename,
                image_full_path=image_full_path,
                section_heading=section_heading,
                pre_text=pre_text,
                post_text=post_text,
            ))

        return results

    def _extract_paragraphs_with_limit(
        self,
        lines: List[str],
        max_chars: int,
        direction: str = "forward"
    ) -> str:
        """从行列表中提取完整段落，总字符数不超过 max_chars"""
        # 合并连续非空行为段落
        paragraphs = []
        current_para = []

        for line in lines:
            stripped = line.strip()
            if stripped == "":
                if current_para:
                    paragraphs.append("\n".join(current_para))
                    current_para = []
            else:
                # 跳过图片行
                if re.match(r"^!\[.*?\]\(.*?\)$", stripped):
                    if current_para:
                        paragraphs.append("\n".join(current_para))
                        current_para = []
                    continue
                current_para.append(stripped)

        if current_para:
            paragraphs.append("\n".join(current_para))

        paragraphs = [p for p in paragraphs if p.strip()]

        if not paragraphs:
            return ""

        # backward 优先取靠近图片的段落
        if direction == "backward":
            paragraphs = list(reversed(paragraphs))

        # 在字符数限制内尽量多取完整段落
        selected = []
        total_chars = 0

        for para in paragraphs:
            para_len = len(para)
            if total_chars + para_len > max_chars and selected:
                break
            selected.append(para)
            total_chars += para_len

        if direction == "backward":
            selected = list(reversed(selected))

        return "\n\n".join(selected)
    def _Image_summary(self, ctx: ImageContext, vlm_client=None) -> str:
        self.log_step("图片摘要", f"处理: {ctx.image_filename}")

        # 读取图片并 base64 编码
        try:
            with open(ctx.image_full_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
        except IOError as e:
            self.logger.error(f"无法读取图片 {ctx.image_full_path}: {e}")
            return "图片描述"

        # 组装上下文信息
        context_parts = []
        if ctx.section_heading:
            context_parts.append(f"所属章节标题：{ctx.section_heading}")
        if ctx.pre_text:
            context_parts.append(f"图片上文：{ctx.pre_text}")
        if ctx.post_text:
            context_parts.append(f"图片下文：{ctx.post_text}")
        context_info = "\n".join(context_parts) if context_parts else "无可用上下文"

        prompt = f"""任务：为Markdown文档中的图片生成一个简短的中文标题。
背景信息：
图片上下文：
{context_info}
请结合图片视觉内容和上述上下文信息，用中文简要总结这张图片的内容，
生成一个精准的中文标题（不要包含"图片"二字）。"""

        # 调用 VLM
        try:
            completion = vlm_client.chat.completions.create(
                model="qwen3.6-plus",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
            )
            summary = completion.choices[0].message.content.strip().replace("\n", " ")
            self.logger.info(f"图片摘要生成成功: {summary}")
            return summary
        except Exception as e:
            self.logger.warning(f"VLM调用失败 {ctx.image_filename}: {e}")
            return "图片描述"

    def _upload_image_to_minio(self, ctx: ImageContext) -> str:
        self.log_step("MinIO上传", f"上传: {ctx.image_filename}")
        config = get_config()

        try:
            from minio import Minio
            from minio.error import S3Error
            import urllib3

            http_client = urllib3.PoolManager(
                timeout=urllib3.Timeout(connect=10, read=30),
                maxsize=10,
            )
            client = Minio(
                config.minio_endpoint,
                access_key=config.minio_access_key,
                secret_key=config.minio_secret_key,
                secure=config.minio_secure,
                http_client=http_client,
            )

            bucket_name = config.minio_bucket
            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
                self.logger.info(f"创建 bucket: {bucket_name}")

            client.fput_object(bucket_name, ctx.image_filename, ctx.image_full_path)

            # 设置 bucket 为公开读取
            policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
                }]
            }
            client.set_bucket_policy(bucket_name, json.dumps(policy))

            url = f"{config.get_minio_base_url()}/{ctx.image_filename}"
            self.logger.info(f"上传成功: {url}")
            return url
        except Exception as e:
            self.logger.warning(f"MinIO上传失败 {ctx.image_filename}: {e}")
            return ""

    def produce_Image_content(self, state: ImportGraphState):
        self.log_step("图片处理", "生成摘要并替换链接")
        md_content = state["md_content"]
        images = state["image_contexts"]

        # 支持 max_images 限制处理数量
        max_images = state.get("max_images", 0)
        if max_images > 0:
            images = images[:max_images]

        total = len(images)

        # 创建 VLM 客户端（只创建一次，加超时）
        from openai import OpenAI
        vlm_client = OpenAI(
            api_key=os.getenv("VLM_MODEL_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=60.0,
        )

        for i, ctx in enumerate(images, 1):
            self.logger.info(f"处理图片 {i}/{total}: {ctx.image_filename}")
            try:
                ctx.summary = self._Image_summary(ctx, vlm_client=vlm_client)
            except Exception as e:
                self.logger.warning(f"VLM异常降级: {ctx.image_filename}: {e}")
                ctx.summary = "图片描述"

            try:
                ctx.minio_url = self._upload_image_to_minio(ctx)
            except Exception as e:
                self.logger.warning(f"MinIO异常降级: {ctx.image_filename}: {e}")
                ctx.minio_url = ""

            md_content = self.replace_md_image(ctx, md_content)

        state["md_content"] = md_content

    def replace_md_image(self, ctx: ImageContext, md_content: str) -> str:
        if not ctx.minio_url:
            self.logger.warning(f"跳过替换（无URL）: {ctx.image_filename}")
            return md_content
        pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(ctx.image_filename) + r".*?\)"
        )
        new_ref = f"![{ctx.summary}]({ctx.minio_url})"
        new_content, count = pattern.subn(new_ref, md_content)
        if count > 0:
            self.logger.info(f"替换成功: {ctx.image_filename} → {ctx.summary}")
        else:
            self.logger.warning(f"未找到匹配: {ctx.image_filename}")
        return new_content

    def write_new_md(self, state: ImportGraphState) -> str:
        md_path = Path(state["md_path"])
        new_md_path = md_path.parent / f"{md_path.stem}_new{md_path.suffix}"
        with open(new_md_path, "w", encoding="utf-8") as f:
            f.write(state["md_content"])
        self.log_step("写入新MD", f"输出: {new_md_path}")
        state["md_path"] = str(new_md_path)
        return str(new_md_path)