from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState, create_default_state
from knowledge.tools.BGE3_client import get_bgem3_client, BGEM3EmbeddingFunction
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import ValidationError


class bge_embedding(BaseNode):
    name = "bge_embedding"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        config = get_config()

        # 获取切片
        chunks = state.get("chunks", [])
        if not isinstance(chunks, list) or not chunks:
            raise ValidationError("chunks 为空或无效", node_name=self.name)

        self.log_step("step_1", f"开始为 {len(chunks)} 个切片生成向量")
        try:
            bge_m3_ef = get_bgem3_client()
        except Exception as e:
            raise ValidationError(f"获取BGE M3模型失败: {e}", node_name=self.name)

        output_data = []
        batch_size = config.embedding_batch_size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_output = self._process_batch(bge_m3_ef, batch, i, len(chunks))
            output_data.extend(batch_output)

        state["chunks"] = output_data
        return state

    def _process_batch(
            self,
            bge_m3_ef: BGEM3EmbeddingFunction,
            batch: list[dict],
            start_idx: int,
            total: int
    ) -> list[dict]:
        try:
            # 拼装 item_name + content 作为编码文本
            texts = [
                (doc.get("item_name", "") or "") + "\n" + (doc.get("content", "") or "")
                for doc in batch
            ]
            embeddings = bge_m3_ef.encode_documents(texts)

            output = []
            for j, doc in enumerate(batch):
                # 提取稠密向量
                dense_vector = embeddings["dense"][j].tolist()

                # 提取稀疏向量
                start = embeddings["sparse"].indptr[j]
                end = embeddings["sparse"].indptr[j + 1]
                token_ids = embeddings["sparse"].indices[start:end].tolist()
                weights = embeddings["sparse"].data[start:end].tolist()
                sparse_vector = dict(zip(token_ids, weights))

                item = {
                    "content": doc.get("content"),
                    "title": doc.get("title"),
                    "parent_title": doc.get("parent_title", ""),
                    "file_title": doc.get("file_title"),
                    "item_name": doc.get("item_name"),
                    "dense_vector": dense_vector,
                    "sparse_vector": sparse_vector,
                }
                output.append(item)

            self.logger.info(f"成功处理批次 {start_idx + 1}-{start_idx + len(batch)}/{total}")
            return output
        except Exception as e:
            self.logger.error(f"批次 {start_idx + 1}-{start_idx + len(batch)} 处理失败: {e}")
            return batch


if __name__ == '__main__':
    from knowledge.processor import setup_logging
    setup_logging()

    node = bge_embedding()
    state = create_default_state()
    state["file_title"] = "测试商品"
    state["chunks"] = [
        {"title": "## 产品简介", "content": "苏伯尔5000W大功率电磁炉", "file_title": "测试商品", "parent_title": "测试商品", "item_name": "苏伯尔电磁炉"},
        {"title": "## 规格参数", "content": "额定功率5000W，电压220V", "file_title": "测试商品", "parent_title": "测试商品", "item_name": "苏伯尔电磁炉"},
    ]

    result = node.process(state)
    for i, chunk in enumerate(result["chunks"]):
        print(f"chunk[{i}]: dense_vector长度={len(chunk.get('dense_vector', []))}, sparse_vector长度={len(chunk.get('sparse_vector', {}))}")
