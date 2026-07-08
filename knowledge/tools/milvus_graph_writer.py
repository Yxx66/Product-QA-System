import logging
from typing import List, Dict, Any
from pymilvus import DataType


class MilvusGraphWriter:
    """负责将知识图谱的实体和关系向量化并写入 Milvus。"""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.logger = logging.getLogger(self.__class__.__name__)

    def clear(self, milvus_client, item_name: str):
        """按 item_name 清理旧数据"""
        try:
            if milvus_client.has_collection(self.collection_name):
                milvus_client.delete(
                    collection_name=self.collection_name,
                    filter=f'item_name == "{item_name}"'
                )
                self.logger.info(f"已清理 {item_name} 的旧数据")
        except Exception as e:
            self.logger.warning(f"清理旧数据失败: {e}")

    def insert(self, milvus_client, entities: List[Dict], relations: List[Dict],
               item_name: str, source_id: str = "", content: str = ""):
        """对外入口：将实体和关系写入 Milvus"""
        from knowledge.tools.BGE3_client import get_bgem3_client

        if not entities and not relations:
            self.logger.warning("无实体和关系数据，跳过写入")
            return

        # 1. 确保集合存在
        self._ensure_collection(milvus_client)

        # 2. 组装文本和元数据
        texts = []
        metadata = []
        for e in entities:
            text = e["name"] + " " + e["type"]
            texts.append(text)
            metadata.append({
                "record_type": "entity", "node_type": e["type"], "name": e["name"],
                "source": "", "target": "", "relation": "",
                "item_name": item_name, "source_id": source_id,
            })
        for r in relations:
            text = r["source"] + " " + r["type"] + " " + r["target"]
            texts.append(text)
            metadata.append({
                "record_type": "relation", "node_type": "relation", "name": "",
                "source": r["source"], "target": r["target"], "relation": r["type"],
                "item_name": item_name, "source_id": source_id,
            })

        # 3. 生成向量
        try:
            bge_ef = get_bgem3_client()
            embeddings = bge_ef.encode_documents(texts)
        except Exception as e:
            self.logger.error(f"向量生成失败: {e}")
            return

        # 4. 组装记录
        records = self._build_records(metadata, embeddings, texts, content)
        if not records:
            self.logger.warning("构建记录为空")
            return

        # 5. 插入 Milvus
        try:
            milvus_client.insert(collection_name=self.collection_name, data=records)
            self.logger.info(f"写入成功: {len(entities)} 实体 + {len(relations)} 关系")
        except Exception as e:
            self.logger.error(f"Milvus 写入失败: {e}")

    def _ensure_collection(self, milvus_client):
        """集合不存在则创建"""
        if milvus_client.has_collection(self.collection_name):
            return

        schema = milvus_client.create_schema()
        schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("record_type", DataType.VARCHAR, max_length=65535)
        schema.add_field("node_type", DataType.VARCHAR, max_length=65535)
        schema.add_field("name", DataType.VARCHAR, max_length=65535)
        schema.add_field("source", DataType.VARCHAR, max_length=65535)
        schema.add_field("target", DataType.VARCHAR, max_length=65535)
        schema.add_field("relation", DataType.VARCHAR, max_length=65535)
        schema.add_field("item_name", DataType.VARCHAR, max_length=65535)
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=65535)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        milvus_client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        self.logger.info(f"集合 {self.collection_name} 创建成功")

    @staticmethod
    def _build_records(metadata: List[Dict], embeddings: Dict[str, Any],
                       texts: List[str], content: str = "") -> List[Dict]:
        """组装插入记录"""
        dense_list = embeddings.get("dense")
        sparse_matrix = embeddings.get("sparse")
        if dense_list is None or sparse_matrix is None:
            return []

        records = []
        for idx, meta in enumerate(metadata):
            if idx >= len(dense_list):
                break

            dense = dense_list[idx]
            if hasattr(dense, "tolist"):
                dense = dense.tolist()

            start = sparse_matrix.indptr[idx]
            end = sparse_matrix.indptr[idx + 1]
            indices = sparse_matrix.indices[start:end].tolist()
            data = sparse_matrix.data[start:end].tolist()
            sparse_dict = dict(zip(indices, data))

            record = {
                "text": texts[idx],
                "record_type": meta["record_type"],
                "node_type": meta.get("node_type", ""),
                "name": meta.get("name", ""),
                "source": meta.get("source", ""),
                "target": meta.get("target", ""),
                "relation": meta.get("relation", ""),
                "item_name": meta.get("item_name", ""),
                "chunk_id": meta.get("source_id", ""),
                "content": content,
                "dense_vector": dense,
                "sparse_vector": sparse_dict,
            }
            records.append(record)

        return records
