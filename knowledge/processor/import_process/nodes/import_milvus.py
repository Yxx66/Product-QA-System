from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import ValidationError
from pymilvus import MilvusClient, DataType


class import_milvus_node(BaseNode):
    name = "import_milvus"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("导入Milvus", "开始导入")

        # 1. 参数校验
        chunks, collection_name = self._validate_inputs(state)
        if not chunks:
            self.logger.warning("chunks 为空，跳过导入")
            return state

        # 2. 获取 Milvus 客户端
        client = self._get_client()

        # 3. 检查并创建集合
        self._ensure_collection(client, collection_name)

        # 4. 插入数据
        self._insert_chunks(client, collection_name, chunks)

        return state

    def _validate_inputs(self, state: ImportGraphState):
        self.log_step("step_1", "参数校验")
        chunks = state.get("chunks", [])
        if not isinstance(chunks, list):
            raise ValidationError("chunks 不是列表", node_name=self.name)

        config = get_config()
        collection_name = config.chunks_collection
        if not collection_name:
            raise ValidationError("chunks_collection 未配置", node_name=self.name)

        return chunks, collection_name

    def _get_client(self) -> MilvusClient:
        self.log_step("step_2", "获取 Milvus 客户端")
        from knowledge.tools.milvus_client import get_milvus_client
        return get_milvus_client()

    def _ensure_collection(self, client: MilvusClient, collection_name: str):
        self.log_step("step_3", f"检查集合 {collection_name}")
        if not client.has_collection(collection_name=collection_name):
            self._create_collection(client, collection_name)
        else:
            self.logger.info(f"集合 {collection_name} 已存在")

    def _create_collection(self, client: MilvusClient, collection_name: str):
        self.logger.info(f"创建集合: {collection_name}")

        schema = client.create_schema(enable_dynamic_fields=True)
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR,
                         is_primary=True, auto_id=True, max_length=100)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="IP"
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        self.logger.info(f"集合 {collection_name} 创建成功")

    def _insert_chunks(self, client: MilvusClient, collection_name: str, chunks: list):
        self.log_step("step_4", f"插入 {len(chunks)} 条数据")

        data = []
        for chunk in chunks:
            item = {
                "title": chunk.get("title", ""),
                "content": chunk.get("content", ""),
                "file_title": chunk.get("file_title", ""),
                "parent_title": chunk.get("parent_title", ""),
                "item_name": chunk.get("item_name", ""),
            }
            if "dense_vector" in chunk:
                item["dense_vector"] = chunk["dense_vector"]
            if "sparse_vector" in chunk:
                item["sparse_vector"] = chunk["sparse_vector"]
            data.append(item)

        result = client.insert(collection_name=collection_name, data=data)
        # 把 Milvus 自动生成的 pk 回写到 chunk，供下游图谱节点关联
        for i, pk_id in enumerate(result['ids']):
            chunks[i]['pk'] = pk_id
        self.logger.info(f"插入成功，ID: {result['ids'][:5]}...")


if __name__ == '__main__':
    from knowledge.processor import setup_logging
    from knowledge.processor.import_process.state import create_default_state
    setup_logging()

    node = import_milvus_node()

    # 测试连接和集合创建
    from knowledge.tools.milvus_client import get_milvus_client
    client = get_milvus_client()
    print(f"Milvus 连接成功，已有集合: {client.list_collections()}")

    # 测试插入（用假数据）
    test_state = create_default_state()
    test_state["chunks"] = [
        {
            "title": "## 产品简介",
            "content": "苏伯尔5000W大功率电磁炉",
            "file_title": "测试商品",
            "parent_title": "测试商品",
            "item_name": "苏伯尔电磁炉",
            "dense_vector": [0.1] * 1024,
            "sparse_vector": {100: 0.5, 200: 0.3},
        }
    ]
    result = node.process(test_state)
    print(f"插入完成，chunks 数量: {len(result.get('chunks', []))}")
