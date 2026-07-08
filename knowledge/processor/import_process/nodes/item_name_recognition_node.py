from knowledge.processor import setup_logging
from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Optional, List

from langgraph_sdk.schema import Config
from pymilvus import DataType

setup_logging()


class item_name_recognition_node(BaseNode):
    name = "item_name_recognition"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("项目名称识别", "开始识别")
        #第一步，参数校验
        file_title, chunks = self._validate_inputs(state)
        #第二步，调用模型识别
        item_name = self._call_model_item_name(file_title, chunks)
        #第三步，回填到每个chunk
        self._back_fill_item_name(state, item_name)
        #第四步，生成向量
        dense_vector, sparse_vector = self._generate_vector(state, item_name)
        #第五步，保存到milvus
        self._save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector, self.config)
        return state

    def _validate_inputs(self, state: ImportGraphState):
        self.log_step("step1", "参数校验")
        file_title = state.get("file_title", "")
        if not file_title:
            raise ValidationError("文件标题为空", node_name="item_name_recognition")
        chunks = state.get("chunks", [])
        if not chunks:
            raise ValidationError("chunks 为空", node_name="item_name_recognition")
        return file_title, chunks

    #构建上下文
    def _build_prompt(self, file_title: str, chunks: list, k: int, max_chars: int = 800):
        parts = []
        total = 0

        for i, chunk in enumerate(chunks[:k]):
            if not isinstance(chunk, dict):
                raise ValidationError(f"chunks[{i}] 不是字典", node_name="item_name_recognition")
            title = chunk.get("title", "")
            content = chunk.get("content", "")
            if not title or not content:
                continue
            if len(content) > max_chars:
                new_content = content[:max_chars] + "..."
            else:
                new_content = content
            #组装title+content
            part = f"切片{i+1}:标题:{title}\n内容:{new_content}"
            parts.append(part)
            total += len(part)
            if total > max_chars:
                break
        return parts

    #调用模型识别商品名字
    def _call_model_item_name(self, file_title, chunks):
        from knowledge.tools.llm_client import get_llm_client
        self.log_step("step2", "获取模型并调用")

        parts = self._build_prompt(file_title, chunks, 10)
        prompt = f"""文件名:{file_title}
切片:{parts}

要求：
1. 返回内容为字符形式，最好是带品牌、型号和名称的完整商品名称。比如：苏伯尓5000W大功率电磁炉；
2. 返回结果应该只包含商品名称，不要添加任何解释或其他内容；
3. 如果无法识别商品名称,请返回空字符串。"""

        try:
            llm_client = get_llm_client()
            resp = llm_client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "你是一个商品名称识别的助手。"},
                    {"role": "user", "content": prompt},
                ]
            )
            item_name = resp.choices[0].message.content.strip()
            if not item_name:
                self.logger.info("模型识别结果为空，使用文件名作为商品名称")
                item_name = file_title
            return item_name
        except Exception:
            self.logger.exception("模型调用失败")
            return file_title

    def _back_fill_item_name(self, state: ImportGraphState, item_name: str):
        """将识别到的商品名称回填到每个 chunk"""
        self.log_step("step3", "回填商品名称到 chunks")
        state["item_name"] = item_name
        for chunk in state.get("chunks", []):
            chunk["item_name"] = item_name
        return state
    #生成向量
    def _generate_vector(self,state:ImportGraphState, item_name:str):
        self.log_step("step4", "生成向量")
        from knowledge.tools.BGE3_client import get_bgem3_client
        try:
            bge_m3_ef = get_bgem3_client()
            vectors = bge_m3_ef.encode_documents([item_name])

            if vectors:
                dense_vector = vectors["dense"][0].tolist()

                # 提取稀疏向量
                start_idx = vectors["sparse"].indptr[0]
                end_idx = vectors["sparse"].indptr[1]
                token_ids = vectors["sparse"].indices[start_idx:end_idx].tolist()
                weights = vectors["sparse"].data[start_idx:end_idx].tolist()
                sparse_vector = dict(zip(token_ids, weights))

                self.logger.info("向量生成成功")
                return dense_vector, sparse_vector

        except Exception as e:
            self.logger.warning(f"向量生成失败: {e}")

        return None, None

    def _save_to_milvus(
            self,
            state: ImportGraphState,
            file_title: str,
            item_name: str,
            dense_vector: Optional[List[float]],
            sparse_vector: Optional[dict],
            config:Config
    ):
        """保存到 Milvus"""
        self.log_step("step_6", "保存到 Milvus")
        from knowledge.processor.import_process.config import get_config
        config=get_config()
        if not config.milvus_url or not config.item_name_collection:
            self.logger.warning("Milvus 配置不完整，跳过保存")
            return

        try:
            # 1. 获取 Milvus 客户端
            from knowledge.tools.milvus_client import get_milvus_client
            client = get_milvus_client()

            # 2. 获取集合名字
            collection_name = "item_name_collection_test"

            # 3. 检查并创建集合
            if not client.has_collection(collection_name=collection_name):
                self._create_item_name_collection(client, collection_name)

            # 4. 准备数据
            data = {
                "file_title": file_title,
                "item_name": item_name
            }

            # 5. 构建稠密向量
            if dense_vector is not None:
                data["dense_vector"] = dense_vector

            # 6. 构建稀疏向量
            if sparse_vector is not None:
                data["sparse_vector"] = sparse_vector

            # 7. 插入数据
            result = client.insert(collection_name=collection_name, data=[data])
            self.logger.info(f"已保存到 Milvus，ID: {result['ids'][0]}")

            state["item_name"] = item_name

        except Exception as e:
            self.logger.warning(f"Milvus 保存失败: {e}")

    def _create_item_name_collection(self, client, collection_name: str):
        """创建 item_name 集合"""
        self.logger.info(f"创建集合: {collection_name}")

        # 1. 定义字段
        schema = client.create_schema(enable_dynamic_field=True)

        schema.add_field(field_name="pk", datatype=DataType.VARCHAR,
                         is_primary=True, auto_id=True, max_length=100)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 2. 创建索引
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

        # 3. 创建集合
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        self.logger.info(f"集合 {collection_name} 创建成功")



if __name__ == '__main__':
    node = item_name_recognition_node()

    print("=" * 50)
    print("测试1: 正常 chunk")
    print("=" * 50)
    result = node._build_prompt("电磁炉说明书", [{"title": "第一章", "content": "这是一款苏泊尔5000W大功率电磁炉"}], 3)
    print(result)

    print("\n" + "=" * 50)
    print("测试2: 空 title/content 被跳过")
    print("=" * 50)
    result = node._build_prompt("测试", [
        {"title": "", "content": "有内容但没标题"},
        {"title": "有标题", "content": ""},
        {"title": "正常标题", "content": "正常内容"},
    ], 5)
    print(result)

    print("\n" + "=" * 50)
    print("测试3: 超长 content 截断")
    print("=" * 50)
    long_content = "A" * 1000
    result = node._build_prompt("测试", [{"title": "标题", "content": long_content}], 1, max_chars=50)
    print(result)

    print("\n" + "=" * 50)
    print("测试4: 多个 chunk 累计")
    print("=" * 50)
    result = node._build_prompt("测试", [
        {"title": f"标题{i}", "content": f"内容{i}"} for i in range(10)
    ], 5, max_chars=200)
    print(f"共 {len(result)} 个切片")
    for p in result:
        print(p)

    print("\n" + "=" * 50)
    print("测试5: _back_fill_item_name 回填")
    print("=" * 50)
    from knowledge.processor.import_process.state import create_default_state
    test_state = create_default_state()
    test_state["file_title"] = "测试商品"
    test_state["chunks"] = [
        {"title": "第一章", "content": "产品介绍", "file_title": "测试商品", "parent_title": "测试商品"},
        {"title": "第二章", "content": "规格参数", "file_title": "测试商品", "parent_title": "测试商品"},
    ]
    node._back_fill_item_name(test_state, "苏泊尔电磁炉")
    print(f"state['item_name']: {test_state['item_name']}")
    for i, chunk in enumerate(test_state["chunks"]):
        print(f"chunk[{i}]['item_name']: {chunk.get('item_name')}")

    print("\n" + "=" * 50)
    print("测试6: Milvus 连接和集合创建")
    print("=" * 50)
    try:
        from knowledge.tools.milvus_client import get_milvus_client
        client = get_milvus_client()
        print(f"Milvus 连接成功")
        collections = client.list_collections()
        print(f"已有集合: {collections}")

        test_collection = "item_name_collection_test"
        if not client.has_collection(collection_name=test_collection):
            node._create_item_name_collection(client, test_collection)
            print(f"集合 {test_collection} 创建成功")
        else:
            print(f"集合 {test_collection} 已存在")
    except Exception as e:
        print(f"Milvus 测试失败: {e}")

    print("\n" + "=" * 50)
    print("测试7: process 完整流程（会调 LLM）")
    print("=" * 50)
    test_state2 = create_default_state()
    test_state2["file_title"] = "苏伯尔5000W大功率电磁炉使用说明书"
    test_state2["chunks"] = [
        {"title": "## 产品简介", "content": "本产品为苏伯尔品牌5000W大功率电磁炉，适用于商用厨房和家庭大功率烹饪需求。", "file_title": "苏伯尔电磁炉", "parent_title": "苏伯尔电磁炉"},
        {"title": "## 规格参数", "content": "额定功率：5000W，额定电压：220V，产品尺寸：450x350x80mm，净重：5.2kg。", "file_title": "苏伯尔电磁炉", "parent_title": "苏伯尔电磁炉"},
        {"title": "## 使用方法", "content": "1. 将电磁炉放置在平稳台面上 2. 接通电源 3. 按下开关按钮 4. 调节火力档位。", "file_title": "苏伯尔电磁炉", "parent_title": "苏伯尔电磁炉"},
    ]
    try:
        result_state = node.process(test_state2)
        print(f"识别结果: {result_state['item_name']}")
        for i, chunk in enumerate(result_state["chunks"]):
            print(f"chunk[{i}]['item_name']: {chunk.get('item_name')}")
    except Exception as e:
        print(f"process 失败: {e}")
