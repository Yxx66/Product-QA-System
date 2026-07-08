from dataclasses import dataclass, field
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import ValidationError

#Milvus类，负责封装milvus的所有操作
class kg_graph_node(BaseNode):
    name = "kg_graph"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        from knowledge.tools.milvus_client import get_milvus_client
        from knowledge.tools.milvus_graph_writer import MilvusGraphWriter
        from knowledge.tools.neo4j_graph_writer import Neo4jGraphWriter

        self.log_step("知识图谱", "开始构建")

        # 1. 参数校验
        validate_chunks = self._validate_inputs(state)
        stats = ProcessingStats(total_chunks=len(validate_chunks))
        item_name = validate_chunks[0].get("item_name", "")

        # 2. 初始化 writer
        config = get_config()
        milvus_client = get_milvus_client()
        milvus_writer = MilvusGraphWriter(collection_name=config.graph_collection)
        neo4j_writer = Neo4jGraphWriter(
            uri=config.neo4j_uri,
            username=config.neo4j_username,
            password=config.neo4j_password,
            database=config.neo4j_database,
        )

        # 3. 清理旧数据
        self._clear_exist_data(item_name, milvus_writer, neo4j_writer, milvus_client)

        # 4. 主线程预加载 BGE 模型（避免多线程加载冲突）
        from knowledge.tools.BGE3_client import get_bgem3_client
        get_bgem3_client()
        self.log_step("预加载", "BGE 模型加载完成")

        # 5. 处理切片
        self._process_chunks(validate_chunks, stats, milvus_writer, neo4j_writer, milvus_client)

        # 5. 关闭连接
        neo4j_writer.close()
        return state

    def _validate_inputs(self, state: ImportGraphState):
        self.log_step("step_1", "参数校验")
        chunks = state.get("chunks", [])
        if not chunks:
            raise ValidationError("chunks 为空", node_name=self.name)
        return chunks

    #清理老数据
    def _clear_exist_data(self, item_name: str, milvus_writer, neo4j_writer, milvus_client):
        """按 item_name 清理 Milvus 和 Neo4j 中的旧数据"""
        self.log_step("step_clear", f"清理 {item_name} 的旧数据")
        milvus_writer.clear(milvus_client, item_name)
        neo4j_writer.clear(item_name)

    def _process_chunks(self, chunks: list, stats, milvus_writer, neo4j_writer, milvus_client):
        self.log_step("step_2", f"处理 {len(chunks)} 个切片（多线程）")
        lock = threading.Lock()

        def _process_one(idx, chunk):
            content = chunk.get("content", "")
            item_name = chunk.get("item_name", "")
            source_id = chunk.get("pk", "") or chunk.get("chunk_id", "") or f"{item_name}_chunk_{idx}"

            # 1. LLM 提取
            llm_response = self._llm_extract(item_name, content)

            # 2. 清洗
            clean_data = self._clean_data(llm_response)
            entities = clean_data.get("entities", [])
            relations = clean_data.get("relations", [])

            # 3. 写入
            self._write_to_storage(entities, relations, item_name, source_id, content,
                                   milvus_writer, neo4j_writer, milvus_client)

            return len(entities), len(relations)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_process_one, idx, chunk): chunk for idx, chunk in enumerate(chunks)}
            for future in as_completed(futures):
                try:
                    entity_count, relation_count = future.result()
                    with lock:
                        stats.processed_chunks += 1
                        stats.total_entities += entity_count
                        stats.total_relations += relation_count
                except Exception as e:
                    with lock:
                        stats.failed_chunks += 1
                    self.logger.error(f"切片处理失败: {e}")

        self.logger.info(stats.summary())
    def _llm_extract(self, item_name: str, content: str) -> dict:
        """用 LLM 从 content 中提取实体和关系"""
        self.logger.info(f"LLM 提取 {item_name} 的实体和关系")
        from knowledge.tools.llm_client import get_llm_client
        from knowledge.tools.prompt.neo4j_prompt import GRAPH_RAG_PROMPT
        import json

        llm_client = get_llm_client()
        prompt = GRAPH_RAG_PROMPT.replace("{chunk_text}", content)

        try:
            resp = llm_client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "你是一个知识图谱实体关系提取助手，只返回JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            result_text = resp.choices[0].message.content.strip()
            print(f"LLM 原始返回:\n{result_text[:500]}")
            # 去除代码围栏
            import re
            result_text = re.sub(r"^```(?:json)?\s*", "", result_text)
            result_text = re.sub(r"\s*```$", "", result_text)
            result_text = result_text.strip()
            # 解析 JSON
            result = json.loads(result_text)
            return result
        except json.JSONDecodeError as e:
            self.logger.warning(f"LLM 返回的 JSON 解析失败: {e}")
            return {"nodes": [], "relationships": []}
        except Exception as e:
            self.logger.warning(f"LLM 提取失败: {e}")
            return {"nodes": [], "relationships": []}
    #清洗数据
    def _strip_fence(self, text: str) -> str:
        """去除文本中的代码围栏"""
        import re
        text = re.sub(r"```(?:\w+)?\s*", "", text)
        text = re.sub(r"```", "", text)
        text = re.sub(r"~~~(?:\w+)?\s*", "", text)
        text = re.sub(r"~~~", "", text)
        return text.strip()

    def _clean_data(self, llm_response: dict) -> dict:
        self.log_step("step_3", "清洗数据")
        raw_nodes = llm_response.get("nodes", [])
        raw_relations = llm_response.get("relationships", [])

        # 清洗节点：去围栏、去空名、去重（按 id 去重）
        seen_ids = set()
        entities = []
        for n in raw_nodes:
            node_id = self._strip_fence(str(n.get("id", "")))
            name = self._strip_fence(str(n.get("name", "")))
            ntype = self._strip_fence(str(n.get("type", "未知")))
            properties = n.get("properties", {})
            if not name or not node_id:
                continue
            if node_id in seen_ids:
                continue
            seen_ids.add(node_id)
            entities.append({"id": node_id, "name": name, "type": ntype, "properties": properties})

        # 清洗关系：去围栏、去空字段、去自环、去重
        seen_relations = set()
        relations = []
        for r in raw_relations:
            source = self._strip_fence(str(r.get("source", "")))
            target = self._strip_fence(str(r.get("target", "")))
            rtype = self._strip_fence(str(r.get("type", "")))
            properties = r.get("properties", {})
            if not source or not target or not rtype:
                continue
            if source == target:
                continue
            key = (source, target, rtype)
            if key in seen_relations:
                continue
            seen_relations.add(key)
            relations.append({"source": source, "target": target, "type": rtype, "properties": properties})

        self.logger.info(f"清洗完成: {len(entities)} 节点, {len(relations)} 关系")
        return {"entities": entities, "relations": relations}
    def _write_to_storage(self, entities, relations, item_name, source_id, content,
                          milvus_writer, neo4j_writer, milvus_client):
        """将实体和关系同时写入 Milvus 和 Neo4j"""
        self.log_step("step_4", "写入存储")
        if not entities and not relations:
            self.logger.warning("无实体和关系数据，跳过写入")
            return

        # 写入 Milvus（向量检索）
        milvus_writer.insert(milvus_client, entities, relations, item_name, source_id, content)

        # 写入 Neo4j（图谱检索）
        neo4j_writer.insert(entities, relations, item_name, source_id)




@dataclass
class ProcessingStats:
    """处理过程统计信息，用于日志和监控。"""

    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    total_entities: int = 0
    total_relations: int = 0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"处理完成: {self.processed_chunks}/{self.total_chunks} 切片成功, "
            f"{self.failed_chunks} 失败, "
            f"共 {self.total_entities} 实体 / {self.total_relations} 关系"
        )


if __name__ == '__main__':
    from knowledge.processor import setup_logging
    from knowledge.processor.import_process.state import create_default_state
    setup_logging()

    node = kg_graph_node()

    # 测试数据
    test_state = create_default_state()
    test_state["file_title"] = "万用表使用说明书"
    test_state["item_name"] = "RS-12万用表"
    test_state["chunks"] = [
        {
            "title": "## 产品简介",
            "content": "RS-12万用表是一款多功能数字万用表，支持直流电压测量、交流电压测量、电阻测量、电容测量等功能。配备HDMI接口和COM端口，适用于家庭和工业维修场景。",
            "file_title": "万用表使用说明书",
            "parent_title": "万用表使用说明书",
            "item_name": "RS-12万用表",
            "chunk_id": "chunk_001",
        },
        {
            "title": "## 安全警告",
            "content": "禁止超过500V进行电压测试。测试前必须断电。若COM端口电压超过500V，请勿进行电压测试，否则可能导致设备损坏或人身伤害。",
            "file_title": "万用表使用说明书",
            "parent_title": "万用表使用说明书",
            "item_name": "RS-12万用表",
            "chunk_id": "chunk_002",
        },
    ]

    try:
        result = node.process(test_state)
        print(f"\n{'='*50}")
        print(f"流程完成")
        print(f"chunks 数量: {len(result.get('chunks', []))}")
    except Exception as e:
        import traceback
        print(f"测试失败: {e}")
        traceback.print_exc()
