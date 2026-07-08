"""
知识图谱查询节点（第一天）

业务逻辑：
1. 从 state 取出 rewritten_query 和 item_names
2. 移除问题中的商品名（避免干扰实体抽取）
3. 调 LLM 抽取实体关键词
4. 在 Milvus ENTITY_NAME_COLLECTION 中对齐已入库实体
5. 结果写入 state

复用：
- get_llm_client() → knowledge/tools/llm_client.py
- get_bgem3_client() → knowledge/tools/BGE3_client.py
- get_milvus_client() → knowledge/tools/milvus_client.py
- AnnSearchRequest + hybrid_search → 参考 item_name_confirm.py
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional

from pymilvus import AnnSearchRequest, Function, FunctionType
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.config import get_config

logger = logging.getLogger(__name__)

# ==================== 常量 ====================
MAX_ENTITY_NAME_LENGTH = 15
ENTITY_ALIGN_TOP_K = 3

ALLOWED_ENTITY_LABELS_CN = (
    "设备(Device)、部件(Component)、功能(Function)、步骤(Procedure)、"
    "参数(Parameter)、故障(Issue)、安全规则(SafetyRule)、概念(Concept)"
)

_ENTITY_EXTRACT_SYSTEM_PROMPT = f"""
你是一个知识图谱问答系统的"实体识别"模块。
请从用户问题中抽取用于查询图数据库(Neo4j)的实体名称。

【图谱中存在的实体类型】
{ALLOWED_ENTITY_LABELS_CN}

【约束】
1) 优先抽取上述类型的名词短语
2) 每个实体名称不超过 {MAX_ENTITY_NAME_LENGTH} 个字符，超过请截取核心部分
3) 不要输出完整句子，只输出实体关键词
4) 输出必须是严格 JSON，只含一个字段 entities（字符串数组）

【输出示例】
{{"entities": ["电池安装", "螺丝刀", "表笔"]}}
"""


# ==================== 工具函数 ====================

def _truncate_entity_name(name: str) -> str:
    """截断实体名至 MAX_ENTITY_NAME_LENGTH 字符"""
    name = name.strip()
    return name[:MAX_ENTITY_NAME_LENGTH] if len(name) > MAX_ENTITY_NAME_LENGTH else name


def _parse_entity_json(llm_response: str) -> List[str]:
    """解析 LLM 返回的实体 JSON，截断 + 去重"""
    if not llm_response:
        return []
    text = llm_response.strip()

    # 清洗 markdown 代码围栏
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"实体 JSON 解析失败: {e} | 片段: {text[:80]}")
        return []

    raw = data.get("entities", [])
    if not isinstance(raw, list):
        return []

    seen = set()
    result = []
    for item in raw:
        if not isinstance(item, str):
            continue
        name = _truncate_entity_name(item)
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return result


# ==================== 内部服务类 ====================

class _EntityExtractor:
    """调用 LLM 从问题中抽取实体关键词"""

    def __init__(self, llm_client):
        self._llm = llm_client
        self._logger = logging.getLogger(self.__class__.__name__)

    def extract(self, question: str) -> List[str]:
        """抽取实体。失败降级返回空列表。"""
        if not question or not question.strip():
            return []

        messages = [
            {"role": "system", "content": _ENTITY_EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]

        try:
            response = self._llm.chat.completions.create(
                model="qwen-plus",
                messages=messages,
                temperature=0.1
            )
            llm_response = response.choices[0].message.content
        except Exception as e:
            self._logger.error(f"LLM 抽取实体失败: {e}")
            return []

        entities = _parse_entity_json(llm_response)
        if not entities:
            self._logger.warning("LLM 抽取实体返回空列表")
        return entities


class _EntityAligner:
    """将 LLM 抽取的实体与 Milvus 中已入库实体做向量对齐"""

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def align(self, entities: List[str], item_names: List[str]) -> Dict[str, Any]:
        """
        对齐入口。
        返回: {"aligned_entities": [...], "alignments": [...]}
        """
        if not entities or not item_names:
            return {"aligned_entities": [], "alignments": []}

        from knowledge.tools.BGE3_client import get_bgem3_client
        from knowledge.tools.milvus_client import get_milvus_client

        bge3 = get_bgem3_client()
        milvus_client = get_milvus_client()
        if not bge3 or not milvus_client:
            return {"aligned_entities": [], "alignments": []}

        # 批量向量化 entities
        embeddings = bge3.encode_documents(entities)
        dense_list = embeddings["dense"]
        sparse_matrix = embeddings["sparse"]

        # 提取 sparse 向量
        sparse_vectors = []
        for i in range(len(entities)):
            start = sparse_matrix.indptr[i]
            end = sparse_matrix.indptr[i + 1]
            indices = sparse_matrix.indices[start:end].tolist()
            data = sparse_matrix.data[start:end].tolist()
            sparse_vectors.append(dict(zip(indices, data)))

        # 逐个实体做混合检索
        collection_name = "knowledge_graph"
        aligned_entities = []
        alignments = []

        for idx, entity in enumerate(entities):
            result = self._align_one(
                entity, idx, dense_list, sparse_vectors,
                collection_name, milvus_client, item_names
            )
            aligned_entities.append(result.get("aligned_name", entity))
            alignments.append(result)

        # 过滤掉没有对齐上的（aligned_name 为空的保留原始名）
        return {"aligned_entities": aligned_entities, "alignments": alignments}

    def _align_one(self, entity: str, idx: int, dense_list, sparse_vectors,
                   collection_name: str, milvus_client, item_names: List[str]) -> Dict[str, Any]:
        """对齐单个实体，返回最佳匹配"""
        dense_vec = dense_list[idx].tolist() if hasattr(dense_list[idx], 'tolist') else dense_list[idx]
        sparse_vec = sparse_vectors[idx]

        # 构建 item_name 过滤条件
        names_str = ", ".join(f'"{n}"' for n in item_names)
        item_filter = f"item_name in [{names_str}]"

        request_dense = AnnSearchRequest(
            data=[dense_vec],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=ENTITY_ALIGN_TOP_K,
            expr=item_filter,
        )
        request_sparse = AnnSearchRequest(
            data=[sparse_vec],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=ENTITY_ALIGN_TOP_K,
            expr=item_filter,
        )

        ranker = Function(
            name="rrf",
            input_field_names=[],
            function_type=FunctionType.RERANK,
            params={"reranker": "rrf", "k": 60},
        )

        results = milvus_client.hybrid_search(
            collection_name=collection_name,
            reqs=[request_dense, request_sparse],
            ranker=ranker,
            limit=ENTITY_ALIGN_TOP_K,
            output_fields=["name", "item_name"],
        )

        # 取最佳匹配
        best = None
        best_score = 0.0
        item_name = ""
        for hits in results:
            for hit in hits:
                score = hit.get("distance", 0.0)
                if score > best_score:
                    best_score = score
                    entity_data = hit.get("entity", {})
                    best = entity_data.get("name", "")
                    item_name = entity_data.get("item_name", "")

        return {
            "original": entity,
            "aligned_name": best or entity,
            "score": round(best_score, 4),
            "item_name":item_name
        }


# ==================== 第二天：Neo4j 查询 + Chunk 回填 ====================

WEIGHT_SEED = 2.0
WEIGHT_NEIGHBOR = 1.0

# Cypher 语句
_CYPHER_EXACT_SEED = """
    MATCH (n)
    WHERE n.item_name = $item_name AND n.name = $entity_name
    RETURN n.name AS name, n.item_name AS item_name
"""

_CYPHER_FUZZY_SEED = """
    MATCH (n)
    WHERE toLower(n.name) CONTAINS toLower($entity_name)
      AND n.item_name = $item_name
    RETURN n.name AS name, n.item_name AS item_name
    LIMIT $limit
"""

_CYPHER_ONE_HOP = """
    MATCH (seed {item_name: $item_name, name: $entity_name})-[r]-(nbr)
    WHERE type(r) <> 'BELONGS_TO' AND nbr.item_name = $item_name
    RETURN
        CASE WHEN startNode(r) = seed THEN seed.name ELSE nbr.name END AS head,
        type(r) AS rel,
        CASE WHEN startNode(r) = seed THEN nbr.name ELSE seed.name END AS tail
    LIMIT $limit
"""

_CYPHER_LOOKUP_CHUNK = """
    UNWIND $nodes_with_weight AS n
    MATCH (e {name: n.entity_name, item_name: n.item_name})
          -[:BELONGS_TO]->(c:Chunk {item_name: n.item_name})
    WITH c, sum(n.weight) AS score, count(e) AS cnt
    RETURN c.id AS source_id, c.item_name AS item_name, score, cnt
    ORDER BY score DESC, cnt DESC, source_id ASC
    LIMIT $limit
"""


def _extract_entity_item_pairs(alignment_results: List[Dict]) -> List[Dict]:
    """从对齐结果中提取 (item_name, entity_name) 配对，去重"""
    # 过滤掉 aligned_name 为空的，
    seen=set()
    results=[]
    if not alignment_results:
        return []
    for result in alignment_results:
        aligned_name = result.get("aligned_name", "")
        item_name = result.get("item_name", "")
        if not aligned_name or not item_name:
            continue
    # key = (item_name, aligned_name) 去重
        key=(item_name,aligned_name)
        if key not in seen:
            seen.add(key)
            results.append({"item_name":item_name,"entity_name":aligned_name})

    return results



def _one_hop_triples_to_texts(triples: List[Dict]) -> List[str]:
    """三元组转文本：[商品名] head -(rel)-> tail"""
    if not triples:
        return []
    texts = []
    for triple in triples:
        item = triple.get("item_name", "")
        head = triple.get("head", "")
        rel = triple.get("rel", "")
        tail = triple.get("tail", "")
        text = f"[{item}] {head} -({rel})-> {tail}"
        texts.append(text)
    return texts



class _Neo4jGraphReader:
    """Neo4j 图谱读取：种子节点查询 + 一跳关系 + chunk 反查"""

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self._logger = logging.getLogger(self.__class__.__name__)

    def find_seed_nodes(self, entity_item_pairs: List[Dict], max_per_node: int = 3,) -> List[Dict]:
        """种子节点查询：精确匹配优先，模糊匹配降级"""
        results=[]
        if not entity_item_pairs:
            self._logger.info("entity_item_pairs is empty")
            return []
        for pair in entity_item_pairs:
            item_name=pair.get("item_name","")
            entity_name=pair.get("entity_name","")
            if not item_name or not entity_name:
                continue
            with self.driver.session(database=self.database) as session:
                result=session.run(_CYPHER_EXACT_SEED,item_name=item_name,entity_name=entity_name).data()
        # 精确匹配无结果时降级 _CYPHER_FUZZY_SEED
                if result==[]:
                    result=session.run(_CYPHER_FUZZY_SEED,item_name=item_name,entity_name=entity_name,limit=max_per_node).data()
            results.extend(result)
        return results


    def find_one_hop_relations(self, seed_nodes: List[Dict], max_per_seed: int = 50) -> List[Dict]:
        """一跳关系扩展：双向查询 + startNode(r) 保留方向 + 过滤 MENTIONED_IN"""
        # TODO: 遍历 seed_nodes，执行 _CYPHER_ONE_HOP
        results=[]
        seen = set()
        if not seed_nodes:
            self._logger.info("seed_nodes is empty")
            return []
        for seed in seed_nodes:
            item_name=seed.get("item_name","")
            entity_name=seed.get("name","")
            if not item_name or not entity_name:
                continue
            with self.driver.session(database=self.database) as session:
                result=session.run(_CYPHER_ONE_HOP,item_name=item_name,entity_name=entity_name,limit=max_per_seed).data()
        # 用 (item_name, head, rel, tail) 去重
            for t in result:  # ← 加这层循环
                key = (item_name, t.get("head", ""), t.get("rel", ""), t.get("tail", ""))
                if key not in seen:
                    seen.add(key)
                    t["item_name"] = item_name
                    results.append(t)
        if len(results) > 10:
            return results[:10]
        return results


    def find_chunk_ids(self, seed_nodes: List[Dict], one_hop_triples: List[Dict], limit: int = 200) -> List[Dict]:
        """加权 chunk 反查：种子 2.0 / 邻居 1.0，UNWIND 聚合打分"""
        nodes_with_weight = self._collect_nodes_with_weight(seed_nodes, one_hop_triples)
        if not nodes_with_weight:
            return []
        with self.driver.session(database=self.database) as session:
            result = session.run(_CYPHER_LOOKUP_CHUNK, nodes_with_weight=nodes_with_weight, limit=limit).data()
        return result

    def _collect_nodes_with_weight(self, seeds, triples) -> List[Dict]:
        """种子 2.0，邻居 1.0，种子优先不覆盖"""
        weight_map = {}
        # 第1步：遍历 seeds，放权重 2.0
        for seed in seeds:
            name = seed.get("name", "")
            item_name = seed.get("item_name", "")
            if not name or not item_name:
                continue
            key = (item_name, name)
            if key not in weight_map:
                weight_map[key] = WEIGHT_SEED
        # 第2步：遍历 triples，head 和 tail 分别放权重 1.0
        for triple in triples:
            item = triple.get("item_name", "")
            head = triple.get("head", "")
            tail = triple.get("tail", "")
            for node_name in [head, tail]:
                if not node_name or not item:
                    continue
                key = (item, node_name)
                if key not in weight_map:
                    weight_map[key] = WEIGHT_NEIGHBOR
        # 第3步：字典转列表
        nodes_with_weight = []
        for (item_name, entity_name), weight in weight_map.items():
            nodes_with_weight.append({"item_name": item_name, "entity_name": entity_name, "weight": weight})
        return nodes_with_weight


    def close(self):
        self.driver.close()


class _ChunkBackfiller:
    """从 Milvus chunks_collection 回填 chunk 内容"""

    def __init__(self, collection_name: str):
        self._collection_name = collection_name
        self._logger = logging.getLogger(self.__class__.__name__)

    def backfill(self, chunk_hits: List[Dict]) -> List[Dict]:
        """按 source_id 从 Milvus 批量回填 content，保持得分排序"""
        source_ids = [hit.get("source_id", "") for hit in chunk_hits]
        if not source_ids:
            self._logger.info("source_ids is empty")
            return []
        from knowledge.tools.milvus_client import get_milvus_client
        client = get_milvus_client()
        results = client.get(self._collection_name, source_ids, output_fields=["pk", "content", "item_name"])
        content_map = {r["pk"]: r.get("content", "") for r in results}
        for hit in chunk_hits:
            hit["content"] = content_map.get(hit.get("source_id"), "")
        return chunk_hits



# ==================== 主节点 ====================

class KGQueryNode(BaseNode):
    """知识图谱查询节点"""
    name = "kg_query"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """完整 pipeline：解析 → 抽取 → 对齐 → 配对 → Neo4j查询 → 回填 → 只返回 chunks/triples"""
        self.log_step("知识图谱查询", "开始")

        # 1. 解析输入
        question, item_names = self._parse_input(state)
        if not question:
            self.logger.warning("问题为空，跳过")
            return {"kg_chunks": [], "kg_triples": []}

        # 2. LLM 抽取实体（第一天）
        from knowledge.tools.llm_client import get_llm_client
        extractor = _EntityExtractor(get_llm_client())
        entities = extractor.extract(question)
        self.log_step("实体抽取", f"抽到 {len(entities)} 个: {entities}")

        # 3. Milvus 实体对齐（第一天）
        aligner = _EntityAligner()
        align_result = aligner.align(entities, item_names)
        self.log_step("实体对齐", f"对齐后 {len(align_result.get('aligned_entities', []))} 个")

        # 4. 提取配对（第二天）
        entity_item_pairs = _extract_entity_item_pairs(align_result.get("alignments", []))
        self.log_step("配对提取", f"{len(entity_item_pairs)} 对")

        # 5. Neo4j 种子节点查询（第二天）
        config = get_config()
        neo4j_reader = _Neo4jGraphReader(
            uri=config.neo4j_uri,
            username=config.neo4j_username,
            password=config.neo4j_password,
            database=config.neo4j_database,
        )
        seed_nodes = neo4j_reader.find_seed_nodes(entity_item_pairs)
        self.log_step("种子节点", f"{len(seed_nodes)} 个")

        # 6. 一跳关系扩展（第二天）
        one_hop_triples = neo4j_reader.find_one_hop_relations(seed_nodes)
        self.log_step("一跳关系", f"{len(one_hop_triples)} 条")

        # 7. 加权 chunk 反查（第二天）
        chunk_hits = neo4j_reader.find_chunk_ids(seed_nodes, one_hop_triples)
        self.log_step("chunk 反查", f"{len(chunk_hits)} 条")

        # 8. Milvus chunk 回填（第二天）
        backfiller = _ChunkBackfiller(collection_name=config.chunks_collection)
        kg_chunks = backfiller.backfill(chunk_hits)
        self.log_step("chunk 回填", f"{len(kg_chunks)} 条")
        # 9. 三元组转文本（第二天）
        triples_docs = _one_hop_triples_to_texts(one_hop_triples)

        neo4j_reader.close()
        self.log_step("知识图谱查询", "完成")
        return {"kg_chunks": kg_chunks, "kg_triples": triples_docs}

    @staticmethod
    def _parse_input(state: QueryGraphState) -> tuple:
        """从 state 提取 question 和 item_names，移除问题中的商品名"""
        question = state.get("rewritten_query") or state.get("original_query", "")
        item_names = state.get("item_names", [])

        for name in item_names:
            if name in question:
                question = question.replace(name, "")
            else:
                found = False
                for length in range(len(name), 4, -1):
                    for i in range(len(name) - length + 1):
                        sub = name[i:i + length]
                        if sub in question:
                            question = question.replace(sub, "")
                            found = True
                            break
                    if found:
                        break

        return question.strip(), item_names


# ==================== 测试 ====================

if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging
    from knowledge.processor.query_process.state import create_default_state
    from knowledge.processor.query_process.nodes.item_name_confirm import ItemNameConfirm
    setup_logging()

    # 1. 先跑 ItemNameConfirm 确认商品名
    print("=" * 50)
    print("Step 1: 商品名确认")
    print("=" * 50)

    test_state = create_default_state()
    test_state["original_query"] = "华为家的B3-211H显示器的底座怎么安装？"

    confirm_node = ItemNameConfirm()
    test_state = confirm_node.process(test_state)

    print(f"  item_names: {test_state.get('item_names')}")
    print(f"  rewritten_query: {test_state.get('rewritten_query')}")

    if not test_state.get("item_names"):
        print("  未识别到商品名，退出")
        exit()

    # 2. 再跑 KGQueryNode
    print(f"\n{'='*50}")
    print("Step 2: 知识图谱查询")
    print("=" * 50)

    kg_node = KGQueryNode()
    result = kg_node.process(test_state)

    print(f"\n--- 实体抽取 ---")
    entities = result.get("kg_entities", [])
    print(f"抽到 {len(entities)} 个: {entities}")
    print(f"\n--- 实体对齐 ---")
    aligned = result.get("kg_aligned_entities", [])
    alignments = result.get("kg_alignments", [])
    print(f"对齐后 {len(aligned)} 个: {aligned}")
    for a in alignments:
        print(f"  {a.get('original')} -> {a.get('aligned_name')} (score: {a.get('score')}, item: {a.get('item_name')})")
    print(f"\n--- 一跳关系 ---")
    triples = result.get("kg_triples", [])
    print(f"{len(triples)} 条")
    for t in triples:
        print(f"  {t}")
    print(f"\n--- KG Chunks ---")
    chunks = result.get("kg_chunks", [])
    print(f"{len(chunks)} 条")
    for c in chunks[:3]:
        print(f"  {c.get('source_id', '')}: {c.get('content', '')[:80]}...")
    print("........................................................................")
    print(test_state.get("kg_chunks"))
