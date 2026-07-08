import logging
from typing import List, Dict


class Neo4jGraphWriter:
    """负责将知识图谱的节点和关系写入 Neo4j。"""

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self.logger = logging.getLogger(self.__class__.__name__)

    def close(self):
        """关闭连接"""
        self.driver.close()
        self.logger.info("Neo4j 连接已关闭")

    def clear(self, item_name: str):
        """删除某商品的所有节点和关系"""
        def _clear_tx(tx, item_name):
            result = tx.run(
                "MATCH (n {item_name: $item_name}) DETACH DELETE n",
                item_name=item_name
            )
            return result.consume()

        try:
            with self.driver.session(database=self.database) as session:
                session.execute_write(_clear_tx, item_name)
            self.logger.info(f"已清理 {item_name} 的图谱数据")
        except Exception as e:
            self.logger.warning(f"清理图谱数据失败: {e}")

    def insert(self, entities: List[Dict], relations: List[Dict], item_name: str, source_id: str = ""):
        """将节点和关系写入 Neo4j"""
        if not entities and not relations:
            self.logger.warning("无节点和关系数据，跳过写入")
            return

        try:
            with self.driver.session(database=self.database) as session:
                # 1. 创建节点
                node_count = session.execute_write(
                    self._create_nodes_tx, entities, item_name, source_id
                )
                # 2. 创建关系
                rel_count = session.execute_write(
                    self._create_relations_tx, relations, source_id
                )
            self.logger.info(f"写入成功: {node_count} 节点 + {rel_count} 关系")
        except Exception as e:
            self.logger.error(f"Neo4j 写入失败: {e}")

    @staticmethod
    def _create_nodes_tx(tx, entities: List[Dict], item_name: str, source_id: str = "") -> int:
        """在一个事务中创建所有节点，并关联到 Chunk 节点"""
        # 1. 创建 Chunk 节点
        tx.run(
            "MERGE (c:Chunk {id: $source_id}) SET c.item_name = $item_name",
            source_id=source_id, item_name=item_name
        )

        # 2. 创建实体节点并关联 Chunk
        count = 0
        for entity in entities:
            node_type = entity.get("type", "Concept")
            node_type = "".join(c for c in node_type if c.isalnum() or c == "_")
            if not node_type:
                node_type = "Concept"

            properties = entity.get("properties", {})
            properties_str = ""
            if properties:
                props = ", ".join(f"n.{k} = ${k}" for k in properties.keys())
                properties_str = f"SET {props}"

            query = f"""
            MERGE (n:{node_type} {{id: $id}})
            SET n.name = $name, n.item_name = $item_name, n.source_id = $source_id
            {properties_str}
            WITH n
            MATCH (c:Chunk {{id: $source_id}})
            MERGE (n)-[:BELONGS_TO]->(c)
            """

            params = {
                "id": entity.get("id", ""),
                "name": entity.get("name", ""),
                "item_name": item_name,
                "source_id": source_id,
                **properties,
            }
            tx.run(query, **params)
            count += 1
        return count

    @staticmethod
    def _create_relations_tx(tx, relations: List[Dict], source_id: str = "") -> int:
        """在一个事务中创建所有关系"""
        count = 0
        for rel in relations:
            rel_type = rel.get("type", "RELATED_TO")
            rel_type = "".join(c for c in rel_type if c.isalnum() or c == "_")
            if not rel_type:
                rel_type = "RELATED_TO"

            query = f"""
            MATCH (a {{id: $source}})
            MATCH (b {{id: $target}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r.source_id = $source_id
            """

            tx.run(query, source=rel["source"], target=rel["target"], source_id=source_id)
            count += 1
        return count

    def verify_connectivity(self):
        """验证连接"""
        self.driver.verify_connectivity()
        self.logger.info("Neo4j 连接成功")
