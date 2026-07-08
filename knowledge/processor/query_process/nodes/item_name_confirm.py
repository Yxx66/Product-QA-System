from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
import re
import json
from knowledge.tools.BGE3_client import get_bgem3_client
from pymilvus import AnnSearchRequest, Function, FunctionType
from typing import List

# ★ 新增：MongoDB 历史记录工具
from knowledge.utils.mongo_history_utils import (
    get_recent_messages,
    update_message_item_names,
)

system_prompt = """
你是一个商品名称提取器，我会给你一个查询，你需要从查询中提取出商品名称，并返回一个json格式的字符串，格式如下：
{
    "item_names": ["商品名称1", "商品名称2", ...],
    "rewritten_query": "重写后的查询"
}
"""

MID_CONFIDENCE = 0.02
HIGH_CONFIDENCE = 0.03
MAX_OPTIONS = 3
SCORE_GAP_THRESHOLD = 0.1


class ItemNameConfirm(BaseNode):
    """商品名称确认节点"""
    name = "item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """完整流程：MongoDB历史读取 → LLM提取 → 向量检索确认 → 决策 → item_names回填"""
        self.log_step("开始商品名称确认")

        session_id = state.get("session_id", "")
        query = state.get("original_query", "")

        # ★ Step 0: 从 MongoDB 读取最近的历史对话（用于 LLM 指代消解）
        if session_id:
            chat_history = get_recent_messages(session_id, limit=10)
        else:
            chat_history = state.get("history", [])

        # Step 1: LLM 提取商品名称（传入历史对话做指代消解）
        extractor = ItemExtractor()
        extract_result = extractor.extract_item_name(query, chat_history)
        item_names = extract_result.get("item_names", [])
        rewritten_query = extract_result.get("rewritten_query", "")

        if not item_names:
            self.log_step("LLM未提取到商品名称，使用原始查询")
            state["item_names"] = []
            # ★ 即使未提取到商品名，仍将历史写入 state 供下游使用
            state["history"] = chat_history
            return state

        self.log_step("LLM提取完成", f"提取到 {item_names}")

        # Step 2: 向量检索 + 评分对齐
        collection_name = "item_name_collection_test"
        confirmed, options = extractor.match_align(item_names, collection_name)

        self.log_step("匹配完成", f"确认: {confirmed}, 候选: {options}")

        # Step 3: 决策
        extractor._decide(state, confirmed, options, rewritten_query, chat_history)

        # ★ Step 4: 回填历史记录中空的 item_names
        if confirmed:
            ids_to_update = [
                str(msg["_id"])
                for msg in chat_history
                if not msg.get("item_names")
            ]
            if ids_to_update:
                try:
                    updated_count = update_message_item_names(ids_to_update, confirmed)
                    if updated_count:
                        self.log_step(
                            "item_names回填",
                            f"更新了 {updated_count} 条历史记录 → {confirmed}",
                        )
                except Exception as e:
                    self.logger.warning(f"回填历史 item_names 失败: {e}")

        # ★ 将历史对话写入 state，供下游 answer_output 使用
        state["history"] = chat_history

        self.log_step("商品名称确认完成")
        return state


class ItemExtractor:
    """调用LLM提取商品名称，向量检索确认"""

    def extract_item_name(self, query: str, history) -> dict:
        """调用LLM从查询中提取商品名称。

        Args:
            query:   用户原始问题
            history: 历史对话，可以是 List[str]（旧格式）或 List[dict]（MongoDB格式）

        Returns:
            {"item_names": [...], "rewritten_query": "..."}
        """
        result = {'item_names': [], 'rewritten_query': ''}

        from knowledge.tools.llm_client import get_llm_client
        llm = get_llm_client()
        if not llm:
            return result

        # ★ 统一转为文本：兼容旧格式 List[str] 和新格式 List[dict]
        if history and isinstance(history[0], dict):
            history_text = ""
            for msg in history:
                role = msg.get("role", "")
                content = msg.get("text", "")
                history_text += f"{role}: {content}\n"
        else:
            history_text = history

        from knowledge.processor.query_process.item_name_extract_prompt import IITEM_NAME_EXTRACT_TEMPLATE
        prompt = IITEM_NAME_EXTRACT_TEMPLATE.format(query=query, history_text=history_text)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        try:
            response = llm.chat.completions.create(
                model="qwen-plus",
                messages=messages,
                temperature=0.1
            )
            llm_response = response.choices[0].message.content
        except Exception as e:
            print(f"LLM调用失败: {e}")
            return result

        if not llm_response:
            return result
        return self._clean_llm_response(llm_response)

    def _clean_llm_response(self, llm_response: str) -> dict:
        """清洗LLM返回的JSON"""
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response.strip())
        content = re.sub(r"\s*```$", "", cleaned)
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM 返回的 JSON 根节点不是字典对象")

        raw_items = parsed.get("item_names")
        item_names = [str(n).strip() for n in raw_items if n] if isinstance(raw_items, list) else []

        raw_query = parsed.get("rewritten_query")
        rewritten_query = str(raw_query).strip() if isinstance(raw_query, str) else ""

        return {"item_names": item_names, "rewritten_query": rewritten_query}

    def _vector_search(self, item_names: List[str], collection_name: str, top_k: int = 5) -> List[dict]:
        """通过混合检索（dense + sparse + RRF）搜索最相关的 item_name。"""
        bge3 = get_bgem3_client()
        if not bge3:
            return []

        from knowledge.tools.milvus_client import get_milvus_client
        milvus_client = get_milvus_client()
        if not milvus_client:
            return []

        # 1. 编码所有 item_name
        embeddings = bge3.encode_documents(item_names)
        dense_vectors = embeddings["dense"]
        sparse_matrix = embeddings["sparse"]

        # 2. 提取每个 item_name 的 sparse 向量
        sparse_vectors = []
        for i in range(len(item_names)):
            start = sparse_matrix.indptr[i]
            end = sparse_matrix.indptr[i + 1]
            indices = sparse_matrix.indices[start:end].tolist()
            data = sparse_matrix.data[start:end].tolist()
            sparse_vectors.append(dict(zip(indices, data)))

        # 3. 逐个 item_name 做混合检索
        all_results = []
        seen = set()
        for i, item_name in enumerate(item_names):
            dense_vec = dense_vectors[i].tolist() if hasattr(dense_vectors[i], 'tolist') else dense_vectors[i]
            sparse_vec = sparse_vectors[i]

            request_dense = AnnSearchRequest(
                data=[dense_vec],
                anns_field="dense_vector",
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=top_k,
            )
            request_sparse = AnnSearchRequest(
                data=[sparse_vec],
                anns_field="sparse_vector",
                param={"metric_type": "IP"},
                limit=top_k,
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
                limit=top_k,
                output_fields=["file_title", "item_name"],
            )

            matches = []
            for hits in results:
                for hit in hits:
                    entity = hit.get("entity", {})
                    name = entity.get("item_name", "")
                    if name and name not in seen:
                        seen.add(name)
                        matches.append({
                            "item_name": name,
                            "file_title": entity.get("file_title", ""),
                            "score": round(hit.get("distance", 0.0), 4),
                        })

            all_results.append({
                "extract_item_name": item_name,
                "matches": matches,
            })

        return all_results

    def _align_by_score(self, all_results: List[dict]) -> tuple[List[str], List[str]]:
        """根据分数对齐：确认项 vs 候选项"""
        confirmed: List[str] = []
        options: List[str] = []

        for result in all_results:
            extract_item_name = result.get("extract_item_name", "")
            matches = result.get("matches", [])
            if not matches:
                continue

            matches.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            high = [m for m in matches if m.get("score", 0.0) > HIGH_CONFIDENCE]

            if high:
                exact = next(
                    (m for m in high if m["item_name"].strip() == extract_item_name), None
                )
                if exact:
                    picked = exact["item_name"]
                    if picked not in confirmed:
                        confirmed.append(picked)
                elif len(high) == 1:
                    picked = high[0]["item_name"]
                    if picked not in confirmed:
                        confirmed.append(picked)
                else:
                    for m in high[:MAX_OPTIONS]:
                        name = m["item_name"]
                        if name not in confirmed and name not in options:
                            options.append(name)
            else:
                mid = [m for m in matches
                       if m["score"] >= MID_CONFIDENCE
                       and m["item_name"] not in confirmed
                       and m["item_name"] not in options]
                for m in mid[:MAX_OPTIONS]:
                    name = m["item_name"]
                    if name not in confirmed and name not in options:
                        options.append(name)

        return confirmed, options[:MAX_OPTIONS]

    def _filter_by_score_gap(self, confirmed: List[str], search_results: List[dict]) -> List[str]:
        """当确认项有多个时，过滤掉分数差距过大的项"""
        if not confirmed or not search_results:
            return confirmed

        score_map: dict = {}
        for res in search_results:
            for m in (res.get("matches") or []):
                name = (m.get("item_name") or "").strip()
                score = m.get("score", 0.0)
                if name in confirmed:
                    score_map[name] = max(score_map.get(name, 0.0), score)

        if len(score_map) < 2:
            return confirmed

        sorted_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        top1_score = sorted_items[0][1]

        kept = [
            name for name, score in sorted_items
            if top1_score - score <= SCORE_GAP_THRESHOLD
        ]
        return kept

    def _decide(self, state, confirmed, options, rewritten_query, history):
        """根据匹配结果决策：确认/反问/无法识别"""
        if confirmed:
            state["item_names"] = confirmed
            state["rewritten_query"] = rewritten_query
        elif options:
            state["answer"] = (
                f"我不确定您指的是哪款产品。"
                f"您是在询问以下产品吗：{'、'.join(options)}？"
            )
        else:
            state["answer"] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"

    def match_align(self, item_names: List[str], collection_name: str) -> tuple[List[str], List[str]]:
        """向量检索 + 评分对齐 + 分数差异过滤"""
        # Step 1: 向量检索
        search_results = self._vector_search(item_names, collection_name)

        # Step 2: 评分对齐
        confirmed, options = self._align_by_score(search_results)

        # Step 3: 分数差异过滤（仅当 confirmed 有多个时触发）
        if len(confirmed) > 1:
            confirmed = self._filter_by_score_gap(confirmed, search_results)

        return confirmed, options


if __name__ == "__main__":
    import logging
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.WARNING)

    # ─────────────────────────────────
    # 快捷造数据
    # ─────────────────────────────────
    def mk(extracted, *pairs):
        """pairs: (item_name, score), ..."""
        return {
            "extract_item_name": extracted,
            "matches": [
                {"item_name": n, "file_title": f"{n}.pdf", "score": s}
                for n, s in pairs
            ],
        }

    extractor = ItemExtractor()
    ok, fail = 0, 0

    def check(label, actual, expected):
        global ok, fail
        if actual == expected:
            ok += 1
            print(f"  ✅ {label}")
        else:
            fail += 1
            print(f"  ❌ {label}: 期望 {expected}, 实际 {actual}")

    # ===========================================
    # 1. _filter_by_score_gap
    # ===========================================
    print("=" * 45)
    print("  _filter_by_score_gap")
    print("=" * 45)

    target = extractor._filter_by_score_gap

    # 边界
    check("空confirmed",            target([], []),                       [])
    check("空search_results",       target(["X"], []),                    ["X"])
    check("单项不比较",              target(["X"], [mk("X", ("X", 0.9))]), ["X"])

    # 差距 ≤ 0.1 全保留
    check("差距0.07保留",
          set(target(["A", "B"], [mk("A", ("A", 0.95)), mk("B", ("B", 0.88))])),
          {"A", "B"})

    # 差距 > 0.1 踢低分
    check("差距0.25过滤",
          target(["A", "B"], [mk("A", ("A", 0.95)), mk("B", ("B", 0.70))]),
          ["A"])

    # 同商品多轮取 max
    check("取max不误杀",
          target(["A", "B"], [mk("X", ("A", 0.60)), mk("A", ("A", 0.95))]),
          ["A", "B"])

    # ===========================================
    # 2. _align_by_score
    # ===========================================
    print("\n" + "=" * 45)
    print("  _align_by_score")
    print("=" * 45)

    # A: 精确命中
    c, o = extractor._align_by_score([mk("A", ("A", 0.95), ("B", 0.72))])
    check("A-精确命中",        "A" in c and "B" not in c,         True)

    # B: 独苗高置信
    c, o = extractor._align_by_score([mk("X万用表", ("A", 0.87))])
    check("B-独苗确认",        c,                                  ["A"])

    # C: 多高无精确
    c, o = extractor._align_by_score([mk("万用表", ("A", 0.88), ("B", 0.85))])
    check("C-confirmed空",     c,                                  [])
    check("C-进options",       set(o),                             {"A", "B"})

    # D: 中置信
    c, o = extractor._align_by_score([mk("电压表", ("A", 0.62), ("B", 0.61), ("C", 0.55))])
    check("D-confirmed空",     c,                                  [])
    check("D-进options",       len(o) >= 2,                        True)

    # 空 matches
    c, o = extractor._align_by_score([mk("未知")])
    check("空matches",         c == [] and o == [],                True)

    # 跨轮去重
    c, o = extractor._align_by_score([mk("A", ("A", 0.95)), mk("X", ("A", 0.87))])
    check("跨轮去重",           c,                                  ["A"])

    # options 截断
    c, o = extractor._align_by_score([mk("X", *[(chr(65+i), 0.82) for i in range(6)])])
    check(f"options≤{MAX_OPTIONS}", len(o) <= MAX_OPTIONS,         True)

    # ===========================================
    # 3. 集成测试（需 LLM + Milvus）
    # ===========================================
    print("\n" + "=" * 45)
    print("  集成测试")
    print("=" * 45)

    test_state = {
        "session_id": "test_123",
        "original_query": "RSPRORS-12数字万用表怎么使用？",
        "item_names": [],
        "rewritten_query": "",
        "answer": "",
        "history": [],
    }
    query = test_state['original_query']
    print(f"输入: {query}\n")

    # Step 1: LLM 提取
    print("--- Step 1: LLM 提取商品名 ---")
    extract_result = extractor.extract_item_name(query, [])
    item_names = extract_result.get("item_names", [])
    rewritten_query = extract_result.get("rewritten_query", "")
    print(f"  item_names: {item_names}")
    print(f"  rewritten_query: {rewritten_query}")

    if not item_names:
        print("  ⚠️ LLM 未提取到商品名，跳过后续步骤")
    else:
        # Step 2: 向量检索
        print(f"\n--- Step 2: 向量检索 (collection: item_name_collection_test) ---")
        search_results = extractor._vector_search(item_names, "item_name_collection_test")
        for r in search_results:
            print(f"  提取名: {r['extract_item_name']}")
            for m in r.get("matches", []):
                print(f"    匹配: {m['item_name']}, 分数: {m['score']}")

        # Step 3: 评分对齐
        print(f"\n--- Step 3: 评分对齐 ---")
        confirmed, options = extractor._align_by_score(search_results)
        print(f"  confirmed: {confirmed}")
        print(f"  options: {options}")

        # Step 3.5: 分数过滤
        if len(confirmed) > 1:
            confirmed = extractor._filter_by_score_gap(confirmed, search_results)
            print(f"  过滤后 confirmed: {confirmed}")

        # Step 4: 决策
        print(f"\n--- Step 4: 决策 ---")
        extractor._decide(test_state, confirmed, options, rewritten_query, [])
        print(f"  item_names: {test_state.get('item_names')}")
        print(f"  answer: {test_state.get('answer', '')}")

    # ===========================================
    # 总结
    # ===========================================
    total = ok + fail
    print(f"\n{'='*45}")
    print(f"  纯逻辑: {ok}/{total} 通过" + (f"  ({fail} 失败)" if fail else "  🎉"))
    print(f"{'='*45}")

