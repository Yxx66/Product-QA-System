from langgraph.graph import StateGraph, START, END

from knowledge.processor.import_process.state import ImportGraphState, create_default_state
from knowledge.processor.import_process.nodes.entry_node import EntryNode
from knowledge.processor.import_process.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_process.nodes.md_image_node import MD_IMAGE
from knowledge.processor.import_process.nodes.split_documents_node import SplitDocumentsNode
from knowledge.processor.import_process.nodes.item_name_recognition_node import item_name_recognition_node
from knowledge.processor.import_process.nodes.bge_embedding import bge_embedding
from knowledge.processor.import_process.nodes.import_milvus import import_milvus_node
from knowledge.processor.import_process.nodes.kg_graph_node import kg_graph_node


def route_after_entry(state: ImportGraphState) -> str:
    """根据 entry 节点的状态决定路由"""
    if state.get("is_pdf_read_enabled"):
        return "pdf_to_md"
    return "md_image"


def build_graph():
    """构建导入处理工作流图"""
    graph = StateGraph(ImportGraphState)

    # 添加节点
    graph.add_node("entry", EntryNode())
    graph.add_node("pdf_to_md", PdfToMdNode())
    graph.add_node("md_image", MD_IMAGE())
    graph.add_node("split_documents", SplitDocumentsNode())
    graph.add_node("item_name_recognition", item_name_recognition_node())
    graph.add_node("bge_embedding", bge_embedding())
    graph.add_node("import_milvus", import_milvus_node())
    graph.add_node("kg_graph", kg_graph_node())

    # START → entry
    graph.add_edge(START, "entry")

    # entry → 条件路由
    graph.add_conditional_edges("entry", route_after_entry)

    # pdf_to_md → md_image
    graph.add_edge("pdf_to_md", "md_image")

    # md_image → split_documents
    graph.add_edge("md_image", "split_documents")

    # split_documents → item_name_recognition
    graph.add_edge("split_documents", "item_name_recognition")

    # item_name_recognition → bge_embedding
    graph.add_edge("item_name_recognition", "bge_embedding")

    # bge_embedding → import_milvus
    graph.add_edge("bge_embedding", "import_milvus")

    # import_milvus → kg_graph
    graph.add_edge("import_milvus", "kg_graph")

    # kg_graph → END
    graph.add_edge("kg_graph", END)

    return graph.compile()


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    from knowledge.processor.import_process.base import setup_logging
    setup_logging()

    project_root = Path(__file__).resolve().parents[3]
    md_path = project_root / "import_temp_dir" / "64971847" / "mineru_output" / "hak180产品安全手册" / "auto" / "hak180产品安全手册.md"

    state = create_default_state(
        task_id="test_001",
        import_file_path=str(md_path),
        is_md_read_enabled=True,
        md_path=str(md_path),
    )

    app = build_graph()
    result = app.invoke(state)
    print(f"\n{'='*50}")
    print(f"流程完成")
    print(f"chunks 数量: {len(result.get('chunks', []))}")
    print(f"item_name: {result.get('item_name')}")
