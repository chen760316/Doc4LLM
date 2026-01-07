"""
修改图连接：对于引用同一实体但来自不同句子/章节/文档的两个顶点，
在图G_D中，我们在两者之间添加一条带有特殊标签的边，表示这两者“匹配”。
与[22]相反，我们实际上并没有合并这两者，以便每个子元素和子元素都可以保持它们的结构和属性;但是我们用一个子元素边缘使两者“局部”，以便我们可以合成两者之间的相互作用。
"""

from src.kg_gen import KGGen
import os
import re
from dotenv import load_dotenv
from tests.utils.visualize_kg import visualize
from typing import List, Tuple, Dict


def locate_entities(text: str, entity: str) -> List[Tuple[int, int]]:
    """查找实体在文本中的所有出现位置"""
    return [(m.start(), m.end()) for m in re.finditer(re.escape(entity), text)]

def find_matched_entities(entities1: List[str], entities2: List[str]) -> List[Tuple[str, str]]:
    """找到两个图中同名的实体（按字符串匹配）"""
    matched = []
    for e1 in entities1:
        for e2 in entities2:
            if e1 == e2:
                matched.append((e1, e2))
    return matched



if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    kg = KGGen(api_key=os.getenv("OPENAI_API_KEY"), api_base=os.getenv("OPENAI_BASE_URL"))

    # 读取文本
    with open("split1.txt", "r", encoding="utf-8") as file:
        text1 = file.read()
    with open("split2.txt", "r", encoding="utf-8") as file:
        text2 = file.read()

    # 生成图谱
    graph1 = kg.generate(input_data=text1, model="openai/gpt-4o-mini", context="Family relationships")
    graph2 = kg.generate(input_data=text2, model="openai/gpt-4o-mini", context="Family relationships")

    print("*"*100)

    print("Graph1 Extracted Triples:")
    for head, relation, tail in graph1.relations:
        print(f"({head}) -[{relation}]-> ({tail})")

    print("*"*100)

    print("Graph2 Extracted Triples:")
    for head, relation, tail in graph2.relations:
        print(f"({head}) -[{relation}]-> ({tail})")

    print("*"*100)

    # 创建关系来源映射 (在融合前记录)
    relation_sources = {}

    # 记录graph1的关系来源
    for head, relation, tail in graph1.relations:
        relation_key = (head, relation, tail)
        relation_sources[relation_key] = "text1"

    # 记录graph2的关系来源
    for head, relation, tail in graph2.relations:
        relation_key = (head, relation, tail)
        relation_sources[relation_key] = "text2"

    # 保持原有的图融合过程不变
    combined_graph = kg.aggregate([graph1, graph2])

    # 🔧 添加 matched_entity 边
    matched_pairs = find_matched_entities(graph1.entities, graph2.entities)
    for head, tail in matched_pairs:
        matched_relation = (head, "matched_entity", tail)
        combined_graph.relations.add(matched_relation)
        combined_graph.edges.add("matched_entity")

    print("组合图中的关系如下：")
    print(combined_graph.relations)
    print("*"*100)

    # 打印带位置信息的三元组
    print("\nCombined_graph Extracted Triples with Locations:")
    for head, relation, tail in combined_graph.relations:
        relation_key = (head, relation, tail)
        source = relation_sources.get(relation_key, "unknown")
        source_text = text1 if source == "text1" else text2

        head_positions = locate_entities(source_text, head)
        tail_positions = locate_entities(source_text, tail)

        print(f"\n({head}) -[{relation}]-> ({tail})")
        print(f"来源: {source}")
        print(f"头实体位置: {head_positions}")
        print(f"尾实体位置: {tail_positions}")

        if head_positions:
            start, end = head_positions[0]
            context = source_text[max(0, start - 20):min(len(source_text), end + 20)]
            print(f"头实体上下文: ...{context}...")
        if tail_positions:
            start, end = tail_positions[0]
            context = source_text[max(0, start - 20):min(len(source_text), end + 20)]
            print(f"尾实体上下文: ...{context}...")

    visualize(combined_graph, "E:/Document cleaning/kg-gen/tests/test_basic_2.html")