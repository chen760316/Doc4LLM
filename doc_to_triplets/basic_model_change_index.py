from src.kg_gen import KGGen
import os
from dotenv import load_dotenv
from tests.utils.visualize_kg import visualize
import re
from typing import List, Dict, Tuple


def print_triples_concise(annotated_triples):
    for i, triple in enumerate(annotated_triples, 1):
        head = triple["head"]
        relation = triple["relation"]
        tail = triple["tail"]

        head_pos = head["positions"][0] if head["positions"] else "未找到"
        tail_pos = tail["positions"][0] if tail["positions"] else "未找到"

        print(f"\n三元组 #{i}:")
        print(f"{head['text']} [位置: {head_pos}] → {relation} → {tail['text']} [位置: {tail_pos}]")

def locate_entities_in_text(text: str, entities: List[str]) -> Dict[str, List[Tuple[int, int]]]:
    """
    定位实体在原始文本中的所有出现位置
    :param text: 原始文本
    :param entities: 需要定位的实体列表
    :return: 字典 {实体: [(start1, end1), (start2, end2), ...]}
    """
    entity_positions = {}
    for entity in entities:
        # 使用正则表达式查找所有匹配项（考虑转义特殊字符）
        matches = list(re.finditer(re.escape(entity), text))
        positions = [(m.start(), m.end()) for m in matches]
        if positions:
            entity_positions[entity] = positions
    return entity_positions

def annotate_triples_with_positions(text: str, triples: List[Tuple[str, str, str]]) -> List[Dict]:
    """
    为三元组添加实体位置信息
    :param text: 原始文本
    :param triples: 原始三元组列表 (head, relation, tail)
    :return: 增强后的三元组列表，包含位置信息
    """
    # 收集所有需要定位的实体
    all_entities = set()
    for head, _, tail in triples:
        all_entities.add(head)
        all_entities.add(tail)
    
    # 定位所有实体的位置
    entity_positions = locate_entities_in_text(text, list(all_entities))
    
    # 构建增强后的三元组
    annotated_triples = []
    for head, relation, tail in triples:
        annotated_triples.append({
            "head": {
                "text": head,
                "positions": entity_positions.get(head, [])
            },
            "relation": relation,
            "tail": {
                "text": tail,
                "positions": entity_positions.get(tail, [])
            }
        })
    return annotated_triples

if __name__ == "__main__":
    # 加载环境变量
    load_dotenv()

    # 初始化KG生成器
    kg = KGGen(api_key=os.getenv("OPENAI_API_KEY"), api_base=os.getenv("OPENAI_BASE_URL"))

    # 读取文本
    with open("book_split.txt", "r", encoding="utf-8") as file:
        text = file.read()

    # 生成知识图谱
    graph = kg.generate(
        input_data=text,
        model="openai/gpt-4o-mini"
    )

    # 为三元组添加位置信息
    annotated_triples = annotate_triples_with_positions(text, graph.relations)

    # 可视化结果（可选）
    visualize(graph, "E:/Document cleaning/kg-gen/tests/test_basic.html")

    # 打印带有位置信息的三元组
    print("\n增强后的三元组（包含实体位置信息）:")
    print_triples_concise(annotated_triples)
    # for triple in annotated_triples:
    #     head = triple["head"]
    #     tail = triple["tail"]
    #     print(f"\n关系: {triple['relation']}")
    #     print(f"头实体: '{head['text']}'")
    #     print(f"  位置: {head['positions']}")
    #     print(f"尾实体: '{tail['text']}'")
    #     print(f"  位置: {tail['positions']}")

    # 打印原始图谱（可选）
    print("\n原始图谱结构:")
    print(graph)