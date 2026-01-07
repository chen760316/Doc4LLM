"""
根据（1）得到的星型图（组成的双星结构），从中生成示例的RICO谓词和规则，提供给生成式模型进行学习
候选谓词w为 violate(𝑥0,𝑦0) ，该谓词在每个双星模式上只有一个，取值为八种具体矛盾类型之一，目标在指出该双星的具体矛盾类型
"""

from glob import glob
import json
import os
import random
import re
from collections import defaultdict
from itertools import combinations
from dotenv import load_dotenv
from knowledge_graph_maker import OpenAIClient
from collections import Counter

# 加载环境变量
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

# 初始化LLM客户端
llm = OpenAIClient(
    model="gpt-4.1-mini",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

def load_text_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def contra_from_spans(span_x, span_y, node_x, node_y, doc1, doc2):
    """
    提取文本片段并返回 contra("文本1", "文本2") 谓词
    """
    text1 = doc1[span_x["start"]:span_x["end"]].strip().replace('"', "'")
    text2 = doc2[span_y["start"]:span_y["end"]].strip().replace('"', "'")
    return f'contra("{text1}", "{text2}")'


def contra(s1: str, s2: str, llm) -> bool:
    """
    Returns True if s1 and s2 contradict each other (semantically), using LLM.
    """
    system_prompt = (
        "You are a natural language inference engine. "
        "Determine whether the following two sentences contradict each other. "
        "Respond with only one word: 'Yes' if they contradict, or 'No' if they do not."
    )

    user_message = f"Sentence 1: {s1}\nSentence 2: {s2}\nDo they contradict?"

    try:
        response = llm.generate(
            user_message=user_message,
            system_message=system_prompt
        )
        answer = response.strip().lower()
        return answer.startswith("yes")
    except Exception as e:
        print(f"⚠️ Error in contra() check:\n{e}")
        return False  # fallback

# ML predicate伪函数
def M_predicate(attrs1: dict, attrs2: dict) -> float:
    common_keys = set(attrs1.keys()) & set(attrs2.keys())
    score = sum(1 for k in common_keys if attrs1[k] == attrs2[k]) / max(len(common_keys), 1)
    return score

def load_dual_star_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_document_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def generate_predicates_and_rules(data, doc1, doc2, x0, y0, ml_threshold=0.5):
    predicates = set()
    rules = []

    nodes_attrs = {}
    edges_attrs = {}
    text_spans = {}

    for triple in data:
        n1 = triple["node_1"]["name"]
        n2 = triple["node_2"]["name"]

        for k, v in triple["node_1"].items():
            if k not in ["name", "text_span", "label"]:
                nodes_attrs.setdefault(n1, {})[k] = v
        for k, v in triple["node_2"].items():
            if k not in ["name", "text_span", "label"]:
                nodes_attrs.setdefault(n2, {})[k] = v

        if "text_span" in triple["node_1"]:
            text_spans[n1] = triple["node_1"]["text_span"]
        if "text_span" in triple["node_2"]:
            text_spans[n2] = triple["node_2"]["text_span"]

        # 边属性：先初始化字典
        edge_key = (n1, n2)
        edges_attrs.setdefault(edge_key, {})

        # 关系字符串
        if "relationship" in triple and triple["relationship"]:
            edges_attrs[edge_key]["relationship"] = triple["relationship"]

        # 加入edge_properties中的所有属性
        if "edge_properties" in triple:
            for ek, ev in triple["edge_properties"].items():
                edges_attrs[edge_key][ek] = ev

    for (x, y) in edges_attrs.keys():
        predicates.add(f"linkage({x}, {y})")


    def gen_unary_preds():
        for node, attrs in nodes_attrs.items():
            for attr, val in attrs.items():
                if isinstance(val, (int, float)):
                    predicates.add(f"{node}.{attr} = {val}")
                    predicates.add(f"{node}.{attr} != {val}")
                    predicates.add(f"{node}.{attr} < {val + 5}")
                    predicates.add(f"{node}.{attr} > {val - 5}")
                    predicates.add(f"{node}.{attr} <= {val}")
                    predicates.add(f"{node}.{attr} >= {val}")
                elif isinstance(val, str):
                    predicates.add(f'{node}.{attr} = "{val}"')
                    predicates.add(f'{node}.{attr} != "{val}"')

    def gen_binary_node_preds():
        nodes = list(nodes_attrs.keys())
        for x, y in combinations(nodes, 2):
            attrs_x = nodes_attrs[x]
            attrs_y = nodes_attrs[y]
            for attr in set(attrs_x) & set(attrs_y):
                if type(attrs_x[attr]) != type(attrs_y[attr]):
                    continue
                if isinstance(attrs_x[attr], (int, float)):
                    predicates.add(f"{x}.{attr} != {y}.{attr}")
                elif isinstance(attrs_x[attr], str):
                    predicates.add(f'{x}.{attr} != {y}.{attr}')

    def gen_unary_edge_preds():
        for (x, y), attrs in edges_attrs.items():
            for attr, val in attrs.items():
                if attr == "text_span":
                    continue  # 跳过文本span字段
                if isinstance(val, (int, float)):
                    predicates.add(f"{x}_{y}.{attr} = {val}")
                    predicates.add(f"{x}_{y}.{attr} != {val}")
                    predicates.add(f"{x}_{y}.{attr} < {val + 1}")
                    predicates.add(f"{x}_{y}.{attr} > {val - 1}")
                    predicates.add(f"{x}_{y}.{attr} <= {val}")
                    predicates.add(f"{x}_{y}.{attr} >= {val}")
                elif isinstance(val, str):
                    predicates.add(f'{x}_{y}.{attr} = "{val}"')
                    predicates.add(f'{x}_{y}.{attr} != "{val}"')
                elif isinstance(val, list):
                    # 简单处理list，转成字符串摘要或长度谓词
                    predicates.add(f"{x}_{y}.{attr}.length = {len(val)}")

    def gen_binary_edge_preds():
        edges = list(edges_attrs.keys())
        for (ex1, ey1), (ex2, ey2) in combinations(edges, 2):
            attr1 = edges_attrs[(ex1, ey1)]["relationship"]
            attr2 = edges_attrs[(ex2, ey2)]["relationship"]
            if attr1 != attr2:
                predicates.add(f'{ex1}_{ey1}.relationship != {ex2}_{ey2}.relationship')

    def gen_contra_preds():
        nodes = list(text_spans.keys())
        for x, y in combinations(nodes, 2):
            span_x = text_spans[x]
            span_y = text_spans[y]

            # 判断两个节点属于哪个文档
            # 假设 node_1 来源于 text1，node_2 来源于 text2
            doc_x = doc1 if "doc1" in x.lower() or "node_1" in x.lower() else doc1  # 默认都来自 doc1
            doc_y = doc2 if "doc2" in y.lower() or "node_2" in y.lower() else doc2  # 可根据实际命名规则改

            contra_pred = contra_from_spans(span_x, span_y, x, y, doc_x, doc_y)
            predicates.add(contra_pred)

    def print_predicate_statistics(predicates):
        stats = defaultdict(int)
        for p in predicates:
            if "contra" in p:
                stats["contra"] += 1
            elif "M_NLI" in p or "M_NLU" in p or "M_TEM" in p:
                stats["ML"] += 1
            elif "linkage" in p:
                stats["linkage"] += 1
            elif "relationship" in p or "_.relationship" in p:
                stats["edge_attr"] += 1
            else:
                stats["entity_attr"] += 1

        print("📊 Predicate Type Statistics:")
        for k, v in stats.items():
            print(f" - {k}: {v}")

    def gen_ml_preds():
        nodes = list(nodes_attrs.keys())
        for x, y in combinations(nodes, 2):
            score = M_predicate(nodes_attrs[x], nodes_attrs[y])
            if score < ml_threshold:
                predicates.add(f'M_NLI({x}.attrs, {y}.attrs)')
                predicates.add(f'M_NLU({x}.attrs, {y}.attrs)')
                predicates.add(f'M_TEM({x}.attrs, {y}.attrs)')

    # 执行所有谓词生成器
    gen_unary_preds()
    gen_binary_node_preds()
    gen_unary_edge_preds()
    gen_binary_edge_preds()
    gen_contra_preds()
    gen_ml_preds()

    # 假设 predicates 已有
    predicates = {p.strip().replace("  ", " ") for p in predicates if p}
    print_predicate_statistics(predicates)
    pred_list = list(predicates)
    rules = []
    rule_count = 0
    max_rules = 10

    # 先按类型分类谓词
    type_buckets = defaultdict(list)
    for p in pred_list:
        if "contra" in p:
            type_buckets["contra"].append(p)
        elif "M(" in p:
            type_buckets["M"].append(p)
        elif "linkage" in p:
            type_buckets["linkage"].append(p)
        else:
            type_buckets["other"].append(p)

    # 先确保每个类型至少出现一次作为后件
    used_heads = set()
    for t, preds_of_type in type_buckets.items():
        # 先随便挑一个谓词作为后件
        head = random.choice(preds_of_type)
        used_heads.add(head)
        # 找个body（不包含head）组合成规则
        bodies = [p for p in pred_list if p != head]
        if not bodies:
            continue
        body = random.choice(bodies)
        rule_body = f"{body} ∧ {head}"
        rules.append(f"Q[{x0},{y0}] ({rule_body} → {head})")
        rule_count += 1
        if rule_count >= max_rules:
            break

    if rule_count < max_rules:
        # 生成剩余规则，随机后件，但不重复用已经用过的head，保证多样性
        combos = list(combinations(pred_list, 2))
        random.shuffle(combos)
        for combo in combos:
            body = " ∧ ".join(combo)
            # 随机选后件
            head = random.choice(combo)
            if head in used_heads:
                continue  # 跳过重复的后件
            rule = f"Q[{x0},{y0}] ({body} → {head})"
            rules.append(rule)
            used_heads.add(head)
            rule_count += 1
            if rule_count >= max_rules:
                break


    return predicates, rules

def find_dual_star_centers(data):
    """
    在双星结构中，寻找中心节点（通过 same_entity 关系连接的两个节点）。
    返回中心节点名 x0, y0。
    """
    for triple in data:
        rel = triple.get("relationship", "").lower()
        if rel == "same_entity":
            node1 = triple.get("node_1", {}).get("name")
            node2 = triple.get("node_2", {}).get("name")
            if node1 and node2:
                return node1, node2

    raise ValueError("❌ 未找到 same_entity 的中心节点连接")


# 示例入口
if __name__ == "__main__":
    # 设定你的目录
    DUAL_STAR_JSON_DIR = "../star_json/dual_graph"
    OUTPUT_DIR = "star_outputs"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 遍历所有 JSON 文件
    json_files = glob(os.path.join(DUAL_STAR_JSON_DIR, "*.json"))

    for json_path in json_files:
        filename = os.path.basename(json_path)
        name_without_ext = os.path.splitext(filename)[0]

        print(f"\n🌀 处理中: {filename}")

        try:
            # 加载 JSON 数据
            data = load_dual_star_json(json_path)
            x0, y0 = find_dual_star_centers(data)

            # 推测对应文本路径，比如 text1/2 位于 texts 文件夹中
            # 你需要保证这些文件存在或按规则命名，例如：
            # dual_star_graph_2_John_Smith.json -> text1_dual_star_graph_2_John_Smith.txt
            text1_path = "text1.txt"
            text2_path = "text2.txt"

            if not os.path.exists(text1_path) or not os.path.exists(text2_path):
                print(f"⚠️ 缺失对应文本文件: {text1_path} 或 {text2_path}")
                continue

            doc1 = load_text_file(text1_path)
            doc2 = load_text_file(text2_path)

            preds, rules = generate_predicates_and_rules(data, doc1, doc2, x0, y0)

            # 输出文件
            output_file = os.path.join(OUTPUT_DIR, f"{name_without_ext}_rico.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("Predicates:\n")
                for p in sorted(preds):
                    f.write(p + "\n")

                f.write("\nRules:\n")
                for r in rules:
                    f.write(r + "\n")

            print(f"✅ 已保存到: {output_file}")

        except Exception as e:
            print(f"❌ 处理失败: {filename} - 错误: {e}")
