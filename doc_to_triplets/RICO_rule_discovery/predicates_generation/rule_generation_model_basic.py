"""
没有借助生成式模型，直接在给定的图模式下推导RICO的谓词和规则
候选谓词w为 violate(𝑥0,𝑦0) ，该谓词在每个双星模式上只有一个，取值为八种具体矛盾类型之一，目标在指出该双星的具体矛盾类型
"""

import json
import os
import re
import random
from itertools import combinations
from typing import List, Set


def load_graph(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_numeric(val: str) -> bool:
    try:
        float(val)
        return True
    except ValueError:
        return False


def generate_predicates_from_graph(graph_json: List[dict]) -> List[str]:
    predicates: Set[str] = set()
    attrs_by_node = {}  # For meta-predicates

    for edge in graph_json:
        node1 = edge.get("node_1", {})
        node2 = edge.get("node_2", {})
        rel = edge.get("relationship", "").strip()
        edge_props = edge.get("edge_properties", {})

        name1 = node1.get("name", "x0")
        name2 = node2.get("name", "y0")

        # Entity-level predicates
        for node, name in [(node1, name1), (node2, name2)]:
            for attr, val in node.items():
                if attr not in ["name", "label", "text_span"] and val != "unknown":
                    val_str = str(val)
                    predicates.add(f'{name}.{attr} = "{val_str}"')

                    # Add != predicate
                    predicates.add(f'{name}.{attr} != "{val_str}"')

                    # If numeric, add comparisons
                    if is_numeric(val_str):
                        predicates.add(f'{name}.{attr} > {val_str}')
                        predicates.add(f'{name}.{attr} < {val_str}')
                        predicates.add(f'{name}.{attr} >= {val_str}')
                        predicates.add(f'{name}.{attr} <= {val_str}')

                    # Record for meta
                    attrs_by_node.setdefault(name, []).append(f'{name}.{attr}')

        # Edge-level predicates
        if rel:
            predicates.add(f'{name1}_{name2}.relationship = "{rel}"')
            predicates.add(f'{name1}_{name2}.relationship != "{rel}"')

        for attr, val in edge_props.items():
            if attr != "text_span":
                val_str = str(val)
                predicates.add(f'{name1}_{name2}.{attr} = "{val_str}"')
                predicates.add(f'{name1}_{name2}.{attr} != "{val_str}"')

                if is_numeric(val_str):
                    predicates.add(f'{name1}_{name2}.{attr} > {val_str}')
                    predicates.add(f'{name1}_{name2}.{attr} < {val_str}')
                    predicates.add(f'{name1}_{name2}.{attr} >= {val_str}')
                    predicates.add(f'{name1}_{name2}.{attr} <= {val_str}')

        # Meta-level
        predicates.add(f"linkage({name1}, {name2})")

        # Generate M_* meta-predicates
        for meta in ["M_NLI", "M_NLU", "M_TEM"]:
            attrs1 = attrs_by_node.get(name1, [])
            attrs2 = attrs_by_node.get(name2, [])
            for a1 in attrs1:
                for a2 in attrs2:
                    predicates.add(f'{meta}({a1}, {a2})')

        # Simple contra predicate (placeholder text)
        predicates.add(f'contra("{name1} text", "{name2} text")')

    return sorted(predicates)


def generate_rules(predicates: List[str], num_rules: int = 30, min_len=2, max_len=6) -> List[str]:
    all_rules = set()
    tried = 0
    max_attempts = 5000  # Prevent infinite loop

    while len(all_rules) < num_rules and tried < max_attempts:
        premise_len = random.randint(min_len, max_len)  # 确定规则长度
        premise = random.sample(predicates, k=premise_len)
        conclusion_candidates = [p for p in predicates if p not in premise]

        if not conclusion_candidates:
            tried += 1
            continue

        conclusion = random.choice(conclusion_candidates)

        # Rule format
        rule = f'Q[x0, y0]({ " ∧ ".join(premise) } → {conclusion})'

        # Avoid trivial rules or duplicate structure
        if is_valid_rule(rule, premise, conclusion):
            all_rules.add(rule)

        tried += 1

    return sorted(all_rules)


def is_valid_rule(rule: str, premise: List[str], conclusion: str) -> bool:
    # Avoid rules like: a=a → a=a
    if conclusion in premise:
        return False

    # Avoid tautologies (same attribute and value)
    premise_attrs = set([re.split(r'\s*=', p)[0].strip() for p in premise])
    conclusion_attr = re.split(r'\s*=', conclusion)[0].strip()
    if conclusion_attr in premise_attrs:
        return False

    return True


def save_list_to_file(lines: List[str], path: str, header: str = ""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        if header:
            f.write(f"{header}\n")
        for line in lines:
            f.write(line.strip() + "\n")


def main():
    # === 输入路径 ===
    graph_path = "../star_json/dual_graph/dual_star_graph_1_Emily_Clark.json"

    # === 输出路径 ===
    pred_path = "./star_outputs/generated_predicates.txt"
    rule_path = "./star_outputs/generated_rules.txt"

    # === 读取图数据 ===
    graph = load_graph(graph_path)

    # === Step 1: 生成谓词 ===
    predicates = generate_predicates_from_graph(graph)
    save_list_to_file(predicates, pred_path, header="Predicates:")
    print(f"✅ Generated {len(predicates)} predicates.")

    # === Step 2: 生成规则 ===
    rules = generate_rules(predicates, num_rules=30)
    save_list_to_file(rules, rule_path, header="Rules:")
    print(f"✅ Generated {len(rules)} rules.")

    print("🚀 All done. Results saved in ./output/")


if __name__ == "__main__":
    main()

