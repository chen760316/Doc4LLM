"""
借助大模型根据已有的RICO谓词和规则，在给定的图模式下学习生成新的谓词和规则
由于llm倾向于生成满足条件的最小长度的规则，因此要求llm生成长度2-6之间的规则
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
from typing import List


# === 初始化环境 ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

llm = OpenAIClient(
    model="gpt-4.1-mini",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)


def parse_rico_file(filepath: str) -> (List[str], List[str]):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    predicate_block = re.search(r'Predicates:\n(.*?)\nRules:', content, re.DOTALL)
    rule_block = re.search(r'Rules:\n(.*)', content, re.DOTALL)

    if not predicate_block or not rule_block:
        raise ValueError("RICO file format is incorrect or missing blocks.")

    predicates = [line.strip() for line in predicate_block.group(1).strip().split('\n') if line.strip()]
    rules = [line.strip() for line in rule_block.group(1).strip().split('\n') if line.strip()]

    return predicates, rules


# === Step 1: 提取谓词 ===
def extract_predicates_from_graph(json_path: str) -> List[str]:
    with open(json_path, 'r') as f:
        data = json.load(f)

    predicates = set()
    for edge in data:
        node1 = edge.get("node_1", {})
        node2 = edge.get("node_2", {})
        rel = edge.get("relationship", "").strip()

        name1 = node1.get("name")
        name2 = node2.get("name")

        for attr, val in node1.items():
            if attr not in ["name", "label", "text_span"] and val != "unknown":
                predicates.add(f'{name1}.{attr} = "{val}"')

        for attr, val in node2.items():
            if attr not in ["name", "label", "text_span"] and val != "unknown":
                predicates.add(f'{name2}.{attr} = "{val}"')

        if rel:
            predicates.add(f'{name1}_{name2}.relationship = "{rel}"')

        if name1 and name2:
            predicates.add(f"linkage({name1}, {name2})")

    return sorted(predicates)

# === Step 2: 定义 prompt 模板 ===

def build_rule_generation_prompt(rico_predicates: List[str], example_rules: List[str]) -> str:
    return f"""You are an expert in logical reasoning and contradiction detection.

Your task is to generate RICO-style rules based on the given list of predicates.

Each rule should follow the format:
Q[x0, y0](P1 ∧ P2 ∧ ... → P_out)

### Rule Constraints:

1. Each rule must include between **2 and 6 predicate conditions** in the premise.
2. You must generate a **diverse set of rules** with varying premise lengths — specifically, make sure to include rules whose premises have 2, 3, 4, 5, and 6 predicate conditions.
3. The number of rules for each premise length should be **as evenly distributed as possible** — for example, if you are generating 20 rules, try to generate about 4 rules for each premise length.
4. Each rule must have **exactly one predicate in the conclusion**.
5. The **conclusion predicate must not be identical to any of the predicates in the premise**. That is, avoid reusing the same predicate in both the premise and the conclusion of a rule.

Use only the provided predicates. Use them in combinations to create **reasonable and minimal** logical implications. Try to make each rule capture a plausible constraint, contradiction, or implication.

--- Examples ---

Rules:
{chr(10).join(example_rules)}

--- Predicates ---
{chr(10).join(rico_predicates)}

--- Now write exactly 30 new rules below ---
Rules:"""



# === Step 3: 主调用 ===

def generate_rico_rules(json_path: str, example_rules: List[str], llm):
    preds = extract_predicates_from_graph(json_path)
    prompt = build_rule_generation_prompt(preds, example_rules)

    system_message = (
        "You are a natural language inference engine that generates RICO-style logical rules "
        "from given predicates and examples. Each rule should capture a plausible implication or contradiction, "
        "and follow the format Q[x0,y0](X → Y). Output only the rules."
    )

    response = llm.generate(
        user_message=prompt,
        system_message=system_message
    )
    return response.strip()

def generate_rico_output(
    json_graph_path: str,
    rico_txt_path: str,
    llm
) -> dict:
    rico_predicates, example_rules = parse_rico_file(rico_txt_path)

    # 可选：从图中提取新谓词（如果你希望）
    # extracted_preds = extract_predicates_from_graph(json_graph_path)

    # 构造 prompt
    prompt = build_rule_generation_prompt(rico_predicates, example_rules)

    system_message = (
        "You are a natural language inference engine that generates RICO-style logical rules "
        "from given predicates and examples. Each rule should capture a plausible implication or contradiction, "
        "and follow the format Q[x0,y0](X → Y). Output only the rules."
    )

    response = llm.generate(
        user_message=prompt,
        system_message=system_message
    )

    return {
        "new_rules": response.strip(),
        "used_predicates": rico_predicates
    }

def generate_predicates_with_llm(graph_json_path: str, llm) -> List[str]:
    with open(graph_json_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    entity_descriptions = []
    for edge in graph_data:
        node1 = edge.get("node_1", {})
        node2 = edge.get("node_2", {})
        rel = edge.get("relationship", "").strip()
        edge_props = edge.get("edge_properties", {})

        name1 = node1.get("name")
        name2 = node2.get("name")

        if name1 and node1:
            for attr, val in node1.items():
                if attr not in ["name", "label", "text_span"]:
                    entity_descriptions.append(f'{name1}.{attr} = "{val}"')

        if name2 and node2:
            for attr, val in node2.items():
                if attr not in ["name", "label", "text_span"]:
                    entity_descriptions.append(f'{name2}.{attr} = "{val}"')

        if rel:
            entity_descriptions.append(f'{name1}_{name2}.relationship = "{rel}"')

        for ep_key, ep_val in edge_props.items():
            if ep_key != "text_span":
                entity_descriptions.append(f'{name1}_{name2}.{ep_key} = "{ep_val}"')

    # 去重，提升质量
    entity_descriptions = list(sorted(set(entity_descriptions)))

    # 构建 Prompt
    system_message = (
        "You are a knowledge graph inference engine. "
        "Given entities and relationships extracted from a document, "
        "generate expressive and diverse logical predicates. "
        "These predicates are used to construct RICO-style logical rules."
    )

    user_prompt = f"""Entities and relationships:
{chr(10).join(entity_descriptions)}

Generate exactly 100 **diverse** logical predicates in the style of:

Entity-level:
- entity.attribute = "value"
- entity.attribute != "value"
- entity.attribute < "value"
- entity.attribute > "value"
- entity.attribute <= "value"
- entity.attribute >= "value"

Edge-level:
- edge.attribute = "value"
- edge.attribute != "value"
- edge.attribute < "value"
- edge.attribute > "value"
- edge.attribute <= "value"
- edge.attribute >= "value"

Meta-level:
- M_NLI(Entity1.attrs, Entity2.attrs)
- M_NLU(Entity1.attrs, Entity2.attrs)
- M_TEM(Entity1.attrs, Entity2.attrs)
- linkage(Entity1, Entity2)
- contra("text1", "text2")

⚠️ Requirements:
- Roughly **balanced types** across the above categories (entity, edge, meta).
- Use a variety of attributes and entity pairs.
- No duplicate predicates.
- Use realistic numeric/string comparisons where applicable.
- Avoid trivial or tautological predicates.

Output only the 100 predicates, one per line.
"""

    response = llm.generate(
        user_message=user_prompt,
        system_message=system_message
    )

    predicates = [line.strip() for line in response.strip().splitlines() if line.strip()]
    return list(dict.fromkeys(predicates))  # 去重保持顺序


def save_generated_rico(path: str, predicates: List[str], rules: str):
    with open(path, "w", encoding="utf-8") as f:
        # f.write("Predicates:\n")
        # f.write("\n".join(predicates))
        f.write("Rules:\n")
        f.write(rules)

def save_predicates_to_file(predicates: List[str], save_path: str, append: bool = False):
    mode = "a" if append else "w"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, mode, encoding="utf-8") as f:
        if append:
            f.write("\n\n# Additional Predicates:\n")
        else:
            f.write("Predicates:\n")
        for p in predicates:
            f.write(p.strip() + "\n")


if __name__ == "__main__":
    result = generate_rico_output(
        json_graph_path="../star_json/dual_graph/dual_star_graph_1_Emily_Clark.json",
        rico_txt_path="./star_outputs/dual_star_graph_1_Emily_Clark_rico.txt",
        llm=llm
    )

    save_generated_rico(
        path="./star_outputs/dual_star_graph_1_Emily_Clark_rico_generated.txt",
        predicates=result["used_predicates"],
        rules=result["new_rules"]
    )

    # print("\n📐 Used Predicates:")
    # print("\n".join(result["used_predicates"]))

    print("\n🧠 Generated RICO Rules:")
    print(result["new_rules"])

    preds = generate_predicates_with_llm("../star_json/dual_graph/dual_star_graph_1_Emily_Clark.json", llm)

    # 2. 保存到指定文件
    save_predicates_to_file(
        predicates=preds,
        save_path="./star_outputs/dual_star_graph_1_Emily_Clark_rico_generated.txt",
        append=True  # ✅ 不覆盖，追加
    )

    print("\n🧠 Generated RICO Predicates:")
    print(preds)

    print("✅ 谓词已保存到 ./star_outputs/dual_star_graph_1_Emily_Clark_rico_generated.txt")
