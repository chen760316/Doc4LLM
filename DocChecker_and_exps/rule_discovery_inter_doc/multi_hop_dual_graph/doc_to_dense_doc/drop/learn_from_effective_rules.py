"""
执行效率太低
"""
import os
import json
from dotenv import load_dotenv
from knowledge_graph_maker import OpenAIClient
from collections import defaultdict
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

# 设定你的目录
llm_folder = './dual_star/llm_evaluations_v2'  # 用来加载已有有效规则的目录
json_data_folder = './dual_star_json_with_attrs'  # 用来加载JSON数据的目录
contradictions_file = '../dense_text_cross.txt'  # 用来加载矛盾类型的文件
output_folder = './dual_star/learn_from_effective_rules'  # 输出评估结果的文件夹路径

os.makedirs(output_folder, exist_ok=True)

def load_contradictions(file_path: str) -> List[str]:
    """加载矛盾类型文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

# 加载矛盾类型
contradictions = load_contradictions(contradictions_file)

def generate_predicates_with_llm(graph_json_path: str, llm, num_predicates: int = 100) -> List[str]:
    """生成谓词"""
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

        if name1:
            for attr, val in node1.items():
                if attr not in ["name", "label", "text_span"]:
                    entity_descriptions.append(f'{name1}.{attr} = "{val}"')

        if name2:
            for attr, val in node2.items():
                if attr not in ["name", "label", "text_span"]:
                    entity_descriptions.append(f'{name2}.{attr} = "{val}"')

        if rel:
            entity_descriptions.append(f'{name1}_{name2}.relationship = "{rel}"')

        for ep_key, ep_val in edge_props.items():
            if ep_key != "text_span":
                entity_descriptions.append(f'{name1}_{name2}.{ep_key} = "{ep_val}"')

    entity_descriptions = list(sorted(set(entity_descriptions)))

    system_message = (
        "You are a knowledge graph inference engine. "
        "Given entities and relationships extracted from a document, "
        "generate expressive and diverse logical predicates. "
        "These predicates are used to construct RICO-style logical rules."
    )

    user_prompt = f"""Entities and relationships:
{chr(10).join(entity_descriptions)}

Generate exactly {num_predicates} **diverse** logical predicates in the style of:

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
- contra("text1", "text2")
- S(x, s): Maps node $x$ to the substring $s$ in the original document
- S(ex, s): Maps edge $ex$ to the substring $s$ in the original document
- T(s, s', tau): Determines the specific contradiction type tau between the strings $s$ and $s'$, where tau can be one of the following types: Factual, Content. Causal, Negation, Relational, Numeric, Temporal
- tau(x.A, y.B): After determining the contradiction type tau in T(s, s', tau), locates where the contradiction occurs between the node attributes x.A and y.B.
tau(ex.A, ey.B): After determining the contradiction type tau in T(s, s', tau$), locates where the contradiction occurs between the edge attributes ex.A and ey.B.

⚠️ Requirements:
- Roughly **balanced types** across the above categories (entity, edge, meta).
- Use a variety of attributes and entity pairs.
- No duplicate predicates.
- Use realistic numeric/string comparisons where applicable.
- Avoid trivial or tautological predicates.

Output only the {num_predicates} predicates, one per line.
"""

    response = llm.generate(
        user_message=user_prompt,
        system_message=system_message
    )

    predicates = [line.strip() for line in response.strip().splitlines() if line.strip()]
    return list(dict.fromkeys(predicates))

def build_rule_generation_prompt(rico_predicates: List[str], example_rules: List[str]) -> str:
    """构建生成规则的提示"""
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

def generate_rico_predicates_and_rules(
        graph_json_path: str,
        output_path: str,
        llm,
        contradictions: List[str],
        num_predicates: int = 100,
        num_rules: int = 100
):
    """生成谓词和规则，并根据矛盾类型评估规则的有效性"""
    # === Step 1: Generate predicates from graph ===
    predicates = generate_predicates_with_llm(graph_json_path, llm, num_predicates=num_predicates)

    # === Step 2: Generate rules based on predicates ===
    rules_prompt = build_rule_generation_prompt(predicates, example_rules=[])
    system_message = (
        "You are a natural language inference engine that generates RICO-style logical rules "
        "from given predicates. Each rule should follow the format Q[x0, y0](P1 ∧ P2 ∧ ... → P_out) "
        "and capture a plausible implication or contradiction."
    )
    response = llm.generate(
        user_message=rules_prompt,
        system_message=system_message
    )
    rules = [line.strip() for line in response.strip().splitlines() if line.strip()]

    # === Step 3: Evaluate rules with contradictions ===
    valid_rules = []
    for rule in rules:
        # Evaluate the rule against contradictions
        is_valid = evaluate_rule_against_contradictions(rule, contradictions)
        if is_valid:
            valid_rules.append(rule)

    # === Step 4: Save everything ===
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Predicates:\n")
        for p in predicates:
            f.write(p + "\n")
        f.write("\nRules:\n")
        for r in valid_rules:
            f.write(r + "\n")

    print(f"\n✅ Successfully saved {len(predicates)} predicates and {len(valid_rules)} valid rules to: {output_path}")


def evaluate_rule_against_contradictions(rule: str, contradictions: List[str]) -> bool:
    """根据矛盾类型评估规则是否有效"""
    # 这里可以实现具体的矛盾类型评估逻辑
    # 假设我们通过LMM生成一个检查规则是否匹配矛盾类型的系统
    prompt = f"Check if the following rule matches any contradiction types (Factual, Content, Causal, Negation, Relational, Numeric, Temporal):\n{rule}"

    response = llm.generate(
        user_message=prompt,
        system_message="You are an expert in contradiction detection. Your task is to check if the given rule violates any contradiction types."
    )

    # 假设返回的响应是一个布尔值，表示是否匹配矛盾类型
    return "contradiction" in response.lower()  # 这里可以进一步调整根据实际的API响应进行调整

# 遍历所有 JSON 文件
json_files = [os.path.join(json_data_folder, f) for f in os.listdir(json_data_folder) if f.endswith(".json")]

if not json_files:
    print(f"⚠️ 没有找到任何 JSON 文件在目录: {json_data_folder}")
    exit(1)

# For each JSON file, generate predicates and rules and save to output
for json_file in json_files:
    # Construct output file path based on the input JSON file
    output_file_path = os.path.join(output_folder, f"generated_rules_llm_{os.path.basename(json_file)}.txt")

    print(f"Processing file: {json_file}")

    # Call the function to generate predicates and rules for each JSON file
    generate_rico_predicates_and_rules(
        graph_json_path=json_file,
        output_path=output_file_path,
        llm=llm,
        contradictions=contradictions,
        num_predicates=100,
        num_rules=100
    )
