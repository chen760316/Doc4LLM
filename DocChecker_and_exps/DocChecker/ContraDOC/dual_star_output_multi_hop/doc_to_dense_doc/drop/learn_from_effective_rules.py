import os
import json
import re
import asyncio
from glob import glob
from functools import partial
from dotenv import load_dotenv
from knowledge_graph_maker import OpenAIClient
from typing import List, Dict, Any, Tuple, Set as TypeSet

# 设定目录
INPUT_DIR = "dual_star_json_with_attrs"  # 原始图数据输入目录
OUTPUT_DIR = "dual_star/llm_learning_rules"
# 新增：有效规则的输入目录
EFFECTIVE_RULES_DIR = "dual_star/llm_evaluations_v3/effective_with_center_name"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 初始化环境 ===
load_dotenv()
# 确保 LLMClient 能够找到 API Key
if os.getenv("OPENAI_API_KEY") is None:
    print("⚠️ 警告: OPENAI_API_KEY 未设置。请检查 .env 文件。")

llm = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

# === 融入的 System Prompt (包含矛盾类型定义) ===
CONTRADICTION_SYSTEM_PROMPT = """
You are an expert in detecting contradiction types between two sentences.
... (保持 Contradiction System Prompt 内容不变，省略以节省空间)
"""  # 保持原有内容


# ... (保持 _llm_generate_async, _extract_graph_data 不变) ...
# 由于篇幅限制，此处不再重复 _llm_generate_async 和 _extract_graph_data 函数

# -------------------- 异步包装器 --------------------
async def _llm_generate_async(llm: OpenAIClient, user_message: str, system_message: str) -> str:
    """使用 asyncio.to_thread 包装同步的 LLM 调用。"""
    try:
        return await asyncio.to_thread(
            llm.generate,
            user_message=user_message,
            system_message=system_message
        )
    except Exception as e:
        print(f"❌ 警告: LLM 生成请求失败 (返回 None): {e}")
        return None


# -------------------- 核心逻辑重构为异步 --------------------

def _extract_graph_data(graph_json_path: str) -> Tuple[List[str], str]:
    """同步函数：从 JSON 文件中提取数据和生成 prompt。"""
    with open(graph_json_path, "r", encoding="utf-8") as f:
        dual_star_data = json.load(f)

    if not isinstance(dual_star_data, dict) or "stars" not in dual_star_data:
        raise ValueError("输入 JSON 文件不是预期的双星结构（缺少 'stars' 键或不是字典）。")

    entity_descriptions = []
    for star in dual_star_data.get("stars", []):
        node_attrs_dict = star.get("node_attrs", {})
        multi_hop_node_attrs_dict = star.get("multi_hop_node_attrs", {})
        all_node_attrs = {**node_attrs_dict, **multi_hop_node_attrs_dict}

        for node_name, attrs in all_node_attrs.items():
            for attr, val in attrs.items():
                if attr not in ["name", "label", "text_span", "center"]:
                    entity_descriptions.append(f'{node_name}.{attr} = "{val}"')

        edges_attrs_dict = star.get("edges_attrs", {})
        multi_hop_edges_attrs_dict = star.get("multi_hop_edges_attrs", {})
        all_edge_attrs = {**edges_attrs_dict, **multi_hop_edges_attrs_dict}

        for relationship, props in all_edge_attrs.items():
            for prop_key, prop_val in props.items():
                if prop_key not in ["text_span", "doc_id"]:
                    entity_descriptions.append(f'"{relationship}".{prop_key} = "{prop_val}"')

    entity_descriptions = list(sorted(set(entity_descriptions)))

    # 构建用户提示的 graph_description 部分
    graph_description = chr(10).join(entity_descriptions)

    return entity_descriptions, graph_description


# 【新增函数】加载有效规则
def _load_effective_rules(effective_rules_dir: str, base_name: str) -> List[str]:
    """
    从指定的有效规则目录中加载对应文件的 Rules with Substituted Entities。
    """
    # 构造有效规则文件的路径，例如: dual_star/llm_evaluations_v3/effective_with_center_name/filled_dual_star_1_final_rules.txt
    file_path = os.path.join(effective_rules_dir, f"{base_name}_final_rules.txt")
    effective_rules = []

    if not os.path.exists(file_path):
        return effective_rules

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 使用正则表达式匹配 "### Rules with Substituted Entities ###" 之后的所有 Rule:
        rules_section_match = re.search(r"### Rules with Substituted Entities ###\s*(\S[\s\S]*?)(?:$|###)", content)

        if rules_section_match:
            rules_text = rules_section_match.group(1)
            # 提取所有 Rule: 后面的内容直到下一个 Rule: 或文件结束
            effective_rules = re.findall(r"Rule: (Q\[.*?\]\(.*?\))", rules_text)

    except Exception as e:
        print(f"⚠️ 警告: 读取有效规则文件 {os.path.basename(file_path)} 失败: {e}")

    return effective_rules


async def generate_predicates_with_llm_async(graph_json_path: str, llm: OpenAIClient, num_predicates: int = 100) -> List[str]:
    # 提取数据（同步操作）
    try:
        entity_descriptions, graph_description = _extract_graph_data(graph_json_path)
    except Exception as e:
        print(f"❌ 数据提取失败: {e}")
        return []

    # === 系统消息：结合矛盾类型定义和谓词生成任务 ===
    system_message = (
            "You are a knowledge graph inference engine focused on contradiction detection. "
            "Given the entity and relationship data below, generate logical predicates. "
            "The predicates must support the detection and classification of the following contradiction types (Factual, Content, Causal, Negation, Relation, Numeric, Temporal) as defined:\n\n"
            + CONTRADICTION_SYSTEM_PROMPT
    )

    user_prompt = f"""Entities and relationships (where "Relationship" is the unique edge identifier):
{graph_description}

Generate exactly {num_predicates} **diverse** logical predicates. **Crucially, ensure the final list includes the target contradiction predicate `violate(x0, y0)`** and its specific type variants (e.g., `Factual(x0, y0)`).

The predicates should follow the style:

Entity-level:
- entity.attribute = "value"
...

Edge-level:
- "Relationship".attribute = "value"
...

Meta-level (Focus on contradiction types for this task):
- M_NLI(Entity1.attrs, Entity2.attrs)
- T(s, s', tau): Determines the specific contradiction type tau between the strings $s$ and $s'$, where tau can be one of the following types: Factual, Content. Causal, Negation, Relational, Numeric, Temporal
- tau(x.A, y.B): Locates where the contradiction occurs between the node attributes x.A and y.B.
- violate(x0, y0): **This is the main target predicate. Include this.**
- Factual(x0, y0), Causal(x0, y0), Numeric(x0, y0), Temporal(x0, y0), etc. **Include several specific contradiction type predicates.**

⚠️ Requirements:
- Roughly **balanced types** across the above categories (entity, edge, meta).
- **MUST** include the target predicate `violate(x0, y0)`.
- **MUST** include several specific contradiction type predicates (e.g., Factual(x0, y0), Numeric(x0, y0)).
- Use a variety of attributes and entity pairs.
- No duplicate predicates.

Output only the {num_predicates} predicates, one per line.
"""

    response = await _llm_generate_async(llm, user_message=user_prompt, system_message=system_message)

    if not response or not isinstance(response, str):
        print(f"❌ 警告: 文件 {os.path.basename(graph_json_path)} LLM 返回空响应，跳过谓词解析。")
        return []

    predicates = [line.strip() for line in response.strip().splitlines() if line.strip()]

    # 强制包含 violate(x0, y0) 以防 LLM 遗漏
    if "violate(x0, y0)" not in predicates:
        predicates.append("violate(x0, y0)")

    return list(dict.fromkeys(predicates))


def build_rule_generation_prompt(
        rico_predicates: List[str],
        num_rules: int,
        effective_rules: List[str]
) -> str:
    # 提取所有看起来像矛盾类型的谓词作为结论候选
    contradiction_predicates = [
        p for p in rico_predicates
        if p.startswith('violate(') or re.match(r'(Factual|Causal|Numeric|Temporal|Negation|Relation|Content)\(x0, y0\)', p)
    ]

    if not contradiction_predicates:
        contradiction_predicates = ["violate(x0, y0)"]

    contradiction_target = " OR ".join(contradiction_predicates[:5])

    # 【修正点】引入有效规则作为参考示例
    effective_rules_section = ""
    if effective_rules:
        effective_rules_section = f"""\n\n--- Reference Effective Rules (Use these to guide the structure and content of the new rules) ---
{chr(10).join(effective_rules[:5])}
"""

    return f"""You are an expert in logical reasoning and contradiction detection.

Your task is to generate RICO-style rules based on the given list of predicates.

Each rule should follow the format:
Q[x0, y0](P1 ∧ P2 ∧ ... → P_out)
{effective_rules_section}

### Rule Constraints:

1. Each rule must include between **2 and 6 predicate conditions** in the premise.
2. You must generate a **diverse set of rules** with varying premise lengths — specifically, make sure to include rules whose premises have 2, 3, 4, 5, and 6 predicate conditions.
3. The number of rules for each premise length should be **as evenly distributed as possible** — for example, if you are generating {num_rules} rules, try to generate about {num_rules // 5} rules for each premise length (length 2 to 6).
4. **Crucially, the conclusion predicate (P_out) MUST be a contradiction predicate**, such as `violate(x0, y0)`, or a specific type like `Factual(x0, y0)`, `Numeric(x0, y0)`, etc. **The rules should aim to predict the contradiction based on the premises.**
5. The conclusion predicate must not be identical to any of the predicates in the premise.
6. **Important**: Reference the provided "Reference Effective Rules" to maintain consistency in style, entity usage, and the overall implication captured.

--- Predicates ---
{chr(10).join(rico_predicates)}

--- Target Conclusion Predicates (P_out must be one of these or similar) ---
Target examples: {contradiction_target}

--- Now write exactly {num_rules} new rules below ---
Rules:"""


async def generate_rico_predicates_and_rules_concurrent(
        graph_json_path: str,
        output_path: str,
        llm: OpenAIClient,
        num_predicates: int = 100,
        num_rules: int = 30
):
    print(f"[START] {os.path.basename(graph_json_path)}")

    # 确定文件名基础部分 (例如: filled_dual_star_1)
    base_name = os.path.basename(graph_json_path).replace('.json', '')

    # === Step 0: 【新增】加载有效规则作为参考 ===
    effective_rules = _load_effective_rules(EFFECTIVE_RULES_DIR, base_name)
    if not effective_rules:
        print(f"[INFO] {base_name}: 未找到对应的有效规则文件作为参考。")

    # === Step 1: Generate predicates from graph (Async LLM Call) ===
    predicates = await generate_predicates_with_llm_async(graph_json_path, llm, num_predicates=num_predicates)

    if not predicates:
        print(f"[FAIL] {base_name}: 谓词生成失败或为空。")
        return

    # === Step 2: Generate rules based on predicates (Async LLM Call) ===
    # 【修正点】传入 effective_rules
    rules_prompt = build_rule_generation_prompt(
        predicates,
        num_rules=num_rules,
        effective_rules=effective_rules
    )

    rule_generation_system_message = (
        "You are a natural language inference engine that generates RICO-style logical rules "
        "from given predicates. Each rule should follow the format Q[x0, y0](P1 ∧ P2 ∧ ... → P_out) "
        "and capture a plausible implication or contradiction. **The conclusion MUST be a contradiction predicate (e.g., violate(x0, y0) or Factual(x0, y0)).**"
    )

    response = await _llm_generate_async(llm, user_message=rules_prompt, system_message=rule_generation_system_message)

    # 检查规则生成响应是否有效
    if not response or not isinstance(response, str):
        print(f"❌ 警告: 文件 {base_name} 规则生成返回空响应，跳过保存。")
        rules = []
    else:
        rules = [line.strip() for line in response.strip().splitlines() if line.strip()]

    # === Step 3: Save everything (同步操作) ===
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Predicates:\n")
        for p in predicates:
            f.write(p + "\n")
        f.write("\nRules:\n")
        for r in rules:
            f.write(r + "\n")

    print(f"✅ [SUCCESS] {base_name}: Saved {len(predicates)} predicates and {len(rules)} rules.")


# -------------------- IV. 异步主函数 --------------------

async def main_concurrent():
    # 查找所有 JSON 文件
    graph_files = glob(os.path.join(INPUT_DIR, "*.json"))

    if not graph_files:
        print(f"⚠️ 警告: 在目录 {INPUT_DIR} 中未找到任何 JSON 文件。请检查路径。")
        return

    print(f"--- 🚀 发现 {len(graph_files)} 个文件，开始并发处理 ---")

    tasks = []
    # 遍历所有找到的文件，创建异步任务
    for graph_json_path in graph_files:
        base_name = os.path.basename(graph_json_path).replace('.json', '')
        output_file_path = os.path.join(OUTPUT_DIR, f"{base_name}_rules_llm.txt")

        # 创建单个文件的处理任务
        task = generate_rico_predicates_and_rules_concurrent(
            graph_json_path=graph_json_path,
            output_path=output_file_path,
            llm=llm,
            num_predicates=100,
            num_rules=30
        )
        tasks.append(task)

    # 并行执行所有任务
    await asyncio.gather(*tasks)

    print("\n--- ✅ 所有文件处理完毕 ---")


if __name__ == "__main__":
    # 使用 asyncio.run 启动异步主函数
    asyncio.run(main_concurrent())