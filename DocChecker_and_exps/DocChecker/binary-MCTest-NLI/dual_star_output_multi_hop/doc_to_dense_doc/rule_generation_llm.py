import json
import os
import re
from glob import glob
import asyncio
from functools import partial
from dotenv import load_dotenv
from knowledge_graph_maker import OpenAIClient
from typing import List, Dict, Any, Tuple, Set as TypeSet

# 设定目录
OUTPUT_DIR = "dual_star/llm_rules"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 初始化环境 ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

llm = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

# === 融入的 System Prompt (包含矛盾类型定义) ===
# 作为常量在全局定义，供 LLM 调用使用
CONTRADICTION_SYSTEM_PROMPT = """
You are an expert in detecting contradiction types between two sentences.

Given two sentences, identify if they are contradictory and, if so, classify the contradiction into one of the following 7 types. Each contradiction type is explained with multiple detailed examples to clarify the scope of each category:

1. **Factual**: Requires knowledge of the external world to verify contradictions (external knowledge bases are needed for detection).
   Example: 
   - Sentence 1: "The Nobel Prize was indeed established in 1901 to commemorate Alfred Nobel."
   - Sentence 2: "Hinton received a world-renowned scientific award in 2024, which was established in 1901 to honor Shannon's contributions to the field of information."
   Contradiction type: **Factual**, because the Nobel Prize was established in 1901 to honor Alfred Nobel, not Shannon, and requires external knowledge to confirm the misattribution.

2. **Content**: Contradictions caused by altering one or more attributes of an event or entity. This includes:
   - **Opinion/Emotion**: A contradiction involving differing emotional reactions or opinions.
     Example:
     - Sentence 1: "William was engaged while reading the paper on computer architecture."
     - Sentence 2: "William found the content of the paper on computer architecture to be extremely boring."
     Contradiction type: **Content (Opinion/Emotion)**, because the contradiction arises from different emotional responses to the same paper (engagement vs. boredom).

3. **Causal**: Contradictions arising from a mismatch between cause and effect (must involve a clear causal relationship).
   Example:
   - Sentence 1: "David’s video, which was published on YouTube in 2024, was permanently removed that year due to violence."
   - Sentence 2: "Taylor came across this video published by David while browsing YouTube in 2025."
   Contradiction type: **Causal**, because the video was supposed to be removed in 2024, but it is still available in 2025.

4. **Negation**: Contradictions resulting from negating the original sentence (must involve explicit negation, such as antonyms or negative words like "do not," "is not").
   Example:
   - Sentence 1: "On April 20, 2025, the router experienced a brief routing disruption at 2:00 PM."
   - Sentence 2: "The router did not report any routing disruptions throughout April 20, 2025."
   Contradiction type: **Negation**, because the second sentence explicitly negates the occurrence of the routing disruption mentioned in the first sentence.

5. **Relation**: Contradictions arising from descriptions of two mutually exclusive relationships between entities (must involve conflicts in relationships between entities), including reversed, missing, or mismatched relationships.
   Example:
   - Sentence 1: "The owner of the brand store, Howard, and his wife Sheila have a harmonious relationship after marriage."
   - Sentence 2: "Howard and his wife Emma have a beloved little daughter named Sheila."
   Contradiction type: **Relation**, because the first sentence implies Howard is married to Sheila, while the second sentence implies he is married to Emma, creating a contradiction in the relationships.

6. **Numeric**: Contradictions due to mismatched numbers or numbers falling outside an acceptable range.
   Example:
   - Sentence 1: "Howard’s brand store sold 29 pairs of women’s high heels on June 15, 2025."
   - Sentence 2: "Howard is concerned about the lack of any sales on June 15, 2025."
   Contradiction type: **Numeric**, because one sentence claims sales of 29 pairs of high heels, while the other suggests there were no sales at all.

7. **Temporal**: Contradictions caused by changes or updates in knowledge over time, affecting temporal order.
   Example:
   - Sentence 1: "Around 1860, Nobel and other researchers believed that nitroglycerin could not treat heart disease."
   - Sentence 2: "After Nobel’s death, further research revealed that nitroglycerin is effective for treating heart disease."
   Contradiction type: **Temporal**, because the belief about nitroglycerin changed over time, from it being ineffective in the 1860s to being effective after Nobel's death.
"""


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

    # *** 修正点：检查 response 是否为 None 或空字符串 ***
    if not response or not isinstance(response, str):
        print(f"❌ 警告: 文件 {os.path.basename(graph_json_path)} LLM 返回空响应，跳过谓词解析。")
        return []

    predicates = [line.strip() for line in response.strip().splitlines() if line.strip()]

    # 强制包含 violate(x0, y0) 以防 LLM 遗漏
    if "violate(x0, y0)" not in predicates:
        predicates.append("violate(x0, y0)")

    return list(dict.fromkeys(predicates))


def build_rule_generation_prompt(rico_predicates: List[str], num_rules: int) -> str:
    # 提取所有看起来像矛盾类型的谓词作为结论候选
    contradiction_predicates = [
        p for p in rico_predicates
        if p.startswith('violate(') or re.match(r'(Factual|Causal|Numeric|Temporal|Negation|Relational|Content)\(x0, y0\)', p)
    ]

    if not contradiction_predicates:
        contradiction_predicates = ["violate(x0, y0)"]

    contradiction_target = " OR ".join(contradiction_predicates[:5])

    return f"""You are an expert in logical reasoning and contradiction detection.

Your task is to generate RICO-style rules based on the given list of predicates.

Each rule should follow the format:
Q[x0, y0](P1 ∧ P2 ∧ ... → P_out)

### Rule Constraints:

1. Each rule must include between **2 and 6 predicate conditions** in the premise.
2. You must generate a **diverse set of rules** with varying premise lengths — specifically, make sure to include rules whose premises have 2, 3, 4, 5, and 6 predicate conditions.
3. The number of rules for each premise length should be **as evenly distributed as possible** — for example, if you are generating {num_rules} rules, try to generate about {num_rules // 5} rules for each premise length (length 2 to 6).
4. **Crucially, the conclusion predicate (P_out) MUST be a contradiction predicate**, such as `violate(x0, y0)`, or a specific type like `Factual(x0, y0)`, `Numeric(x0, y0)`, etc. **The rules should aim to predict the contradiction based on the premises.**
5. The conclusion predicate must not be identical to any of the predicates in the premise.

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

    # === Step 1: Generate predicates from graph (Async LLM Call) ===
    predicates = await generate_predicates_with_llm_async(graph_json_path, llm, num_predicates=num_predicates)

    if not predicates:
        print(f"[FAIL] {os.path.basename(graph_json_path)}: 谓词生成失败或为空。")
        return

    # === Step 2: Generate rules based on predicates (Async LLM Call) ===
    rules_prompt = build_rule_generation_prompt(predicates, num_rules=num_rules)

    # === 规则生成的系统消息：简短而强调任务 ===
    # 注意：这里我们使用一个更聚焦于规则生成的系统消息，而不是重复整个矛盾定义
    rule_generation_system_message = (
        "You are a natural language inference engine that generates RICO-style logical rules "
        "from given predicates. Each rule should follow the format Q[x0, y0](P1 ∧ P2 ∧ ... → P_out) "
        "and capture a plausible implication or contradiction. **The conclusion MUST be a contradiction predicate (e.g., violate(x0, y0) or Factual(x0, y0)).**"
    )

    response = await _llm_generate_async(llm, user_message=rules_prompt, system_message=rule_generation_system_message)

    # 检查规则生成响应是否有效
    if not response or not isinstance(response, str):
        print(f"❌ 警告: 文件 {os.path.basename(graph_json_path)} 规则生成返回空响应，跳过保存。")
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

    print(f"✅ [SUCCESS] {os.path.basename(graph_json_path)}: Saved {len(predicates)} predicates and {len(rules)} rules.")


# -------------------- III. 异步主函数 --------------------

async def main_concurrent():
    # 设置输入和输出目录
    INPUT_DIR = "dual_star_json_with_attrs"

    # 查找所有JSON文件
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