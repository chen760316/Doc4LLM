"""
==================================================
           ✨ 规则学习任务处理完成 ✨
==================================================
📋 处理文件总数: 40
⏳ 总运行耗时: 78.95 秒 (约 1.32 分钟)
🚀 并发吞吐率: 30.40 文件/分钟
⏱️ 平均单文件等待时间: 1.97 秒 (并发环境下)
📅 结束时间: 2025-12-24 19:59:58
==================================================
"""
import os
import json
import re
import asyncio
from glob import glob
from functools import partial
from dotenv import load_dotenv
from knowledge_graph_maker import OpenAIClient
from typing import List, Dict, Any, Tuple, Set as TypeSet
import time

# 设定目录
INPUT_DIR = "dual_star_json_with_attrs"  # 原始图数据输入目录
OUTPUT_DIR = "dual_star/llm_learning_rules"
# 新增：有效规则的输入目录
EFFECTIVE_RULES_DIR = "dual_star/llm_evaluations_v3/effective_with_center_name"

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
CONTRADICTION_SYSTEM_PROMPT = """
You are an expert in detecting contradiction types between two sentences.

Given two sentences, identify if they are contradictory and, if so, classify the contradiction into one of the following 7 types. Each contradiction type is explained with multiple detailed examples to clarify the scope of each category:

1. **Factual**: Requires knowledge of the external world to verify contradictions (external knowledge bases are needed for detection).
2. **Content**: Contradictions caused by altering one or more attributes of an event or entity. This includes:
   - **Opinion/Emotion**: A contradiction involving differing emotional reactions or opinions.
3. **Causal**: Contradictions arising from a mismatch between cause and effect (must involve a clear causal relationship).
4. **Negation**: Contradictions resulting from negating the original sentence (must involve explicit negation, such as antonyms or negative words like "do not," "is not").
5. **Relation**: Contradictions arising from descriptions of two mutually exclusive relationships between entities (must involve conflicts in relationships between entities), including reversed, missing, or mismatched relationships.
6. **Numeric**: Contradictions due to mismatched numbers or numbers falling outside an acceptable range.
7. **Temporal**: Contradictions caused by changes or updates in knowledge over time, affecting temporal order.
"""  # 恢复完整的 System Prompt 内容


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
        # 捕捉 API 错误并打印，而非直接返回 None
        print(f"❌ 警告: LLM 生成请求失败: {e}")
        return None


# -------------------- 核心逻辑重构为异步 --------------------

def _extract_graph_data(graph_json_path: str) -> Tuple[List[str], str]:
    # 保持不变
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
    graph_description = chr(10).join(entity_descriptions)

    return entity_descriptions, graph_description


# --- 修正 2a：新增函数：从原始 JSON 提取中心节点名称 ---
def _extract_center_names(graph_json_path: str) -> Tuple[str, str]:
    """从原始 dual-star JSON 文件中提取 x0 和 y0 的中心节点名称。"""
    try:
        with open(graph_json_path, "r", encoding="utf-8") as f:
            dual_star_data = json.load(f)

        stars = dual_star_data.get("stars", [])

        # 假设 x0 是第一个 star 的中心节点，y0 是第二个 star 的中心节点
        # 如果只有一个 star，则 x0 和 y0 相同

        x0_name = stars[0].get("center", f"x0_default_{os.path.basename(graph_json_path)}")

        if len(stars) > 1:
            y0_name = stars[1].get("center", f"y0_default_{os.path.basename(graph_json_path)}")
        else:
            y0_name = x0_name  # 如果只有一个 star，则 y0 复制 x0

        return x0_name, y0_name
    except Exception as e:
        print(f"❌ 警告: 无法从原始 JSON {os.path.basename(graph_json_path)} 中提取中心节点: {e}")
        # 回退到文件名作为默认名称
        base_name = os.path.basename(graph_json_path).replace('.json', '')
        return f"x0_fallback_{base_name}", f"y0_fallback_{base_name}"


# 【修正函数】加载有效规则，并提取 x0, y0 的具体名称
def _load_effective_rules(effective_rules_dir: str, base_name: str) -> Tuple[List[str], str, str]:
    """
    从指定的有效规则目录中加载对应文件的 Rules with Substituted Entities，并提取 x0/y0 名称。
    返回: (规则列表, x0 名称, y0 名称)
    """
    file_path = os.path.join(effective_rules_dir, f"{base_name}_final_rules.txt")
    effective_rules = []
    # 暂时使用占位符，如果提取失败，调用方会用 _extract_center_names 替换
    x0_name_extracted = "x0_placeholder"
    y0_name_extracted = "y0_placeholder"

    if not os.path.exists(file_path):
        return effective_rules, x0_name_extracted, y0_name_extracted

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 1. 提取 x0 和 y0 的具体名称
        x0_match = re.search(r'^x0: "(.*?)"', content, re.MULTILINE)
        y0_match = re.search(r'^y0: "(.*?)"', content, re.MULTILINE)

        if x0_match:
            x0_name_extracted = x0_match.group(1)
        if y0_match:
            y0_name_extracted = y0_match.group(1)

        # 2. 提取有效规则
        rules_section_match = re.search(r"### Rules with Substituted Entities ###\s*(\S[\s\S]*?)(?:$|###)", content)

        if rules_section_match:
            rules_text = rules_section_match.group(1)
            effective_rules = re.findall(r"Rule: (Q\[.*?\]\(.*?\))", rules_text)

    except Exception as e:
        print(f"⚠️ 警告: 读取有效规则文件 {os.path.basename(file_path)} 失败: {e}")

    return effective_rules, x0_name_extracted, y0_name_extracted


async def generate_predicates_with_llm_async(graph_json_path: str, llm: OpenAIClient, x0_name: str, y0_name: str, num_predicates: int = 100) -> List[str]:
    # 保持逻辑不变，但现在 x0_name 和 y0_name 是保证有效的
    try:
        entity_descriptions, graph_description = _extract_graph_data(graph_json_path)
    except Exception as e:
        print(f"❌ 数据提取失败: {e}")
        return []

    system_message = (
            "You are a knowledge graph inference engine focused on contradiction detection. "
            "Given the entity and relationship data below, generate logical predicates. "
            "The predicates must support the detection and classification of the following contradiction types (Factual, Content, Causal, Negation, Relation, Numeric, Temporal) as defined:\n\n"
            + CONTRADICTION_SYSTEM_PROMPT
    )

    user_prompt = f"""Entities and relationships (where "Relationship" is the unique edge identifier):
{graph_description}

--- Entity Mapping ---
x0 (Center 1): "{x0_name}"
y0 (Center 2): "{y0_name}"

Generate exactly {num_predicates} **diverse** logical predicates. **Crucially, ensure the final list includes the target contradiction predicate `violate("{x0_name}", "{y0_name}")`** and its specific type variants (e.g., `Factual("{x0_name}", "{y0_name}")`).

The predicates should follow the style, using the **entity names** directly where applicable (e.g., instead of `x0.type`, use `{x0_name}.type`):
... (Predicate style instructions)

Meta-level (Focus on contradiction types for this task):
...
- violate("{x0_name}", "{y0_name}"): **This is the main target predicate. Include this.**
- Factual("{x0_name}", "{y0_name}"), Causal("{x0_name}", "{y0_name}"), Numeric("{x0_name}", "{y0_name}"), Temporal("{x0_name}", "{y0_name}"), etc. **Include several specific contradiction type predicates.**

⚠️ Requirements:
- Roughly **balanced types** across the above categories (entity, edge, meta).
- **MUST** include the target predicate `violate("{x0_name}", "{y0_name}")`.
- **MUST** include several specific contradiction type predicates (e.g., Factual("{x0_name}", "{y0_name}"), Numeric("{x0_name}", "{y0_name}")).
- Use a variety of attributes and entity pairs.
- No duplicate predicates.

Output only the {num_predicates} predicates, one per line.
"""
    # 注意：这里将 {x0_name} 改为 "{x0_name}" 以确保引号正确传递
    user_prompt = user_prompt.replace(f"violate({x0_name}, {y0_name})", f"violate(\"{x0_name}\", \"{y0_name}\")")
    # 使用常规字符串定义 lambda，避免 f-string 嵌套
    def quote_predicate_args(m):
        func_name = m.group(1)
        # 获取所有参数，移除潜在的空格，并用引号包裹
        quoted_args = ', '.join(f'"{arg.strip()}"' for arg in m.group(2).split(','))
        return f'{func_name}({quoted_args})'

    user_prompt = re.sub(
        r'([A-Za-z]+)\((.*?)\)',
        quote_predicate_args,
        user_prompt
    )
    response = await _llm_generate_async(llm, user_message=user_prompt, system_message=system_message)

    if not response or not isinstance(response, str):
        print(f"❌ 警告: 文件 {os.path.basename(graph_json_path)} LLM 返回空响应，跳过谓词解析。")
        return []

    predicates = [line.strip() for line in response.strip().splitlines() if line.strip()]

    # 强制包含 violate(x0_name, y0_name)
    target_violate = f"violate(\"{x0_name}\", \"{y0_name}\")"
    if target_violate not in predicates:
        predicates.append(target_violate)

    return list(dict.fromkeys(predicates))


def build_rule_generation_prompt(
        rico_predicates: List[str],
        num_rules: int,
        effective_rules: List[str],
        x0_name: str,
        y0_name: str
) -> str:
    # 保持逻辑不变
    # ... (省略重复代码) ...
    # 根据新的谓词列表，提取所有看起来像矛盾类型的谓词作为结论候选
    contradiction_predicates = [
        p for p in rico_predicates
        if p.startswith('violate(') or re.match(r'(Factual|Causal|Numeric|Temporal|Negation|Relation|Content)\(.*?\"(.*?)\".*?\"(.*?)\".*?\)', p)
    ]

    if not contradiction_predicates:
        contradiction_predicates = [f"violate(\"{x0_name}\", \"{y0_name}\")"]

    contradiction_target = " OR ".join(contradiction_predicates[:5])

    # 引入有效规则作为参考示例
    effective_rules_section = ""
    if effective_rules:
        effective_rules_section = f"""\n\n--- Reference Effective Rules (Use these to guide the structure and content of the new rules) ---
{chr(10).join(effective_rules[:5])}
"""

    return f"""You are an expert in logical reasoning and contradiction detection.

Your task is to generate RICO-style rules based on the given list of predicates.

Each rule must follow the exact format, using the instantiated entity names:
Q["{x0_name}", "{y0_name}"](P1 ∧ P2 ∧ ... → P_out)
{effective_rules_section}

### Rule Constraints:

1. Each rule must include between **2 and 6 predicate conditions** in the premise.
2. You must generate a **diverse set of rules** with varying premise lengths — specifically, make sure to include rules whose premises have 2, 3, 4, 5, and 6 predicate conditions.
3. The number of rules for each premise length should be **as evenly distributed as possible** — for example, if you are generating {num_rules} rules, try to generate about {num_rules // 5} rules for each premise length (length 2 to 6).
4. **Crucially, the conclusion predicate (P_out) MUST be a contradiction predicate**, such as `violate("{x0_name}", "{y0_name}")`, or a specific type like `Factual("{x0_name}", "{y0_name}")`. **The rules should aim to predict the contradiction based on the premises.**
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

    base_name = os.path.basename(graph_json_path).replace('.json', '')

    # === Step 0: 【修正】加载有效规则并进行回退处理 ===
    effective_rules, x0_name, y0_name = _load_effective_rules(EFFECTIVE_RULES_DIR, base_name)

    # 如果从有效规则中提取失败，则从原始 JSON 中提取中心节点作为默认名称
    if x0_name == "x0_placeholder":
        x0_name, y0_name = _extract_center_names(graph_json_path)
        print(f"[INFO] {base_name}: 切换到从原始 JSON 提取的中心节点: x0='{x0_name}', y0='{y0_name}'")

    if not effective_rules:
        print(f"[INFO] {base_name}: 未找到对应的有效规则文件作为参考。")

    # === Step 1: Generate predicates from graph (Async LLM Call) ===
    predicates = await generate_predicates_with_llm_async(
        graph_json_path,
        llm,
        x0_name=x0_name,
        y0_name=y0_name,
        num_predicates=num_predicates
    )

    if not predicates:
        print(f"[FAIL] {base_name}: 谓词生成失败或为空。")
        return

    # === Step 2: Generate rules based on predicates (Async LLM Call) ===
    rules_prompt = build_rule_generation_prompt(
        predicates,
        num_rules=num_rules,
        effective_rules=effective_rules,
        x0_name=x0_name,
        y0_name=y0_name
    )

    rule_generation_system_message = (
        "You are a natural language inference engine that generates RICO-style logical rules "
        "from given predicates. The rules MUST use the explicit entity names provided in the Q[] scope and the conclusion (P_out). "
        "The conclusion MUST be a contradiction predicate (e.g., violate(\"Name1\", \"Name2\"))."
    )

    response = await _llm_generate_async(llm, user_message=rules_prompt, system_message=rule_generation_system_message)

    # 检查规则生成响应是否有效
    if not response or not isinstance(response, str):
        print(f"❌ 警告: 文件 {base_name} 规则生成返回空响应，跳过保存。")
        rules = []
    else:
        # 移除 Rules: 标题行
        rules = [line.strip() for line in response.strip().splitlines() if line.strip() and not line.strip().startswith('Rules:')]

    # === Step 3: Save everything (同步操作) ===
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"### Entity Mapping ###\n")
        f.write(f"x0: \"{x0_name}\"\n")
        f.write(f"y0: \"{y0_name}\"\n\n")
        f.write("Predicates:\n")
        for p in predicates:
            f.write(p + "\n")
        f.write("\nRules:\n")
        for r in rules:
            f.write(r + "\n")

    print(f"✅ [SUCCESS] {base_name}: Saved {len(predicates)} predicates and {len(rules)} rules.")


# -------------------- IV. 异步主函数 --------------------
# 保持不变
async def main_concurrent():
    graph_files = glob(os.path.join(INPUT_DIR, "*.json"))

    if not graph_files:
        print(f"⚠️ 警告: 在目录 {INPUT_DIR} 中未找到任何 JSON 文件。请检查路径。")
        return

    # === ⏱️ 计时开始 ===
    start_time = time.perf_counter()

    print(f"--- 🚀 发现 {len(graph_files)} 个文件，开始并发处理 ---")
    print(f"--- ⚙️ 配置: 并发任务数={len(graph_files)}")

    tasks = []
    for graph_json_path in graph_files:
        base_name = os.path.basename(graph_json_path).replace('.json', '')
        output_file_path = os.path.join(OUTPUT_DIR, f"{base_name}_rules_llm.txt")

        task = generate_rico_predicates_and_rules_concurrent(
            graph_json_path=graph_json_path,
            output_path=output_file_path,
            llm=llm,
            num_predicates=100,
            num_rules=30
        )
        tasks.append(task)

    # 并行执行所有任务并等待结束
    await asyncio.gather(*tasks)

    # === ⏱️ 计时结束 ===
    end_time = time.perf_counter()
    total_duration = end_time - start_time

    # === 📊 生成统计报告 ===
    print("\n" + "=" * 50)
    print("           ✨ 规则学习任务处理完成 ✨")
    print("=" * 50)
    print(f"📋 处理文件总数: {len(graph_files)}")
    print(f"⏳ 总运行耗时: {total_duration:.2f} 秒 (约 {total_duration / 60:.2f} 分钟)")

    if len(graph_files) > 0:
        # 吞吐量指标
        avg_speed = total_duration / len(graph_files)
        throughput = (len(graph_files) / total_duration) * 60
        print(f"🚀 并发吞吐率: {throughput:.2f} 文件/分钟")
        print(f"⏱️ 平均单文件等待时间: {avg_speed:.2f} 秒 (并发环境下)")

    print(f"📅 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print("=" * 50)


if __name__ == "__main__":
    # 使用 asyncio.run 启动异步主函数
    asyncio.run(main_concurrent())